"""This module encapsulates language modeling features."""
#imports
import torch
import copy
#typing
from typing import Iterator, List, Optional, Tuple, assert_never
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

import logging
logging.basicConfig(filename='logs/app.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Tokenstring():
    """A Tokenstring is an abstraction of strings and their tokenizations."""
    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer|PreTrainedTokenizerFast, cache:Optional[DynamicCache]) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._string = ""
        self._input_ids : Tensor = torch.tensor([[]]).to(self.device)
        self._logits : Tensor = torch.tensor([[[]]]).to(self.device)

        self._cache : Optional[DynamicCache] = cache
        if self._model._supports_cache_class and cache is None:
            logging.warning("No cache was provided but the model supports a cache class.")
        elif not self._model._supports_cache_class and cache is not None:
            logging.warning("A cache was provided but the model does not support a cache class; setting to none")
            self._cache = None

        self._output : Optional[CausalLMOutputWithPast] = None
        self._probabilities : Tensor = torch.tensor([[]])
        self._perplexities : Tensor = torch.tensor([[]])

    
    @property
    def device(self): 
        """The device this Tokenstring is on; always synced to underlying language model."""
        return self._model.device

    @property
    def string(self) -> str:
        """The string this Tokenstring represents; attempts to assign this value will trigger `self.pivot`."""
        return self._string
    @string.setter
    def string(self, incoming:str):
        self.pivot(incoming)

    @property
    def input_ids(self) -> Tensor: 
        """A Tensor of input ids for this Tokenstring.
        `Tensor (1,seq)@device:long`"""
        assert self._input_ids.device == self.device
        return self._input_ids 
    
    @property
    def tokens(self) -> List[str]:
        """The tokenization (as strings) of this Tokenstring.
        `List (seq)`"""
        return list(self._tokenizer.convert_ids_to_tokens(self.input_ids.flatten().tolist()))
    
    @property
    def word_count(self) -> int:
        "The number of 'words' in this Tokenstring, a 'word' can be anything but whitespace."
        return len(self._string.split())

    @property
    def output(self) -> Optional[CausalLMOutputWithPast]:
        """The model output on this Tokenstring or None if this Tokenstring is empty."""
        return self._output
    
    @property
    def logits(self) -> Tensor:
        """The logits/activations of the model on this Tokenstring.
        `Tensor (1,seq,dim)@device:float`"""
        assert self._logits.device == self.device
        return self._logits
    
    @property
    def is_using_cache(self) -> bool:
        return self._model._supports_cache_class and self._cache is not None

    def past_key_values(self, up_to:int) -> DynamicCache|List[Tuple[Tensor, ...]]:
        "TODO: Docs"
        assert self.output is not None
        assert self.output.past_key_values is not None
        if self.is_using_cache:
            assert self._cache is not None
            self._cache.crop(up_to)
            return self._cache
        else:
            return [tuple(key_value[:,:,:up_to,:]
                          for key_value in layer)
                          for layer in self.output.past_key_values]
    
    @property
    def probabilities(self) -> Tensor:
        """The per-token probabilities as computed by `self.__compute_probabilities`."""
        return self._probabilities

    @property
    def perplexities(self) -> Tensor:
        """The per-token perplexities as computed by `self.__compute_perplexities`."""
        return self._perplexities
    
    def __compute_probabilities(self) -> Tensor: 
        """The per-token probabilities under the language model on this Tokenstring.
        Causally, there is no way to predict the first token, so this is set to 1.0.
        `Tensor (1,seq)@cpu:float`"""
        if self.output:
            probs = self.logits.softmax(dim=-1)[0,torch.arange(self.input_ids.size(-1)-1),self.input_ids[:,1:]]
            return torch.cat((torch.tensor([[1.0]]).to(self.device),probs), dim=-1).to("cpu")
        else:
            return torch.tensor([[]],device="cpu")

    def __compute_perplexities(self) -> Tensor:
        """The per-token perplexities under the language model on this Tokenstring.
        The first token's perplexity is defined to be `self._tokenizer.vocab_size`.
        `Tensor (1,seq)@cpu:float`."""
        if self.output:
            probs = self.logits.softmax(dim=-1)
            return torch.cat((torch.tensor([[self._tokenizer.vocab_size]]).to(self.device),
                                (-probs*probs.log()).sum(dim=-1).exp()), dim=-1).to("cpu")
        else:
            return torch.tensor([[]], device="cpu")

    # NOTE TODO: https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache
    # Newer models are aiming to deprecate past_key_values and move to ^^^. Should change this soon. 
    @torch.no_grad
    def pivot(self, incoming:'str|Tokenstring') -> 'Tokenstring':
        """Align this Tokenstring's representation with the `incoming` string. If the incoming
        object is a Tokenstring, becomes a clone of that Tokenstring, returns `self`."""
        match incoming:
            case "":
                #clear fields
                self._string = "" 
                self._input_ids = torch.tensor([[]]).to(self.device)
                self._logits = torch.tensor([[[]]]).to(self.device)
                self.past_key_values(up_to=0)
                self._output = None
                self._perplexities = torch.tensor([[]]).to(self.device)
                self._probabilities = torch.tensor([[]]).to(self.device)
            case self.string:
                pass
            case str() as next:
                self._string = next
                next_input = self._tokenizer(next, return_tensors="pt").to(self.device)
                next_attn = next_input["attention_mask"]
                next_ids = next_input["input_ids"] # type:ignore (c.f. assert)
                assert type(next_ids) == Tensor

                # find first point of difference
                same_until = 0
                for id, n_id in zip(self.input_ids.flatten().tolist(),next_ids.flatten().tolist()):
                    if id != n_id: 
                        break
                    same_until += 1
                
                # new input
                self._input_ids = next_ids
                new_ids = next_ids[:,same_until:]

                # update efficiently
                if self.output is not None:
                    assert self.output.past_key_values; assert self.output.logits is not None
                    past_key_values = self.past_key_values(up_to=same_until)
                    if new_ids.size(-1) > 0:
                        self._output = self._model(input_ids=new_ids,attention_mask=next_attn, 
                                                    past_key_values=past_key_values, use_cache=True)
                        self._logits = torch.cat((self.logits[:,:same_until],self.output.logits),dim=-2)
                    else:
                        self._logits = self.logits[:,:same_until]
                else:
                    self._output = self._model(**next_input.to(self.device),
                                               past_key_values=self._cache,
                                               use_cache=True)
                    assert self.output is not None
                    self._logits = self.output.logits
                self._probabilities = self.__compute_probabilities()
                self._perplexities = self.__compute_perplexities()
            case Tokenstring() as ts:
                #pass reference
                self._model     = ts._model
                self._tokenizer = ts._tokenizer
                #copy fields
                self._string    = ts._string
                self._input_ids = ts._input_ids.clone()
                self._logits    = ts._logits.clone()
                self._cache     = copy.deepcopy(ts._cache)
                self._output    = copy.deepcopy(ts._output)
                self._probabilities = ts._probabilities.clone()
                self._perplexities = ts._perplexities.clone()
            case _ as unreachable: 
                assert_never(unreachable)
        return self

    def append(self, incoming:str):
        """Append `incoming` to the end of this Tokenstring."""
        self.string = self.string + incoming

    def pop(self, num_popped:int) -> str:
        """Remove and return `num_popped` characters from the end of this Tokenstring."""
        popped = self.string[-num_popped:]
        self.string = self.string[:-num_popped]
        return popped

    def clone(self) -> 'Tokenstring':
        """Initialize a new instance of a Tokenstring using `self` as a template."""
        return Tokenstring(self._model, self._tokenizer, copy.deepcopy(self._cache)).pivot(self)
    def __deepcopy__(self, memo) -> 'Tokenstring':
        memo[id(self)] = (clone:=self.clone())
        return clone

    def __iter__(self) -> Iterator[Tuple[str, float, float]]:
        """An iterator over tuples of this Tokenstring's (token, probability, perperplexity).
        `Iterator (seq):(str * float * float)`"""
        return zip(
            self.tokens,
            map(lambda tnsr : tnsr.item(), self.probabilities.flatten().to("cpu")),
            map(lambda tnsr : tnsr.item(), self.perplexities.flatten().to("cpu"))
        )
    
    def __str__(self) -> str:
        "The string representation of this Tokenstring (simply the string it encapsulates)."
        return self.string

    def __len__(self) -> int:
        """The `__len__` of a Tokenstring is the number of tokens in its tokenization."""
        return len(self.tokens) 

if __name__ == "__main__":
    import os
    from huggingface_hub import login
    import transformers

    try:
        login(os.environ["HF_TOKEN"])
    except KeyError:
        print("WARNING: HF_TOKEN environment variable not set.")

    device, model_name= [
        (torch.device("cuda"), "gpt2-xl"),
        (torch.device("cpu"), "meta-llama/Meta-Llama-3.1-8B"),
        (torch.device("cpu"), "meta-llama/Llama-2-7b-chat-hf"),
        ][1]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    cache = DynamicCache()

    tokenstring = Tokenstring(model,tokenizer, cache).pivot("Hello").pivot("Hello there")
    for token, prob, perp in tokenstring:
        print(token)
    breakpoint()
