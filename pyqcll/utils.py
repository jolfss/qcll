import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from typing import Tuple
import os

def setup_lm_and_tokenizer(model_id:int=0) -> Tuple[PreTrainedModel,PreTrainedTokenizer|PreTrainedTokenizerFast]:
    login(os.environ["HF_TOKEN"])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device, model_name= [
        (torch.device("cuda"), "gpt2-xl"),
        (torch.device("cpu"), "meta-llama/Meta-Llama-3.1-8B"),
        (torch.device("cpu"), "meta-llama/Llama-2-7b-chat-hf"),
        ][model_id]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer