from blessed import Terminal
from blessed.keyboard import Keystroke

### LOGGING

#------------#
#   Typing   #
#------------#
from dataclasses import dataclass
from typing import List, Tuple, assert_never 
from enum import Enum

@dataclass
class Alphanumeric: key:Keystroke
class Arrow(Enum): UP,DOWN,LEFT,RIGHT=range(4)
class Backspace:...
class Empty:...
class Esc:...
class Newline:...
@dataclass
class Printable: key:Keystroke
@dataclass
class Unsupported: key:Keystroke
type Input = Alphanumeric | Arrow | Backspace | Empty | Esc | Newline | Printable | Unsupported

class Mode(Enum): MENU, PLAYGROUND, EXIT = range(3)

#-------------#
#   Globals   #
#-------------#
### Language Model
import torch
import transformers
from huggingface_hub import login
from tokenstring import Tokenstring
### Misc.
import os
import numpy as np
import time

try:
    login(os.environ["HF_TOKEN"])
except KeyError:
    print("WARNING: HF_TOKEN environment variable not set.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device, model_name= [
    (torch.device("cuda"), "gpt2-xl"),
    (torch.device("cpu"), "meta-llama/Meta-Llama-3.1-8B"),
    (torch.device("cpu"), "meta-llama/Llama-2-7b-chat-hf"),
    ][0]
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to(device)
cache = transformers.DynamicCache()

### App State
blocks : List[Tokenstring] = [Tokenstring(model, tokenizer, cache)]
cur_id : int = 0
cur_pos : int = 0

def is_previewing():
    global cur_id, blocks
    return cur_id != len(blocks)-1

def get_preview():
    global blocks, cur_id
    return blocks[:-1] + [blocks[cur_id]]

def ratify():
    global cur_id, cur_pos, blocks
    if is_previewing():
        blocks = blocks[:-1] + [blocks[cur_id].clone()] 
        cur_id = len(blocks) - 1
        cur_pos = len(blocks[-1].string)

def push_block():
    global model, tokenizer
    global blocks, cur_id, cur_pos
    blocks.append(Tokenstring(model, tokenizer, transformers.DynamicCache()))
    cur_id = len(blocks) - 1
    cur_pos = 0

def pop_block():
    global blocks, cur_id, cur_pos
    if len(blocks) > 1:
        blocks.pop()
        assert cur_id == len(blocks)
        cur_id -= 1
        cur_pos = len(blocks[-1].string)

def scroll(amount:int):
    global blocks, cur_id, cur_pos
    if amount > 0 and cur_id < len(blocks) - 1:
        cur_id += 1
    elif amount < 0 and cur_id > 0:
        cur_id -= 1

### Statistics
def word_count():
    return sum(map(lambda tokenstring : tokenstring.word_count, blocks)) 

def token_count():
    return sum(map(lambda tokenstring : len(tokenstring), blocks))
#-----------------#
#   Application   #
#-----------------#
def flush(string:str):
    print(string, flush=True, end="")

def parse_keystroke(term:Terminal) -> Input:
    """Receive and interpret key-presses to the terminal."""
    keystroke = term.inkey()
    if keystroke.isalnum():
        return Alphanumeric(keystroke)
    elif keystroke.isprintable():
        return Printable(keystroke)
    elif keystroke.name == "KEY_BACKSPACE":
        return Backspace()
    elif keystroke.code == term.KEY_ENTER:
        return Newline()
    elif keystroke.code == term.KEY_ESCAPE:
        return Esc()
    elif keystroke.code == term.KEY_UP:
        return Arrow.UP
    elif keystroke.code == term.KEY_DOWN:
        return Arrow.DOWN
    elif keystroke.code == term.KEY_LEFT:
        return Arrow.LEFT
    elif keystroke.code == term.KEY_RIGHT:
        return Arrow.RIGHT
    else:
        return Unsupported(keystroke)

### CONSTANTS
SIDE_PAD = 4
BOTTOM_PAD = 3
BG = (10,0,20)

### BAR
"""Bar:
#-------------------------#
| ***information***       | ^^^BOTTOM_PAD
#-------------------------#"""
def draw_bar(term: Terminal):
    _row = term.height-2
    flush(term.move_yx(_row,0) + term.normal + term.on_color_rgb(*BG) + 
          F" qcll: alpha | language model:{model_name} | words:{word_count()} | tokens: {token_count()}")

### MENU
def draw_menu(term:Terminal): ...

### PLAYGROUND
"""Playground Layout:
#-----------------------------------#
|SIDE_ |                     |      |
|   PAD|                     |      |
|<---->| 1) First sentence   |      |
|      |    continues here.  |      |
|      | 2)  Second line.    |      |
|------|---------------------|------|
|      |   ^^^BOTTOM_PAD     |      |
#-----------------------------------#"""
def draw_playground(term:Terminal):
    _row = term.height - BOTTOM_PAD 
    _last_line = None
    _block_number = len(blocks)
    _completed = False
    for tokenstring in reversed(get_preview()): 
        # add highlighting
        flush(term.color_rgb(220,220,220) if _block_number-1 == cur_id else term.color_rgb(150,150,130))
        color = ""
        for token, prob, perp in tokenstring:
            token = token.replace("Ġ"," ")
            score = 1/(prob*perp)
            color += term.on_color_rgb(BG[0]+max(BG[0],min(255,int(score))), BG[1], BG[2]) + F"{token}"#({int(np.log2(score))})" 
        color += term.on_color_rgb(*BG)
        # wrap and center
        color_wrap = term.wrap(color, term.width - 2*SIDE_PAD, drop_whitespace=False)
        _last_line = color_wrap[-1] if _last_line is None else _last_line
        color_wrap_center = list(map(lambda s :term.center(s, term.width - 2*SIDE_PAD), color_wrap))
        
        # print to terminal
        _offset = len(color_wrap_center)
        for line in color_wrap_center:
            if _row - _offset <= 0: 
                _completed = True
            if _offset == len(color_wrap_center):
                flush(term.move_yx(_row - _offset,1) + F"{_block_number})")
            flush(term.move_yx(_row - _offset, SIDE_PAD) + line)
            _offset -= 1
        
        # track iterators
        if _completed: 
            break
        _row -= len(color_wrap_center)
        _block_number -= 1

    # find cursor placement
    assert _last_line is not None
    flush(term.move_yx(term.height - BOTTOM_PAD - 1, (term.width + term.length(_last_line))//2))

def main():
    """The application code."""
    term = Terminal()
    mode = Mode.PLAYGROUND
    with term.fullscreen(), term.cbreak():
        #--------------#
        #   APP LOOP   #
        #--------------#
        while mode != Mode.EXIT:
            #-----------------#
            #   MODE : Menu   #
            #-----------------#
            if mode == Mode.MENU:
                draw_menu(term)
                match parse_keystroke(term):
                    case Alphanumeric("q")|Alphanumeric("Q"): 
                        mode = Mode.EXIT
                    case Alphanumeric("i")|Alphanumeric("I"): 
                        mode = Mode.PLAYGROUND 
                continue
            #-----------------------#
            #   MODE : Playground   #
            #-----------------------#
            elif mode == Mode.PLAYGROUND:
                ### Draw Playground
                flush(term.on_color_rgb(*BG) + term.clear)
                draw_bar(term)
                draw_playground(term)
                ### Process Input
                match parse_keystroke(term):
                    case Alphanumeric(key) | Printable(key):
                        ratify()
                        blocks[-1].append(key)
                    case Backspace():
                        ratify()
                        pop_block() if len(blocks[-1].string)==0 else blocks[-1].pop(1)
                    case Newline(): 
                        ratify()
                        if blocks[-1].string.strip() != "":
                            push_block()
                    case Esc(): 
                        mode = Mode.MENU
                    case Arrow() as arrow:
                        if arrow == Arrow.UP:
                            scroll(-1) 
                        elif arrow == Arrow.DOWN: 
                            scroll(1)
                        elif arrow == Arrow.LEFT:  
                            ...
                        else:#arrow == Arrow.RIGHT:
                            ...
                    case Unsupported(key):
                        print(F"<UNSUPPORTED:{key.code}>", end="")
                    case Empty():
                        continue
                    case _ as x:
                        assert_never(x)

if __name__ == "__main__":
    main()