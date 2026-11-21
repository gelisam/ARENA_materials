# %%

import arena_lib

exercises_dir, section_dir = arena_lib.setup("1.1")

# %%

import importlib.metadata
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from liveplot import LivePlot
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utilities import gelu_new, tokenize_and_concatenate
from transformers import GPT2TokenizerFast

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

import part1_transformer_from_scratch.solutions as solutions
import part1_transformer_from_scratch.tests as tests

MAIN = __name__ == "__main__"


# %%

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
    device=device,
)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()

# %%

print(reference_gpt2.to_str_tokens("Ralph"))
print(reference_gpt2.to_str_tokens(" Ralph"))
print(reference_gpt2.to_str_tokens(" ralph"))
print(reference_gpt2.to_str_tokens("ralph"))

# %%

print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))

# %%

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# %%

logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)

# %%

probs = logits.softmax(dim=-1)
print(probs.shape)

# %%

most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0].unsqueeze(-1))

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

# %%

next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))

# %%
print(next_token.item())
print(next_token)
print(next_token.shape)
print(next_token[None].shape)
print(next_token[None,None].shape)

# %%

# We use new variable names here, so that `tokens` & `logits` still refer to our original sequence in later cells
gen_tokens = tokens
print(f"Sequence so far: {reference_gpt2.to_string(gen_tokens)[0]!r}")

for i in range(10):
    print(f"Token {gen_tokens.shape[-1] + 1}: {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    gen_tokens = t.cat([gen_tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    gen_logits = reference_gpt2(gen_tokens)
    # Get the predicted token at the end of our sequence
    next_token = gen_logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)

# %%

for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")

# %%

for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

# %%

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)