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