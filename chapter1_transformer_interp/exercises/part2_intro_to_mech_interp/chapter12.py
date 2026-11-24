# %%

import arena_lib

exercises_dir, section_dir = arena_lib.setup("1.2")

# %%

import functools
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

import part2_intro_to_mech_interp.tests as tests
from arena_lib.plot import hist, imshow
from arena_lib.circuits_vis import attention_patterns
from plotly_utils import plot_comp_scores, plot_logit_attribution, plot_loss_difference

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# %%

# skipping everything which isn't required for the Bonus, because I already did those at https://colab.research.google.com/github/gelisam/colab/blob/main/tara/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb#scrollTo=2stIZawZtId-

# %%

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
    device=device,
)

# %%

from huggingface_hub import hf_hub_download

REPO_ID = "arena-education/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# %%

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

# %%

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# %%

def generate_repeated_tokens(
    model: HookedTransformer, half_seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    half = (t.rand(batch_size, half_seq_len) * model.cfg.d_vocab).int()
    # YOUR CODE HERE - build `rep_tokens` from `prefix` and two copies of a random half, and return it
    return t.cat([prefix, half, half], dim=1)


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    tokens = generate_repeated_tokens(model, seq_len, batch_size)  # b seq
    logits, cache = model.run_with_cache(tokens)  # b seq d_vocab
    return (tokens, logits, cache)


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs

tokens = generate_repeated_tokens(model, 4)
print(tokens)
print(model.to_str_tokens(tokens))


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

# %%

def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l0_results: Float[Tensor, "seq nheads d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l0_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
        l1_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]  # d_model seq-1
    direct_path_attrs = einops.repeat(
        einops.einsum(
           embed[:-1],
            W_U_correct_tokens,
            'seq_1 d_model, d_model seq_1 -> seq_1'
        ),
        'seq_1 -> seq_1 1'
    )
    layer0_attrs = einops.einsum(
        l0_results[:-1],
        W_U_correct_tokens,
        'seq_1 n_heads d_model, d_model seq_1 -> seq_1 n_heads'
    )
    layer1_attrs = einops.einsum(
        l1_results[:-1],
        W_U_correct_tokens,
        'seq_1 n_heads d_model, d_model seq_1 -> seq_1 n_heads'
    )

    return t.cat([direct_path_attrs, layer0_attrs, layer1_attrs], dim=-1)

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")

# %%

def head_zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = t.zeros((1,))
    return z


def get_ablation_scores(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
    neuron_name: str = "z"
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss
    from ablating the output of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            logits = model.run_with_hooks(
                tokens,
                return_type='logits',
                fwd_hooks=[(utils.get_act_name(neuron_name, layer), functools.partial(ablation_function, head_index_to_ablate=head))],
            )  # b seq n_vocab
            loss_with_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()
            ablation_scores[layer, head] = loss_with_ablation - loss_no_ablation

    return ablation_scores

# z = t.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(z)
# z[:, 1, :] = t.zeros((1,))
# print(z)


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
# %%

imshow(
    ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%

def head_mean_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> None:
    batch, seq, n_heads, d_head = z.shape
    z[:, :, head_index_to_ablate, :] = einops.repeat(
        einops.reduce(
            z[:, :, head_index_to_ablate, :],
            'batch seq d_head -> seq d_head',
            'mean'
        ),
        'seq d_head -> batch seq d_head',
        batch=batch,
    )
    return z

tests.test_head_mean_ablation_hook(head_mean_ablation_hook)

# %%

rep_tokens_batch = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=10)[0]
mean_ablation_scores = get_ablation_scores(
    model,
    rep_tokens_batch,
    ablation_function=head_mean_ablation_hook
)

# %%

imshow(
    mean_ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)

# %% Bonus: understand heads 0.4 and 0.11

display(
    cv.attention.attention_patterns(
        tokens=model.to_str_tokens(rep_tokens[0]),
        attention=rep_cache["pattern", 0],
    )
)

# %%

def head_self_attention_ablation_hook(
    attn_scores: Float[Tensor, "batch n_heads seqQ seqK"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    batch, n_heads, seqQ, seqK = attn_scores.shape
    attn_scores[:, head_index_to_ablate, :, :] -= t.eye(seqQ) * float('inf')
    return attn_scores

# %%

rep_tokens_batch = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=10)[0]
self_attention_ablation_scores = get_ablation_scores(
    model,
    rep_tokens_batch,
    ablation_function=head_self_attention_ablation_hook,
    neuron_name = "attn_scores"
)

# %%

imshow(
    self_attention_ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Ablating Self-Attention",
    text_auto=".2f",
    width=900,
    height=350,
)
