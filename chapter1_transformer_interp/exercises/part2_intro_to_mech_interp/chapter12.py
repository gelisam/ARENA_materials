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
    permanent_hooks = [],
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
    logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=permanent_hooks
    )
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            logits = model.run_with_hooks(
                tokens,
                return_type='logits',
                fwd_hooks=permanent_hooks+[(utils.get_act_name(neuron_name, layer), functools.partial(ablation_function, head_index_to_ablate=head))],
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

rep_tokens_batch, _, _ = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=10)
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

# %%

# There is an effect on 0.11, so head 0.11 does attend to itself. Perhaps it is
# computing a calculation, since there are no MLP layers?

# %%

def head_self_attention_override_hook(
    attn_scores: Float[Tensor, "batch n_heads seqQ seqK"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    batch, n_heads, seqQ, seqK = attn_scores.shape
    attn_scores[:, head_index_to_ablate, :, :] = t.where(t.eye(seqQ) == 1, 1.0, -float('inf'))
    return attn_scores

# %%

self_attention_override_scores = get_ablation_scores(
    model,
    rep_tokens_batch,
    ablation_function=head_self_attention_override_hook,
    neuron_name = "attn_scores"
)

# %%

imshow(
    self_attention_override_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Forcing Self-Attention",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%

# Forcing self-attention _improves_ the loss, so head 0.11 is trying to learn to
# compute something about its own position. But what?
#
# Well, there are only two pieces of data at that position: the token's
# embedding, and the position's embedding. It's probably the position.

# %%

def record_to_cell(
    x: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    cell: list[Float[Tensor, "batch seq n_heads d_head"]]
):
    cell[0] = x

def zero_out_embed(
    embed: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
):
    return embed * 0.0

recorded_q = [None]
recorded_k = [None]
recorded_v = [None]

model.run_with_hooks(
    rep_tokens,
    return_type=None,
    fwd_hooks=[
        ("hook_embed", zero_out_embed),
        (utils.get_act_name("q", 0), functools.partial(record_to_cell, cell=recorded_q)),
        (utils.get_act_name("k", 0), functools.partial(record_to_cell, cell=recorded_k)),
        (utils.get_act_name("v", 0), functools.partial(record_to_cell, cell=recorded_v)),
    ],
)
q_without_token = recorded_q[0]
k_without_token = recorded_k[0]
v_without_token = recorded_v[0]

model.run_with_hooks(
    rep_tokens,
    return_type=None,
    fwd_hooks=[
        ("hook_pos_embed", zero_out_embed),
        (utils.get_act_name("q", 0), functools.partial(record_to_cell, cell=recorded_q)),
        (utils.get_act_name("k", 0), functools.partial(record_to_cell, cell=recorded_k)),
        (utils.get_act_name("v", 0), functools.partial(record_to_cell, cell=recorded_v)),
    ],
)
q_without_pos = recorded_q[0]
k_without_pos = recorded_k[0]
v_without_pos = recorded_v[0]
# %%

def ablate_info(
    x: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
    without_info: Float[Tensor, "batch seq n_heads d_head"],
) -> None:
    x[:,:,head_index_to_ablate] = without_info[:,:,head_index_to_ablate]

q_without_token_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=q_without_token),
    neuron_name = "q"
)
k_without_token_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=k_without_token),
    neuron_name = "k"
)
v_without_token_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=v_without_token),
    neuron_name = "v"
)
q_without_pos_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=q_without_pos),
    neuron_name = "q"
)
k_without_pos_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=k_without_pos),
    neuron_name = "k"
)
v_without_pos_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=v_without_pos),
    neuron_name = "v"
)

# %%

imshow(
    q_without_token_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping tokens from Q",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    k_without_token_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping tokens from K",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    v_without_token_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping tokens from V",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    q_without_pos_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping pos from Q",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    k_without_pos_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping pos from K",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    v_without_pos_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping pos from V",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%

# Head 0.11 attends to itself, so messing with Q or K but not both is increasing
# the loss, but that does not give us new information. What is more interesting
# is that dropping the token information from V increases the loss, while
# removing the position does not have any effect. This would allow us to
# conclude that head 0.11 is computing something about its token, except that
# ablating the position had zero effect on every head in layer 0, which seems
# very unlikely! Is the position not given to the V matrix?
#
# Ah, indeed, the document said this earlier:
# > The positional embeddings are only added to the residual stream before
# > calculating each key and query vector in the attention layers as opposed to
# > the token embeddings - i.e. we compute queries as Q = (resid + pos_embed) @
# > W_Q + b_Q and same for keys, but values as V = resid @ W_V + b_V. This means
# > that the residual stream can't directly encode positional information.

# %%

# New approach: let's find a circuit! Which heads in layer 1 depend on 0.11?
# Let's see what happens when each layer 1 head receives a residual which does
# not include 0.11's contribution.

recorded_q = [None]
recorded_k = [None]
recorded_v = [None]

model.run_with_hooks(
    rep_tokens,
    return_type=None,
    fwd_hooks=[
        (utils.get_act_name("z", 0), functools.partial(head_zero_ablation_hook, head_index_to_ablate=11)),
        (utils.get_act_name("q", 1), functools.partial(record_to_cell, cell=recorded_q)),
        (utils.get_act_name("k", 1), functools.partial(record_to_cell, cell=recorded_k)),
        (utils.get_act_name("v", 1), functools.partial(record_to_cell, cell=recorded_v)),
    ],
)

q_without_head011 = recorded_q[0]
k_without_head011 = recorded_k[0]
v_without_head011 = recorded_v[0]

# %%

def ablate_info(
    x: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
    without_info: Float[Tensor, "batch seq n_heads d_head"],
) -> None:
    if hook.layer() == 1:
        x[:,:,head_index_to_ablate] = without_info[:,:,head_index_to_ablate]
        return x

q_without_head011_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=q_without_head011),
    neuron_name = "q"
)
k_without_head011_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=k_without_head011),
    neuron_name = "k"
)
v_without_head011_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=v_without_head011),
    neuron_name = "v"
)

# %%

imshow(
    q_without_head011_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping head 0.11 from Q",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    k_without_head011_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping head 0.11 from K",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    v_without_head011_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping head 0.11 from V",
    text_auto=".2f",
    width=900,
    height=350,
)
# %%

# Head 1.10 (and to a lesser extend 1.4) uses the value computed by head 0.11 in
# order to construct its own V. Those are precisely the induction heads, so we
# are on the right track!

# In an induction circuit, the layer 0 head uses the position embedding to
# obtain the token at the previous position. Then, the layer 1 head uses that
# previous position as the K, so that Q can find the position whose preceding
# token is the same as Q's, and return that position's token.
#
# Layer 0's task is difficult, because the position embedding for the previous
# token is very similar to the position embedding of the current token. Thus, it
# seems likely that layer 0 will return a mix of the previous and current token,
# and thus that layer 1 will find a mix of the correct position (the one whose
# _previous_ position holds Q's token) and a few incorrect positions: those
# whose _current_ position hold's Q's token. Both incorrect cases have a V which
# is the same as Q's token, so layer 1's result will be a mix of the correct
# token and Q's token. It thus makes sense to take this mixture, and to reduce
# the probability of the current token. Might this be how 1.10 uses 0.11's
# output?
#
# To determine that, let's ablate 0.11 again, but this time let's also do its
# work ourselves, by reducing the probability of Q's token in the final logits.
# If I am right, then the effect of ablating 0.11 should disappear.

# %%

a = t.tensor([[[1,2],[3,4]], [[4,5],[6,7]]])
b = t.tensor([1, 0])
r = t.arange(a.shape[0])
a[:, r, b] = 0
print(a)

# %%

def never_return_self_token(
      out: Float[Tensor, "batch seq d_vocab"],
      hook: HookPoint
):
    # rep_tokens :: batch seq
    r = t.arange(out.shape[1])
    out[:, r, rep_tokens] = -float('inf')
    return out

q_without_head011_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=q_without_head011),
    permanent_hooks=[("unembed.hook_out", never_return_self_token)],
    neuron_name = "q"
)
k_without_head011_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=k_without_head011),
    permanent_hooks=[("unembed.hook_out", never_return_self_token)],
    neuron_name = "k"
)
v_without_head011_scores = get_ablation_scores(
    model,
    rep_tokens,
    ablation_function=functools.partial(ablate_info, without_info=v_without_head011),
    permanent_hooks=[("unembed.hook_out", never_return_self_token)],
    neuron_name = "v"
)

# %%

imshow(
    q_without_head011_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping head 0.11 from Q",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    k_without_head011_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping head 0.11 from K",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    v_without_head011_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping head 0.11 from V",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%

# No effect, I was wrong. Maybe it has a direct effect on the final logits, via
# the residual stream?

mean_ablation_scores = get_ablation_scores(
    model,
    rep_tokens_batch,
    ablation_function=head_mean_ablation_hook,
    permanent_hooks=[("unembed.hook_out", never_return_self_token)]
)

# %%

imshow(
    mean_ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Dropping Head Mean",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%

# No effect either. What else could it be?
#
# Well, the loss is different. So it must be affecting the final logits in some
# way. Let's examine what that effect is.

all_tokens_batch = rep_tokens_batch # batch seq

# batch seq d_vocab
all_logits = model(all_tokens_batch, return_type="logits")
all_logits_without011 = model.run_with_hooks(
    all_tokens_batch,
    return_type="logits",
    fwd_hooks=[(utils.get_act_name("result", 0), functools.partial(head_zero_ablation_hook, head_index_to_ablate=11))],
)

# %%

batch, seq = all_tokens_batch.shape

# batch seq d_vocab
all_differences = all_logits - all_logits_without011

# batch seq-1
correct_token = all_tokens_batch[:, 1:]
current_token = all_tokens_batch[:, :-1]

# batch seq-1 d_vocab
logits = all_logits[:, :-1]
differences = all_differences[:, :-1]

batch_ix = einops.repeat(t.arange(batch), "batch -> batch seq", seq=seq-1)
seq_ix = einops.repeat(t.arange(seq-1), "seq -> batch seq", batch=batch)

# batch seq-1
current_token_impact = differences[batch_ix, seq_ix, current_token]
correct_token_impact = differences[batch_ix, seq_ix, correct_token]

# seq-1
current_token_impact = current_token_impact.mean(dim=0)
correct_token_impact = correct_token_impact.mean(dim=0)

# 2 seq-1
results = t.stack([current_token_impact, correct_token_impact], dim=0)

imshow(results)

# %%

# Result: it increases the correct token's logit, it does not decrease the
# current token's logit. Darn.
#
# I am giving up and reading the hint. "Can you figure out which source
# positions are important to attend to for the model to perform well?" That's
# not much of a hint, I already know that 0.4 attends to the last few positions,
# and that 0.11 attends to itself.
#
# Let's read the solution. "Below is some sample code which plots the effect of
# ablating the inputs to heads 0.4 and 0.11 at all offset positions minus a few
# (e.g. the first row shows the effect on loss of mean ablating all inputs to
# the heads except for those that come from self-attention, and the second row
# shows the effect when we ablate all inputs except for those that come from the
# token immediately before it in the sequence)." Let me at least write that code
# myself.

def hook_self_attention_only(
    attn_scores: Float[Tensor, "batch n_heads seqQ seqK"],
    hook: HookPoint,
    head_index_to_ablate: int,
):
    batch, n_heads, seqQ, seqK = attn_scores.shape
    q_ix = einops.repeat(
        t.arange(seqQ),
        'seqQ -> seqQ seqK',
        seqK=seqK
    )
    k_ix = einops.repeat(
        t.arange(seqK),
        'seqK -> seqQ seqK',
        seqQ=seqQ
    )
    attn_scores[:,head_index_to_ablate] = t.where(q_ix == k_ix, 1.0, -float('inf'))
    return attn_scores

def hook_prev_attention_only(
    attn_scores: Float[Tensor, "batch n_heads seqQ seqK"],
    hook: HookPoint,
    head_index_to_ablate: int,
):
    batch, n_heads, seqQ, seqK = attn_scores.shape
    q_ix = einops.repeat(
        t.arange(seqQ),
        'seqQ -> seqQ seqK',
        seqK=seqK
    )
    k_ix = einops.repeat(
        t.arange(seqK),
        'seqK -> seqQ seqK',
        seqQ=seqQ
    )
    attn_scores[:,head_index_to_ablate] = t.where(k_ix == q_ix - 1, 1.0, -float('inf'))
    return attn_scores

self_attention_only_scores = get_ablation_scores(
    model,
    rep_tokens_batch,
    ablation_function=hook_self_attention_only,
    neuron_name = "attn_scores"
)

prev_attention_only_scores = get_ablation_scores(
    model,
    rep_tokens_batch,
    ablation_function=hook_prev_attention_only,
    neuron_name = "attn_scores"
)

# %%

imshow(
    self_attention_only_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After forcing self-attention",
    text_auto=".2f",
    width=900,
    height=350,
)
imshow(
    prev_attention_only_scores,
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After forcing prev-attention",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%

# Unsurprisingly, making head 0.11 attend to itself is helping, not hindering,
# since that is the head's goal, while making it attend to the previous token
# does have a consequence since it prevents that goal.
#
# I didn't look much at head 0.4 yet, but I do find it surprising is that head
# 0.4 is affected by both changes, as I expected its job to be attending the
# previous token.
#
# Let's try the solution's code:

def head_z_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
    seq_posns: list[int],
    cache: ActivationCache,
) -> None:
    """
    We perform ablation at the z vector, by doing the equivalent of mean ablating all the inputs to this attention head
    except for those which come from the tokens `n` positions back, where `n` is in the `seq_posns` list.
    """
    batch, seq = z.shape[:2]
    v = cache["v", hook.layer()][:, :, head_index_to_ablate]  # shape [batch seq_K d_head]
    pattern = cache["pattern", hook.layer()][:, head_index_to_ablate]  # shape [batch seq_Q seq_K]

    # Get a repeated version of v, and mean ablate all but the previous token values
    v_repeated = einops.repeat(v, "b sK h -> b sQ sK h", sQ=seq)
    v_ablated = einops.repeat(v_repeated.mean(0), "sQ sK h -> b sQ sK h", b=batch).clone()
    for offset in seq_posns:
        seqQ_slice = t.arange(offset, seq)
        v_ablated[:, seqQ_slice, seqQ_slice - offset] = v_repeated[:, seqQ_slice, seqQ_slice - offset]

    # Take weighted sum of this new v, and use it to edit `z` inplace.
    z[:, :, head_index_to_ablate] = einops.einsum(v_ablated, pattern, "b sQ sK h, b sQ sK -> b sQ h")


def get_ablation_scores_cache_assisted(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
    seq_posns: list[int] = [0],
    layers: list[int] = [0],
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Version of `get_ablation_scores` which can use the cache to assist with the ablation.
    """
    ablation_scores = t.zeros((len(layers), model.cfg.n_heads), device=model.cfg.device)

    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits, cache = model.run_with_cache(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for i, layer in enumerate(layers):
        for head in range(model.cfg.n_heads):
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head, cache=cache, seq_posns=seq_posns)
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)])
            loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
            ablation_scores[i, head] = loss - loss_no_ablation

    return ablation_scores


rep_tokens_batch = run_and_cache_model_repeated_tokens(model, seq_len=50, batch_size=50)[0]

offsets = [[0], [1], [2], [3], [1, 2], [1, 2, 3]]
z_ablation_scores = [
    get_ablation_scores_cache_assisted(model, rep_tokens_batch, head_z_ablation_hook, offset).squeeze()
    for offset in tqdm(offsets)
]

imshow(
    t.stack(z_ablation_scores),
    labels={"x": "Head", "y": "Position offset", "color": "Loss diff"},
    title="Loss Difference (ablating heads everywhere except for certain offset positions)",
    text_auto=".2f",
    y=[str(offset) for offset in offsets],
    width=900,
    height=400,
)

# %%

# Looks like the solution didn't manage to go further than I did, at least about
# 0.11? They are just better at interpreting the results!