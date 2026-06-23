"""03 · Indexing / gather / scatter / masking
Level: 0

Language model loss, sampling, and attention masks all rely on "reading/writing values by index." Three key tools:
  • gather(dim, index): read values along a dimension at the indices given by index.
    In an LM, use it to pull the "true next token" entry out of the (B,T,V) logprob -> (B,T).
  • scatter_(dim, index, value): the inverse of gather, writing values at the given indices. Can hand-write one-hot.
  • masked_fill(mask, value): fill positions where mask is True with value.
    The causal mask in attention fills "future" positions with -inf, which become 0 after softmax.

Shape alignment point: the index for gather/scatter must have "the same number of dimensions" as the tensor being operated on,
so it's often paired with unsqueeze(-1) / squeeze(-1).

## Your task
Run: python practice.py check 03
"""
import torch


def gather_token_logprobs(logprobs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logprobs shape (B, T, V), targets shape (B, T) as long (the target token id at each position).
    Return (B, T), out[b,t] = logprobs[b, t, targets[b,t]]."""
    raise NotImplementedError("TODO: logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)")


def one_hot_scatter(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Hand-write one-hot using scatter_ (F.one_hot not allowed).
    indices shape (N,) as long. Return an (N, num_classes) float tensor,
    out[i, indices[i]] = 1, all others 0."""
    raise NotImplementedError("TODO: first zeros, then scatter_(1, indices.unsqueeze(1), 1.0)")


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Apply a causal mask to attention scores. scores shape (..., T, T),
    where the second-to-last dim is query position i and the last dim is key position j.
    Fill all positions with j > i (future) with -inf, and return a new tensor."""
    raise NotImplementedError("TODO: use torch.triu to build an upper-triangular mask, then masked_fill")
