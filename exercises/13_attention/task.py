"""13 · Scaled Dot-Product Attention (single-head)
Level: 2

Attention is the heart of the Transformer. In one sentence: each query scores its
similarity against all keys, softmax turns those into weights, then take a weighted
sum of the values.

    scores = Q @ K^T / sqrt(d)        # (Tq, Tk), divide by sqrt(d) to keep magnitudes in check
    attn   = softmax(scores, dim=-1)  # each query's weights over all keys
    out    = attn @ V                  # (Tq, dv)

Interview follow-up: why divide by sqrt(d)? → at large dimensions the dot-product variance grows, so softmax saturates and gradients vanish.
Causal mask: in autoregressive generation, position i must not see the future j>i, so set those scores to -inf.

## Your task
Run: python practice.py check 13
"""
import math
import torch


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """q (B, Tq, d), k (B, Tk, d), v (B, Tk, dv). Returns (B, Tq, dv).
    When causal=True, apply the causal mask (assumes Tq == Tk)."""
    raise NotImplementedError("TODO: QK^T/sqrt(d) -> (optional causal mask) -> softmax -> @V")
