"""14 · Multi-Head Attention (MHA)
Level: 2

Split the D dimension into H heads, let each head run attention independently in
its own head_dim = D/H subspace, then concatenate and pass through an output
projection. Intuition: different heads attend to different relationships (syntax / coreference / position...).

The full pipeline (this is what an interview expects you to write from memory):
  1. q = x @ Wq, k = x @ Wk, v = x @ Wv      # (B,T,D)
  2. split heads: (B,T,D) -> (B,T,H,hd) -> transpose -> (B,H,T,hd)
  3. run scaled dot-product attention per head (optionally with a causal mask)
  4. merge heads: (B,H,T,hd) -> (B,T,H,hd) -> (B,T,D)
  5. out = merged result @ Wo

Convention: Wq/Wk/Wv/Wo are all plain (D, D) matrices, used as x @ W.

## Your task
Run: python practice.py check 14
"""
import math
import torch


def multi_head_attention(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                         Wv: torch.Tensor, Wo: torch.Tensor,
                         num_heads: int, causal: bool = False) -> torch.Tensor:
    """x (B, T, D). Returns (B, T, D)."""
    raise NotImplementedError("TODO: project -> split heads -> attention -> merge heads -> output projection")
