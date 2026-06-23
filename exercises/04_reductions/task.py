"""04 · Reductions & statistics
Level: 0

A reduction "collapses" along some dimensions: sum / mean / var / max ...
These are the basis for LayerNorm, RMSNorm, and gradient clipping.

Key points:
  • x.mean(dim=d, keepdim=True) takes the mean along dim d and keeps that dim (it becomes 1).
  • There are two kinds of variance: biased (/N, population) and unbiased (/(N-1)).
    Normalization layers (LayerNorm/RMSNorm) use the [biased] variance = ((x-mean)**2).mean().
  • Gradient clipping first computes the "global L2 norm" of all parameter gradients concatenated together.

## Your task
Run: python practice.py check 04
"""
import torch
from typing import List, Tuple


def mean_and_var(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean and [biased] variance along dim, both keeping the dimension (keepdim=True).
    Return (mean, var). var = mean((x - mean)**2), dividing by N not N-1."""
    raise NotImplementedError("TODO: mean = x.mean(...); var = ((x-mean)**2).mean(...)")


def standardize(x: torch.Tensor, dim: int, eps: float = 1e-5) -> torch.Tensor:
    """Standardize along dim: (x - mean) / sqrt(var + eps), using the biased variance.
    This is the core step of LayerNorm."""
    raise NotImplementedError("TODO: reuse mean_and_var")


def global_grad_norm(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Given a list of tensors (simulating each parameter's gradient), return the global L2 norm
    of them concatenated (a scalar tensor): sqrt( sum over all tensors of sum(t**2) ). This is the first step of gradient clipping."""
    raise NotImplementedError("TODO: accumulate each tensor's sum of squares, then take the square root")
