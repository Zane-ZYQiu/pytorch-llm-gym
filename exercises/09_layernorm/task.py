"""09 · LayerNorm from scratch
Level: 1

LayerNorm standardizes **each token's own feature vector** (along the last dim D),
independent of the batch——this is crucial for variable-length sequences.

    y = (x - mean) / sqrt(var + eps) * gamma + beta
where mean and var are computed along the last dim (var is biased, dividing by D);
gamma and beta are learnable per-channel scale/shift, both of shape (D,).

## Your task
Run: python practice.py check 09
"""
import torch


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """x shape (..., D), gamma/beta shape (D,). Standardize along the last dim, then apply the affine transform."""
    raise NotImplementedError("TODO: compute mean/var along -1, standardize, then *gamma + beta")
