"""10 · RMSNorm from scratch
Level: 1

Modern large models (LLaMA, Qwen, etc.) use RMSNorm instead of LayerNorm: cheaper and faster.
The difference: **no mean subtraction, no beta**, just root-mean-square (RMS) scaling:

    RMS(x) = sqrt( mean(x^2) + eps )      # along the last dim
    y = x / RMS(x) * gamma

Interviewers often ask about LayerNorm vs RMSNorm: RMSNorm drops "re-centering" (mean subtraction)
and only does "re-scaling"; in practice it performs comparably but more efficiently.

## Your task
Run: python practice.py check 10
"""
import torch


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x shape (..., D), gamma shape (D,). RMS-normalize along the last dim, then scale per channel."""
    raise NotImplementedError("TODO: x / sqrt(mean(x^2)+eps) * gamma")
