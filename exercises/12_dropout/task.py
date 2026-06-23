"""12 · Dropout (inverted dropout)
Level: 1

During training, dropout randomly "drops" some neurons (sets them to 0) to prevent overfitting;
at inference it drops nothing. Modern implementations use **inverted dropout**: during training
the kept values are scaled up by 1/(1-p), so nothing needs to be done at inference (the expectation
is already aligned).

Rules:
  • training=False or p==0: return x unchanged (identity).
  • training=True: independently set each element to 0 with probability p; multiply kept elements by 1/(1-p).
  This makes E[output] = x, so the training/inference distributions match.

Interview follow-up: why divide by (1-p)? → to keep the expectation unchanged, so no scaling is needed at inference.

## Your task
Run: python practice.py check 12
"""
import torch


def dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """inverted dropout. p is the drop probability (0<=p<1)."""
    raise NotImplementedError("TODO: see the rules above; use torch.rand_like to generate the mask")
