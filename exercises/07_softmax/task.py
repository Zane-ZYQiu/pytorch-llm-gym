"""07 · Numerically stable softmax / log-softmax / logsumexp
Level: 1

softmax(x)_i = exp(x_i) / sum_j exp(x_j).
Applying the formula directly overflows: if x contains a 1000, exp(1000)=inf.

The life-saving trick: **subtract the max**. Because softmax is shift-invariant:
    softmax(x) = softmax(x - max(x))
After subtracting the max, the largest exponent becomes exp(0)=1, so nothing overflows.
This "numerical stability" is a guaranteed interview topic.

  • logsumexp(x) = max(x) + log( sum exp(x - max(x)) )   same trick
  • log_softmax(x) = x - logsumexp(x)                    more stable than log(softmax(x))

## Your task (handle numerical stability yourself; do not call torch.softmax / logsumexp)
Run: python practice.py check 07
"""
import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax along dim."""
    raise NotImplementedError("TODO: subtract max -> exp -> normalize")


def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable logsumexp along dim. The dim axis is removed from the result (not kept)."""
    raise NotImplementedError("TODO: m + log(sum(exp(x-m))), remember to drop dim at the end")


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-softmax along dim."""
    raise NotImplementedError("TODO: x - logsumexp(x, dim, keepdim)")
