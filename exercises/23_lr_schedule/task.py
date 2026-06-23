"""23 · Learning rate schedule (warmup + cosine) and gradient clipping
Level: 4

The standard learning-rate curve for large-model training: first a **linear warmup** (rising from 0 to base_lr),
then a **cosine decay** (smoothly dropping to min_lr). Warmup prevents large gradients from derailing training at the start.

  warmup phase (step < warmup): lr = base_lr * step / warmup
  cosine phase:                 progress = (step-warmup)/(total-warmup)
                                lr = min_lr + (base_lr-min_lr) * 0.5*(1+cos(pi*progress))

**Gradient clipping**: treat all gradients as one big vector; if its L2 norm exceeds max_norm,
scale the whole thing down proportionally to max_norm (prevents gradient explosion).

## Your task
Run: python practice.py check 23
"""
import math
import torch
from typing import List


def lr_at_step(step: int, base_lr: float, warmup_steps: int,
               total_steps: int, min_lr: float = 0.0) -> float:
    """Return the learning rate at step `step` (starting from 0)."""
    raise NotImplementedError("TODO: linear warmup, then cosine decay")


def clip_grad_norm_(grads: List[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Clip in place. grads is a list of gradient tensors, one per parameter.
    Compute the global L2 norm; if > max_norm, multiply each gradient by max_norm/total_norm.
    Return the norm before clipping (scalar tensor)."""
    raise NotImplementedError("TODO: compute total norm -> scale in place with mul_ if needed -> return original norm")
