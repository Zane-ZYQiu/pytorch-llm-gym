"""25 · Implement AdamW from Scratch
Level: 5

Adam = momentum (first moment m) + adaptive learning rate (second moment v) + bias correction.
AdamW = Adam + **decoupled weight decay** (weight decay acts directly on the parameters, not mixed into the gradient).

Single update step (t starts from 1):
  1. Decoupled weight decay:  θ ← θ * (1 - lr * weight_decay)
  2. m ← β1·m + (1-β1)·g
  3. v ← β2·v + (1-β2)·g²
  4. Bias correction:      m̂ = m/(1-β1^t),  v̂ = v/(1-β2^t)
  5. Update:          θ ← θ - lr · m̂ / (√v̂ + eps)

Interview point: why bias correction? → m and v start at 0, so early estimates are too small and need amplifying to compensate.
Why is AdamW better than Adam? → mixing L2 regularization into the adaptive learning rate distorts it via scaling; decoupling is cleaner.

__init__ and zero_grad are given. You implement step().

## Your task
Run: python practice.py check 25
"""
import torch
from typing import Iterable


class AdamW:
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        """Update each parameter in place using the 5 steps above. Parameter gradients are in p.grad."""
        raise NotImplementedError("TODO: t+=1, run the 5 AdamW update steps for each p")
