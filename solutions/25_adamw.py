HINTS = [
    "self.t += 1. Then for i, p in enumerate(self.params): g = p.grad; skip if g is None.",
    "Put decoupled weight decay first: p.mul_(1 - self.lr * self.wd).",
    "Update m, v: self.m[i].mul_(self.b1).add_(g, alpha=1-self.b1); self.v[i].mul_(self.b2).addcmul_(g, g, value=1-self.b2).",
    "Bias correction and update: mhat = self.m[i]/(1-self.b1**self.t); vhat = self.v[i]/(1-self.b2**self.t); p.addcdiv_(mhat, vhat.sqrt()+self.eps, value=-self.lr).",
]

# ===== reference solution =====
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
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            # 1) decoupled weight decay
            p.mul_(1 - self.lr * self.wd)
            # 2,3) first/second moments
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            # 4) bias correction
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            # 5) update
            p.addcdiv_(mhat, vhat.sqrt() + self.eps, value=-self.lr)
