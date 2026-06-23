HINTS = [
    "grad_of: x = x.detach().clone().requires_grad_(True); y = f(x); y.backward(); return x.grad.",
    "Why clone: turning on requires_grad directly on the passed-in x would pollute the caller, and x may already be part of another computation graph.",
    "numerical_grad: first x = x.detach().clone(); grad = torch.zeros_like(x). view(-1) them into 1-D for easy element-by-element editing.",
    "For each i: save the original value -> set to +eps and compute f -> set to -eps and compute f -> restore -> grad_flat[i] = (fp - fm)/(2*eps). Use .item() to get the scalar.",
]

# ===== reference solution =====
import torch
from typing import Callable


def grad_of(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    y = f(x)
    y.backward()
    return x.grad


def numerical_grad(f: Callable[[torch.Tensor], torch.Tensor],
                   x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.detach().clone()
    grad = torch.zeros_like(x)
    xf = x.view(-1)
    gf = grad.view(-1)
    for i in range(xf.numel()):
        orig = xf[i].item()
        xf[i] = orig + eps
        fp = f(x).item()
        xf[i] = orig - eps
        fm = f(x).item()
        xf[i] = orig
        gf[i] = (fp - fm) / (2 * eps)
    return grad
