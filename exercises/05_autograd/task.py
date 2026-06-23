"""05 · Autograd
Level: 0

PyTorch computes gradients for you automatically -- but interviews ask "how does it work under the hood" and have you write a "gradient check."

The three core steps:
  1. x.requires_grad_(True) makes the tensor participate in differentiation (it becomes a leaf node of the computation graph).
  2. The forward pass produces a scalar loss; call loss.backward().
  3. Gradients accumulate in x.grad.

Gradient check: use numerical differencing to approximate the true gradient, verifying that backprop is correct.
  Central difference: df/dx_i ≈ (f(x + eps·e_i) - f(x - eps·e_i)) / (2·eps)

## Your task
Run: python practice.py check 05
"""
import torch
from typing import Callable


def grad_of(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Use autograd to compute the gradient df/dx of f at x, returning a tensor with the same shape as x.
    f takes a tensor and returns a scalar tensor. Do not modify the passed-in x.
    Hint: first detach().clone() a copy and turn on requires_grad."""
    raise NotImplementedError("TODO: clone -> requires_grad_ -> f -> backward -> .grad")


def numerical_grad(f: Callable[[torch.Tensor], torch.Tensor],
                   x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Estimate df/dx numerically with central differences, returning a tensor with the same shape as x.
    For each element i of x: (f(x with +eps at i) - f(x with -eps at i)) / (2*eps).
    A for loop over elements is allowed here (numerical gradients inherently perturb element by element)."""
    raise NotImplementedError("TODO: element-by-element +eps/-eps, central difference")
