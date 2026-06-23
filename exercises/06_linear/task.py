"""06 · Linear Layer (forward + hand-derived backward)
Level: 1

The linear layer y = xW^T + b is the foundation of everything. A classic interview
question: "Derive its backward pass by hand."

Conventions (matching nn.Linear):
  • x shape (B, in)    one batch, B samples
  • W shape (out, in)  weights (note: out comes first!)
  • b shape (out,)     bias
  • y = x @ W.T + b   shape (B, out)

Backward (let the upstream gradient dL/dy = grad_out, shape (B, out)):
  • dL/dx = grad_out @ W            shape (B, in)
  • dL/dW = grad_out.T @ x          shape (out, in)
  • dL/db = grad_out.sum(over batch) shape (out,)
Being able to write these three from memory is almost guaranteed to come up.

## Your task
Run: python practice.py check 06
"""
import torch
from typing import Tuple


def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return y = x @ W.T + b, shape (B, out)."""
    raise NotImplementedError("TODO: x @ W.t() + b")


def linear_backward(x: torch.Tensor, W: torch.Tensor,
                    grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-written backward. Given the upstream gradient grad_out (B, out),
    return (grad_x, grad_W, grad_b) with shapes (B,in), (out,in), (out,) respectively."""
    raise NotImplementedError("TODO: three lines of matmul / sum")
