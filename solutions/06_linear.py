HINTS = [
    "forward: x @ W.t() + b. W is (out,in); transposed it's (in,out), so (B,in)@(in,out)=(B,out).",
    "grad_x: chain rule, y=xW^T, so grad_x = grad_out @ W, (B,out)@(out,in)=(B,in).",
    "grad_W: grad_out.t() @ x, (out,B)@(B,in)=(out,in). Note how the dimensions line up.",
    "grad_b: b is broadcast-added to every sample, so the gradient sums over the batch dim: grad_out.sum(dim=0).",
]

# ===== reference solution =====
import torch
from typing import Tuple


def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x @ W.t() + b


def linear_backward(x: torch.Tensor, W: torch.Tensor,
                    grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_x = grad_out @ W           # (B, out) @ (out, in) -> (B, in)
    grad_W = grad_out.t() @ x       # (out, B) @ (B, in)  -> (out, in)
    grad_b = grad_out.sum(dim=0)    # (out,)
    return grad_x, grad_W, grad_b
