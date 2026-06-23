HINTS = [
    "mean_and_var: mean = x.mean(dim, keepdim=True); var = ((x - mean)**2).mean(dim, keepdim=True).",
    "Note this is the biased variance (dividing by N), so use .mean() directly instead of .sum()/(N-1). Equivalent to x.var(dim, unbiased=False).",
    "standardize: m, v = mean_and_var(x, dim); return (x - m) / torch.sqrt(v + eps).",
    "global_grad_norm: total = sum((t**2).sum() for t in tensors); return total.sqrt(). Equivalent to torch.cat([t.flatten()...]).norm().",
]

# ===== reference solution =====
import torch
from typing import List, Tuple


def mean_and_var(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=dim, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dim, keepdim=True)
    return mean, var


def standardize(x: torch.Tensor, dim: int, eps: float = 1e-5) -> torch.Tensor:
    mean, var = mean_and_var(x, dim)
    return (x - mean) / torch.sqrt(var + eps)


def global_grad_norm(tensors: List[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros(())
    for t in tensors:
        total = total + (t.float() ** 2).sum()
    return torch.sqrt(total)
