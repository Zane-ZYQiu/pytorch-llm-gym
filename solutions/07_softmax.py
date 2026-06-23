HINTS = [
    "softmax: m = x.max(dim, keepdim=True).values; e = (x - m).exp(); return e / e.sum(dim, keepdim=True).",
    "x.max(dim, keepdim=True) returns a (values, indices) named tuple; take .values.",
    "logsumexp: m = x.max(dim, keepdim=True).values; result = m + (x-m).exp().sum(dim, keepdim=True).log(). Finally .squeeze(dim) to drop that axis.",
    "log_softmax: subtract directly using the keepdim logsumexp: x - (m + log(sum(exp(x-m)))); keep keepdim=True throughout so it broadcasts.",
]

# ===== reference solution =====
import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)


def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    out = m + (x - m).exp().sum(dim=dim, keepdim=True).log()
    return out.squeeze(dim)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    return z - z.exp().sum(dim=dim, keepdim=True).log()
