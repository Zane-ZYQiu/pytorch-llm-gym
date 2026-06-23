HINTS = [
    "pairwise_sq_dist: diff = a.unsqueeze(1) - b.unsqueeze(0)  # (N,M,D); then (diff**2).sum(-1).",
    "outer_sum: a.unsqueeze(1) + b.unsqueeze(0), i.e. a[:,None] + b[None,:].",
    "normalize_last_dim: x / x.norm(dim=-1, keepdim=True).clamp_min(eps). keepdim lets the shape broadcast back.",
    "masked_mean: m = mask.unsqueeze(-1).to(x.dtype); numerator (x*m).sum(1), denominator m.sum(1).clamp_min(1).",
]

# ===== reference solution =====
import torch


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a.unsqueeze(1) - b.unsqueeze(0)        # (N, M, D)
    return (diff ** 2).sum(dim=-1)                # (N, M)


def outer_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.unsqueeze(1) + b.unsqueeze(0)


def normalize_last_dim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)            # (B, T, 1)
    summed = (x * m).sum(dim=1)                   # (B, D)
    count = m.sum(dim=1).clamp_min(1.0)           # (B, 1)
    return summed / count
