HINTS = [
    "mean = x.mean(-1, keepdim=True); var = ((x-mean)**2).mean(-1, keepdim=True) (biased, dividing by D).",
    "standardize: xhat = (x - mean) / torch.sqrt(var + eps).",
    "affine: return xhat * gamma + beta. gamma/beta of shape (D,) broadcast automatically to (...,D).",
]

# ===== reference solution =====
import torch


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    xhat = (x - mean) / torch.sqrt(var + eps)
    return xhat * gamma + beta
