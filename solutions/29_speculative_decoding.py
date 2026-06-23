HINTS = [
    "acceptance_prob: return torch.clamp(p[x] / q[x], max=1.0).",
    "residual_dist: r = torch.clamp(p - q, min=0.0); return r / r.sum().",
    "speculative_sample: ratio = (p[draft_token] / q[draft_token]).item(); if u <= min(1.0, ratio): return draft_token; else return residual_token.",
    "Intuition: the acceptance probability exactly cancels the draft's bias, and the residual distribution makes up the mass missing on a rejection, so together they come out to exactly p.",
]

# ===== reference solution =====
import torch


def acceptance_prob(p: torch.Tensor, q: torch.Tensor, x: int) -> torch.Tensor:
    return torch.clamp(p[x] / q[x], max=1.0)


def residual_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r = torch.clamp(p - q, min=0.0)
    return r / r.sum()


def speculative_sample(p: torch.Tensor, q: torch.Tensor, draft_token: int,
                       u: float, residual_token: int) -> int:
    ratio = (p[draft_token] / q[draft_token]).item()
    if u <= min(1.0, ratio):
        return draft_token
    return residual_token
