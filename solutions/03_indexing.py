HINTS = [
    "gather_token_logprobs: targets.unsqueeze(-1) becomes (B,T,1), logprobs.gather(-1, it) gives (B,T,1), then squeeze(-1).",
    "one_hot_scatter: out = torch.zeros(N, num_classes); out.scatter_(1, indices.unsqueeze(1), 1.0).",
    "apply_causal_mask: T = scores.shape[-1]; mask = torch.triu(torch.ones(T,T,dtype=bool), diagonal=1). diagonal=1 means the strict upper triangle (excluding the diagonal), i.e. the future.",
    "apply_causal_mask: return scores.masked_fill(mask, float('-inf')). mask is (T,T) and broadcasts automatically over the leading batch/head dims.",
]

# ===== reference solution =====
import torch


def gather_token_logprobs(logprobs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def one_hot_scatter(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(indices.shape[0], num_classes)
    out.scatter_(1, indices.unsqueeze(1), 1.0)
    return out


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device),
                      diagonal=1)
    return scores.masked_fill(mask, float("-inf"))
