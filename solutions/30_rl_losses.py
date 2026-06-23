HINTS = [
    "dpo: logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)); return -F.logsigmoid(logits).mean().",
    "ppo: ratio = torch.exp(logp - old_logp); unclipped = ratio*advantages; clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantages; return -torch.min(unclipped, clipped).mean().",
    "grpo: mean = rewards.mean(-1, keepdim=True); std = rewards.std(-1, unbiased=False, keepdim=True); return (rewards-mean)/(std+eps).",
    "The min in PPO is the key: take the more conservative (smaller) objective, which is equivalent to limiting how far ratio deviates from 1 in the advantage direction.",
]

# ===== reference solution =====
import torch
import torch.nn.functional as F


def dpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float) -> torch.Tensor:
    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(logits).mean()


def ppo_clip_loss(logp: torch.Tensor, old_logp: torch.Tensor,
                  advantages: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)
