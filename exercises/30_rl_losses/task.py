"""30 · RLHF Alignment Losses: DPO / PPO-clip / GRPO
Level: 5

Three common post-training alignment losses, each writable in a few lines:

DPO (Direct Preference Optimization, RL-free): given chosen (better) / rejected (worse) responses,
  loss = -log σ( β·[(logπ_c - logπ_ref_c) - (logπ_r - logπ_ref_r)] )
  Intuition: widen the gap between "better than reference" and "worse than reference".

PPO clipped objective (with ratio clipping, to prevent the policy from updating too aggressively in one step):
  ratio = exp(logπ - logπ_old)
  loss = -mean( min( ratio·A, clip(ratio, 1-ε, 1+ε)·A ) )

GRPO group-relative advantage (DeepSeek, critic-free): sample a group of responses for the same prompt,
  A_i = (r_i - mean(r)) / (std(r) + eps)     # standardize rewards within the group

## Your task
Run: python practice.py check 30
"""
import torch
import torch.nn.functional as F


def dpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float) -> torch.Tensor:
    """All four inputs are (B,) sequence log-probabilities. Returns the scalar mean DPO loss."""
    raise NotImplementedError("TODO: -logsigmoid(beta*((pi_c-ref_c)-(pi_r-ref_r))).mean()")


def ppo_clip_loss(logp: torch.Tensor, old_logp: torch.Tensor,
                  advantages: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """logp/old_logp/advantages are all (B,). Returns the scalar PPO clipped loss."""
    raise NotImplementedError("TODO: ratio=exp(logp-old_logp); -min(ratio*A, clip*A).mean()")


def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """rewards (B, G): B prompts, each with G sampled responses' scalar rewards.
    Returns (B, G) within-group standardized advantages. Use the population std (unbiased=False)."""
    raise NotImplementedError("TODO: (r - mean) / (std + eps), along the group dim")
