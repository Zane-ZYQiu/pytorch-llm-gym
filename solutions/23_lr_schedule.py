HINTS = [
    "warmup: if step < warmup_steps: return base_lr * step / warmup_steps.",
    "cosine: progress = (step - warmup_steps) / max(1, total_steps - warmup_steps), and clamp to [0,1].",
    "lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)).",
    "clip: total = torch.sqrt(sum((g**2).sum() for g in grads)); if total > max_norm: coef = max_norm/(total+1e-6); apply g.mul_(coef) to each g. return total.",
]

# ===== reference solution =====
import math
import torch
from typing import List


def lr_at_step(step: int, base_lr: float, warmup_steps: int,
               total_steps: int, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * coeff


def clip_grad_norm_(grads: List[torch.Tensor], max_norm: float) -> torch.Tensor:
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    if total > max_norm:
        coef = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(coef)
    return total
