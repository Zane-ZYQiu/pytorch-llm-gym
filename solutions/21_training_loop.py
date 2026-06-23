HINTS = [
    "shift_for_lm: return tokens[:, :-1], tokens[:, 1:].",
    "lm_loss: B,T,V = logits.shape; return F.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T)).",
    "train_step: optimizer.zero_grad(); logits = model(inputs); loss = lm_loss(logits, targets).",
    "loss.backward(); optimizer.step(); return loss.item(). Mind the order: zero_grad first, then backward, then step.",
]

# ===== reference solution =====
import torch
import torch.nn.functional as F
from typing import Tuple


def shift_for_lm(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return tokens[:, :-1], tokens[:, 1:]


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))


def train_step(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> float:
    optimizer.zero_grad()
    logits = model(inputs)
    loss = lm_loss(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()
