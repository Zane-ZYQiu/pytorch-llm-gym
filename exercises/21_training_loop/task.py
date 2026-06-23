"""21 · Training loop + teacher forcing
Level: 4

How do you train a language model? **teacher forcing**: feed the true sequence as input and predict the next token at each position.
  • input  = sequence without the last token: tokens[:, :-1]
  • target = sequence without the first token: tokens[:, 1:]
  This way the output at position t should predict the true token at position t+1.

Loss = mean cross-entropy over all positions. The standard four steps of one training step:
    optimizer.zero_grad(); loss.backward(); optimizer.step()

## Your task
Run: python practice.py check 21
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def shift_for_lm(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """tokens (B, T+1) -> (inputs, targets), each (B, T).
    inputs = tokens[:, :-1], targets = tokens[:, 1:]."""
    raise NotImplementedError("TODO: slice")


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V), targets (B, T). Return the mean cross-entropy over all positions (scalar).
    Hint: flatten the first two dims to (B*T, V) and (B*T,), then use F.cross_entropy."""
    raise NotImplementedError("TODO: reshape then F.cross_entropy")


def train_step(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> float:
    """Run one training step: forward -> lm_loss -> backward -> update. Return this step's loss value (float)."""
    raise NotImplementedError("TODO: zero_grad -> forward -> loss -> backward -> step")
