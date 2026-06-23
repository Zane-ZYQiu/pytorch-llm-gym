"""18 · RoPE (Rotary Position Embedding)
Level: 3

Modern LLMs (LLaMA/Qwen/...) all use RoPE. The core idea: "rotate" q and k by an
angle that grows in proportion to the position. The magic: after rotation, the dot
product of two vectors depends only on their relative position m−n, which naturally
encodes relative-position information into the attention scores.

Implementation (the HF/LLaMA rotate_half convention):
  • inv_freq = 1 / base^(2i/hd), i=0..hd/2-1
  • the angles for position pos are angles = pos ⊗ inv_freq, shape (T, hd/2)
  • emb = cat([angles, angles], -1), cos/sin have shape (T, hd)
  • rotate_half(x): negate the second half and move it to the front -> cat([-x2, x1])
  • x_rot = x * cos + rotate_half(x) * sin

Only Q and K are rotated; V is left untouched.

## Your task
Run: python practice.py check 18
"""
import torch
from typing import Tuple


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k have shape (B, H, T, hd), hd is even. Positions are 0..T-1.
    Return the rotated (q_rot, k_rot), with shapes unchanged."""
    raise NotImplementedError("TODO: compute cos/sin, rotate_half, apply the formula")
