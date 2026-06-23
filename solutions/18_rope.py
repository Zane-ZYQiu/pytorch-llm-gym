HINTS = [
    "hd = q.shape[-1]; inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))  # (hd/2,).",
    "pos = torch.arange(T).float(); angles = torch.outer(pos, inv_freq)  # (T, hd/2).",
    "emb = torch.cat([angles, angles], dim=-1)  # (T, hd); cos = emb.cos()[None,None]; sin = emb.sin()[None,None]  # (1,1,T,hd).",
    "rotate_half(x): x1,x2 = x[...,:hd//2], x[...,hd//2:]; return cat([-x2, x1], -1). Finally q_rot = q*cos + rotate_half(q)*sin.",
]

# ===== reference solution =====
import torch
from typing import Tuple


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, T, hd = q.shape
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32, device=q.device) / hd))
    pos = torch.arange(T, dtype=torch.float32, device=q.device)
    angles = torch.outer(pos, inv_freq)               # (T, hd/2)
    emb = torch.cat([angles, angles], dim=-1)         # (T, hd)
    cos = emb.cos()[None, None, :, :]                 # (1,1,T,hd)
    sin = emb.sin()[None, None, :, :]

    def rotate_half(x):
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
