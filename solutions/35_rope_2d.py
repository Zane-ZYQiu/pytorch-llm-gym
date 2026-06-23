HINTS = [
    "coordinates: idx = torch.arange(T); rows = idx // width; cols = idx % width.",
    "write a 1D RoPE helper: for a (B,H,T,d) tensor and positions pos(T,), compute inv_freq=1/base^(arange(0,d,2)/d), angles=outer(pos,inv_freq), emb=cat([angles,angles]), cos/sin.",
    "rotate_half(x): x1,x2=x[...,:d//2],x[...,d//2:]; cat([-x2,x1]). x_rot = x*cos + rotate_half(x)*sin.",
    "split q in half: apply 1D RoPE to the first hd/2 dims with rows, to the last hd/2 dims with cols, then cat back. Same for k.",
]

# ===== reference solution =====
import torch
from typing import Tuple


def _rope_1d(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    # x: (B, H, T, d), pos: (T,)
    d = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d))
    angles = torch.outer(pos.float(), inv_freq)         # (T, d/2)
    emb = torch.cat([angles, angles], dim=-1)           # (T, d)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


def rope_2d(q: torch.Tensor, k: torch.Tensor, height: int, width: int,
            base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    T = q.shape[2]
    half = q.shape[-1] // 2
    idx = torch.arange(T, device=q.device)
    rows = idx // width
    cols = idx % width

    def apply(x):
        xr = _rope_1d(x[..., :half], rows, base)        # first half by row
        xc = _rope_1d(x[..., half:], cols, base)        # second half by col
        return torch.cat([xr, xc], dim=-1)

    return apply(q), apply(k)
