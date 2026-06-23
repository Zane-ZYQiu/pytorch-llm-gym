"""35 · 2D-RoPE (Variable-Resolution Visual Positional Encoding)
Level: 6

Image tokens are laid out on a 2D grid (H rows × W cols). Plain 1D RoPE only encodes "which token index",
losing the 2D structure. Qwen2-VL uses **2D-RoPE**: split head_dim in half,
rotate the first half by the [row coordinate] and the second half by the [column coordinate].

Benefit: attention scores depend only on the [relative row offset and relative column offset] of two tokens,
and they don't depend on absolute resolution, so they extrapolate directly to any H×W (any aspect ratio/resolution).

Implementation: for q/k of shape (B,H,T,hd), T = height*width (row-major layout).
  • rows[i] = i // width, cols[i] = i % width
  • first hd/2 dims: apply 1D RoPE with rows; last hd/2 dims: apply 1D RoPE with cols.

## Your task
Run: python practice.py check 35
"""
import torch
from typing import Tuple


def rope_2d(q: torch.Tensor, k: torch.Tensor, height: int, width: int,
            base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k of shape (B, H, T, hd), T == height*width, hd even (hd/2 is also recommended to be even).
    Returns the rotated (q_rot, k_rot)."""
    raise NotImplementedError("TODO: rotate the first half by row coords, the second half by col coords, each via one 1D RoPE, then concatenate")
