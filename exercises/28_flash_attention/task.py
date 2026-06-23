"""28 · Online softmax / Flash-Attention tiling
Level: 5

Standard attention has to store the full (Tq, Tk) score matrix, costing O(T²) memory.
Flash-Attention never materializes this big matrix: it streams K/V block by block, using **online softmax**
to update the output as it reads. The result is **exactly the same** as standard attention (exact, not approximate), but memory drops to O(T).

Online softmax core (while processing block j, maintain: running max m, running denominator l, running output o):
  s        = q @ k_block^T / sqrt(d)
  m_new    = max(m, rowmax(s))
  p        = exp(s - m_new)
  alpha    = exp(m - m_new)               # rescaling factor for the old statistics
  l        = alpha * l + rowsum(p)
  o        = alpha * o + p @ v_block
  m        = m_new
Finally out = o / l.

## Your task (non-causal version is fine)
Run: python practice.py check 28
"""
import math
import torch


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    block_size: int) -> torch.Tensor:
    """q,k,v have shape (B, H, T, d). Compute attention by streaming in blocks of block_size.
    Returns (B, H, T, d), which should numerically match standard softmax attention."""
    raise NotImplementedError("TODO: tile along the key dim, maintain m/l/o, update via online softmax")
