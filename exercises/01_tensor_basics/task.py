"""01 · Tensor basics
Level: 0

In PyTorch everything is a tensor. The first thing to master in an interview is
reshaping a tensor's "shape" fluently -- reshape / view / transpose / split heads.
These are the foundation for writing attention later.

A few key points:
  • torch.arange(k) produces 0..k-1, with default dtype int64 (long).
  • x.reshape(...) / x.view(...) change the shape, reordering elements in row-major order, total count unchanged.
  • x.transpose(d1, d2) swaps two dimensions (note: after transpose the memory is non-contiguous, call .contiguous() when needed).
  • Negative dimensions count from the end: -1 is the last dim, -2 is the second-to-last.
  • Multi-head attention splits the D dimension into (num_heads, head_dim) and moves the head dim to the front,
    shape (B, T, D) -> (B, num_heads, T, head_dim). This is a must-know move.

## Your task
Implement the 4 functions below (replace raise NotImplementedError with your implementation). No for loops allowed.

Run: python practice.py check 01
"""
import torch


def arange_matrix(n: int, m: int) -> torch.Tensor:
    """Return a tensor of shape (n, m), filled in row-major order with 0,1,2,...,n*m-1, dtype long.
    Example: arange_matrix(2, 3) -> [[0,1,2],[3,4,5]]"""
    raise NotImplementedError("TODO: use torch.arange and reshape")


def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    """Swap the last two dimensions of a tensor. Example: (2,3,4) -> (2,4,3)."""
    raise NotImplementedError("TODO: use transpose with negative dimensions")


def merge_first_two_dims(x: torch.Tensor) -> torch.Tensor:
    """Merge the first two dimensions. Input (B, T, D) -> output (B*T, D).
    (This is often used when flattening all tokens in a batch to compute loss.)"""
    raise NotImplementedError("TODO: use reshape")


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Split into heads. Input (B, T, D), where D is divisible by num_heads.
    Output (B, num_heads, T, head_dim), where head_dim = D // num_heads.
    Require out[b, h, t, :] to equal x[b, t, h*head_dim:(h+1)*head_dim]."""
    raise NotImplementedError("TODO: first reshape out the head dim, then transpose it to the front")
