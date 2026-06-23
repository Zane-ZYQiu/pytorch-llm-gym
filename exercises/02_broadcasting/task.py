"""02 · Broadcasting & vectorization
Level: 0

Interviewers often say: "rewrite this without for loops." The trick is **broadcasting**:
when two tensors have different dimensions, PyTorch automatically "replicates" size-1 dimensions to align them.

  • a shape (N, 1, D), b shape (1, M, D)  ->  a - b shape (N, M, D)
  • Use x.unsqueeze(k) to insert a new size-1 dimension (equivalent to x[:, None]).
  • Pairing a reduction with keepdim=True keeps the dimension for easy later broadcasting.
  • For padding, a mask is common: multiply invalid positions by 0, then divide by the valid count.

## Your task
Implement everything with pure tensor operations, no Python for loops.

Run: python practice.py check 02
"""
import torch


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance. a shape (N, D), b shape (M, D).
    Return (N, M), where out[i, j] = sum_d (a[i,d] - b[j,d])**2."""
    raise NotImplementedError("TODO: a.unsqueeze(1) - b.unsqueeze(0) then square and sum")


def outer_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Outer sum. a shape (N,), b shape (M,). Return (N, M), out[i,j] = a[i] + b[j]."""
    raise NotImplementedError("TODO: use unsqueeze to create broadcastable shapes")


def normalize_last_dim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize the last dimension to unit L2 norm. Return a tensor with the same shape as x.
    Each vector v -> v / (||v||_2 + eps)."""
    raise NotImplementedError("TODO: x.norm(dim=-1, keepdim=True)")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean (handles variable-length sequences / padding).
    x shape (B, T, D), mask shape (B, T) as bool (True = valid token).
    Return (B, D): for each sample, average only over the valid time steps."""
    raise NotImplementedError("TODO: multiply by mask.unsqueeze(-1) and sum, then divide by the valid count")
