HINTS = [
    "arange_matrix: torch.arange(n*m) gives a 1-D 0..nm-1, then .reshape(n, m). It is long by default.",
    "transpose_last_two: return x.transpose(-1, -2).",
    "merge_first_two_dims: B, T = x.shape[:2]; return x.reshape(B*T, x.shape[2]). Or x.reshape(-1, x.shape[-1]).",
    "split_heads: first x.reshape(B, T, num_heads, head_dim) to split D into two dims; then .transpose(1, 2) to move the head dim in front of T.",
]

# ===== reference solution =====
import torch


def arange_matrix(n: int, m: int) -> torch.Tensor:
    return torch.arange(n * m).reshape(n, m)


def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-1, -2)


def merge_first_two_dims(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, T, D = x.shape
    head_dim = D // num_heads
    return x.reshape(B, T, num_heads, head_dim).transpose(1, 2)
