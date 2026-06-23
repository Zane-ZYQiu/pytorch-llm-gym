"""15 · Grouped-Query Attention GQA / MQA
Level: 2

At inference time the KV cache dominates memory. GQA lets multiple query heads
[share] one group of K/V heads, shrinking the KV cache substantially while barely
losing any quality.

  • MHA: N query heads, N KV heads (one-to-one)
  • MQA: N query heads, 1 KV head (shared by all queries)
  • GQA: N query heads, K KV heads (every N/K query heads share one KV head)

Implementation key: "expand" the K and V heads from K to N using repeat_interleave(N//K, dim=head).
After expanding, it's just like ordinary multi-head attention.

Convention: hd = D // N. Wq:(D, N*hd)=(D,D), Wk/Wv:(D, K*hd), Wo:(N*hd, D)=(D,D).

## Your task
Run: python practice.py check 15
"""
import math
import torch


def grouped_query_attention(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                            Wv: torch.Tensor, Wo: torch.Tensor,
                            num_heads: int, num_kv_heads: int,
                            causal: bool = True) -> torch.Tensor:
    """x (B, T, D). num_heads=N query heads, num_kv_heads=K KV heads (N divisible by K).
    Returns (B, T, D)."""
    raise NotImplementedError("TODO: project -> split heads -> repeat_interleave to expand KV -> attention -> merge heads")
