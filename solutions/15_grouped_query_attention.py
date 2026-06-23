HINTS = [
    "B,T,D = x.shape; N,K = num_heads, num_kv_heads; hd = D // N; rep = N // K.",
    "q = (x @ Wq).reshape(B,T,N,hd).transpose(1,2)  # (B,N,T,hd). For k, v use K heads: reshape(B,T,K,hd).transpose(1,2).",
    "expand KV: k = k.repeat_interleave(rep, dim=1)  # (B,K,T,hd)->(B,N,T,hd). repeat_interleave orders them 0,0,1,1... which matches the grouping exactly.",
    "after that it's plain multi-head attention: scores=q@k^T/sqrt(hd) -> (causal mask) -> softmax -> @v -> merge heads -> @Wo.",
]

# ===== reference solution =====
import math
import torch


def grouped_query_attention(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                            Wv: torch.Tensor, Wo: torch.Tensor,
                            num_heads: int, num_kv_heads: int,
                            causal: bool = True) -> torch.Tensor:
    B, T, D = x.shape
    N, K = num_heads, num_kv_heads
    hd = D // N
    rep = N // K
    q = (x @ Wq).reshape(B, T, N, hd).transpose(1, 2)   # (B,N,T,hd)
    k = (x @ Wk).reshape(B, T, K, hd).transpose(1, 2)   # (B,K,T,hd)
    v = (x @ Wv).reshape(B, T, K, hd).transpose(1, 2)
    k = k.repeat_interleave(rep, dim=1)                 # (B,N,T,hd)
    v = v.repeat_interleave(rep, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    o = (attn @ v).transpose(1, 2).reshape(B, T, D)
    return o @ Wo
