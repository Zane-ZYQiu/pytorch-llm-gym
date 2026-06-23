HINTS = [
    "B,T,D = x.shape; H = num_heads; hd = D // H.",
    "project and split heads: q = (x @ Wq).reshape(B,T,H,hd).transpose(1,2), giving (B,H,T,hd). Same for k, v.",
    "score: scores = q @ k.transpose(-2,-1) / math.sqrt(hd). When causal, masked_fill the upper triangle to -inf.",
    "merge heads: o = (attn @ v).transpose(1,2).reshape(B,T,D); finally return o @ Wo.",
]

# ===== reference solution =====
import math
import torch


def multi_head_attention(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                         Wv: torch.Tensor, Wo: torch.Tensor,
                         num_heads: int, causal: bool = False) -> torch.Tensor:
    B, T, D = x.shape
    H = num_heads
    hd = D // H
    q = (x @ Wq).reshape(B, T, H, hd).transpose(1, 2)   # (B,H,T,hd)
    k = (x @ Wk).reshape(B, T, H, hd).transpose(1, 2)
    v = (x @ Wv).reshape(B, T, H, hd).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(hd)    # (B,H,T,T)
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    o = attn @ v                                         # (B,H,T,hd)
    o = o.transpose(1, 2).reshape(B, T, D)
    return o @ Wo
