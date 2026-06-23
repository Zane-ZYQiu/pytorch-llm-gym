HINTS = [
    "d = q.shape[-1]; scores = q @ k.transpose(-2, -1) / math.sqrt(d). transpose(-2,-1) turns (B,Tk,d) into (B,d,Tk).",
    "causal: T = scores.shape[-1]; mask = torch.triu(torch.ones(T,T,bool), 1); scores = scores.masked_fill(mask, -inf).",
    "attn = torch.softmax(scores, dim=-1)  # normalize along the key dim.",
    "return attn @ v  # (B,Tq,Tk) @ (B,Tk,dv) -> (B,Tq,dv).",
]

# ===== reference solution =====
import math
import torch


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        T = scores.shape[-1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
