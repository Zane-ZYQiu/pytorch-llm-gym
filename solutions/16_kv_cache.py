HINTS = [
    "append to cache: if k_cache is None, k_all = k_t; otherwise k_all = torch.cat([k_cache, k_t], dim=2). Same for v.",
    "dim=2 is the seq dim (the 2nd dim of shape B,H,S,hd).",
    "score: hd = q_t.shape[-1]; scores = q_t @ k_all.transpose(-2,-1) / sqrt(hd)  # (B,H,1,S+1).",
    "out = softmax(scores,-1) @ v_all  # (B,H,1,hd). return out, k_all, v_all. No causal mask is needed since there's only one newest query.",
]

# ===== reference solution =====
import math
import torch
from typing import Optional, Tuple


def append_and_attend(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor,
                      k_cache: Optional[torch.Tensor],
                      v_cache: Optional[torch.Tensor]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if k_cache is None:
        k_all, v_all = k_t, v_t
    else:
        k_all = torch.cat([k_cache, k_t], dim=2)
        v_all = torch.cat([v_cache, v_t], dim=2)
    hd = q_t.shape[-1]
    scores = q_t @ k_all.transpose(-2, -1) / math.sqrt(hd)   # (B,H,1,S+1)
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v_all                                        # (B,H,1,hd)
    return out, k_all, v_all
