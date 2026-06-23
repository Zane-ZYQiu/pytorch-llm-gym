HINTS = [
    "Init: B,H,Tq,d = q.shape; scale=1/sqrt(d); o=zeros(B,H,Tq,d); m=full((B,H,Tq,1), -inf); l=zeros(B,H,Tq,1).",
    "Loop over key blocks start in range(0, Tk, block_size): kb=k[:,:,start:start+bs]; vb=v[:,:,start:start+bs]; s = (q@kb.transpose(-2,-1))*scale  # (B,H,Tq,bk).",
    "m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values); p = torch.exp(s - m_new); alpha = torch.exp(m - m_new).",
    "l = alpha*l + p.sum(-1, keepdim=True); o = alpha*o + p @ vb; m = m_new. After the loop return o / l.",
]

# ===== reference solution =====
import math
import torch


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    block_size: int) -> torch.Tensor:
    B, H, Tq, d = q.shape
    Tk = k.shape[2]
    scale = 1.0 / math.sqrt(d)
    o = torch.zeros(B, H, Tq, d, dtype=q.dtype, device=q.device)
    m = torch.full((B, H, Tq, 1), float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.zeros(B, H, Tq, 1, dtype=q.dtype, device=q.device)
    for start in range(0, Tk, block_size):
        kb = k[:, :, start:start + block_size]
        vb = v[:, :, start:start + block_size]
        s = (q @ kb.transpose(-2, -1)) * scale          # (B,H,Tq,bk)
        m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        alpha = torch.exp(m - m_new)                     # rescaling factor for old statistics
        l = alpha * l + p.sum(dim=-1, keepdim=True)
        o = alpha * o + p @ vb
        m = m_new
    return o / l
