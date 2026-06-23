HINTS = [
    "B,Tx,D = x.shape; Tc = context.shape[1]; H = num_heads; hd = D//H.",
    "q = (x @ Wq).reshape(B,Tx,H,hd).transpose(1,2)  # query from x.",
    "k = (context @ Wk).reshape(B,Tc,H,hd).transpose(1,2); v likewise from context. Note that q's sequence length Tx and k/v's sequence length Tc can differ.",
    "scores=q@k.transpose(-2,-1)/sqrt(hd) -> softmax(-1) -> @v -> merge heads reshape(B,Tx,D) -> @Wo. No causal mask.",
]

# ===== reference solution =====
import math
import torch


def cross_attention(x: torch.Tensor, context: torch.Tensor,
                    Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
                    Wo: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, Tx, D = x.shape
    Tc = context.shape[1]
    H = num_heads
    hd = D // H
    q = (x @ Wq).reshape(B, Tx, H, hd).transpose(1, 2)        # (B,H,Tx,hd)
    k = (context @ Wk).reshape(B, Tc, H, hd).transpose(1, 2)  # (B,H,Tc,hd)
    v = (context @ Wv).reshape(B, Tc, H, hd).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
    attn = torch.softmax(scores, dim=-1)
    o = (attn @ v).transpose(1, 2).reshape(B, Tx, D)
    return o @ Wo
