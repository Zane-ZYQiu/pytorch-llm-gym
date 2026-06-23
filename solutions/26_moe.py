HINTS = [
    "gate = x @ router_weight  # (N,E). topv, topi = torch.topk(gate, top_k, dim=-1)  # both (N,k).",
    "weights = torch.softmax(topv, dim=-1)  # normalize only over the chosen k scores.",
    "Loop over experts e: sel = (topi == e)  # (N,k) which tokens picked e in some slot; w_e = (weights*sel).sum(-1)  # (N,), 0 if not picked.",
    "out += (x @ expert_weights[e]) * w_e.unsqueeze(-1). Summing over all experts gives the result.",
]

# ===== reference solution =====
import torch


def moe_layer(x: torch.Tensor, router_weight: torch.Tensor,
              expert_weights: torch.Tensor, top_k: int) -> torch.Tensor:
    N, D = x.shape
    E = expert_weights.shape[0]
    gate = x @ router_weight                          # (N, E)
    topv, topi = torch.topk(gate, top_k, dim=-1)      # (N, k)
    weights = torch.softmax(topv, dim=-1)             # (N, k)
    out = torch.zeros_like(x)
    for e in range(E):
        sel = (topi == e)                             # (N, k) bool
        w_e = (weights * sel).sum(dim=-1)             # (N,)
        out = out + (x @ expert_weights[e]) * w_e.unsqueeze(-1)
    return out
