HINTS = [
    "cross_entropy: first a stable log_softmax: z = logits - logits.max(-1,keepdim).values; logp = z - z.exp().sum(-1,keepdim).log().",
    "Pick out the target terms: logp[torch.arange(N), targets], or logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1). loss = -that.mean().",
    "cross_entropy_grad: p = softmax(logits) (again subtract the max for stability).",
    "onehot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0); return (p - onehot) / N.",
]

# ===== reference solution =====
import torch


def _log_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    return z - z.exp().sum(dim=dim, keepdim=True).log()


def _softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    n = logits.shape[0]
    logp = _log_softmax(logits, dim=-1)
    picked = logp[torch.arange(n), targets]
    return -picked.mean()


def cross_entropy_grad(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    n = logits.shape[0]
    p = _softmax(logits, dim=-1)
    onehot = torch.zeros_like(logits)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return (p - onehot) / n
