HINTS = [
    "greedy: logits.argmax(dim=-1).",
    "top_k_filter: kth = torch.topk(logits, k, dim=-1).values[..., -1:]  # the k-th largest as the threshold; return logits.masked_fill(logits < kth, -inf).",
    "top_p_filter: first sorted_logits, sorted_idx = torch.sort(logits, -1, descending=True); probs = softmax(sorted_logits, -1); cum = probs.cumsum(-1).",
    "remove = (cum - probs) > p  # remove only when 'the cumulative before me' exceeds p, guaranteeing the largest token (cum-probs=0) is always kept. Then scatter back to the original order: zeros.scatter(-1, sorted_idx, remove), masked_fill.",
]

# ===== reference solution =====
import torch


def greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]   # (..., 1) the k-th largest
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove_sorted = (cum - probs) > p                      # the ones to remove in sorted space
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))
