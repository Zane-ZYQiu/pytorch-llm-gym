"""22 · Sampling strategies: greedy / temperature / top-k / top-p
Level: 4

How do you pick the next token from logits during generation?
  • greedy: just take the argmax (deterministic, prone to repetition).
  • temperature T: logits / T, then softmax. T<1 is sharper, T>1 is more random.
  • top-k: keep only the k highest-probability tokens, set the rest to -inf, then sample.
  • top-p (nucleus): accumulate probabilities from high to low, keep the smallest set whose cumulative probability just exceeds p.

Interview point: top-k is "a fixed count", top-p is "a fixed cumulative probability"; the latter is more adaptive.

## Your task (the filter functions set discarded positions to -inf)
Run: python practice.py check 22
"""
import torch


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """logits (..., V) -> argmax over the last dim, returns (...,) long."""
    raise NotImplementedError("TODO: argmax")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits in each row, set the rest to -inf. Return a tensor of the same shape."""
    raise NotImplementedError("TODO: use topk to find the threshold, masked_fill values below it")


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """nucleus sampling filter: keep the smallest set whose cumulative probability reaches p, set the rest to -inf.
    Always keep at least the highest-probability token. Return a tensor of the same shape."""
    raise NotImplementedError("TODO: sort -> softmax -> cumsum -> mark for removal -> restore order")
