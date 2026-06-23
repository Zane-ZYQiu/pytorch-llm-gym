"""08 · Cross-Entropy Loss + Gradient
Level: 1

The training objective of a language model is exactly cross-entropy (predicting the next token).
  loss = -1/N * sum_i log p_i[target_i], where p = softmax(logits)

The most elegant result (guaranteed to come up): the gradient of cross-entropy w.r.t. logits
is remarkably simple——
    dL/d(logits) = (softmax(logits) - onehot(targets)) / N
"predicted probability minus the true one-hot". Understanding this matters more than memorizing the formula.

## Your task
Implement it yourself (must be numerically stable; you may inline the log-softmax trick; do not call F.cross_entropy).
Run: python practice.py check 08
"""
import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits shape (N, V), targets shape (N,) as long.
    Return a scalar: the negative log-likelihood averaged over the batch. Must be numerically stable."""
    raise NotImplementedError("TODO: log_softmax, gather the target terms, negate, average")


def cross_entropy_grad(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return dL/d(logits), same shape as logits (N, V).
    Should equal (softmax(logits) - onehot(targets)) / N."""
    raise NotImplementedError("TODO: softmax - onehot, then divide by N")
