"""24 · Beam Search
Level: 4

Greedy keeps only the single best token at each step, easily missing sequences that are better overall.
Beam search keeps the num_beams highest-scoring candidate sequences at once, expanding then pruning each step.

  • Each beam records (tokens, score), where score = sum of cumulative log probabilities.
  • Each step: for every surviving beam, take the top-(num_beams) next-token expansions;
    collect all expanded candidates, sort by score, and keep the top num_beams.
  • A beam that hits eos moves into finished and is no longer expanded.
  • At the end, return the highest-scoring beam among all of them.

Note: beam search is an [approximate] search; the wider the beam, the closer to the global optimum.

## Your task
Run: python practice.py check 24
"""
import torch
from typing import Callable, List, Optional


def beam_search(next_logprobs: Callable[[torch.Tensor], torch.Tensor],
                start_token: int, num_beams: int, max_len: int,
                eos_token: Optional[int] = None) -> List[int]:
    """next_logprobs(seq) takes a 1-D long tensor (the current sequence) and returns a (V,) log-probability for the next token.
    Return the token-id list of the highest-scoring sequence (including start_token). Expand at most max_len steps."""
    raise NotImplementedError("TODO: maintain a list of (tokens, score); each step expand + sort + prune")
