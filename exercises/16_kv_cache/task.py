"""16 · KV cache (incremental decoding)
Level: 2

In autoregressive generation, each new token must attend to [all past tokens].
Recomputing every K/V at each step wastes a huge amount of work. The KV cache
stores the past K and V, so each step only computes the new token's q/k/v, appends
the new k/v to the cache, then uses the new q to attend to all cached k/v.

Key insight (and what the test verifies): **incremental decoding == one-shot causal
attention** — the result is exactly identical, the work is just spread across steps
to avoid repetition.

## Your task
Implement a single step: given the current token's q/k/v and the existing cache,
append and compute this step's attention output.
(The newest token can see all history plus itself, so no causal mask is needed here.)
Run: python practice.py check 16
"""
import math
import torch
from typing import Optional, Tuple


def append_and_attend(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor,
                      k_cache: Optional[torch.Tensor],
                      v_cache: Optional[torch.Tensor]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q_t, k_t, v_t all have shape (B, H, 1, hd) (the single token of this step).
    k_cache, v_cache have shape (B, H, S, hd), the cache from the past S steps; None on the first step.
    Returns (out_t, new_k_cache, new_v_cache):
      out_t has shape (B, H, 1, hd), new_*_cache has shape (B, H, S+1, hd)."""
    raise NotImplementedError("TODO: cat-append along the seq dim -> use q_t to attend over the whole cache")
