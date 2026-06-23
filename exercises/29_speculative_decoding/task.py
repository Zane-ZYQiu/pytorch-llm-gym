"""29 · Speculative Decoding (accept/resample)
Level: 5

Use a small draft model q to quickly guess several tokens, then use the large target model p to verify them all at once;
accepted tokens skip the large model's token-by-token decoding — faster, and **the output distribution is exactly the same as using p alone**.

Single-token verification rule (draft samples token x ~ q):
  • Accept x with probability min(1, p[x]/q[x]);
  • Otherwise reject and resample a token from the "residual distribution" normalize(max(p - q, 0)).

It can be proven that the marginal distribution of the resulting token is exactly p. (This exercise verifies the theorem via Monte Carlo.)

## Your task
Run: python practice.py check 29
"""
import torch


def acceptance_prob(p: torch.Tensor, q: torch.Tensor, x: int) -> torch.Tensor:
    """Return the probability of accepting draft token x: min(1, p[x]/q[x]) (a scalar tensor)."""
    raise NotImplementedError("TODO: clamp(p[x]/q[x], max=1)")


def residual_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Resampling distribution after a rejection: normalize(max(p - q, 0)), returns (V,)."""
    raise NotImplementedError("TODO: clamp(p-q, min=0) then normalize")


def speculative_sample(p: torch.Tensor, q: torch.Tensor, draft_token: int,
                       u: float, residual_token: int) -> int:
    """Given the draft-sampled token, a uniform random number u∈[0,1), and a residual_token
    pre-sampled from the residual distribution: if u <= p[x]/q[x] accept and return draft_token, otherwise return residual_token."""
    raise NotImplementedError("TODO: compare u with the acceptance probability")
