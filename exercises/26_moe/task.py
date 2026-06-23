"""26 · Mixture of Experts (MoE)
Level: 5

MoE replaces one big FFN with many small "experts"; each token only goes through its top-k,
so the parameter count explodes while the per-token compute stays constant (sparse activation).

Flow (token-level top-k routing):
  1. Routing scores: gate = x @ router_weight          # (N, E), E experts
  2. Each token picks its top-k experts: topk(gate, k)    # gives scores and expert indices
  3. Softmax over the chosen k scores as weights         # normalize only within the chosen ones
  4. Output = weighted sum of the chosen experts' outputs

Here each expert is just a linear transform (D, D) (simplified).

## Your task
Run: python practice.py check 26
"""
import torch


def moe_layer(x: torch.Tensor, router_weight: torch.Tensor,
              expert_weights: torch.Tensor, top_k: int) -> torch.Tensor:
    """x (N, D); router_weight (D, E); expert_weights (E, D, D).
    Returns (N, D). Each token goes through its top_k highest-scoring experts, weighted by softmax."""
    raise NotImplementedError("TODO: routing scores -> top-k -> softmax weights -> weighted sum of expert outputs")
