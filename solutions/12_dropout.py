HINTS = [
    "Handle the edge cases first: if (not training) or p == 0.0: return x.",
    "keep = 1 - p. Use torch.rand_like(x) to draw uniform values in [0,1), mask = (rand < keep).",
    "Cast mask to x's dtype and multiply to zero out the dropped positions.",
    "Finally divide by keep to scale up: return x * mask / keep. That's inverted dropout.",
]

# ===== reference solution =====
import torch


def dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    if (not training) or p == 0.0:
        return x
    keep = 1.0 - p
    mask = (torch.rand_like(x) < keep).to(x.dtype)
    return x * mask / keep
