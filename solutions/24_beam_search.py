HINTS = [
    "beams = [([start_token], 0.0)]; finished = []. Each beam is (token list, cumulative logprob).",
    "Each step iterate over surviving beams: if the last token is eos, move into finished and skip; otherwise call next_logprobs and expand with torch.topk(lp, min(num_beams, V)).",
    "Collect all expansions (tokens+[i], score+v), sort by score descending, beams = candidates[:num_beams].",
    "When done, all = finished + beams, sort by score and return the tokens of the first one.",
]

# ===== reference solution =====
import torch
from typing import Callable, List, Optional


def beam_search(next_logprobs: Callable[[torch.Tensor], torch.Tensor],
                start_token: int, num_beams: int, max_len: int,
                eos_token: Optional[int] = None) -> List[int]:
    beams = [([start_token], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for tokens, score in beams:
            if eos_token is not None and tokens[-1] == eos_token:
                finished.append((tokens, score))
                continue
            lp = next_logprobs(torch.tensor(tokens, dtype=torch.long))
            k = min(num_beams, lp.numel())
            topv, topi = torch.topk(lp, k)
            for v, i in zip(topv.tolist(), topi.tolist()):
                candidates.append((tokens + [i], score + v))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:num_beams]
    all_beams = finished + beams
    all_beams.sort(key=lambda x: x[1], reverse=True)
    return all_beams[0][0]
