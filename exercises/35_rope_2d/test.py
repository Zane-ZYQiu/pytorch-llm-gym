import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_shape_and_norm():
    H_grid, W_grid, hd = 3, 4, 8
    T = H_grid * W_grid
    q = torch.randn(2, 2, T, hd)
    k = torch.randn(2, 2, T, hd)
    qr, kr = task.rope_2d(q, k, H_grid, W_grid)
    assert qr.shape == q.shape
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-4), "rotation should preserve the norm"


def test_2d_relative_property():
    """The essence of 2D-RoPE: score(token_a, token_b) depends only on (Δrow, Δcol)."""
    Hg, Wg, hd = 3, 3, 8
    T = Hg * Wg
    qvec = torch.randn(hd)
    kvec = torch.randn(hd)
    q = qvec.expand(1, 1, T, hd).contiguous()
    k = kvec.expand(1, 1, T, hd).contiguous()
    qr, kr = task.rope_2d(q, k, Hg, Wg)
    S = (qr[0, 0] @ kr[0, 0].t()).reshape(Hg, Wg, Hg, Wg)  # S[r1,c1,r2,c2]

    for dr in range(-(Hg - 1), Hg):
        for dc in range(-(Wg - 1), Wg):
            vals = []
            for r2 in range(Hg):
                for c2 in range(Wg):
                    r1, c1 = r2 + dr, c2 + dc
                    if 0 <= r1 < Hg and 0 <= c1 < Wg:
                        vals.append(S[r1, c1, r2, c2])
            vals = torch.stack(vals)
            assert torch.allclose(vals, vals[0].expand_as(vals), atol=1e-4), \
                f"scores for relative offset (dr={dr}, dc={dc}) should be constant"
