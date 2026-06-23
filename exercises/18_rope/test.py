import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_shape_and_norm_preserved():
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    qr, kr = task.apply_rope(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape
    # rotation preserves the norm: each vector's magnitude is unchanged
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    assert torch.allclose(kr.norm(dim=-1), k.norm(dim=-1), atol=1e-4)


def test_position_zero_unchanged():
    # position 0 has a rotation angle of 0, so it should be unchanged
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    qr, kr = task.apply_rope(q, k)
    assert torch.allclose(qr[:, :, 0], q[:, :, 0], atol=1e-5)


def test_relative_position_property():
    """The soul of RoPE: score(q_m, k_n) depends only on the relative position m-n.
    Place the same vector at every position; after rotation, each diagonal of the
    inner-product matrix (fixed m-n) should be constant."""
    hd, T = 8, 6
    qvec = torch.randn(hd)
    kvec = torch.randn(hd)
    q = qvec.expand(1, 1, T, hd).contiguous()
    k = kvec.expand(1, 1, T, hd).contiguous()
    qr, kr = task.apply_rope(q, k)
    S = qr[0, 0] @ kr[0, 0].t()  # (T, T), S[m,n] = q_m . k_n
    for offset in range(-(T - 1), T):
        diag = torch.diagonal(S, offset=offset)
        assert torch.allclose(diag, diag[0].expand_as(diag), atol=1e-4), \
            f"diagonal offset={offset} should be constant (depends only on relative position)"
