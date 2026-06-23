import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_patchify_shape():
    img = torch.randn(2, 3, 4, 6)
    out = task.patchify(img, patch=2)
    assert out.shape == (2, 6, 12)  # N=(4/2)*(6/2)=6, dim=3*2*2=12


def test_patchify_content():
    img = torch.randn(2, 3, 4, 6)
    out = task.patchify(img, patch=2)
    # patch 0 = top-left corner (rows 0:2, cols 0:2), flattened in (C,p,p) order
    assert torch.allclose(out[:, 0], img[:, :, 0:2, 0:2].reshape(2, -1), atol=1e-6)
    # patch 1 = next in row-major order = (rows 0:2, cols 2:4)
    assert torch.allclose(out[:, 1], img[:, :, 0:2, 2:4].reshape(2, -1), atol=1e-6)
    # patch 3 = first patch of the second row = (rows 2:4, cols 0:2)
    assert torch.allclose(out[:, 3], img[:, :, 2:4, 0:2].reshape(2, -1), atol=1e-6)


def test_vit_embed():
    B, C, H, W, patch, D = 2, 3, 4, 6, 2, 16
    img = torch.randn(B, C, H, W)
    N = (H // patch) * (W // patch)
    proj = torch.randn(C * patch * patch, D)
    cls = torch.randn(1, 1, D)
    pos = torch.randn(1, N + 1, D)
    out = task.vit_embed(img, patch, proj, cls, pos)
    assert out.shape == (B, N + 1, D)
    # token 0 should be cls + pos[0]
    assert torch.allclose(out[:, 0], cls[:, 0] + pos[:, 0], atol=1e-5)
    # token 1 should be patch0 projection + pos[1]
    patches = task.patchify(img, patch)
    assert torch.allclose(out[:, 1], patches[:, 0] @ proj + pos[:, 1], atol=1e-5)
