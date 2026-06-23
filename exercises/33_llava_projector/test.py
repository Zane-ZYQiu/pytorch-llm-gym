import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def _setup():
    B, N_img, N_txt = 2, 5, 7
    D_v, D_h, D_t = 8, 16, 12
    img = torch.randn(B, N_img, D_v)
    txt = torch.randn(B, N_txt, D_t)
    W1 = torch.randn(D_v, D_h); b1 = torch.randn(D_h)
    W2 = torch.randn(D_h, D_t); b2 = torch.randn(D_t)
    return img, txt, W1, b1, W2, b2, (B, N_img, N_txt, D_t)


def test_shape_and_order():
    img, txt, W1, b1, W2, b2, (B, N_img, N_txt, D_t) = _setup()
    out = task.build_multimodal_input(img, txt, W1, b1, W2, b2)
    assert out.shape == (B, N_img + N_txt, D_t)
    # the text part (tail) should be preserved as-is
    assert torch.allclose(out[:, N_img:], txt, atol=1e-5), "tail should be the original text embeddings"


def test_projection_value():
    img, txt, W1, b1, W2, b2, (B, N_img, N_txt, D_t) = _setup()
    out = task.build_multimodal_input(img, txt, W1, b1, W2, b2)
    expected_visual = F.gelu(img @ W1 + b1) @ W2 + b2
    assert torch.allclose(out[:, :N_img], expected_visual, atol=1e-5), "visual part should be the MLP projection result"
