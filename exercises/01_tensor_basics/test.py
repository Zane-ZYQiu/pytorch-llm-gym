import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_arange_matrix_values_and_dtype():
    out = task.arange_matrix(2, 3)
    assert out.shape == (2, 3), f"shape should be (2,3), got {tuple(out.shape)}"
    assert out.dtype == torch.long, f"dtype should be long, got {out.dtype}"
    assert torch.equal(out, torch.tensor([[0, 1, 2], [3, 4, 5]]))


def test_arange_matrix_other_shape():
    out = task.arange_matrix(4, 1)
    assert out.shape == (4, 1)
    assert torch.equal(out.flatten(), torch.arange(4))


def test_transpose_last_two():
    x = torch.randn(2, 3, 4)
    out = task.transpose_last_two(x)
    assert out.shape == (2, 4, 3)
    assert torch.equal(out, x.transpose(-1, -2))


def test_merge_first_two_dims():
    x = torch.randn(2, 3, 5)
    out = task.merge_first_two_dims(x)
    assert out.shape == (6, 5)
    assert torch.equal(out, x.reshape(6, 5))


def test_split_heads_shape():
    x = torch.randn(2, 5, 8)
    out = task.split_heads(x, num_heads=2)
    assert out.shape == (2, 2, 5, 4), f"expected (2,2,5,4), got {tuple(out.shape)}"


def test_split_heads_values():
    x = torch.randn(2, 5, 8)
    hd = 4
    out = task.split_heads(x, num_heads=2)
    for h in range(2):
        # out[:, h, :, :] should be the h-th segment of x along the channel dim
        assert torch.equal(out[:, h, :, :], x[:, :, h * hd:(h + 1) * hd]), \
            "data layout after splitting heads is wrong: out[b,h,t,:] should equal x[b,t,h*hd:(h+1)*hd]"
