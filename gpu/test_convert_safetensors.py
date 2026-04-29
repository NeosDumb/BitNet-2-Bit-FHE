import torch
import pytest
from gpu.convert_safetensors import ModelArgs, invert_convert_q, invert_convert_k
from einops import EinopsError

def test_invert_convert_q():
    config = ModelArgs(n_head=2, dim=8, n_local_heads=2)
    # w: (h l d) i -> (h d l) i
    # h = 2, l = 2, d = 4, i = 1
    # total dim = 2 * 2 * 4 = 16
    w = torch.arange(16).view(16, 1)

    expected = torch.tensor([
        [0], [4], [1], [5], [2], [6], [3], [7],
        [8], [12], [9], [13], [10], [14], [11], [15]
    ])

    result = invert_convert_q(w, config)
    assert torch.equal(result, expected)

def test_invert_convert_k():
    config = ModelArgs(n_head=4, n_local_heads=2, dim=16)
    # w: (h l d) i -> (h d l) i
    # h = 2, l = 2, d = 8, i = 1
    # total dim = 2 * 2 * 8 = 32
    w = torch.arange(32).view(32, 1)

    result = invert_convert_k(w, config)

    # expected shape is (32, 1)
    # The rearrangement is:
    # chunk into h=2, l=2, d=8
    # w_reshaped = w.view(2, 2, 8, 1)
    # w_transposed = w_reshaped.transpose(1, 2) # shape (2, 8, 2, 1)
    # w_final = w_transposed.reshape(32, 1)

    expected = w.view(2, 2, 8, 1).transpose(1, 2).reshape(32, 1)

    assert torch.equal(result, expected)

def test_invert_convert_q_multidim():
    config = ModelArgs(n_head=2, dim=8, n_local_heads=2)
    # w: (h l d) i -> (h d l) i
    # h = 2, l = 2, d = 4, i = 2
    # first dim = 2 * 2 * 4 = 16
    w = torch.arange(32).view(16, 2)

    result = invert_convert_q(w, config)
    expected = w.view(2, 2, 4, 2).transpose(1, 2).reshape(16, 2)

    assert torch.equal(result, expected)

def test_invert_convert_k_multidim():
    config = ModelArgs(n_head=4, n_local_heads=2, dim=16)
    # w: (h l d) i -> (h d l) i
    # h = 2, l = 2, d = 8, i = 3
    # first dim = 2 * 2 * 8 = 32
    w = torch.arange(96).view(32, 3)

    result = invert_convert_k(w, config)
    expected = w.view(2, 2, 8, 3).transpose(1, 2).reshape(32, 3)

    assert torch.equal(result, expected)

def test_invert_convert_q_invalid_shape():
    config = ModelArgs(n_head=2, dim=8, n_local_heads=2)
    # Need first dim divisible by (h * l) = 2 * 2 = 4
    # Let's provide 15 instead of 16
    w = torch.arange(15).view(15, 1)

    with pytest.raises(EinopsError):
        invert_convert_q(w, config)

def test_invert_convert_k_invalid_shape():
    config = ModelArgs(n_head=4, n_local_heads=2, dim=16)
    # Need first dim divisible by (h * l) = 2 * 2 = 4
    # Let's provide 31 instead of 32
    w = torch.arange(31).view(31, 1)

    with pytest.raises(EinopsError):
        invert_convert_k(w, config)
