"""Tests for the position-wise feed-forward network. Run: uv run python -m pytest tests/ -q"""

import torch

from nanogpt.feed_forward import FeedForward


def test_output_shape_preserved():
    ff = FeedForward(n_embd=32).eval()
    x = torch.randn(4, 8, 32)
    assert ff(x).shape == (4, 8, 32)


def test_hidden_layer_is_4x():
    ff = FeedForward(n_embd=32, expansion=4)
    first_linear = ff.net[0]
    assert first_linear.out_features == 128  # 4 * 32


def test_is_position_wise():
    """Applying the FFN to the whole sequence equals applying it per position separately."""
    ff = FeedForward(n_embd=16).eval()
    x = torch.randn(1, 5, 16)
    full = ff(x)
    per_pos = torch.stack([ff(x[:, t : t + 1]) for t in range(5)], dim=1).squeeze(2)
    assert torch.allclose(full, per_pos, atol=1e-6)


def test_is_nonlinear():
    """A purely linear map would satisfy f(2x) == 2 f(x); GELU breaks that."""
    ff = FeedForward(n_embd=16).eval()
    x = torch.randn(1, 4, 16)
    assert not torch.allclose(ff(2 * x), 2 * ff(x), atol=1e-4)


def test_gradients_flow():
    ff = FeedForward(n_embd=16).eval()
    ff(torch.randn(2, 4, 16)).sum().backward()
    assert ff.net[0].weight.grad.abs().sum() > 0
    assert ff.net[2].weight.grad.abs().sum() > 0
