"""Tests for the hand-written LayerNorm. Run: uv run python -m pytest tests/ -q"""

import torch

from nanogpt.layer_norm import LayerNorm


def test_output_shape_preserved():
    ln = LayerNorm(32)
    x = torch.randn(4, 8, 32)
    assert ln(x).shape == (4, 8, 32)


def test_normalizes_to_zero_mean_unit_var():
    ln = LayerNorm(64)  # gamma=1, beta=0 initially
    x = torch.randn(2, 5, 64) * 7 + 3  # arbitrary scale/shift
    out = ln(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-3)


def test_matches_pytorch_layernorm():
    ours = LayerNorm(32)
    ref = torch.nn.LayerNorm(32)
    # Give them the same learnable params so outputs must match.
    with torch.no_grad():
        ours.gamma.copy_(ref.weight)
        ours.beta.copy_(ref.bias)
    x = torch.randn(3, 7, 32)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_learnable_params_get_gradients():
    ln = LayerNorm(16)
    ln(torch.randn(2, 4, 16)).sum().backward()
    assert ln.gamma.grad is not None and ln.gamma.grad.abs().sum() > 0
    assert ln.beta.grad is not None


def test_gamma_beta_affect_output():
    ln = LayerNorm(8)
    x = torch.randn(1, 3, 8)
    base = ln(x)
    with torch.no_grad():
        ln.gamma.mul_(2.0)
        ln.beta.add_(1.0)
    assert not torch.allclose(ln(x), base)
