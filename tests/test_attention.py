"""Tests for single-head causal self-attention. Run: uv run python -m pytest tests/ -q"""

import torch

from nanogpt.attention import Head


def make_head(n_embd=32, head_size=16, block_size=8, dropout=0.0):
    return Head(n_embd, head_size, block_size, dropout).eval()  # eval: dropout off


def test_output_shape():
    head = make_head()
    x = torch.randn(4, 8, 32)
    assert head(x).shape == (4, 8, 16)


def test_attention_weights_rows_sum_to_one():
    head = make_head()
    _, w = head(torch.randn(2, 8, 32), return_weights=True)
    assert w.shape == (2, 8, 8)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 8), atol=1e-6)


def test_causal_mask_zeros_future():
    head = make_head()
    _, w = head(torch.randn(1, 8, 32), return_weights=True)
    # Upper triangle (strictly above diagonal) must be exactly zero: no peeking ahead.
    upper = torch.triu(w[0], diagonal=1)
    assert torch.all(upper == 0)


def test_first_token_attends_only_to_itself():
    head = make_head()
    _, w = head(torch.randn(1, 8, 32), return_weights=True)
    # Row 0 can only see position 0, so its weight there is 1.
    assert torch.allclose(w[0, 0, 0], torch.tensor(1.0), atol=1e-6)


def test_future_tokens_do_not_affect_past_outputs():
    """Changing token at position t must not change outputs at positions < t (causality)."""
    head = make_head(block_size=8)
    x = torch.randn(1, 8, 32)
    out1 = head(x)
    x2 = x.clone()
    x2[0, 5] = torch.randn(32)  # perturb position 5 only
    out2 = head(x2)
    # Positions 0..4 must be identical; position 5+ may change.
    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-6)
    assert not torch.allclose(out1[0, 5], out2[0, 5], atol=1e-6)


def test_gradients_flow_to_projections():
    head = make_head()
    head(torch.randn(2, 8, 32)).sum().backward()
    assert head.query.weight.grad is not None
    assert head.key.weight.grad.abs().sum() > 0
    assert head.value.weight.grad.abs().sum() > 0
