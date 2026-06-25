"""Tests for multi-head attention. Run: uv run python -m pytest tests/ -q"""

import pytest
import torch

from nanogpt.multi_head_attention import MultiHeadAttention


def make_mha(n_embd=32, n_head=4, block_size=8, dropout=0.0):
    return MultiHeadAttention(n_embd, n_head, block_size, dropout).eval()


def test_output_preserves_shape():
    mha = make_mha()
    x = torch.randn(4, 8, 32)
    assert mha(x).shape == (4, 8, 32)  # (B, T, C) in and out


def test_head_count_and_size():
    mha = make_mha(n_embd=32, n_head=4)
    assert len(mha.heads) == 4
    assert mha.heads[0].head_size == 8  # 32 // 4


def test_rejects_indivisible_dims():
    with pytest.raises(ValueError):
        MultiHeadAttention(n_embd=30, n_head=4, block_size=8)


def test_per_head_weights_shape():
    mha = make_mha(n_embd=32, n_head=4, block_size=8)
    _, w = mha(torch.randn(2, 8, 32), return_weights=True)
    assert w.shape == (2, 4, 8, 8)  # (B, n_head, T, T)


def test_each_head_is_causal():
    mha = make_mha(n_embd=32, n_head=4, block_size=8)
    _, w = mha(torch.randn(1, 8, 32), return_weights=True)
    for h in range(4):
        assert torch.all(torch.triu(w[0, h], diagonal=1) == 0)


def test_causality_preserved_end_to_end():
    mha = make_mha(block_size=8)
    x = torch.randn(1, 8, 32)
    out1 = mha(x)
    x2 = x.clone()
    x2[0, 6] = torch.randn(32)
    out2 = mha(x2)
    assert torch.allclose(out1[0, :6], out2[0, :6], atol=1e-6)
