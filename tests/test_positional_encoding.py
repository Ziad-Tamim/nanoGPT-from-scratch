"""Tests for positional encoding. Run: uv run python -m pytest tests/ -q"""

import math

import torch

from nanogpt.config import GPTConfig
from nanogpt.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    build_positional_encoding,
)


def test_sinusoidal_shape_preserved():
    pe = SinusoidalPositionalEncoding(n_embd=384, block_size=256)
    x = torch.zeros(4, 16, 384)
    assert pe(x).shape == (4, 16, 384)


def test_sinusoidal_matches_formula():
    C, base = 8, 10000.0
    pe = SinusoidalPositionalEncoding(n_embd=C, block_size=32, base=base)
    pos, k = 5, 1  # check dimension 2k=2 (sin) and 2k+1=3 (cos)
    denom = base ** (2 * k / C)
    assert math.isclose(pe.pe[pos, 2 * k].item(), math.sin(pos / denom), abs_tol=1e-5)
    assert math.isclose(pe.pe[pos, 2 * k + 1].item(), math.cos(pos / denom), abs_tol=1e-5)


def test_sinusoidal_has_no_parameters():
    pe = SinusoidalPositionalEncoding(n_embd=16, block_size=32)
    assert sum(p.numel() for p in pe.parameters()) == 0  # fixed, not learnable


def test_sinusoidal_values_in_range():
    pe = SinusoidalPositionalEncoding(n_embd=16, block_size=32)
    assert pe.pe.min() >= -1.0 and pe.pe.max() <= 1.0  # sin/cos bounded


def test_learned_shape_and_is_parametric():
    pe = LearnedPositionalEncoding(n_embd=32, block_size=64)
    x = torch.zeros(2, 10, 32)
    assert pe(x).shape == (2, 10, 32)
    assert sum(p.numel() for p in pe.parameters()) == 64 * 32  # block_size * C


def test_learned_rejects_too_long_sequence():
    pe = LearnedPositionalEncoding(n_embd=8, block_size=4)
    try:
        pe(torch.zeros(1, 5, 8))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_factory_respects_config():
    assert isinstance(
        build_positional_encoding(GPTConfig(pos_encoding="sinusoidal")),
        SinusoidalPositionalEncoding,
    )
    assert isinstance(
        build_positional_encoding(GPTConfig(pos_encoding="learned")),
        LearnedPositionalEncoding,
    )
