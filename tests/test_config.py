"""Sanity tests for GPTConfig. Run with: uv run python -m pytest tests/ -q"""

import pytest

from nanogpt import GPTConfig, get_device


def test_head_size_divides_evenly():
    cfg = GPTConfig(n_embd=384, n_head=6)
    assert cfg.head_size == 64


def test_invalid_head_count_raises():
    with pytest.raises(ValueError):
        GPTConfig(n_embd=384, n_head=5)  # 384 not divisible by 5


def test_invalid_pos_encoding_raises():
    with pytest.raises(ValueError):
        GPTConfig(pos_encoding="rotary")


def test_device_is_known():
    assert get_device() in ("cuda", "cpu")


def test_resolve_device_expands_auto():
    cfg = GPTConfig(device="auto")
    assert cfg.resolve_device() in ("cuda", "cpu")
    assert GPTConfig(device="cpu").resolve_device() == "cpu"
