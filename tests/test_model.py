"""Tests for the full GPT model. Run: uv run python -m pytest tests/ -q"""

import math

import torch

from nanogpt.config import GPTConfig
from nanogpt.model import GPT


def small_config(**kw):
    base = dict(vocab_size=65, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_logits_shape():
    model = GPT(small_config()).eval()
    idx = torch.randint(0, 65, (4, 16))
    logits, loss = model(idx)
    assert logits.shape == (4, 16, 65)  # (B, T, vocab)
    assert loss is None  # no targets passed


def test_loss_computed_with_targets():
    model = GPT(small_config()).eval()
    idx = torch.randint(0, 65, (4, 16))
    targets = torch.randint(0, 65, (4, 16))
    _, loss = model(idx, targets)
    assert loss.ndim == 0  # scalar


def test_initial_loss_near_ln_vocab():
    """At init, the model is ~uniform, so cross-entropy should be ~ln(vocab_size)."""
    torch.manual_seed(0)
    model = GPT(small_config()).eval()
    idx = torch.randint(0, 65, (32, 16))
    targets = torch.randint(0, 65, (32, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(65)) < 0.3  # ln(65) ~ 4.17


def test_weight_tying_shares_storage():
    model = GPT(small_config())
    # The LM head and the token embedding must be the exact same tensor.
    assert model.lm_head.weight.data_ptr() == model.token_embedding.table.weight.data_ptr()


def test_generate_extends_sequence():
    model = GPT(small_config()).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20)
    assert out.shape == (1, 21)
    assert out.min() >= 0 and out.max() < 65  # valid token ids


def test_generate_respects_block_size():
    """Generating beyond block_size must not error (context is cropped)."""
    model = GPT(small_config(block_size=8)).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20)  # exceeds block_size of 8
    assert out.shape == (1, 21)


def test_return_weights_shape():
    cfg = small_config(n_layer=2, n_head=4, block_size=8)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 65, (2, 8))
    _, _, w = model(idx, return_weights=True)
    assert w.shape == (2, 2, 4, 8, 8)  # (B, n_layer, n_head, T, T)
