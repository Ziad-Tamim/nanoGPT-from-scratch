"""Tests for the tokenizer and data pipeline. Run: uv run python -m pytest tests/ -q"""

import torch

from nanogpt.tokenizer import CharTokenizer
from nanogpt.data import TextData

SAMPLE = "hello world\nthis is a tiny corpus for testing.\n"


def test_tokenizer_roundtrip_is_lossless():
    tok = CharTokenizer.from_text(SAMPLE)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_vocab_is_sorted_unique_chars():
    tok = CharTokenizer.from_text(SAMPLE)
    assert tok.chars == sorted(set(SAMPLE))
    assert tok.vocab_size == len(set(SAMPLE))


def test_tokenizer_save_load_roundtrip(tmp_path):
    tok = CharTokenizer.from_text(SAMPLE)
    p = tmp_path / "vocab.json"
    tok.save(p)
    tok2 = CharTokenizer.load(p)
    assert tok2.chars == tok.chars
    assert tok2.encode(SAMPLE) == tok.encode(SAMPLE)


def test_batch_shapes():
    data = TextData(SAMPLE * 50)
    x, y = data.get_batch("train", batch_size=4, block_size=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long


def test_targets_are_inputs_shifted_by_one():
    # With a fixed generator we can recompute the exact windows and check y = x shifted.
    data = TextData(SAMPLE * 50)
    g = torch.Generator().manual_seed(0)
    x, y = data.get_batch("train", batch_size=3, block_size=8, generator=g)
    # y[:, t] should equal the token that follows x[:, t]; i.e. x[:, 1:] == y[:, :-1].
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_train_val_split_disjoint_sizes():
    data = TextData("abcdefghij" * 100, train_frac=0.9)
    assert len(data.train_data) + len(data.val_data) == 1000
    assert len(data.train_data) == 900
