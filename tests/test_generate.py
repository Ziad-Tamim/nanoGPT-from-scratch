"""Tests for checkpoint loading and text generation. Run: uv run python -m pytest tests/ -q"""

import torch

from nanogpt.config import GPTConfig
from nanogpt.generate import generate_text, load_checkpoint
from nanogpt.train import train


def make_checkpoint(tmp_path):
    cfg = GPTConfig(
        vocab_size=0, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0,
        batch_size=8, max_iters=20, eval_interval=20, eval_iters=3, device="cpu", seed=0,
    )
    corpus = tmp_path / "input.txt"
    corpus.write_text("To be, or not to be.\n" * 200, encoding="utf-8")
    train(cfg, data_path=corpus, out_dir=tmp_path, log_fn=lambda *_: None)
    return tmp_path / "ckpt.pt"


def test_load_checkpoint_rebuilds_model_and_tokenizer(tmp_path):
    model, tok = load_checkpoint(make_checkpoint(tmp_path), "cpu")
    assert model.config.vocab_size == tok.vocab_size
    assert not model.training  # load_checkpoint puts it in eval mode


def test_generate_text_starts_with_prompt(tmp_path):
    model, tok = load_checkpoint(make_checkpoint(tmp_path), "cpu")
    text = generate_text(model, tok, "cpu", prompt="To be", max_new_tokens=30, top_k=None)
    assert text.startswith("To be")
    assert len(text) > len("To be")


def test_generated_chars_are_in_vocab(tmp_path):
    model, tok = load_checkpoint(make_checkpoint(tmp_path), "cpu")
    text = generate_text(model, tok, "cpu", prompt="\n", max_new_tokens=50)
    assert all(c in tok.stoi for c in text)


def test_temperature_and_topk_variants_run(tmp_path):
    model, tok = load_checkpoint(make_checkpoint(tmp_path), "cpu")
    for temp, k in [(0.5, None), (1.0, 5), (1.2, 10)]:
        out = generate_text(model, tok, "cpu", prompt="T", max_new_tokens=10,
                            temperature=temp, top_k=k)
        assert isinstance(out, str) and len(out) > 1


def test_loaded_model_produces_valid_logits(tmp_path):
    model, tok = load_checkpoint(make_checkpoint(tmp_path), "cpu")
    idx = torch.zeros((1, 4), dtype=torch.long)
    logits, _ = model(idx)
    assert logits.shape == (1, 4, tok.vocab_size)
    assert torch.isfinite(logits).all()
