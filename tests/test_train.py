"""Tests for the training loop. Run: uv run python -m pytest tests/ -q

Uses a tiny model + few iterations on CPU so it runs fast.
"""

import torch

from nanogpt.config import GPTConfig
from nanogpt.model import GPT
from nanogpt.train import configure_optimizers, estimate_loss, train


def tiny_config(**kw):
    base = dict(
        vocab_size=0, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0,
        batch_size=8, max_iters=30, eval_interval=15, eval_iters=5, device="cpu", seed=0,
    )
    base.update(kw)
    return GPTConfig(**base)


def write_corpus(tmp_path):
    text = ("To be, or not to be, that is the question.\n" * 200)
    p = tmp_path / "input.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_optimizer_splits_decay_groups():
    cfg = tiny_config(vocab_size=10)
    model = GPT(cfg)
    opt = configure_optimizers(model, cfg)
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == cfg.weight_decay
    assert no_decay_group["weight_decay"] == 0.0
    # 1-D params (LayerNorm gains/shifts) must be in the no-decay group.
    assert all(p.dim() == 1 for p in no_decay_group["params"])


def test_estimate_loss_returns_both_splits(tmp_path):
    from nanogpt.data import TextData

    cfg = tiny_config()
    data = TextData.from_file(write_corpus(tmp_path))
    cfg.vocab_size = data.vocab_size
    model = GPT(cfg)
    losses = estimate_loss(model, data, cfg, "cpu")
    assert set(losses) == {"train", "val"}
    assert losses["train"] > 0


def test_training_reduces_loss_and_writes_checkpoint(tmp_path):
    cfg = tiny_config(max_iters=60, eval_interval=20)
    _, _, history = train(cfg, data_path=write_corpus(tmp_path), out_dir=tmp_path, log_fn=lambda *_: None)
    first_val = history[0][2]
    last_val = history[-1][2]
    assert last_val < first_val  # the model learned something
    assert (tmp_path / "ckpt.pt").exists()


def test_checkpoint_is_loadable(tmp_path):
    cfg = tiny_config(max_iters=20, eval_interval=20)
    train(cfg, data_path=write_corpus(tmp_path), out_dir=tmp_path, log_fn=lambda *_: None)
    ckpt = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert "model" in ckpt and "config" in ckpt and "vocab" in ckpt
    assert len(ckpt["vocab"]) == ckpt["config"]["vocab_size"]
