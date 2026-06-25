"""Training loop — fit the GPT on Tiny Shakespeare.

See docs/12_training.md.

Run with defaults:

    uv run python -m nanogpt.train

or with overrides:

    uv run python -m nanogpt.train --max-iters 5000 --device cuda

The loop: sample a batch, compute the next-token loss, backprop, clip gradients, and step
AdamW. Every ``eval_interval`` steps it estimates train/val loss over several batches and
saves a checkpoint whenever validation loss improves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from nanogpt.config import GPTConfig
from nanogpt.data import TextData
from nanogpt.model import GPT


def configure_optimizers(model: GPT, config: GPTConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay on matmul weights only (not biases / LayerNorm gains).

    Tensors with >= 2 dims (embeddings, linear weights) are regularized; 1-D tensors
    (biases, LayerNorm gamma/beta) are not — the standard GPT recipe.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(
    model: GPT, data: TextData, config: GPTConfig, device: str
) -> dict[str, float]:
    """Average loss over ``eval_iters`` batches for train and val (smooths out noise)."""
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = data.get_batch(split, config.batch_size, config.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: Path, model: GPT, optimizer: torch.optim.Optimizer,
    config: GPTConfig, data: TextData, step: int, best_val: float,
) -> None:
    """Persist everything needed to resume training or to generate text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(config),
            "vocab": data.tokenizer.chars,  # so generation uses the identical mapping
            "step": step,
            "best_val": best_val,
        },
        path,
    )


def train(
    config: GPTConfig | None = None,
    data_path: str | Path = "data/input.txt",
    out_dir: str | Path = "checkpoints",
    log_fn: Callable[[str], None] = print,
) -> tuple[GPT, TextData, list[tuple[int, float, float]]]:
    """Train a GPT and return ``(model, data, history)`` where history is (step, train, val)."""
    config = config or GPTConfig()
    torch.manual_seed(config.seed)
    device = config.resolve_device()

    data = TextData.from_file(data_path)
    config.vocab_size = data.vocab_size  # keep model vocab in sync with the corpus
    model = GPT(config).to(device)
    optimizer = configure_optimizers(model, config)
    log_fn(f"device={device}  params={model.num_params()/1e6:.2f}M  vocab={config.vocab_size}")

    # Optional Weights & Biases logging. Imported lazily so the dependency is only touched
    # when explicitly enabled (tests never enable it). See docs/12_training.md.
    run = None
    if config.wandb_log:
        import wandb

        run = wandb.init(
            project=config.wandb_project, name=config.wandb_run_name, config=vars(config)
        )

    ckpt_path = Path(out_dir) / "ckpt.pt"
    history: list[tuple[int, float, float]] = []
    best_val = float("inf")

    for step in range(config.max_iters + 1):
        metrics: dict[str, object] = {}

        # Periodic evaluation + checkpointing.
        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = estimate_loss(model, data, config, device)
            history.append((step, losses["train"], losses["val"]))
            log_fn(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            metrics["train/loss"] = losses["train"]
            metrics["val/loss"] = losses["val"]
            if run is not None:
                metrics["samples/text"] = wandb.Html(
                    f"<pre>{_sample_text(model, data, config, device)}</pre>"
                )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(ckpt_path, model, optimizer, config, data, step, best_val)

        if step == config.max_iters:
            if run is not None and metrics:
                run.log(metrics, step=step)
            break

        # One optimization step.
        x, y = data.get_batch("train", config.batch_size, config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if run is not None:
            metrics["train/batch_loss"] = loss.item()
            run.log(metrics, step=step)  # one log call per step keeps the x-axis aligned

    if run is not None:
        run.finish()
    log_fn(f"done. best val loss {best_val:.4f}. checkpoint: {ckpt_path}")
    return model, data, history


@torch.no_grad()
def _sample_text(model: GPT, data: TextData, config: GPTConfig, device: str, n: int = 200) -> str:
    """Generate a short sample (starting from a newline) to eyeball progress in W&B."""
    model.eval()
    start = torch.tensor([[data.tokenizer.stoi.get("\n", 0)]], device=device)
    out = model.generate(start, max_new_tokens=n)
    model.train()
    return data.tokenizer.decode(out[0].tolist())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train nanoGPT on Tiny Shakespeare.")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--data", type=str, default="data/input.txt")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {}
    if args.max_iters is not None:
        overrides["max_iters"] = args.max_iters
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.wandb:
        overrides["wandb_log"] = True
    if args.wandb_project is not None:
        overrides["wandb_project"] = args.wandb_project
    if args.run_name is not None:
        overrides["wandb_run_name"] = args.run_name
    train(GPTConfig(**overrides), data_path=args.data, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
