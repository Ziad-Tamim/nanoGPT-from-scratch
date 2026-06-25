"""Model & training configuration, plus device selection.

All hyperparameters live here so no module contains magic numbers. The defaults match
docs/00_plan.md: a small model that trains in minutes on a single GPU yet still produces
readable Shakespeare.

Shape convention used throughout the codebase: (B, T, C) = (batch, time/sequence, channels).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    """Hyperparameters for the GPT model and training loop.

    Attributes
    ----------
    vocab_size : int
        Number of distinct tokens. Set from the dataset (~65 for char-level Shakespeare).
    block_size : int
        Context length T — the maximum number of tokens the model attends over.
    n_embd : int
        Embedding/model width C. Must be divisible by ``n_head``.
    n_head : int
        Number of attention heads. ``head_size = n_embd // n_head``.
    n_layer : int
        Number of stacked Transformer blocks.
    dropout : float
        Dropout probability used in attention and the feed-forward network.
    pos_encoding : str
        "sinusoidal" (fixed, matches the diagram) or "learned" (GPT-style).
    """

    # --- model ---
    vocab_size: int = 65          # overwritten once the dataset/tokenizer is built
    block_size: int = 256         # context length (T)
    n_embd: int = 384             # model width (C)
    n_head: int = 6               # head_size = n_embd // n_head = 64
    n_layer: int = 6              # number of Transformer blocks
    dropout: float = 0.2
    pos_encoding: str = "sinusoidal"  # or "learned"

    # --- training ---
    batch_size: int = 64          # sequences per step (B)
    learning_rate: float = 3e-4   # AdamW
    max_iters: int = 5000
    eval_interval: int = 500      # evaluate train/val loss every N steps
    eval_iters: int = 200         # batches to average when estimating loss
    weight_decay: float = 0.1
    grad_clip: float = 1.0        # max global grad norm (0 disables)

    # --- io ---
    seed: int = 1337
    device: str = "auto"          # "auto" -> get_device(); or "cuda"/"cpu"

    # --- logging (Weights & Biases) ---
    wandb_log: bool = False                       # enable W&B logging
    wandb_project: str = "nanogpt-shakespeare"    # project name on wandb.ai
    wandb_run_name: str | None = None             # run name (None -> W&B auto-generates)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
            )
        if self.pos_encoding not in ("sinusoidal", "learned"):
            raise ValueError(
                f"pos_encoding must be 'sinusoidal' or 'learned', got {self.pos_encoding!r}."
            )

    @property
    def head_size(self) -> int:
        """Dimension of each attention head."""
        return self.n_embd // self.n_head

    def resolve_device(self) -> str:
        """Return the concrete device string, expanding 'auto'."""
        return get_device() if self.device == "auto" else self.device


def get_device() -> str:
    """Pick the best available device: CUDA GPU if present, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
