"""Data pipeline: corpus -> token ids -> train/val split -> random batches.

See docs/02_dataset.md and docs/assets/02_dataset_batching.png.

The model is trained to predict the *next* token at every position. So a batch is a set of
windows ``x`` of length ``block_size`` and targets ``y`` equal to ``x`` shifted one step
right:

    x = data[i      : i + T]      # (B, T)
    y = data[i + 1  : i + 1 + T]  # (B, T) -- the next token at each position

We sample ``batch_size`` random start positions each step, so the model sees varied context.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nanogpt.tokenizer import CharTokenizer


class TextData:
    """Holds the encoded corpus split into train/val and serves random batches."""

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer | None = None,
        train_frac: float = 0.9,
    ) -> None:
        # Build (or reuse) the tokenizer, then encode the whole corpus once.
        self.tokenizer = tokenizer or CharTokenizer.from_text(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)  # (N,)

        n_train = int(len(data) * train_frac)
        self.train_data = data[:n_train]
        self.val_data = data[n_train:]

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "TextData":
        """Load a UTF-8 text corpus from disk."""
        text = Path(path).read_text(encoding="utf-8")
        return cls(text, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one random batch ``(x, y)``, each of shape ``(batch_size, block_size)``.

        ``y`` is ``x`` shifted by one position, so position t's target is token t+1.
        """
        data = self.train_data if split == "train" else self.val_data
        # Highest valid start index so that x[i:i+T] and y[i+1:i+1+T] stay in bounds.
        high = len(data) - block_size
        ix = torch.randint(high, (batch_size,), generator=generator)  # (B,)
        x = torch.stack([data[i : i + block_size] for i in ix])          # (B, T)
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # (B, T)
        # ``non_blocking`` helps overlap host->GPU copy when data is pinned; harmless on CPU.
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
