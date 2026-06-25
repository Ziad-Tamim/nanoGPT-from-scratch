"""Positional encoding — inject token order into the otherwise order-blind model.

See docs/05_positional_encoding.md and docs/assets/05_positional_encoding.png, and the
interactive widget at web/widgets/05_positional_encoding.js.

Self-attention is permutation-invariant: it sees a *set* of token vectors, not a sequence.
We restore order by *adding* a position-dependent vector to each token embedding.

Two variants (selected by ``GPTConfig.pos_encoding``):

* ``SinusoidalPositionalEncoding`` — fixed sin/cos of different frequencies (Vaswani et al.).
  No parameters; generalizes to longer sequences; matches the architecture diagram.
* ``LearnedPositionalEncoding`` — a trainable table indexed by position (GPT-2 style).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    r"""Fixed sinusoidal positional encoding.

    For position ``pos`` and dimension index ``i`` (with ``C = n_embd``):

        PE(pos, 2k)   = sin( pos / base^(2k / C) )
        PE(pos, 2k+1) = cos( pos / base^(2k / C) )

    Each dimension is a sinusoid whose wavelength grows geometrically from ``2π`` up to
    ``base * 2π``. The encoding is precomputed once and stored as a (non-learnable) buffer.
    """

    def __init__(self, n_embd: int, block_size: int, base: float = 10000.0) -> None:
        super().__init__()
        pe = torch.zeros(block_size, n_embd)                       # (T_max, C)
        position = torch.arange(block_size).unsqueeze(1).float()   # (T_max, 1)
        # div_term[k] = 1 / base^(2k / C) = exp(-2k/C * ln(base)), shape (C/2,)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * (-math.log(base) / n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims -> sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims  -> cos
        # register_buffer: moves with .to(device)/state_dict but is NOT a learnable parameter.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to token embeddings. ``x``: (B, T, C) -> (B, T, C)."""
        T = x.size(1)
        return x + self.pe[:T]  # (T, C) broadcasts over the batch dimension


class LearnedPositionalEncoding(nn.Module):
    """Trainable positional embedding table indexed by position (GPT-2 style)."""

    def __init__(self, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.table = nn.Embedding(block_size, n_embd)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, C) -> (B, T, C). Adds the learned vector for each position."""
        T = x.size(1)
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        pos = torch.arange(T, device=x.device)   # (T,)
        return x + self.table(pos)               # (T, C) broadcasts over batch


def build_positional_encoding(config) -> nn.Module:
    """Construct the positional encoding chosen by ``config.pos_encoding``."""
    if config.pos_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(config.n_embd, config.block_size)
    return LearnedPositionalEncoding(config.n_embd, config.block_size)
