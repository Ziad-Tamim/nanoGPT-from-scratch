"""Position-wise feed-forward network (MLP) — where each token "thinks".

See docs/08_feed_forward.md and docs/assets/08_feed_forward.png.

Attention (docs 06-07) mixes information *across* positions. The feed-forward network then
processes each position *independently* with a small 2-layer MLP, giving the model non-linear
capacity to transform what attention gathered:

    FFN(x) = Dropout( W2 @ GELU(W1 @ x + b1) + b2 )

The hidden layer is wider than the model (the standard 4x expansion), so the block can route
through a richer intermediate representation before projecting back to width C. "Position-wise"
means the *same* MLP is applied to every position with no mixing across the sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP applied independently at each sequence position."""

    def __init__(self, n_embd: int, dropout: float = 0.0, expansion: int = 4) -> None:
        super().__init__()
        hidden = expansion * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),   # expand: (B, T, C) -> (B, T, 4C)
            nn.GELU(),                   # smooth non-linearity (GPT-2 style)
            nn.Linear(hidden, n_embd),   # project back: (B, T, 4C) -> (B, T, C)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, C) -> (B, T, C). No mixing across the T (sequence) axis."""
        return self.net(x)
