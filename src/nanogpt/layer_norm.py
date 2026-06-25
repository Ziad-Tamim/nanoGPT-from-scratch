"""Layer normalization, implemented from scratch.

See docs/09_layernorm_residual.md and docs/assets/09_layernorm_residual.png.

LayerNorm stabilizes training by normalizing each token's feature vector to zero mean and
unit variance (over the C dimension), then rescaling with learnable parameters:

    x_hat = (x - mean) / sqrt(var + eps)        # normalize over the last (C) axis
    out   = gamma * x_hat + beta                # learnable per-feature scale/shift

Unlike BatchNorm, the statistics are computed per token (per position), independent of the
batch — so it behaves identically at train and inference time and works with any batch size.

This module pairs with the **residual connection** ``x = x + sublayer(x)`` used in the
Transformer block (see block.py / doc 09); together they make deep stacks trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Normalize over the last dimension (features), with learnable scale and shift."""

    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))   # scale, initialized to 1
        self.beta = nn.Parameter(torch.zeros(n_embd))   # shift, initialized to 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (..., C) -> (..., C). Normalizes over the last axis."""
        mean = x.mean(dim=-1, keepdim=True)                  # (..., 1)
        # Population variance (unbiased=False) to match torch.nn.LayerNorm.
        var = x.var(dim=-1, keepdim=True, unbiased=False)    # (..., 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
