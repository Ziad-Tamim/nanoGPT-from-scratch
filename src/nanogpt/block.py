"""Transformer block — the repeated unit (the ``Nx`` box in the architecture diagram).

See docs/10_transformer_block.md and docs/assets/10_transformer_block.png.

One block interleaves the two ways tokens are processed:

    communication : multi-head self-attention (tokens exchange information)  -- doc 07
    computation   : position-wise feed-forward (each token thinks)           -- doc 08

Each sublayer is wrapped in **pre-LayerNorm + a residual connection** (doc 09):

    x = x + MultiHeadAttention( LayerNorm(x) )
    x = x + FeedForward(        LayerNorm(x) )

Pre-LN keeps the residual path un-normalized, giving a clean gradient highway through the
whole stack. Stacking ``n_layer`` of these blocks builds the model's depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanogpt.feed_forward import FeedForward
from nanogpt.layer_norm import LayerNorm
from nanogpt.multi_head_attention import MultiHeadAttention


class Block(nn.Module):
    """A single pre-LN Transformer block: attention + feed-forward, each with a residual."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``x``: (B, T, C) -> (B, T, C). Optionally also return attention weights."""
        # Attention sublayer (pre-LN, residual). Note we normalize a *copy* via ln1 and add
        # the sublayer output back to the untouched residual stream ``x``.
        if return_weights:
            attn_out, weights = self.attn(self.ln1(x), return_weights=True)
            x = x + attn_out
            x = x + self.ffwd(self.ln2(x))
            return x, weights

        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
