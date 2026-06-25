"""Multi-head attention — run several attention heads in parallel and recombine.

See docs/07_multi_head_attention.md and docs/assets/07_multi_head_attention.png.

One head can only track one kind of relationship at a time. Multi-head attention splits the
model width ``C`` into ``n_head`` subspaces of size ``head_size = C // n_head``, runs an
independent causal-attention head (see attention.py) in each, concatenates their outputs back
to width ``C``, and mixes them with a final linear projection:

    head_i = Attention_i(x)                 # each (B, T, head_size)
    concat = [head_1, ..., head_h]          # (B, T, C)
    out    = Dropout(concat @ W_o)          # (B, T, C)

We build it from individual ``Head`` modules for clarity. (A production implementation fuses
the per-head projections into one big matmul and reshapes to (B, n_head, T, head_size); the
math is identical, just faster.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanogpt.attention import Head


class MultiHeadAttention(nn.Module):
    """``n_head`` parallel causal-attention heads followed by an output projection."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head}).")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        )
        # Output projection mixes information across heads, back into the residual stream.
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``x``: (B, T, C) -> (B, T, C).

        If ``return_weights``, also return per-head attention matrices (B, n_head, T, T).
        """
        if return_weights:
            outs, weights = zip(*(h(x, return_weights=True) for h in self.heads))
            out = torch.cat(outs, dim=-1)                 # (B, T, C)
            out = self.dropout(self.proj(out))
            return out, torch.stack(weights, dim=1)       # (B, n_head, T, T)

        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))
        return out
