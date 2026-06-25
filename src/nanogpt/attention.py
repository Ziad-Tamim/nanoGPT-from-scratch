"""Single-head causal self-attention — how tokens communicate.

See docs/06_self_attention.md and docs/assets/06_self_attention.png.

Every token emits three vectors via learned linear projections of its embedding:

    query  q : "what am I looking for?"
    key    k : "what do I contain?"
    value  v : "what will I share if attended to?"

A token's output is a weighted average of all *values*, where the weight from token i to
token j is how well i's query matches j's key:

    scores  = Q Kᵀ / sqrt(head_size)      # (B, T, T) pairwise similarities
    scores  = mask future positions to -inf   # causal: token t sees only <= t
    weights = softmax(scores)             # each row sums to 1
    out     = weights @ V                 # (B, T, head_size)

The sqrt(head_size) scaling keeps the dot products from growing with dimension (which would
saturate the softmax). The causal mask is what makes this a left-to-right language model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # No bias: these are pure projections into query/key/value subspaces.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular matrix of ones; tril[i, j] = 1 iff j <= i (j is allowed for i).
        # Stored as a buffer so it moves with .to(device) but is not a learnable parameter.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``x``: (B, T, C) -> (B, T, head_size).

        If ``return_weights`` is True, also return the attention matrix (B, T, T) — handy
        for visualization (see the planned self-attention web widget).
        """
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # Pairwise similarity of every query with every key, scaled. (B,T,hs)@(B,hs,T)->(B,T,T)
        scores = q @ k.transpose(-2, -1) * self.head_size**-0.5
        # Causal mask: forbid attending to future positions by sending their score to -inf,
        # so softmax assigns them exactly zero weight.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, T, T), each row sums to 1
        weights = self.dropout(weights)

        v = self.value(x)        # (B, T, head_size)
        out = weights @ v        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        if return_weights:
            return out, weights
        return out
