"""Token embeddings — turn discrete token ids into dense, learnable vectors.

See docs/04_embeddings.md and docs/assets/04_embeddings.png.

This is the first *learnable* component. An embedding is a lookup table

    E in R^{vocab_size x C}

where row ``E[i]`` is the C-dimensional vector for token id ``i``. Looking up a batch of ids
is a gather (indexing), not a matrix multiply, though it is mathematically equivalent to
multiplying a one-hot ``(B, T, vocab_size)`` tensor by ``E``. Gradients flow only to the rows
that were actually used in the batch (a sparse update).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Maps token ids ``(B, T)`` to embedding vectors ``(B, T, C)``."""

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, n_embd)
        # GPT-2 initializes embeddings from N(0, 0.02). Small weights keep the initial
        # activations (and thus the softmax logits) in a stable range before training.
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """``idx`` is a ``(B, T)`` LongTensor of token ids -> ``(B, T, C)`` embeddings."""
        return self.table(idx)
