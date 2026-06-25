"""The full GPT model — assemble every component into a language model.

See docs/11_gpt_model.md and docs/assets/11_gpt_model.png.

Forward pass (decoder-only / GPT):

    idx (B, T) token ids
      -> token embedding            (B, T, C)      doc 04
      -> + positional encoding      (B, T, C)      doc 05
      -> dropout
      -> n_layer x Block            (B, T, C)      doc 10
      -> final LayerNorm            (B, T, C)      doc 09
      -> linear LM head             (B, T, vocab)
      -> (optional) cross-entropy loss against targets (B, T)

Also provides ``generate`` for autoregressive sampling (see docs/13_sampling.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogpt.block import Block
from nanogpt.config import GPTConfig
from nanogpt.embeddings import TokenEmbedding
from nanogpt.layer_norm import LayerNorm
from nanogpt.positional_encoding import build_positional_encoding


class GPT(nn.Module):
    """A decoder-only Transformer language model built from scratch components."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.n_embd)
        self.pos_encoding = build_positional_encoding(config)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            Block(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        )
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output projection. The input
        # embedding (V x C) and the LM head (C x V) describe the same token<->vector map, so
        # tying them saves V*C parameters and usually improves results (Press & Wolf, 2017).
        self.lm_head.weight = self.token_embedding.table.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style init: linears ~ N(0, 0.02), biases 0, embeddings ~ N(0, 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Total trainable parameters (counts the tied weight once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``idx`` (B, T) -> logits (B, T, vocab_size), and loss if ``targets`` is given."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        x = self.token_embedding(idx)   # (B, T, C)
        x = self.pos_encoding(x)        # (B, T, C)
        x = self.drop(x)

        all_weights = []
        for block in self.blocks:
            if return_weights:
                x, w = block(x, return_weights=True)
                all_weights.append(w)
            else:
                x = block(x)

        x = self.ln_f(x)                # (B, T, C)
        logits = self.lm_head(x)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy expects (N, vocab) logits and (N,) targets, so flatten B and T.
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T)
            )

        if return_weights:
            return logits, loss, torch.stack(all_weights, dim=1)  # (B, n_layer, n_head, T, T)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` (B, T) by ``max_new_tokens`` tokens.

        ``temperature`` < 1 sharpens the distribution (more greedy), > 1 flattens it.
        ``top_k`` restricts sampling to the k most likely tokens. See docs/13_sampling.md.
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (the model can't attend further).
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)          # (B, T, vocab)
            logits = logits[:, -1, :] / temperature  # focus on the last step -> (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
