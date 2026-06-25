"""Character-level tokenizer — the reversible map between text and integer ids.

See docs/03_tokenization.md and docs/assets/03_tokenization.png.

A character-level model treats every distinct character as one token. The "vocabulary" is
just the sorted set of characters in the training corpus, and encoding/decoding are simple
dictionary lookups:

    stoi:  char -> int        (string-to-int)
    itos:  int  -> char       (int-to-string)

There are no learnable parameters here; this is fixed preprocessing. Learning begins at the
embedding table (see docs/04_embeddings.md). The class is deliberately small and behind a
simple ``encode`` / ``decode`` interface so a BPE/subword tokenizer can replace it later.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Maps characters to integer ids and back, built from a training corpus."""

    def __init__(self, chars: list[str]) -> None:
        # ``chars`` is the vocabulary: a sorted list of unique characters.
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from raw text by collecting its unique characters."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        """Text -> list of token ids. ``decode(encode(s)) == s`` (lossless)."""
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text."""
        return "".join(self.itos[int(i)] for i in ids)

    # --- persistence: train and generate must share the exact same vocab ---

    def save(self, path: str | Path) -> None:
        """Save the vocabulary to JSON so generation uses the identical mapping."""
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text("utf-8"))
        return cls(data["chars"])
