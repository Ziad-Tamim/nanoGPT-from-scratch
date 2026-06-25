# 03 — Tokenization (character-level)

> **Code:** `src/nanogpt/tokenizer.py` · **Diagram:** `assets/03_tokenization.png`
> **Status:** stub — fill in during Step 2.

## Intuition

Neural nets work on numbers, not text. Tokenization is the reversible map between characters
and integer ids. Character-level: one character = one token.

## Math / mechanics

- Vocabulary `V = sorted(set(text))`, `vocab_size = |V|` (≈ 65).
- `stoi: char → int`, `itos: int → char`.
- `encode(s) = [stoi[c] for c in s]`, `decode(ids) = ''.join(itos[i] for i in ids)`.
- `decode(encode(s)) == s` (lossless).

## Shapes

- `encode`: string of length `L` → `(L,)` int tensor.

## Backprop note

No learnable parameters here — tokenization is fixed preprocessing, not part of the
computation graph. (The *embedding* that follows is where learning starts; see doc 04.)

## Extension

Swap for BPE/subword later behind the same `encode`/`decode` interface.

## References

- Karpathy, *Let's build GPT* (tokenizer section).
