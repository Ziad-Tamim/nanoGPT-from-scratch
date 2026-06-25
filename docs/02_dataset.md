# Dataset — Tiny Shakespeare

## What it is

A single ~1 MB plain-text file containing a concatenation of Shakespeare's plays. It is the
canonical nanoGPT starter dataset: small enough to train in minutes, large enough to learn
real structure (character names, line breaks, archaic spelling, dialogue formatting).

Source: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

`scripts/prepare_data.py` downloads it into `data/` (gitignored).

## From text to training batches

The whole pipeline for a **character-level** model:

1. **Read** the file as one long string `text`.
2. **Build the vocabulary:** `chars = sorted(set(text))` → ~65 unique characters.
   Create two maps: `stoi` (char→int) and `itos` (int→char). This *is* the tokenizer
   (see [03_tokenization.md](03_tokenization.md)).
3. **Encode** the entire text to a 1-D tensor of ints: `data = encode(text)`, shape `(N,)`.
4. **Split** into train/val, e.g. first 90% train, last 10% val — so we can measure
   overfitting honestly.
5. **Sample batches.** To make one batch we pick `batch_size` random start positions `i`
   and slice:
   - inputs  `x = data[i : i + block_size]`            → shape `(B, T)`
   - targets `y = data[i+1 : i+1 + block_size]`        → shape `(B, T)`
   `y` is `x` shifted by one: at each position the model predicts the **next** character.
   This is the "Outputs (shifted right)" idea from the architecture diagram.

## Shapes

| Name | Shape | Meaning |
|---|---|---|
| `data` | `(N,)` | whole corpus as token ids |
| `x` | `(B, T)` | inputs; `T = block_size` |
| `y` | `(B, T)` | targets = inputs shifted by +1 |

## Why character-level first

- No external tokenizer library; the mapping is a few lines you can read.
- Tiny vocab keeps the embedding table and LM head small.
- You can *see* the model learn structure (spaces, line breaks, names) quickly.

Trade-off: sequences are long in characters, so the model spends capacity learning spelling.
A BPE/subword tokenizer (GPT-2 style) is the natural next upgrade; the code keeps the
tokenizer behind an interface so it can be swapped. See [03_tokenization.md](03_tokenization.md).

## Diagram

`docs/assets/02_dataset_batching.png` — illustrate the sliding-window sampling of `x` and
the shifted `y`.
