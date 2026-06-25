# 12 — Training Loop

> **Code:** `src/nanogpt/train.py` · **Diagram:** `assets/12_training.png`

## Intuition

Training is a loop: draw a batch, let the model predict the next token at every position,
measure how wrong it is (the loss), and nudge every weight a little to be less wrong. Repeat
thousands of times. Periodically we check the loss on held-out validation data so we can tell
learning from memorizing.

## The optimization step

Each iteration does four things:

```python
x, y = data.get_batch("train", ...)   # (B, T) inputs and shifted targets
_, loss = model(x, y)                 # forward: logits + cross-entropy (doc 11)
optimizer.zero_grad(set_to_none=True) # clear last step's gradients
loss.backward()                       # backprop: fill every .grad
clip_grad_norm_(..., grad_clip)       # rescale gradients if too large
optimizer.step()                      # AdamW updates the weights
```

### AdamW and the optimizer

We use **AdamW** (Adam with decoupled weight decay). Adam keeps per-parameter running
estimates of the gradient's first moment (mean) and second moment (variance) and scales each
update by them, which adapts the effective learning rate per parameter and makes training
robust. Betas `(0.9, 0.95)` follow the GPT recipe.

**Selective weight decay.** We apply weight decay (L2-style regularization) only to matmul
weights (`dim >= 2`: embeddings, linear layers) and **not** to 1-D parameters (biases,
LayerNorm `γ`/`β`). Decaying a normalization gain or a bias would just fight the layer's job.
See `configure_optimizers` and `test_optimizer_splits_decay_groups`.

### Gradient clipping

If a batch produces a huge gradient, a single step could destabilize training. We rescale the
global gradient so its norm never exceeds `grad_clip` (default 1.0):

$$g \leftarrow g \cdot \min\!\left(1, \frac{\texttt{grad\_clip}}{\lVert g\rVert_2}\right).$$

## Evaluation & checkpointing

Every `eval_interval` steps we call `estimate_loss`, which averages the loss over `eval_iters`
batches for both splits (under `torch.no_grad()` and `model.eval()` so dropout is off and no
graph is built). This smooths out the per-batch noise. Whenever validation loss hits a new
best, we save a checkpoint containing the model weights, optimizer state, config, and the
tokenizer vocab — everything needed to resume training or generate text.

### train() vs eval() mode

`model.train()` enables dropout (regularization during learning); `model.eval()` disables it
for deterministic evaluation and generation. `estimate_loss` flips to eval and back. Forgetting
this is a classic bug — your "validation" loss would secretly include dropout noise.

## What to expect on Tiny Shakespeare

Loss starts near `ln(65) ≈ 4.17` and, with the default config on a single GPU, falls to roughly
**1.5** within a few thousand steps. Around there the samples stop being random characters and
start producing Shakespeare-shaped lines (names, colons, line breaks, plausible words).

## Experiment tracking (Weights & Biases)

Training logs to [Weights & Biases](https://wandb.ai/) when enabled, so you can watch the
loss curves live in a dashboard. It is **off by default** and lazily imported, so it never
affects tests or runs that don't ask for it.

One-time login with your free API key (from <https://wandb.ai/authorize>):

```bash
uv run wandb login
```

Then enable it on a run:

```bash
uv run python -m nanogpt.train --wandb --run-name shakespeare-v1
```

What gets logged each step / eval:

| Metric | When | Meaning |
|---|---|---|
| `train/batch_loss` | every step | the single-batch training loss (noisy) |
| `train/loss`, `val/loss` | every `eval_interval` | averaged, smoothed losses |
| `samples/text` | every `eval_interval` | a short generated sample, to *see* progress |

Only scalar metrics, the config, and generated text samples are sent — the training corpus
itself is not uploaded. Controlled by `GPTConfig.wandb_log` / `wandb_project` /
`wandb_run_name`, or the `--wandb`, `--wandb-project`, `--run-name` CLI flags. To log locally
without an account, prefix with `WANDB_MODE=offline` (view later via `wandb sync`).

## Possible upgrades

- **Mixed precision** (`torch.autocast` + `GradScaler`) for faster GPU training.
- **Cosine learning-rate schedule** with warmup.
- **`torch.compile(model)`** for a speedup on recent PyTorch.

These are intentionally left out of the first version to keep the loop readable.

## References

- Kingma & Ba (2015), *Adam*; Loshchilov & Hutter (2019), *AdamW*.
- Karpathy, *nanoGPT* `train.py`.
