# 12 — Training Loop

> **Code:** `src/nanogpt/train.py` · **Diagram:** `assets/12_training.png`
> **Status:** stub — fill in during Step 11.

## Intuition

Repeatedly: grab a batch, predict next tokens, measure the loss, and nudge the weights to do
better. Periodically check validation loss so we know we're learning, not memorizing.

## Mechanics

1. Build `GPTConfig`, model, move to device (CUDA if available).
2. Optimizer: **AdamW** (`lr ≈ 3e-4`, weight decay on matmul params, not on biases/norms).
3. Loop `max_iters`:
   - `x, y = get_batch('train')`
   - `logits, loss = model(x, y)`
   - `optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()`
   - every `eval_interval`: estimate mean train/val loss over a few batches (no_grad).
4. **Checkpoints:** save `{model_state, optimizer_state, config, iter, val_loss}` to
   `checkpoints/`. Save best-val and/or latest.
5. Log loss; optionally collect for a loss-curve plot (matplotlib/seaborn).

## Things to get right

- `model.train()` vs `model.eval()` (dropout/▷ behaviour).
- `torch.no_grad()` during evaluation.
- Optional: gradient clipping, cosine LR schedule, `torch.autocast` mixed precision on CUDA.
- Seed everything for reproducibility.

## References

- Loshchilov & Hutter (2019), AdamW; Karpathy, nanoGPT `train.py`.
