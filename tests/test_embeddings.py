"""Tests for TokenEmbedding. Run: uv run python -m pytest tests/ -q"""

import torch

from nanogpt.embeddings import TokenEmbedding


def test_output_shape_and_dtype():
    emb = TokenEmbedding(vocab_size=65, n_embd=384)
    idx = torch.randint(0, 65, (4, 16))  # (B, T)
    out = emb(idx)
    assert out.shape == (4, 16, 384)  # (B, T, C)
    assert out.dtype == torch.float32


def test_same_id_maps_to_same_vector():
    emb = TokenEmbedding(vocab_size=10, n_embd=8)
    idx = torch.tensor([[3, 3, 5]])
    out = emb(idx)
    assert torch.equal(out[0, 0], out[0, 1])      # both are token 3
    assert not torch.equal(out[0, 0], out[0, 2])  # token 3 vs token 5


def test_equivalent_to_onehot_matmul():
    emb = TokenEmbedding(vocab_size=6, n_embd=4)
    idx = torch.tensor([[0, 2, 5]])
    onehot = torch.nn.functional.one_hot(idx, num_classes=6).float()  # (1, 3, 6)
    expected = onehot @ emb.table.weight                              # (1, 3, 4)
    assert torch.allclose(emb(idx), expected, atol=1e-6)


def test_gradient_is_sparse_to_used_rows():
    emb = TokenEmbedding(vocab_size=10, n_embd=4)
    idx = torch.tensor([[1, 1, 3]])  # only rows 1 and 3 are used
    emb(idx).sum().backward()
    grad = emb.table.weight.grad
    assert grad[1].abs().sum() > 0    # used
    assert grad[3].abs().sum() > 0    # used
    assert grad[0].abs().sum() == 0   # unused -> zero gradient
    assert grad[2].abs().sum() == 0   # unused -> zero gradient
