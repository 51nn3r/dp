import pytest
import torch
from torch import nn

from shapley_dreamer.world_model.encoder import WMEncoder


class _FakeTokenizer:
    """Trivial char-level tokenizer for tests."""

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) % 32 for c in text]], dtype=torch.long)
        return type("Out", (), {"input_ids": ids})()


def _make_encoder(K=3, T_max=4, embed_dim=8, d_model=8, nhead=2):
    tok = _FakeTokenizer()
    emb = nn.Embedding(32, embed_dim)
    return WMEncoder(
        tok, emb, K=K, T_max=T_max,
        d_model=d_model, nhead=nhead, num_layers=1,
    )


def test_state_dim_consistent():
    enc = _make_encoder(K=3, T_max=4, d_model=8)
    assert enc.state_dim == 3 * 8 + (4 + 1)


def test_forward_returns_correct_shape():
    enc = _make_encoder()
    out = enc(["abc", "", "x"], step=0)
    assert out.shape == (enc.state_dim,)


def test_step_one_hot_set_correctly():
    enc = _make_encoder(T_max=4)
    out = enc(["a", "b", "c"], step=2)
    step_part = out[-(enc.T_max + 1):]
    assert step_part.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_token_embedding_is_frozen():
    enc = _make_encoder()
    assert all(not p.requires_grad for p in enc.token_embedding.parameters())
    assert enc.in_proj.weight.requires_grad
    # Transformer params are trainable
    assert any(
        p.requires_grad for p in enc.transformer.parameters()
    )


def test_empty_cell_does_not_crash():
    enc = _make_encoder()
    out = enc(["", "", ""], step=0)
    assert out.shape == (enc.state_dim,)
    assert torch.isfinite(out).all()


def test_wrong_K_raises():
    enc = _make_encoder(K=3)
    with pytest.raises(ValueError):
        enc(["a", "b"], step=0)


def test_different_cells_produce_different_states():
    enc = _make_encoder()
    a = enc(["abc", "", ""], step=0)
    b = enc(["xyz", "", ""], step=0)
    assert not torch.allclose(a, b)
