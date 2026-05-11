import pytest
import torch
from torch import nn

from shapley_dreamer.world_model.encoder import WMEncoder
from shapley_dreamer.world_model.model import WorldModel
from shapley_dreamer.world_model.training import train_wm, wm_step_loss


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) % 32 for c in text]], dtype=torch.long)
        return type("Out", (), {"input_ids": ids})()


def _make_wm(K=3, T_max=4, embed_dim=8, d_model=8, nhead=2):
    tok = _FakeTokenizer()
    emb = nn.Embedding(32, embed_dim)
    enc = WMEncoder(tok, emb, K=K, T_max=T_max, d_model=d_model, nhead=nhead, num_layers=1)
    return WorldModel(enc, K=K)


def test_encode_returns_state_vector():
    wm = _make_wm(K=3)
    s = wm.encode(["a", "b", "c"], step=0)
    assert s.shape == (wm.state_dim,)


def test_dynamics_step_preserves_shape():
    wm = _make_wm(K=3)
    s = wm.encode(["a", "b", "c"], step=0)
    s_next = wm.dynamics_step(s, action_idx=2)
    assert s_next.shape == s.shape
    assert not torch.allclose(s, s_next)


def test_predict_reward_returns_scalar():
    wm = _make_wm()
    s = wm.encode(["a", "b", "c"], step=0)
    r = wm.predict_reward(s)
    assert r.shape == ()


def test_predict_terminal_runs_full_sequence():
    wm = _make_wm(K=3, T_max=4)
    r = wm.predict_terminal(["q", "", ""], action_indices=[1, 2, 1, 2])
    assert r.shape == ()


def test_predict_terminal_keep_mask_skips_ablated():
    wm = _make_wm(K=3, T_max=4)
    full = wm.predict_terminal(["q", "", ""], action_indices=[1, 2, 1, 2])
    none_kept = wm.predict_terminal(
        ["q", "", ""], action_indices=[1, 2, 1, 2],
        keep_mask=[False, False, False, False],
    )
    # Empty coalition should differ from full coalition (almost surely).
    assert not torch.allclose(full, none_kept, atol=1e-6)


def test_training_reduces_loss_on_synthetic_data():
    wm = _make_wm(K=3, T_max=2)
    # Synthetic deterministic mapping: terminal_R = 1.0 if last action is 2, else 0.0
    dataset = []
    for last in (0, 1, 2):
        for first in (0, 1, 2):
            cells = ["q", "", ""]
            actions = [first, last]
            R = 1.0 if last == 2 else 0.0
            dataset.append((cells, actions, R))

    losses = train_wm(wm, dataset, epochs=200, lr=5e-3)
    assert losses[-1] < 0.5 * losses[0]
    # After training, full predict should be close to ground truth on held set.
    pred_solved = wm.predict_terminal(["q", "", ""], action_indices=[0, 2]).item()
    pred_failed = wm.predict_terminal(["q", "", ""], action_indices=[0, 1]).item()
    assert pred_solved > pred_failed