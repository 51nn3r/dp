import numpy as np
import torch
from torch import nn

from shapley_dreamer.agent.actor_critic import ActorCritic
from shapley_dreamer.agent.imagination import imagine_rollout
from shapley_dreamer.shapley.exact import ExactShapley
from shapley_dreamer.shapley.imagined import compute_phi_imagined
from shapley_dreamer.world_model.encoder import WMEncoder
from shapley_dreamer.world_model.model import WorldModel


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) % 32 for c in text]], dtype=torch.long)
        return type("Out", (), {"input_ids": ids})()


def _make_wm_ac(K=3, T_max=4, embed_dim=8, d_model=8, nhead=2):
    tok = _FakeTokenizer()
    emb = nn.Embedding(32, embed_dim)
    enc = WMEncoder(tok, emb, K=K, T_max=T_max, d_model=d_model, nhead=nhead, num_layers=1)
    wm = WorldModel(enc, K=K)
    ac = ActorCritic(K=K, T_max=T_max, d_model=d_model, nhead=nhead, num_layers=1)
    return wm, ac


def test_actor_logits_shape():
    _, ac = _make_wm_ac(K=3)
    s = torch.randn(ac.state_dim)
    logits = ac.logits(s)
    assert logits.shape == (3,)


def test_critic_returns_scalar():
    _, ac = _make_wm_ac()
    s = torch.randn(ac.state_dim)
    v = ac.value(s)
    assert v.shape == ()


def test_sample_returns_int_and_log_prob():
    _, ac = _make_wm_ac(K=3)
    s = torch.randn(ac.state_dim)
    a, log_p = ac.sample(s)
    assert isinstance(a, int)
    assert 0 <= a < 3
    assert log_p.shape == ()
    assert torch.isfinite(log_p)


def test_imagine_rollout_lengths():
    wm, ac = _make_wm_ac(K=3, T_max=4)
    roll = imagine_rollout(wm, ac, ["q", "", ""], T_max=4)
    assert len(roll.states) == 5
    assert len(roll.actions) == 4
    assert len(roll.log_probs) == 4
    assert roll.terminal_reward.shape == ()


def test_compute_phi_imagined_satisfies_efficiency():
    wm, _ = _make_wm_ac(K=3, T_max=3)
    actions = [1, 2, 0]
    phi = compute_phi_imagined(wm, ["q", "", ""], actions, ExactShapley())
    # Efficiency: sum(phi) == v(full) - v(empty)
    full = wm.predict_terminal(["q", "", ""], actions).item()
    empty = wm.predict_terminal(
        ["q", "", ""], actions, keep_mask=[False] * len(actions),
    ).item()
    assert abs(phi.sum() - (full - empty)) < 1e-5