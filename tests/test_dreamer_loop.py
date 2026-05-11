import random

import torch
from torch import nn, optim

from shapley_dreamer.agent.actor_critic import ActorCritic
from shapley_dreamer.envs.sot import SoTEnv
from shapley_dreamer.training.dreamer_loop import (
    collect_random_episodes,
    evaluate_on_env,
    imagined_train_step,
)
from shapley_dreamer.world_model.encoder import WMEncoder
from shapley_dreamer.world_model.model import WorldModel


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) % 32 for c in text]], dtype=torch.long)
        return type("Out", (), {"input_ids": ids})()


class _MockLLM:
    def generate(self, cells, target_cell, max_new_tokens):
        return f"x{target_cell}"


class _FixedTask:
    def sample(self):
        return ("3 + 5 = ?", "8")

    def eval(self, answer, gt):
        return 1.0 if "8" in answer else 0.0


def _make_wm_ac(K=3, T_max=4, embed_dim=8, d_model=8, nhead=2):
    tok = _FakeTokenizer()
    emb = nn.Embedding(32, embed_dim)
    enc = WMEncoder(tok, emb, K=K, T_max=T_max, d_model=d_model, nhead=nhead, num_layers=1)
    wm = WorldModel(enc, K=K)
    ac = ActorCritic(K=K, T_max=T_max, d_model=d_model, nhead=nhead, num_layers=1)
    return wm, ac


def test_imagined_train_step_terminal_runs():
    wm, ac = _make_wm_ac()
    opt = optim.Adam(ac.parameters(), lr=1e-3)
    stats = imagined_train_step(
        wm, ac, opt,
        initial_cells=["q", "", ""], T_max=4, reward_mode="terminal",
    )
    assert isinstance(stats.actor_loss, float)
    assert isinstance(stats.critic_loss, float)


def test_imagined_train_step_shapley_runs():
    wm, ac = _make_wm_ac()
    opt = optim.Adam(ac.parameters(), lr=1e-3)
    stats = imagined_train_step(
        wm, ac, opt,
        initial_cells=["q", "", ""], T_max=3, reward_mode="shapley",
    )
    assert isinstance(stats.actor_loss, float)


def test_collect_random_episodes_shapes():
    env = SoTEnv(K=3, N=8, T_max=4, llm=_MockLLM(), task_gen=_FixedTask())
    eps = collect_random_episodes(env, n_episodes=5, rng=random.Random(0))
    assert len(eps) == 5
    for cells, actions, R in eps:
        assert len(cells) == 3
        assert len(actions) == 4
        assert R in (0.0, 1.0)


def test_evaluate_on_env_returns_rate_in_unit_interval():
    wm, ac = _make_wm_ac()
    env = SoTEnv(K=3, N=8, T_max=4, llm=_MockLLM(), task_gen=_FixedTask())
    rate = evaluate_on_env(ac, wm, env, n_episodes=3)
    assert 0.0 <= rate <= 1.0