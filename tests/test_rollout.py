import random
from types import SimpleNamespace

import pytest

from shapley_dreamer.envs.sot import SoTEnv
from shapley_dreamer.llm.factory import build_llm
from shapley_dreamer.tasks.factory import build_task
from shapley_dreamer.training.rollout import collect_rollout, random_policy
from shapley_dreamer.training.trajectory import Trajectory


def _llm_settings():
    return SimpleNamespace(LLM_TYPE="mock", LLM_MAX_NEW_TOKENS=8, RANDOM_SEED=0)


def _task_settings():
    return SimpleNamespace(
        TASK_TYPE="word_problem",
        RANDOM_SEED=0,
        TASK_NUM_OPERANDS=3,
        TASK_MAX_VALUE=9,
        TASK_OPS="+,-",
    )


def _build_env(K=3, T_max=4):
    return SoTEnv(
        K=K,
        N=8,
        T_max=T_max,
        llm=build_llm(_llm_settings()),
        task_gen=build_task(_task_settings()),
    )


def test_rollout_runs_to_completion():
    env = _build_env(K=3, T_max=4)
    policy = random_policy(random.Random(0), K=3)

    traj = collect_rollout(env, policy)

    assert isinstance(traj, Trajectory)
    assert traj.length == 4
    assert len(traj.cells_history) == 5
    assert len(traj.rewards) == 4
    assert traj.rewards[:-1] == [0.0, 0.0, 0.0]


def test_rollout_terminal_reward_in_unit_interval():
    env = _build_env(K=2, T_max=3)
    policy = random_policy(random.Random(1), K=2)

    traj = collect_rollout(env, policy)

    assert 0.0 <= traj.terminal_reward <= 1.0


def test_rollout_actions_within_action_space():
    K = 4
    env = _build_env(K=K, T_max=6)
    policy = random_policy(random.Random(42), K=K)

    traj = collect_rollout(env, policy)

    assert all(code == SoTEnv.ACTION_LLM_WRITE for code, _ in traj.actions)
    assert all(0 <= idx < K for _, idx in traj.actions)


def test_rollout_carries_ground_truth():
    env = _build_env(K=2, T_max=2)
    policy = random_policy(random.Random(3), K=2)

    traj = collect_rollout(env, policy)

    assert traj.gt != ""
    int(traj.gt)  # gt must parse as int for the arithmetic task


def test_rollout_seed_determinism():
    K, T_max = 3, 5

    env1 = _build_env(K=K, T_max=T_max)
    env2 = _build_env(K=K, T_max=T_max)
    p1 = random_policy(random.Random(7), K=K)
    p2 = random_policy(random.Random(7), K=K)

    traj1 = collect_rollout(env1, p1)
    traj2 = collect_rollout(env2, p2)

    assert traj1.actions == traj2.actions
    assert traj1.cells_history == traj2.cells_history


def test_rollout_can_be_run_twice_on_same_env():
    K, T_max = 2, 3
    env = _build_env(K=K, T_max=T_max)
    policy = random_policy(random.Random(0), K=K)

    traj1 = collect_rollout(env, policy)
    traj2 = collect_rollout(env, policy)

    assert traj1.length == traj2.length == T_max