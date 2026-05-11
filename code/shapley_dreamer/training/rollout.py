from __future__ import annotations

import random
from typing import Callable

from shapley_dreamer.envs.sot import Action, Cells, SoTEnv

from .trajectory import Trajectory


Policy = Callable[[Cells, int], Action]


def random_policy(rng: random.Random, K: int) -> Policy:
    def pick(cells: Cells, t: int) -> Action:
        return (SoTEnv.ACTION_LLM_WRITE, rng.randrange(K))

    return pick


def collect_rollout(env: SoTEnv, policy: Policy) -> Trajectory:
    cells = env.reset()
    cells_history: list[Cells] = [list(cells)]
    actions: list[Action] = []
    rewards: list[float] = []
    gt = ""

    done = False
    while not done:
        action = policy(cells, env.t)
        cells, reward, done, info = env.step(action)
        actions.append(action)
        cells_history.append(list(cells))
        rewards.append(reward)
        gt = info.get("gt", gt)

    return Trajectory(
        cells_history=cells_history,
        actions=actions,
        rewards=rewards,
        gt=gt,
    )