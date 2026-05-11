from __future__ import annotations

import numpy as np

from shapley_dreamer.envs.sot import Action, Cells, LLMCallable, TaskCallable
from shapley_dreamer.shapley.base import CoalitionValueFn, ShapleyEstimator
from shapley_dreamer.training.trajectory import Trajectory


def rerun_with_ablation(
    initial_cells: Cells,
    actions: list[Action],
    keep_set: set[int],
    llm: LLMCallable,
    task: TaskCallable,
    gt: str,
    max_new_tokens: int,
) -> tuple[Cells, float]:
    cells = list(initial_cells)
    for t, action in enumerate(actions):
        if t not in keep_set:
            continue
        _, idx = action
        cells[idx] = llm.generate(cells, idx, max_new_tokens=max_new_tokens)
    return cells, task.eval(cells[-1], gt)


def make_coalition_value(
    initial_cells: Cells,
    actions: list[Action],
    llm: LLMCallable,
    task: TaskCallable,
    gt: str,
    max_new_tokens: int,
) -> CoalitionValueFn:
    def v(mask: np.ndarray) -> float:
        keep_set = {i for i, b in enumerate(mask.tolist()) if b}
        _, reward = rerun_with_ablation(
            initial_cells, actions, keep_set, llm, task, gt, max_new_tokens,
        )
        return reward

    return v


def compute_phi_for_trajectory(
    traj: Trajectory,
    llm: LLMCallable,
    task: TaskCallable,
    estimator: ShapleyEstimator,
    max_new_tokens: int,
) -> np.ndarray:
    initial = traj.cells_history[0]
    v = make_coalition_value(initial, traj.actions, llm, task, traj.gt, max_new_tokens)
    return estimator.estimate(v, n_players=len(traj.actions))