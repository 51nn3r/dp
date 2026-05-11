import numpy as np
import pytest

from shapley_dreamer.shapley.counterfactual import (
    compute_phi_for_trajectory,
    make_coalition_value,
    rerun_with_ablation,
)
from shapley_dreamer.shapley.exact import ExactShapley
from shapley_dreamer.training.trajectory import Trajectory


class _ScriptedLLM:
    """Returns a fixed string per call, ignoring inputs."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, cells, target_cell, max_new_tokens):
        s = self.outputs[self.calls]
        self.calls += 1
        return s


class _ExactMatch:
    def sample(self):
        return ("q", "")

    def eval(self, answer, gt):
        return 1.0 if answer == gt else 0.0


def test_rerun_empty_keep_set_keeps_initial_cells():
    llm = _ScriptedLLM(["x", "y"])
    cells, _ = rerun_with_ablation(
        ["q", "", ""], actions=[(1, 1), (1, 2)], keep_set=set(),
        llm=llm, task=_ExactMatch(), gt="x", max_new_tokens=4,
    )
    assert cells == ["q", "", ""]
    assert llm.calls == 0


def test_rerun_full_keep_set_runs_all_steps():
    llm = _ScriptedLLM(["x", "y"])
    cells, _ = rerun_with_ablation(
        ["q", "", ""], actions=[(1, 1), (1, 2)], keep_set={0, 1},
        llm=llm, task=_ExactMatch(), gt="x", max_new_tokens=4,
    )
    assert cells == ["q", "x", "y"]
    assert llm.calls == 2


def test_rerun_partial_keep_set_skips_ablated_steps():
    llm = _ScriptedLLM(["only-call"])
    cells, _ = rerun_with_ablation(
        ["q", "", ""], actions=[(1, 1), (1, 2)], keep_set={1},
        llm=llm, task=_ExactMatch(), gt="x", max_new_tokens=4,
    )
    assert cells == ["q", "", "only-call"]
    assert llm.calls == 1


def test_rerun_returns_terminal_reward_via_task_eval():
    llm = _ScriptedLLM(["x", "8"])
    _, r = rerun_with_ablation(
        ["q", "", ""], actions=[(1, 1), (1, 2)], keep_set={0, 1},
        llm=llm, task=_ExactMatch(), gt="8", max_new_tokens=4,
    )
    assert r == 1.0


def test_coalition_value_empty_and_full():
    llm = _ScriptedLLM(["x", "8"])
    v = make_coalition_value(
        ["q", "", ""], [(1, 1), (1, 2)], llm, _ExactMatch(), gt="8", max_new_tokens=4,
    )
    # empty coalition: cells stay initial, answer cell == "" != "8"
    assert v(np.array([False, False])) == 0.0
    # full coalition: writes "8" into answer
    llm.calls = 0  # reset between coalition queries
    assert v(np.array([True, True])) == 1.0


def test_compute_phi_sums_to_full_minus_empty_with_exact():
    # 3-step trajectory; LLM scripted so only specific subsets give r=1.
    # Player 2 must run AND player 0 must have run (to set up cell 1) for
    # the answer to be "8". This makes the problem non-trivial for Shapley.
    class _ConditionalLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, cells, target_cell, max_new_tokens):
            self.calls += 1
            # cell 1 is "ready" only if it was previously written non-empty
            if target_cell == 2 and cells[1]:
                return "8"
            if target_cell == 1:
                return "ready"
            return ""

    actions = [(1, 1), (1, 0), (1, 2)]
    traj = Trajectory(
        cells_history=[["q", "", ""]],  # only initial state needed for compute_phi
        actions=actions,
        rewards=[0.0] * len(actions),
        gt="8",
    )
    phi = compute_phi_for_trajectory(
        traj=traj, llm=_ConditionalLLM(), task=_ExactMatch(),
        estimator=ExactShapley(), max_new_tokens=4,
    )
    # Efficiency: sum(phi) == v(full) - v(empty)
    v_full = 1.0  # actions [(1,1),(1,0),(1,2)]: cell1='ready' then cell0='', then cell2='8'
    v_empty = 0.0  # empty cells -> answer == "" != "8"
    assert phi.sum() == pytest.approx(v_full - v_empty, abs=1e-9)


def test_compute_phi_assigns_zero_to_useless_step():
    # Step 0 writes into cell 0 (overwrites question, useless).
    # Step 1 writes "8" into cell 2 (the answer cell). Should get all the credit.
    class _SimpleLLM:
        def generate(self, cells, target_cell, max_new_tokens):
            if target_cell == 2:
                return "8"
            return "garbage"

    traj = Trajectory(
        cells_history=[["q", "", ""]],
        actions=[(1, 0), (1, 2)],
        rewards=[0.0, 1.0],
        gt="8",
    )
    phi = compute_phi_for_trajectory(
        traj=traj, llm=_SimpleLLM(), task=_ExactMatch(),
        estimator=ExactShapley(), max_new_tokens=4,
    )
    # Step 0 (useless overwrite) should get phi ~ 0; step 1 should get ~1.0
    assert abs(phi[0]) < 1e-9
    assert phi[1] == pytest.approx(1.0)