from __future__ import annotations

import numpy as np
import torch

from shapley_dreamer.shapley.base import CoalitionValueFn, ShapleyEstimator
from shapley_dreamer.world_model.model import WorldModel


def make_imagined_coalition_value(
    wm: WorldModel,
    initial_cells: list[str],
    action_indices: list[int],
) -> CoalitionValueFn:
    @torch.no_grad()
    def v(mask: np.ndarray) -> float:
        keep = [bool(b) for b in mask.tolist()]
        r = wm.predict_terminal(initial_cells, action_indices, keep_mask=keep)
        return float(r.item())

    return v


def compute_phi_imagined(
    wm: WorldModel,
    initial_cells: list[str],
    action_indices: list[int],
    estimator: ShapleyEstimator,
) -> np.ndarray:
    v = make_imagined_coalition_value(wm, initial_cells, action_indices)
    return estimator.estimate(v, n_players=len(action_indices))