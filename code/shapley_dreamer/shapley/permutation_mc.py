from __future__ import annotations

import numpy as np

from .base import CoalitionValueFn, ShapleyEstimator, register


@register("perm_mc")
class PermutationMC(ShapleyEstimator):
    def __init__(self, n_samples: int = 32, seed: int = 0) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_settings(cls, settings) -> "PermutationMC":
        return cls(n_samples=settings.SHAPLEY_M, seed=settings.RANDOM_SEED)

    def estimate(
        self,
        coalition_value: CoalitionValueFn,
        n_players: int,
    ) -> np.ndarray:
        phi = np.zeros(n_players, dtype=float)
        for _ in range(self.n_samples):
            perm = self._rng.permutation(n_players)
            mask = np.zeros(n_players, dtype=bool)
            v_prev = coalition_value(mask.copy())
            for i in perm:
                mask[i] = True
                v_curr = coalition_value(mask.copy())
                phi[i] += v_curr - v_prev
                v_prev = v_curr
        return phi / self.n_samples