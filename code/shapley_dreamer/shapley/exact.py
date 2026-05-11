from __future__ import annotations

from math import factorial

import numpy as np

from .base import CoalitionValueFn, ShapleyEstimator, register


@register("exact")
class ExactShapley(ShapleyEstimator):
    @classmethod
    def from_settings(cls, settings) -> "ExactShapley":
        return cls()

    def estimate(
        self,
        coalition_value: CoalitionValueFn,
        n_players: int,
    ) -> np.ndarray:
        n = n_players
        phi = np.zeros(n, dtype=float)
        n_fact = factorial(n)

        cache: dict[int, float] = {}

        def v(subset: int) -> float:
            if subset not in cache:
                mask = np.array(
                    [(subset >> i) & 1 for i in range(n)], dtype=bool
                )
                cache[subset] = float(coalition_value(mask))
            return cache[subset]

        for subset in range(1 << n):
            v_S = v(subset)
            size_S = bin(subset).count("1")
            for i in range(n):
                if subset & (1 << i):
                    continue
                v_Si = v(subset | (1 << i))
                weight = factorial(size_S) * factorial(n - size_S - 1) / n_fact
                phi[i] += weight * (v_Si - v_S)

        return phi
