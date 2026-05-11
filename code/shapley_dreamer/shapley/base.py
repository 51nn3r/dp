from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


CoalitionValueFn = Callable[[np.ndarray], float]
"""(keep_mask: bool array of shape (n,)) -> scalar coalition value."""


class ShapleyEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        coalition_value: CoalitionValueFn,
        n_players: int,
    ) -> np.ndarray:
        """Return per-player Shapley values, shape (n_players,)."""

    @classmethod
    @abstractmethod
    def from_settings(cls, settings) -> "ShapleyEstimator":
        ...


_REGISTRY: dict[str, type[ShapleyEstimator]] = {}


def register(name: str):
    def deco(cls: type[ShapleyEstimator]) -> type[ShapleyEstimator]:
        if name in _REGISTRY:
            raise ValueError(f"Shapley estimator {name!r} already registered")
        if not issubclass(cls, ShapleyEstimator):
            raise TypeError(f"{cls.__name__} must subclass ShapleyEstimator")
        _REGISTRY[name] = cls
        return cls

    return deco


def get_estimator(name: str) -> type[ShapleyEstimator]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Shapley estimator {name!r} not registered. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def registered_estimators() -> list[str]:
    return sorted(_REGISTRY)