from __future__ import annotations

from abc import ABC, abstractmethod


class TaskGenerator(ABC):
    """Contract used by SoTEnv. Matches envs.sot.TaskCallable protocol."""

    @abstractmethod
    def sample(self) -> tuple[str, str]:
        """Return (question, ground_truth)."""

    @abstractmethod
    def eval(self, answer: str, gt: str) -> float:
        """Score a model answer against gt. Return value in [0, 1]."""

    @classmethod
    @abstractmethod
    def from_settings(cls, settings) -> "TaskGenerator":
        ...


_REGISTRY: dict[str, type[TaskGenerator]] = {}


def register(name: str):
    def deco(cls: type[TaskGenerator]) -> type[TaskGenerator]:
        if name in _REGISTRY:
            raise ValueError(f"Task {name!r} already registered")
        if not issubclass(cls, TaskGenerator):
            raise TypeError(f"{cls.__name__} must subclass TaskGenerator")
        _REGISTRY[name] = cls
        return cls

    return deco


def get_task(name: str) -> type[TaskGenerator]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Task {name!r} not registered. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def registered_tasks() -> list[str]:
    return sorted(_REGISTRY)