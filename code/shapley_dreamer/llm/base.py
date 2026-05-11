from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Contract used by SoTEnv. Matches envs.sot.LLMCallable protocol."""

    @abstractmethod
    def generate(
        self,
        cells: list[str],
        target_cell: int,
        max_new_tokens: int,
    ) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_settings(cls, settings) -> "LLMBackend":
        ...


_REGISTRY: dict[str, type[LLMBackend]] = {}


def register(name: str):
    def deco(cls: type[LLMBackend]) -> type[LLMBackend]:
        if name in _REGISTRY:
            raise ValueError(f"LLM backend {name!r} already registered")
        if not issubclass(cls, LLMBackend):
            raise TypeError(f"{cls.__name__} must subclass LLMBackend")
        _REGISTRY[name] = cls
        return cls

    return deco


def get_backend(name: str) -> type[LLMBackend]:
    if name not in _REGISTRY:
        raise KeyError(
            f"LLM backend {name!r} not registered. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def registered_backends() -> list[str]:
    return sorted(_REGISTRY)