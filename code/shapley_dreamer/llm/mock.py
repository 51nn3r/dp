from __future__ import annotations

from .base import LLMBackend, register


@register("mock")
class MockBackend(LLMBackend):
    def __init__(self, max_new_tokens: int = 16) -> None:
        self.max_new_tokens = max_new_tokens
        self._calls = 0

    @classmethod
    def from_settings(cls, settings) -> "MockBackend":
        return cls(max_new_tokens=settings.LLM_MAX_NEW_TOKENS)

    def generate(
        self,
        cells: list[str],
        target_cell: int,
        max_new_tokens: int,
    ) -> str:
        self._calls += 1
        return f"mock[c={target_cell},n={self._calls}]"