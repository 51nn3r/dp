from __future__ import annotations

from typing import Protocol


Cells = list[str]
Action = tuple[int, int]


class LLMCallable(Protocol):
    def generate(self, cells: Cells, target_cell: int, max_new_tokens: int) -> str: ...


class TaskCallable(Protocol):
    def sample(self) -> tuple[str, str]: ...
    def eval(self, answer: str, gt: str) -> float: ...


class SoTEnv:
    """Среда SoT.

    State: K строковых ячеек (`""` = пусто). C0 — входной вопрос, C_{K-1} — ответ.
    Action: `(action_code, cell_idx)`.
    Reward: 0 в промежутке, терминальная при `t == T_max`.
    """

    ACTION_LLM_WRITE = 1

    def __init__(
        self,
        K: int,
        N: int,
        T_max: int,
        llm: LLMCallable,
        task_gen: TaskCallable,
    ) -> None:
        if K < 2:
            raise ValueError("K must be >= 2 (input + answer cell).")
        self.K = K
        self.N = N
        self.T_max = T_max
        self.llm = llm
        self.task_gen = task_gen
        self.cells: Cells = [""] * K
        self.t = 0
        self._gt: str = ""
        self._started = False
        self._done = False

    @property
    def answer_cell(self) -> int:
        return self.K - 1

    def reset(self) -> Cells:
        question, gt = self.task_gen.sample()
        self.cells = [""] * self.K
        self.cells[0] = question
        self._gt = gt
        self.t = 0
        self._started = True
        self._done = False
        return list(self.cells)

    def step(self, action: Action) -> tuple[Cells, float, bool, dict]:
        if not self._started:
            raise RuntimeError("step() called before reset().")
        if self._done:
            raise RuntimeError("step() called after episode is done; call reset() first.")

        code, idx = action
        if code != self.ACTION_LLM_WRITE:
            raise ValueError(f"Unsupported action_code: {code}")
        if not 0 <= idx < self.K:
            raise ValueError(f"cell_idx {idx} out of range [0, {self.K})")

        out = self.llm.generate(self.cells, idx, max_new_tokens=self.N)
        self.cells[idx] = out
        self.t += 1

        done = self.t >= self.T_max
        reward = self.task_gen.eval(self.cells[self.answer_cell], self._gt) if done else 0.0
        self._done = done
        return list(self.cells), reward, done, {"t": self.t, "gt": self._gt}
