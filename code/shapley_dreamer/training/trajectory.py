from __future__ import annotations

from dataclasses import dataclass, field

from shapley_dreamer.envs.sot import Action, Cells


@dataclass(frozen=True)
class Trajectory:
    cells_history: list[Cells] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    gt: str = ""

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def terminal_reward(self) -> float:
        return self.rewards[-1] if self.rewards else 0.0