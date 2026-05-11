from __future__ import annotations

from dataclasses import dataclass

import torch

from shapley_dreamer.agent.actor_critic import ActorCritic
from shapley_dreamer.world_model.model import WorldModel


@dataclass
class ImaginedRollout:
    states: list[torch.Tensor]       # length T_max + 1
    actions: list[int]               # length T_max
    log_probs: list[torch.Tensor]    # length T_max, each scalar
    terminal_reward: torch.Tensor    # scalar


def imagine_rollout(
    wm: WorldModel,
    ac: ActorCritic,
    initial_cells: list[str],
    T_max: int,
) -> ImaginedRollout:
    state = wm.encode(initial_cells, step=0)
    states = [state]
    actions: list[int] = []
    log_probs: list[torch.Tensor] = []
    for _ in range(T_max):
        a, log_p = ac.sample(state)
        state = wm.dynamics_step(state, a)
        states.append(state)
        actions.append(a)
        log_probs.append(log_p)
    terminal_r = wm.predict_reward(state)
    return ImaginedRollout(states, actions, log_probs, terminal_r)