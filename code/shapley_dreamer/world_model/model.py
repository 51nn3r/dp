from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .encoder import WMEncoder


class WorldModel(nn.Module):
    def __init__(self, encoder: WMEncoder, K: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.K = K
        self.state_dim = encoder.state_dim
        self.dynamics = nn.GRUCell(input_size=K, hidden_size=self.state_dim)
        self.reward_head = nn.Linear(self.state_dim, 1)

    def encode(self, cells: list[str], step: int) -> torch.Tensor:
        return self.encoder(cells, step)

    def dynamics_step(self, state: torch.Tensor, action_idx: int) -> torch.Tensor:
        device = state.device
        action_oh = torch.zeros(self.K, device=device)
        action_oh[action_idx] = 1.0
        return self.dynamics(action_oh.unsqueeze(0), state.unsqueeze(0)).squeeze(0)

    def predict_reward(self, state: torch.Tensor) -> torch.Tensor:
        return self.reward_head(state).squeeze(-1)

    def predict_terminal(
        self,
        initial_cells: list[str],
        action_indices: Sequence[int],
        keep_mask: Sequence[bool] | None = None,
    ) -> torch.Tensor:
        state = self.encode(initial_cells, step=0)
        for t, idx in enumerate(action_indices):
            if keep_mask is None or keep_mask[t]:
                state = self.dynamics_step(state, idx)
        return self.predict_reward(state)