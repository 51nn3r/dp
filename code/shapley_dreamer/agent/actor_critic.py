from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        K: int,
        T_max: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.K = K
        self.T_max = T_max
        self.d_model = d_model
        self.state_dim = K * d_model + (T_max + 1)

        self.step_proj = nn.Linear(T_max + 1, d_model)
        self.pos_emb = nn.Embedding(K + 1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.actor_head = nn.Linear(d_model, K)
        self.critic_head = nn.Linear(d_model, 1)

    def _pooled(self, state: torch.Tensor) -> torch.Tensor:
        cells = state[: self.K * self.d_model].view(self.K, self.d_model)
        step_oh = state[self.K * self.d_model:]
        step_emb = self.step_proj(step_oh).unsqueeze(0)
        seq = torch.cat([cells, step_emb], dim=0)
        positions = torch.arange(self.K + 1, device=state.device)
        seq = seq + self.pos_emb(positions)
        out = self.attn(seq.unsqueeze(0)).squeeze(0)
        return out.mean(dim=0)

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self._pooled(state))

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._pooled(state)).squeeze(-1)

    def sample(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=self.logits(state))
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)