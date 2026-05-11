from __future__ import annotations

from typing import Sequence

import torch
from torch import nn, optim

from .model import WorldModel


def wm_step_loss(
    wm: WorldModel,
    initial_cells: list[str],
    action_indices: Sequence[int],
    terminal_reward: float,
) -> torch.Tensor:
    pred = wm.predict_terminal(initial_cells, action_indices)
    target = torch.tensor(terminal_reward, dtype=pred.dtype, device=pred.device)
    return (pred - target) ** 2


def train_wm(
    wm: WorldModel,
    dataset: list[tuple[list[str], list[int], float]],
    epochs: int = 50,
    lr: float = 1e-3,
    log_every: int | None = None,
) -> list[float]:
    optimiser = optim.Adam(wm.parameters(), lr=lr)
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for cells, actions, R in dataset:
            optimiser.zero_grad()
            loss = wm_step_loss(wm, cells, actions, R)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
        epoch_loss /= max(1, len(dataset))
        losses.append(epoch_loss)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[wm epoch {epoch + 1}/{epochs}] mse={epoch_loss:.4f}")
    return losses