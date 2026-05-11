from __future__ import annotations

import torch
from torch import nn


class WMEncoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        token_embedding: nn.Embedding,
        K: int,
        T_max: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.token_embedding = token_embedding
        for p in self.token_embedding.parameters():
            p.requires_grad = False

        embed_dim = token_embedding.embedding_dim
        self.in_proj = nn.Linear(embed_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.K = K
        self.T_max = T_max
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.state_dim = K * d_model + (T_max + 1)

    def _encode_cell(self, cell: str) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = self.in_proj.weight.dtype
        if not cell:
            return torch.zeros(self.d_model, device=device, dtype=dtype)
        ids = self.tokenizer(cell, return_tensors="pt").input_ids[0].to(device)
        ids = ids[: self.max_seq_len]
        with torch.no_grad():
            emb = self.token_embedding.weight[ids].to(dtype=dtype)
        x = self.in_proj(emb)
        positions = torch.arange(x.size(0), device=device)
        x = x + self.pos_emb(positions)
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        return x.mean(dim=0)

    def forward(self, cells: list[str], step: int) -> torch.Tensor:
        if len(cells) != self.K:
            raise ValueError(f"Expected {self.K} cells, got {len(cells)}")
        cell_vecs = [self._encode_cell(c) for c in cells]
        cells_concat = torch.cat(cell_vecs, dim=0)
        device = cells_concat.device
        step_oh = torch.zeros(self.T_max + 1, device=device)
        step_oh[step] = 1.0
        return torch.cat([cells_concat, step_oh], dim=0)