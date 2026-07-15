from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}.")
        if z1.ndim != 2:
            raise ValueError("NT-Xent expects embeddings shaped [batch, dim].")
        batch_size = z1.shape[0]
        if batch_size < 2:
            raise ValueError("NT-Xent requires batch size at least 2.")

        z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits.masked_fill(torch.eye(2 * batch_size, device=z.device, dtype=torch.bool), float("-inf"))
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z.device),
                torch.arange(0, batch_size, device=z.device),
            ]
        )
        return F.cross_entropy(logits, labels)

