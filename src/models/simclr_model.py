from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .yolov8_backbone import YOLOv8Backbone


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(
        self,
        variant: str = "yolov8n",
        pretrained: bool = True,
        embedding_dim: int = 128,
        projection_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = YOLOv8Backbone(variant=variant, pretrained=pretrained)
        self.projection_head = ProjectionHead(
            in_dim=self.backbone.out_channels,
            hidden_dim=projection_hidden_dim,
            out_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone.pooled_features(x)
        embedding = self.projection_head(pooled)
        return F.normalize(embedding, dim=1)

    def backbone_state_dict_for_transfer(self) -> dict[str, torch.Tensor]:
        return self.backbone.backbone_state_dict_for_transfer()

