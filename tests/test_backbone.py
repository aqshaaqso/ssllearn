from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from src.models import SimCLRModel, YOLOv8Backbone


def test_backbone_accepts_image_tensor() -> None:
    backbone = YOLOv8Backbone(variant="yolov8n", pretrained=False)
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 64, 64))
    assert torch.is_tensor(features)
    assert features.ndim == 4


def test_projection_head_embedding_dim() -> None:
    model = SimCLRModel(variant="yolov8n", pretrained=False, embedding_dim=64, projection_hidden_dim=128)
    model.eval()
    with torch.no_grad():
        embedding = model(torch.randn(2, 3, 64, 64))
    assert tuple(embedding.shape) == (2, 64)

