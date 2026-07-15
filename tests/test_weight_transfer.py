from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.utils.weight_transfer import transfer_backbone_weights


class TinyDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.ModuleList([torch.nn.Conv2d(3, 4, 1), torch.nn.Conv2d(4, 4, 1)])


def test_weight_transfer_loads_only_matching_shapes() -> None:
    detector = TinyDetector()
    state = {
        "model.0.weight": torch.ones_like(detector.state_dict()["model.0.weight"]),
        "model.0.bias": torch.ones_like(detector.state_dict()["model.0.bias"]),
        "model.1.weight": torch.randn(3, 3, 1, 1),
        "model.99.weight": torch.randn(1),
    }

    report = transfer_backbone_weights(detector, state)

    assert report.loaded_count == 2
    assert len(report.skipped_shape) == 1
    assert len(report.skipped_missing) == 1
    assert torch.allclose(detector.state_dict()["model.0.weight"], torch.ones_like(detector.state_dict()["model.0.weight"]))

