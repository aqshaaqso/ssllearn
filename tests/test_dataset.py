from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.datasets import ImageFolderDataset, SimCLRTransform


def test_dataset_returns_two_tensor_views(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), color=(120, 40, 20)).save(image_dir / "sample.jpg")

    dataset = ImageFolderDataset(image_dir, transform=SimCLRTransform(image_size=32))
    view1, view2 = dataset[0]

    assert torch.is_tensor(view1)
    assert torch.is_tensor(view2)
    assert tuple(view1.shape) == (3, 32, 32)
    assert tuple(view2.shape) == (3, 32, 32)

