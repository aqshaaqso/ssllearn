from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from src.utils.subset import make_fraction_dataset_yaml, select_labeled_subset


def _write_detection_dataset(root: Path, count: int = 10) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for idx in range(count):
        Image.new("RGB", (32, 32), color=(idx, idx, idx)).save(root / "images" / "train" / f"img_{idx}.jpg")
        (root / "labels" / "train" / f"img_{idx}.txt").write_text(f"{idx % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(root / "images" / "val" / "val.jpg")
    (root / "labels" / "val" / "val.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "a", 1: "b"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def test_subset_keeps_image_label_pairs(tmp_path: Path) -> None:
    dataset_yaml = _write_detection_dataset(tmp_path / "data")
    selected = select_labeled_subset(tmp_path / "data" / "images" / "train", tmp_path / "data" / "labels" / "train", 0.5, 42)

    assert len(selected) == 5
    for image in selected:
        label = tmp_path / "data" / "labels" / "train" / image.with_suffix(".txt").name
        assert label.exists()

    subset_yaml = make_fraction_dataset_yaml(dataset_yaml, tmp_path / "outputs", 0.5, 42)
    payload = yaml.safe_load(subset_yaml.read_text(encoding="utf-8"))
    train_list = Path(payload["train"])
    assert train_list.exists()
    assert "images/val" == payload["val"]
    assert str(train_list) != payload["val"]

