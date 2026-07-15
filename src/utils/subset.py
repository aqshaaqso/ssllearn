from __future__ import annotations

import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images found in: {image_dir}")
    return images


def label_for_image(image_path: Path, train_images_dir: Path, train_labels_dir: Path) -> Path:
    relative = image_path.relative_to(train_images_dir).with_suffix(".txt")
    return train_labels_dir / relative


def classes_in_label(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    classes: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def select_labeled_subset(
    train_images_dir: str | Path,
    train_labels_dir: str | Path,
    fraction: float,
    seed: int,
) -> list[Path]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the range (0, 1].")
    train_images_dir = Path(train_images_dir)
    train_labels_dir = Path(train_labels_dir)
    images = find_images(train_images_dir)
    target_count = max(1, round(len(images) * fraction))

    rng = random.Random(seed)
    if fraction >= 1:
        selected = images
    else:
        by_class: dict[int, list[Path]] = defaultdict(list)
        no_label: list[Path] = []
        for image in images:
            label = label_for_image(image, train_images_dir, train_labels_dir)
            classes = classes_in_label(label)
            if not label.exists():
                raise FileNotFoundError(f"Missing YOLO label for train image: {image} -> {label}")
            if not classes:
                no_label.append(image)
            for cls in classes:
                by_class[cls].append(image)

        pools = [sorted(set(paths)) for _, paths in sorted(by_class.items())]
        for pool in pools:
            rng.shuffle(pool)
        rng.shuffle(no_label)

        selected_set: set[Path] = set()
        while len(selected_set) < target_count and any(pools):
            for pool in pools:
                if pool and len(selected_set) < target_count:
                    selected_set.add(pool.pop())
        remaining = [image for image in images if image not in selected_set]
        rng.shuffle(remaining)
        for image in remaining:
            if len(selected_set) >= target_count:
                break
            selected_set.add(image)
        selected = sorted(selected_set)

    LOGGER.info("Selected %d/%d train images for fraction %.3f.", len(selected), len(images), fraction)
    return selected


def write_subset_file(selected_images: Iterable[Path], subset_file: str | Path) -> Path:
    subset_file = Path(subset_file)
    subset_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(path.resolve()).replace("\\", "/") for path in selected_images]
    subset_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return subset_file


def load_dataset_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {path}")
    for key in ("train", "val"):
        if key not in data:
            raise ValueError(f"Dataset YAML missing key: {key}")
    return data


def make_fraction_dataset_yaml(
    source_yaml: str | Path,
    output_dir: str | Path,
    fraction: float,
    seed: int,
) -> Path:
    data = load_dataset_yaml(source_yaml)
    source_yaml = Path(source_yaml)
    base = Path(data.get("path", source_yaml.parent)).expanduser()
    if not base.is_absolute():
        base = (source_yaml.parent / base).resolve()

    train_value = Path(str(data["train"]))
    train_images_dir = train_value if train_value.is_absolute() else base / train_value
    train_labels_dir = Path(str(train_images_dir)).as_posix().replace("/images/", "/labels/")
    train_labels_dir = Path(train_labels_dir)

    selected = select_labeled_subset(train_images_dir, train_labels_dir, fraction, seed)
    output_dir = Path(output_dir)
    subset_file = write_subset_file(selected, output_dir / "subsets" / f"train_{int(fraction * 100)}_percent.txt")

    subset_yaml = dict(data)
    subset_yaml["train"] = str(subset_file.resolve()).replace("\\", "/")
    subset_yaml_path = output_dir / "subsets" / f"dataset_train_{int(fraction * 100)}_percent.yaml"
    subset_yaml_path.write_text(yaml.safe_dump(subset_yaml, sort_keys=False), encoding="utf-8")

    if Path(str(data["val"])).resolve() == subset_file.resolve():
        raise ValueError("Validation data resolved to the train subset file; refusing data leakage.")
    return subset_yaml_path

