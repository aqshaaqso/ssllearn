from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SimCLRTransform:
    """Create two independently augmented views from one image."""

    def __init__(
        self,
        image_size: int = 224,
        crop_scale: tuple[float, float] = (0.5, 1.0),
        horizontal_flip_p: float = 0.5,
        color_jitter_p: float = 0.8,
        color_jitter_strength: float = 0.5,
        grayscale_p: float = 0.2,
        gaussian_blur_p: float = 0.2,
    ) -> None:
        jitter = transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=min(0.5, color_jitter_strength / 2),
        )
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=crop_scale),
                transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                transforms.RandomApply([jitter], p=color_jitter_p),
                transforms.RandomGrayscale(p=grayscale_p),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=gaussian_blur_p),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SimCLRTransform":
        augment = config.get("ssl", {}).get("augment", {})
        crop_scale = tuple(augment.get("crop_scale", (0.5, 1.0)))
        return cls(
            image_size=int(config["ssl"]["image_size"]),
            crop_scale=(float(crop_scale[0]), float(crop_scale[1])),
            horizontal_flip_p=float(augment.get("horizontal_flip_p", 0.5)),
            color_jitter_p=float(augment.get("color_jitter_p", 0.8)),
            color_jitter_strength=float(augment.get("color_jitter_strength", 0.5)),
            grayscale_p=float(augment.get("grayscale_p", 0.2)),
            gaussian_blur_p=float(augment.get("gaussian_blur_p", 0.2)),
        )

    def __call__(self, image: Image.Image):
        return self.transform(image), self.transform(image)


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        transform: SimCLRTransform | None = None,
        corrupt_image_policy: str = "error",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform or SimCLRTransform()
        self.corrupt_image_policy = corrupt_image_policy
        if corrupt_image_policy not in {"error", "skip"}:
            raise ValueError("corrupt_image_policy must be either 'error' or 'skip'.")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"SSL image directory not found: {self.image_dir}")
        self.image_paths = sorted(path for path in self.image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.image_paths:
            raise ValueError(f"No image files found in SSL image directory: {self.image_dir}")
        if self.corrupt_image_policy == "skip":
            self.image_paths = self._filter_valid_images(self.image_paths)
        LOGGER.info("Loaded %d SSL images from %s.", len(self.image_paths), self.image_dir)

    def _filter_valid_images(self, image_paths: list[Path]) -> list[Path]:
        valid: list[Path] = []
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                valid.append(path)
            except (OSError, UnidentifiedImageError) as exc:
                LOGGER.warning("Skipping corrupt image %s: %s", path, exc)
        if not valid:
            raise ValueError(f"All images were corrupt or unreadable in: {self.image_dir}")
        return valid

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                view1, view2 = self.transform(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to read image for SSL training: {path}") from exc
        return view1, view2

