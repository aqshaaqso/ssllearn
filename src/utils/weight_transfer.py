from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class TransferReport:
    loaded: list[str]
    skipped_missing: list[str]
    skipped_shape: list[str]

    @property
    def loaded_count(self) -> int:
        return len(self.loaded)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_missing) + len(self.skipped_shape)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loaded_count"] = self.loaded_count
        payload["skipped_count"] = self.skipped_count
        return payload


def normalize_backbone_key(key: str) -> str:
    if key.startswith("backbone."):
        key = key.removeprefix("backbone.")
    match = re.match(r"^layers\.(\d+)\.(.+)$", key)
    if match:
        return f"model.{match.group(1)}.{match.group(2)}"
    return key


def extract_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    candidates = [payload.get("state_dict"), payload.get("backbone_state_dict"), payload]
    for candidate in candidates:
        if isinstance(candidate, dict) and all(torch.is_tensor(v) for v in candidate.values()):
            return {normalize_backbone_key(k): v for k, v in candidate.items()}
    raise ValueError("Checkpoint does not contain a tensor state_dict or backbone_state_dict.")


def transfer_backbone_weights(
    detector_model: torch.nn.Module,
    backbone_state_dict: dict[str, torch.Tensor],
    report_path: str | Path | None = None,
) -> TransferReport:
    current = detector_model.state_dict()
    updated = dict(current)
    loaded: list[str] = []
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []

    for source_key, source_tensor in backbone_state_dict.items():
        key = normalize_backbone_key(source_key)
        if key not in current:
            skipped_missing.append(key)
            continue
        if tuple(current[key].shape) != tuple(source_tensor.shape):
            skipped_shape.append(f"{key}: checkpoint={tuple(source_tensor.shape)} detector={tuple(current[key].shape)}")
            continue
        updated[key] = source_tensor.detach().to(device=current[key].device, dtype=current[key].dtype)
        loaded.append(key)

    detector_model.load_state_dict(updated, strict=True)
    report = TransferReport(loaded=loaded, skipped_missing=skipped_missing, skipped_shape=skipped_shape)
    LOGGER.info("Transferred %d tensors; skipped %d tensors.", report.loaded_count, report.skipped_count)

    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if not loaded:
        raise ValueError("No backbone weights were transferred. Check checkpoint compatibility.")
    return report

