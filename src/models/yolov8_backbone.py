from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


def create_yolov8_model(variant: str = "yolov8n", pretrained: bool = True) -> YOLO:
    suffix = ".pt" if pretrained else ".yaml"
    model_name = variant if variant.endswith((".pt", ".yaml")) else f"{variant}{suffix}"
    LOGGER.info("Creating YOLO model from %s.", model_name)
    return YOLO(model_name)


def unwrap_yolo_module(yolo: YOLO) -> nn.Module:
    module = getattr(yolo, "model", None)
    if module is None:
        raise ValueError("Ultralytics YOLO object does not expose a .model module.")
    return module


def backbone_layer_count(yolo_module: nn.Module) -> int:
    yaml_data: dict[str, Any] | None = getattr(yolo_module, "yaml", None)
    if isinstance(yaml_data, dict) and "backbone" in yaml_data:
        count = len(yaml_data["backbone"])
        if count > 0:
            return count
    raise ValueError("Could not infer YOLOv8 backbone layer count from model.yaml['backbone'].")


def _layer_sequence(yolo_module: nn.Module) -> nn.ModuleList:
    layers = getattr(yolo_module, "model", None)
    if layers is None:
        raise ValueError("YOLO model module does not expose a .model layer sequence.")
    return layers


class YOLOv8Backbone(nn.Module):
    """Backbone-only forward wrapper for Ultralytics YOLOv8 models.

    The split point is read from Ultralytics model YAML metadata via the
    documented `backbone` section length. This avoids hardcoding "layer 10"
    while still matching the YOLOv8 graph definition used by Ultralytics.
    """

    def __init__(self, variant: str = "yolov8n", pretrained: bool = True) -> None:
        super().__init__()
        yolo = create_yolov8_model(variant=variant, pretrained=pretrained)
        object.__setattr__(self, "_yolo", yolo)
        self.yolo_module = unwrap_yolo_module(yolo)
        self.num_backbone_layers = backbone_layer_count(self.yolo_module)
        layers = _layer_sequence(self.yolo_module)
        self.layers = nn.ModuleList([layers[i] for i in range(self.num_backbone_layers)])
        self.save = set(int(i) for i in getattr(self.yolo_module, "save", []))
        LOGGER.info("Using %d YOLOv8 backbone layers.", self.num_backbone_layers)

    @property
    def out_channels(self) -> int:
        channels = getattr(self.layers[-1], "c2", None)
        if channels is not None:
            return int(channels)
        for module in reversed(list(self.layers[-1].modules())):
            if isinstance(module, nn.Conv2d):
                return int(module.out_channels)
        raise ValueError("Could not infer backbone output channels from the final backbone layer.")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        saved: list[torch.Tensor | None] = []
        for layer in self.layers:
            from_idx = getattr(layer, "f", -1)
            if from_idx != -1:
                if isinstance(from_idx, int):
                    x = saved[from_idx]
                else:
                    x = [x if j == -1 else saved[j] for j in from_idx]
            x = layer(x)
            layer_index = int(getattr(layer, "i", len(saved)))
            saved.append(x if layer_index in self.save else None)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if not torch.is_tensor(x):
            raise RuntimeError("Backbone output is not a tensor.")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, output_size=1), start_dim=1)

    def backbone_state_dict_for_transfer(self) -> dict[str, torch.Tensor]:
        state = self.yolo_module.state_dict()
        selected: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            parts = key.split(".")
            if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
                if int(parts[1]) < self.num_backbone_layers:
                    selected[key] = value.detach().cpu()
        if not selected:
            raise RuntimeError("No YOLOv8 backbone tensors were selected for transfer.")
        return selected
