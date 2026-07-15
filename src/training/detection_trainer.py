from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.utils.checkpoint import load_checkpoint
from src.utils.seed import resolve_device, seed_everything
from src.utils.subset import make_fraction_dataset_yaml
from src.utils.weight_transfer import extract_state_dict, transfer_backbone_weights

LOGGER = logging.getLogger(__name__)


def _yolo_model_name(variant: str, initialization: str) -> str:
    if initialization == "pretrained":
        return variant if variant.endswith(".pt") else f"{variant}.pt"
    if initialization in {"random", "ssl"}:
        return variant if variant.endswith(".yaml") else f"{variant}.yaml"
    raise ValueError(f"Unknown initialization: {initialization}")


def _metric_value(results: Any, *names: str) -> float | None:
    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        for name in names:
            if name in results_dict:
                return float(results_dict[name])
    box = getattr(results, "box", None)
    if box is not None:
        mapping = {
            "precision": getattr(box, "mp", None),
            "recall": getattr(box, "mr", None),
            "mAP50": getattr(box, "map50", None),
            "mAP50-95": getattr(box, "map", None),
        }
        for name in names:
            if name in mapping and mapping[name] is not None:
                return float(mapping[name])
    return None


def collect_detection_metrics(results: Any, model: YOLO, training_time: float, best_epoch: int | None) -> dict[str, Any]:
    speed = getattr(results, "speed", {}) or {}
    model_module = getattr(model, "model", None)
    param_count = sum(p.numel() for p in model_module.parameters()) if model_module is not None else None
    return {
        "precision": _metric_value(results, "metrics/precision(B)", "precision"),
        "recall": _metric_value(results, "metrics/recall(B)", "recall"),
        "mAP50": _metric_value(results, "metrics/mAP50(B)", "mAP50"),
        "mAP50-95": _metric_value(results, "metrics/mAP50-95(B)", "mAP50-95"),
        "training_time": training_time,
        "inference_time": speed.get("inference") if isinstance(speed, dict) else None,
        "num_parameters": param_count,
        "best_epoch": best_epoch,
    }


def train_detector(
    config: dict[str, Any],
    initialization: str,
    experiment_name: str,
    backbone_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    output_dir = Path(config["output_dir"])
    fraction = float(config["detection"]["labeled_fraction"])
    dataset_yaml = make_fraction_dataset_yaml(
        config["detection"]["dataset_yaml"],
        output_dir,
        fraction=fraction,
        seed=int(config["seed"]),
    )

    model_name = _yolo_model_name(str(config["model"]["variant"]), initialization)
    model = YOLO(model_name)
    run_dir = output_dir / "detection" / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    transfer_report = None
    if initialization == "ssl":
        if backbone_checkpoint is None:
            raise ValueError("SSL initialization requires --backbone-checkpoint.")
        payload = load_checkpoint(backbone_checkpoint, map_location="cpu")
        state_dict = extract_state_dict(payload)
        transfer_report = transfer_backbone_weights(
            model.model,
            state_dict,
            report_path=run_dir / "weight_transfer_report.json",
        ).to_dict()

    LOGGER.info(
        "Starting detection training: experiment=%s initialization=%s fraction=%.3f device=%s",
        experiment_name,
        initialization,
        fraction,
        device,
    )
    started_at = time.time()
    train_results = model.train(
        data=str(dataset_yaml),
        imgsz=int(config["detection"]["image_size"]),
        epochs=int(config["detection"]["epochs"]),
        batch=int(config["detection"]["batch_size"]),
        workers=int(config["detection"].get("num_workers", 0)),
        device=str(device),
        project=str(run_dir.resolve()),
        name="train",
        exist_ok=True,
        pretrained=False if initialization in {"random", "ssl"} else True,
        optimizer=config["detection"].get("optimizer", "auto"),
        patience=int(config["detection"].get("patience", 20)),
    )
    training_time = time.time() - started_at
    val_results = model.val(
        data=str(dataset_yaml),
        imgsz=int(config["detection"]["image_size"]),
        device=str(device),
        project=str(run_dir.resolve()),
        name="val",
        exist_ok=True,
    )

    best_epoch = getattr(train_results, "epoch", None)
    metrics = collect_detection_metrics(val_results, model, training_time=training_time, best_epoch=best_epoch)
    metrics.update(
        {
            "experiment": experiment_name,
            "labeled_fraction": fraction,
            "initialization": initialization,
            "dataset_yaml": str(dataset_yaml),
            "transfer_report": transfer_report,
        }
    )
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved detection metrics: %s", metrics_path)
    return metrics
