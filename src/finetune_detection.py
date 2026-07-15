from __future__ import annotations

import argparse
import logging

from src.training.detection_trainer import train_detector
from src.utils.config import load_config
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 detection with optional SSL backbone weights.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--experiment-name", default="ssl_backbone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.env_config)
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    train_detector(
        config,
        initialization="ssl",
        experiment_name=args.experiment_name,
        backbone_checkpoint=args.backbone_checkpoint,
    )
    LOGGER.info("Detection fine-tuning finished.")


if __name__ == "__main__":
    main()

