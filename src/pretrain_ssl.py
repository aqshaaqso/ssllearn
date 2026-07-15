from __future__ import annotations

import argparse
import logging

from src.training import SSLTrainer
from src.utils.config import load_config
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised SimCLR pretraining for a YOLOv8 backbone.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to the base YAML config.")
    parser.add_argument("--env-config", default=None, help="Optional environment override YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.env_config)
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    LOGGER.info("Starting SSL pretraining.")
    metrics = SSLTrainer(config).train()
    LOGGER.info("SSL pretraining finished. Best loss: %.6f", metrics["best_loss"])


if __name__ == "__main__":
    main()

