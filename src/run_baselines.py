from __future__ import annotations

import argparse
import copy
import logging

from src.training.detection_trainer import train_detector
from src.utils.config import load_config
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection baselines and SSL-backbone experiment.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--include-ssl", action="store_true", help="Also run experiment C with SSL backbone weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config, args.env_config)
    configure_logging(base_config.get("logging", {}).get("level", "INFO"))
    fractions = base_config["detection"].get("fractions", [base_config["detection"]["labeled_fraction"]])

    for fraction in fractions:
        config = copy.deepcopy(base_config)
        config["detection"]["labeled_fraction"] = float(fraction)
        suffix = int(float(fraction) * 100)
        train_detector(config, initialization="pretrained", experiment_name=f"baseline_pretrained_{suffix}pct")
        train_detector(config, initialization="random", experiment_name=f"baseline_random_{suffix}pct")
        if args.include_ssl:
            train_detector(
                config,
                initialization="ssl",
                experiment_name=f"ssl_backbone_{suffix}pct",
                backbone_checkpoint=args.backbone_checkpoint,
            )
    LOGGER.info("Baseline runs finished.")


if __name__ == "__main__":
    main()

