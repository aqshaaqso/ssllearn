from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TABLE_COLUMNS = [
    "experiment",
    "labeled_fraction",
    "initialization",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "best_epoch",
    "training_time",
]


def find_metrics(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(output_dir.glob("detection/*/metrics.json")):
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"No detection metrics found under {output_dir / 'detection'}")
    return rows


def write_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df[TABLE_COLUMNS].to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(df[TABLE_COLUMNS].to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    markdown = ["| " + " | ".join(TABLE_COLUMNS) + " |"]
    markdown.append("| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |")
    for _, row in df[TABLE_COLUMNS].iterrows():
        markdown.append("| " + " | ".join("" if pd.isna(row[col]) else str(row[col]) for col in TABLE_COLUMNS) + " |")
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    if metric not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    for initialization, group in df.groupby("initialization"):
        group = group.sort_values("labeled_fraction")
        plt.plot(group["labeled_fraction"] * 100, group[metric], marker="o", label=initialization)
    plt.xlabel("Labeled fraction (%)")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric.replace('-', '_')}_vs_fraction.png", dpi=160)
    plt.close()


def plot_training_time(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    labels = df["experiment"].astype(str)
    plt.bar(labels, df["training_time"].fillna(0))
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Training time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "training_time_by_experiment.png", dpi=160)
    plt.close()


def plot_ssl_loss(output_dir: Path) -> None:
    metrics_path = output_dir / "ssl" / "metrics.json"
    if not metrics_path.exists():
        return
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not history:
        return
    df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"] + 1, df["loss"], marker="o")
    plt.xlabel("SSL epoch")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ssl_training_loss.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLOv8 SSL experiment results.")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = pd.DataFrame(find_metrics(output_dir))
    for column in TABLE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    write_comparison(df, output_dir)
    for metric in ("mAP50", "mAP50-95", "precision", "recall"):
        plot_metric(df, metric, output_dir)
    plot_training_time(df, output_dir)
    plot_ssl_loss(output_dir)


if __name__ == "__main__":
    main()
