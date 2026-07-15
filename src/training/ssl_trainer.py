from __future__ import annotations

import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets import ImageFolderDataset, SimCLRTransform
from src.losses import NTXentLoss
from src.models import SimCLRModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.seed import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


class SSLTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        seed_everything(int(config["seed"]))
        self.device = resolve_device(str(config["device"]))
        self.output_dir = Path(config["output_dir"])
        self.ssl_dir = self.output_dir / "ssl"
        self.checkpoint_dir = self.ssl_dir / "checkpoints"
        self.log_dir = self.ssl_dir / "logs"
        self.metrics_path = self.ssl_dir / "metrics.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        transform = SimCLRTransform.from_config(config)
        self.dataset = ImageFolderDataset(
            config["ssl"]["image_dir"],
            transform=transform,
            corrupt_image_policy=config["ssl"].get("corrupt_image_policy", "error"),
        )
        if len(self.dataset) < 2:
            raise ValueError("SSL dataset must contain at least 2 images.")

        self.loader = DataLoader(
            self.dataset,
            batch_size=int(config["ssl"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["ssl"].get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        if len(self.loader) == 0:
            raise ValueError("SSL DataLoader has zero batches. Reduce ssl.batch_size or add images.")

        self.model = SimCLRModel(
            variant=str(config["model"]["variant"]),
            pretrained=bool(config["model"].get("pretrained", True)),
            embedding_dim=int(config["model"]["embedding_dim"]),
            projection_hidden_dim=int(config["model"].get("projection_hidden_dim", 512)),
        ).to(self.device)
        self.criterion = NTXentLoss(float(config["ssl"]["temperature"]))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["ssl"]["learning_rate"]),
            weight_decay=float(config["ssl"]["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(config["ssl"]["epochs"])),
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if config.get("logging", {}).get("tensorboard", True) else None

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        LOGGER.info("Device: %s", self.device)
        LOGGER.info("SSL images: %d", len(self.dataset))
        LOGGER.info("SSL batch size: %d", int(config["ssl"]["batch_size"]))
        LOGGER.info("Model parameters: total=%d trainable=%d", total, trainable)
        LOGGER.info("Learning rate: %.6g", float(config["ssl"]["learning_rate"]))

        self.start_epoch = 0
        self.best_loss = math.inf
        resume_path = config["ssl"].get("resume")
        if resume_path:
            self.resume(resume_path)

    def resume(self, checkpoint_path: str | Path) -> None:
        payload = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.start_epoch = int(payload.get("epoch", 0)) + 1
        self.best_loss = float(payload.get("best_loss", math.inf))
        LOGGER.info("Resumed SSL training from %s at epoch %d.", checkpoint_path, self.start_epoch)

    def _checkpoint_payload(self, epoch: int, epoch_loss: float) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "epoch_loss": epoch_loss,
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def _save_backbone(self, path: Path, epoch: int, epoch_loss: float) -> None:
        save_checkpoint(
            path,
            {
                "epoch": epoch,
                "best_loss": epoch_loss,
                "state_dict": self.model.backbone_state_dict_for_transfer(),
                "model_variant": self.config["model"]["variant"],
            },
        )
        LOGGER.info("Saved transferable backbone checkpoint: %s", path)

    def train(self) -> dict[str, Any]:
        epochs = int(self.config["ssl"]["epochs"])
        gradient_clip = float(self.config["ssl"].get("gradient_clip_norm", 0) or 0)
        history: list[dict[str, Any]] = []
        started_at = time.time()

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(self.loader, desc=f"SSL epoch {epoch + 1}/{epochs}", leave=False)
            for view1, view2 in progress:
                view1 = view1.to(self.device, non_blocking=True)
                view2 = view2.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                if self.device.type == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    autocast_context = torch.amp.autocast(device_type="cuda", enabled=True)
                elif self.device.type == "cuda":
                    autocast_context = torch.cuda.amp.autocast(enabled=True)
                else:
                    autocast_context = nullcontext()

                with autocast_context:
                    z1 = self.model(view1)
                    z2 = self.model(view2)
                    loss = self.criterion(z1, z2)

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"NaN or Inf SSL loss detected at epoch {epoch + 1}.")

                self.scaler.scale(loss).backward()
                if gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += float(loss.detach().cpu())
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

            self.scheduler.step()
            epoch_loss = total_loss / len(self.loader)
            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                    "seconds_elapsed": time.time() - started_at,
                }
            )
            LOGGER.info("SSL epoch %d finished: loss=%.6f", epoch + 1, epoch_loss)
            if self.writer:
                self.writer.add_scalar("ssl/loss", epoch_loss, epoch)
                self.writer.add_scalar("ssl/lr", self.scheduler.get_last_lr()[0], epoch)

            save_checkpoint(self.checkpoint_dir / "last.pt", self._checkpoint_payload(epoch, epoch_loss))
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_checkpoint(self.checkpoint_dir / "best.pt", self._checkpoint_payload(epoch, epoch_loss))
                self._save_backbone(self.checkpoint_dir / "backbone_best.pt", epoch, epoch_loss)

            self.metrics_path.write_text(
                json.dumps({"best_loss": self.best_loss, "history": history}, indent=2),
                encoding="utf-8",
            )

        if self.writer:
            self.writer.close()
        return {"best_loss": self.best_loss, "history": history}
