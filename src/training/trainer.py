"""
Phase 8 (Simplified) — Segmentation Training Loop.

Handles:
  - AdamW optimizer with cosine annealing LR scheduler
  - FP16 mixed precision training (saves ~40% VRAM)
  - Gradient accumulation (effective batch = batch_size × accum_steps)
  - Early stopping on validation mIoU
  - Best model checkpointing
  - Optional W&B logging
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """
    Training loop for crack segmentation model.

    Features:
      - Mixed precision (FP16) training via torch.amp
      - Gradient accumulation for larger effective batch sizes
      - Cosine annealing LR schedule
      - Early stopping on validation mIoU
      - Best model checkpoint saving
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_min: float = 1e-6,
        epochs: int = 100,
        grad_accumulation_steps: int = 4,
        use_amp: bool = True,
        patience: int = 15,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "crack-segmentation",
        wandb_run_name: str = "resnet34-unet",
        device: Optional[str] = None,
    ):
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr_min
        )

        # Mixed precision
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Training config
        self.epochs = epochs
        self.grad_accumulation_steps = grad_accumulation_steps
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.best_miou = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_bce": [], "train_dice": [], "train_boundary": [],
            "val_miou": [], "val_f1": [], "val_pixel_acc": [],
            "lr": [],
        }

        # W&B
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=wandb_run_name)
                wandb.watch(model, log_freq=100)
            except ImportError:
                logger.warning("wandb not installed. Disabling W&B logging.")
                self.use_wandb = False

    def _compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute segmentation metrics from logits and targets."""
        with torch.no_grad():
            probs = torch.sigmoid(pred)
            binary_pred = (probs > 0.5).float()

            # Flatten
            pred_flat = binary_pred.view(-1)
            target_flat = target.view(-1)

            tp = (pred_flat * target_flat).sum().item()
            fp = (pred_flat * (1 - target_flat)).sum().item()
            fn = ((1 - pred_flat) * target_flat).sum().item()
            tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

            # Pixel accuracy
            pixel_acc = (tp + tn) / max(tp + tn + fp + fn, 1)

            # IoU (per crack class)
            iou = tp / max(tp + fp + fn, 1)

            # Mean IoU (background IoU + crack IoU) / 2
            bg_iou = tn / max(tn + fp + fn, 1)
            miou = (iou + bg_iou) / 2.0

            # F1 / Dice
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            return {
                "pixel_acc": pixel_acc,
                "iou": iou,
                "miou": miou,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()

        running_loss = 0.0
        running_bce = 0.0
        running_dice = 0.0
        running_boundary = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [Train]",
            leave=False,
        )

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # Mixed precision forward
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict["total"] / self.grad_accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Step optimizer every grad_accumulation_steps
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss_dict["total"].item()
            running_bce += loss_dict["bce"].item()
            running_dice += loss_dict["dice"].item()
            running_boundary += loss_dict["boundary"].item()
            num_batches += 1

            pbar.set_postfix(
                loss=f"{running_loss / num_batches:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        # Handle remaining gradients
        if num_batches % self.grad_accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {
            "loss": running_loss / max(num_batches, 1),
            "bce": running_bce / max(num_batches, 1),
            "dice": running_dice / max(num_batches, 1),
            "boundary": running_boundary / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()

        running_loss = 0.0
        all_metrics = {"pixel_acc": [], "iou": [], "miou": [], "f1": [],
                       "precision": [], "recall": []}

        pbar = tqdm(self.val_loader, desc="Validating", leave=False)

        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss_dict = self.criterion(logits, masks)

            running_loss += loss_dict["total"].item()

            # Compute metrics per batch
            metrics = self._compute_metrics(logits, masks)
            for key, value in metrics.items():
                all_metrics[key].append(value)

        num_batches = len(self.val_loader)
        avg_loss = running_loss / max(num_batches, 1)
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        avg_metrics["loss"] = avg_loss

        return avg_metrics

    def save_checkpoint(self, epoch: int, metrics: dict, filename: str = "best_segmentation.pth"):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_miou": self.best_miou,
            "metrics": metrics,
            "history": self.history,
        }, str(path))
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_miou = checkpoint.get("best_miou", 0.0)
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Loaded checkpoint from {path} (best mIoU: {self.best_miou:.4f})")
        return checkpoint.get("epoch", 0)

    def fit(self) -> dict:
        """
        Full training loop with early stopping.

        Returns:
            Training history dict.
        """
        print(f"\n{'=' * 60}")
        print(f"  TRAINING: ResNet34 U-Net")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.train_loader.batch_size} × {self.grad_accumulation_steps} "
              f"= {self.train_loader.batch_size * self.grad_accumulation_steps} effective")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Early stopping patience: {self.patience}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update LR
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_bce"].append(train_metrics["bce"])
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["train_boundary"].append(train_metrics["boundary"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_miou"].append(val_metrics["miou"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_pixel_acc"].append(val_metrics["pixel_acc"])
            self.history["lr"].append(current_lr)

            # W&B logging
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/bce": train_metrics["bce"],
                        "train/dice": train_metrics["dice"],
                        "val/loss": val_metrics["loss"],
                        "val/miou": val_metrics["miou"],
                        "val/f1": val_metrics["f1"],
                        "val/pixel_acc": val_metrics["pixel_acc"],
                        "lr": current_lr,
                    })
                except Exception:
                    pass

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            improved = ""
            if val_metrics["miou"] > self.best_miou:
                self.best_miou = val_metrics["miou"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_metrics)
                improved = " ★ NEW BEST"
            else:
                self.epochs_without_improvement += 1

            print(
                f"Epoch {epoch + 1:3d}/{self.epochs} │ "
                f"Loss: {train_metrics['loss']:.4f} / {val_metrics['loss']:.4f} │ "
                f"mIoU: {val_metrics['miou']:.4f} │ "
                f"F1: {val_metrics['f1']:.4f} │ "
                f"Acc: {val_metrics['pixel_acc']:.4f} │ "
                f"LR: {current_lr:.2e} │ "
                f"{epoch_time:.1f}s{improved}"
            )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"  Training complete!")
        print(f"  Total time: {total_time / 60:.1f} minutes")
        print(f"  Best mIoU: {self.best_miou:.4f}")
        print(f"  Checkpoint: {self.checkpoint_dir / 'best_segmentation.pth'}")
        print(f"{'=' * 60}\n")

        # Save final history
        history_path = self.checkpoint_dir / "training_history.npy"
        np.save(str(history_path), self.history)

        return self.history
