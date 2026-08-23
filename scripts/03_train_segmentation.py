"""
Script 03: Train Segmentation Model.

Trains the ResNet34 U-Net on CrackVision12K for crack segmentation.

Usage:
    python scripts/03_train_segmentation.py
    python scripts/03_train_segmentation.py --epochs 5 --batch_size 4  # Quick test
    python scripts/03_train_segmentation.py --resume checkpoints/best_segmentation.pth
"""

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import get_config
from src.data.dataset import get_segmentation_dataloaders
from src.data.transforms import get_train_augmentation, get_val_augmentation
from src.models.loss import CombinedSegmentationLoss
from src.models.unet import ResNetUNet
from src.training.trainer import SegmentationTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Faster for fixed input sizes


def main():
    parser = argparse.ArgumentParser(description="Train crack segmentation model")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder name (resnet34/resnet50)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    config = get_config()

    # Apply overrides
    epochs = args.epochs or config.training.epochs
    batch_size = args.batch_size or config.training.batch_size
    lr = args.lr or config.training.learning_rate
    encoder = args.encoder or config.model.encoder_name
    use_amp = not args.no_amp and config.training.use_amp

    # Set seed
    set_seed(config.training.seed)

    print(f"\n{'=' * 60}")
    print(f"  CRACK SEGMENTATION TRAINING")
    print(f"{'=' * 60}")
    print(f"  Encoder:     {encoder}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    print(f"  AMP:         {use_amp}")
    print(f"  Image size:  {config.preprocess.image_size}")
    print(f"{'=' * 60}\n")

    # ── Data ──
    print("[1/4] Loading datasets...")
    train_transform = get_train_augmentation(
        image_size=config.preprocess.image_size,
        imagenet_mean=config.preprocess.imagenet_mean,
        imagenet_std=config.preprocess.imagenet_std,
    )
    val_transform = get_val_augmentation(
        image_size=config.preprocess.image_size,
        imagenet_mean=config.preprocess.imagenet_mean,
        imagenet_std=config.preprocess.imagenet_std,
    )

    dataloaders = get_segmentation_dataloaders(
        processed_dir=str(config.data.processed_dir),
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    print(f"  Train: {len(dataloaders['train'].dataset)} images")
    print(f"  Val:   {len(dataloaders['val'].dataset)} images")
    for key in dataloaders:
        if key.startswith("test"):
            print(f"  {key}: {len(dataloaders[key].dataset)} images")

    # ── Model ──
    print("\n[2/4] Building model...")
    model = ResNetUNet(
        encoder_name=encoder,
        pretrained=config.model.encoder_pretrained,
        num_classes=config.model.num_classes,
        decoder_channels=config.model.decoder_channels,
    )

    param_info = model.count_parameters()
    print(f"  Total parameters: {param_info['total_millions']:.1f}M")
    print(f"  Encoder: {param_info['encoder'] / 1e6:.1f}M")
    print(f"  Decoder: {param_info['decoder'] / 1e6:.1f}M")

    # ── Loss ──
    print("\n[3/4] Setting up training...")
    criterion = CombinedSegmentationLoss(
        bce_weight=config.training.bce_weight,
        dice_weight=config.training.dice_weight,
        boundary_weight=config.training.boundary_weight,
        boundary_pixel_weight=3.0,
        dilation_kernel_size=config.training.boundary_dilation_kernel,
    )

    # ── Trainer ──
    trainer = SegmentationTrainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        criterion=criterion,
        learning_rate=lr,
        weight_decay=config.training.weight_decay,
        lr_min=config.training.lr_min,
        epochs=epochs,
        grad_accumulation_steps=config.training.grad_accumulation_steps,
        use_amp=use_amp,
        patience=config.training.patience,
        checkpoint_dir=str(config.training.checkpoint_dir),
        use_wandb=args.wandb or config.training.use_wandb,
        wandb_project=config.training.wandb_project,
        wandb_run_name=config.training.wandb_run_name,
        device=args.device,
    )

    # Resume from checkpoint
    if args.resume:
        print(f"  Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # ── Train ──
    print("\n[4/4] Training...")
    history = trainer.fit()

    # ── Post-training evaluation ──
    print("\nRunning post-training evaluation on all available datasets...")
    from src.training.evaluate import cross_dataset_evaluation

    # Load best model
    best_path = config.training.checkpoint_dir / "best_segmentation.pth"
    if best_path.exists():
        checkpoint = torch.load(str(best_path), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    eval_loaders = {k: v for k, v in dataloaders.items() if k != "train"}
    results = cross_dataset_evaluation(
        model=model,
        dataloaders=eval_loaders,
        device=str(trainer.device),
        save_dir=str(config.training.results_dir),
    )

    # Check 70% target
    val_miou = results.get("val", {}).get("miou_mean", 0)
    target_met = val_miou >= 0.70
    print(f"\n  Target mIoU ≥ 0.70: {'✓ MET' if target_met else '✗ NOT MET'} "
          f"(actual: {val_miou:.4f})")


if __name__ == "__main__":
    main()
