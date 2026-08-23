"""
Script 06: Full Pipeline Evaluation.

Runs the complete evaluation pipeline:
  1. Load trained segmentation model
  2. Evaluate on all test datasets (CrackForest, DeepCrack)
  3. Load trained severity classifier
  4. Generate comprehensive evaluation report with visualizations

Usage:
    python scripts/06_evaluate_full.py
    python scripts/06_evaluate_full.py --seg-checkpoint checkpoints/best_segmentation.pth
    python scripts/06_evaluate_full.py --visualize  # Generate prediction grids
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import get_config
from src.data.dataset import get_segmentation_dataloaders
from src.data.transforms import denormalize_image, get_val_augmentation
from src.models.unet import ResNetUNet
from src.training.evaluate import cross_dataset_evaluation
from src.utils.visualization import (
    create_prediction_grid,
    generate_report,
    overlay_mask,
    plot_training_curves,
)


def load_segmentation_model(checkpoint_path: str, config, device: str = "cuda"):
    """Load trained segmentation model from checkpoint."""
    model = ResNetUNet(
        encoder_name=config.model.encoder_name,
        pretrained=False,
        num_classes=config.model.num_classes,
        decoder_channels=config.model.decoder_channels,
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dev)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    best_miou = checkpoint.get("best_miou", checkpoint.get("metrics", {}).get("miou", "?"))
    print(f"  Loaded model from epoch {epoch}, best mIoU: {best_miou}")

    return model, checkpoint


def generate_visual_samples(model, data_loader, config, output_dir, n_samples=20, device="cuda"):
    """Generate visual prediction samples for manual inspection."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    images_list = []
    gt_list = []
    pred_list = []

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(dev)

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            for i in range(images.shape[0]):
                if len(images_list) >= n_samples:
                    break

                # Denormalize image for visualization
                img = images[i].cpu().numpy()
                img = denormalize_image(img,
                                        mean=config.preprocess.imagenet_mean,
                                        std=config.preprocess.imagenet_std)

                gt = masks[i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()

                images_list.append(img)
                gt_list.append(gt)
                pred_list.append(pred)

            if len(images_list) >= n_samples:
                break

    if images_list:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison grid
        create_prediction_grid(
            images_list, gt_list, pred_list,
            n=min(8, len(images_list)),
            save_path=str(output_dir / "prediction_grid.png"),
        )
        import matplotlib.pyplot as plt
        plt.close()

        # Save individual overlays
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)
        for i in range(min(n_samples, len(images_list))):
            overlay = overlay_mask(images_list[i], pred_list[i], alpha=0.5, color=(255, 0, 0))
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_dir / f"overlay_{i:03d}.png"), overlay_bgr)

        print(f"  Saved {min(n_samples, len(images_list))} visual samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation")
    parser.add_argument("--seg-checkpoint", type=str, default=None,
                        help="Segmentation model checkpoint")
    parser.add_argument("--sev-checkpoint", type=str, default=None,
                        help="Severity model checkpoint")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visual prediction samples")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of visual samples")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()
    results_dir = config.training.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FULL PIPELINE EVALUATION")
    print("=" * 60)

    # ── Find checkpoints ──
    seg_ckpt = args.seg_checkpoint or str(config.training.checkpoint_dir / "best_segmentation.pth")
    sev_ckpt = args.sev_checkpoint or str(config.training.checkpoint_dir / "severity_best.pth")

    if not Path(seg_ckpt).exists():
        print(f"  ✗ Segmentation checkpoint not found: {seg_ckpt}")
        print(f"  Run scripts/03_train_segmentation.py first.")
        return

    # ── Load Model ──
    print("\n[1/5] Loading segmentation model...")
    model, checkpoint = load_segmentation_model(seg_ckpt, config, args.device)

    # ── Load Data ──
    print("\n[2/5] Loading datasets...")
    val_transform = get_val_augmentation(
        image_size=config.preprocess.image_size,
        imagenet_mean=config.preprocess.imagenet_mean,
        imagenet_std=config.preprocess.imagenet_std,
    )

    dataloaders = get_segmentation_dataloaders(
        processed_dir=str(config.data.processed_dir),
        train_transform=val_transform,  # No augmentation for evaluation
        val_transform=val_transform,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    for name, loader in dataloaders.items():
        if name != "train":
            print(f"  {name}: {len(loader.dataset)} images")

    # ── Segmentation Evaluation ──
    print("\n[3/5] Running segmentation evaluation...")
    eval_loaders = {k: v for k, v in dataloaders.items() if k != "train"}
    seg_results = cross_dataset_evaluation(
        model=model,
        dataloaders=eval_loaders,
        device=args.device,
        save_dir=str(results_dir),
    )

    # ── Visual Samples ──
    if args.visualize:
        print("\n[4/5] Generating visual samples...")
        for name, loader in eval_loaders.items():
            generate_visual_samples(
                model, loader, config,
                output_dir=results_dir / name,
                n_samples=args.n_samples,
                device=args.device,
            )
    else:
        print("\n[4/5] Skipping visualization (use --visualize to enable)")

    # ── Severity Evaluation ──
    severity_results = None
    if Path(sev_ckpt).exists():
        print("\n[5/5] Loading severity classifier...")
        from src.models.severity_mlp import SeverityClassifier
        sev_checkpoint = torch.load(sev_ckpt, map_location="cpu", weights_only=False)
        input_dim = sev_checkpoint.get("input_dim", 35)

        sev_model = SeverityClassifier(input_dim=input_dim)
        sev_model.load_state_dict(sev_checkpoint["model_state_dict"])

        severity_results = {
            "best_f1": sev_checkpoint.get("best_f1", "N/A"),
            "val_acc": sev_checkpoint.get("val_acc", "N/A"),
        }
        print(f"  Severity Best F1:  {severity_results['best_f1']}")
        print(f"  Severity Val Acc:  {severity_results['val_acc']}")
    else:
        print("\n[5/5] Severity checkpoint not found, skipping severity evaluation")

    # ── Generate Report ──
    print("\n  Generating report...")

    # Load training history if available
    history = None
    history_path = config.training.checkpoint_dir / "training_history.npy"
    if history_path.exists():
        history = np.load(str(history_path), allow_pickle=True).item()
        plot_training_curves(history, save_path=str(results_dir / "training_curves.png"))
        import matplotlib.pyplot as plt
        plt.close()

    generate_report(
        seg_results=seg_results,
        severity_results=severity_results,
        history=history,
        output_dir=str(results_dir),
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    for name, results in seg_results.items():
        miou = results.get("miou_mean", 0)
        f1 = results.get("f1_mean", 0)
        target = "✓" if miou >= 0.70 else "✗"
        print(f"  [{target}] {name:<20} mIoU: {miou:.4f}  F1: {f1:.4f}")

    if severity_results:
        sev_f1 = severity_results.get("best_f1", 0)
        target = "✓" if (isinstance(sev_f1, (int, float)) and sev_f1 >= 0.70) else "?"
        print(f"  [{target}] {'severity':<20} Macro F1: {sev_f1}")

    print(f"\n  Results saved to: {results_dir}/")
    print(f"  Report: {results_dir / 'evaluation_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
