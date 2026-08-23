"""
Evaluation Metrics and Cross-Dataset Evaluation.

Provides:
  - Pixel-level segmentation metrics (accuracy, IoU, F1, precision, recall)
  - Per-dataset evaluation
  - Cross-dataset generalization analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_segmentation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute segmentation metrics for a single prediction-ground truth pair.

    Args:
        pred: Predicted probabilities or binary mask (H, W), values in [0, 1].
        gt: Ground truth binary mask (H, W), values in {0, 1}.
        threshold: Binarization threshold for predictions.

    Returns:
        Dict with pixel_acc, iou, miou, f1, precision, recall.
    """
    # Binarize prediction
    if pred.max() > 1.0:
        pred = pred / 255.0
    binary_pred = (pred > threshold).astype(np.float32)

    if gt.max() > 1.0:
        gt = gt / 255.0
    gt = (gt > 0.5).astype(np.float32)

    # Confusion matrix elements
    tp = np.sum(binary_pred * gt)
    fp = np.sum(binary_pred * (1 - gt))
    fn = np.sum((1 - binary_pred) * gt)
    tn = np.sum((1 - binary_pred) * (1 - gt))

    total = tp + fp + fn + tn

    # Metrics
    pixel_acc = (tp + tn) / max(total, 1)
    iou_crack = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    miou = (iou_crack + iou_bg) / 2.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "pixel_acc": pixel_acc,
        "iou": iou_crack,
        "miou": miou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


@torch.no_grad()
def evaluate_on_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    dataset_name: str = "test",
    device: str = "cuda",
    save_predictions: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a complete dataset.

    Args:
        model: Trained segmentation model.
        data_loader: DataLoader for the evaluation dataset.
        dataset_name: Name for logging.
        device: Device to run inference on.
        save_predictions: Whether to save predicted masks.
        save_dir: Directory to save predictions.

    Returns:
        Dict with mean metrics and standard deviations.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_metrics = {
        "pixel_acc": [], "iou": [], "miou": [],
        "f1": [], "precision": [], "recall": [],
    }

    if save_predictions and save_dir:
        save_path = Path(save_dir) / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)

    sample_idx = 0

    for images, masks in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            probs = torch.sigmoid(logits)

        # Move to CPU for metric computation
        probs_np = probs.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(probs_np.shape[0]):
            pred = probs_np[i, 0]  # (H, W)
            gt = masks_np[i, 0]    # (H, W)

            metrics = compute_segmentation_metrics(pred, gt)
            for key, value in metrics.items():
                all_metrics[key].append(value)

            # Save prediction
            if save_predictions and save_dir:
                pred_mask = (pred > 0.5).astype(np.uint8) * 255
                cv2.imwrite(
                    str(save_path / f"pred_{sample_idx:05d}.png"),
                    pred_mask,
                )
                sample_idx += 1

    # Compute mean and std
    results = {}
    for key, values in all_metrics.items():
        results[f"{key}_mean"] = np.mean(values)
        results[f"{key}_std"] = np.std(values)
    results["num_samples"] = len(all_metrics["iou"])

    # Print results
    print(f"\n{'─' * 50}")
    print(f"  {dataset_name.upper()} Evaluation Results ({results['num_samples']} samples)")
    print(f"{'─' * 50}")
    print(f"  {'Metric':<15} {'Mean':>10} {'Std':>10}")
    print(f"  {'─' * 35}")
    for metric in ["pixel_acc", "miou", "iou", "f1", "precision", "recall"]:
        mean = results[f"{metric}_mean"]
        std = results[f"{metric}_std"]
        print(f"  {metric:<15} {mean:>10.4f} {std:>10.4f}")
    print(f"{'─' * 50}\n")

    return results


def cross_dataset_evaluation(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: str = "cuda",
    save_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model across multiple datasets and compute domain gap.

    Args:
        model: Trained segmentation model.
        dataloaders: Dict of {dataset_name: DataLoader}.
        device: Device for inference.
        save_dir: Directory to save predictions and report.

    Returns:
        Dict of {dataset_name: metrics}.
    """
    all_results = {}

    for name, loader in dataloaders.items():
        results = evaluate_on_dataset(
            model, loader, dataset_name=name, device=device,
            save_predictions=save_dir is not None, save_dir=save_dir,
        )
        all_results[name] = results

    # Compute domain adaptation gaps
    if "val" in all_results:
        val_miou = all_results["val"]["miou_mean"]
        print(f"\n{'=' * 60}")
        print(f"  CROSS-DATASET COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Dataset':<20} {'mIoU':>10} {'F1':>10} {'Gap vs Val':>12}")
        print(f"  {'─' * 52}")
        for name, results in all_results.items():
            miou = results["miou_mean"]
            f1 = results["f1_mean"]
            gap = miou - val_miou
            gap_str = f"{gap:+.4f}" if name != "val" else "baseline"
            print(f"  {name:<20} {miou:>10.4f} {f1:>10.4f} {gap_str:>12}")
        print(f"{'=' * 60}\n")

    # Save report
    if save_dir:
        report_path = Path(save_dir) / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write("SEGMENTATION EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            for name, results in all_results.items():
                f.write(f"Dataset: {name}\n")
                f.write(f"  Samples: {results['num_samples']}\n")
                for metric in ["pixel_acc", "miou", "iou", "f1", "precision", "recall"]:
                    f.write(f"  {metric}: {results[f'{metric}_mean']:.4f} "
                            f"± {results[f'{metric}_std']:.4f}\n")
                f.write("\n")
        logger.info(f"Report saved to {report_path}")

    return all_results
