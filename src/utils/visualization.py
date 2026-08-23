"""
Visualization Utilities.

Provides:
  - Mask overlay on images
  - Training curve plots
  - Confusion matrix visualization
  - Prediction comparison grids
  - Evaluation report generation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/cloud
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Global style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Overlay predicted mask on original image with color tinting.

    Args:
        image: RGB image (H, W, 3), uint8.
        mask: Binary mask (H, W), values in {0, 1} or {0, 255}.
        alpha: Overlay transparency.
        color: RGB color for crack overlay.

    Returns:
        Overlaid image (H, W, 3), uint8.
    """
    if mask.max() > 1:
        mask = (mask > 127).astype(np.float32)
    else:
        mask = mask.astype(np.float32)

    overlay = image.copy().astype(np.float32)
    color_mask = np.zeros_like(image, dtype=np.float32)
    color_mask[:, :] = color

    # Blend only where mask is active
    mask_3d = np.stack([mask] * 3, axis=-1)
    overlay = overlay * (1 - mask_3d * alpha) + color_mask * mask_3d * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot training loss and validation mIoU over epochs.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'val_miou', 'val_f1', 'lr'.
        save_path: Path to save figure.
        show: Whether to display figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # ── Loss ──
    ax = axes[0, 0]
    ax.plot(epochs, history.get("train_loss", []), label="Train Loss", linewidth=2)
    ax.plot(epochs, history.get("val_loss", []), label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── mIoU ──
    ax = axes[0, 1]
    ax.plot(epochs, history.get("val_miou", []), label="Val mIoU",
            linewidth=2, color="green")
    ax.axhline(y=0.70, color="red", linestyle="--", alpha=0.7, label="Target (0.70)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Validation mIoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── F1 & Pixel Accuracy ──
    ax = axes[1, 0]
    ax.plot(epochs, history.get("val_f1", []), label="Val F1", linewidth=2)
    ax.plot(epochs, history.get("val_pixel_acc", []), label="Val Pixel Acc",
            linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("F1 Score & Pixel Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Learning Rate ──
    ax = axes[1, 1]
    ax.plot(epochs, history.get("lr", []), linewidth=2, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot confusion matrix with percentages and counts.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save figure.
    """
    if class_names is None:
        class_names = ["Low", "Moderate", "Severe", "Critical"]

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Normalize for percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"label": "Proportion"},
    )

    # Add counts as secondary annotation
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j + 0.5, i + 0.75, f"(n={cm[i, j]})",
                    ha="center", va="center", fontsize=8, color="gray")

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()

    return fig


def create_prediction_grid(
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    n: int = 8,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create a comparison grid: original | ground truth | prediction.

    Args:
        images: List of RGB images.
        gt_masks: List of ground truth masks.
        pred_masks: List of predicted masks.
        n: Number of samples to show.
        save_path: Path to save figure.
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        # Original image with GT overlay
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis("off")

        # Ground truth
        gt_overlay = overlay_mask(images[i], gt_masks[i], alpha=0.5, color=(0, 255, 0))
        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis("off")

        # Prediction
        pred_overlay = overlay_mask(images[i], pred_masks[i], alpha=0.5, color=(255, 0, 0))
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title("Prediction" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.suptitle("Segmentation Predictions (Green=GT, Red=Pred)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Prediction grid saved to {save_path}")

    if show:
        plt.show()

    return fig


def generate_report(
    seg_results: Dict[str, dict],
    severity_results: Optional[dict] = None,
    history: Optional[dict] = None,
    output_dir: str = "results",
):
    """
    Generate a comprehensive evaluation report.

    Args:
        seg_results: Segmentation results per dataset.
        severity_results: Severity classification results.
        history: Training history.
        output_dir: Directory to save report and figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training curves ──
    if history:
        plot_training_curves(
            history,
            save_path=str(output_dir / "training_curves.png"),
        )
        plt.close()

    # ── Write text report ──
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Pipeline Evaluation Report\n\n")
        f.write("## Segmentation Results\n\n")
        f.write(f"| Dataset | mIoU | F1 | Pixel Acc | Precision | Recall |\n")
        f.write(f"|---------|------|----|-----------|-----------|---------|\n")

        for name, results in seg_results.items():
            f.write(
                f"| {name} | "
                f"{results.get('miou_mean', 0):.4f} | "
                f"{results.get('f1_mean', 0):.4f} | "
                f"{results.get('pixel_acc_mean', 0):.4f} | "
                f"{results.get('precision_mean', 0):.4f} | "
                f"{results.get('recall_mean', 0):.4f} |\n"
            )

        if severity_results:
            f.write("\n## Severity Classification Results\n\n")
            for key, value in severity_results.items():
                f.write(f"- **{key}**: {value}\n")

        f.write("\n---\n")
        f.write("*Generated by the Crack Detection Pipeline*\n")

    logger.info(f"Report saved to {report_path}")
    print(f"\n  Report saved to {report_path}")
