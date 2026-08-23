"""
Script 05: Train Severity Classifier.

Trains the MLP severity classifier on structural features extracted from
Mendeley Pavement images (real severity labels).

Pipeline:
  1. Load pre-extracted structural features (from script 04)
  2. Normalize features
  3. Train SeverityClassifier MLP
  4. Evaluate: accuracy, macro F1, per-class F1, confusion matrix

Usage:
    python scripts/05_train_severity.py
    python scripts/05_train_severity.py --epochs 100 --lr 0.001
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import get_config
from src.models.severity_mlp import (
    FeatureNormalizer,
    SeverityClassifier,
    generate_pseudo_severity_labels,
)


def train_severity(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
):
    """Train severity classifier with early stopping."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    class_weights = class_weights.to(dev)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_f1 = 0.0
    epochs_no_improve = 0

    print(f"\n{'=' * 50}")
    print(f"  SEVERITY CLASSIFIER TRAINING")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  Device: {dev}")
    print(f"{'=' * 50}\n")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        train_loss = running_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(dev)
                labels = labels.to(dev)

                logits = model(features)
                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / max(val_total, 1)

        # Macro F1
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        per_class_f1 = []
        for c in range(4):
            tp = np.sum((all_preds == c) & (all_labels == c))
            fp = np.sum((all_preds == c) & (all_labels != c))
            fn = np.sum((all_preds != c) & (all_labels == c))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            per_class_f1.append(f1)
        macro_f1 = np.mean(per_class_f1)

        scheduler.step(macro_f1)

        # Check improvement
        improved = ""
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_no_improve = 0
            # Save best model
            ckpt_path = Path(checkpoint_dir) / "severity_best.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "val_acc": val_acc,
                "input_dim": model.input_dim,
            }, str(ckpt_path))
            improved = " ★"
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {epoch + 1:3d}/{epochs} │ "
            f"Loss: {train_loss:.4f} │ "
            f"Train Acc: {train_acc:.4f} │ "
            f"Val Acc: {val_acc:.4f} │ "
            f"Macro F1: {macro_f1:.4f}{improved}"
        )

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    print(f"\n  Best Macro F1: {best_f1:.4f}")
    return best_f1


def evaluate_severity(model, val_loader, device="cuda"):
    """Print detailed evaluation: confusion matrix, per-class metrics."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(dev)
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = ["Low", "Moderate", "Severe", "Critical"]

    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n  Overall Accuracy: {accuracy:.4f}")

    # Per-class metrics
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-' * 52}")

    for c in range(4):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))
        support = np.sum(all_labels == c)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"  {class_names[c]:<12} {precision:>10.4f} {recall:>10.4f} "
              f"{f1:>10.4f} {support:>10d}")

    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>12}", end="")
    for name in class_names:
        print(f" {name[:8]:>8}", end="")
    print()

    for i, name in enumerate(class_names):
        print(f"  {name:<12}", end="")
        for j in range(4):
            count = np.sum((all_labels == i) & (all_preds == j))
            print(f" {count:>8d}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Train severity classifier")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-pseudo-labels", action="store_true",
                        help="Use pseudo-labels instead of Mendeley labels")
    args = parser.parse_args()

    config = get_config()
    features_dir = project_root / "features"

    epochs = args.epochs or config.severity.epochs
    lr = args.lr or config.severity.learning_rate
    batch_size = args.batch_size or config.severity.batch_size

    print("=" * 60)
    print("  SEVERITY CLASSIFIER TRAINING")
    print("=" * 60)

    # ── Load features ──
    print("\n[1/3] Loading features...")

    # Try severity features first (from Mendeley Pavement)
    train_file = features_dir / "severity_train.npz"
    val_file = features_dir / "severity_val.npz"

    if train_file.exists() and not args.use_pseudo_labels:
        print("  Using Mendeley Pavement severity features (real labels)")
        train_data = np.load(str(train_file))
        val_data = np.load(str(val_file))
        train_features = train_data["features"]
        train_labels = train_data["labels"]
        val_features = val_data["features"]
        val_labels = val_data["labels"]
        feature_names = list(train_data.get("feature_names", []))
    else:
        print("  ⚠ Mendeley features not found. Using GT mask features with pseudo-labels.")
        gt_train = features_dir / "structural_train_gt.npz"
        gt_val = features_dir / "structural_val_gt.npz"

        if not gt_train.exists():
            print("  ✗ No features found. Run script 04 first.")
            return

        train_data = np.load(str(gt_train))
        val_data = np.load(str(gt_val))
        train_features = train_data["features"]
        val_features = val_data["features"]
        feature_names = list(train_data.get("feature_names", []))

        # Generate pseudo-labels
        train_labels = generate_pseudo_severity_labels(train_features, feature_names)
        val_labels = generate_pseudo_severity_labels(val_features, feature_names)

    print(f"  Train: {len(train_features)} samples, {train_features.shape[1]} features")
    print(f"  Val:   {len(val_features)} samples")
    print(f"  Train label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"  Val label distribution:   {dict(zip(*np.unique(val_labels, return_counts=True)))}")

    # ── Normalize features ──
    print("\n[2/3] Normalizing features...")
    normalizer = FeatureNormalizer()
    train_features_norm = normalizer.fit_transform(train_features)
    val_features_norm = normalizer.transform(val_features)
    normalizer.save(str(features_dir / "feature_normalizer.npz"))

    # Handle NaN/Inf
    train_features_norm = np.nan_to_num(train_features_norm, nan=0, posinf=0, neginf=0)
    val_features_norm = np.nan_to_num(val_features_norm, nan=0, posinf=0, neginf=0)

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_features_norm.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.int64)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_features_norm.astype(np.float32)),
        torch.from_numpy(val_labels.astype(np.int64)),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class weights (inverse frequency)
    class_counts = np.bincount(train_labels, minlength=4).astype(np.float32)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 4
    class_weights = torch.from_numpy(class_weights)
    print(f"  Class weights: {class_weights.numpy()}")

    # ── Train ──
    print("\n[3/3] Training...")
    input_dim = train_features.shape[1]
    model = SeverityClassifier(
        input_dim=input_dim,
        hidden_dims=config.severity.hidden_dims,
        num_classes=config.severity.num_classes,
        dropout_rate=config.severity.dropout_rate,
    )

    best_f1 = train_severity(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=epochs,
        lr=lr,
        patience=config.severity.patience,
        device=args.device,
        checkpoint_dir=str(config.training.checkpoint_dir),
    )

    # ── Final Evaluation ──
    # Load best model
    ckpt_path = config.training.checkpoint_dir / "severity_best.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("\n" + "=" * 60)
    print("  SEVERITY EVALUATION (Best Model)")
    print("=" * 60)
    evaluate_severity(model, val_loader, device=args.device)

    # Check 70% target
    target_met = best_f1 >= 0.70
    print(f"\n  Target Macro F1 ≥ 0.70: {'✓ MET' if target_met else '✗ NOT MET'} "
          f"(actual: {best_f1:.4f})")


if __name__ == "__main__":
    main()
