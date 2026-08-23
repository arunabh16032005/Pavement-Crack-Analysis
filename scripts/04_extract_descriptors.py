"""
Script 04: Extract Structural Descriptors from Segmentation Masks.

Runs the structural descriptor extractor on predicted masks (from trained
segmentation model) and on Mendeley Pavement images (for severity training).

Usage:
    python scripts/04_extract_descriptors.py
    python scripts/04_extract_descriptors.py --checkpoint checkpoints/best_segmentation.pth
    python scripts/04_extract_descriptors.py --from-gt  # Use ground truth masks instead
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import get_config
from src.features.structural import StructuralDescriptorExtractor


def extract_from_masks(
    mask_dir: Path,
    extractor: StructuralDescriptorExtractor,
    desc: str = "Extracting",
) -> tuple:
    """Extract features from all masks in a directory."""
    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    mask_files = sorted(f for f in mask_dir.iterdir() if f.suffix.lower() in img_extensions)

    all_features = []
    all_names = []
    feature_names = None

    for mask_path in tqdm(mask_files, desc=desc):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        feature_vector, feature_dict = extractor.extract_all(mask)
        all_features.append(feature_vector)
        all_names.append(mask_path.stem)

        if feature_names is None:
            feature_names = list(feature_dict.keys())

    features_array = np.stack(all_features) if all_features else np.array([])
    return features_array, all_names, feature_names


def extract_from_model(
    model_path: str,
    image_dir: Path,
    extractor: StructuralDescriptorExtractor,
    image_size: tuple = (512, 512),
    device: str = "cuda",
    desc: str = "Predicting + Extracting",
) -> tuple:
    """Run segmentation model on images, then extract features from predictions."""
    from src.data.transforms import get_val_augmentation
    from src.models.unet import ResNetUNet

    # Load model
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ResNetUNet()
    checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dev)
    model.eval()

    transform = get_val_augmentation(image_size=image_size)

    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in img_extensions)

    all_features = []
    all_names = []
    feature_names = None

    for img_path in tqdm(image_files, desc=desc):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image_resized = cv2.resize(image, (image_size[1], image_size[0]))

        # Transform
        augmented = transform(image=image_resized)
        tensor = augmented["image"].unsqueeze(0).to(dev)

        # Predict
        with torch.no_grad():
            logits = model(tensor)
            pred = (torch.sigmoid(logits) > 0.5).float()

        # Convert to mask
        mask = pred[0, 0].cpu().numpy().astype(np.uint8)

        # Extract features
        feature_vector, feature_dict = extractor.extract_all(mask)
        all_features.append(feature_vector)
        all_names.append(img_path.stem)

        if feature_names is None:
            feature_names = list(feature_dict.keys())

    features_array = np.stack(all_features) if all_features else np.array([])
    return features_array, all_names, feature_names


def main():
    parser = argparse.ArgumentParser(description="Extract structural descriptors")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to segmentation model checkpoint")
    parser.add_argument("--from-gt", action="store_true",
                        help="Extract from ground truth masks (no model needed)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()
    extractor = StructuralDescriptorExtractor()
    features_dir = config.data.processed_dir.parent.parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  STRUCTURAL DESCRIPTOR EXTRACTION")
    print("=" * 60)

    # Determine checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_ckpt = config.training.checkpoint_dir / "best_segmentation.pth"
        if default_ckpt.exists():
            checkpoint_path = str(default_ckpt)

    # ── Extract from segmentation training data (GT masks) ──
    if args.from_gt or checkpoint_path is None:
        print("\n[1/3] Extracting from ground truth training masks...")
        train_mask_dir = config.data.processed_dir / "train" / "masks"
        if train_mask_dir.exists():
            features, names, feat_names = extract_from_masks(
                train_mask_dir, extractor, desc="Train GT masks"
            )
            np.savez(
                str(features_dir / "structural_train_gt.npz"),
                features=features, names=names, feature_names=feat_names,
            )
            print(f"  Saved {len(features)} feature vectors to structural_train_gt.npz")

        print("\n[2/3] Extracting from ground truth validation masks...")
        val_mask_dir = config.data.processed_dir / "val" / "masks"
        if val_mask_dir.exists():
            features, names, feat_names = extract_from_masks(
                val_mask_dir, extractor, desc="Val GT masks"
            )
            np.savez(
                str(features_dir / "structural_val_gt.npz"),
                features=features, names=names, feature_names=feat_names,
            )
            print(f"  Saved {len(features)} feature vectors to structural_val_gt.npz")
    else:
        print(f"\n  Using model: {checkpoint_path}")

    # ── Extract from Mendeley Pavement (for severity training) ──
    print("\n[3/3] Extracting from Mendeley Pavement severity data...")
    severity_dir = config.data.severity_dir

    for split in ["train", "val"]:
        split_dir = severity_dir / split
        if not split_dir.exists():
            print(f"  ⚠ {split_dir} not found, skipping")
            continue

        all_features = []
        all_labels = []
        all_names = []
        feature_names = None

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            try:
                label = int(class_dir.name.split("_")[0])
            except ValueError:
                continue

            print(f"\n  Processing severity/{split}/{class_dir.name}...")

            if checkpoint_path:
                features, names, feat_names = extract_from_model(
                    checkpoint_path, class_dir, extractor,
                    device=args.device,
                    desc=f"  {class_dir.name}",
                )
            else:
                # If no model, use image-level features (less accurate)
                # Just extract from whatever masks exist
                features, names, feat_names = extract_from_masks(
                    class_dir, extractor, desc=f"  {class_dir.name}",
                )

            if feature_names is None and feat_names:
                feature_names = feat_names

            labels = np.full(len(features), label, dtype=np.int64)
            all_features.append(features)
            all_labels.append(labels)
            all_names.extend(names)

        if all_features:
            combined_features = np.concatenate(all_features, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)

            np.savez(
                str(features_dir / f"severity_{split}.npz"),
                features=combined_features,
                labels=combined_labels,
                names=all_names,
                feature_names=feature_names,
            )
            print(f"\n  Saved {len(combined_features)} severity feature vectors ({split})")
            print(f"  Label distribution: {dict(zip(*np.unique(combined_labels, return_counts=True)))}")

    print(f"\n✓ All features saved to {features_dir}/")


if __name__ == "__main__":
    main()
