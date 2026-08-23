import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config import get_config
from src.data.transforms import preprocess_image, preprocess_mask


def validate_pairs(image_dir: Path, mask_dir: Path) -> int:
    """Validate that image-mask pairs match in count and dimensions."""
    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    images = {f.stem: f for f in image_dir.iterdir() if f.suffix.lower() in img_extensions}
    masks = {f.stem: f for f in mask_dir.iterdir() if f.suffix.lower() in img_extensions}

    paired = set(images.keys()) & set(masks.keys())
    only_images = set(images.keys()) - set(masks.keys())
    only_masks = set(masks.keys()) - set(images.keys())

    if only_images:
        print(f"  ⚠ {len(only_images)} images without masks")
    if only_masks:
        print(f"  ⚠ {len(only_masks)} masks without images")

    # Check dimensions of a sample
    errors = 0
    for stem in list(paired)[:10]:
        img = cv2.imread(str(images[stem]))
        mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            errors += 1
            continue
        if img.shape[:2] != mask.shape[:2]:
            print(f"  ⚠ Size mismatch: {stem} img={img.shape[:2]} mask={mask.shape[:2]}")
            errors += 1

        # Check mask is binary
        unique = np.unique(mask)
        if not all(v in [0, 255] for v in unique) and not all(v in [0, 1] for v in unique):
            print(f"  ⚠ Non-binary mask: {stem} unique values={unique[:10]}")

    print(f"  ✓ {len(paired)} valid pairs, {errors} errors in sample check")
    return len(paired)


def preprocess_and_cache(
    src_image_dir: Path,
    src_mask_dir: Path,
    dst_image_dir: Path,
    dst_mask_dir: Path,
    target_size: tuple = (512, 512),
    apply_denoise: bool = True,
    apply_clahe: bool = True,
):

    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted(
        f for f in src_image_dir.iterdir() if f.suffix.lower() in img_extensions
    )

    for img_path in tqdm(image_files, desc=f"Preprocessing {src_image_dir.name}"):
        # Process image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(
            image, target_size=target_size,
            apply_denoise=apply_denoise, apply_clahe=apply_clahe,
        )
        # Save as PNG
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_image_dir / f"{img_path.stem}.png"), processed_bgr)

        # Process corresponding mask
        mask_found = False
        for ext in img_extensions:
            mask_path = src_mask_dir / f"{img_path.stem}{ext}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = preprocess_mask(mask, target_size=target_size)
                    cv2.imwrite(str(dst_mask_dir / f"{img_path.stem}.png"), mask)
                    mask_found = True
                break

        if not mask_found:
            # If masks are in the same directory as images with different naming
            pass


def compute_dataset_stats(image_dir: Path, num_samples: int = 1000) -> dict:
    """Compute mean and std of the dataset for normalization."""
    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in img_extensions
    )

    # Sample subset for speed
    if len(image_files) > num_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(image_files), num_samples, replace=False)
        image_files = [image_files[i] for i in indices]

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    num_pixels = 0

    for img_path in tqdm(image_files, desc="Computing stats"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
        num_pixels += img.shape[0] * img.shape[1]

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)

    return {"mean": mean.tolist(), "std": std.tolist(), "num_images": len(image_files)}


def main():
    parser = argparse.ArgumentParser(description="Preprocess and cache images")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip caching, just validate")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Skip denoising (faster)")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Skip CLAHE enhancement")
    args = parser.parse_args()

    config = get_config()
    processed = config.data.processed_dir

    print("=" * 60)
    print("  IMAGE PREPROCESSING PIPELINE")
    print("=" * 60)

    # Validate existing pairs
    print("\n[1/3] Validating image-mask pairs...")
    for split in ["train", "val"]:
        img_dir = processed / split / "images"
        mask_dir = processed / split / "masks"
        if img_dir.exists():
            print(f"\n  {split}:")
            validate_pairs(img_dir, mask_dir)

    for test_name in ["crackforest", "deepcrack"]:
        img_dir = processed / "test" / test_name / "images"
        mask_dir = processed / "test" / test_name / "masks"
        if img_dir.exists() and any(img_dir.iterdir()):
            print(f"\n  test/{test_name}:")
            validate_pairs(img_dir, mask_dir)

    if args.skip_cache:
        print("\n  Skipping cache step (--skip-cache)")
        return

    # Compute dataset statistics
    print("\n[2/3] Computing dataset statistics...")
    train_img_dir = processed / "train" / "images"
    if train_img_dir.exists():
        stats = compute_dataset_stats(train_img_dir)
        print(f"  Mean: {[f'{v:.4f}' for v in stats['mean']]}")
        print(f"  Std:  {[f'{v:.4f}' for v in stats['std']]}")
        print(f"  (Computed from {stats['num_images']} images)")
        print(f"  Note: Using ImageNet stats for pretrained ResNet34 encoder")

        # Save stats
        stats_path = processed / "dataset_stats.npy"
        np.save(str(stats_path), stats)
        print(f"  Saved to {stats_path}")

    print("\n[3/3] Preprocessing complete!")
    print("  Images are already organized. Augmentation applied on-the-fly during training.")


if __name__ == "__main__":
    main()
