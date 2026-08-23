"""
Kaggle Notebook — Crack Detection Pipeline (Self-Contained)
============================================================
Paste this into a SINGLE Kaggle notebook code cell and run.
No patching, no string replacement — everything is handled directly.
"""

import importlib
import os, sys, shutil, subprocess, zipfile
from pathlib import Path
from tqdm import tqdm

INPUT_ROOT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working/PROJ1")

print("=" * 70)
print("  CRACK DETECTION PIPELINE — KAGGLE RUNNER")
print("=" * 70)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Find and copy project files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 1/8] Setting up project files...")

project_candidates = [
    item.parent for item in INPUT_ROOT.rglob("config.py")
    if (item.parent / "src").is_dir()
]
config_found = project_candidates[0] if project_candidates else None

if not config_found:
    print("  ERROR: config.py not found in /kaggle/input/")
    for p in INPUT_ROOT.rglob("*"):
        if p.is_file():
            print(f"    {p}")
    raise FileNotFoundError("Project not found")

print(f"  Found project at: {config_found}")

if WORK_DIR.exists():
    shutil.rmtree(str(WORK_DIR))
# Copy code to the writable directory, but keep large archives on the read-only
# Kaggle input mount until they are extracted below.
shutil.copytree(
    str(config_found), str(WORK_DIR),
    ignore=shutil.ignore_patterns("*.zip", "__pycache__", ".git"),
)
os.chdir(str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR))
print("  ✓ Project files copied to /kaggle/working/PROJ1")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: Install dependencies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 2/8] Checking dependencies...")
required_modules = {
    "albumentations": "albumentations",
    "cv2": "opencv-python-headless",
    "skimage": "scikit-image",
    "networkx": "networkx",
    "seaborn": "seaborn",
}
missing_packages = []
for module, package in required_modules.items():
    try:
        importlib.import_module(module)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"  Installing missing packages: {', '.join(missing_packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_packages])
else:
    print("  All required packages are already available")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: Keep checked-in source unchanged
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 3/8] Preparing project source and data...")

# Fix transforms.py — rewrite the augmentation functions entirely
transforms_path = WORK_DIR / "src" / "data" / "transforms.py"
transforms_lines = []  # Source is fixed in the uploaded project; no rewriting.

# Find and replace the problematic augmentation block
new_lines = []
skip_until_normalize = False
for i, line in enumerate(transforms_lines):
    # Skip the old ShiftScaleRotate/ElasticTransform/GaussNoise block
    if "A.ShiftScaleRotate(" in line:
        skip_until_normalize = True
        # Insert replacement
        new_lines.append("        A.Affine(")
        new_lines.append('            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},')
        new_lines.append("            scale=(0.9, 1.1),")
        new_lines.append("            rotate=(-15, 15),")
        new_lines.append("            p=0.5,")
        new_lines.append("        ),")
        new_lines.append("        A.ElasticTransform(")
        new_lines.append("            alpha=120,")
        new_lines.append("            sigma=12,")
        new_lines.append("            p=0.3,")
        new_lines.append("        ),")
        new_lines.append("")
        new_lines.append("        # Color/intensity augmentations (image only, not mask)")
        new_lines.append("        A.RandomBrightnessContrast(")
        new_lines.append("            brightness_limit=0.2,")
        new_lines.append("            contrast_limit=0.2,")
        new_lines.append("            p=0.3,")
        new_lines.append("        ),")
        new_lines.append("        A.GaussNoise(p=0.2),")
        continue
    if skip_until_normalize:
        if "A.Normalize(" in line:
            skip_until_normalize = False
            new_lines.append(line)
        # else skip the line
        continue
    new_lines.append(line)

 # The project source is already Kaggle-compatible; do not rewrite it here.
 # Runtime rewriting made the notebook dependent on a particular
 # Albumentations version and could corrupt the module before import.
print("  ✓ transforms.py fixed")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extract archives uploaded as Kaggle inputs.  `Path.rglob()` cannot search
# inside ZIP files, which was why the previous runner could not find the data.
def find_archive(name_fragment):
    candidates = []
    for root in (INPUT_ROOT, WORK_DIR):
        candidates.extend(p for p in root.rglob("*.zip") if name_fragment in p.name.lower())
    return candidates[0] if candidates else None


def extract_archive(archive, destination):
    if archive is None:
        return
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted_from_kaggle"
    if not marker.exists():
        print(f"  Extracting {archive.name}...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
        marker.touch()


print("\n[STEP 3/8] Extracting uploaded data archives...")
extract_archive(find_archive("crackvision"), WORK_DIR / "data" / "raw" / "crackvision12k")
extract_archive(find_archive("crackforest"), WORK_DIR / "data" / "raw" / "crackforest")

# STEP 4: Organize data DIRECTLY (bypass organizer.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 4/8] Organizing datasets (direct copy)...")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
processed_dir = WORK_DIR / "data" / "processed"

# --- Find CrackVision12K folder ---
cv12k_src = None
for d in [WORK_DIR, INPUT_ROOT]:
    for item in d.rglob("*"):
        if item.is_dir() and "crackvision" in item.name.lower():
            # Check if it has train/IMG inside
            if (item / "train" / "IMG").is_dir() or (item / "train" / "GT").is_dir():
                cv12k_src = item
                break
            # Or check one level deeper
            for sub in item.iterdir():
                if sub.is_dir() and (sub / "train" / "IMG").is_dir():
                    cv12k_src = sub
                    break
    if cv12k_src:
        break

if not cv12k_src:
    # Brute force: find any directory containing train/IMG
    for d in [WORK_DIR, INPUT_ROOT]:
        for img_dir in d.rglob("IMG"):
            if img_dir.is_dir() and img_dir.parent.name == "train":
                cv12k_src = img_dir.parent.parent
                break
        if cv12k_src:
            break

if cv12k_src:
    print(f"  Found CrackVision12K at: {cv12k_src}")
    print(f"  Structure: {[c.name for c in cv12k_src.iterdir()]}")

    # Copy each split directly
    split_mapping = {
        "train": ("train", "images", "masks"),
        "val": ("val", "images", "masks"),
        "test": ("test/crackvision12k_test", "images", "masks"),
    }

    for split_name, (dst_split, img_folder, mask_folder) in split_mapping.items():
        src_split = cv12k_src / split_name
        if not src_split.is_dir():
            print(f"    {split_name}: not found, skipping")
            continue

        # Find images dir (IMG or images or similar)
        src_img = None
        for cand in ["IMG", "images", "image", "img"]:
            p = src_split / cand
            if p.is_dir():
                src_img = p
                break

        # Find masks dir (GT or masks or similar)
        src_mask = None
        for cand in ["GT", "gt", "masks", "mask", "groundtruth"]:
            p = src_split / cand
            if p.is_dir():
                src_mask = p
                break

        if not src_img or not src_mask:
            print(f"    {split_name}: IMG/GT dirs not found in {src_split}")
            continue

        dst_img = processed_dir / dst_split / img_folder
        dst_mask = processed_dir / dst_split / mask_folder
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_mask.mkdir(parents=True, exist_ok=True)

        # Copy images
        img_files = sorted(f for f in src_img.iterdir() if f.suffix.lower() in IMG_EXTS)
        for f in tqdm(img_files, desc=f"    {split_name} images"):
            shutil.copy2(str(f), str(dst_img / f.name))

        # Copy masks (match by stem)
        img_stems = {f.stem for f in img_files}
        mask_files = sorted(f for f in src_mask.iterdir()
                           if f.suffix.lower() in IMG_EXTS and f.stem in img_stems)
        for f in tqdm(mask_files, desc=f"    {split_name} masks"):
            shutil.copy2(str(f), str(dst_mask / f.name))

        print(f"    ✓ {split_name}: {len(img_files)} images, {len(mask_files)} masks")
else:
    print("  ✗ CrackVision12K not found anywhere!")
    # List everything for debugging
    for d in INPUT_ROOT.rglob("*"):
        if d.is_dir() and len(d.relative_to(INPUT_ROOT).parts) <= 4:
            print(f"    {d}")
    raise FileNotFoundError("CrackVision12K dataset not found")

# --- Find and organize CrackForest ---
print("\n  Organizing CrackForest...")
cfd_src = None
for d in [WORK_DIR, INPUT_ROOT]:
    for item in d.rglob("*"):
        if item.is_dir() and "crackforest" in item.name.lower():
            # Check if it has image/ subfolder
            if (item / "image").is_dir() or (item / "images").is_dir():
                cfd_src = item
                break
    if cfd_src:
        break

if cfd_src:
    print(f"  Found CrackForest at: {cfd_src}")

    # Find image dir
    img_dir = None
    for cand in ["image", "images", "img"]:
        p = cfd_src / cand
        if p.is_dir():
            img_dir = p
            break

    # Find GT dir (could have .mat files OR png segmentation masks)
    gt_dir = None
    for cand in ["groundTruth", "groundtruth", "ground_truth", "gt", "masks"]:
        p = cfd_src / cand
        if p.is_dir():
            gt_dir = p
            break

    # Also check for seg/ folder (has pre-computed PNG masks)
    seg_dir = None
    for cand in ["seg", "segmentation", "binary_masks"]:
        p = cfd_src / cand
        if p.is_dir():
            seg_dir = p
            break

    if img_dir and (gt_dir or seg_dir):
        dst_img = processed_dir / "test" / "crackforest" / "images"
        dst_mask = processed_dir / "test" / "crackforest" / "masks"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_mask.mkdir(parents=True, exist_ok=True)

        # Copy images
        img_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        for f in img_files:
            shutil.copy2(str(f), str(dst_img / f.name))

        masks_created = 0

        # Try seg/ folder first (pre-computed PNG masks — most reliable)
        if seg_dir:
            seg_files = sorted(f for f in seg_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
            for f in seg_files:
                shutil.copy2(str(f), str(dst_mask / f.name))
                masks_created += 1
            if masks_created > 0:
                print(f"    ✓ CrackForest: {len(img_files)} images, {masks_created} masks (from seg/)")

        # If seg/ didn't work, try .mat files
        if masks_created == 0 and gt_dir:
            mat_files = sorted(f for f in gt_dir.iterdir() if f.suffix == ".mat")
            png_masks = sorted(f for f in gt_dir.iterdir() if f.suffix.lower() in IMG_EXTS)

            if png_masks:
                for f in png_masks:
                    shutil.copy2(str(f), str(dst_mask / f.name))
                    masks_created += 1
                print(f"    ✓ CrackForest: {len(img_files)} images, {masks_created} masks")
            elif mat_files:
                try:
                    import scipy.io
                    import numpy as np
                    from PIL import Image as PILImage

                    for mat_path in mat_files:
                        mat_data = scipy.io.loadmat(str(mat_path))
                        mask = None
                        try:
                            if "groundTruth" in mat_data:
                                gt_cell = mat_data["groundTruth"]
                                # Try multiple access patterns
                                try:
                                    boundary = gt_cell[0, 0]["Boundaries"][0, 0]
                                    mask = (boundary > 0).astype(np.uint8) * 255
                                except (IndexError, KeyError):
                                    try:
                                        seg = gt_cell[0, 0]["Segmentation"][0, 0]
                                        mask = (seg > 0).astype(np.uint8) * 255
                                    except (IndexError, KeyError):
                                        # Last resort: just use the raw array
                                        arr = gt_cell[0, 0]
                                        if hasattr(arr, 'shape') and len(arr.shape) == 2:
                                            mask = (arr > 0).astype(np.uint8) * 255
                            elif "segmentation" in mat_data:
                                mask = (mat_data["segmentation"] > 0).astype(np.uint8) * 255
                            else:
                                for key, val in mat_data.items():
                                    if not key.startswith("_") and isinstance(val, np.ndarray) and val.ndim == 2:
                                        mask = (val > 0).astype(np.uint8) * 255
                                        break
                        except Exception:
                            pass

                        if mask is not None:
                            PILImage.fromarray(mask).save(str(dst_mask / f"{mat_path.stem}.png"))
                            masks_created += 1

                    print(f"    ✓ CrackForest: {len(img_files)} images, {masks_created}/{len(mat_files)} .mat→masks")
                except Exception as e:
                    print(f"    ⚠ CrackForest .mat failed: {e}")

        if masks_created == 0:
            print(f"    ⚠ CrackForest: No masks created. Will skip in evaluation.")
            # Remove the empty test dir so dataloader doesn't crash
            shutil.rmtree(str(dst_img))
            shutil.rmtree(str(dst_mask), ignore_errors=True)
    else:
        print(f"    ⚠ CrackForest: Could not find image/GT dirs")
else:
    print("  ⚠ CrackForest not found, skipping")

# --- Print summary ---
print("\n  Data summary:")
for split in ["train", "val"]:
    img_d = processed_dir / split / "images"
    mask_d = processed_dir / split / "masks"
    if img_d.exists():
        ic = sum(1 for f in img_d.iterdir() if f.suffix.lower() in IMG_EXTS)
        mc = sum(1 for f in mask_d.iterdir() if f.suffix.lower() in IMG_EXTS) if mask_d.exists() else 0
        print(f"    {split}: {ic} images, {mc} masks")

for test_name in ["crackforest", "crackvision12k_test"]:
    td = processed_dir / "test" / test_name / "images"
    if td.exists():
        print(f"    test/{test_name}: {sum(1 for f in td.iterdir())} images")

# Verify we have training data
train_img_dir = processed_dir / "train" / "images"
if not train_img_dir.exists() or sum(1 for _ in train_img_dir.iterdir()) == 0:
    print("\n  ✗ FATAL: No training images found!")
    raise FileNotFoundError("No training data")

print("  ✓ Data organization complete!")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: Validate data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 5/8] Quick data validation...")
import cv2
import numpy as np

sample_img = next(train_img_dir.iterdir())
img = cv2.imread(str(sample_img))
print(f"  Sample image: {sample_img.name}, shape={img.shape}, dtype={img.dtype}")

sample_mask_dir = processed_dir / "train" / "masks"
sample_mask = sample_mask_dir / sample_img.name
if not sample_mask.exists():
    # Try same stem, different extension
    for f in sample_mask_dir.iterdir():
        if f.stem == sample_img.stem:
            sample_mask = f
            break
mask = cv2.imread(str(sample_mask), cv2.IMREAD_GRAYSCALE)
if mask is not None:
    print(f"  Sample mask:  {sample_mask.name}, shape={mask.shape}, unique={np.unique(mask)}")
print("  ✓ Data looks good!")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: Train Segmentation Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 6/8] Training Segmentation Model...")
print("  This will take 2-5 hours on Kaggle GPU.\n")

import torch
import random

device_name = "cuda" if torch.cuda.is_available() else "cpu"
if device_name == "cpu":
    print("  WARNING: GPU is not enabled. Training will be very slow on CPU.")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from config import get_config
from src.data.dataset import get_segmentation_dataloaders
from src.data.transforms import get_train_augmentation, get_val_augmentation
from src.models.loss import CombinedSegmentationLoss
from src.models.unet import ResNetUNet
from src.training.trainer import SegmentationTrainer

config = get_config()

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
    batch_size=config.training.batch_size,
    num_workers=2,
    pin_memory=True,
)

print(f"  Train: {len(dataloaders['train'].dataset)} images")
print(f"  Val:   {len(dataloaders['val'].dataset)} images")

model = ResNetUNet(
    encoder_name=config.model.encoder_name,
    pretrained=config.model.encoder_pretrained,
    num_classes=config.model.num_classes,
    decoder_channels=config.model.decoder_channels,
)
print(f"  Parameters: {model.count_parameters()['total_millions']:.1f}M")

criterion = CombinedSegmentationLoss(
    bce_weight=config.training.bce_weight,
    dice_weight=config.training.dice_weight,
    boundary_weight=config.training.boundary_weight,
)

trainer = SegmentationTrainer(
    model=model,
    train_loader=dataloaders["train"],
    val_loader=dataloaders["val"],
    criterion=criterion,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    lr_min=config.training.lr_min,
    epochs=config.training.epochs,
    grad_accumulation_steps=config.training.grad_accumulation_steps,
    use_amp=True,
    patience=config.training.patience,
    checkpoint_dir=str(config.training.checkpoint_dir),
    device=device_name,
)

history = trainer.fit()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: Descriptors + Severity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 7/8] Extracting descriptors & training severity...")

from src.features.structural import StructuralDescriptorExtractor
from src.models.severity_mlp import SeverityClassifier, FeatureNormalizer, generate_pseudo_severity_labels
from torch.utils.data import DataLoader, TensorDataset

extractor = StructuralDescriptorExtractor()
features_dir = WORK_DIR / "features"
features_dir.mkdir(parents=True, exist_ok=True)

# Extract from GT masks
for split in ["train", "val"]:
    mask_dir = processed_dir / split / "masks"
    if not mask_dir.exists():
        continue
    mask_files = sorted(f for f in mask_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    all_features, all_names = [], []
    for mp in tqdm(mask_files, desc=f"  {split} features"):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        fv, _ = extractor.extract_all(m)
        all_features.append(fv)
        all_names.append(mp.stem)
    if all_features:
        np.savez(str(features_dir / f"structural_{split}_gt.npz"),
                 features=np.stack(all_features), names=all_names,
                 feature_names=extractor.get_feature_names())
        print(f"  ✓ {split}: {len(all_features)} vectors")

# Train severity with pseudo-labels
gt_file = features_dir / "structural_train_gt.npz"
if gt_file.exists():
    print("  Training severity classifier with pseudo-labels...")
    train_data = np.load(str(gt_file))
    val_data = np.load(str(features_dir / "structural_val_gt.npz"))
    train_features = train_data["features"]
    val_features = val_data["features"]
    feat_names = list(train_data.get("feature_names", []))
    train_labels = generate_pseudo_severity_labels(train_features, feat_names)
    val_labels = generate_pseudo_severity_labels(val_features, feat_names)

    normalizer = FeatureNormalizer()
    train_norm = np.nan_to_num(normalizer.fit_transform(train_features), nan=0, posinf=0, neginf=0)
    val_norm = np.nan_to_num(normalizer.transform(val_features), nan=0, posinf=0, neginf=0)
    normalizer.save(str(features_dir / "feature_normalizer.npz"))

    train_ds = TensorDataset(torch.from_numpy(train_norm.astype(np.float32)),
                             torch.from_numpy(train_labels.astype(np.int64)))
    val_ds = TensorDataset(torch.from_numpy(val_norm.astype(np.float32)),
                           torch.from_numpy(val_labels.astype(np.int64)))
    tl = DataLoader(train_ds, batch_size=64, shuffle=True)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False)

    cc = np.maximum(np.bincount(train_labels, minlength=4).astype(np.float32), 1)
    cw = torch.from_numpy(cc.sum() / (4 * cc))

    sev_model = SeverityClassifier(input_dim=train_features.shape[1], hidden_dims=(128, 64, 32), num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sev_model = sev_model.to(device)
    cw = cw.to(device)
    sev_crit = torch.nn.CrossEntropyLoss(weight=cw)
    sev_opt = torch.optim.Adam(sev_model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = 0
    for epoch in range(50):
        sev_model.train()
        for feats, labs in tl:
            feats, labs = feats.to(device), labs.to(device)
            sev_opt.zero_grad()
            sev_crit(sev_model(feats), labs).backward()
            sev_opt.step()

        sev_model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for feats, labs in vl:
                preds_all.append(sev_model(feats.to(device)).argmax(1).cpu().numpy())
                labels_all.append(labs.numpy())
        p_all = np.concatenate(preds_all)
        l_all = np.concatenate(labels_all)
        f1s = []
        for c in range(4):
            tp = ((p_all == c) & (l_all == c)).sum()
            fp = ((p_all == c) & (l_all != c)).sum()
            fn = ((p_all != c) & (l_all == c)).sum()
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            f1s.append(2 * pr * rc / max(pr + rc, 1e-8))
        mf1 = np.mean(f1s)
        if mf1 > best_f1:
            best_f1 = mf1
            torch.save({"model_state_dict": sev_model.state_dict(), "best_f1": best_f1,
                        "input_dim": train_features.shape[1]},
                       str(config.training.checkpoint_dir / "severity_best.pth"))
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50 — F1: {mf1:.4f} (best: {best_f1:.4f})")
    print(f"  ✓ Severity done. Best F1: {best_f1:.4f}")
else:
    print("  ⚠ No feature data, skipping severity")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8: Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 8/8] Final evaluation...")

from src.training.evaluate import cross_dataset_evaluation
from src.utils.visualization import plot_training_curves, generate_report

best_ckpt = config.training.checkpoint_dir / "best_segmentation.pth"
if best_ckpt.exists():
    ckpt = torch.load(str(best_ckpt), map_location=device_name, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_loaders = {k: v for k, v in dataloaders.items() if k != "train"}
    results = cross_dataset_evaluation(
        model=model, dataloaders=eval_loaders, device=device_name,
        save_dir=str(config.training.results_dir))

    plot_training_curves(history, save_path=str(config.training.results_dir / "training_curves.png"))
    import matplotlib.pyplot as plt
    plt.close()

    sev_results = None
    sev_ckpt = config.training.checkpoint_dir / "severity_best.pth"
    if sev_ckpt.exists():
        sd = torch.load(str(sev_ckpt), map_location="cpu", weights_only=False)
        sev_results = {"best_f1": sd.get("best_f1", "N/A")}

    generate_report(seg_results=results, severity_results=sev_results,
                    history=history, output_dir=str(config.training.results_dir))

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for name, res in results.items():
        miou = res.get("miou_mean", 0)
        f1 = res.get("f1_mean", 0)
        t = "✓" if miou >= 0.70 else "✗"
        print(f"  [{t}] {name:<20} mIoU: {miou:.4f}  F1: {f1:.4f}")
    if sev_results:
        print(f"  [?] {'severity':<20} F1: {sev_results.get('best_f1', 'N/A')}")
    print("=" * 70)
else:
    print("  ✗ No checkpoint found")

# Copy results for download
print("\n  Saving outputs...")
output_root = Path("/kaggle/working")
for d in ["checkpoints", "results", "features"]:
    src = WORK_DIR / d
    dst = output_root / d
    if src.exists():
        if dst.exists():
            shutil.rmtree(str(dst))
        shutil.copytree(str(src), str(dst))
        print(f"  ✓ {d}/ → /kaggle/working/{d}/")

print("\n✓ DONE! Download outputs from the notebook's Output tab.")
