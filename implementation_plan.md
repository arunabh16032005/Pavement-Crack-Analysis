# 50% Pipeline Implementation Plan — Targeting 70% Accuracy (v2)

## Goal

Implement the **core functional half** of the Quantum-Enhanced Multi-Domain Framework for Crack Segmentation, Severity Assessment & PCI Estimation. The target is a working end-to-end pipeline (data in → segmentation mask + severity label out) that achieves **≥70% accuracy** (mIoU for segmentation, classification accuracy for severity) on the CrackForest and DeepCrack test sets.

**Key constraints:**
- All code written locally; training happens on cloud (8 GB VRAM GPU)
- Datasets downloaded manually by user
- Severity classifier trained on **real labels** from Mendeley Pavement dataset

---

## Strategy: What "50% of the Pipeline" Means

The full architecture has **8 phases**. We implement **4 phases fully** and **2 phases in simplified form**, deliberately deferring the quantum and VMamba components.

### Phases Included (50% Scope)

| Phase | Status | What We Build |
|-------|--------|---------------|
| **Phase 1 — Data Acquisition** | ✅ Full | Directory structure, deduplication, splits |
| **Phase 2 — Preprocessing & Enhancement** | ✅ Full | Standardization, CLAHE, augmentation pipeline |
| **Phase 4 — Segmentation Network** | ⚠️ Classical (ResNet34 U-Net) | Produces `M_seg` — quantum VM-UNet swapped in later |
| **Phase 5 — Structural Descriptors** | ✅ Full | Geometry, topology, network analysis from masks |
| **Phase 6 — Severity Prediction** | ⚠️ Classical (MLP on real labels) | Produces severity `S` — quantum NN swapped in later |
| **Phase 8 — Training Loop** | ⚠️ Simplified | Single-task segmentation loss, basic hyperparams |

### Phases Deferred

| Phase | Reason |
|-------|--------|
| **Phase 3 — Multi-Domain Features & Quantum Encoding** | Frangi/Gabor/Wavelet/VMamba/PennyLane — classical U-Net learns its own features |
| **Phase 7 — PCI/DI Assessment** | Downstream of severity; needs calibrated predictions first |
| **All quantum circuits (3.5, 3.6, 4.1, 4.2, 6.2)** | Architecture doc states they are "drop-in replacements" — classical-only mode first |

> [!IMPORTANT]
> The architecture document explicitly states: *"The system can run in classical-only mode by bypassing quantum modules."* This plan builds that classical baseline. Quantum enhancements are added in the next iteration to measure their actual contribution.

---

## Dataset Download Instructions (Manual)

You will need to download **4 datasets** manually and place them in the correct folders. I will create a `data/raw/` directory with README instructions, and the code will handle organization from there.

### Downloads Required

| # | Dataset | Download URL | Size | What to Download |
|---|---------|-------------|------|-----------------|
| 1 | **CrackVision12K** | [UCL Repository](https://rdr.ucl.ac.uk/articles/dataset/CrackVision12K/26946472) | ~2.5 GB | Full ZIP (images + masks) |
| 2 | **CrackForest (CFD)** | [GitHub](https://github.com/cuilimeng/CrackForest-dataset) | ~50 MB | Clone or download ZIP (`image/` + `groundTruth/`) |
| 3 | **DeepCrack** | [GitHub](https://github.com/yhlleo/DeepCrack) | ~200 MB | Clone or download ZIP (train/test splits) |
| 4 | **Mendeley Pavement** | [Mendeley](https://data.mendeley.com/datasets/gsbmknrhkv/6) | ~1.5 GB | Full download (~7,000+ images with severity folders) |

### Placement Instructions

```
PROJ1/
└── data/
    └── raw/                           ← Place downloads here
        ├── crackvision12k/            ← Extract CrackVision12K ZIP contents here
        │   ├── images/
        │   └── masks/
        ├── crackforest/               ← Clone/extract CrackForest here
        │   ├── image/
        │   └── groundTruth/
        ├── deepcrack/                 ← Clone/extract DeepCrack here
        │   ├── train_img/
        │   ├── train_lab/
        │   ├── test_img/
        │   └── test_lab/
        └── mendeley_pavement/         ← Extract Mendeley Pavement here
            ├── Low/                   ← Severity-labeled folders
            ├── Moderate/
            ├── Severe/
            └── Critical/             (folder names may vary — code handles mapping)
```

> [!NOTE]
> The code includes a **`verify_raw_data()`** function that checks if all required files are present and reports any missing datasets before proceeding. You don't need to get the folder names exactly right — the organizer script auto-detects the structure.

---

## Severity Labels: Mendeley Pavement Dataset

> [!TIP]
> **Key insight from the architecture doc:** Dataset #4 — **Mendeley Pavement (`gsbmknrhkv`)** — has ~7,000+ images with **classification-with-severity labels**. This gives us real human-annotated severity labels instead of noisy pseudo-labels.

### How we use it:

1. **Mendeley Pavement** images are organized into severity-labeled folders (the dataset groups images by damage severity)
2. We map these folder labels to our 4-level severity scale:
   - **Low (0):** Hairline/minor cracks
   - **Moderate (1):** Visible cracks, minor branching
   - **Severe (2):** Wide cracks, significant damage pattern
   - **Critical (3):** Alligator cracking, structural failure
3. We run the trained segmentation model on these images → extract structural descriptors (Phase 5) → train severity MLP with **real severity labels**
4. This gives us a properly supervised severity classifier, not a rule-based approximation

### Fallback
If the Mendeley folder structure doesn't map cleanly to 4 levels, the code includes a `SeverityMapper` class that handles flexible mapping and logs any ambiguous cases for manual review.

---

## Proposed Changes

### Component 1: Project Structure & Configuration

#### [NEW] [requirements.txt](file:///c:/Users/Arunabh/Desktop/PROJ1/requirements.txt)
```
torch>=2.1
torchvision>=0.16
opencv-python>=4.8
albumentations>=1.3
scikit-image>=0.21
scipy>=1.11
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
Pillow>=10.0
imagehash>=4.3
scikit-learn>=1.3
networkx>=3.1
wandb>=0.16
```

#### [NEW] [config.py](file:///c:/Users/Arunabh/Desktop/PROJ1/config.py)
Centralized dataclass configuration:
- Dataset paths (raw, processed, splits)
- Preprocessing params (image size=512, CLAHE clipLimit=2.0, normalization stats)
- Model hyperparams (encoder=ResNet34, decoder channels=[256,128,64,32])
- Training hyperparams (lr=1e-4, batch=8, epochs=100, patience=15)
- Severity class mapping dictionary
- Random seed=42 for reproducibility

#### [NEW] Full project directory structure
```
PROJ1/
├── architecture.md                  # Existing
├── config.py                        # Centralized config
├── requirements.txt                 # Dependencies
├── data/
│   ├── raw/                         # ← User places downloads here
│   │   └── README.md                # Download & placement instructions
│   ├── processed/                   # Auto-generated: preprocessed images
│   │   ├── train/images/            # 9,600 images (80% of CV12K)
│   │   ├── train/masks/
│   │   ├── val/images/              # 2,400 images (20% of CV12K)
│   │   ├── val/masks/
│   │   └── test/
│   │       ├── crackforest/{images,masks}/
│   │       └── deepcrack/{images,masks}/
│   └── severity/                    # Auto-generated: Mendeley → severity split
│       ├── train/{0_low,1_moderate,2_severe,3_critical}/
│       └── val/{0_low,1_moderate,2_severe,3_critical}/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── organizer.py             # Phase 1: verify, organize, dedup, split
│   │   ├── dataset.py               # PyTorch Dataset classes
│   │   └── transforms.py            # Phase 2: preprocessing & augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py                  # Phase 4: ResNet34 U-Net
│   │   ├── loss.py                  # BCE + Dice + Boundary loss
│   │   └── severity_mlp.py          # Phase 6: Severity classifier
│   ├── features/
│   │   ├── __init__.py
│   │   └── structural.py            # Phase 5: Structural descriptors
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Phase 8: Training loop
│   │   └── evaluate.py              # Metrics & evaluation
│   └── utils/
│       ├── __init__.py
│       └── visualization.py         # Plotting & overlays
├── scripts/
│   ├── 01_organize_data.py          # Organize raw → processed
│   ├── 02_preprocess.py             # Cache preprocessed images
│   ├── 03_train_segmentation.py     # Train U-Net
│   ├── 04_extract_descriptors.py    # Extract F_g from masks
│   ├── 05_train_severity.py         # Train severity MLP
│   └── 06_evaluate_full.py          # Full pipeline evaluation
└── checkpoints/                     # Saved models
```

---

### Component 2: Phase 1 — Data Organization (No Auto-Download)

Since datasets are downloaded manually, this component focuses on **verification, organization, deduplication, and splitting**.

#### [NEW] [data/raw/README.md](file:///c:/Users/Arunabh/Desktop/PROJ1/data/raw/README.md)
- Exact download URLs for all 4 datasets
- Expected folder structure after extraction
- Checksum values where available

#### [NEW] [src/data/organizer.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/data/organizer.py)
- Function `verify_raw_data(raw_dir)`:
  - Checks each dataset folder exists and has expected contents
  - Reports counts: found X images, Y masks
  - Raises clear error messages for missing/malformed datasets
- Function `organize_crackvision12k(raw_dir, processed_dir)`:
  - Copies images/masks from raw to processed structure
  - Creates 80/20 train/val split (`sklearn.train_test_split`, `random_state=42`, stratified by crack pixel percentage buckets)
  - Logs split assignments to `splits.csv`
- Function `organize_test_sets(raw_dir, processed_dir)`:
  - Organizes CrackForest: maps `image/` → `images/`, converts `.mat` groundTruth to PNG masks
  - Organizes DeepCrack: uses pre-defined train/test split (test only for us)
- Function `organize_severity_data(raw_dir, severity_dir)`:
  - Scans Mendeley Pavement folder structure
  - Maps folder names to severity levels {0,1,2,3} via `SeverityMapper`
  - Creates 80/20 train/val split preserving class balance
  - Logs mapping + split to `severity_splits.csv`
- Function `compute_phashes(image_dir)`:
  - Perceptual hashes for all images using `imagehash` (pHash)
- Function `remove_duplicates(train_hashes, test_hashes, threshold=10)`:
  - Finds near-duplicates between train and test (Hamming distance ≤ 10)
  - Moves duplicates to `data/quarantine/`, logs to `dedup_log.csv`
- Class `SeverityMapper`:
  - Handles flexible mapping of Mendeley folder names to 4-level scale
  - Logs ambiguous mappings for manual review
  - Supports custom override dictionary

---

### Component 3: Phase 2 — Preprocessing & Enhancement

#### [NEW] [src/data/transforms.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/data/transforms.py)

**Preprocessing pipeline (applied to ALL images — train, val, test):**

| Step | Function | Parameters | Notes |
|------|----------|------------|-------|
| 1 | `standardize()` | Resize 512×512 bilinear, RGB, float32 [0,1] | Masks resized with nearest-neighbor |
| 2 | `denoise()` | `cv2.fastNlMeansDenoisingColored(h=10, templateWindowSize=7)` | Preserves crack edges |
| 3 | `enhance_clahe()` | LAB space, L-channel CLAHE (clipLimit=2.0, tileGridSize=8×8) | No color artifacts |
| 4 | `normalize_imagenet()` | μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225] | Standard for pretrained encoders |

**Augmentation pipeline (training only):**
```python
albumentations.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ElasticTransform(alpha=120, sigma=12, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(0, 0.01*255), p=0.2),
])
# All transforms applied jointly to image + mask
```

> [!NOTE]
> Skipping Step 2.5 (GrabCut ROI Localization) in this iteration. CrackVision12K is already pre-filtered for pavement-dominant images. Will add in next iteration if non-pavement content hurts performance.

#### [NEW] [src/data/dataset.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/data/dataset.py)
- Class `CrackSegmentationDataset(Dataset)`:
  - Loads image-mask pairs from processed directory
  - `__getitem__`: preprocessing → augmentation → tensors
  - Supports both from-disk and from-cached-array loading
- Class `SeverityDataset(Dataset)`:
  - Loads images from severity-labeled folders (Mendeley Pavement)
  - Returns `(image_tensor, severity_label)` pairs
  - Class-balanced sampling via `WeightedRandomSampler`
- Function `get_dataloaders(config, task='segmentation')`:
  - Returns dict of `{'train': DataLoader, 'val': DataLoader, 'test_cfd': DataLoader, 'test_dc': DataLoader}`
  - batch_size=8, num_workers=4, pin_memory=True

#### [NEW] [scripts/02_preprocess.py](file:///c:/Users/Arunabh/Desktop/PROJ1/scripts/02_preprocess.py)
- Pre-computes denoised + CLAHE-enhanced images → caches to `data/processed/`
- Computes dataset-level statistics (mean, std) on training set
- Validates all image-mask pairs (dimension match, mask is binary)
- Reports dataset statistics: image count, size distribution, crack pixel % histogram

---

### Component 4: Phase 4 — Segmentation Network (Classical U-Net)

#### [NEW] [src/models/unet.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/models/unet.py)

- Class `ResNetUNet(nn.Module)`:
  - **Encoder**: `torchvision.models.resnet34(pretrained=True)` — first 4 layer groups
    - Stage 1: 512×512 → 256×256, C=64
    - Stage 2: 256×256 → 128×128, C=128
    - Stage 3: 128×128 → 64×64, C=256
    - Stage 4: 64×64 → 32×32, C=512
  - **Decoder**: `DecoderBlock` modules — ConvTranspose2d + skip concat + 2×(Conv+BN+ReLU)
    - Up4: 32→64, channels 512+256→256
    - Up3: 64→128, channels 256+128→128
    - Up2: 128→256, channels 128+64→64
    - Up1: 256→512, channels 64+64→32
  - **Output Head**: `Conv2d(32, 1, kernel_size=1)` — raw logits (sigmoid applied in loss)
  - ~24.4M parameters | fits comfortably in 8 GB VRAM at batch=8, 512×512

> [!TIP]
> **Why ResNet34 U-Net instead of VM-UNet?** Proven baseline that achieves 65-80% mIoU on crack benchmarks. Trains faster, well-understood, solid classical baseline. VM-UNet + quantum injection will be swapped in as the encoder later — the decoder stays the same.

#### [NEW] [src/models/loss.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/models/loss.py)

- Class `DiceLoss(nn.Module)`:
  - Soft Dice: `1 - (2*intersection + smooth) / (sum_pred + sum_target + smooth)`
  - smooth=1e-6 for numerical stability
- Class `BoundaryAwareLoss(nn.Module)`:
  - Computes morphological dilation of GT mask (3px kernel) → boundary mask
  - Boundary pixels get 3× weight in BCE computation
- Class `CombinedSegmentationLoss(nn.Module)`:
  - `L = 0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss + 0.3 × BoundaryBCE`
  - All components differentiable end-to-end

---

### Component 5: Phase 5 — Structural Descriptor Extraction

#### [NEW] [src/features/structural.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/features/structural.py)

- Class `StructuralDescriptorExtractor`:

  | Method | Features Extracted | Libraries |
  |--------|-------------------|-----------|
  | `extract_geometry(mask)` | crack_length, width_mean/max/std, area, aspect_ratio | `skimage.morphology.skeletonize`, `scipy.ndimage.distance_transform_edt` |
  | `extract_structural(mask)` | density, num_components, num_junctions, fragmentation_index | `skimage.measure.label` |
  | `extract_network(mask)` | orientation_histogram (18-bin), avg_path_length, clustering_coeff, degree_dist | `networkx`, custom numpy |
  | `extract_all(mask)` → `np.ndarray` | Concatenated feature vector F_g ∈ ℝ^d (d≈35) | All above |

- No ML training required — pure computation on binary masks
- Handles edge cases: empty masks (all zeros), tiny masks (<10px), disconnected fragments

#### [NEW] [scripts/04_extract_descriptors.py](file:///c:/Users/Arunabh/Desktop/PROJ1/scripts/04_extract_descriptors.py)
- Runs extraction on predicted masks from trained segmentation model
- Also runs on Mendeley Pavement images (after segmenting them) for severity training features
- Saves to `features/structural_train.npz` and `features/structural_val.npz`

---

### Component 6: Phase 6 — Severity Prediction (Classical MLP with Real Labels)

#### [NEW] [src/models/severity_mlp.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/models/severity_mlp.py)

- Class `SeverityClassifier(nn.Module)`:
  ```
  FC(d_input → 128) → BatchNorm → ReLU → Dropout(0.3)
  FC(128 → 64)      → BatchNorm → ReLU → Dropout(0.3)
  FC(64 → 32)       → ReLU
  FC(32 → 4)        → Output logits
  ```
  - Input: Structural feature vector `F_g` from Phase 5 (d≈35 features)
  - Output: 4-class severity logits
  - Small model (~25K params) — trains in minutes

- Class `SeverityMapper`:
  - Maps Mendeley Pavement folder names to 4-level scale
  - Default mapping (adjustable via config):
    ```python
    SEVERITY_MAP = {
        'good': 0, 'low': 0, 'minor': 0,
        'moderate': 1, 'medium': 1, 'fair': 1,
        'severe': 2, 'high': 2, 'poor': 2,
        'critical': 3, 'very_severe': 3, 'failed': 3,
    }
    ```
  - Handles case-insensitive matching, logs unmapped folders

- Training details:
  - Loss: `CrossEntropyLoss` with inverse-frequency class weights
  - Optimizer: Adam(lr=1e-3)
  - Epochs: 50 with early stopping (patience=10)
  - Trains on features extracted from Mendeley Pavement images

#### [NEW] [scripts/05_train_severity.py](file:///c:/Users/Arunabh/Desktop/PROJ1/scripts/05_train_severity.py)
- Pipeline: Load Mendeley images → segment with trained U-Net → extract F_g → train severity MLP
- Reports: accuracy, macro F1, per-class F1, confusion matrix
- Saves best model to `checkpoints/severity_best.pth`

---

### Component 7: Phase 8 — Training Loop (Simplified)

#### [NEW] [src/training/trainer.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/training/trainer.py)

- Class `SegmentationTrainer`:
  - **Optimizer**: `AdamW(lr=1e-4, weight_decay=1e-4)`
  - **Scheduler**: `CosineAnnealingLR(T_max=100, eta_min=1e-6)`
  - **Loss**: `CombinedSegmentationLoss` (BCE + Dice + Boundary)
  - **Mixed precision**: `torch.amp.GradScaler` for FP16 training (saves VRAM)
  - **Gradient accumulation**: 4 steps → effective batch=32
  - Methods:
    - `train_one_epoch()` → logs loss, lr per batch
    - `validate()` → pixel acc, mIoU, F1, precision, recall (no gradients)
    - `fit(num_epochs=100)` → full loop with early stopping (patience=15 on val mIoU)
    - `save_checkpoint()` / `load_checkpoint()` → `checkpoints/best_segmentation.pth`
  - W&B logging (optional, toggle in config)

- Training config for 8 GB VRAM:

  | Parameter | Value | Rationale |
  |-----------|-------|-----------|
  | Image size | 512×512 | Full resolution, fits in 8GB with FP16 |
  | Batch size | 8 | Max for 8GB VRAM with ResNet34 U-Net |
  | Grad accumulation | 4 | Effective batch = 32 |
  | Epochs | 100 | Early stopping prevents waste |
  | Mixed precision | FP16 | ~40% VRAM reduction |
  | Num workers | 4 | Async data loading |

#### [NEW] [src/training/evaluate.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/training/evaluate.py)

- Function `compute_segmentation_metrics(pred, gt)` → dict:
  - Pixel Accuracy, IoU, F1 (Dice), Precision, Recall
  - All computed at threshold=0.5 on sigmoid output
- Function `evaluate_on_dataset(model, loader, name)` → dict:
  - Mean ± std of all metrics across the dataset
  - Saves predicted masks to `results/{name}/`
- Function `cross_dataset_evaluation(model, config)` → report:
  - Evaluates on CrackForest AND DeepCrack
  - Computes domain adaptation gap
  - Generates comparison table

---

### Component 8: Utilities

#### [NEW] [src/utils/visualization.py](file:///c:/Users/Arunabh/Desktop/PROJ1/src/utils/visualization.py)
- `overlay_mask(image, mask, alpha=0.5, color=(255,0,0))`: Red-tinted overlay
- `plot_training_curves(history)`: Loss + mIoU over epochs (dual y-axis)
- `plot_confusion_matrix(y_true, y_pred, classes)`: Severity CM with percentages
- `create_prediction_grid(images, gt_masks, pred_masks, n=8)`: 3-column comparison grid
- `generate_report(results, output_dir)`: Markdown report with embedded figures

---

## Execution Order (Step-by-Step)

| Step | What I Build | Depends On | Deliverable |
|------|-------------|------------|-------------|
| **1** | Create full project directory structure | — | All folders + `__init__.py` files |
| **2** | Write `config.py` | Step 1 | Centralized configuration |
| **3** | Write `requirements.txt` | — | Dependency list |
| **4** | Write `data/raw/README.md` | — | Download instructions for user |
| **5** | Write `src/data/organizer.py` | Step 2 | Verify + organize + dedup + split |
| **6** | Write `scripts/01_organize_data.py` | Step 5 | Orchestration script |
| **7** | Write `src/data/transforms.py` | Step 2 | Preprocessing + augmentation |
| **8** | Write `src/data/dataset.py` | Step 7 | PyTorch Dataset + DataLoader |
| **9** | Write `scripts/02_preprocess.py` | Steps 7-8 | Cache preprocessing script |
| **10** | Write `src/models/unet.py` | Step 2 | ResNet34 U-Net model |
| **11** | Write `src/models/loss.py` | — | BCE + Dice + Boundary loss |
| **12** | Write `src/training/trainer.py` | Steps 10-11 | Training loop |
| **13** | Write `src/training/evaluate.py` | Step 10 | Evaluation metrics |
| **14** | Write `scripts/03_train_segmentation.py` | Steps 8,12,13 | Training orchestration |
| **15** | Write `src/features/structural.py` | — | Descriptor extraction |
| **16** | Write `scripts/04_extract_descriptors.py` | Step 15 | Extraction orchestration |
| **17** | Write `src/models/severity_mlp.py` | Step 15 | Severity classifier |
| **18** | Write `scripts/05_train_severity.py` | Steps 16-17 | Severity training script |
| **19** | Write `src/utils/visualization.py` | — | Plotting utilities |
| **20** | Write `scripts/06_evaluate_full.py` | Steps 13-19 | Full pipeline evaluation |

> [!IMPORTANT]
> After I complete all 20 steps, you will need to:
> 1. Download the 4 datasets and place them in `data/raw/` per the README
> 2. Install dependencies: `pip install -r requirements.txt`
> 3. Run scripts 01 through 06 **in order** on the cloud machine
> 4. Review the evaluation report generated by script 06

---

## Verification Plan

### Automated Tests (run on cloud)

```bash
# Step 1: Verify data organization
python scripts/01_organize_data.py --verify-only

# Step 2: Preprocess + validate
python scripts/02_preprocess.py

# Step 3: Train segmentation (full run)
python scripts/03_train_segmentation.py

# Step 4: Extract structural descriptors
python scripts/04_extract_descriptors.py

# Step 5: Train severity classifier
python scripts/05_train_severity.py

# Step 6: Full pipeline evaluation
python scripts/06_evaluate_full.py --checkpoint checkpoints/best_segmentation.pth
```

### Accuracy Targets (70% Goal)

| Metric | Target | Literature Benchmark |
|--------|--------|---------------------|
| **Segmentation mIoU** (CrackForest) | ≥ 0.70 | ResNet U-Net on CFD: ~0.65-0.75 |
| **Segmentation F1** (CrackForest) | ≥ 0.75 | Classical methods: 0.70-0.85 |
| **Segmentation mIoU** (DeepCrack) | ≥ 0.65 | Cross-dataset: expect 5-10% drop |
| **Severity Accuracy** (Mendeley) | ≥ 0.70 | Real labels — achievable with balanced data |
| **Pixel Accuracy** | ≥ 0.90 | High baseline due to class imbalance |

> [!NOTE]
> **70% accuracy is well within reach** with a classical ResNet34 U-Net. Published benchmarks show 65-80% mIoU on crack datasets. CrackVision12K (12K images) + proper augmentation + CLAHE + boundary-aware loss should comfortably hit this target. The quantum enhancements in the next iteration are expected to push this to 80-85%.

### Manual Verification
- Visual inspection of 20 random predictions (mask overlays)
- Check severity predictions against visual crack appearance
- Review train/val loss curves for overfitting signs
- Verify cross-dataset generalization (CFD → DeepCrack gap)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Mendeley severity folders don't map to 4 levels cleanly | `SeverityMapper` handles flexible mapping + logs ambiguous cases |
| mIoU < 70% with ResNet34 | Upgrade to ResNet50 or EfficientNet-B4; increase boundary loss weight |
| Training OOM on 8GB GPU | FP16 already enabled; fallback: reduce batch to 4, keep grad_accum=8 |
| Cross-dataset gap too large | Add test-time augmentation (TTA); fine-tune on small subset of test domain |
| Class imbalance in severity | WeightedRandomSampler + inverse-frequency class weights in CE loss |
| Mendeley Pavement images don't have masks | We segment them first with our trained U-Net → extract F_g → train severity |
