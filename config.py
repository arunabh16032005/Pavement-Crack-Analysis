"""
Centralized configuration for the Quantum-Enhanced Crack Detection Pipeline.
All hyperparameters, paths, and settings in one place.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ── Project Root ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()


@dataclass
class DataConfig:
    """Dataset paths and split configuration."""
    # Root directories
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    severity_dir: Path = PROJECT_ROOT / "data" / "severity"
    quarantine_dir: Path = PROJECT_ROOT / "data" / "quarantine"

    # Raw dataset subdirectories (user places downloads here)
    raw_crackvision12k: Path = raw_dir / "crackvision12k"
    raw_crackforest: Path = raw_dir / "crackforest"
    raw_deepcrack: Path = raw_dir / "deepcrack"
    raw_mendeley_pavement: Path = raw_dir / "mendeley_pavement"

    # Processed splits
    train_images: Path = processed_dir / "train" / "images"
    train_masks: Path = processed_dir / "train" / "masks"
    val_images: Path = processed_dir / "val" / "images"
    val_masks: Path = processed_dir / "val" / "masks"
    test_cfd_images: Path = processed_dir / "test" / "crackforest" / "images"
    test_cfd_masks: Path = processed_dir / "test" / "crackforest" / "masks"
    test_dc_images: Path = processed_dir / "test" / "deepcrack" / "images"
    test_dc_masks: Path = processed_dir / "test" / "deepcrack" / "masks"

    # Split ratios
    train_val_split: float = 0.8  # 80% train, 20% val
    severity_train_val_split: float = 0.8

    # Deduplication
    phash_threshold: int = 10  # Hamming distance threshold for near-duplicates

    # Random seed
    seed: int = 42


@dataclass
class PreprocessConfig:
    """Image preprocessing and augmentation parameters."""
    # Image standardization
    image_size: Tuple[int, int] = (512, 512)
    interpolation: str = "bilinear"  # For images
    mask_interpolation: str = "nearest"  # For masks — preserve binary values

    # Noise suppression
    nlm_h: int = 10  # Non-Local Means filter strength
    nlm_template_window: int = 7
    nlm_search_window: int = 21

    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # ImageNet normalization
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation parameters (training only)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    shift_limit: float = 0.05
    scale_limit: float = 0.1
    rotate_limit: int = 15
    shift_scale_rotate_p: float = 0.5
    elastic_alpha: float = 120.0
    elastic_sigma: float = 12.0
    elastic_p: float = 0.3
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    brightness_contrast_p: float = 0.3
    gauss_noise_var_limit: Tuple[float, float] = (0.0, 10.0)
    gauss_noise_p: float = 0.2


@dataclass
class ModelConfig:
    """Segmentation model architecture parameters."""
    # Encoder
    encoder_name: str = "resnet34"
    encoder_pretrained: bool = True
    encoder_channels: Tuple[int, ...] = (64, 64, 128, 256, 512)

    # Decoder
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)

    # Output
    num_classes: int = 1  # Binary segmentation (crack / no-crack)
    output_activation: str = "none"  # Raw logits; sigmoid applied in loss


@dataclass
class SeverityConfig:
    """Severity classifier configuration."""
    # Architecture
    hidden_dims: Tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.3
    num_classes: int = 4  # Low, Moderate, Severe, Critical

    # Severity label mapping (flexible — handles various folder naming conventions)
    severity_map: Dict[str, int] = field(default_factory=lambda: {
        # Level 0 — Low severity
        "good": 0, "low": 0, "minor": 0, "light": 0, "hairline": 0,
        "0": 0, "class_0": 0, "level_0": 0, "0_low": 0,
        # Level 1 — Moderate severity
        "moderate": 1, "medium": 1, "fair": 1, "intermediate": 1,
        "1": 1, "class_1": 1, "level_1": 1, "1_moderate": 1,
        # Level 2 — Severe
        "severe": 2, "high": 2, "poor": 2, "heavy": 2, "serious": 2,
        "2": 2, "class_2": 2, "level_2": 2, "2_severe": 2,
        # Level 3 — Critical
        "critical": 3, "very_severe": 3, "failed": 3, "extreme": 3,
        "very_high": 3, "3": 3, "class_3": 3, "level_3": 3, "3_critical": 3,
    })

    # Training
    learning_rate: float = 1e-3
    epochs: int = 50
    patience: int = 10
    batch_size: int = 64


@dataclass
class TrainingConfig:
    """Segmentation training configuration."""
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_min: float = 1e-6  # Cosine annealing minimum

    # Training loop
    epochs: int = 100
    batch_size: int = 8  # For 8GB VRAM with FP16
    grad_accumulation_steps: int = 4  # Effective batch = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Early stopping
    patience: int = 15
    monitor_metric: str = "val_miou"  # Metric to monitor for early stopping

    # Mixed precision
    use_amp: bool = True  # FP16 mixed precision — saves ~40% VRAM

    # Loss weights
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    boundary_weight: float = 0.3
    boundary_dilation_kernel: int = 3  # Pixels for boundary dilation

    # Checkpointing
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    save_best_only: bool = True

    # Logging
    use_wandb: bool = False  # Toggle W&B logging
    wandb_project: str = "crack-segmentation"
    wandb_run_name: str = "resnet34-unet-baseline"

    # Results
    results_dir: Path = PROJECT_ROOT / "results"

    # Random seed
    seed: int = 42


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Ensure all directories exist."""
        dirs_to_create = [
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.severity_dir,
            self.data.quarantine_dir,
            self.training.checkpoint_dir,
            self.training.results_dir,
        ]
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)


def get_config() -> Config:
    """Get the default configuration. Modify returned object as needed."""
    return Config()
