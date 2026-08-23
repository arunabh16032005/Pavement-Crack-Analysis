"""
Phase 6 — Severity Classification (Classical MLP Baseline).

Classifies crack severity into 4 levels using structural descriptors:
  0 — Low:      Hairline cracks, <1mm width, isolated
  1 — Moderate: 1–3mm width, minor branching
  2 — Severe:   3–10mm width, significant branching, networked
  3 — Critical: >10mm width, alligator cracking, structural failure

This is the classical baseline for the Hybrid Quantum Neural Network
described in Phase 6 of the architecture document. The quantum VQC
classifier will be swapped in later.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SeverityClassifier(nn.Module):
    """
    MLP classifier for 4-level crack severity prediction.

    Architecture:
        FC(d_input → 128) → BN → ReLU → Dropout(0.3)
        FC(128 → 64)      → BN → ReLU → Dropout(0.3)
        FC(64 → 32)       → ReLU
        FC(32 → 4)        → Output logits

    Input: Structural feature vector F_g from Phase 5 (d ≈ 35 features)
    Output: 4-class logits (apply softmax for probabilities)
    """

    def __init__(
        self,
        input_dim: int = 35,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        num_classes: int = 4,
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input feature vector.
            hidden_dims: Tuple of hidden layer dimensions.
            num_classes: Number of severity classes.
            dropout_rate: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:
                # BatchNorm + ReLU + Dropout for first layers
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
            else:
                # Last hidden layer: just ReLU
                layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Feature vector (B, d_input).

        Returns:
            Logits (B, num_classes).
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Feature Normalization ───────────────────────────────────────────────────────

class FeatureNormalizer:
    """
    Z-score normalization for structural features.
    Fit on training set, apply to all sets.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        """Compute mean and std from training features."""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        # Avoid division by zero for constant features
        self.std[self.std < 1e-8] = 1.0
        self.is_fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (features - self.mean) / self.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)

    def save(self, path: str):
        """Save normalization parameters."""
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str) -> "FeatureNormalizer":
        """Load normalization parameters."""
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        self.is_fitted = True
        return self


# ── Pseudo-label Generation (Fallback) ──────────────────────────────────────────

def generate_pseudo_severity_labels(
    features: np.ndarray,
    feature_names: list,
) -> np.ndarray:
    """
    Generate severity pseudo-labels from structural features.
    Fallback for when Mendeley Pavement dataset is not available.

    Thresholds calibrated against ASTM D6433 severity definitions:
      Low (0):      density < 2%, max_width < 3px, components < 3
      Moderate (1): density 2-5%, max_width 3-8px, moderate branching
      Severe (2):   density 5-15%, max_width 8-20px, significant branching
      Critical (3): density > 15%, max_width > 20px, high junction count

    Args:
        features: Feature array (N, d).
        feature_names: List of feature names matching columns.

    Returns:
        Labels array (N,) with values in {0, 1, 2, 3}.
    """
    # Build name-to-index mapping
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    labels = np.zeros(len(features), dtype=np.int64)

    for i in range(len(features)):
        density = features[i, name_to_idx.get("density", 0)]
        max_width = features[i, name_to_idx.get("crack_width_max", 0)]
        num_components = features[i, name_to_idx.get("num_components", 0)]
        num_junctions = features[i, name_to_idx.get("num_junctions", 0)]

        # Rule-based severity assignment
        if density > 15 or max_width > 20 or num_junctions > 20:
            labels[i] = 3  # Critical
        elif density > 5 or max_width > 8 or num_junctions > 10:
            labels[i] = 2  # Severe
        elif density > 2 or max_width > 3 or num_components > 3:
            labels[i] = 1  # Moderate
        else:
            labels[i] = 0  # Low

    # Report distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"Pseudo-label distribution: {dict(zip(unique, counts))}")

    return labels
