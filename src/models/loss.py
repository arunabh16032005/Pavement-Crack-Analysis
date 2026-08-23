"""
Loss Functions for Crack Segmentation.

Implements:
  - DiceLoss: Directly optimizes F1/Dice coefficient
  - BoundaryAwareLoss: Higher weight on crack boundary pixels
  - CombinedSegmentationLoss: BCE + Dice + Boundary (default training loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Directly optimizes the Dice coefficient (F1-score), making it effective
    for class-imbalanced problems like crack segmentation where crack pixels
    are a small fraction of the total.

    Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
    DiceLoss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W). Sigmoid applied internally.
            targets: Ground truth binary mask (B, 1, H, W), values in {0, 1}.

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-sensitive BCE loss.

    Applies higher weight (default 3×) to crack boundary pixels,
    forcing the model to learn precise crack edges rather than
    producing blobby predictions.

    Boundary pixels are identified by dilating the ground truth mask
    and taking the difference (dilation - original = boundary ring).
    """

    def __init__(
        self,
        boundary_weight: float = 3.0,
        dilation_kernel_size: int = 3,
    ):
        """
        Args:
            boundary_weight: Extra weight for boundary pixels.
            dilation_kernel_size: Size of morphological dilation kernel.
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kernel_size = dilation_kernel_size

        # Create dilation kernel (circular)
        self.register_buffer(
            "dilation_kernel",
            torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size),
        )

    def _compute_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary pixels via morphological dilation.

        Args:
            mask: Binary mask (B, 1, H, W).

        Returns:
            Boundary mask (B, 1, H, W), 1 at boundary pixels.
        """
        padding = self.kernel_size // 2
        dilated = F.conv2d(mask, self.dilation_kernel, padding=padding)
        dilated = (dilated > 0).float()
        boundary = dilated - mask
        boundary = torch.clamp(boundary, 0, 1)
        return boundary

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W).
            targets: Ground truth binary mask (B, 1, H, W).

        Returns:
            Scalar boundary-weighted BCE loss.
        """
        boundary = self._compute_boundary(targets)

        # Weight map: 1.0 everywhere, boundary_weight at boundaries
        weights = torch.ones_like(targets) + boundary * (self.boundary_weight - 1.0)

        # Weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction="mean"
        )

        return loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for crack segmentation training.

    L_total = α × BCE + β × Dice + γ × BoundaryBCE

    This combination provides:
      - BCE: Stable gradients at all prediction values
      - Dice: Directly optimizes F1, handles class imbalance
      - Boundary: Forces precise crack edge delineation
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.3,
        boundary_pixel_weight: float = 3.0,
        dilation_kernel_size: int = 3,
        smooth: float = 1e-6,
    ):
        """
        Args:
            bce_weight: Weight for standard BCE loss (α).
            dice_weight: Weight for Dice loss (β).
            boundary_weight: Weight for boundary-aware BCE (γ).
            boundary_pixel_weight: Extra weight on boundary pixels.
            dilation_kernel_size: Kernel size for boundary computation.
            smooth: Dice loss smoothing factor.
        """
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryAwareLoss(
            boundary_weight=boundary_pixel_weight,
            dilation_kernel_size=dilation_kernel_size,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            logits: Raw model output (B, 1, H, W).
            targets: Ground truth binary mask (B, 1, H, W), values in {0, 1}.

        Returns:
            Dict with 'total', 'bce', 'dice', 'boundary' loss values.
        """
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.boundary_weight * boundary
        )

        return {
            "total": total,
            "bce": bce.detach(),
            "dice": dice.detach(),
            "boundary": boundary.detach(),
        }
