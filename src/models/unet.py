"""
Phase 4 — ResNet34 U-Net for Crack Segmentation.

Classical baseline encoder-decoder architecture:
  Encoder: ResNet34 pretrained on ImageNet
  Decoder: ConvTranspose2d upsampling + skip connections

This is the classical equivalent of the Quantum-Enhanced VM-UNet.
The decoder architecture stays the same when swapping in VMamba + quantum later.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Single decoder block: upsample + skip concatenation + double convolution.

    Architecture:
        ConvTranspose2d(upsample 2×) → Concat(skip) → Conv-BN-ReLU → Conv-BN-ReLU
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels: Channels from the lower decoder level.
            skip_channels: Channels from the corresponding encoder skip connection.
            out_channels: Output channels after this block.
        """
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

        # After concatenation: in_channels + skip_channels
        concat_channels = in_channels + skip_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from lower level (B, in_channels, H, W).
            skip: Skip connection from encoder (B, skip_channels, 2H, 2W).
        """
        x = self.upsample(x)

        # Handle size mismatches (can happen with odd input dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class ResNetUNet(nn.Module):
    """
    U-Net with ResNet34 encoder for binary crack segmentation.

    Encoder stages (from ResNet34 pretrained on ImageNet):
        Stage 0: 512×512 → 256×256,  C=64   (conv1 + bn1 + relu + maxpool)
        Stage 1: 256×256 → 128×128,  C=64   (layer1)
        Stage 2: 128×128 → 64×64,    C=128  (layer2)
        Stage 3: 64×64   → 32×32,    C=256  (layer3)
        Stage 4: 32×32   → 16×16,    C=512  (layer4)

    Decoder stages (ConvTranspose2d + skip connections):
        Up4: 16→32,   512+256 → 256
        Up3: 32→64,   256+128 → 128
        Up2: 64→128,  128+64  → 64
        Up1: 128→256, 64+64   → 32

    Output: 1×1 Conv → raw logits (sigmoid applied in loss function)

    Total params: ~24.4M
    VRAM: ~3.5 GB at batch=8, 512×512, FP16
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()

        self.num_classes = num_classes

        # ── Encoder (ResNet34) ──
        if encoder_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            try:
                resnet = models.resnet34(weights=weights)
            except Exception as exc:
                if not pretrained:
                    raise
                print(f"Warning: pretrained ResNet34 weights unavailable ({exc}). "
                      "Continuing with random initialization.")
                resnet = models.resnet34(weights=None)
        elif encoder_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            try:
                resnet = models.resnet50(weights=weights)
            except Exception as exc:
                if not pretrained:
                    raise
                print(f"Warning: pretrained ResNet50 weights unavailable ({exc}). "
                      "Continuing with random initialization.")
                resnet = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        # Extract encoder stages
        self.encoder0 = nn.Sequential(
            resnet.conv1,    # 3 → 64, stride=2, 512→256
            resnet.bn1,
            resnet.relu,
        )
        self.pool0 = resnet.maxpool  # stride=2, 256→128

        self.encoder1 = resnet.layer1  # 64 → 64,   128×128
        self.encoder2 = resnet.layer2  # 64 → 128,  64×64
        self.encoder3 = resnet.layer3  # 128 → 256, 32×32
        self.encoder4 = resnet.layer4  # 256 → 512, 16×16

        # Encoder channel sizes (ResNet34)
        if encoder_name == "resnet34":
            enc_channels = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet50":
            enc_channels = [64, 256, 512, 1024, 2048]

        # ── Bottleneck ──
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], enc_channels[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[4]),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──
        # Up4: bottleneck(512) + skip3(256) → 256
        self.decoder4 = DecoderBlock(enc_channels[4], enc_channels[3], decoder_channels[0])
        # Up3: 256 + skip2(128) → 128
        self.decoder3 = DecoderBlock(decoder_channels[0], enc_channels[2], decoder_channels[1])
        # Up2: 128 + skip1(64) → 64
        self.decoder2 = DecoderBlock(decoder_channels[1], enc_channels[1], decoder_channels[2])
        # Up1: 64 + skip0(64) → 32
        self.decoder1 = DecoderBlock(decoder_channels[2], enc_channels[0], decoder_channels[3])

        # ── Final upsample to original resolution ──
        self.final_upsample = nn.ConvTranspose2d(
            decoder_channels[3], decoder_channels[3], kernel_size=2, stride=2
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        # ── Output head ──
        self.output_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

        # Initialize decoder weights
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Initialize decoder with Kaiming normal (encoder is pretrained)."""
        for module in [self.bottleneck, self.decoder4, self.decoder3,
                       self.decoder2, self.decoder1, self.final_upsample,
                       self.final_conv, self.output_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W), H=W=512.

        Returns:
            Raw logits (B, 1, H, W). Apply sigmoid for probabilities.
        """
        # ── Encoder ──
        # x: (B, 3, 512, 512)
        e0 = self.encoder0(x)       # (B, 64, 256, 256)
        e0_pool = self.pool0(e0)    # (B, 64, 128, 128)

        e1 = self.encoder1(e0_pool) # (B, 64, 128, 128)
        e2 = self.encoder2(e1)      # (B, 128, 64, 64)
        e3 = self.encoder3(e2)      # (B, 256, 32, 32)
        e4 = self.encoder4(e3)      # (B, 512, 16, 16)

        # ── Bottleneck ──
        b = self.bottleneck(e4)     # (B, 512, 16, 16)

        # ── Decoder (with skip connections) ──
        d4 = self.decoder4(b, e3)   # (B, 256, 32, 32)
        d3 = self.decoder3(d4, e2)  # (B, 128, 64, 64)
        d2 = self.decoder2(d3, e1)  # (B, 64, 128, 128)
        d1 = self.decoder1(d2, e0)  # (B, 32, 256, 256)

        # ── Final upsample to 512×512 ──
        out = self.final_upsample(d1)  # (B, 32, 512, 512)
        out = self.final_conv(out)     # (B, 32, 512, 512)

        # ── Output ──
        logits = self.output_head(out) # (B, 1, 512, 512)

        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Inference: returns binary mask.

        Args:
            x: Input image tensor.
            threshold: Probability threshold for binarization.

        Returns:
            Binary mask (B, 1, H, W) with values in {0, 1}.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()

    def count_parameters(self) -> dict:
        """Count trainable parameters by component."""
        encoder_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "encoder" in name and p.requires_grad
        )
        decoder_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "decoder" in name or "bottleneck" in name or "final" in name or "output" in name
            and p.requires_grad
        )
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "total": total,
            "total_millions": total / 1e6,
        }
