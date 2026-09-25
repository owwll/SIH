"""
=============================================================================
NAVICORE AI: 1D TEMPORAL CONVOLUTIONAL NETWORK (TCN) VIRTUAL ODOMETER
Smart India Hackathon 2026 | Problem Statement ID: 260168
Theme: Smart Vehicles | Team: @enigm@ (Team ID: 132834)
=============================================================================
Exact architectural specification matching TRD.md §4.1 and AI_ARCHITECTURE.md §3.1:

Input: [Batch, 6 channels (Ax, Ay, Az, Gx, Gy, Gz), window_len timesteps]
  │
  ├──► Stem: Conv1D(in_channels=6, filters=32, k=7, padding=3, bias=False)
  │          + BatchNorm1D(32) + ReLU + SpatialDropout1D(p=0.1)
  │
  ├──► Residual Block 1 (Dilation=1):
  │     ├── Conv1D(32 -> 32, k=3, d=1, pad=1) -> BatchNorm1D -> ReLU
  │     ├── Conv1D(32 -> 32, k=3, d=1, pad=1) -> BatchNorm1D
  │     ├── Skip Connection (Identity)
  │     └── Squeeze-and-Excitation (SE Block, reduction=4) -> (+) -> ReLU
  │
  ├──► Residual Block 2 (Dilation=2):
  │     ├── Conv1D(32 -> 64, k=3, d=2, pad=2) -> BatchNorm1D -> ReLU
  │     ├── Conv1D(64 -> 64, k=3, d=2, pad=2) -> BatchNorm1D
  │     ├── Skip Connection (1x1 Conv 32 -> 64)
  │     └── Squeeze-and-Excitation (SE Block, reduction=4) -> (+) -> ReLU
  │
  ├──► Residual Block 3 (Dilation=4):
  │     ├── Conv1D(64 -> 128, k=3, d=4, pad=4) -> BatchNorm1D -> ReLU
  │     ├── Conv1D(128 -> 128, k=3, d=4, pad=4) -> BatchNorm1D
  │     ├── Skip Connection (1x1 Conv 64 -> 128)
  │     └── Squeeze-and-Excitation (SE Block, reduction=4) -> (+) -> ReLU
  │
  ├──► Global Average Pooling 1D (AdaptiveAvgPool1d(1))
  ├──► Dense Bottleneck (128 -> 64 units, ReLU) + Dropout(0.2)
  │
  ├──► Head A: Forward Speed Regression (Vx, scalar m/s)
  ├──► Head B: Speed Variance / Uncertainty (σ_v², Softplus > 0, adaptive covariance)
  └──► Head C: ZUPT / Idle Classifier Logit (Scalar logit for BCEWithLogitsLoss / P(stop))
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeAndExcitationBlock1D(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation Attention for 1D time series.
    TRD.md §4.1: SE Block with reduction ratio r=4.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, TimeSteps]
        b, c, _ = x.size()
        squeeze = x.mean(dim=2)  # [Batch, Channels]
        excitation = F.relu(self.fc1(squeeze), inplace=True)
        excitation = torch.sigmoid(self.fc2(excitation)).view(b, c, 1)
        return x * excitation


class DilatedResidualBlock1D(nn.Module):
    """
    Dilated 1D Convolutional Residual Block with Batch Normalization and SE Attention.
    TRD.md §4.1: Conv1D + BatchNorm -> Conv1D + BatchNorm -> SE -> Skip (+) -> ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_p: float = 0.1,
        se_reduction: int = 4,
    ):
        super().__init__()
        # Padding to preserve temporal sequence length: padding = dilation * (kernel_size - 1) // 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.se = SqueezeAndExcitationBlock1D(out_channels, reduction=se_reduction)

        # 1x1 projection for skip connection on blocks 2 and 3 where channel dimension expands
        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm1d(out_channels)
        else:
            self.skip_proj = nn.Identity()
            self.skip_bn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_bn(self.skip_proj(x))

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        return F.relu(out + residual, inplace=True)


class VirtualOdometerTCN(nn.Module):
    """
    1D-TCN Deep Virtual Odometer with Multi-Task Prediction Heads.
    TRD.md §4.1 & AI_ARCHITECTURE.md §3.1.

    Inputs:
      - x: [Batch, 6, window_len] (Ax, Ay, Az, Gx, Gy, Gz)
    Outputs (3 heads):
      - Head A: forward speed regression (Vx, scalar m/s) -> [Batch, 1]
      - Head B: speed variance / uncertainty (sigma_v^2, scalar > 0) -> [Batch, 1]
      - Head C: ZUPT / idle classifier logit -> [Batch, 1]
    """

    def __init__(
        self,
        in_channels: int = 6,
        stem_filters: int = 32,
        stem_kernel_size: int = 7,
        spatial_dropout_p: float = 0.1,
        bottleneck_units: int = 64,
        dropout_p: float = 0.2,
        se_reduction: int = 4,
    ):
        super().__init__()

        # Stem: Conv1D(32, k=7) + BatchNorm + SpatialDropout(0.1)
        self.stem_conv = nn.Conv1d(
            in_channels,
            stem_filters,
            kernel_size=stem_kernel_size,
            padding=stem_kernel_size // 2,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm1d(stem_filters)
        self.stem_relu = nn.ReLU(inplace=True)
        # Dropout1d drops entire channels across the time dimension (SpatialDropout1D)
        self.stem_spatial_dropout = nn.Dropout1d(p=spatial_dropout_p)

        # 3 Dilated Residual Blocks (dilation 1, 2, 4; channels 32 -> 64 -> 128)
        self.res1 = DilatedResidualBlock1D(
            stem_filters, 32, kernel_size=3, dilation=1, se_reduction=se_reduction
        )
        self.res2 = DilatedResidualBlock1D(
            32, 64, kernel_size=3, dilation=2, se_reduction=se_reduction
        )
        self.res3 = DilatedResidualBlock1D(
            64, 128, kernel_size=3, dilation=4, se_reduction=se_reduction
        )

        # Global Average Pooling 1D: dynamically aggregates across any temporal window length W
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Dense Bottleneck: Dense(64, ReLU) + Dropout(0.2)
        self.dense_bottleneck = nn.Sequential(
            nn.Linear(128, bottleneck_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # Three Output Heads:
        # Head A: Forward Longitudinal Speed Vx (m/s)
        self.head_vx = nn.Linear(bottleneck_units, 1)
        # Head B: Speed Variance sigma^2 (strictly positive uncertainty via Softplus)
        self.head_variance = nn.Linear(bottleneck_units, 1)
        # Head C: ZUPT / Idle Classifier Logit (unbounded logit for BCEWithLogitsLoss)
        self.head_zupt = nn.Linear(bottleneck_units, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: [Batch, 6, TimeSteps]
        :param return_prob: If True, applies sigmoid to Head C output for inference probability.
                            If False (default), returns raw logit for numerical stability with BCEWithLogitsLoss.
        :return: (vx [Batch, 1], variance [Batch, 1], zupt [Batch, 1])
        """
        # Stem
        feat = self.stem_conv(x)
        feat = self.stem_bn(feat)
        feat = self.stem_relu(feat)
        feat = self.stem_spatial_dropout(feat)

        # Dilated residual backbone
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)

        # Pooling & bottleneck
        pooled = self.global_pool(feat).squeeze(dim=2)  # [Batch, 128]
        bottleneck = self.dense_bottleneck(pooled)      # [Batch, 64]

        # Head A: Forward speed
        vx = self.head_vx(bottleneck)

        # Head B: Speed variance (strictly positive via Softplus + numerical epsilon)
        variance = F.softplus(self.head_variance(bottleneck)) + 1e-4

        # Head C: ZUPT / Idle classifier logit
        zupt_logit = self.head_zupt(bottleneck)
        zupt_out = torch.sigmoid(zupt_logit) if return_prob else zupt_logit

        return vx, variance, zupt_out

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method returning (vx, variance, zupt_probability) during mobile inference."""
        return self.forward(x, return_prob=True)
