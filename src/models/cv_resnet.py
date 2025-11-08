"""Complex-Valued ResNet for Fourier space denoising."""

import torch
import torch.nn as nn
from .base import BaseFourierModel, ComplexResidualBlock
from ..utils.complex_ops import ComplexConv3d, ComplexBatchNorm3d


class CVResNet(BaseFourierModel):
    """
    Complex-Valued ResNet - Baseline architecture.

    Direct end-to-end processing with complex residual blocks.
    Treats all frequencies the same (no frequency-specific processing).

    Architecture:
        Input [B, 64, 64, 33, 2]
        ↓
        Initial Projection: Conv3D [2] → [C]
        ↓
        Residual Blocks × N
        ↓
        Output Projection: Conv3D [C] → [2]
        ↓
        Output [B, 64, 64, 33, 2]

    Args:
        input_shape: Input dimensions (H, W, D_rfft)
        hidden_channels: Number of channels in residual blocks
        num_blocks: Number of residual blocks
        kernel_size: Convolution kernel size
        activation: Activation type ('crelu' or 'modrelu')
    """

    def __init__(
        self,
        input_shape=(64, 64, 33),
        hidden_channels=32,
        num_blocks=16,
        kernel_size=3,
        activation='crelu',
    ):
        super().__init__(input_shape, hidden_channels)

        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.activation = activation

        # Initial projection: 1 channel (complex = 2 real channels) -> hidden_channels
        self.input_proj = ComplexConv3d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ComplexResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output projection: hidden_channels -> 1 channel
        self.output_proj = ComplexConv3d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Fourier coefficients [B, H, W, D, 2]

        Returns:
            Denoised Fourier coefficients [B, H, W, D, 2]
        """
        B, H, W, D, _ = x.shape

        # Add channel dimension: [B, H, W, D, 2] -> [B, 1, H, W, D, 2]
        x = x.unsqueeze(1)

        # Rearrange to [B, C, H, W, D, 2] for processing
        # Initial projection
        x = self.input_proj(x)  # [B, C, H, W, D, 2]

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)  # [B, 1, H, W, D, 2]

        # Remove channel dimension: [B, 1, H, W, D, 2] -> [B, H, W, D, 2]
        x = x.squeeze(1)

        return x

    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                'num_blocks': self.num_blocks,
                'kernel_size': self.kernel_size,
                'activation': self.activation,
            }
        )
        return info
