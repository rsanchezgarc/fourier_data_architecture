"""Base model class for Fourier space architectures."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseFourierModel(nn.Module, ABC):
    """
    Abstract base class for Fourier space denoising models.

    All models take Fourier coefficients as input and output denoised coefficients.

    Input shape: [B, H, W, D, 2] where:
        - B: Batch size
        - H, W: Spatial dimensions (e.g., 64, 64)
        - D: rFFT dimension (e.g., 33 for 64x64x64 real FFT)
        - 2: Real and imaginary parts

    Output shape: Same as input [B, H, W, D, 2]
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (64, 64, 33),
        hidden_channels: int = 32,
    ):
        """
        Args:
            input_shape: Expected input dimensions (H, W, D_rfft)
            hidden_channels: Number of hidden channels for processing
        """
        super().__init__()
        self.input_shape = input_shape
        self.hidden_channels = hidden_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input Fourier coefficients [B, H, W, D, 2]

        Returns:
            Denoised Fourier coefficients [B, H, W, D, 2]
        """
        pass

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'name': self.__class__.__name__,
            'parameters': self.count_parameters(),
            'input_shape': self.input_shape,
            'hidden_channels': self.hidden_channels,
        }


class ComplexResidualBlock(nn.Module):
    """
    Residual block for complex-valued data.

    Architecture:
        x -> BatchNorm -> Activation -> Conv -> BatchNorm -> Activation -> Conv -> (+) -> out
        |_______________________________________________________________|
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        activation: str = 'crelu',
    ):
        super().__init__()

        from ..utils.complex_ops import ComplexConv3d, ComplexBatchNorm3d, complex_relu, mod_relu

        self.bn1 = ComplexBatchNorm3d(channels)
        self.conv1 = ComplexConv3d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn2 = ComplexBatchNorm3d(channels)
        self.conv2 = ComplexConv3d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.activation = complex_relu if activation == 'crelu' else mod_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input [B, C, H, W, D, 2]

        Returns:
            Complex output [B, C, H, W, D, 2]
        """
        residual = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        return out + residual


class ComplexDownsample(nn.Module):
    """Downsample complex data using strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        from ..utils.complex_ops import ComplexConv3d

        self.conv = ComplexConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ComplexUpsample(nn.Module):
    """Upsample complex data using transposed convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()

        # For complex upsampling, we need to handle real/imag separately
        self.conv_real = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1,
        )
        self.conv_imag = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input [B, C_in, H, W, D, 2]

        Returns:
            Complex output [B, C_out, 2*H, 2*W, 2*D, 2]
        """
        real = x[..., 0]
        imag = x[..., 1]

        # Complex multiplication for transposed conv
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        return torch.stack([real_out, imag_out], dim=-1)
