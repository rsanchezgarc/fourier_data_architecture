"""Complex-valued operations for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ComplexConv3d(nn.Module):
    """
    Complex-valued 3D convolution.

    Implements true complex multiplication:
    (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Zero-padding added to all sides
        bias: If True, adds a learnable bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Real and imaginary kernels
        self.conv_real = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.conv_imag = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input [B, C_in, H, W, D, 2] where last dim is [real, imag]

        Returns:
            Complex output [B, C_out, H, W, D, 2]
        """
        real = x[..., 0]  # [B, C_in, H, W, D]
        imag = x[..., 1]  # [B, C_in, H, W, D]

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        return torch.stack([real_out, imag_out], dim=-1)


class ComplexBatchNorm3d(nn.Module):
    """
    Batch normalization for complex-valued data.
    Normalizes real and imaginary parts independently.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn_real = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input [B, C, H, W, D, 2]

        Returns:
            Normalized complex output [B, C, H, W, D, 2]
        """
        real = self.bn_real(x[..., 0])
        imag = self.bn_imag(x[..., 1])
        return torch.stack([real, imag], dim=-1)


def complex_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ReLU independently to real and imaginary parts.

    Args:
        x: Complex input [B, C, H, W, D, 2]

    Returns:
        ReLU-activated complex output [B, C, H, W, D, 2]
    """
    real = F.relu(x[..., 0])
    imag = F.relu(x[..., 1])
    return torch.stack([real, imag], dim=-1)


def mod_relu(x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    modReLU activation for complex numbers.

    modReLU(z) = ReLU(|z| + b) * z / |z|

    Preserves phase, applies ReLU to magnitude.

    Args:
        x: Complex input [B, C, H, W, D, 2]
        bias: Optional bias to add to magnitude [C]

    Returns:
        Activated complex output [B, C, H, W, D, 2]
    """
    real = x[..., 0]
    imag = x[..., 1]

    magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)  # Add epsilon for stability

    if bias is not None:
        # Expand bias to match dimensions
        bias = bias.view(1, -1, 1, 1, 1)
        magnitude = magnitude + bias

    # Apply ReLU to magnitude
    activated_magnitude = F.relu(magnitude)

    # Scale by activated magnitude / original magnitude (preserves phase)
    scale = activated_magnitude / (magnitude + 1e-8)

    real_out = real * scale
    imag_out = imag * scale

    return torch.stack([real_out, imag_out], dim=-1)


def complex_to_magnitude_phase(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert complex representation to magnitude and phase.

    Args:
        x: Complex input [B, C, H, W, D, 2] where last dim is [real, imag]

    Returns:
        magnitude: [B, C, H, W, D]
        phase: [B, C, H, W, D] in radians
    """
    real = x[..., 0]
    imag = x[..., 1]

    magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
    phase = torch.atan2(imag, real)

    return magnitude, phase


def magnitude_phase_to_complex(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Convert magnitude and phase to complex representation.

    Args:
        magnitude: [B, C, H, W, D]
        phase: [B, C, H, W, D] in radians

    Returns:
        Complex tensor [B, C, H, W, D, 2] where last dim is [real, imag]
    """
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)

    return torch.stack([real, imag], dim=-1)


def complex_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
) -> ComplexConv3d:
    """Factory function for creating complex 3D convolution layers."""
    return ComplexConv3d(in_channels, out_channels, kernel_size, stride, padding, bias)


def complex_batch_norm(num_features: int) -> ComplexBatchNorm3d:
    """Factory function for creating complex batch normalization layers."""
    return ComplexBatchNorm3d(num_features)
