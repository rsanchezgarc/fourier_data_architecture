"""Fourier Neural Operator for 3D Fourier space denoising."""

import torch
import torch.nn as nn
from .base import BaseFourierModel


class SpectralConv3d(nn.Module):
    """
    Spectral Convolution layer for Fourier Neural Operator.

    Learns frequency-specific transformations by multiplying Fourier modes
    with learned weight matrices.

    For each frequency mode k, applies: v_out(k) = R(k) @ v_in(k)
    where R(k) is a learnable C_in × C_out complex matrix.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes_x, modes_y, modes_z: Number of Fourier modes to use
            (truncated to low frequencies for efficiency)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int = 12,
        modes_y: int = 12,
        modes_z: int = 12,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        # Learnable weights for each frequency mode (complex-valued)
        # Shape: [in_channels, out_channels, modes_x, modes_y, modes_z]
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, modes_y, modes_z) * scale
        )
        self.weights_imag = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, modes_y, modes_z) * scale
        )

    def complex_multiply_3d(self, input_real, input_imag, weights_real, weights_imag):
        """
        Complex multiplication in Fourier space.

        (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

        Args:
            input_real, input_imag: [B, C_in, modes_x, modes_y, modes_z]
            weights_real, weights_imag: [C_in, C_out, modes_x, modes_y, modes_z]

        Returns:
            output_real, output_imag: [B, C_out, modes_x, modes_y, modes_z]
        """
        # Einstein summation for matrix multiply over channel dimension
        # b=batch, i=input_channels, o=output_channels, x,y,z=spatial modes
        out_real = torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_real) - torch.einsum(
            "bixyz,ioxyz->boxyz", input_imag, weights_imag
        )

        out_imag = torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_imag) + torch.einsum(
            "bixyz,ioxyz->boxyz", input_imag, weights_real
        )

        return out_real, out_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input in Fourier space [B, C_in, H, W, D, 2]

        Returns:
            Output in Fourier space [B, C_out, H, W, D, 2]
        """
        B, C_in, H, W, D, _ = x.shape

        # Extract real and imaginary parts
        x_real = x[..., 0]  # [B, C_in, H, W, D]
        x_imag = x[..., 1]

        # Initialize output
        out_real = torch.zeros(B, self.out_channels, H, W, D, device=x.device, dtype=x.dtype)
        out_imag = torch.zeros(B, self.out_channels, H, W, D, device=x.device, dtype=x.dtype)

        # Only process low-frequency modes (for computational efficiency)
        # Top-left corner of Fourier space
        mx, my, mz = min(self.modes_x, H), min(self.modes_y, W), min(self.modes_z, D)

        x_real_modes = x_real[:, :, :mx, :my, :mz]
        x_imag_modes = x_imag[:, :, :mx, :my, :mz]

        weights_real = self.weights_real[:, :, :mx, :my, :mz]
        weights_imag = self.weights_imag[:, :, :mx, :my, :mz]

        # Apply spectral convolution
        out_real_modes, out_imag_modes = self.complex_multiply_3d(
            x_real_modes, x_imag_modes, weights_real, weights_imag
        )

        # Place back in output (only low frequencies are modified)
        out_real[:, :, :mx, :my, :mz] = out_real_modes
        out_imag[:, :, :mx, :my, :mz] = out_imag_modes

        return torch.stack([out_real, out_imag], dim=-1)


class FNOLayer(nn.Module):
    """
    Single FNO layer combining spectral convolution and local features.

    Architecture:
        x -> SpectralConv (global) + Conv1x1x1 (local) -> activation -> out
    """

    def __init__(
        self,
        channels: int,
        modes_x: int = 12,
        modes_y: int = 12,
        modes_z: int = 12,
    ):
        super().__init__()

        self.spectral_conv = SpectralConv3d(channels, channels, modes_x, modes_y, modes_z)

        # Local features via 1x1x1 conv (pointwise)
        # We treat real/imag as separate channels for simplicity
        self.local_conv = nn.Conv3d(channels * 2, channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W, D, 2]

        Returns:
            Output [B, C, H, W, D, 2]
        """
        B, C, H, W, D, _ = x.shape

        # Branch 1: Spectral convolution (global receptive field)
        x1 = self.spectral_conv(x)

        # Branch 2: Local convolution (pointwise channel mixing)
        # Reshape to treat real/imag as channels
        x_local = x.reshape(B, C * 2, H, W, D)  # [B, C*2, H, W, D]
        x2 = self.local_conv(x_local)  # [B, C*2, H, W, D]
        x2 = x2.reshape(B, C, H, W, D, 2)  # [B, C, H, W, D, 2]

        # Combine and activate
        out = x1 + x2
        # Apply activation separately to real/imag
        out_real = torch.nn.functional.gelu(out[..., 0])
        out_imag = torch.nn.functional.gelu(out[..., 1])
        out = torch.stack([out_real, out_imag], dim=-1)

        return out


class FNO3D(BaseFourierModel):
    """
    Fourier Neural Operator for 3D denoising.

    Operates entirely in Fourier space with frequency-specific learned kernels.
    Never converts back to real space (except for evaluation).

    Architecture:
        Input [B, 64, 64, 33, 2] (already in Fourier space!)
        ↓
        Lifting: Project to higher dimensions [2] → [v]
        ↓
        FNO Layers × N (spectral conv + local conv)
        ↓
        Projection: [v] → [2]
        ↓
        Output [B, 64, 64, 33, 2]

    Args:
        input_shape: Input dimensions (H, W, D_rfft)
        hidden_channels: Number of channels (v in FNO paper)
        num_layers: Number of FNO layers
        modes_x, modes_y, modes_z: Truncated Fourier modes to use
    """

    def __init__(
        self,
        input_shape=(64, 64, 33),
        hidden_channels=32,
        num_layers=4,
        modes_x=12,
        modes_y=12,
        modes_z=12,
    ):
        super().__init__(input_shape, hidden_channels)

        self.num_layers = num_layers
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        # Lifting layer: Project from 2 real channels (real/imag) to hidden_channels
        self.lifting = nn.Conv3d(2, hidden_channels * 2, kernel_size=1)

        # FNO layers
        self.fno_layers = nn.ModuleList(
            [FNOLayer(hidden_channels, modes_x, modes_y, modes_z) for _ in range(num_layers)]
        )

        # Projection layer: hidden_channels back to 2 (real/imag)
        self.projection = nn.Conv3d(hidden_channels * 2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Fourier coefficients [B, H, W, D, 2]

        Returns:
            Denoised Fourier coefficients [B, H, W, D, 2]
        """
        B, H, W, D, _ = x.shape

        # Rearrange to [B, 2, H, W, D] for lifting
        x = x.permute(0, 4, 1, 2, 3)  # [B, 2, H, W, D]

        # Lifting
        x = self.lifting(x)  # [B, C*2, H, W, D]

        # Reshape to [B, C, H, W, D, 2]
        x = x.reshape(B, self.hidden_channels, H, W, D, 2)

        # FNO layers
        for layer in self.fno_layers:
            x = layer(x)

        # Reshape for projection
        x = x.reshape(B, self.hidden_channels * 2, H, W, D)

        # Projection
        x = self.projection(x)  # [B, 2, H, W, D]

        # Rearrange back to [B, H, W, D, 2]
        x = x.permute(0, 2, 3, 4, 1)

        return x

    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                'num_layers': self.num_layers,
                'modes': (self.modes_x, self.modes_y, self.modes_z),
            }
        )
        return info
