"""Frequency Shell CNN for 3D Fourier space denoising."""

import torch
import torch.nn as nn
import numpy as np
from .base import BaseFourierModel
from ..utils.complex_ops import ComplexConv3d, ComplexBatchNorm3d, complex_relu


class ShellEncoder(nn.Module):
    """
    Encoder for processing a single frequency shell.

    Each shell gets its own encoder with appropriate capacity.
    """

    def __init__(self, embedding_dim: int = 256, num_conv_layers: int = 3):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Build conv layers
        channels = [2, 16, 32, 64, embedding_dim]
        self.conv_layers = nn.ModuleList()

        for i in range(num_conv_layers):
            in_ch = channels[i] if i < len(channels) - 1 else channels[-2]
            out_ch = channels[i + 1] if i + 1 < len(channels) else channels[-1]

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shell coefficients [B, N_coeffs, 2] (irregular shape)
               or [B, 2, H, W, D] if scattered to grid

        Returns:
            Embedding [B, embedding_dim]
        """
        # Assume input is already scattered to grid: [B, 2, H, W, D]
        for layer in self.conv_layers:
            x = layer(x)

        # Pool to fixed size
        x = self.adaptive_pool(x)  # [B, embedding_dim, 1, 1, 1]
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, embedding_dim]

        return x


class ShellDecoder(nn.Module):
    """
    Decoder for reconstructing a single frequency shell from its embedding.
    """

    def __init__(self, embedding_dim: int = 256, target_size: tuple = (8, 8, 8)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.target_size = target_size

        # Initial projection
        self.initial_proj = nn.Linear(embedding_dim, embedding_dim * 2 * 2 * 2)

        # Upsampling convolutions
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 2, kernel_size=3, padding=1),  # Output real/imag
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shell embedding [B, embedding_dim]

        Returns:
            Reconstructed shell [B, 2, H, W, D]
        """
        B = x.shape[0]

        # Project to 3D
        x = self.initial_proj(x)  # [B, embedding_dim * 8]
        x = x.reshape(B, self.embedding_dim, 2, 2, 2)  # [B, embedding_dim, 2, 2, 2]

        # Upsample
        x = self.upsample(x)  # [B, 2, H, W, D]

        return x


class CrossShellFusion(nn.Module):
    """
    Fuses information across frequency shells.

    Uses 1D convolutions or transformer to mix shell embeddings.
    """

    def __init__(self, num_shells: int, embedding_dim: int):
        super().__init__()

        self.num_shells = num_shells
        self.embedding_dim = embedding_dim

        # Simple MLP-based fusion
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # Cross-shell attention (simplified)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=4, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, shell_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shell_embeddings: [B, num_shells, embedding_dim]

        Returns:
            Fused embeddings: [B, num_shells, embedding_dim]
        """
        # Self-attention across shells
        attn_out, _ = self.cross_attention(
            shell_embeddings, shell_embeddings, shell_embeddings
        )

        # Residual connection
        x = self.layer_norm(shell_embeddings + attn_out)

        # Per-shell MLP
        x = x + self.mlp(x)

        return x


class FSCNet(BaseFourierModel):
    """
    Frequency Shell CNN - Groups Fourier coefficients by frequency magnitude.

    Core innovation: Process different frequency ranges with different networks.

    Architecture:
        1. Assign each coefficient to a shell based on |k|
        2. Process each shell with dedicated encoder
        3. Fuse shell embeddings (cross-shell attention)
        4. Decode each shell separately
        5. Reconstruct full Fourier space

    Args:
        input_shape: Input dimensions (H, W, D_rfft)
        num_shells: Number of frequency shells
        embedding_dim: Embedding dimension per shell
        shell_spacing: 'linear', 'log', or 'quantile'
    """

    def __init__(
        self,
        input_shape=(64, 64, 33),
        hidden_channels=32,
        num_shells=16,
        embedding_dim=256,
        shell_spacing='log',
    ):
        super().__init__(input_shape, hidden_channels)

        self.num_shells = num_shells
        self.embedding_dim = embedding_dim
        self.shell_spacing = shell_spacing

        # Precompute shell boundaries
        self.register_buffer('shell_bounds', self._create_shell_boundaries())

        # Precompute shell assignments for the input grid
        self.register_buffer('shell_masks', self._create_shell_masks())

        # Per-shell encoders (different networks for different shells)
        self.shell_encoders = nn.ModuleList(
            [ShellEncoder(embedding_dim, num_conv_layers=3) for _ in range(num_shells)]
        )

        # Cross-shell fusion
        self.fusion = CrossShellFusion(num_shells, embedding_dim)

        # Per-shell decoders
        self.shell_decoders = nn.ModuleList(
            [ShellDecoder(embedding_dim, target_size=(8, 8, 8)) for _ in range(num_shells)]
        )

    def _create_shell_boundaries(self) -> torch.Tensor:
        """Create shell boundaries based on spacing strategy."""
        H, W, D = self.input_shape
        max_k = np.sqrt((H // 2) ** 2 + (W // 2) ** 2 + D**2)

        if self.shell_spacing == 'linear':
            bounds = torch.linspace(0, max_k, self.num_shells + 1)
        elif self.shell_spacing == 'log':
            # Logarithmic spacing (more shells at low frequencies)
            log_bounds = torch.linspace(0, np.log(max_k + 1), self.num_shells + 1)
            bounds = torch.exp(log_bounds) - 1
        else:  # quantile
            bounds = torch.linspace(0, max_k, self.num_shells + 1)

        return bounds

    def _create_shell_masks(self) -> torch.Tensor:
        """
        Precompute which shell each Fourier coefficient belongs to.

        Returns:
            shell_masks: [num_shells, H, W, D] boolean masks
        """
        H, W, D = self.input_shape

        # Create frequency grid
        kx = torch.arange(H) - H // 2
        ky = torch.arange(W) - W // 2
        kz = torch.arange(D)  # rfft only has positive frequencies in z

        # Meshgrid
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')

        # Compute frequency magnitude for each point
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

        # Assign to shells
        shell_masks = []
        for i in range(self.num_shells):
            lower = self.shell_bounds[i]
            upper = self.shell_bounds[i + 1]
            mask = (k_mag >= lower) & (k_mag < upper)
            shell_masks.append(mask)

        return torch.stack(shell_masks, dim=0)  # [num_shells, H, W, D]

    def scatter_shell_to_grid(
        self, coeffs: torch.Tensor, shell_mask: torch.Tensor, target_size: int = 16
    ) -> torch.Tensor:
        """
        Scatter irregular shell coefficients to a regular grid.

        Args:
            coeffs: Full Fourier coefficients [B, H, W, D, 2]
            shell_mask: Boolean mask [H, W, D]
            target_size: Target grid size

        Returns:
            Grid [B, 2, target_size, target_size, target_size]
        """
        B = coeffs.shape[0]

        # Extract shell coefficients
        # Rearrange to [B, 2, H, W, D]
        coeffs_reorg = coeffs.permute(0, 4, 1, 2, 3)  # [B, 2, H, W, D]

        # For simplicity, use adaptive pooling
        # In practice, you might want more sophisticated scattering
        shell_coeffs = coeffs_reorg * shell_mask.unsqueeze(0).unsqueeze(0)

        # Interpolate to target size
        shell_grid = torch.nn.functional.interpolate(
            shell_coeffs, size=(target_size, target_size, target_size), mode='trilinear'
        )

        return shell_grid

    def gather_grid_to_shell(
        self,
        grid: torch.Tensor,
        shell_mask: torch.Tensor,
        original_shape: tuple,
    ) -> torch.Tensor:
        """
        Gather from regular grid back to original shell positions.

        Args:
            grid: [B, 2, target_size, target_size, target_size]
            shell_mask: [H, W, D]
            original_shape: (H, W, D)

        Returns:
            Coefficients at shell positions [B, H, W, D, 2]
        """
        B = grid.shape[0]
        H, W, D = original_shape

        # Interpolate back to original size
        coeffs = torch.nn.functional.interpolate(grid, size=(H, W, D), mode='trilinear')

        # Apply mask (only keep shell coefficients)
        coeffs = coeffs * shell_mask.unsqueeze(0).unsqueeze(0)

        # Rearrange back to [B, H, W, D, 2]
        coeffs = coeffs.permute(0, 2, 3, 4, 1)

        return coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Fourier coefficients [B, H, W, D, 2]

        Returns:
            Denoised Fourier coefficients [B, H, W, D, 2]
        """
        B, H, W, D, _ = x.shape

        # Process each shell
        shell_embeddings = []
        for shell_idx in range(self.num_shells):
            # Scatter shell coefficients to grid
            shell_grid = self.scatter_shell_to_grid(
                x, self.shell_masks[shell_idx], target_size=16
            )

            # Encode
            embedding = self.shell_encoders[shell_idx](shell_grid)  # [B, embedding_dim]
            shell_embeddings.append(embedding)

        # Stack shell embeddings
        shell_embeddings = torch.stack(shell_embeddings, dim=1)  # [B, num_shells, embedding_dim]

        # Cross-shell fusion
        fused_embeddings = self.fusion(shell_embeddings)  # [B, num_shells, embedding_dim]

        # Decode each shell and reconstruct
        output = torch.zeros_like(x)
        for shell_idx in range(self.num_shells):
            # Decode shell
            shell_grid = self.shell_decoders[shell_idx](
                fused_embeddings[:, shell_idx]
            )  # [B, 2, H', W', D']

            # Gather back to original positions
            shell_coeffs = self.gather_grid_to_shell(
                shell_grid, self.shell_masks[shell_idx], (H, W, D)
            )

            # Add to output
            output = output + shell_coeffs

        return output

    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                'num_shells': self.num_shells,
                'embedding_dim': self.embedding_dim,
                'shell_spacing': self.shell_spacing,
            }
        )
        return info
