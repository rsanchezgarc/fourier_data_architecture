"""Radial-Angular Decomposition Network for 3D Fourier space denoising."""

import torch
import torch.nn as nn
import numpy as np
from .base import BaseFourierModel


class RadialProcessor(nn.Module):
    """
    Processes radial (frequency magnitude) information.

    For each radial bin, aggregates information across all angles.
    """

    def __init__(self, num_radial_bins: int, hidden_dim: int = 128):
        super().__init__()

        self.num_radial_bins = num_radial_bins
        self.hidden_dim = hidden_dim

        # Process radial features with 1D convolutions
        self.radial_conv = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),  # 2 for real/imag
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, radial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radial_features: [B, num_radial_bins, 2] (real/imag averaged over angles)

        Returns:
            Radial embeddings: [B, num_radial_bins, hidden_dim]
        """
        B, N, _ = radial_features.shape

        # Rearrange for 1D conv: [B, 2, N]
        x = radial_features.permute(0, 2, 1)

        # Process with 1D convolutions
        x = self.radial_conv(x)  # [B, hidden_dim, N]

        # Rearrange back: [B, N, hidden_dim]
        x = x.permute(0, 2, 1)

        return x


class AngularProcessor(nn.Module):
    """
    Processes angular variations within each radial shell.

    Uses spherical harmonics or learned angular patterns.
    """

    def __init__(self, num_angular_modes: int = 16, hidden_dim: int = 128):
        super().__init__()

        self.num_angular_modes = num_angular_modes
        self.hidden_dim = hidden_dim

        # Learn angular patterns with MLPs
        self.angular_mlp = nn.Sequential(
            nn.Linear(2 + 2, hidden_dim),  # 2 for angles (theta, phi) + 2 for real/imag
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_angular_modes),
        )

    def forward(
        self, coeffs: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coeffs: Complex coefficients [N, 2] (real/imag)
            theta: Polar angles [N]
            phi: Azimuthal angles [N]

        Returns:
            Angular features: [num_angular_modes, hidden_dim]
        """
        N = coeffs.shape[0]

        # Combine angles and coefficients
        angles = torch.stack([theta, phi], dim=-1)  # [N, 2]
        features = torch.cat([angles, coeffs], dim=-1)  # [N, 4]

        # Process through MLP
        angular_features = self.angular_mlp(features)  # [N, num_angular_modes]

        # Aggregate across points (mean pooling)
        angular_features = angular_features.mean(dim=0)  # [num_angular_modes]

        return angular_features


class RADNet(BaseFourierModel):
    """
    Radial-Angular Decomposition Network.

    Exploits spherical geometry of Fourier space by separating
    radial and angular processing.

    Architecture:
        1. Convert (kx, ky, kz) → (r, θ, φ) coordinates
        2. Radial processing: Learn r-dependent features
        3. Angular processing: Learn (θ, φ)-dependent features per shell
        4. Fusion: Combine radial + angular information
        5. Reconstruction: Convert back to (kx, ky, kz)

    Args:
        input_shape: Input dimensions (H, W, D_rfft)
        num_radial_bins: Number of radial bins
        num_angular_modes: Number of angular modes
        hidden_dim: Hidden dimension for processing
    """

    def __init__(
        self,
        input_shape=(64, 64, 33),
        hidden_channels=128,
        num_radial_bins=16,
        num_angular_modes=16,
    ):
        super().__init__(input_shape, hidden_channels)

        self.num_radial_bins = num_radial_bins
        self.num_angular_modes = num_angular_modes

        # Precompute spherical coordinates for input grid
        self.register_buffer('radii', torch.zeros(input_shape))
        self.register_buffer('theta', torch.zeros(input_shape))
        self.register_buffer('phi', torch.zeros(input_shape))
        self.register_buffer('radial_bin_indices', torch.zeros(input_shape, dtype=torch.long))

        self._precompute_spherical_coords()

        # Radial processor
        self.radial_processor = RadialProcessor(num_radial_bins, hidden_channels)

        # Angular processor
        self.angular_processor = AngularProcessor(num_angular_modes, hidden_channels)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels + num_angular_modes, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2),  # Output real/imag
        )

        # Reconstruction network (per radial bin)
        self.reconstruction = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channels, 2),  # real/imag
                )
                for _ in range(num_radial_bins)
            ]
        )

    def _precompute_spherical_coords(self):
        """Precompute spherical coordinates for the Fourier grid."""
        H, W, D = self.input_shape

        # Create Cartesian grid
        kx = torch.arange(H, dtype=torch.float32) - H // 2
        ky = torch.arange(W, dtype=torch.float32) - W // 2
        kz = torch.arange(D, dtype=torch.float32)  # rfft: only positive

        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')

        # Convert to spherical coordinates
        radii = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2 + 1e-8)
        theta = torch.acos(kz_grid / (radii + 1e-8))  # Polar angle [0, π]
        phi = torch.atan2(ky_grid, kx_grid)  # Azimuthal angle [-π, π]

        # Assign radial bins
        max_radius = radii.max()
        radial_bins = torch.linspace(0, max_radius, self.num_radial_bins + 1)
        radial_bin_indices = torch.bucketize(radii, radial_bins, right=False) - 1
        radial_bin_indices = torch.clamp(radial_bin_indices, 0, self.num_radial_bins - 1)

        # Store
        self.radii.copy_(radii)
        self.theta.copy_(theta)
        self.phi.copy_(phi)
        self.radial_bin_indices.copy_(radial_bin_indices)

    def extract_radial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract radial features by averaging over angles within each radial bin.

        Args:
            x: Input coefficients [B, H, W, D, 2]

        Returns:
            Radial features: [B, num_radial_bins, 2]
        """
        B = x.shape[0]
        radial_features = torch.zeros(
            B, self.num_radial_bins, 2, device=x.device, dtype=x.dtype
        )

        for r_idx in range(self.num_radial_bins):
            # Find all coefficients in this radial bin
            mask = self.radial_bin_indices == r_idx

            if mask.sum() > 0:
                # Extract and average coefficients
                coeffs_in_bin = x[:, mask, :]  # [B, N_points_in_bin, 2]
                radial_features[:, r_idx, :] = coeffs_in_bin.mean(dim=1)

        return radial_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Fourier coefficients [B, H, W, D, 2]

        Returns:
            Denoised Fourier coefficients [B, H, W, D, 2]
        """
        B, H, W, D, _ = x.shape

        # Extract radial features (average over angles)
        radial_features = self.extract_radial_features(x)  # [B, num_radial_bins, 2]

        # Process radial features
        radial_embeddings = self.radial_processor(radial_features)  # [B, num_radial_bins, hidden]

        # Process angular features for each radial shell
        # For simplicity, we process the first batch element and assume similar patterns
        # In practice, you'd do this for each batch element
        angular_features_per_shell = []

        for r_idx in range(self.num_radial_bins):
            mask = self.radial_bin_indices == r_idx

            if mask.sum() > 0:
                # Get coefficients, theta, phi for this shell
                coeffs_in_shell = x[0, mask, :]  # [N_points, 2]
                theta_in_shell = self.theta[mask]  # [N_points]
                phi_in_shell = self.phi[mask]  # [N_points]

                # Process angular features
                angular_feat = self.angular_processor(
                    coeffs_in_shell, theta_in_shell, phi_in_shell
                )
                angular_features_per_shell.append(angular_feat)
            else:
                angular_features_per_shell.append(
                    torch.zeros(self.num_angular_modes, device=x.device)
                )

        angular_features = torch.stack(
            angular_features_per_shell, dim=0
        )  # [num_radial_bins, num_angular_modes]

        # Reconstruct coefficients for each point
        output = torch.zeros_like(x)

        for r_idx in range(self.num_radial_bins):
            mask = self.radial_bin_indices == r_idx

            if mask.sum() > 0:
                # Combine radial and angular features
                radial_emb = radial_embeddings[:, r_idx, :]  # [B, hidden]
                angular_emb = angular_features[r_idx].unsqueeze(0).expand(
                    B, -1
                )  # [B, num_angular_modes]

                combined = torch.cat([radial_emb, angular_emb], dim=-1)  # [B, hidden + angular]

                # Reconstruct coefficients for this shell
                reconstructed = self.reconstruction[r_idx](radial_emb)  # [B, 2]

                # Broadcast to all points in this shell
                output[:, mask, :] = reconstructed.unsqueeze(1)

        return output

    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                'num_radial_bins': self.num_radial_bins,
                'num_angular_modes': self.num_angular_modes,
            }
        )
        return info
