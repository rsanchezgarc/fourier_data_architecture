"""Frequency-Grouped Transformer for 3D Fourier space denoising."""

import torch
import torch.nn as nn
import numpy as np
from .base import BaseFourierModel


class FrequencyBandTokenizer(nn.Module):
    """
    Tokenizes a frequency band into a fixed number of tokens.

    Uses adaptive pooling to create uniform representation.
    """

    def __init__(self, tokens_per_band: int = 256, token_dim: int = 64):
        super().__init__()

        self.tokens_per_band = tokens_per_band
        self.token_dim = token_dim

        # Projection to token dimension
        self.projection = nn.Linear(2, token_dim)  # 2 for real/imag

        # Adaptive pooling to fixed number of tokens
        # We'll use a learned pooling approach
        self.token_pool = nn.AdaptiveAvgPool1d(tokens_per_band)

    def forward(self, band_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_coeffs: [B, H, W, D_band, 2]

        Returns:
            Tokens: [B, tokens_per_band, token_dim]
        """
        B, H, W, D, _ = band_coeffs.shape

        # Flatten spatial dimensions
        coeffs_flat = band_coeffs.reshape(B, H * W * D, 2)  # [B, N, 2]

        # Project to token dimension
        tokens = self.projection(coeffs_flat)  # [B, N, token_dim]

        # Pool to fixed number of tokens
        # Transpose for adaptive pooling
        tokens = tokens.permute(0, 2, 1)  # [B, token_dim, N]
        tokens = self.token_pool(tokens)  # [B, token_dim, tokens_per_band]
        tokens = tokens.permute(0, 2, 1)  # [B, tokens_per_band, token_dim]

        return tokens


class FrequencyBandDetokenizer(nn.Module):
    """
    Reconstructs frequency band from tokens.
    """

    def __init__(
        self, tokens_per_band: int = 256, token_dim: int = 64, output_shape: tuple = (64, 64, 8)
    ):
        super().__init__()

        self.tokens_per_band = tokens_per_band
        self.token_dim = token_dim
        self.output_shape = output_shape

        H, W, D = output_shape
        total_size = H * W * D

        # Upsample tokens to output size
        self.upsample = nn.Linear(tokens_per_band, total_size)

        # Project back to real/imag
        self.output_proj = nn.Linear(token_dim, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, tokens_per_band, token_dim]

        Returns:
            Reconstructed band: [B, H, W, D, 2]
        """
        B = tokens.shape[0]
        H, W, D = self.output_shape

        # Upsample tokens
        # [B, tokens_per_band, token_dim] -> [B, token_dim, tokens_per_band]
        tokens_t = tokens.permute(0, 2, 1)

        # Upsample to output size
        upsampled = self.upsample(tokens_t)  # [B, token_dim, H*W*D]

        # Transpose back
        upsampled = upsampled.permute(0, 2, 1)  # [B, H*W*D, token_dim]

        # Project to real/imag
        coeffs = self.output_proj(upsampled)  # [B, H*W*D, 2]

        # Reshape to 3D
        coeffs = coeffs.reshape(B, H, W, D, 2)

        return coeffs


class WithinBandTransformer(nn.Module):
    """
    Transformer for processing tokens within a single frequency band.
    """

    def __init__(self, token_dim: int = 64, num_layers: int = 2, num_heads: int = 4):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N_tokens, token_dim]

        Returns:
            Processed tokens: [B, N_tokens, token_dim]
        """
        return self.transformer(tokens)


class CrossBandTransformer(nn.Module):
    """
    Transformer for exchanging information across frequency bands.
    """

    def __init__(self, num_bands: int, token_dim: int = 64, num_heads: int = 4):
        super().__init__()

        self.num_bands = num_bands

        # Cross-attention between bands
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=num_heads, batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(token_dim)
        self.layer_norm2 = nn.LayerNorm(token_dim)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
        )

    def forward(self, band_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_tokens: [B, num_bands, N_tokens_per_band, token_dim]

        Returns:
            Cross-attended tokens: [B, num_bands, N_tokens_per_band, token_dim]
        """
        B, num_bands, N_tokens, token_dim = band_tokens.shape

        # Reshape for cross-band attention
        # Treat each band's tokens as a sequence
        tokens_flat = band_tokens.reshape(B, num_bands * N_tokens, token_dim)

        # Self-attention across all tokens (from all bands)
        attn_out, _ = self.cross_attention(tokens_flat, tokens_flat, tokens_flat)

        # Residual and norm
        tokens_flat = self.layer_norm1(tokens_flat + attn_out)

        # Feedforward
        ffn_out = self.ffn(tokens_flat)
        tokens_flat = self.layer_norm2(tokens_flat + ffn_out)

        # Reshape back
        band_tokens = tokens_flat.reshape(B, num_bands, N_tokens, token_dim)

        return band_tokens


class FrequencyGroupedTransformer(BaseFourierModel):
    """
    Frequency-Grouped Transformer (FGT).

    Groups frequencies into bands, tokenizes each band, and uses
    attention mechanisms for processing.

    Architecture:
        1. Split Fourier space into frequency bands (by |k|)
        2. Tokenize each band → fixed number of tokens
        3. Within-band transformer: Process each band independently
        4. Cross-band transformer: Exchange information between bands
        5. Detokenize: Reconstruct each band
        6. Concatenate bands → output

    Args:
        input_shape: Input dimensions (H, W, D_rfft)
        num_bands: Number of frequency bands
        tokens_per_band: Number of tokens per band
        token_dim: Dimension of each token
        num_within_layers: Number of within-band transformer layers
        num_cross_layers: Number of cross-band transformer layers
    """

    def __init__(
        self,
        input_shape=(64, 64, 33),
        hidden_channels=64,
        num_bands=16,
        tokens_per_band=256,
        num_within_layers=2,
        num_cross_layers=1,
        num_heads=4,
    ):
        super().__init__(input_shape, hidden_channels)

        self.num_bands = num_bands
        self.tokens_per_band = tokens_per_band
        self.token_dim = hidden_channels

        H, W, D = input_shape

        # Precompute frequency band assignments
        self.register_buffer('band_masks', self._create_band_masks())

        # Calculate output shapes for each band
        self.band_shapes = []
        for i in range(num_bands):
            mask = self.band_masks[i]
            # For simplicity, we'll use fixed shapes based on input_shape
            # In practice, you'd calculate based on actual mask
            band_depth = D // num_bands
            self.band_shapes.append((H, W, band_depth))

        # Tokenizers and detokenizers for each band
        self.tokenizers = nn.ModuleList(
            [FrequencyBandTokenizer(tokens_per_band, self.token_dim) for _ in range(num_bands)]
        )

        self.detokenizers = nn.ModuleList(
            [
                FrequencyBandDetokenizer(tokens_per_band, self.token_dim, shape)
                for shape in self.band_shapes
            ]
        )

        # Within-band transformers
        self.within_band_transformers = nn.ModuleList(
            [
                WithinBandTransformer(self.token_dim, num_within_layers, num_heads)
                for _ in range(num_bands)
            ]
        )

        # Cross-band transformer
        self.cross_band_transformer = CrossBandTransformer(num_bands, self.token_dim, num_heads)

    def _create_band_masks(self) -> torch.Tensor:
        """
        Create masks for frequency bands based on |k|.

        Returns:
            Band masks: [num_bands, H, W, D]
        """
        H, W, D = self.input_shape

        # Create frequency grid
        kx = torch.arange(H, dtype=torch.float32) - H // 2
        ky = torch.arange(W, dtype=torch.float32) - W // 2
        kz = torch.arange(D, dtype=torch.float32)

        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')

        # Compute frequency magnitude
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

        # Create band boundaries (logarithmic spacing)
        max_k = k_mag.max()
        log_bounds = torch.linspace(0, np.log(max_k + 1), self.num_bands + 1)
        bounds = torch.exp(log_bounds) - 1

        # Assign to bands
        band_masks = []
        for i in range(self.num_bands):
            lower = bounds[i]
            upper = bounds[i + 1]
            mask = (k_mag >= lower) & (k_mag < upper)
            band_masks.append(mask)

        return torch.stack(band_masks, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Fourier coefficients [B, H, W, D, 2]

        Returns:
            Denoised Fourier coefficients [B, H, W, D, 2]
        """
        B, H, W, D, _ = x.shape

        # Split into frequency bands and tokenize
        band_tokens = []
        band_inputs = []

        for i in range(self.num_bands):
            # Extract band using mask
            mask = self.band_masks[i].unsqueeze(0).unsqueeze(-1)  # [1, H, W, D, 1]
            band_coeffs = x * mask  # [B, H, W, D, 2]

            # For tokenization, we need to extract non-zero regions
            # For simplicity, we use the full grid
            band_inputs.append(band_coeffs)

            # Tokenize (using full grid for now)
            tokens = self.tokenizers[i](band_coeffs)  # [B, tokens_per_band, token_dim]
            band_tokens.append(tokens)

        # Stack band tokens
        band_tokens = torch.stack(band_tokens, dim=1)  # [B, num_bands, tokens_per_band, token_dim]

        # Within-band processing
        processed_bands = []
        for i in range(self.num_bands):
            processed = self.within_band_transformers[i](
                band_tokens[:, i]
            )  # [B, tokens_per_band, token_dim]
            processed_bands.append(processed)

        processed_bands = torch.stack(
            processed_bands, dim=1
        )  # [B, num_bands, tokens_per_band, token_dim]

        # Cross-band processing
        cross_processed = self.cross_band_transformer(
            processed_bands
        )  # [B, num_bands, tokens_per_band, token_dim]

        # Detokenize each band
        output = torch.zeros_like(x)
        for i in range(self.num_bands):
            # Detokenize
            band_reconstructed = self.detokenizers[i](
                cross_processed[:, i]
            )  # [B, H', W', D', 2]

            # Resize to match original band shape if needed
            if band_reconstructed.shape[1:4] != (H, W, D // self.num_bands):
                # Interpolate to correct size
                band_reconstructed = torch.nn.functional.interpolate(
                    band_reconstructed.permute(0, 4, 1, 2, 3),
                    size=(H, W, D // self.num_bands),
                    mode='trilinear',
                )
                band_reconstructed = band_reconstructed.permute(0, 2, 3, 4, 1)

            # Apply mask and add to output
            mask = self.band_masks[i].unsqueeze(0).unsqueeze(-1)
            output = output + band_reconstructed * mask

        return output

    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                'num_bands': self.num_bands,
                'tokens_per_band': self.tokens_per_band,
                'token_dim': self.token_dim,
            }
        )
        return info
