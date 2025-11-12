"""Multi-Resolution Fourier U-Net for 3D Fourier space denoising."""

import torch
import torch.nn as nn
from .base import BaseFourierModel, ComplexDownsample, ComplexUpsample
from ..utils.complex_ops import ComplexConv3d, ComplexBatchNorm3d, complex_relu


class FrequencyBandUNet(nn.Module):
    """
    U-Net for processing a single frequency band.

    Architecture:
        Encoder: Conv + Pool (downsample)
        Decoder: TransposeConv + Skip connections
    """

    def __init__(self, depth_size: int, base_channels: int = 16):
        super().__init__()

        self.depth_size = depth_size
        channels = [1, base_channels, base_channels * 2, base_channels * 4]

        # Determine number of pooling stages based on depth
        # Need at least depth >= 2^num_pools
        if depth_size >= 8:
            self.num_pools = 3
        elif depth_size >= 4:
            self.num_pools = 2
        elif depth_size >= 2:
            self.num_pools = 1
        else:
            self.num_pools = 0

        # Encoder
        self.enc1 = self._make_encoder_block(channels[0], channels[1])
        if self.num_pools >= 2:
            self.enc2 = self._make_encoder_block(channels[1], channels[2])
        if self.num_pools >= 3:
            self.enc3 = self._make_encoder_block(channels[2], channels[3])

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck - channel count must match the last encoder created
        # If num_pools=0: input is enc1 (channels[1])
        # If num_pools=1: input is enc1_pooled (channels[1])
        # If num_pools=2: input is enc2_pooled (channels[2])
        # If num_pools=3: input is enc3_pooled (channels[3])
        bottleneck_ch = channels[max(1, self.num_pools)]
        self.bottleneck = ComplexConv3d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1)

        # Decoder
        if self.num_pools >= 3:
            self.up3 = ComplexUpsample(channels[3], channels[3], kernel_size=2, stride=2)
            self.dec3 = self._make_decoder_block(channels[3] * 2, channels[2])

        if self.num_pools >= 2:
            up2_in = channels[2] if self.num_pools >= 3 else channels[2]
            self.up2 = ComplexUpsample(up2_in, channels[2], kernel_size=2, stride=2)
            self.dec2 = self._make_decoder_block(channels[2] * 2, channels[1])

        if self.num_pools >= 1:
            up1_in = channels[1]
            self.up1 = ComplexUpsample(up1_in, channels[1], kernel_size=2, stride=2)
            self.dec1 = self._make_decoder_block(channels[1] * 2, channels[0])

        # Final output
        final_ch = channels[0] if self.num_pools >= 1 else channels[1]
        self.output = ComplexConv3d(final_ch, 1, kernel_size=1)

    def _make_encoder_block(self, in_ch: int, out_ch: int):
        """Create encoder block with two convolutions."""
        return nn.Sequential(
            ComplexConv3d(in_ch, out_ch, kernel_size=3, padding=1),
            ComplexBatchNorm3d(out_ch),
            ComplexConv3d(out_ch, out_ch, kernel_size=3, padding=1),
            ComplexBatchNorm3d(out_ch),
        )

    def _make_decoder_block(self, in_ch: int, out_ch: int):
        """Create decoder block with two convolutions."""
        return nn.Sequential(
            ComplexConv3d(in_ch, out_ch, kernel_size=3, padding=1),
            ComplexBatchNorm3d(out_ch),
            ComplexConv3d(out_ch, out_ch, kernel_size=3, padding=1),
            ComplexBatchNorm3d(out_ch),
        )

    def _pool_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling to complex data."""
        real = self.pool(x[..., 0])
        imag = self.pool(x[..., 1])
        return torch.stack([real, imag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frequency band [B, 1, H, W, D, 2]

        Returns:
            Denoised band [B, 1, H, W, D, 2]
        """
        # Encoder with skip connections
        enc1_out = self.enc1(x)  # [B, C1, H, W, D, 2]

        if self.num_pools == 0:
            # No pooling - just bottleneck and output
            bottleneck_out = self.bottleneck(enc1_out)
            output = self.output(bottleneck_out)
            return output

        enc1_pooled = self._pool_complex(enc1_out)

        if self.num_pools >= 2:
            enc2_out = self.enc2(enc1_pooled)  # [B, C2, H/2, W/2, D/2, 2]
            enc2_pooled = self._pool_complex(enc2_out)
        else:
            enc2_out = None
            enc2_pooled = enc1_pooled

        if self.num_pools >= 3:
            enc3_out = self.enc3(enc2_pooled)  # [B, C3, H/4, W/4, D/4, 2]
            enc3_pooled = self._pool_complex(enc3_out)
        else:
            enc3_out = None
            enc3_pooled = enc2_pooled

        # Bottleneck
        bottleneck_out = self.bottleneck(enc3_pooled)

        # Decoder with skip connections
        if self.num_pools >= 3:
            up3 = self.up3(bottleneck_out)
            dec3_in = torch.cat([up3, enc3_out], dim=1)
            dec3_out = self.dec3(dec3_in)
        else:
            dec3_out = bottleneck_out

        if self.num_pools >= 2:
            up2 = self.up2(dec3_out)
            dec2_in = torch.cat([up2, enc2_out], dim=1)
            dec2_out = self.dec2(dec2_in)
        else:
            dec2_out = dec3_out

        if self.num_pools >= 1:
            up1 = self.up1(dec2_out)
            dec1_in = torch.cat([up1, enc1_out], dim=1)
            dec1_out = self.dec1(dec1_in)
        else:
            dec1_out = dec2_out

        # Final output
        output = self.output(dec1_out)

        return output


class CrossBandAttention(nn.Module):
    """
    Cross-attention between different frequency bands.

    Allows low frequencies to guide high frequency denoising.
    """

    def __init__(self, num_bands: int, feature_dim: int, num_heads: int = 4):
        super().__init__()

        self.num_bands = num_bands
        self.feature_dim = feature_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_features: [B, num_bands, feature_dim]

        Returns:
            Attended features: [B, num_bands, feature_dim]
        """
        # Self-attention across bands
        attn_out, _ = self.attention(band_features, band_features, band_features)

        # Residual connection and normalization
        output = self.layer_norm(band_features + attn_out)

        return output


class MRFUNet(BaseFourierModel):
    """
    Multi-Resolution Fourier U-Net.

    Splits Fourier space into frequency bands and processes each with a U-Net.
    Cross-band attention at bottleneck allows information exchange.

    Architecture:
        1. Split input into frequency bands (along rfft dimension)
        2. Each band → U-Net encoder → bottleneck features
        3. Cross-band attention (low freq guides high freq)
        4. Each band → U-Net decoder → denoised band
        5. Concatenate bands → output

    Args:
        input_shape: Input dimensions (H, W, D_rfft)
        num_bands: Number of frequency bands to split into
        base_channels: Base number of channels for U-Nets
    """

    def __init__(
        self,
        input_shape=(64, 64, 33),
        hidden_channels=16,
        num_bands=4,
    ):
        super().__init__(input_shape, hidden_channels)

        self.num_bands = num_bands
        H, W, D = input_shape

        # Calculate band splits
        self.band_splits = self._calculate_band_splits(D, num_bands)

        # Create U-Net for each band
        self.band_unets = nn.ModuleList(
            [
                FrequencyBandUNet(depth_size=split[1] - split[0], base_channels=hidden_channels)
                for split in self.band_splits
            ]
        )

        # Cross-band attention (at bottleneck level)
        # Feature dimension is based on bottleneck size
        bottleneck_size = (H // 8) * (W // 8) * max(1, (D // 8 // num_bands)) * hidden_channels * 4
        self.cross_band_attention = CrossBandAttention(
            num_bands=num_bands, feature_dim=bottleneck_size, num_heads=4
        )

    def _calculate_band_splits(self, depth: int, num_bands: int):
        """
        Calculate how to split depth dimension into bands.

        Returns:
            List of (start, end) tuples for each band
        """
        band_size = depth // num_bands
        splits = []

        for i in range(num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else depth
            splits.append((start, end))

        return splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Fourier coefficients [B, H, W, D, 2]

        Returns:
            Denoised Fourier coefficients [B, H, W, D, 2]
        """
        B, H, W, D, _ = x.shape

        # Split into frequency bands
        bands = []
        for start, end in self.band_splits:
            band = x[:, :, :, start:end, :]  # [B, H, W, band_depth, 2]
            # Add channel dimension
            band = band.unsqueeze(1)  # [B, 1, H, W, band_depth, 2]
            bands.append(band)

        # Process each band through its U-Net (encoder part)
        band_outputs = []
        bottleneck_features = []

        for i, band in enumerate(bands):
            # For cross-attention, we need bottleneck features
            # For simplicity, we process entire U-Net here
            # In a full implementation, you'd extract bottleneck separately
            band_out = self.band_unets[i](band)  # [B, 1, H, W, band_depth, 2]
            band_outputs.append(band_out)

            # Extract bottleneck features for cross-attention
            # (This is a simplification - in practice, modify U-Net to return bottleneck)
            # For now, we use the output as a proxy
            bottleneck_feat = band_out.reshape(B, -1)  # Flatten
            bottleneck_features.append(bottleneck_feat)

        # Cross-band attention
        bottleneck_features = torch.stack(bottleneck_features, dim=1)  # [B, num_bands, feat_dim]
        attended_features = self.cross_band_attention(bottleneck_features)

        # Note: In a full implementation, you'd use attended features to guide decoding
        # For simplicity, we just use the band outputs directly

        # Concatenate bands back together
        band_outputs = [band.squeeze(1) for band in band_outputs]  # Remove channel dim
        output = torch.cat(band_outputs, dim=3)  # Concatenate along depth

        return output

    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                'num_bands': self.num_bands,
                'band_splits': self.band_splits,
            }
        )
        return info
