"""Evaluation metrics for denoising quality."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


def psnr(prediction: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    Args:
        prediction: Predicted volume [B, H, W, D] or [H, W, D]
        target: Ground truth volume
        max_value: Maximum possible pixel value

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(prediction, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(mse)


def ssim_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_value: float = 1.0,
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) for 3D volumes.

    Simplified implementation for 3D data.

    Args:
        prediction: Predicted volume [B, H, W, D] or [H, W, D]
        target: Ground truth volume
        window_size: Size of sliding window
        max_value: Maximum possible pixel value

    Returns:
        SSIM value (0-1, higher is better)
    """
    # Add batch and channel dimensions if needed
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif prediction.ndim == 4:
        prediction = prediction.unsqueeze(1)
        target = target.unsqueeze(1)

    # Constants for stability
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    # Create 3D Gaussian window
    sigma = 1.5
    gauss = torch.exp(
        -torch.arange(-window_size // 2 + 1, window_size // 2 + 1, dtype=torch.float32) ** 2
        / (2 * sigma**2)
    )
    gauss = gauss / gauss.sum()

    # 3D window
    window_3d = gauss.unsqueeze(1).unsqueeze(2) * gauss.unsqueeze(0).unsqueeze(2) * gauss.unsqueeze(0).unsqueeze(1)
    window_3d = window_3d.unsqueeze(0).unsqueeze(0).to(prediction.device)

    # Mean
    mu1 = F.conv3d(prediction, window_3d, padding=window_size // 2)
    mu2 = F.conv3d(target, window_3d, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Variance
    sigma1_sq = F.conv3d(prediction * prediction, window_3d, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv3d(target * target, window_3d, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv3d(prediction * target, window_3d, padding=window_size // 2) - mu1_mu2

    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


def fourier_mse(
    prediction_fourier: torch.Tensor,
    target_fourier: torch.Tensor,
) -> float:
    """
    Calculate MSE in Fourier space.

    Args:
        prediction_fourier: Predicted Fourier coefficients [B, H, W, D, 2]
        target_fourier: Target Fourier coefficients

    Returns:
        MSE in Fourier space
    """
    return F.mse_loss(prediction_fourier, target_fourier).item()


def phase_error(
    prediction_fourier: torch.Tensor,
    target_fourier: torch.Tensor,
) -> float:
    """
    Calculate mean absolute phase error.

    Args:
        prediction_fourier: Predicted Fourier coefficients [B, H, W, D, 2]
        target_fourier: Target Fourier coefficients

    Returns:
        Mean absolute phase difference (radians)
    """
    # Extract real and imaginary parts
    pred_real = prediction_fourier[..., 0]
    pred_imag = prediction_fourier[..., 1]
    tgt_real = target_fourier[..., 0]
    tgt_imag = target_fourier[..., 1]

    # Calculate phases
    pred_phase = torch.atan2(pred_imag, pred_real)
    tgt_phase = torch.atan2(tgt_imag, tgt_real)

    # Phase difference (wrapped to [-π, π])
    phase_diff = torch.atan2(
        torch.sin(pred_phase - tgt_phase),
        torch.cos(pred_phase - tgt_phase),
    )

    return torch.abs(phase_diff).mean().item()


def magnitude_error(
    prediction_fourier: torch.Tensor,
    target_fourier: torch.Tensor,
    relative: bool = True,
) -> float:
    """
    Calculate magnitude error in Fourier space.

    Args:
        prediction_fourier: Predicted Fourier coefficients [B, H, W, D, 2]
        target_fourier: Target Fourier coefficients
        relative: If True, compute relative error

    Returns:
        Magnitude error
    """
    # Calculate magnitudes
    pred_mag = torch.sqrt(
        prediction_fourier[..., 0] ** 2 + prediction_fourier[..., 1] ** 2
    )
    tgt_mag = torch.sqrt(target_fourier[..., 0] ** 2 + target_fourier[..., 1] ** 2)

    if relative:
        error = torch.abs(pred_mag - tgt_mag) / (tgt_mag + 1e-8)
        return error.mean().item()
    else:
        return F.mse_loss(pred_mag, tgt_mag).item()


def frequency_band_snr(
    prediction_fourier: torch.Tensor,
    target_fourier: torch.Tensor,
    num_bands: int = 4,
) -> Dict[str, float]:
    """
    Calculate SNR for different frequency bands.

    Args:
        prediction_fourier: Predicted Fourier coefficients [B, H, W, D, 2]
        target_fourier: Target Fourier coefficients
        num_bands: Number of frequency bands to evaluate

    Returns:
        Dictionary with SNR per band
    """
    B, H, W, D, _ = prediction_fourier.shape

    # Create frequency magnitude grid
    kx = torch.arange(H, device=prediction_fourier.device) - H // 2
    ky = torch.arange(W, device=prediction_fourier.device) - W // 2
    kz = torch.arange(D, device=prediction_fourier.device)

    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    # Divide into bands
    max_k = k_mag.max()
    band_bounds = torch.linspace(0, max_k, num_bands + 1)

    snr_per_band = {}

    for i in range(num_bands):
        mask = (k_mag >= band_bounds[i]) & (k_mag < band_bounds[i + 1])

        if mask.sum() > 0:
            # Extract coefficients in this band
            pred_band = prediction_fourier[:, mask, :]
            tgt_band = target_fourier[:, mask, :]

            # Calculate signal power and noise power
            signal_power = (tgt_band**2).mean()
            noise_power = ((pred_band - tgt_band) ** 2).mean()

            if noise_power > 0:
                snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                snr_per_band[f'band_{i}'] = snr.item()
            else:
                snr_per_band[f'band_{i}'] = float('inf')

    return snr_per_band


def compute_all_metrics(
    prediction_fourier: torch.Tensor,
    target_fourier: torch.Tensor,
    compute_ssim: bool = True,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        prediction_fourier: Predicted Fourier coefficients [B, H, W, D, 2]
        target_fourier: Target Fourier coefficients
        compute_ssim: Whether to compute SSIM (slow for large volumes)

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Convert to real space for PSNR/SSIM
    pred_real = torch.fft.irfftn(
        torch.view_as_complex(prediction_fourier.contiguous()),
        dim=(1, 2, 3),
    )
    tgt_real = torch.fft.irfftn(
        torch.view_as_complex(target_fourier.contiguous()),
        dim=(1, 2, 3),
    )

    # Real space metrics
    metrics['psnr'] = psnr(pred_real, tgt_real).item()

    if compute_ssim:
        metrics['ssim'] = ssim_3d(pred_real, tgt_real)

    metrics['mse_real'] = F.mse_loss(pred_real, tgt_real).item()

    # Fourier space metrics
    metrics['mse_fourier'] = fourier_mse(prediction_fourier, target_fourier)
    metrics['phase_error'] = phase_error(prediction_fourier, target_fourier)
    metrics['magnitude_error'] = magnitude_error(prediction_fourier, target_fourier)

    # Frequency band SNR
    band_snr = frequency_band_snr(prediction_fourier, target_fourier, num_bands=4)
    metrics.update(band_snr)

    return metrics


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics_dict: Dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def get_average(self) -> Dict[str, float]:
        """Get average metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def summary_string(self) -> str:
        """Get formatted summary string."""
        avg_metrics = self.get_average()
        lines = []
        for key, value in avg_metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)
