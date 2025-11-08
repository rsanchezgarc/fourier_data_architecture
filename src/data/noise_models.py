"""Noise models for Fourier space data augmentation."""

import torch
import numpy as np


def add_gaussian_noise_fourier(
    fourier_coeffs: torch.Tensor,
    noise_level: float = 0.1,
    frequency_dependent: bool = True,
) -> torch.Tensor:
    """
    Add Gaussian noise to Fourier coefficients.

    Args:
        fourier_coeffs: Complex Fourier coefficients [B, H, W, D, 2]
        noise_level: Noise standard deviation (relative to signal)
        frequency_dependent: If True, add more noise to high frequencies

    Returns:
        Noisy Fourier coefficients [B, H, W, D, 2]
    """
    B, H, W, D, _ = fourier_coeffs.shape

    if frequency_dependent:
        # Create frequency-dependent noise scaling
        kx = torch.arange(H, dtype=torch.float32, device=fourier_coeffs.device) - H // 2
        ky = torch.arange(W, dtype=torch.float32, device=fourier_coeffs.device) - W // 2
        kz = torch.arange(D, dtype=torch.float32, device=fourier_coeffs.device)

        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

        # Normalize to [0, 1]
        k_mag_norm = k_mag / k_mag.max()

        # Higher noise at high frequencies (more realistic)
        noise_scale = 1.0 + 2.0 * k_mag_norm  # 1x at DC, 3x at max frequency
        noise_scale = noise_scale.unsqueeze(0).unsqueeze(-1)  # [1, H, W, D, 1]
    else:
        noise_scale = 1.0

    # Compute signal power for relative noise level
    signal_power = (fourier_coeffs**2).mean()

    # Generate complex Gaussian noise
    noise_real = torch.randn_like(fourier_coeffs[..., 0]) * noise_level * torch.sqrt(signal_power)
    noise_imag = torch.randn_like(fourier_coeffs[..., 1]) * noise_level * torch.sqrt(signal_power)

    noise = torch.stack([noise_real, noise_imag], dim=-1) * noise_scale

    return fourier_coeffs + noise


def add_poisson_noise_fourier(
    fourier_coeffs: torch.Tensor,
    peak_count: float = 1000.0,
) -> torch.Tensor:
    """
    Add Poisson noise (in real space, then FFT).

    This is more realistic for imaging applications.

    Args:
        fourier_coeffs: Complex Fourier coefficients [B, H, W, D, 2]
        peak_count: Peak photon count in real space

    Returns:
        Noisy Fourier coefficients [B, H, W, D, 2]
    """
    # Convert to real space
    real_space = torch.fft.irfftn(
        torch.view_as_complex(fourier_coeffs.contiguous()),
        dim=(1, 2, 3),
    )

    # Normalize to [0, peak_count]
    real_space_normalized = (real_space - real_space.min()) / (
        real_space.max() - real_space.min() + 1e-8
    )
    real_space_scaled = real_space_normalized * peak_count

    # Add Poisson noise
    # Use Gaussian approximation for efficiency (valid for peak_count > ~20)
    noisy_real_space = real_space_scaled + torch.randn_like(real_space_scaled) * torch.sqrt(
        real_space_scaled.clamp(min=1e-8)
    )

    # Convert back to Fourier space
    noisy_fourier = torch.fft.rfftn(noisy_real_space, dim=(1, 2, 3))
    noisy_fourier_real_imag = torch.stack(
        [noisy_fourier.real, noisy_fourier.imag], dim=-1
    )

    return noisy_fourier_real_imag


def add_mixed_noise_fourier(
    fourier_coeffs: torch.Tensor,
    gaussian_level: float = 0.05,
    poisson_peak: float = 500.0,
    frequency_dependent: bool = True,
) -> torch.Tensor:
    """
    Add both Gaussian and Poisson noise.

    This simulates realistic imaging conditions.

    Args:
        fourier_coeffs: Complex Fourier coefficients [B, H, W, D, 2]
        gaussian_level: Gaussian noise level
        poisson_peak: Poisson peak count
        frequency_dependent: Frequency-dependent Gaussian noise

    Returns:
        Noisy Fourier coefficients [B, H, W, D, 2]
    """
    # Add Poisson noise first (in real space)
    noisy_fourier = add_poisson_noise_fourier(fourier_coeffs, peak_count=poisson_peak)

    # Then add Gaussian noise in Fourier space
    noisy_fourier = add_gaussian_noise_fourier(
        noisy_fourier,
        noise_level=gaussian_level,
        frequency_dependent=frequency_dependent,
    )

    return noisy_fourier


def add_phase_noise_fourier(
    fourier_coeffs: torch.Tensor,
    phase_noise_std: float = 0.1,
) -> torch.Tensor:
    """
    Add noise to phase only (preserve magnitude).

    Useful for testing phase sensitivity.

    Args:
        fourier_coeffs: Complex Fourier coefficients [B, H, W, D, 2]
        phase_noise_std: Standard deviation of phase noise (radians)

    Returns:
        Noisy Fourier coefficients [B, H, W, D, 2]
    """
    real = fourier_coeffs[..., 0]
    imag = fourier_coeffs[..., 1]

    # Convert to magnitude/phase
    magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
    phase = torch.atan2(imag, real)

    # Add phase noise
    phase_noise = torch.randn_like(phase) * phase_noise_std
    noisy_phase = phase + phase_noise

    # Convert back to real/imag
    noisy_real = magnitude * torch.cos(noisy_phase)
    noisy_imag = magnitude * torch.sin(noisy_phase)

    return torch.stack([noisy_real, noisy_imag], dim=-1)
