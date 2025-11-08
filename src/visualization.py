"""Visualization utilities for 3D volumes and Fourier space."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path


def visualize_volume_slices(
    volume: torch.Tensor,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    title: str = "Volume Slices",
    save_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Visualize three orthogonal slices through a 3D volume.

    Args:
        volume: 3D volume [H, W, D]
        slice_indices: Indices for slices (x, y, z). If None, use center
        title: Plot title
        save_path: Path to save figure
        vmin, vmax: Color scale limits
    """
    if volume.ndim == 4:  # Remove batch dimension if present
        volume = volume[0]

    H, W, D = volume.shape

    # Default to center slices
    if slice_indices is None:
        slice_indices = (H // 2, W // 2, D // 2)

    ix, iy, iz = slice_indices

    # Convert to numpy
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # X slice (YZ plane)
    im0 = axes[0].imshow(volume[ix, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'X slice (index {ix})')
    axes[0].set_xlabel('Z')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0])

    # Y slice (XZ plane)
    im1 = axes[1].imshow(volume[:, iy, :], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Y slice (index {iy})')
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('X')
    plt.colorbar(im1, ax=axes[1])

    # Z slice (XY plane)
    im2 = axes[2].imshow(volume[:, :, iz], cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Z slice (index {iz})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('X')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_fourier_spectrum(
    fourier_coeffs: torch.Tensor,
    log_scale: bool = True,
    title: str = "Fourier Spectrum",
    save_path: Optional[Path] = None,
):
    """
    Visualize Fourier spectrum magnitude.

    Args:
        fourier_coeffs: Fourier coefficients [H, W, D, 2] or [B, H, W, D, 2]
        log_scale: Use log scale for magnitude
        title: Plot title
        save_path: Path to save figure
    """
    if fourier_coeffs.ndim == 5:  # Remove batch dimension
        fourier_coeffs = fourier_coeffs[0]

    # Compute magnitude
    magnitude = torch.sqrt(
        fourier_coeffs[..., 0] ** 2 + fourier_coeffs[..., 1] ** 2
    )

    if log_scale:
        magnitude = torch.log(magnitude + 1e-8)

    # Visualize center slices
    visualize_volume_slices(
        magnitude,
        title=f"{title} {'(log scale)' if log_scale else ''}",
        save_path=save_path,
    )


def visualize_denoising_comparison(
    noisy_fourier: torch.Tensor,
    denoised_fourier: torch.Tensor,
    clean_fourier: torch.Tensor,
    slice_idx: Optional[int] = None,
    save_path: Optional[Path] = None,
):
    """
    Compare noisy, denoised, and clean volumes side by side.

    Args:
        noisy_fourier: Noisy Fourier coefficients [H, W, D, 2]
        denoised_fourier: Denoised Fourier coefficients
        clean_fourier: Clean Fourier coefficients
        slice_idx: Which Z slice to show
        save_path: Path to save figure
    """
    # Convert to real space
    noisy_real = torch.fft.irfftn(
        torch.view_as_complex(noisy_fourier.contiguous()),
        dim=(0, 1, 2),
    )
    denoised_real = torch.fft.irfftn(
        torch.view_as_complex(denoised_fourier.contiguous()),
        dim=(0, 1, 2),
    )
    clean_real = torch.fft.irfftn(
        torch.view_as_complex(clean_fourier.contiguous()),
        dim=(0, 1, 2),
    )

    if slice_idx is None:
        slice_idx = noisy_real.shape[2] // 2

    # Convert to numpy
    noisy_np = noisy_real[:, :, slice_idx].cpu().numpy()
    denoised_np = denoised_real[:, :, slice_idx].cpu().numpy()
    clean_np = clean_real[:, :, slice_idx].cpu().numpy()
    error_np = np.abs(denoised_np - clean_np)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Noisy
    im0 = axes[0, 0].imshow(noisy_np, cmap='gray')
    axes[0, 0].set_title('Noisy')
    plt.colorbar(im0, ax=axes[0, 0])

    # Denoised
    im1 = axes[0, 1].imshow(denoised_np, cmap='gray')
    axes[0, 1].set_title('Denoised')
    plt.colorbar(im1, ax=axes[0, 1])

    # Clean
    im2 = axes[1, 0].imshow(clean_np, cmap='gray')
    axes[1, 0].set_title('Clean (Ground Truth)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Error map
    im3 = axes[1, 1].imshow(error_np, cmap='hot')
    axes[1, 1].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(f'Denoising Comparison (Z slice {slice_idx})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_frequency_response(
    clean_fourier: torch.Tensor,
    denoised_fourier: torch.Tensor,
    num_bins: int = 50,
    save_path: Optional[Path] = None,
):
    """
    Plot frequency response (SNR vs frequency).

    Args:
        clean_fourier: Clean Fourier coefficients [H, W, D, 2]
        denoised_fourier: Denoised Fourier coefficients
        num_bins: Number of frequency bins
        save_path: Path to save figure
    """
    H, W, D, _ = clean_fourier.shape

    # Create frequency grid
    kx = torch.arange(H) - H // 2
    ky = torch.arange(W) - W // 2
    kz = torch.arange(D)

    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2).to(clean_fourier.device)

    max_k = k_mag.max()
    k_bins = torch.linspace(0, max_k, num_bins + 1)

    # Compute SNR per bin
    snr_values = []
    k_centers = []

    for i in range(num_bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])

        if mask.sum() > 0:
            clean_band = clean_fourier[mask]
            denoised_band = denoised_fourier[mask]

            signal_power = (clean_band**2).mean()
            noise_power = ((denoised_band - clean_band) ** 2).mean()

            if noise_power > 0:
                snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                snr_values.append(snr.item())
                k_centers.append((k_bins[i] + k_bins[i + 1]) / 2)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_centers, snr_values, 'b-', linewidth=2)
    plt.xlabel('Frequency Magnitude |k|', fontsize=12)
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.title('Frequency Response', fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history_file: Path,
    save_path: Optional[Path] = None,
):
    """
    Plot training history from JSON file.

    Args:
        history_file: Path to training_history.json
        save_path: Path to save figure
    """
    import json

    with open(history_file, 'r') as f:
        history = json.load(f)

    train_history = history['train']
    val_history = history['val']

    # Extract losses
    train_losses = [h['loss'] for h in train_history]
    val_losses = [h['loss'] for h in val_history]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, label='Train Loss', linewidth=2)
    ax.plot(
        range(0, len(train_losses), len(train_losses) // len(val_losses)),
        val_losses,
        label='Validation Loss',
        linewidth=2,
        marker='o',
    )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()
