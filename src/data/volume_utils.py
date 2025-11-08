"""Utilities for loading and processing 3D volumes."""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional


def load_volume(
    file_path: Union[str, Path],
    volume_size: int = 64,
) -> torch.Tensor:
    """
    Load a 3D volume from file.

    Supports:
    - .npy files (NumPy arrays)
    - .npz files (compressed NumPy)
    - .raw files (raw binary)
    - .mrc files (electron microscopy, requires mrcfile)

    Args:
        file_path: Path to volume file
        volume_size: Target size for resizing

    Returns:
        Volume tensor [volume_size, volume_size, volume_size]
    """
    file_path = Path(file_path)

    if file_path.suffix == '.npy':
        volume = np.load(file_path)
    elif file_path.suffix == '.npz':
        data = np.load(file_path)
        # Assume first array is the volume
        volume = data[list(data.keys())[0]]
    elif file_path.suffix == '.raw':
        # Assume raw binary, try to infer size
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        size = int(round(len(data) ** (1 / 3)))
        volume = data.reshape(size, size, size)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Convert to tensor
    volume = torch.from_numpy(volume).float()

    # Resize if needed
    if volume.shape != (volume_size, volume_size, volume_size):
        volume = torch.nn.functional.interpolate(
            volume.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W, D]
            size=(volume_size, volume_size, volume_size),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return volume


def normalize_volume(
    volume: torch.Tensor,
    method: str = 'zscore',
    clip_percentile: Optional[float] = None,
) -> torch.Tensor:
    """
    Normalize 3D volume.

    Args:
        volume: Input volume [H, W, D]
        method: 'zscore', 'minmax', or 'none'
        clip_percentile: If set, clip outliers (e.g., 99.5)

    Returns:
        Normalized volume [H, W, D]
    """
    if clip_percentile is not None:
        lower = torch.quantile(volume, (100 - clip_percentile) / 100)
        upper = torch.quantile(volume, clip_percentile / 100)
        volume = torch.clamp(volume, lower, upper)

    if method == 'zscore':
        mean = volume.mean()
        std = volume.std()
        volume = (volume - mean) / (std + 1e-8)
    elif method == 'minmax':
        vmin = volume.min()
        vmax = volume.max()
        volume = (volume - vmin) / (vmax - vmin + 1e-8)
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return volume


def voxelize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    volume_size: int = 64,
) -> torch.Tensor:
    """
    Voxelize a triangle mesh.

    Args:
        vertices: Vertex positions [N, 3]
        faces: Triangle indices [M, 3]
        volume_size: Output voxel grid size

    Returns:
        Binary voxel grid [volume_size, volume_size, volume_size]
    """
    # Normalize vertices to [0, volume_size]
    vertices = vertices - vertices.min(axis=0)
    vertices = vertices / vertices.max() * (volume_size - 1)

    # Create empty voxel grid
    voxels = np.zeros((volume_size, volume_size, volume_size), dtype=np.float32)

    # Simple voxelization: mark occupied voxels
    # For production, use a proper voxelization library
    for face in faces:
        v0, v1, v2 = vertices[face]

        # Get bounding box of triangle
        bbox_min = np.floor(np.min([v0, v1, v2], axis=0)).astype(int)
        bbox_max = np.ceil(np.max([v0, v1, v2], axis=0)).astype(int)

        bbox_min = np.clip(bbox_min, 0, volume_size - 1)
        bbox_max = np.clip(bbox_max, 0, volume_size - 1)

        # Mark voxels in bounding box (conservative)
        voxels[
            bbox_min[0]:bbox_max[0] + 1,
            bbox_min[1]:bbox_max[1] + 1,
            bbox_min[2]:bbox_max[2] + 1,
        ] = 1.0

    return torch.from_numpy(voxels)


def generate_synthetic_volume(
    volume_size: int = 64,
    shape: str = 'sphere',
    noise_level: float = 0.0,
) -> torch.Tensor:
    """
    Generate synthetic 3D volume for testing.

    Args:
        volume_size: Size of volume
        shape: 'sphere', 'cube', 'torus', or 'random'
        noise_level: Additive Gaussian noise

    Returns:
        Synthetic volume [volume_size, volume_size, volume_size]
    """
    # Create coordinate grid
    x = torch.linspace(-1, 1, volume_size)
    y = torch.linspace(-1, 1, volume_size)
    z = torch.linspace(-1, 1, volume_size)

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    if shape == 'sphere':
        R = torch.sqrt(X**2 + Y**2 + Z**2)
        volume = (R < 0.7).float()
    elif shape == 'cube':
        volume = ((torch.abs(X) < 0.5) & (torch.abs(Y) < 0.5) & (torch.abs(Z) < 0.5)).float()
    elif shape == 'torus':
        R = torch.sqrt(X**2 + Y**2)
        volume = (torch.abs(R - 0.5) < 0.2).float() * (torch.abs(Z) < 0.3).float()
    elif shape == 'random':
        volume = torch.randn(volume_size, volume_size, volume_size)
        volume = (volume > 0.5).float()
    else:
        raise ValueError(f"Unknown shape: {shape}")

    # Add noise
    if noise_level > 0:
        volume = volume + torch.randn_like(volume) * noise_level

    return volume


def volume_to_fourier(volume: torch.Tensor) -> torch.Tensor:
    """
    Convert real-space volume to Fourier coefficients.

    Args:
        volume: Real-space volume [H, W, D]

    Returns:
        Fourier coefficients [H, W, D//2+1, 2] (real, imag)
    """
    # Apply real FFT
    fourier_complex = torch.fft.rfftn(volume, dim=(0, 1, 2))

    # Convert to real/imag representation
    fourier_real_imag = torch.stack([fourier_complex.real, fourier_complex.imag], dim=-1)

    return fourier_real_imag


def fourier_to_volume(fourier_coeffs: torch.Tensor, output_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
    """
    Convert Fourier coefficients to real-space volume.

    Args:
        fourier_coeffs: Fourier coefficients [H, W, D_rfft, 2]
        output_size: Optional output size (H, W, D)

    Returns:
        Real-space volume [H, W, D]
    """
    # Convert to complex representation
    fourier_complex = torch.view_as_complex(fourier_coeffs.contiguous())

    # Apply inverse real FFT
    if output_size is not None:
        volume = torch.fft.irfftn(fourier_complex, s=output_size, dim=(0, 1, 2))
    else:
        volume = torch.fft.irfftn(fourier_complex, dim=(0, 1, 2))

    return volume
