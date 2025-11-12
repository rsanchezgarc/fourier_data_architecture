"""PyTorch Dataset for Fourier space denoising."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import random

from .volume_utils import (
    load_volume,
    normalize_volume,
    volume_to_fourier,
    generate_synthetic_volume,
)
from .noise_models import (
    add_gaussian_noise_fourier,
    add_poisson_noise_fourier,
    add_mixed_noise_fourier,
)


class FourierDenoisingDataset(Dataset):
    """
    Dataset for training Fourier space denoising models.

    Loads 3D volumes, converts to Fourier space, and adds noise.

    Args:
        data_dir: Directory containing volume files (or None for synthetic)
        volume_size: Size of volumes (cubic)
        noise_type: 'gaussian', 'poisson', or 'mixed'
        noise_params: Dictionary of noise parameters
        normalize_method: 'zscore', 'minmax', or 'none'
        synthetic: If True, use synthetic data instead of loading files
        num_synthetic_samples: Number of synthetic samples to generate
        cache_in_memory: Cache processed data in RAM
        augmentation: Apply random rotations/flips
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        volume_size: int = 64,
        noise_type: str = 'gaussian',
        noise_params: Optional[dict] = None,
        normalize_method: str = 'zscore',
        synthetic: bool = False,
        num_synthetic_samples: int = 1000,
        cache_in_memory: bool = False,
        augmentation: bool = True,
    ):
        self.volume_size = volume_size
        self.noise_type = noise_type
        self.noise_params = noise_params or {}
        self.normalize_method = normalize_method
        self.synthetic = synthetic
        self.cache_in_memory = cache_in_memory
        self.augmentation = augmentation

        # Default noise parameters
        if 'noise_level' not in self.noise_params and noise_type == 'gaussian':
            self.noise_params['noise_level'] = 0.1
        if 'peak_count' not in self.noise_params and noise_type == 'poisson':
            self.noise_params['peak_count'] = 500.0

        if synthetic:
            # Generate synthetic data
            self.num_samples = num_synthetic_samples
            self.file_paths = None
        else:
            # Load file paths
            if data_dir is None:
                raise ValueError("data_dir must be provided when synthetic=False")

            data_dir = Path(data_dir)
            self.file_paths = sorted(list(data_dir.glob('*.npy')) + list(data_dir.glob('*.npz')))

            if len(self.file_paths) == 0:
                raise ValueError(f"No volume files found in {data_dir}")

            self.num_samples = len(self.file_paths)

        # Cache
        self.cache = {} if cache_in_memory else None

        print(f"Created dataset with {self.num_samples} samples")
        print(f"  Synthetic: {synthetic}")
        print(f"  Volume size: {volume_size}")
        print(f"  Noise type: {noise_type}")
        print(f"  Noise params: {self.noise_params}")

    def __len__(self) -> int:
        return self.num_samples

    def _load_clean_volume(self, idx: int) -> torch.Tensor:
        """Load or generate a clean volume."""
        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]

        if self.synthetic:
            # Generate synthetic volume
            shape = random.choice(['sphere', 'cube', 'torus', 'random'])
            volume = generate_synthetic_volume(
                volume_size=self.volume_size,
                shape=shape,
                noise_level=0.0,
            )
        else:
            # Load from file
            volume = load_volume(self.file_paths[idx], volume_size=self.volume_size)

        # Normalize
        volume = normalize_volume(volume, method=self.normalize_method)

        if self.cache_in_memory:
            self.cache[idx] = volume

        return volume

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to volume."""
        if not self.augmentation:
            return volume

        # Random 90-degree rotations
        if random.random() > 0.5:
            k = random.randint(1, 3)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            volume = torch.rot90(volume, k=k, dims=axes)

        # Random flips
        for dim in range(3):
            if random.random() > 0.5:
                volume = torch.flip(volume, dims=[dim])

        return volume

    def _add_noise(self, fourier_coeffs: torch.Tensor) -> torch.Tensor:
        """Add noise to Fourier coefficients."""
        if self.noise_type == 'gaussian':
            return add_gaussian_noise_fourier(fourier_coeffs.unsqueeze(0), **self.noise_params).squeeze(0)
        elif self.noise_type == 'poisson':
            return add_poisson_noise_fourier(fourier_coeffs.unsqueeze(0), **self.noise_params).squeeze(0)
        elif self.noise_type == 'mixed':
            return add_mixed_noise_fourier(fourier_coeffs.unsqueeze(0), **self.noise_params).squeeze(0)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            noisy_fourier: Noisy Fourier coefficients [H, W, D_rfft, 2]
            clean_fourier: Clean Fourier coefficients [H, W, D_rfft, 2]
        """
        # Load clean volume
        clean_volume = self._load_clean_volume(idx)

        # Apply augmentation
        clean_volume = self._apply_augmentation(clean_volume)

        # Convert to Fourier space
        clean_fourier = volume_to_fourier(clean_volume)

        # Add noise
        noisy_fourier = self._add_noise(clean_fourier)

        return noisy_fourier, clean_fourier


class SyntheticFourierDataset(FourierDenoisingDataset):
    """Convenience class for synthetic data."""

    def __init__(
        self,
        num_samples: int = 1000,
        volume_size: int = 64,
        noise_type: str = 'gaussian',
        noise_params: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(
            data_dir=None,
            volume_size=volume_size,
            noise_type=noise_type,
            noise_params=noise_params,
            synthetic=True,
            num_synthetic_samples=num_samples,
            **kwargs
        )
