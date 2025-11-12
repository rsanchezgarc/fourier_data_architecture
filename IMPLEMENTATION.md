# Implementation Guide

This directory contains implementations of 6 deep learning architectures for 3D Fourier space denoising.

## Architecture Implementations

All architectures are located in `src/models/`:

1. **CV-ResNet** (`cv_resnet.py`) - Complex-Valued ResNet
   - Baseline architecture with complex residual blocks
   - ~5-10M parameters

2. **FNO-3D** (`fno_3d.py`) - Fourier Neural Operator
   - Operates entirely in Fourier space with spectral convolutions
   - Frequency-specific learned kernels
   - ~3-8M parameters

3. **FSC-Net** (`fsc_net.py`) - Frequency Shell CNN
   - Groups coefficients by frequency magnitude |k|
   - Shell-specific processing with cross-shell fusion
   - ~5-10M parameters

4. **MRFU-Net** (`mrfu_net.py`) - Multi-Resolution Fourier U-Net
   - Splits Fourier space into frequency bands
   - U-Net per band with cross-band attention
   - ~8-15M parameters

5. **RAD-Net** (`rad_net.py`) - Radial-Angular Decomposition Network
   - Exploits spherical geometry (r, θ, φ)
   - Separate radial and angular processing
   - ~4-7M parameters

6. **FGT** (`fgt.py`) - Frequency-Grouped Transformer
   - Tokenizes frequency bands
   - Within-band and cross-band attention
   - ~15-25M parameters

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- einops >= 0.7.0

## Usage

### Basic Usage

```python
from src.models import CVResNet, FNO3D, FSCNet, MRFUNet, RADNet, FrequencyGroupedTransformer

# Create model
model = CVResNet(
    input_shape=(64, 64, 33),  # H, W, D_rfft
    hidden_channels=32,
    num_blocks=16
)

# Input: Fourier coefficients [B, H, W, D, 2]
import torch
batch_size = 4
fourier_coeffs = torch.randn(batch_size, 64, 64, 33, 2)

# Forward pass
denoised_coeffs = model(fourier_coeffs)

# Get model info
info = model.get_model_info()
print(f"Model: {info['name']}")
print(f"Parameters: {info['parameters']:,}")
```

### Testing All Models

```bash
python tests/test_models.py
```

This will:
- Instantiate all 6 architectures
- Run forward passes with dummy data
- Verify output shapes and values
- Print summary of results

### Input/Output Format

All models expect:
- **Input**: `[B, H, W, D, 2]` where last dimension is `[real, imaginary]`
  - Typically `H=64, W=64, D=33` (from `rfft` of 64×64×64 volume)
- **Output**: Same shape as input `[B, H, W, D, 2]`

The input is already in Fourier space (after `torch.fft.rfftn`).

### Model Selection Guide

**Quick Baseline**: Start with **CV-ResNet**
- Simple, proven architecture
- Easy to train and debug
- No frequency-specific processing

**Best Theoretical Foundation**: Use **FNO-3D**
- Operates entirely in Fourier space
- Frequency-specific kernels
- Resolution-invariant (theoretically)

**Novel Frequency Processing**: Try **FSC-Net** or **RAD-Net**
- Explicit frequency structure exploitation
- Shell-based or spherical decomposition
- May work better for specific data types

**Maximum Capacity**: Use **FGT** (Transformer)
- Attention-based processing
- Most parameters
- May need more data/compute

## Model Architecture Details

### Complex Number Handling

All models handle complex numbers using one of these approaches:

**Option A: Separate Real/Imaginary (used in most models)**
```python
# Last dimension is [real, imag]
real = x[..., 0]
imag = x[..., 1]
```

**Option B: True Complex Arithmetic (used in FNO-3D)**
```python
# Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
real_out = conv(real, W_real) - conv(imag, W_imag)
imag_out = conv(real, W_imag) + conv(imag, W_real)
```

### Frequency Organization

Different models organize Fourier space differently:

- **CV-ResNet**: Cartesian grid, no frequency awareness
- **FNO-3D**: Per-mode weights, truncated to low frequencies
- **FSC-Net**: Radial shells based on |k|
- **MRFU-Net**: Bands along rfft dimension
- **RAD-Net**: Spherical coordinates (r, θ, φ)
- **FGT**: Frequency bands with tokenization

## Training

### Example Training Loop

```python
import torch
import torch.nn as nn
from src.models import FNO3D

# Setup
model = FNO3D(input_shape=(64, 64, 33), hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training
model.train()
for epoch in range(num_epochs):
    for noisy_fourier, clean_fourier in dataloader:
        optimizer.zero_grad()

        # Forward
        denoised = model(noisy_fourier)

        # Loss in Fourier space
        loss = criterion(denoised, clean_fourier)

        # Backward
        loss.backward()
        optimizer.step()
```

### Loss Functions

You can use loss in:
- **Fourier space**: MSE between Fourier coefficients (fast)
- **Real space**: Convert to real space with IRFFT and compute MSE (more accurate)
- **Hybrid**: Combine both losses

```python
# Fourier space loss
loss_fourier = F.mse_loss(denoised_fourier, clean_fourier)

# Real space loss
denoised_real = torch.fft.irfftn(
    torch.view_as_complex(denoised_fourier.contiguous()),
    dim=(1, 2, 3)
)
clean_real = torch.fft.irfftn(
    torch.view_as_complex(clean_fourier.contiguous()),
    dim=(1, 2, 3)
)
loss_real = F.mse_loss(denoised_real, clean_real)

# Combined
loss = 0.5 * loss_fourier + 0.5 * loss_real
```

## File Structure

```
fourier_data_architecture/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Base classes and common components
│   │   ├── cv_resnet.py         # CV-ResNet
│   │   ├── fno_3d.py            # FNO-3D
│   │   ├── fsc_net.py           # FSC-Net
│   │   ├── mrfu_net.py          # MRFU-Net
│   │   ├── rad_net.py           # RAD-Net
│   │   └── fgt.py               # FGT
│   └── utils/
│       ├── __init__.py
│       └── complex_ops.py       # Complex-valued operations
├── tests/
│   ├── __init__.py
│   └── test_models.py           # Test all models
├── requirements.txt
├── README.md
├── REQUIREMENTS.md              # Original requirements
├── ARCHITECTURE_PROPOSALS.md    # Architecture designs
├── ARCHITECTURE_DETAILS.md      # Detailed mechanics
└── IMPLEMENTATION.md           # This file
```

## Next Steps

1. **Implement data pipeline**: Load ModelNet40, voxelize, add noise
2. **Create training script**: Full training loop with logging
3. **Benchmarking**: Compare all architectures
4. **Hyperparameter tuning**: Grid search for optimal settings
5. **Evaluation**: PSNR, SSIM, inference time

## Notes

- All models are currently untrained
- Parameter counts are approximate and depend on configuration
- GPU is highly recommended for training
- Models can be made smaller/larger by adjusting `hidden_channels`
- For production use, consider:
  - Mixed precision training (`torch.cuda.amp`)
  - Gradient checkpointing for memory
  - DistributedDataParallel for multi-GPU

## Troubleshooting

**Out of memory**:
- Reduce `batch_size`
- Reduce `hidden_channels`
- Reduce `input_shape` for testing
- Use gradient checkpointing

**Slow training**:
- Use mixed precision training
- Profile bottlenecks with PyTorch profiler
- Consider smaller models (FNO-3D is most efficient)

**NaN/Inf values**:
- Check learning rate (may be too high)
- Add gradient clipping
- Check for division by zero in custom ops
- Initialize weights carefully
