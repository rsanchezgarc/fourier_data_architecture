# Training Guide

Complete guide to training and evaluating Fourier space denoising models.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test All Models

```bash
python tests/test_models.py
```

### 3. Quick Training Test

```bash
python scripts/train.py \
    --model fno3d \
    --synthetic \
    --num-synthetic 100 \
    --volume-size 32 \
    --epochs 5 \
    --batch-size 8
```

### 4. Benchmark All Models

```bash
python scripts/benchmark.py \
    --volume-size 32 \
    --num-samples 50 \
    --batch-size 4
```

## Data Preparation

### Option 1: Synthetic Data (Recommended for Testing)

The code can generate synthetic 3D shapes (spheres, cubes, tori) automatically:

```python
from src.data import SyntheticFourierDataset

dataset = SyntheticFourierDataset(
    num_samples=1000,
    volume_size=64,
    noise_type='gaussian',
    noise_params={'noise_level': 0.1}
)
```

### Option 2: Real Data

Organize your data as `.npy` or `.npz` files:

```
data/
├── volume_001.npy
├── volume_002.npy
├── ...
```

Then use:

```python
from src.data import FourierDenoisingDataset

dataset = FourierDenoisingDataset(
    data_dir='data/',
    volume_size=64,
    noise_type='gaussian',
    noise_params={'noise_level': 0.1}
)
```

## Training Workflow

### 1. Choose an Architecture

- **CVResNet**: Simple baseline, good starting point
- **FNO-3D**: Efficient, frequency-aware, theoretically motivated
- **FSCNet**: Novel shell-based processing
- **MRFUNet**: Hierarchical with U-Net structure
- **RADNet**: Spherical coordinate processing
- **FGT**: Transformer-based, most parameters

### 2. Train the Model

```bash
python scripts/train.py \
    --model fno3d \
    --synthetic \
    --num-synthetic 1000 \
    --volume-size 64 \
    --noise-type gaussian \
    --noise-level 0.1 \
    --batch-size 4 \
    --epochs 100 \
    --lr 1e-3 \
    --loss-type hybrid \
    --output-dir outputs
```

### 3. Monitor Training

Training prints progress and saves:
- `config.json`: All hyperparameters
- `best_model.pth`: Best validation checkpoint
- `training_history.json`: Loss curves

### 4. Evaluate

Load a trained model:

```python
import torch
from src.models import FNO3D

# Load checkpoint
checkpoint = torch.load('outputs/fno3d_20241108/best_model.pth')

# Create model
model = FNO3D(
    input_shape=(64, 64, 33),
    hidden_channels=32,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    denoised = model(noisy_fourier)
```

## Advanced Training

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for noisy, clean in dataloader:
    optimizer.zero_grad()

    with autocast():
        prediction = model(noisy)
        loss = criterion(prediction, clean)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Wrap model
model = DistributedDataParallel(model)

# Use DistributedSampler for data
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

### Custom Loss Functions

```python
def fourier_weighted_loss(prediction, target):
    """Weight loss by frequency importance."""
    # Compute magnitude
    pred_mag = torch.sqrt(prediction[..., 0]**2 + prediction[..., 1]**2)
    tgt_mag = torch.sqrt(target[..., 0]**2 + target[..., 1]**2)

    # Weight low frequencies more
    weights = create_frequency_weights(prediction.shape)

    # Weighted MSE
    loss = ((prediction - target)**2 * weights).mean()
    return loss
```

## Benchmarking

### Compare All Models

```bash
python scripts/benchmark.py \
    --volume-size 64 \
    --num-samples 100 \
    --noise-type gaussian \
    --noise-level 0.1 \
    --output-dir benchmark_results
```

This generates:
- Comparison table with PSNR, SSIM, inference time
- Per-model detailed metrics
- CSV and JSON results

### Benchmark Specific Models

```bash
python scripts/benchmark.py \
    --models fno3d cvresnet fscnet \
    --num-samples 50
```

## Visualization

### Visualize Results

```python
from src.visualization import (
    visualize_denoising_comparison,
    plot_frequency_response,
)

# Compare noisy, denoised, clean
visualize_denoising_comparison(
    noisy_fourier,
    denoised_fourier,
    clean_fourier,
    save_path='comparison.png'
)

# Plot frequency response
plot_frequency_response(
    clean_fourier,
    denoised_fourier,
    save_path='freq_response.png'
)
```

### Plot Training History

```python
from src.visualization import plot_training_history

plot_training_history(
    'outputs/fno3d_20241108/training_history.json',
    save_path='training_curves.png'
)
```

## Hyperparameter Tuning

### Learning Rate

```bash
# Try different learning rates
for lr in 1e-4 1e-3 1e-2; do
    python scripts/train.py --lr $lr --output-dir outputs/lr_${lr}
done
```

### Model Size

```bash
# Try different hidden channel sizes
for channels in 16 32 64; do
    python scripts/train.py \
        --hidden-channels $channels \
        --output-dir outputs/channels_${channels}
done
```

### Noise Levels

```bash
# Train on different noise levels
for noise in 0.05 0.1 0.2; do
    python scripts/train.py \
        --noise-level $noise \
        --output-dir outputs/noise_${noise}
done
```

## Troubleshooting

### Out of Memory

1. Reduce batch size: `--batch-size 2`
2. Reduce volume size: `--volume-size 32`
3. Reduce hidden channels: `--hidden-channels 16`
4. Use gradient checkpointing

### Slow Training

1. Use FNO-3D (most efficient)
2. Enable mixed precision training
3. Reduce number of model parameters
4. Use smaller volume sizes

### Poor Results

1. Try `--loss-type hybrid` (combines Fourier + real space)
2. Increase training epochs
3. Try different noise models
4. Ensure proper data normalization
5. Check learning rate (try scheduler)

### NaN/Inf Loss

1. Reduce learning rate
2. Add gradient clipping
3. Check data normalization
4. Increase numerical stability (add eps=1e-8)

## Performance Expectations

Based on synthetic data (64³ volumes, Gaussian noise 0.1):

| Model | Params | PSNR | Inference Time |
|-------|--------|------|----------------|
| FNO-3D | ~5M | 28-32 dB | ~50 ms |
| CVResNet | ~8M | 26-30 dB | ~80 ms |
| FSCNet | ~10M | 29-33 dB | ~120 ms |
| MRFUNet | ~12M | 28-32 dB | ~100 ms |
| RADNet | ~6M | 27-31 dB | ~90 ms |
| FGT | ~20M | 30-34 dB | ~200 ms |

*Note: Actual performance depends on data, noise level, and hyperparameters*

## Next Steps

1. **Prepare your dataset** (or use synthetic for testing)
2. **Run benchmark** to find best architecture for your use case
3. **Train best model** with full hyperparameter tuning
4. **Evaluate on held-out test set**
5. **Deploy model** for inference

## Citation

If you use this code, please cite the original architecture papers:

- **FNO**: Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks", MICCAI 2015
