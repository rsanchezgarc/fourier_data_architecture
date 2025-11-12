## Training and Benchmarking Scripts

This directory contains scripts for training and evaluating Fourier space denoising models.

### Training a Model

Train a single model:

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
    --output-dir outputs
```

**Available models**: `cvresnet`, `fno3d`, `fscnet`, `mrfunet`, `radnet`, `fgt`

**Arguments**:
- `--model`: Which architecture to use
- `--synthetic`: Use synthetic data (recommended for quick testing)
- `--data-dir`: Path to real data directory (if not using synthetic)
- `--volume-size`: Size of 3D volumes (default: 64)
- `--noise-type`: Type of noise (`gaussian`, `poisson`, `mixed`)
- `--noise-level`: Noise level for Gaussian noise
- `--batch-size`: Batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--loss-type`: Loss function (`fourier`, `real`, `hybrid`)
- `--output-dir`: Where to save checkpoints and logs

### Benchmarking All Models

Compare all architectures on the same test set:

```bash
python scripts/benchmark.py \
    --volume-size 64 \
    --num-samples 100 \
    --batch-size 4 \
    --noise-type gaussian \
    --noise-level 0.1 \
    --output-dir benchmark_results
```

This will:
1. Create a test dataset
2. Run inference with all 6 models
3. Compute detailed metrics (PSNR, SSIM, inference time, etc.)
4. Generate a comparison table
5. Save results to CSV and JSON

**Arguments**:
- `--volume-size`: Size of test volumes
- `--num-samples`: Number of test samples
- `--models`: Specific models to benchmark (default: all)
- `--output-dir`: Where to save results

### Example Usage

**Quick test with small synthetic dataset**:
```bash
# Train FNO-3D for 10 epochs
python scripts/train.py \
    --model fno3d \
    --synthetic \
    --num-synthetic 100 \
    --volume-size 32 \
    --epochs 10 \
    --batch-size 8

# Benchmark on small test set
python scripts/benchmark.py \
    --volume-size 32 \
    --num-samples 20 \
    --models fno3d cvresnet
```

**Full training with real data**:
```bash
python scripts/train.py \
    --model fscnet \
    --data-dir /path/to/volumes \
    --volume-size 64 \
    --noise-type mixed \
    --batch-size 4 \
    --epochs 100 \
    --lr 1e-3 \
    --scheduler plateau
```

### Output Structure

Training outputs are saved to `<output-dir>/<model>_<timestamp>/`:
```
outputs/fno3d_20241108_123456/
├── config.json              # Training configuration
├── best_model.pth           # Best model checkpoint
├── final_model.pth          # Final model
├── checkpoint_epoch10.pth   # Periodic checkpoints
└── training_history.json    # Training curves
```

Benchmark results are saved to `<output-dir>/`:
```
benchmark_results/
├── benchmark_results.csv    # Comparison table
└── benchmark_results.json   # Detailed results
```

### Tips

1. **Start with synthetic data** for quick iteration and debugging
2. **Use smaller volume sizes** (32 or 48) for faster training
3. **Monitor GPU memory** - reduce batch size if OOM
4. **FNO-3D is most efficient** for parameter count
5. **CVResNet is the simplest baseline** - start here
6. **Use `--loss-type hybrid`** for best results (combines Fourier + real space loss)
