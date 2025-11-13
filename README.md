# Fourier Space Deep Learning Architectures for 3D Denoising

Six state-of-the-art deep learning architectures that operate directly in Fourier space for 3D volume denoising. All models process complex-valued Fourier coefficients rather than real-space voxels.

## 🎯 Architectures

| Model | Description | Parameters | Key Features |
|-------|-------------|------------|--------------|
| **CV-ResNet** | Complex-Valued ResNet | 1.8M | Baseline with complex convolutions |
| **FNO-3D** | Fourier Neural Operator | 14.2M | Spectral convolutions with learned kernels per frequency mode |
| **FSC-Net** | Frequency Shell Convolution | 29.2M | Groups frequencies into concentric shells |
| **MRFU-Net** | Multi-Resolution Frequency U-Net | 5.2M | U-Net with frequency band decomposition |
| **RAD-Net** | Radial-Angular Decomposition | 458K | Spherical coordinate processing |
| **FGT** | Frequency-Grouped Transformer | 5.9M | Transformer on frequency band tokens |

## 🚀 Quick Start (Kaggle/Colab)

Copy and paste these cells into your Kaggle or Colab notebook:

### Cell 1: Setup and Installation

```python
# Clone repository
!git clone https://github.com/rsanchezgarc/fourier_data_architecture.git
%cd fourier_data_architecture

# Install dependencies
!pip install -q torch numpy scipy einops matplotlib pandas tqdm optuna plotly

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2: Quick Training Test (Single Model)

```python
# ==== CONFIGURABLE PARAMETERS ====
MODEL = "fno3d"              # Options: cvresnet, fno3d, fscnet, mrfunet, radnet, fgt
EPOCHS = 50                  # Number of epochs (use 5-10 for quick test, 100+ for full training)
NUM_SAMPLES = 500            # Number of synthetic samples (use 100 for quick test, 1000+ for full)
VOLUME_SIZE = 64             # Volume size (32 for quick test, 64 for full)
BATCH_SIZE = 4               # Batch size (increase if you have more GPU memory)
HIDDEN_CHANNELS = 32         # Hidden channels (16/32/64)
# =================================

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

!python scripts/train.py \
    --model {MODEL} \
    --synthetic \
    --num-synthetic {NUM_SAMPLES} \
    --volume-size {VOLUME_SIZE} \
    --batch-size {BATCH_SIZE} \
    --hidden-channels {HIDDEN_CHANNELS} \
    --epochs {EPOCHS} \
    --lr 1e-3 \
    --loss-type hybrid \
    --device {device}
```

### Cell 3: Automatic Hyperparameter Optimization (All Models)

```python
# ==== CONFIGURABLE PARAMETERS ====
MODELS_TO_OPTIMIZE = ["fno3d", "radnet", "mrfunet"]  # Select models or use all 6
TRIALS_PER_MODEL = 20        # Number of trials per model (use 10 for quick, 50+ for thorough)
SEARCH_EPOCHS = 20           # Epochs per trial (use 10-20 for search)
NUM_SAMPLES = 500            # Training samples
VOLUME_SIZE = 64             # Volume size
ENABLE_PRUNING = True        # Prune unpromising trials early
# =================================

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
models_arg = " ".join(MODELS_TO_OPTIMIZE)
pruning_flag = "--pruning" if ENABLE_PRUNING else ""

!python scripts/hyperparam_search.py \
    --models {models_arg} \
    --synthetic \
    --num-synthetic {NUM_SAMPLES} \
    --volume-size {VOLUME_SIZE} \
    --n-trials {TRIALS_PER_MODEL} \
    --epochs {SEARCH_EPOCHS} \
    {pruning_flag} \
    --device {device}

# Note: Best models are trained for 3x epochs and saved automatically
```

### Cell 4: View Results and Best Model

```python
import json
from pathlib import Path
import pandas as pd

# Find latest optuna study
optuna_dirs = sorted(Path("optuna_studies").glob("optuna_*"))
if optuna_dirs:
    latest_study = optuna_dirs[-1]
    print(f"📊 Latest study: {latest_study}\n")

    # Load summary
    summary_path = latest_study / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results = json.load(f)

        # Create comparison table
        data = []
        for model, result in results.items():
            data.append({
                'Model': model,
                'Val Loss': result['best_val_loss'],
                'Learning Rate': result['best_params'].get('lr', 'N/A'),
                'Batch Size': result['best_params'].get('batch_size', 'N/A'),
                'Hidden Channels': result['best_params'].get('hidden_channels', 'N/A'),
            })

        df = pd.DataFrame(data).sort_values('Val Loss')
        print("🏆 Model Comparison:")
        print(df.to_string(index=False))

        # Best model details
        best_model = df.iloc[0]['Model']
        print(f"\n⭐ Best Model: {best_model.upper()}")
        print(f"   Location: {latest_study}/{best_model}/best_model.pth")

        # Show visualization links
        print(f"\n📈 Visualizations for {best_model}:")
        viz_dir = latest_study / best_model
        for viz_file in viz_dir.glob("*.html"):
            print(f"   - {viz_file.name}")
else:
    print("No optimization studies found. Run Cell 3 first.")
```

### Cell 5: Test Best Models on New Data

```python
# Evaluate the optimized models from Cell 3 on fresh test data
import torch
import json
import numpy as np
from pathlib import Path
from src.data import SyntheticFourierDataset
from src.metrics import compute_all_metrics
from src.models import CVResNet, FNO3D, FSCNet, MRFUNet, RADNet, FrequencyGroupedTransformer
import pandas as pd

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODELS = {
    'cvresnet': CVResNet,
    'fno3d': FNO3D,
    'fscnet': FSCNet,
    'mrfunet': MRFUNet,
    'radnet': RADNet,
    'fgt': FrequencyGroupedTransformer,
}

# Find latest optuna study to get volume_size from config
optuna_dirs = sorted(Path("optuna_studies").glob("optuna_*"))
if not optuna_dirs:
    print("No optimization studies found. Run Cell 3 first.")
else:
    latest_study = optuna_dirs[-1]

    # Load study config to get volume_size
    with open(latest_study / 'config.json') as f:
        study_config = json.load(f)
        volume_size = study_config['volume_size']

    print(f"Testing models from: {latest_study}")
    print(f"Volume size: {volume_size}\n")

    # Generate test dataset with same volume size as training
    test_dataset = SyntheticFourierDataset(
        num_samples=50,
        volume_size=volume_size,
        noise_type='gaussian',
        noise_params={'noise_level': 0.1},
        augmentation=False
    )

    results = []
    for model_name in MODELS.keys():
        model_dir = latest_study / model_name
        checkpoint_path = model_dir / "best_model.pth"

        if not checkpoint_path.exists():
            print(f"⚠ {model_name}: No trained model found, skipping")
            continue

        # Load best params
        with open(model_dir / 'best_params.json') as f:
            data = json.load(f)
            params = data['best_params']

        # Create model with best params
        config = {
            'input_shape': (volume_size, volume_size, volume_size // 2 + 1),
            'hidden_channels': params['hidden_channels'],
        }

        # Add model-specific params (same logic as hyperparam_search.py)
        if model_name == 'cvresnet':
            config['num_blocks'] = params['num_blocks']
        elif model_name == 'fno3d':
            config['num_layers'] = params['num_layers']
            config['modes_x'] = params['modes_x']
            config['modes_y'] = params['modes_y']
            config['modes_z'] = params['modes_z']
        elif model_name == 'fscnet':
            config['num_shells'] = params['num_shells']
            config['embedding_dim'] = params['embedding_dim']
        elif model_name == 'mrfunet':
            config['num_bands'] = params['num_bands']
        elif model_name == 'radnet':
            config['num_radial_bins'] = params['num_radial_bins']
            config['num_angular_modes'] = params['num_angular_modes']
        elif model_name == 'fgt':
            config['num_bands'] = params['num_bands']
            config['tokens_per_band'] = params['tokens_per_band']

        model = MODELS[model_name](**config)

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()

        # Evaluate
        all_metrics = []
        with torch.no_grad():
            for noisy, clean in test_dataset:
                noisy = noisy.unsqueeze(0).to(device)
                clean = clean.unsqueeze(0).to(device)
                pred = model(noisy)
                metrics = compute_all_metrics(pred, clean, compute_ssim=True)
                all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        results.append({
            'Model': model_name,
            'PSNR (dB)': avg_metrics['psnr'],
            'SSIM': avg_metrics.get('ssim', 0),
            'MSE (Fourier)': avg_metrics['mse_fourier'],
            'Phase Error': avg_metrics['phase_error'],
        })

        print(f"✓ {model_name}: PSNR = {avg_metrics['psnr']:.2f} dB")

    # Display results
    if results:
        df = pd.DataFrame(results).sort_values('PSNR (dB)', ascending=False)
        print("\n" + "="*70)
        print("📊 TRAINED MODEL PERFORMANCE ON TEST SET")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
```

### Cell 6: Download Best Model

```python
from google.colab import files  # For Colab
# For Kaggle, results are saved in the output directory automatically

# Download best model and visualizations
import shutil
from pathlib import Path

# Find latest study
optuna_dirs = sorted(Path("optuna_studies").glob("optuna_*"))
if optuna_dirs:
    latest_study = optuna_dirs[-1]

    # Create archive
    shutil.make_archive("best_models", "zip", latest_study)
    print(f"✓ Created archive: best_models.zip")

    # Download (Colab only)
    # files.download("best_models.zip")
else:
    print("No studies found")
```

## 📁 Project Structure

```
fourier_data_architecture/
├── src/
│   ├── models/           # 6 model architectures
│   │   ├── cvresnet.py
│   │   ├── fno_3d.py
│   │   ├── fsc_net.py
│   │   ├── mrfu_net.py
│   │   ├── radnet.py
│   │   └── fgt.py
│   ├── data/            # Dataset and noise models
│   └── metrics/         # Evaluation metrics (PSNR, SSIM, phase error, etc.)
├── scripts/
│   ├── train.py         # Single model training
│   ├── hyperparam_search.py  # Automatic HPO with Optuna
│   └── benchmark.py     # Compare all models
└── tests/              # Unit tests
```

## 🔬 Technical Details

### Input Format
All models accept 3D Fourier coefficients of shape `[B, H, W, D, 2]`:
- `B`: Batch size
- `H, W, D`: Spatial dimensions (rfft format: D = original_depth // 2 + 1)
- `2`: Real and imaginary components

### Loss Functions
- **Fourier Loss**: MSE in Fourier space
- **Hybrid Loss**: 0.5 × Fourier MSE + 0.5 × Real-space MSE (recommended)

### Noise Models
- Gaussian noise (configurable σ)
- Poisson noise (photon counting)
- Mixed noise (Gaussian + Poisson)
- Phase-only noise

### Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MSE**: Mean Squared Error (Fourier & Real space)
- **Phase Error**: Angular difference in complex plane
- **Magnitude Error**: Amplitude preservation
- **Per-Band PSNR**: Frequency-band specific quality

## 🎯 Recommended Configurations

### Quick Test (2-5 minutes on GPU)
```python
EPOCHS = 10
NUM_SAMPLES = 100
VOLUME_SIZE = 32
TRIALS = 10
```

### Standard Training (30-60 minutes on GPU)
```python
EPOCHS = 50
NUM_SAMPLES = 500
VOLUME_SIZE = 64
TRIALS = 20
```

### Full Optimization (2-4 hours on GPU)
```python
EPOCHS = 100
NUM_SAMPLES = 1000
VOLUME_SIZE = 64
TRIALS = 50
```

## 📊 Expected Results

After proper training (100+ epochs on synthetic data with Gaussian noise σ=0.1), expect:
- **PSNR**: 25-35 dB (higher is better) - varies by architecture and noise level
- **SSIM**: 0.85-0.95 (1.0 is perfect)
- **Training time**: ~0.1-0.5s per batch on modern GPU
- **Best architectures**: FNO-3D and RAD-Net typically perform best for Fourier denoising

**Note**: Untrained models show ~0 dB PSNR (random weights). Results depend on noise type, noise level, and dataset characteristics.

## 🏆 Hyperparameter Optimization

The automatic HPO system:
1. **Optimizes per architecture** - Each model gets its own Optuna study
2. **Bayesian optimization** - Smarter than grid/random search using TPE sampler
3. **Automatic pruning** - Stops unpromising trials early
4. **Final training** - Best config trained for 3× epochs
5. **Visualizations** - Optimization history, parameter importance, parallel coordinates

### Optimized Hyperparameters

**Common (all models):**
- Learning rate (1e-5 to 1e-2, log scale)
- Batch size (2, 4, 8)
- Scheduler (Plateau, Cosine, None)
- Hidden channels (16, 32, 64)

**Architecture-specific:**
- FNO-3D: Number of layers, frequency modes (x/y/z)
- FSC-Net: Number of shells, embedding dimension
- MRFU-Net: Number of frequency bands
- RAD-Net: Radial bins, angular modes
- FGT: Number of bands, tokens per band
- CV-ResNet: Number of residual blocks

**Fixed (not optimized):**
- Optimizer: AdamW
- Loss: Hybrid (Fourier + Real space)

## 🔧 Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
einops>=0.7.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
optuna>=3.0.0
plotly>=5.14.0
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{fourier_data_architecture,
  title={Fourier Space Deep Learning Architectures for 3D Denoising},
  author={Sanchez-Garcia, Ruben},
  year={2025},
  url={https://github.com/rsanchezgarc/fourier_data_architecture}
}
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 🐛 Issues

Report issues at: https://github.com/rsanchezgarc/fourier_data_architecture/issues

## 🙏 Acknowledgments

- Fourier Neural Operator (FNO) architecture inspired by Li et al.
- All models implemented from scratch for 3D Fourier space processing
- Synthetic data generation based on common cryo-EM noise models
