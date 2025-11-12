"""Benchmark all architectures on same data."""

import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import pandas as pd
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CVResNet, FNO3D, FSCNet, MRFUNet, RADNet, FrequencyGroupedTransformer
from src.data import SyntheticFourierDataset
from src.metrics import compute_all_metrics, MetricsTracker


# Model configurations
MODEL_CONFIGS = {
    'CVResNet': {
        'class': CVResNet,
        'config': {
            'hidden_channels': 32,
            'num_blocks': 16,
        }
    },
    'FNO3D': {
        'class': FNO3D,
        'config': {
            'hidden_channels': 32,
            'num_layers': 4,
            'modes_x': 12,
            'modes_y': 12,
            'modes_z': 12,
        }
    },
    'FSCNet': {
        'class': FSCNet,
        'config': {
            'hidden_channels': 32,
            'num_shells': 16,
            'embedding_dim': 256,
        }
    },
    'MRFUNet': {
        'class': MRFUNet,
        'config': {
            'hidden_channels': 16,
            'num_bands': 4,
        }
    },
    'RADNet': {
        'class': RADNet,
        'config': {
            'hidden_channels': 128,
            'num_radial_bins': 16,
            'num_angular_modes': 16,
        }
    },
    'FGT': {
        'class': FrequencyGroupedTransformer,
        'config': {
            'hidden_channels': 64,
            'num_bands': 16,
            'tokens_per_band': 256,
        }
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark all architectures')

    parser.add_argument('--volume-size', type=int, default=64,
                       help='Size of 3D volumes')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--noise-type', type=str, default='gaussian',
                       choices=['gaussian', 'poisson', 'mixed'],
                       help='Type of noise')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Noise level')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to benchmark (default: all)')

    return parser.parse_args()


def benchmark_model(
    model,
    model_name: str,
    dataloader: DataLoader,
    device: str,
) -> Dict:
    """Benchmark a single model."""
    print(f"\nBenchmarking {model_name}")
    print("-" * 60)

    model = model.to(device)
    model.eval()

    # Get model info
    model_info = model.get_model_info()
    print(f"Parameters: {model_info['parameters']:,}")

    metrics_tracker = MetricsTracker()
    total_time = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Time inference
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            prediction = model(noisy)

            if device == 'cuda':
                torch.cuda.synchronize()

            batch_time = time.time() - start_time
            total_time += batch_time
            num_batches += 1

            # Compute metrics
            batch_metrics = compute_all_metrics(prediction, clean, compute_ssim=True)
            metrics_tracker.update(batch_metrics)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")

    # Average metrics
    avg_metrics = metrics_tracker.get_average()
    avg_time_per_batch = total_time / num_batches
    avg_time_per_sample = total_time / (num_batches * dataloader.batch_size)

    results = {
        'model': model_name,
        'parameters': model_info['parameters'],
        'avg_time_per_batch_ms': avg_time_per_batch * 1000,
        'avg_time_per_sample_ms': avg_time_per_sample * 1000,
        **avg_metrics,
    }

    print(f"\nResults:")
    for key, value in results.items():
        if key != 'model':
            print(f"  {key}: {value:.4f}")

    return results


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Benchmarking Fourier Space Denoising Architectures")
    print("=" * 80)
    print(f"Volume size: {args.volume_size}")
    print(f"Test samples: {args.num_samples}")
    print(f"Noise: {args.noise_type} (level: {args.noise_level})")
    print(f"Device: {args.device}")
    print()

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = SyntheticFourierDataset(
        num_samples=args.num_samples,
        volume_size=args.volume_size,
        noise_type=args.noise_type,
        noise_params={'noise_level': args.noise_level},
        augmentation=False,  # No augmentation for testing
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Determine which models to benchmark
    if args.models is not None:
        models_to_benchmark = {k: v for k, v in MODEL_CONFIGS.items() if k in args.models}
    else:
        models_to_benchmark = MODEL_CONFIGS

    print(f"Benchmarking {len(models_to_benchmark)} models: {list(models_to_benchmark.keys())}")
    print()

    # Benchmark each model
    all_results = []

    for model_name, model_config in models_to_benchmark.items():
        try:
            # Create model
            model_class = model_config['class']
            config = model_config['config'].copy()
            config['input_shape'] = (
                args.volume_size,
                args.volume_size,
                args.volume_size // 2 + 1
            )

            model = model_class(**config)

            # Benchmark
            results = benchmark_model(model, model_name, test_loader, args.device)
            all_results.append(results)

            # Clean up
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n✗ Error benchmarking {model_name}:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    df = pd.DataFrame(all_results)

    # Sort by PSNR (higher is better)
    if 'psnr' in df.columns:
        df = df.sort_values('psnr', ascending=False)

    print(df.to_string(index=False))

    # Save results
    results_file = output_dir / 'benchmark_results.csv'
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Save detailed JSON
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if 'psnr' in df.columns:
        best_model = df.iloc[0]['model']
        best_psnr = df.iloc[0]['psnr']
        print(f"Best PSNR: {best_model} ({best_psnr:.2f} dB)")

    if 'parameters' in df.columns:
        smallest_model = df.loc[df['parameters'].idxmin()]['model']
        smallest_params = df.loc[df['parameters'].idxmin()]['parameters']
        print(f"Smallest model: {smallest_model} ({smallest_params:,} params)")

    if 'avg_time_per_sample_ms' in df.columns:
        fastest_model = df.loc[df['avg_time_per_sample_ms'].idxmin()]['model']
        fastest_time = df.loc[df['avg_time_per_sample_ms'].idxmin()]['avg_time_per_sample_ms']
        print(f"Fastest inference: {fastest_model} ({fastest_time:.2f} ms/sample)")

    print("=" * 80)


if __name__ == '__main__':
    main()
