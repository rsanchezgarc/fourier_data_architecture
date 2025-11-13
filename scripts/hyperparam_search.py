"""Automatic hyperparameter optimization using Optuna for all architectures."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CVResNet, FNO3D, FSCNet, MRFUNet, RADNet, FrequencyGroupedTransformer
from src.data import FourierDenoisingDataset, SyntheticFourierDataset
from src.metrics import compute_all_metrics, MetricsTracker

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


# Model registry
MODELS = {
    'cvresnet': CVResNet,
    'fno3d': FNO3D,
    'fscnet': FSCNet,
    'mrfunet': MRFUNet,
    'radnet': RADNet,
    'fgt': FrequencyGroupedTransformer,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for Fourier denoising models')

    # Models to optimize
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                       choices=list(MODELS.keys()),
                       help='Models to optimize (default: all)')

    # Data
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing volume files')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data')
    parser.add_argument('--num-synthetic', type=int, default=500,
                       help='Number of synthetic samples')
    parser.add_argument('--volume-size', type=int, default=64,
                       help='Size of 3D volumes')
    parser.add_argument('--noise-type', type=str, default='gaussian',
                       choices=['gaussian', 'poisson', 'mixed'],
                       help='Type of noise to add')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Noise level (for Gaussian noise)')

    # Optuna settings
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials per model')
    parser.add_argument('--n-startup-trials', type=int, default=10,
                       help='Number of random trials before optimization')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for optimization')
    parser.add_argument('--pruning', action='store_true',
                       help='Enable pruning of unpromising trials')

    # Training settings for trials
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs per trial')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')

    # Output
    parser.add_argument('--output-dir', type=str, default='optuna_studies',
                       help='Output directory for studies')
    parser.add_argument('--storage', type=str, default=None,
                       help='Optuna storage URL (e.g., sqlite:///optuna.db)')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    return parser.parse_args()


def suggest_hyperparameters(trial: optuna.Trial, model_name: str, volume_size: int):
    """Suggest hyperparameters for a specific model."""

    # Common hyperparameters
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8]),
        'optimizer': 'adamw',  # Fixed
        'scheduler': trial.suggest_categorical('scheduler', ['plateau', 'cosine', 'none']),
        'loss_type': 'hybrid',  # Fixed
        'hidden_channels': trial.suggest_categorical('hidden_channels', [16, 32, 64]),
    }

    # Model-specific hyperparameters
    rfft_depth = volume_size // 2 + 1

    if model_name == 'cvresnet':
        params['num_blocks'] = trial.suggest_int('num_blocks', 8, 24, step=4)

    elif model_name == 'fno3d':
        params['num_layers'] = trial.suggest_int('num_layers', 3, 6)
        max_modes = min(volume_size // 4, rfft_depth - 1)
        params['modes_x'] = trial.suggest_int('modes_x', 8, max_modes)
        params['modes_y'] = trial.suggest_int('modes_y', 8, max_modes)
        params['modes_z'] = trial.suggest_int('modes_z', 4, min(16, rfft_depth - 1))

    elif model_name == 'fscnet':
        params['num_shells'] = trial.suggest_int('num_shells', 8, 32, step=4)
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [128, 256, 512])

    elif model_name == 'mrfunet':
        params['num_bands'] = trial.suggest_int('num_bands', 3, 8)

    elif model_name == 'radnet':
        params['num_radial_bins'] = trial.suggest_int('num_radial_bins', 8, 32, step=4)
        params['num_angular_modes'] = trial.suggest_int('num_angular_modes', 8, 32, step=4)

    elif model_name == 'fgt':
        params['num_bands'] = trial.suggest_int('num_bands', 8, 24, step=4)
        params['tokens_per_band'] = trial.suggest_categorical('tokens_per_band', [128, 256, 512])

    return params


def create_model(model_name: str, params: dict, volume_size: int):
    """Create model with given hyperparameters."""
    model_class = MODELS[model_name]

    # Base config
    config = {
        'input_shape': (volume_size, volume_size, volume_size // 2 + 1),
        'hidden_channels': params['hidden_channels'],
    }

    # Add model-specific parameters
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

    return model_class(**config)


def compute_loss(prediction, target, loss_type='fourier'):
    """Compute loss based on loss type."""
    if loss_type == 'fourier':
        return nn.functional.mse_loss(prediction, target)

    elif loss_type == 'hybrid':
        loss_fourier = nn.functional.mse_loss(prediction, target)

        pred_real = torch.fft.irfftn(
            torch.view_as_complex(prediction.contiguous()),
            dim=(1, 2, 3),
        )
        tgt_real = torch.fft.irfftn(
            torch.view_as_complex(target.contiguous()),
            dim=(1, 2, 3),
        )
        loss_real = nn.functional.mse_loss(pred_real, tgt_real)

        return 0.5 * loss_fourier + 0.5 * loss_real


def train_and_evaluate(model, train_loader, val_loader, params, device, epochs, trial=None):
    """Train model and return validation loss."""
    model = model.to(device)

    # Create optimizer (use fixed value if not in params)
    optimizer_type = params.get('optimizer', 'adamw')
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=params['lr'])
    else:
        optimizer = AdamW(model.parameters(), lr=params['lr'])

    # Create scheduler
    scheduler_type = params.get('scheduler', 'plateau')
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    best_val_loss = float('inf')
    loss_type = params.get('loss_type', 'hybrid')

    for epoch in range(epochs):
        # Train
        model.train()
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            prediction = model(noisy)
            loss = compute_loss(prediction, clean, loss_type)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                prediction = model(noisy)
                loss = compute_loss(prediction, clean, loss_type)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)

        # Update scheduler
        if scheduler is not None:
            if params['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Report intermediate value for pruning
        if trial is not None:
            trial.report(val_loss, epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_loss


def create_objective(model_name, train_loader, val_loader, args):
    """Create objective function for Optuna."""

    def objective(trial):
        # Suggest hyperparameters
        params = suggest_hyperparameters(trial, model_name, args.volume_size)

        # Create model
        try:
            model = create_model(model_name, params, args.volume_size)
        except Exception as e:
            # Invalid hyperparameter combination
            print(f"Failed to create model: {e}")
            raise optuna.TrialPruned()

        # Update dataloaders with new batch size
        current_train_loader = DataLoader(
            train_loader.dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False,
        )

        current_val_loader = DataLoader(
            val_loader.dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False,
        )

        # Train and evaluate
        try:
            val_loss = train_and_evaluate(
                model, current_train_loader, current_val_loader,
                params, args.device, args.epochs, trial
            )
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM error, pruning trial")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

        return val_loss

    return objective


def optimize_model(model_name, train_dataset, val_dataset, args, output_dir):
    """Run hyperparameter optimization for a single model."""

    print("=" * 80)
    print(f"Optimizing {model_name.upper()}")
    print("=" * 80)

    # Create dataloaders (initial batch size, will be updated per trial)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    # Create study
    study_name = f"{model_name}_study"

    if args.pruning:
        pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=5)
    else:
        pruner = None

    sampler = TPESampler(n_startup_trials=args.n_startup_trials, seed=42)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    # Create objective
    objective = create_objective(model_name, train_loader, val_loader, args)

    # Optimize
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - Best Trial Results")
    print(f"{'='*60}")
    print(f"  Validation Loss: {study.best_trial.value:.6f}")
    print(f"  Trial Number: {study.best_trial.number}")
    print(f"\n  Best Hyperparameters:")
    for key, value in sorted(study.best_trial.params.items()):
        if isinstance(value, float):
            print(f"    {key:20s}: {value:.6e}" if value < 0.01 else f"    {key:20s}: {value:.4f}")
        else:
            print(f"    {key:20s}: {value}")

    # Save study results
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Save best params (include fixed values)
    best_params_full = dict(study.best_trial.params)
    best_params_full['optimizer'] = 'adamw'  # Add fixed value
    best_params_full['loss_type'] = 'hybrid'  # Add fixed value

    with open(model_output_dir / 'best_params.json', 'w') as f:
        json.dump({
            'best_value': study.best_trial.value,
            'best_params': best_params_full,
            'n_trials': len(study.trials),
        }, f, indent=2)

    # Save all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
        })

    with open(model_output_dir / 'all_trials.json', 'w') as f:
        json.dump(trials_data, f, indent=2)

    # Generate and save plots
    try:
        import plotly

        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(model_output_dir / 'optimization_history.html'))

        # Parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(model_output_dir / 'param_importances.html'))

        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(model_output_dir / 'parallel_coordinate.html'))

        print(f"  Visualization saved to {model_output_dir}")
    except ImportError:
        print("  Install plotly for visualization: pip install plotly")

    # Train final model with best params
    print(f"\n{'─'*60}")
    print(f"Training final {model_name} model with best hyperparameters...")
    print(f"  Epochs: {args.epochs * 3} (3x search epochs)")

    # Use the full params (with fixed values) for final training
    best_params = best_params_full

    # Create final model
    model = create_model(model_name, best_params, args.volume_size)

    # Create final dataloaders
    final_train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    final_val_loader = DataLoader(
        val_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    # Train final model for more epochs
    final_epochs = args.epochs * 3  # Train longer for final model
    final_val_loss = train_and_evaluate(
        model, final_train_loader, final_val_loader,
        best_params, args.device, final_epochs
    )

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': best_params,
        'val_loss': final_val_loss,
    }, model_output_dir / 'best_model.pth')

    print(f"  ✓ Final model saved: {model_output_dir / 'best_model.pth'}")
    print(f"  ✓ Final val_loss: {final_val_loss:.6f}")
    print(f"{'─'*60}\n")

    return study.best_trial.value, best_params_full


def main():
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"optuna_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("AUTOMATIC HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Models to optimize: {', '.join(args.models)}")
    print(f"Trials per model: {args.n_trials}")
    print()

    # Create dataset once (shared across all models)
    print("Creating dataset...")
    noise_params = {'noise_level': args.noise_level}

    if args.synthetic:
        dataset = SyntheticFourierDataset(
            num_samples=args.num_synthetic,
            volume_size=args.volume_size,
            noise_type=args.noise_type,
            noise_params=noise_params,
            augmentation=True,
        )
    else:
        dataset = FourierDenoisingDataset(
            data_dir=args.data_dir,
            volume_size=args.volume_size,
            noise_type=args.noise_type,
            noise_params=noise_params,
            augmentation=True,
        )

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Optimize each model
    results = {}

    for model_name in args.models:
        try:
            best_value, best_params = optimize_model(
                model_name, train_dataset, val_dataset, args, output_dir
            )
            results[model_name] = {
                'best_val_loss': best_value,
                'best_params': best_params,
            }
        except Exception as e:
            print(f"ERROR optimizing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print final summary table
    print("\n" + "=" * 100)
    print("OPTIMIZATION COMPLETE - SUMMARY")
    print("=" * 100)

    if results:
        # Sort by best validation loss
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_loss'])

        # Print header
        print(f"\n{'Rank':<6} {'Model':<12} {'Val Loss':<12} {'Learning Rate':<14} {'Batch Size':<12} {'Hidden Ch':<12}")
        print("-" * 100)

        # Print each model
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            val_loss = result['best_val_loss']
            params = result['best_params']
            lr = params.get('lr', 'N/A')
            batch_size = params.get('batch_size', 'N/A')
            hidden_ch = params.get('hidden_channels', 'N/A')

            # Format learning rate
            if isinstance(lr, float):
                lr_str = f"{lr:.2e}"
            else:
                lr_str = str(lr)

            marker = "★" if rank == 1 else " "
            print(f"{rank:<6} {model_name:<12} {val_loss:<12.6f} {lr_str:<14} {batch_size!s:<12} {hidden_ch!s:<12} {marker}")

        print("-" * 100)

        # Print best model details
        best_model, best_result = sorted_results[0]
        print(f"\n★ BEST MODEL: {best_model.upper()}")
        print(f"  Val Loss: {best_result['best_val_loss']:.6f}")
        print(f"  Best hyperparameters:")
        for key, value in sorted(best_result['best_params'].items()):
            if isinstance(value, float):
                print(f"    {key:20s}: {value:.6e}" if value < 0.01 else f"    {key:20s}: {value:.4f}")
            else:
                print(f"    {key:20s}: {value}")

        print(f"\n  Trained model saved: {output_dir}/{best_model}/best_model.pth")
    else:
        print("\nNo models were successfully optimized.")

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 100)


if __name__ == '__main__':
    main()
