"""Training script for Fourier space denoising models."""

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
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CVResNet, FNO3D, FSCNet, MRFUNet, RADNet, FrequencyGroupedTransformer
from src.data import FourierDenoisingDataset, SyntheticFourierDataset
from src.metrics import compute_all_metrics, MetricsTracker


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
    parser = argparse.ArgumentParser(description='Train Fourier space denoising model')

    # Model
    parser.add_argument('--model', type=str, default='cvresnet', choices=list(MODELS.keys()),
                       help='Model architecture')
    parser.add_argument('--hidden-channels', type=int, default=32,
                       help='Number of hidden channels')

    # Data
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing volume files')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data')
    parser.add_argument('--num-synthetic', type=int, default=1000,
                       help='Number of synthetic samples')
    parser.add_argument('--volume-size', type=int, default=64,
                       help='Size of 3D volumes')
    parser.add_argument('--noise-type', type=str, default='gaussian',
                       choices=['gaussian', 'poisson', 'mixed'],
                       help='Type of noise to add')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Noise level (for Gaussian noise)')

    # Training
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--loss-type', type=str, default='fourier',
                       choices=['fourier', 'real', 'hybrid'],
                       help='Loss function type')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')

    # Logging and checkpointing
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval-every', type=int, default=1,
                       help='Evaluate every N epochs')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    return parser.parse_args()


def create_model(args):
    """Create model based on arguments."""
    model_class = MODELS[args.model]

    # Base config
    config = {
        'input_shape': (args.volume_size, args.volume_size, args.volume_size // 2 + 1),
        'hidden_channels': args.hidden_channels,
    }

    # Model-specific configs
    if args.model == 'cvresnet':
        config['num_blocks'] = 16
    elif args.model == 'fno3d':
        config['num_layers'] = 4
        config['modes_x'] = 12
        config['modes_y'] = 12
        config['modes_z'] = 12
    elif args.model == 'fscnet':
        config['num_shells'] = 16
        config['embedding_dim'] = 256
    elif args.model == 'mrfunet':
        config['num_bands'] = 4
    elif args.model == 'radnet':
        config['num_radial_bins'] = 16
        config['num_angular_modes'] = 16
    elif args.model == 'fgt':
        config['num_bands'] = 16
        config['tokens_per_band'] = 256

    model = model_class(**config)
    return model


def create_datasets(args):
    """Create train and validation datasets."""
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

    return train_dataset, val_dataset


def compute_loss(prediction, target, loss_type='fourier'):
    """Compute loss based on loss type."""
    if loss_type == 'fourier':
        # MSE in Fourier space
        return nn.functional.mse_loss(prediction, target)

    elif loss_type == 'real':
        # MSE in real space
        pred_real = torch.fft.irfftn(
            torch.view_as_complex(prediction.contiguous()),
            dim=(1, 2, 3),
        )
        tgt_real = torch.fft.irfftn(
            torch.view_as_complex(target.contiguous()),
            dim=(1, 2, 3),
        )
        return nn.functional.mse_loss(pred_real, tgt_real)

    elif loss_type == 'hybrid':
        # Combination of both
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


def train_epoch(model, dataloader, optimizer, device, loss_type):
    """Train for one epoch."""
    model.train()
    metrics_tracker = MetricsTracker()

    for batch_idx, (noisy, clean) in enumerate(dataloader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Forward
        optimizer.zero_grad()
        prediction = model(noisy)

        # Loss
        loss = compute_loss(prediction, clean, loss_type)

        # Backward
        loss.backward()
        optimizer.step()

        # Track metrics
        metrics_tracker.update({'loss': loss.item()})

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")

    return metrics_tracker.get_average()


def validate(model, dataloader, device, loss_type):
    """Validate model."""
    model.eval()
    metrics_tracker = MetricsTracker()

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Forward
            prediction = model(noisy)

            # Loss
            loss = compute_loss(prediction, clean, loss_type)

            # Detailed metrics (on first batch only for speed)
            if len(metrics_tracker.metrics) == 0:
                detailed_metrics = compute_all_metrics(prediction, clean, compute_ssim=False)
                metrics_tracker.update(detailed_metrics)

            metrics_tracker.update({'loss': loss.item()})

    return metrics_tracker.get_average()


def main():
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print(f"Training {args.model}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print()

    # Create model
    print("Creating model...")
    model = create_model(args)
    model = model.to(args.device)

    model_info = model.get_model_info()
    print(f"Model: {model_info['name']}")
    print(f"Parameters: {model_info['parameters']:,}")
    print()

    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    # Create scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # Training loop
    best_val_loss = float('inf')
    train_history = []
    val_history = []

    print("Starting training...")
    print()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, args.loss_type)
        epoch_time = time.time() - start_time

        print(f"Training metrics:")
        print(f"  Loss: {train_metrics['loss']:.6f}")
        print(f"  Time: {epoch_time:.2f}s")

        train_history.append(train_metrics)

        # Validate
        if epoch % args.eval_every == 0:
            print("Validating...")
            val_metrics = validate(model, val_loader, args.device, args.loss_type)

            print(f"Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.6f}")

            val_history.append(val_metrics)

            # Update scheduler
            if scheduler is not None:
                if args.scheduler == 'plateau':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': vars(args),
                }, output_dir / 'best_model.pth')
                print(f"  → Saved best model (val_loss: {best_val_loss:.6f})")

        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args),
            }, output_dir / f'checkpoint_epoch{epoch}.pth')

        print()

    # Save final model and training history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
    }, output_dir / 'final_model.pth')

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'train': train_history,
            'val': val_history,
        }, f, indent=2)

    print("=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
