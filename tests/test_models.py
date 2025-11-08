"""Test script for all Fourier space architectures."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CVResNet, FNO3D, FSCNet, MRFUNet, RADNet, FrequencyGroupedTransformer


def test_model(model_class, model_name, **kwargs):
    """
    Test a model with dummy input.

    Args:
        model_class: Model class to instantiate
        model_name: Name for logging
        **kwargs: Arguments to pass to model constructor
    """
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name}")
    print(f"{'=' * 60}")

    try:
        # Create model
        model = model_class(**kwargs)
        print(f"✓ Model created successfully")

        # Print model info
        info = model.get_model_info()
        print(f"\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Create dummy input
        batch_size = 2
        H, W, D = model.input_shape
        x = torch.randn(batch_size, H, W, D, 2)
        print(f"\nInput shape: {x.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"Output shape: {output.shape}")

        # Check output shape matches input
        assert output.shape == x.shape, f"Output shape mismatch! Expected {x.shape}, got {output.shape}"
        print(f"✓ Output shape matches input")

        # Check for NaNs
        assert not torch.isnan(output).any(), "Output contains NaNs!"
        print(f"✓ No NaNs in output")

        # Check for Infs
        assert not torch.isinf(output).any(), "Output contains Infs!"
        print(f"✓ No Infs in output")

        print(f"\n✓ {model_name} PASSED all tests")
        return True

    except Exception as e:
        print(f"\n✗ {model_name} FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all models."""
    print("=" * 60)
    print("Testing Fourier Space Deep Learning Architectures")
    print("=" * 60)

    # Common settings for smaller/faster testing
    input_shape = (32, 32, 17)  # Smaller for faster testing (would be 64, 64, 33 in practice)
    hidden_channels = 16  # Reduced for faster testing

    results = {}

    # Test 1: CV-ResNet
    results['CVResNet'] = test_model(
        CVResNet,
        "CV-ResNet (Complex-Valued ResNet)",
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        num_blocks=4,  # Reduced for testing
    )

    # Test 2: FNO-3D
    results['FNO3D'] = test_model(
        FNO3D,
        "FNO-3D (Fourier Neural Operator)",
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        num_layers=2,  # Reduced for testing
        modes_x=8,
        modes_y=8,
        modes_z=8,
    )

    # Test 3: FSC-Net
    results['FSCNet'] = test_model(
        FSCNet,
        "FSC-Net (Frequency Shell CNN)",
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        num_shells=8,  # Reduced for testing
        embedding_dim=128,
    )

    # Test 4: MRFU-Net
    results['MRFUNet'] = test_model(
        MRFUNet,
        "MRFU-Net (Multi-Resolution Fourier U-Net)",
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        num_bands=4,
    )

    # Test 5: RAD-Net
    results['RADNet'] = test_model(
        RADNet,
        "RAD-Net (Radial-Angular Decomposition)",
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        num_radial_bins=8,  # Reduced for testing
        num_angular_modes=8,
    )

    # Test 6: FGT (Frequency-Grouped Transformer)
    results['FGT'] = test_model(
        FrequencyGroupedTransformer,
        "FGT (Frequency-Grouped Transformer)",
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        num_bands=8,  # Reduced for testing
        tokens_per_band=64,  # Reduced for testing
        num_within_layers=1,
        num_cross_layers=1,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for model_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{model_name:20s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
