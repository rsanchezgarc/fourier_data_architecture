# Project Requirements: Scalable Deep Learning for 3D Fourier Space Processing

## 1. Problem Statement

Develop and evaluate scalable deep learning architectures that can effectively process 3D volumetric data in Fourier space, specifically for denoising tasks where both phase and amplitude information have been corrupted.

## 2. Data Specifications

### Input Data
- **Type**: 3D volumetric objects from ModelNet40 dataset
- **Dimensions**: 64×64×64 voxels
- **Transform**: Real FFT (rfft) to reduce memory footprint → 64×64×33 complex coefficients
- **Perturbation**: Uniform random corruption:
  - **Amplitude noise**: 10-30% uniform perturbation
  - **Phase noise**: 0-π uniform perturbation

### Output Data
- **Type**: Reconstructed 3D volume in real space (via inverse FFT)
- **Evaluation**: Compare against ground truth original signal

## 3. Architecture Requirements

### Scalability Constraints
- **Cannot** use one token per Fourier coefficient (would result in ~33K tokens for rfft of 64³)
- Must be memory-efficient for 3D data processing
- Should handle complex-valued Fourier coefficients (amplitude + phase)

### Design Considerations
- [ ] Tokenization strategy for Fourier coefficients (grouping/pooling?)
- [ ] Handling of complex-valued data (separate real/imag channels vs. magnitude/phase?)
- [ ] Architecture backbone (Transformer, CNN, hybrid, or novel?)
- [ ] Multi-scale or hierarchical processing?
- [ ] Symmetry preservation in Fourier space?

## 4. Task Definition

### Training Task
1. Load 3D volumetric data
2. Apply rfft to convert to Fourier space
3. Add random perturbations to:
   - Amplitudes: `perturbed_amp = original_amp * noise_factor`
   - Phases: `perturbed_phase = original_phase + noise_offset`
4. Feed perturbed Fourier representation to model
5. Model outputs denoised Fourier representation
6. Apply inverse FFT to get reconstructed volume
7. Compare reconstruction to original in real space

### Evaluation Metrics
- [ ] Peak Signal-to-Noise Ratio (PSNR)
- [ ] Structural Similarity Index (SSIM)
- [ ] Mean Squared Error (MSE)
- [ ] Fourier space metrics (optional)?

## 5. Datasets

### Primary Dataset: ModelNet40
- **Size**: ~12,311 3D CAD models across 40 categories
- **Download**: ~500MB compressed, ~2GB uncompressed
- **Format**: OFF mesh files (will be voxelized to 64³)
- **Split**: 9,843 training / 2,468 test
- **Categories**: Airplane, car, chair, desk, lamp, plant, sofa, etc.
- **Voxelization**: Using binvox or custom voxelizer

### Alternative: ModelNet10
- **Size**: 4,899 models across 10 categories
- **Use case**: Faster prototyping and debugging

## 6. Architecture Candidates

See `ARCHITECTURE_PROPOSALS.md` for detailed designs. Summary:

1. **Complex-Valued ResNet (CV-ResNet)** - Baseline, simple residual architecture
2. **Frequency Shell CNN (FSC-Net)** ⭐ - Group by |k|, process shells with complex CNNs
3. **Fourier Neural Operator (FNO-3D)** - Spectral convolutions in Fourier space
4. **Multi-Resolution Fourier U-Net (MRFU-Net)** - Hierarchical frequency band processing
5. **Radial-Angular Decomposition Network (RAD-Net)** - Spherical coordinate processing
6. **Octave Fourier Network (OFN)** - Multi-rate frequency octave processing
7. **Frequency-Grouped Transformer (FGT)** - Attention within/across frequency bands

### Key Innovation: Frequency Shell Grouping
- Group coefficients by |k| = √(kx² + ky² + kz²)
- Process similar frequencies together
- Natural multi-scale hierarchical structure

## 7. Implementation Phases

### Phase 1: Setup & Data Pipeline
- [ ] Dataset selection and loading
- [ ] FFT/RFFT utilities
- [ ] Perturbation mechanisms
- [ ] Evaluation metrics

### Phase 2: Baseline Implementation
- [ ] Simple CNN/MLP baseline in Fourier space
- [ ] Validate training pipeline

### Phase 3: Advanced Architectures
- [ ] Implement 2-3 scalable architectures
- [ ] Comparative evaluation

### Phase 4: Optimization & Analysis
- [ ] Hyperparameter tuning
- [ ] Ablation studies
- [ ] Performance analysis

## 8. Technical Specifications

### Hardware
- **GPUs**: 4× NVIDIA RTX 3090 (24GB VRAM each)
- **Training**: DistributedDataParallel across GPUs
- **Batch size**: 8-16 per GPU → 32-64 total
- **Mixed precision**: FP16 with torch.cuda.amp

### Framework
- **PyTorch**: 2.0+ with native complex number support
- **Additional libraries**:
  - `trimesh` or `pyvista` for mesh loading
  - `binvox` for voxelization
  - `torch-complex` (if needed for complex layers)

### Training Configuration
- **Optimizer**: AdamW with cosine annealing
- **Learning rate**: 1e-3 with warmup
- **Epochs**: 50-100 (early stopping based on validation)
- **Loss**: MSE in real space + optional Fourier space loss

---

## 9. Open Questions

1. **Rotation augmentation**: Should we augment with rotations before FFT?
2. **Curriculum learning**: Start with easy noise → increase difficulty?
3. **Synthetic warmstart**: Prototype on simple shapes (spheres, cubes) first?
4. **Fourier space loss**: Should we add loss directly in frequency domain?

---

**Status**: Requirements finalized. Ready for implementation.
