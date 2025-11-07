# Project Requirements: Scalable Deep Learning for 3D Fourier Space Processing

## 1. Problem Statement

Develop and evaluate scalable deep learning architectures that can effectively process 3D volumetric data in Fourier space, specifically for denoising tasks where both phase and amplitude information have been corrupted.

## 2. Data Specifications

### Input Data
- **Type**: 3D volumetric objects (real-world or synthetic)
- **Dimensions**: 64×64×64 voxels
- **Transform**: Real FFT (rfft) to reduce memory footprint
- **Perturbation**: Random corruption of both:
  - Fourier amplitudes
  - Fourier phases

### Output Data
- **Type**: Reconstructed 3D volume in real space
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

### Options to Consider
- [ ] **ShapeNet**: Large-scale 3D object dataset (synthetic)
- [ ] **ModelNet**: 3D CAD models (synthetic)
- [ ] **Medical imaging**: CT/MRI volumes (if available)
- [ ] **Synthetic**: Generated geometric primitives
- [ ] **Other**: [To be determined]

## 6. Architecture Candidates

### Baseline Approaches
1. **Fourier Image Transformer (FIT)** - Adapted for 3D with reduced tokens
2. **Fourier Neural Operator (FNO)** - If applicable to this task
3. **U-Net in Fourier Space** - CNN-based baseline

### Novel Scalable Approaches
- [ ] Hierarchical Fourier tokenization
- [ ] Frequency-band grouped processing
- [ ] Sparse attention on important frequencies
- [ ] Compressed/learned Fourier representations

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

## 8. Questions to Resolve

1. **Perturbation strategy**: What noise levels? Gaussian, uniform, or other distributions?
2. **Dataset**: Which 3D dataset should we prioritize?
3. **Tokenization**: How should we group Fourier coefficients to reduce token count?
4. **Training regime**: Batch size, epochs, learning rate schedule?
5. **Hardware constraints**: Available GPU memory and compute?

---

**Next Steps**: Review and refine these requirements, then begin with Phase 1 implementation.
