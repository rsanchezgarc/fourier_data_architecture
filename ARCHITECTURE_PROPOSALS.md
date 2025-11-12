# Architecture Proposals for 3D Fourier Space Processing

## Problem Analysis

**Input**: 64×64×64 volume → **rfft** → 64×64×33 complex coefficients (131,072 values)
- Real FFT exploits conjugate symmetry: only need positive frequencies in one dimension
- Each coefficient has amplitude and phase information
- Challenge: Too many coefficients for naive token-per-coefficient approach

**Key Insight**: Organize by frequency shells (|k| = √(kx² + ky² + kz²))
- Low frequencies (small |k|): Global structure, few coefficients, critical for reconstruction
- High frequencies (large |k|): Fine details, many coefficients, noise-sensitive
- Natural multi-scale hierarchical decomposition

## Proposed Architectures

### 1. **Frequency Shell CNN (FSC-Net)** ⭐ Recommended Baseline

**Core Idea**: Group Fourier coefficients into concentric frequency shells, process with complex-valued CNNs

**Architecture**:
```
Input: 64×64×33 complex coefficients

1. Shell Grouping:
   - Compute |k| = sqrt(kx² + ky² + kz²) for each coefficient
   - Group into N shells (e.g., N=16): [0-2], [2-4], ..., [28-30], [30-32], [32-max]
   - Each shell: irregular 3D points with similar frequency magnitude

2. Per-Shell Processing:
   - Scatter coefficients to 3D grid per shell (padding for uniform shape)
   - Complex Conv3D blocks (separate real/imag or use complex layers)
   - BatchNorm → ReLU/CReLU → Complex Conv3D
   - Reduce spatial dims: 64³ → 32³ → 16³ → 8³

3. Cross-Shell Fusion:
   - Concatenate shell embeddings: [B, N_shells, embedding_dim]
   - 1D convolution or MLP to mix information across shells
   - Low-freq ↔ high-freq interactions

4. Reconstruction:
   - Decoder mirrors encoder
   - Upsample and de-scatter back to original shell positions
   - Output: Denoised 64×64×33 complex coefficients
```

**Variants**:
- **FSC-Net-Lite**: Fewer shells (N=8), smaller conv kernels (3³ → 1³)
- **FSC-Net-Deep**: More residual blocks per shell, deeper fusion network

**Parameters**: ~5-10M (tunable)

---

### 2. **Multi-Resolution Fourier U-Net (MRFU-Net)**

**Core Idea**: Hierarchical encoder-decoder operating on frequency bands

**Architecture**:
```
Input: 64×64×33 complex coefficients

1. Frequency Band Splitting:
   - Split into 4 bands: [0-8], [8-16], [16-24], [24-33] (along rfft dimension)
   - Each band: 64×64×8 or 64×64×9 complex values

2. U-Net per Band:
   - Encoder: Complex Conv3D (k=3) + MaxPool
     64×64×8 → 32×32×4 → 16×16×2 → 8×8×1 (bottleneck)
   - Decoder: TransposeConv3D + Skip Connections
     8×8×1 → 16×16×2 → 32×32×4 → 64×64×8

3. Cross-Band Attention:
   - At bottleneck level, exchange info between bands
   - Multi-head attention: Q,K,V from different bands
   - Low freq guides high freq denoising

4. Fusion:
   - Concatenate band outputs → 64×64×33
   - Final conv layer for refinement
```

**Advantages**:
- Natural hierarchical decomposition
- Skip connections preserve fine details
- Proven U-Net architecture

**Parameters**: ~8-15M

---

### 3. **Fourier Neural Operator (FNO-3D)**

**Core Idea**: Learn convolution kernels directly in Fourier space (spectral convolutions)

**Architecture**:
```
Input: 64×64×33 complex coefficients (already in Fourier space!)

1. Spectral Convolution Layer:
   - Multiply Fourier coefficients by learned weights
   - R(k) = W(k) ⊙ F(k)  where ⊙ is element-wise complex multiply
   - W(k) learned per frequency mode (but shared across spatial locations)

2. FNO Block:
   - Branch 1: Spectral Convolution (in Fourier space)
   - Branch 2: Pointwise 1×1 conv (local features)
   - Combine: σ(SpectralConv + PointwiseConv)

3. Stack FNO Blocks:
   - 4-6 blocks with residual connections
   - Each block: [B, 64, 64, 33, C] → [B, 64, 64, 33, C]

4. Output Layer:
   - Final spectral convolution to reduce channels
   - Output: [B, 64, 64, 33, 2] (real + imag)
```

**Advantages**:
- Operates directly on Fourier coefficients (no conversion needed)
- Global receptive field by design
- Theoretically grounded in operator learning

**Parameters**: ~3-8M (very efficient)

**Challenge**: Original FNO designed for solving PDEs; needs adaptation for denoising

---

### 4. **Radial-Angular Decomposition Network (RAD-Net)**

**Core Idea**: Exploit spherical geometry of Fourier space - separate radial and angular processing

**Architecture**:
```
Input: 64×64×33 complex coefficients

1. Spherical Coordinate Transform:
   - Convert (kx, ky, kz) → (|k|, θ, φ) [radius, polar, azimuthal]
   - Group by radial bins: |k| ∈ [0, 0.5, 1, 2, 4, 8, 16, 32, 46]

2. Radial Processing:
   - For each radial bin, pool angular information
   - 1D convolutions along radial direction: learn k-dependent filters
   - Output: radial embeddings [N_radial_bins, hidden_dim]

3. Angular Processing:
   - For each shell, process angular variations
   - Spherical harmonics basis or learned angular filters
   - Captures directional patterns in Fourier space

4. Recombination:
   - MLP to fuse radial + angular features
   - Decode back to (kx, ky, kz) grid
   - Output: Denoised coefficients
```

**Advantages**:
- Rotation-invariant features (potentially)
- Physically motivated decomposition
- Efficient for isotropic structures

**Parameters**: ~4-7M

**Novel**: Haven't seen this exact approach in literature

---

### 5. **Frequency-Grouped Transformer (FGT)**

**Core Idea**: Group frequencies into bands, use attention within/across bands

**Architecture**:
```
Input: 64×64×33 complex coefficients

1. Frequency Band Tokenization:
   - Divide into B=16 bands along |k| (frequency shells)
   - Each band: Pool to fixed number of tokens (e.g., 256 tokens per band)
   - Total tokens: 16 × 256 = 4,096 tokens

2. Within-Band Self-Attention:
   - Transformer blocks process each band independently
   - Learn local frequency patterns
   - 2-3 transformer layers per band

3. Cross-Band Attention:
   - Exchange information between bands
   - Low-freq tokens attend to high-freq tokens
   - Learn frequency coupling

4. Reconstruction:
   - Upsample tokens back to original 64×64×33 grid
   - Learned upsampling or transposed convolutions
```

**Advantages**:
- Flexible attention mechanism
- Good for long-range dependencies

**Disadvantages**:
- More parameters (~15-25M)
- Slower training

---

### 6. **Octave Fourier Network (OFN)**

**Core Idea**: Split into frequency octaves (like OctConv), process at different rates

**Architecture**:
```
Input: 64×64×33 complex coefficients

1. Octave Splitting:
   - Low freq (0-25% of |k|max): 64×64×8
   - Mid freq (25-75%): 64×64×17
   - High freq (75-100%): 64×64×8

2. Multi-Rate Processing:
   - Low freq: Deeper network (8 layers), small spatial size
   - Mid freq: Medium network (4 layers)
   - High freq: Shallow network (2 layers), focus on denoising

3. Frequency Exchange:
   - Low → High: Upsampling guidance
   - High → Low: Detail refinement feedback
   - Bidirectional information flow

4. Fusion:
   - Combine octaves with learned weights
   - Final refinement layer
```

**Advantages**:
- Computational efficiency (process high-freq cheaply)
- Matches natural frequency importance

**Parameters**: ~6-12M

---

### 7. **Complex-Valued ResNet (CV-ResNet)**

**Core Idea**: Direct end-to-end processing with complex residual blocks

**Architecture**:
```
Input: 64×64×33 complex coefficients

1. Input Projection:
   - Complex Conv3D: [64, 64, 33, 2] → [64, 64, 33, 32]
   - Treat real/imag as channels or use complex-valued layers

2. Complex Residual Blocks (×16):
   - ComplexBatchNorm
   - CReLU or modReLU activation
   - Complex Conv3D (3×3×3)
   - Residual connection: out = in + block(in)

3. Bottleneck:
   - Reduce spatial dims: 64×64×33 → 32×32×17 → 16×16×9
   - Then expand back with transpose convolutions

4. Output Projection:
   - Complex Conv3D: [64, 64, 33, 32] → [64, 64, 33, 2]
```

**Advantages**:
- Simple, proven architecture
- Easy to implement and train
- Strong baseline

**Disadvantages**:
- Doesn't explicitly use frequency structure

**Parameters**: ~10-20M

---

## Recommended Testing Order

1. **CV-ResNet** (baseline - simple and effective)
2. **FSC-Net** (frequency shell grouping - core innovation)
3. **FNO-3D** (spectral method - theoretically motivated)
4. **MRFU-Net** (hierarchical - proven U-Net design)
5. **RAD-Net** (novel - radial-angular decomposition)
6. **OFN** (efficient - octave processing)
7. **FGT** (if others underperform - attention-based)

## Implementation Considerations

### Complex Number Handling

**Option A**: Separate Real/Imaginary Channels
- Stack as [B, H, W, D, 2] where last dim is [real, imag]
- Use standard PyTorch layers
- Simpler implementation

**Option B**: Complex-Valued Layers
- Use libraries like `torch.complex` or `complexPyTorch`
- True complex arithmetic (multiplication, etc.)
- More principled but complex

**Recommendation**: Start with Option A, try Option B if needed

### Frequency Shell Grouping Strategies

1. **Linear shells**: Equal |k| spacing
2. **Logarithmic shells**: log(|k|) spacing (more shells at low freq)
3. **Quantile shells**: Equal number of coefficients per shell
4. **Learned shells**: Learn optimal grouping (advanced)

### Multi-GPU Training Strategy

With 4× RTX 3090 (24GB each):
- **DataParallel**: Simple, replicate model on each GPU
- **DistributedDataParallel**: More efficient, use for final training
- **Batch size**: 8-16 per GPU → 32-64 total
- **Mixed precision**: Use `torch.cuda.amp` for faster training

## Evaluation Metrics

### Primary Metrics
1. **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
2. **SSIM** (Structural Similarity Index): 0-1, higher is better
3. **MSE** (Mean Squared Error): Lower is better

### Secondary Metrics
4. **Fourier Space MSE**: Loss in frequency domain
5. **Phase Error**: Absolute phase difference
6. **Amplitude Error**: Relative amplitude error
7. **Inference Time**: Speed (ms per volume)
8. **Parameters**: Model size

## Next Steps

1. Implement data pipeline (ModelNet40 + voxelization)
2. Implement training framework with metrics
3. Implement all 7 architectures
4. Create automated benchmarking script
5. Run experiments and compare

---

**Questions/Notes**:
- Should we also test on synthetic data (spheres, cubes) for rapid iteration?
- Do we want rotation augmentation in real space before FFT?
- Should we add curriculum learning (easy noise → hard noise)?
