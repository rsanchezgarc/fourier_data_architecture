# Detailed Architecture Mechanics

This document explains exactly how each architecture processes data, answering key questions about kernel application, frequency handling, and data flow.

---

## 1. Complex-Valued ResNet (CV-ResNet)

### Question: Does it apply the same kernels at all frequencies?

**Answer: YES** - This is actually a limitation of the naive approach.

### Data Flow

```
Input: [B, 64, 64, 33, 2]  # Batch, Height, Width, Depth_rfft, [real, imag]
       ↓
[Initial Projection]
Conv3D (k=3×3×3, stride=1)
       ↓
[B, 64, 64, 33, C=32]  # C = channels
       ↓
[Residual Block 1-16]
For each block:
  x_in → ComplexBatchNorm
       → Activation (CReLU or separate ReLU on real/imag)
       → Conv3D (k=3×3×3, padding=1)
       → ComplexBatchNorm
       → Activation
       → Conv3D (k=3×3×3, padding=1)
       → x_out = x_in + Conv(x_in)  # Residual connection
       ↓
[B, 64, 64, 33, 32]
       ↓
[Output Projection]
Conv3D (k=1×1×1) to reduce channels
       ↓
[B, 64, 64, 33, 2]  # Back to [real, imag]
       ↓
Output: Denoised Fourier coefficients
```

### How Convolutions Work Here

**Spatial sliding window**: A 3×3×3 kernel slides over the 64×64×33 grid
- At position (kx=10, ky=20, kz=5), the kernel sees a 3×3×3 neighborhood
- Same kernel weights are applied everywhere (translation invariance)
- **Problem**: Low frequencies (small |k|) and high frequencies (large |k|) are treated identically!

### Complex Arithmetic

**Option A: Separate Real/Imaginary Channels**
```python
# Input: [B, H, W, D, 2] where last dim is [real, imag]
real_part = x[..., 0]  # [B, H, W, D]
imag_part = x[..., 1]  # [B, H, W, D]

# Stack as channels
x = torch.stack([real_part, imag_part], dim=1)  # [B, 2, H, W, D]

# Standard Conv3D (treats real/imag as separate channels)
out = conv3d(x)  # [B, C_out, H, W, D]

# This is NOT true complex multiplication!
# Just treating real/imag as independent channels
```

**Option B: True Complex Convolution**
```python
# For complex conv: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
# Need two sets of kernels: W_real and W_imag

real_out = conv(real_in, W_real) - conv(imag_in, W_imag)
imag_out = conv(real_in, W_imag) + conv(imag_in, W_real)
```

**Most implementations use Option A** for simplicity, but it's not mathematically correct for complex numbers.

### Why This Is Naive

1. **No frequency awareness**: Kernel at low freq (kz=2) same as high freq (kz=30)
2. **Ignores spherical geometry**: Fourier space has radial structure, but convolution is Cartesian
3. **DC component mixed with high freq**: (0,0,0) processed same as (32,32,16)

**Advantage**: Simple, proven architecture, good baseline

---

## 2. Fourier Neural Operator (FNO-3D)

### Question: Does it go from real to Fourier and back repeatedly?

**Answer: NO!** - It operates **entirely in Fourier space**. Your input is already in Fourier space, so it never goes back to real space (except at the very end for evaluation).

### Key Insight

FNO is designed for learning operators on function spaces. In our case:
- **Input function**: Noisy Fourier coefficients F_noisy(k)
- **Output function**: Clean Fourier coefficients F_clean(k)
- **Operator to learn**: Denoising map F_clean = G(F_noisy)

### Data Flow

```
Input: [B, 64, 64, 33, 2]  # Already in Fourier space!
       ↓
[Lifting Layer] - Project to higher dimensional space
P: R² → R^v  (2 channels → v channels, typically v=32-64)
       ↓
[B, 64, 64, 33, v]
       ↓
[FNO Layer 1]
  ┌─────────────────────────────┐
  │  Branch 1: Spectral Conv    │
  │  - Multiply by learned R(k) │
  │  - Global in Fourier space  │
  └─────────────────────────────┘
  ┌─────────────────────────────┐
  │  Branch 2: Local Features   │
  │  - 1×1×1 Conv (pointwise)   │
  └─────────────────────────────┘
  Combine: σ(SpectralConv + Local)
       ↓
[FNO Layers 2-4] - Repeat
       ↓
[Projection Layer] - Back to 2 channels
Q: R^v → R²
       ↓
[B, 64, 64, 33, 2]  # Denoised Fourier coefficients
       ↓
IFFT (only for evaluation/loss)
       ↓
[B, 64, 64, 64]  # Real space (for computing loss)
```

### Spectral Convolution Layer - The Magic

**Standard Convolution in Real Space**:
```python
# Real space: (f * g)(x) = ∫ f(x-y) g(y) dy
# In Fourier: F{f * g} = F{f} · F{g}  (multiplication!)
```

**FNO Spectral Convolution**:
```python
# Input: Fourier coefficients v(k) of shape [B, Kx, Ky, Kz, C]
# Learn: Weight matrix R(k) for each frequency k

# For each frequency mode k:
v_out(k) = R(k) @ v_in(k)  # Matrix-vector product

# Where R(k) is a learnable C×C matrix
# Different matrix for each k!
```

**Detailed Implementation**:
```python
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # We only learn weights for low frequency modes (truncated)
        # modes_x, modes_y, modes_z = how many modes to keep (e.g., 12, 12, 12)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        # Learnable weights for each frequency mode
        # Complex-valued weights
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels,
                       modes_x, modes_y, modes_z,
                       dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        # x: [B, C_in, Kx, Ky, Kz] - Already in Fourier space!
        B, C, Kx, Ky, Kz = x.shape

        # Initialize output
        out = torch.zeros(B, self.out_channels, Kx, Ky, Kz,
                         dtype=torch.cfloat, device=x.device)

        # Only process low frequency modes (computational efficiency)
        # For each mode (kx, ky, kz) in the truncated set:
        out[:, :, :self.modes_x, :self.modes_y, :self.modes_z] = \
            self.compl_mul3d(
                x[:, :, :self.modes_x, :self.modes_y, :self.modes_z],
                self.weights
            )

        return out

    def compl_mul3d(self, input, weights):
        # input: [B, C_in, modes_x, modes_y, modes_z]
        # weights: [C_in, C_out, modes_x, modes_y, modes_z]
        # Einstein summation for complex matrix multiply
        return torch.einsum("bicxy,iocxy->bocxy", input, weights)
```

### Why FNO is Different

1. **Global receptive field**: Each R(k) can see all spatial locations (because we're in Fourier space)
2. **Frequency-specific**: Different weights for different k (unlike CV-ResNet)
3. **Resolution invariant**: Can train on 64³ and test on 128³ (theoretically)
4. **Never leaves Fourier space**: Perfect for our use case!

### FNO Layer Structure

```python
class FNOLayer(nn.Module):
    def forward(self, x):
        # x is in Fourier space: [B, C, Kx, Ky, Kz]

        # Branch 1: Spectral convolution (global)
        x1 = self.spectral_conv(x)  # Learns frequency-specific filters

        # Branch 2: Pointwise convolution (local)
        x2 = self.pointwise_conv(x)  # 1x1x1 conv, channel mixing

        # Combine
        out = self.activation(x1 + x2)

        return out
```

**Key difference from CV-ResNet**:
- CV-ResNet: Same kernel everywhere, 3×3×3 spatial
- FNO: Different weights per frequency, 1×1×1 spatial but global due to Fourier

---

## 3. Frequency Shell CNN (FSC-Net) - The Key Innovation

### Question: How does shell grouping work mechanistically?

**Answer**: Group coefficients by frequency magnitude |k|, process each shell separately, then fuse.

### Mathematical Setup

**Frequency magnitude**:
```python
# For each Fourier coefficient at grid position (kx, ky, kz):
k_mag = sqrt(kx² + ky² + kz²)

# Example positions:
# (0, 0, 0): |k| = 0 (DC component)
# (1, 0, 0): |k| = 1
# (1, 1, 0): |k| = √2 ≈ 1.41
# (32, 32, 16): |k| ≈ 42.4
```

**Shell boundaries**:
```python
# For 64×64×33 volume, max |k| ≈ sqrt(32² + 32² + 32²) ≈ 55
# Divide into N=16 shells with logarithmic spacing:

shells = [
    [0, 1],      # Shell 0: DC and very low freq
    [1, 2],      # Shell 1
    [2, 4],      # Shell 2
    [4, 8],      # Shell 3
    [8, 12],     # ...
    [12, 16],
    [16, 20],
    [20, 24],
    [24, 28],
    [28, 32],
    [32, 36],
    [36, 40],
    [40, 44],
    [44, 48],
    [48, 52],
    [52, 60]     # Shell 15: Highest frequencies
]
```

### Data Flow - Detailed

```
Input: [B, 64, 64, 33, 2]  # Fourier coefficients
       ↓
[Shell Assignment]
For each coefficient at (kx, ky, kz):
  1. Compute |k| = sqrt(kx² + ky² + kz²)
  2. Assign to shell_idx based on |k| range
  3. Create shell_mask[shell_idx, kx, ky, kz] = True
       ↓
{Shell 0: coefficients with |k| ∈ [0, 1],
 Shell 1: coefficients with |k| ∈ [1, 2],
 ...
 Shell 15: coefficients with |k| ∈ [52, 60]}
       ↓
[Per-Shell Processing - PARALLEL]
For shell_idx in range(16):
  # Extract coefficients in this shell
  shell_data = x[shell_masks[shell_idx]]  # Irregular shape!

  # Scatter to regular grid (with padding)
  # E.g., Shell 0 might have ~100 coefficients → pad to 8×8×8
  # Shell 15 might have ~10,000 coefficients → reduce to 32×32×16

  grid_shell = scatter_to_grid(shell_data, target_size)

  # 3D CNN processing
  features = conv_block(grid_shell)  # [B, 8, 8, 8, C]

  # Pool to fixed embedding size
  shell_embedding[shell_idx] = adaptive_pool(features)  # [B, embedding_dim]
       ↓
shell_embeddings: [B, 16, 256]  # 16 shells, 256-dim each
       ↓
[Cross-Shell Fusion]
# Mix information across frequency shells
# Low frequencies ↔ High frequencies
fused = transformer_or_mlp(shell_embeddings)  # [B, 16, 256]
       ↓
[Reconstruction]
For shell_idx in range(16):
  # Decode shell embedding back to Fourier coefficients
  shell_coeffs = decoder(fused[shell_idx])

  # Gather back to original (kx, ky, kz) positions
  output[shell_masks[shell_idx]] = shell_coeffs
       ↓
Output: [B, 64, 64, 33, 2]  # Denoised Fourier coefficients
```

### Pseudocode

```python
class FSCNet(nn.Module):
    def __init__(self, num_shells=16, embedding_dim=256):
        self.num_shells = num_shells

        # Create shell boundaries (logarithmic spacing)
        self.shell_bounds = create_shell_boundaries(num_shells)

        # Per-shell encoders (different CNN for each shell!)
        self.shell_encoders = nn.ModuleList([
            ShellEncoder(expected_size_for_shell_i)
            for i in range(num_shells)
        ])

        # Cross-shell fusion
        self.fusion = CrossShellFusion(num_shells, embedding_dim)

        # Per-shell decoders
        self.shell_decoders = nn.ModuleList([
            ShellDecoder(expected_size_for_shell_i)
            for i in range(num_shells)
        ])

    def forward(self, fourier_coeffs):
        B, Kx, Ky, Kz, C = fourier_coeffs.shape

        # 1. Assign each coefficient to a shell
        shell_assignments = self.assign_to_shells(fourier_coeffs)
        # shell_assignments[i] = list of (kx, ky, kz) indices in shell i

        # 2. Process each shell
        shell_embeddings = []
        for shell_idx in range(self.num_shells):
            # Extract coefficients in this shell
            indices = shell_assignments[shell_idx]
            shell_data = fourier_coeffs[:, indices[:, 0],
                                         indices[:, 1],
                                         indices[:, 2], :]

            # Encode (CNN processing)
            embedding = self.shell_encoders[shell_idx](shell_data)
            shell_embeddings.append(embedding)

        shell_embeddings = torch.stack(shell_embeddings, dim=1)
        # Shape: [B, num_shells, embedding_dim]

        # 3. Cross-shell fusion
        fused = self.fusion(shell_embeddings)  # [B, num_shells, embedding_dim]

        # 4. Decode each shell
        output = torch.zeros_like(fourier_coeffs)
        for shell_idx in range(self.num_shells):
            # Decode
            shell_coeffs = self.shell_decoders[shell_idx](fused[:, shell_idx])

            # Place back in output
            indices = shell_assignments[shell_idx]
            output[:, indices[:, 0], indices[:, 1], indices[:, 2], :] = shell_coeffs

        return output
```

### Why This Works Better

**Problem with CV-ResNet**:
- Shell 0 (|k|=0-1): ~100 coefficients, but processed with 3×3×3 kernel
- Shell 15 (|k|=52-60): ~10,000 coefficients, same 3×3×3 kernel
- **Not appropriate!**

**FSC-Net solution**:
- Shell 0: Use small CNN (deeper, more capacity) - these are critical low freqs
- Shell 15: Use lightweight CNN (shallow) - just denoising high freq
- **Adaptive processing per frequency range**

---

## 4. Multi-Resolution Fourier U-Net (MRFU-Net)

### How It Works

**Key idea**: Split Fourier space into frequency bands along the rfft dimension.

```
Original: [B, 64, 64, 33, 2]
           ↓
Split into 4 bands along dim=2 (the rfft dimension):
Band 0: [:, :, :, 0:8, :]   → [B, 64, 64, 8, 2]   # Lowest frequencies
Band 1: [:, :, :, 8:16, :]  → [B, 64, 64, 8, 2]   # Low-mid frequencies
Band 2: [:, :, :, 16:25, :] → [B, 64, 64, 9, 2]   # High-mid frequencies
Band 3: [:, :, :, 25:33, :] → [B, 64, 64, 8, 2]   # Highest frequencies
```

**Each band goes through a U-Net**:
```
Band i: [B, 64, 64, depth_i, 2]
        ↓
Encoder:
  64×64×depth → Conv+Pool → 32×32×(depth/2)
              → Conv+Pool → 16×16×(depth/4)
              → Conv+Pool → 8×8×(depth/8)
                           ↓
                      Bottleneck
                           ↓
Decoder:
  8×8×(depth/8) → TransConv → 16×16×(depth/4) ⊕ skip1
                → TransConv → 32×32×(depth/2) ⊕ skip2
                → TransConv → 64×64×depth ⊕ skip3
        ↓
Output: [B, 64, 64, depth_i, 2]
```

**Cross-band attention at bottleneck**:
```python
# At bottleneck level, exchange information
bottleneck_0 = encoder_0(band_0)  # [B, 8, 8, 1, C]
bottleneck_1 = encoder_1(band_1)  # [B, 8, 8, 1, C]
bottleneck_2 = encoder_2(band_2)  # [B, 8, 8, 1, C]
bottleneck_3 = encoder_3(band_3)  # [B, 8, 8, 1, C]

# Attention: each band can query other bands
# "Low frequencies, what should high frequencies look like?"
attended_0 = attention(Q=bottleneck_0, K=bottleneck_1..3, V=bottleneck_1..3)
attended_1 = attention(Q=bottleneck_1, K=bottleneck_0,2,3, V=bottleneck_0,2,3)
...

# Decode with attended features
output_0 = decoder_0(attended_0)
output_1 = decoder_1(attended_1)
...
```

### Why Band Splitting?

- **Frequency correlation**: Nearby frequencies are related
- **kz=0-8**: Global structure (low freq)
- **kz=25-33**: Fine details (high freq)
- Process each range with appropriate depth/capacity

---

## 5. Radial-Angular Decomposition Network (RAD-Net)

### Spherical Coordinates in Fourier Space

```python
# Cartesian: (kx, ky, kz)
# Spherical: (r, θ, φ) where:
r = sqrt(kx² + ky² + kz²)          # Radius (frequency magnitude)
θ = arccos(kz / r)                  # Polar angle [0, π]
φ = arctan2(ky, kx)                 # Azimuthal angle [0, 2π]
```

### Data Flow

```
Input: [B, 64, 64, 33, 2] in (kx, ky, kz) coordinates
       ↓
[Coordinate Transform]
Convert to (r, θ, φ) representation
       ↓
[Radial Processing]
Group by r bins: [0, 1, 2, 4, 8, 16, 32, 48]
For each r:
  - Pool over all angles (θ, φ) at this radius
  - Radial embedding: [B, num_r_bins, C_r]
       ↓
[Angular Processing]
For each r shell:
  - Process angular variations (θ, φ)
  - Learn directional patterns
  - Angular embedding: [B, num_r_bins, num_angular_modes, C_a]
       ↓
[Fusion]
Combine radial + angular → Full 3D reconstruction
       ↓
[Inverse Transform]
Convert back to (kx, ky, kz) grid
       ↓
Output: [B, 64, 64, 33, 2]
```

**Why this could work**: Many 3D objects have near-isotropic Fourier transforms (especially after random rotations during training).

---

## Summary Table

| Architecture | Frequency Awareness | Kernel Strategy | Stays in Fourier? | Novel? |
|--------------|-------------------|-----------------|-------------------|--------|
| **CV-ResNet** | ❌ No | Same 3×3×3 everywhere | ✅ Yes | ❌ Standard |
| **FNO-3D** | ✅ Yes (per-mode weights) | Different per k | ✅ Yes | ✅ Adapted |
| **FSC-Net** | ✅ Yes (shell-specific) | Different per shell | ✅ Yes | ✅ Novel |
| **MRFU-Net** | ✅ Partial (band-specific) | Different per band | ✅ Yes | ⚡ Hybrid |
| **RAD-Net** | ✅ Yes (radial) | Radial vs angular | ✅ Yes | ✅ Very novel |

## Key Takeaways

1. **CV-ResNet**: Treats all frequencies the same - limitation but simple baseline
2. **FNO-3D**: Learns frequency-specific transformations, never leaves Fourier space, global receptive field
3. **FSC-Net**: Groups similar frequencies together, processes each group appropriately
4. **MRFU-Net**: Hierarchical frequency bands with cross-attention
5. **RAD-Net**: Exploits spherical symmetry of Fourier space

All architectures (except CV-ResNet) explicitly leverage frequency structure!
