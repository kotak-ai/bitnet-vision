# BitNet b1.58 & BitNet v2 PyTorch Implementation

A PyTorch implementation of Microsoft's BitNet architecture, featuring 1.58-bit weight quantization for efficient large language models.

## Overview

BitNet b1.58 is a novel neural network architecture that uses ternary weights `{-1, 0, +1}` instead of full-precision floating-point weights. This dramatically reduces memory footprint and enables efficient inference while maintaining competitive model quality.

BitNet v2 extends this with **H-BitLinear**, which applies an online Hadamard transformation before activation quantization to smooth outlier-prone distributions.

**References:**
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- [BitNet v2: Native Support of Variable-Length Sequences in 1-bit LLMs](https://arxiv.org/abs/2504.00097)

## Features

- **BitLinear**: Drop-in replacement for `nn.Linear` with ternary weight quantization
- **HBitLinear**: BitLinear with online Hadamard transformation (BitNet v2)
- **RMSNorm**: Root Mean Square Layer Normalization
- **Fast Walsh-Hadamard Transform**: O(n log n) efficient transformation
- **Straight-Through Estimator (STE)**: Enables gradient flow through quantization
- **Full-precision latent weights**: Maintained for gradient accumulation during training

## Installation

```bash
pip install torch>=2.0.0
```

## Quick Start

```python
import torch
from bitnet import BitLinear, HBitLinear, RMSNorm

# Create a BitLinear layer (BitNet b1.58)
layer = BitLinear(in_features=512, out_features=1024)

# Create an HBitLinear layer (BitNet v2 with Hadamard transform)
h_layer = HBitLinear(in_features=512, out_features=1024)

# Create RMSNorm
norm = RMSNorm(dim=512)

# Forward pass
x = torch.randn(2, 10, 512)  # (batch, seq_len, features)
x = norm(x)
output = layer(x)      # BitNet b1.58
output_h = h_layer(x)  # BitNet v2

print(output.shape)  # torch.Size([2, 10, 1024])
```

## Modules

### BitLinear

A linear layer with ternary weight quantization `{-1, 0, +1}` and 8-bit activation quantization.

```python
from bitnet import BitLinear

# Basic usage
layer = BitLinear(
    in_features=512,
    out_features=1024,
    bias=False,        # Default: False (BitNet typically doesn't use bias)
    bits=8,            # Activation quantization bits (default: 8)
)
```

**Mathematical Formulation:**

1. **Weight Quantization (absmean):**
   - Scale: `γ = mean(|W|)`
   - Normalize: `W_normalized = W / (γ + ε)`
   - Quantize: `W_ternary = RoundClip(W_normalized, -1, 1)`

2. **Activation Quantization (absmax per-token):**
   - Scale: `η = max(|x|)` per token
   - Range: `Q_b = 2^(b-1)` where `b=8`
   - Quantize: `x_quant = Clip(round(x × Q_b / η), -Q_b, Q_b-1)`

3. **Forward Computation:**
   - `y = (x_quant × W_ternary) × (γ × η / Q_b)`

### HBitLinear (BitNet v2)

BitLinear with online Hadamard transformation for improved quantization.

```python
from bitnet import HBitLinear

# Works with any dimension (pads to power of 2 internally)
layer = HBitLinear(
    in_features=512,
    out_features=1024,
    bias=False,
    bits=8,
)
```

**Why Hadamard Transform?**

The Hadamard transformation smooths sharp, outlier-prone activation distributions into more Gaussian-like forms. This makes activations more suitable for low-bit quantization by reducing the dynamic range.

**Forward Pass:**
1. Pad input to next power of 2 (if needed)
2. Apply Fast Walsh-Hadamard Transform: `x_h = FWHT(x) / √n`
3. Unpad to original dimension
4. Apply standard BitLinear quantization and computation

**Hadamard Properties:**
- `H_m × H_m^T = m × I` (orthogonal)
- `H_m^(-1) = H_m / m` (self-inverse)
- Recursive: `H_2n = [[H_n, H_n], [H_n, -H_n]]`

**Backward Pass:**
- Due to orthogonality, gradients also go through the Hadamard transform

### RMSNorm

Root Mean Square Layer Normalization without centering.

```python
from bitnet import RMSNorm

norm = RMSNorm(dim=512, eps=1e-6)
```

**Formula:** `rmsnorm(x) = x × rsqrt(mean(x²) + ε) × weight`

## Building a Model

```python
import torch
import torch.nn as nn
from bitnet import BitLinear, RMSNorm

class BitNetBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = BitLinear(dim, hidden_dim)
        self.fc2 = BitLinear(hidden_dim, dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual

# Create model
model = nn.Sequential(
    BitNetBlock(512, 2048),
    BitNetBlock(512, 2048),
    BitNetBlock(512, 2048),
)

# Training works normally with full-precision gradients
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

## Training

Training works exactly like standard PyTorch models. The Straight-Through Estimator (STE) enables gradient flow through the quantization operations:

```python
model = BitLinear(512, 1024)
optimizer = torch.optim.Adam(model.parameters())

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Gradients flow through STE
    optimizer.step()  # Updates full-precision latent weights
```

## API Reference

### BitLinear

```python
BitLinear(
    in_features: int,      # Input feature dimension
    out_features: int,     # Output feature dimension
    bias: bool = False,    # Whether to include bias (default: False)
    eps: float = 1e-5,     # Epsilon for numerical stability
    bits: int = 8,         # Activation quantization bits
)
```

### HBitLinear

```python
HBitLinear(
    in_features: int,      # Input feature dimension (any size)
    out_features: int,     # Output feature dimension
    bias: bool = False,    # Whether to include bias (default: False)
    eps: float = 1e-5,     # Epsilon for numerical stability
    bits: int = 8,         # Activation quantization bits
)
# Note: Automatically pads to power of 2 for Hadamard transform
```

### RMSNorm

```python
RMSNorm(
    dim: int,              # Feature dimension
    eps: float = 1e-6,     # Epsilon for numerical stability
)
```

### Quantization Functions

```python
from bitnet import quantize_weights_absmean, quantize_activations_absmax

# Quantize weights to ternary {-1, 0, +1}
w_ternary, gamma = quantize_weights_absmean(weights, eps=1e-5)

# Quantize activations to 8-bit per-token
x_quant, eta = quantize_activations_absmax(x, bits=8)
```

### Hadamard Transform Functions

```python
from bitnet import (
    generate_hadamard_matrix,
    fast_walsh_hadamard_transform,
    fast_walsh_hadamard_transform_vectorized,
    hadamard_transform,
)

# Generate Hadamard matrix (power of 2 only)
H = generate_hadamard_matrix(n=64)

# Fast Walsh-Hadamard Transform (O(n log n))
y = fast_walsh_hadamard_transform_vectorized(x, normalize=True)

# With autograd support
y = hadamard_transform(x)  # Gradients flow through correctly
```

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v
```

## License

MIT License
