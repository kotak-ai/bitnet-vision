"""
BitNet b1.58 and BitNet v2 PyTorch Implementation

This package provides the core modules for Microsoft's BitNet architecture,
featuring 1.58-bit weight quantization and efficient inference.

Modules:
    - BitLinear: Linear layer with ternary weight quantization {-1, 0, +1}
    - HBitLinear: BitLinear with online Hadamard transformation (BitNet v2)
    - RMSNorm: Root Mean Square Layer Normalization

Example:
    >>> from bitnet import BitLinear, HBitLinear, RMSNorm
    >>> 
    >>> # Create a BitLinear layer (BitNet b1.58)
    >>> layer = BitLinear(in_features=512, out_features=1024)
    >>> 
    >>> # Create an HBitLinear layer (BitNet v2)
    >>> h_layer = HBitLinear(in_features=512, out_features=1024)
    >>> 
    >>> # Create RMSNorm
    >>> norm = RMSNorm(dim=512)
    >>> 
    >>> # Forward pass
    >>> x = torch.randn(2, 10, 512)
    >>> x = norm(x)
    >>> output = layer(x)
"""

from bitnet.modules import (
    BitLinear,
    BitLinear1p58,
    HadamardBitLinear,
    HadamardTransform,
    HBitLinear,
    RMSNorm,
    fast_walsh_hadamard_transform,
    fast_walsh_hadamard_transform_vectorized,
    generate_hadamard_matrix,
    hadamard_transform,
    quantize_activations_absmax,
    quantize_weights_absmean,
)

__version__ = "0.2.0"
__all__ = [
    # Core layers
    "BitLinear",
    "BitLinear1p58",
    "HBitLinear",
    "HadamardBitLinear",
    "RMSNorm",
    # Quantization functions
    "quantize_weights_absmean",
    "quantize_activations_absmax",
    # Hadamard utilities
    "generate_hadamard_matrix",
    "fast_walsh_hadamard_transform",
    "fast_walsh_hadamard_transform_vectorized",
    "hadamard_transform",
    "HadamardTransform",
]
