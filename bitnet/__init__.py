"""
BitNet b1.58 PyTorch Implementation

This package provides the core modules for Microsoft's BitNet b1.58 architecture,
featuring 1.58-bit weight quantization and efficient inference.

Modules:
    - BitLinear: Linear layer with ternary weight quantization {-1, 0, +1}
    - RMSNorm: Root Mean Square Layer Normalization

Example:
    >>> from bitnet import BitLinear, RMSNorm
    >>> 
    >>> # Create a BitLinear layer
    >>> layer = BitLinear(in_features=512, out_features=1024)
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
    RMSNorm,
    quantize_activations_absmax,
    quantize_weights_absmean,
)

__version__ = "0.1.0"
__all__ = [
    "BitLinear",
    "BitLinear1p58",
    "RMSNorm",
    "quantize_weights_absmean",
    "quantize_activations_absmax",
]
