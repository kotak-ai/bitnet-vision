"""bitnet_vision: ternary (1.58-bit) Vision Transformers in PyTorch, following BitNet b1.58 and BitNet v2."""

from .export import BitLinearInference, convert_for_inference, pack_ternary, unpack_ternary
from .hadamard import HadamardTransform, hadamard_matrix
from .layers import BitLinear, HBitLinear, RMSNorm, bitlinear_modules, convert_linear_layers, set_quant_strength
from .models import MODEL_PRESETS, BitViT, BitViTConfig, create_model
from .quant import fake_quant_activations, fake_quant_weights, quantize_activations, quantize_weights, ste

__version__ = "0.1.0"

__all__ = [
    "BitLinear",
    "HBitLinear",
    "RMSNorm",
    "bitlinear_modules",
    "set_quant_strength",
    "convert_linear_layers",
    "HadamardTransform",
    "hadamard_matrix",
    "quantize_activations",
    "quantize_weights",
    "fake_quant_activations",
    "fake_quant_weights",
    "ste",
    "BitViT",
    "BitViTConfig",
    "MODEL_PRESETS",
    "create_model",
    "BitLinearInference",
    "convert_for_inference",
    "pack_ternary",
    "unpack_ternary",
]
