"""Ternary image generators built on diffusers models: FLUX.2 [klein] / Bonsai Image 4B with existing LoRAs.

Optional: needs ``pip install 'bitnet-vision[flux]'`` (diffusers 0.41 or newer). Importing this package does not
import diffusers; the functions that need it import it when called.
"""

from .flux2 import (
    BONSAI_GROUP_SIZE,
    BONSAI_MLX,
    BONSAI_UNPACKED,
    FLUX2_TERNARY_TARGETS,
    Flux2LoRASummary,
    load_bonsai_transformer,
    load_flux2_lora,
    round_linears_to_ternary,
    ternarize_transformer,
)

__all__ = [
    "BONSAI_GROUP_SIZE",
    "BONSAI_MLX",
    "BONSAI_UNPACKED",
    "FLUX2_TERNARY_TARGETS",
    "Flux2LoRASummary",
    "load_bonsai_transformer",
    "load_flux2_lora",
    "round_linears_to_ternary",
    "ternarize_transformer",
]
