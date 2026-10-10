"""Saving and loading BitViT models, in training form (latent weights) or packed inference form."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .export import convert_for_inference
from .layers import HBitLinear
from .models.vit import BitViT, BitViTConfig

__all__ = ["save_model", "load_model", "transfer_weights", "save_packed", "load_packed", "PACKED_FORMAT"]

PACKED_FORMAT = "bitnet_vision.packed.v1"


def save_model(model: BitViT, path: str, **extra: Any) -> None:
    """Save a trainable BitViT (full-precision latent weights plus config)."""
    torch.save({"model_config": model.config.to_dict(), "model": model.state_dict(), **extra}, path)


def load_model(path: str, map_location: str | torch.device = "cpu", **config_overrides: Any) -> BitViT:
    """Load a checkpoint written by ``save_model`` or the training loop's ``last.pt``.

    ``config_overrides`` changes config fields on load, e.g. ``quantize=True`` to start a ternary model
    from a full-precision checkpoint (all parameter names match).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format") == PACKED_FORMAT:
        raise ValueError(f"{path} is a packed inference checkpoint; use load_packed()")
    cfg = BitViTConfig(**{**ckpt["model_config"], **config_overrides})
    model = BitViT(cfg)
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def transfer_weights(dst: BitViT, src: BitViT) -> BitViT:
    """Copy all weights from ``src`` into ``dst`` (same architecture, any ``quantize`` setting).

    If ``dst`` uses HBitLinear where ``src`` has a plain BitLinear, the copied weights are rotated
    (``W H``) so that the layer computes the same function on its rotated input.
    """
    dst.load_state_dict(src.state_dict())
    src_modules = dict(src.named_modules())
    for name, m in dst.named_modules():
        if isinstance(m, HBitLinear) and not isinstance(src_modules.get(name), HBitLinear):
            m.weight.copy_(m._weight_for_input_space(m.weight))
    return dst


def save_packed(model: nn.Module, path: str) -> dict[str, Any]:
    """Convert (if needed) to packed 2-bit weights and save. Returns the saved payload."""
    if not hasattr(model, "config"):
        raise TypeError("save_packed needs a BitViT (it stores the model config)")
    packed = convert_for_inference(model)  # copies; a no-op swap if the model is already packed
    payload = {"format": PACKED_FORMAT, "model_config": model.config.to_dict(), "model": packed.state_dict()}
    torch.save(payload, path)
    return payload


def load_packed(path: str, map_location: str | torch.device = "cpu") -> nn.Module:
    """Load a packed checkpoint into a BitViT whose BitLinear layers are ``BitLinearInference``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format") != PACKED_FORMAT:
        raise ValueError(f"{path} is not a packed checkpoint ({PACKED_FORMAT})")
    model = convert_for_inference(BitViT(BitViTConfig(**ckpt["model_config"])), inplace=True)
    model.load_state_dict(ckpt["model"])
    return model.eval()
