"""Saving and loading BitViT models, in training form (latent weights) or packed inference form.

Both forms keep LoRA adapters: the payload's optional ``"lora"`` entry (``bitnet_vision.lora.lora_config``)
describes them, and loading re-creates them before the state dict is loaded. Checkpoints without adapters
have no such entry and load as before.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .export import PACKED_METADATA, convert_for_inference
from .layers import BitLinear, HBitLinear
from .lora import add_lora_from_config, lora_config
from .models.vit import BitViT, BitViTConfig

__all__ = ["save_model", "load_model", "upgrade_config", "config_differences", "transfer_weights", "save_packed",
           "load_packed", "PACKED_FORMAT"]

PACKED_FORMAT = "bitnet_vision.packed.v1"


def upgrade_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in fields added after a checkpoint was written, keeping the behavior it was trained with."""
    return {"hadamard_random_signs": False, **config}  # rotations had no random signs before the field existed


def config_differences(saved: BitViTConfig, current: BitViTConfig) -> list[str]:
    """Config fields that differ in a way that changes the model.

    Activation bits are ignored (a run may switch them itself, as in BitNet v2's final 4-bit phase), and
    Hadamard-only settings are ignored when neither config uses Hadamard layers.
    """
    ignore = {"act_bits", "hadamard_act_bits"}
    if not saved.hadamard and not current.hadamard:
        ignore.add("hadamard_random_signs")
    return [f for f, v in current.to_dict().items() if f not in ignore and getattr(saved, f) != v]


def _with_lora(payload: dict[str, Any], model: nn.Module) -> dict[str, Any]:
    lora = lora_config(model)
    return {**payload, "lora": lora} if lora else payload


def _restore_lora(model: nn.Module, ckpt: dict[str, Any]) -> None:
    if ckpt.get("lora"):
        add_lora_from_config(model, ckpt["lora"])


def save_model(model: BitViT, path: str, **extra: Any) -> None:
    """Save a trainable BitViT (full-precision latent weights plus config, and LoRA adapters if any)."""
    torch.save(_with_lora({"model_config": model.config.to_dict(), "model": model.state_dict(), **extra}, model), path)


def load_model(path: str, map_location: str | torch.device = "cpu", **config_overrides: Any) -> BitViT:
    """Load a checkpoint written by ``save_model`` or the training loop's ``last.pt``.

    ``config_overrides`` changes config fields on load, e.g. ``quantize=True`` to start a ternary model
    from a full-precision checkpoint (all parameter names match).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format") == PACKED_FORMAT:
        raise ValueError(f"{path} is a packed inference checkpoint; use load_packed()")
    if "hadamard" in config_overrides and config_overrides["hadamard"] != ckpt["model_config"].get("hadamard"):
        raise ValueError("changing 'hadamard' on load would scramble the weights; build the new model and use "
                         "transfer_weights() instead")
    cfg = BitViTConfig(**{**upgrade_config(ckpt["model_config"]), **config_overrides})
    model = BitViT(cfg)
    _restore_lora(model, ckpt)
    model.load_state_dict(ckpt["model"])
    return model


def _rotation(module: nn.Module) -> torch.Tensor | None:
    return module.hadamard.matrix() if isinstance(module, HBitLinear) else None


@torch.no_grad()
def transfer_weights(dst: BitViT, src: BitViT) -> BitViT:
    """Copy all weights from ``src`` into ``dst`` (same architecture, any ``quantize`` setting).

    Layers whose input rotation differs (HBitLinear vs BitLinear, or different sign settings) get their
    weights re-expressed for the new rotation, ``W_dst = W_src R_src^T R_dst``, so every layer computes the
    same function before quantization. LoRA adapters of ``src`` are re-created on ``dst`` if it has none;
    they read the raw input, so they need no rotation.
    """
    if not lora_config(dst):
        add_lora_from_config(dst, lora_config(src))
    dst.load_state_dict(src.state_dict())
    src_modules = dict(src.named_modules())
    for name, m in dst.named_modules():
        if not isinstance(m, BitLinear):
            continue
        r_src, r_dst = _rotation(src_modules[name]), _rotation(m)
        if r_src is None and r_dst is None:
            continue
        if r_src is not None and r_dst is not None and torch.equal(r_src, r_dst):
            continue
        w = m.weight.double()
        if r_src is not None:
            w = w @ r_src.T.to(w.device)  # undo the source rotation
        if r_dst is not None:
            w = w @ r_dst.to(w.device)
        m.weight.copy_(w.to(m.weight.dtype))
    return dst


def save_packed(model: nn.Module, path: str) -> dict[str, Any]:
    """Convert (if needed) to packed 2-bit weights and save. Returns the saved payload."""
    if not hasattr(model, "config"):
        raise TypeError("save_packed needs a BitViT (it stores the model config)")
    packed = convert_for_inference(model)  # copies; a no-op swap if the model is already packed
    config = {**model.config.to_dict(), "quantize": True}  # a packed model is quantized by definition
    payload = _with_lora({"format": PACKED_FORMAT, "conventions": dict(PACKED_METADATA), "model_config": config,
                          "model": packed.state_dict()}, packed)
    torch.save(payload, path)
    return payload


def load_packed(path: str, map_location: str | torch.device = "cpu", **inference_kwargs: Any) -> nn.Module:
    """Load a packed checkpoint into a BitViT whose BitLinear layers are ``BitLinearInference``.

    ``inference_kwargs`` (``int_matmul``, ``keep_unpacked``) are passed to every inference layer.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format") != PACKED_FORMAT:
        raise ValueError(f"{path} is not a packed checkpoint ({PACKED_FORMAT})")
    config = BitViTConfig(**{**upgrade_config(ckpt["model_config"]), "quantize": True})
    model = convert_for_inference(BitViT(config), inplace=True, **inference_kwargs)
    _restore_lora(model, ckpt)
    model.load_state_dict(ckpt["model"])
    return model.eval()
