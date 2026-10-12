"""LoRA adapters: low-rank, full-precision side branches on ternary and ordinary linear layers.

An adapter adds ``strength * scale * B(A(dropout(x)))`` to a layer's output, with ``A`` of shape ``[r, in]``,
``B`` of shape ``[out, r]`` and ``scale = alpha / r`` (``alpha / sqrt(r)`` with rsLoRA). ``A`` starts like
PEFT's (Kaiming uniform) and ``B`` at zero, so a new adapter changes nothing until it is trained.

The branch reads the layer's raw input ``x``, before BitLinear's inner RMSNorm and Hadamard rotation, and is
added to the final output after the bias. It runs in float32 (float64 for float64 inputs, also under autocast)
and is never quantized. A LoRA trained on an ordinary ``nn.Linear`` expects exactly this, and the branch stays
exact through export: the packed 2-bit layers, the fused Apple layers and the MLX port run the same branch,
and so does the weight-only layer for imported ternary models (``bitnet_vision.weight_only``).

Layout, PEFT's, so files map one to one: a layer with adapters holds ModuleDicts ``lora_A`` and ``lora_B``
keyed by adapter name (state-dict keys ``<module>.lora_A.<adapter>.weight``; the base layer's keys do not
change) and ``lora_settings``, a dict of :class:`LoRASettings` in the same order. BitLinear, HBitLinear,
BitLinearInference, FusedBitLinearInference and PackedTernaryLinear call :func:`lora_forward` at the end of
their forward and declare ``lora_settings = None`` at class level, which is how another layer class (with
``in_features``, ``out_features`` and a ``weight`` or ``weight_scale`` tensor) opts in. A plain ``nn.Linear``
gets a forward hook that calls :func:`lora_forward`.

Files: :func:`save_lora` writes PEFT's format (``adapter_config.json`` and ``adapter_model.safetensors``, or
``adapter_model.bin`` without the safetensors package), with keys ``base_model.model.<module>.lora_A.weight``.
:func:`load_lora` reads that format, PEFT's own files for the same module names, and plain state dicts. A
loader for another format only has to rename its keys to ``<module>.lora_A.weight`` (``[r, in]``) and
``<module>.lora_B.weight`` (``[out, r]``), express its scaling as PEFT's ``lora_alpha`` / ``alpha_pattern``
(and ``use_rslora``), and pass both to :func:`load_lora`.

With ``torch.compile``, add or load adapters before compiling. PyTorch 2.4 and newer pick up later changes to
strength, enabled flags and BitLinear adapters; 2.2 does not notice any of them (call ``torch._dynamo.reset()``).
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import json
import math
import os
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hadamard import autocast_disabled

__all__ = [
    "DEFAULT_TARGETS",
    "LoRASettings",
    "lora_forward",
    "supports_lora",
    "in_lora_adapter",
    "add_lora_adapter",
    "copy_lora",
    "add_lora",
    "lora_layers",
    "adapter_names",
    "set_lora_strength",
    "enable_lora",
    "disable_lora",
    "delete_lora",
    "mark_only_lora_trainable",
    "lora_state_dict",
    "lora_config",
    "add_lora_from_config",
    "save_lora",
    "load_lora",
    "merge_lora_into_weights",
]

DEFAULT_TARGETS = ("qkv", "proj", "fc1", "fc2")
PEFT_PREFIX = "base_model.model."
CONFIG_NAME = "adapter_config.json"
SAFETENSORS_NAME = "adapter_model.safetensors"
BIN_NAME = "adapter_model.bin"
_PARTS = ("lora_A", "lora_B")


@dataclass
class LoRASettings:
    """One adapter on one layer. ``strength`` scales its output (for mixing several adapters) and
    ``enabled=False`` switches it off; neither changes the weights."""

    r: int
    alpha: float
    dropout: float = 0.0
    use_rslora: bool = False
    strength: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r < 1:
            raise ValueError(f"LoRA rank must be a positive int, got {self.r!r}")
        if not math.isfinite(self.alpha) or not math.isfinite(self.strength):
            raise ValueError("LoRA alpha and strength must be finite")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")

    @property
    def scale(self) -> float:
        """``alpha / r``, or ``alpha / sqrt(r)`` with rsLoRA (Kalajdzievski, arXiv 2312.03732)."""
        return self.alpha / (math.sqrt(self.r) if self.use_rslora else self.r)


# ---------------------------------------------------------------------------------------------------------
# The branch itself


def lora_forward(layer: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """``y`` plus the enabled adapters of ``layer`` evaluated on its raw input ``x``.

    The branches run in float32 (float64 for float64 ``x``) with autocast off, are summed, added to ``y`` in
    that precision and rounded once to ``y``'s dtype. Returns ``y`` itself when nothing is enabled.
    """
    settings = getattr(layer, "lora_settings", None)
    if not settings:
        return y
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    delta = None
    with autocast_disabled(x.device.type):
        xf = x.to(dtype)
        for name, s in settings.items():
            if not s.enabled or s.strength == 0:
                continue
            h = F.dropout(xf, s.dropout, layer.training) if s.dropout else xf
            a, b = layer.lora_A[name].weight.to(dtype), layer.lora_B[name].weight.to(dtype)
            d = F.linear(F.linear(h, a), b) * (s.strength * s.scale)
            delta = d if delta is None else delta + d
    if delta is None:
        return y
    return (y.to(dtype) + delta).to(y.dtype)


def _linear_hook(module: nn.Module, args: tuple, kwargs: dict, output: Tensor) -> Tensor:
    return lora_forward(module, args[0] if args else kwargs["input"], output)


def _native(layer: nn.Module) -> bool:
    """True for layer classes that call :func:`lora_forward` in their own forward."""
    return hasattr(type(layer), "lora_settings")


def supports_lora(layer: nn.Module) -> bool:
    """BitLinear-family layers and plain ``nn.Linear`` (not its subclasses, whose forward may differ, such as
    the output projection inside ``nn.MultiheadAttention``, which is never called as a module)."""
    return _native(layer) or type(layer) is nn.Linear


def in_lora_adapter(module_name: str) -> bool:
    """True for the qualified names of adapter parts (``...lora_A``, ``...lora_B.<adapter>``)."""
    return any(part in _PARTS for part in module_name.split("."))


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not name or "." in name or hasattr(nn.ModuleDict(), name):
        raise ValueError(f"invalid adapter name {name!r}: use a non-empty string without '.' that is not an "
                         "attribute of nn.ModuleDict")


def _settings(layer: nn.Module) -> dict[str, LoRASettings]:
    return getattr(layer, "lora_settings", None) or {}


def _device_dtype(layer: nn.Module) -> tuple[torch.device, torch.dtype]:
    ref = layer.weight if isinstance(getattr(layer, "weight", None), Tensor) else layer.weight_scale
    return ref.device, (ref.dtype if ref.is_floating_point() else torch.float32)


def _attach(layer: nn.Module, a: nn.ModuleDict, b: nn.ModuleDict, settings: dict[str, LoRASettings]) -> None:
    layer.lora_A, layer.lora_B, layer.lora_settings = a, b, settings
    if not _native(layer) and _linear_hook not in layer._forward_hooks.values():
        layer.register_forward_hook(_linear_hook, with_kwargs=True)  # inert once its adapters are deleted


def add_lora_adapter(
    layer: nn.Module,
    name: str = "default",
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    use_rslora: bool = False,
    dtype: torch.dtype | None = None,
) -> LoRASettings:
    """Add one adapter to one layer and return its settings. ``A`` and ``B`` get the layer's device, and its
    dtype unless ``dtype`` is given (float32 adapters on a bfloat16 model avoid rounding imported weights)."""
    if not supports_lora(layer):
        raise TypeError(f"LoRA adapters need a BitLinear-family layer or a plain nn.Linear, got {type(layer).__name__}")
    _check_name(name)
    if name in _settings(layer):
        raise ValueError(f"the layer already has an adapter named {name!r}")
    settings = LoRASettings(r=r, alpha=float(alpha), dropout=float(dropout), use_rslora=bool(use_rslora))
    device, layer_dtype = _device_dtype(layer)
    dtype = layer_dtype if dtype is None else dtype
    a = nn.Linear(layer.in_features, r, bias=False, device=device, dtype=dtype)
    b = nn.Linear(r, layer.out_features, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))  # PEFT's init for A
        nn.init.zeros_(b.weight)  # so a new adapter adds exactly 0
    if not _settings(layer):
        _attach(layer, nn.ModuleDict(), nn.ModuleDict(), {})
    layer.lora_A[name], layer.lora_B[name] = a, b
    layer.lora_settings[name] = settings
    return settings


def _remove_adapter(layer: nn.Module, name: str) -> None:
    del layer.lora_A[name], layer.lora_B[name], layer.lora_settings[name]
    if not layer.lora_settings:
        del layer.lora_A, layer.lora_B
        layer.lora_settings = None


def copy_lora(src: nn.Module, dst: nn.Module, share: bool = False) -> None:
    """Give ``dst`` the adapters of ``src`` (copies, or the same modules and settings with ``share=True``).

    Used when one layer type is converted to another; adapters read the raw input, so they carry over exactly.
    """
    settings = _settings(src)
    if not settings:
        return
    if _settings(dst):
        raise ValueError("the destination layer already has adapters")
    if share:
        _attach(dst, src.lora_A, src.lora_B, settings)
    else:
        _attach(dst, copy.deepcopy(src.lora_A), copy.deepcopy(src.lora_B),
                {k: dataclasses.replace(v) for k, v in settings.items()})


# ---------------------------------------------------------------------------------------------------------
# Whole models


def _matches(name: str, targets: str | Iterable[str] | None) -> bool:
    if targets is None:
        return True
    if isinstance(targets, str):
        return re.fullmatch(targets, name) is not None
    return any(name == t or name.endswith("." + t) for t in targets)


def add_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    targets: str | Iterable[str] | None = DEFAULT_TARGETS,
    name: str = "default",
    dropout: float = 0.0,
    use_rslora: bool = False,
    dtype: torch.dtype | None = None,
) -> list[str]:
    """Add adapter ``name`` to every supported layer whose qualified name matches ``targets``.

    ``targets``: module-name suffixes (``"qkv"`` matches ``blocks.3.attn.qkv``), a string taken as a regular
    expression that must match the whole name (as in PEFT), or None for every supported layer. Matching
    modules of other types (``patch_embed.proj`` is a Conv2d) are skipped. ``dtype`` is the adapters' dtype
    (default: each layer's). Returns the targeted names.
    """
    _check_name(name)
    targets = list(targets) if targets is not None and not isinstance(targets, str) else targets
    chosen = [(n, m) for n, m in model.named_modules() if not in_lora_adapter(n) and supports_lora(m)
              and _matches(n, targets)]
    if not chosen:
        raise ValueError(f"no supported layer matches targets={targets!r}")
    taken = [n for n, m in chosen if name in _settings(m)]
    if taken:
        raise ValueError(f"adapter {name!r} already exists on {taken}")
    LoRASettings(r=r, alpha=float(alpha), dropout=float(dropout))  # validate before changing anything
    for _, m in chosen:
        add_lora_adapter(m, name, r=r, alpha=alpha, dropout=dropout, use_rslora=use_rslora, dtype=dtype)
    return [n for n, _ in chosen]


def lora_layers(model: nn.Module, name: str | None = None) -> list[tuple[str, nn.Module]]:
    """``(qualified name, layer)`` for every layer that has adapter ``name`` (any adapter if None)."""
    return [(n, m) for n, m in model.named_modules() if not in_lora_adapter(n) and _settings(m)
            and (name is None or name in _settings(m))]


def adapter_names(model: nn.Module) -> list[str]:
    """Names of all adapters in ``model``, in the order they first appear."""
    return list(dict.fromkeys(k for _, m in lora_layers(model) for k in _settings(m)))


def _select(model: nn.Module, name: str | None) -> list[tuple[str, nn.Module, str]]:
    """``(layer name, layer, adapter)`` for adapter ``name`` (every adapter if None); raises if there is none."""
    found = [(n, m, k) for n, m in lora_layers(model, name) for k in list(_settings(m)) if name in (None, k)]
    if not found:
        raise ValueError(f"the model has no adapter named {name!r}" if name else "the model has no LoRA adapters")
    return found


def set_lora_strength(model: nn.Module, strength: float, name: str | None = "default") -> None:
    """Scale adapter ``name`` (every adapter if None) by ``strength``; 0 turns it off, 1 is as trained."""
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    for _, m, k in _select(model, name):
        m.lora_settings[k].strength = float(strength)


def enable_lora(model: nn.Module, name: str | None = None) -> None:
    """Switch adapter ``name`` (every adapter if None) on."""
    for _, m, k in _select(model, name):
        m.lora_settings[k].enabled = True


def disable_lora(model: nn.Module, name: str | None = None) -> None:
    """Switch adapter ``name`` (every adapter if None) off; the model then computes exactly its base output."""
    for _, m, k in _select(model, name):
        m.lora_settings[k].enabled = False


def delete_lora(model: nn.Module, name: str | None = "default") -> list[str]:
    """Remove adapter ``name`` (every adapter if None) and return the layer names it was removed from."""
    found = _select(model, name)
    for _, m, k in found:
        _remove_adapter(m, k)
    return list(dict.fromkeys(n for n, _, _ in found))


def mark_only_lora_trainable(
    model: nn.Module, names: str | Iterable[str] | None = None, extra: str | Iterable[str] = ()
) -> list[str]:
    """Freeze every parameter except those of adapters ``names`` (all adapters if None).

    ``extra`` lists qualified module or parameter names that train too, e.g. ``("head",)``. Returns the names
    of the trainable parameters.
    """
    known = adapter_names(model)
    names = known if names is None else [names] if isinstance(names, str) else list(names)
    unknown = sorted(set(names) - set(known))
    if unknown or not names:
        raise ValueError(f"unknown adapters {unknown}" if unknown else "the model has no LoRA adapters")
    extra = (extra,) if isinstance(extra, str) else tuple(extra)
    used, trainable = set(), []
    for pname, p in model.named_parameters():
        parts = pname.split(".")
        on = any(part in _PARTS and parts[i + 1] in names for i, part in enumerate(parts[:-1]))
        for e in extra:
            if pname == e or pname.startswith(e + "."):
                on = True
                used.add(e)
        p.requires_grad_(on)
        if on:
            trainable.append(pname)
    if set(extra) - used:
        raise ValueError(f"extra names match no parameter: {sorted(set(extra) - used)}")
    return trainable


def lora_state_dict(model: nn.Module, name: str = "default") -> dict[str, Tensor]:
    """The entries of ``model.state_dict()`` that belong to adapter ``name`` (keys keep the adapter name)."""
    out = {}
    for n, m, k in _select(model, name):
        out[_join(n, f"lora_A.{k}.weight")] = m.lora_A[k].weight.detach()
        out[_join(n, f"lora_B.{k}.weight")] = m.lora_B[k].weight.detach()
    return out


def lora_config(model: nn.Module) -> dict[str, dict[str, dict[str, Any]]]:
    """``{layer name: {adapter: settings}}`` for every adapter: what :func:`add_lora_from_config` needs to
    rebuild them before loading a state dict. Plain Python types, so it fits in a ``weights_only`` checkpoint."""
    return {n: {k: dataclasses.asdict(s) for k, s in _settings(m).items()} for n, m in lora_layers(model)}


def add_lora_from_config(model: nn.Module, config: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    """Re-create the adapters described by :func:`lora_config` (weights still to be loaded)."""
    modules = dict(model.named_modules())
    missing = [n for n in config if n not in modules]
    if missing:
        raise ValueError(f"the model has no layers named {missing}")
    for layer_name, adapters in config.items():
        for adapter, s in adapters.items():
            add_lora_adapter(modules[layer_name], adapter, r=s["r"], alpha=s["alpha"], dropout=s.get("dropout", 0.0),
                             use_rslora=s.get("use_rslora", False))
            settings = modules[layer_name].lora_settings[adapter]
            settings.strength, settings.enabled = float(s.get("strength", 1.0)), bool(s.get("enabled", True))


# ---------------------------------------------------------------------------------------------------------
# PEFT-compatible files


def _safetensors_torch() -> Any:
    try:
        import safetensors.torch
    except ImportError:
        return None
    return safetensors.torch


def _most_common(values: list) -> Any:
    return collections.Counter(values).most_common(1)[0][0]


def save_lora(
    model: nn.Module,
    path: str,
    name: str = "default",
    *,
    modules_to_save: Iterable[str] = (),
    safe_serialization: bool | None = None,
) -> dict[str, Any]:
    """Write adapter ``name`` to directory ``path`` in PEFT's format and return the config written.

    Files: ``adapter_config.json`` and ``adapter_model.safetensors`` (or ``adapter_model.bin`` via
    ``torch.save`` when ``safe_serialization`` is False, or None and the safetensors package is missing).
    Weight keys are ``base_model.model.<module>.lora_A.weight`` / ``lora_B.weight``, without the adapter name,
    as PEFT saves them. Ranks or alphas that differ between layers go to PEFT's ``rank_pattern`` and
    ``alpha_pattern``. ``modules_to_save`` adds whole modules trained alongside the adapter (e.g. ``"head"``),
    stored under their own keys as PEFT does; as in PEFT, they cannot carry the adapter themselves. Strength
    and enabled flags are not saved: they are choices made when loading.
    """
    layers = [(n, m) for n, m, _ in _select(model, name)]
    settings = {n: m.lora_settings[name] for n, m in layers}
    for field in ("dropout", "use_rslora"):
        if len({getattr(s, field) for s in settings.values()}) > 1:
            raise ValueError(f"adapter {name!r} uses different {field} values on different layers, which PEFT's "
                             "format cannot express")
    modules_to_save = list(modules_to_save)
    both = [n for n in settings for m in modules_to_save if n == m or n.startswith(m + ".")]
    if both:
        raise ValueError(f"{both} carry adapter {name!r} and are also in modules_to_save, which PEFT's format "
                         "cannot hold; merge those adapters into the weights first (merge_lora_into_weights)")
    first = next(iter(settings.values()))
    r = _most_common([s.r for s in settings.values()])
    alpha = _most_common([s.alpha for s in settings.values()])
    tensors = {}
    for n, m in layers:
        tensors[PEFT_PREFIX + _join(n, "lora_A.weight")] = m.lora_A[name].weight.detach().cpu().clone()
        tensors[PEFT_PREFIX + _join(n, "lora_B.weight")] = m.lora_B[name].weight.detach().cpu().clone()
    for module_name in modules_to_save:
        for key, value in model.get_submodule(module_name).state_dict().items():
            if not in_lora_adapter(key):
                tensors[PEFT_PREFIX + _join(module_name, key)] = value.detach().cpu().clone()
    config = {
        "peft_type": "LORA",
        "task_type": None,
        "base_model_name_or_path": None,
        "inference_mode": True,
        "r": r,
        "lora_alpha": alpha,
        "target_modules": list(settings),
        "lora_dropout": first.dropout,
        "use_rslora": first.use_rslora,
        "bias": "none",
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "modules_to_save": modules_to_save or None,
        "rank_pattern": {n: s.r for n, s in settings.items() if s.r != r},
        "alpha_pattern": {n: s.alpha for n, s in settings.items() if s.alpha != alpha},
    }
    st = _safetensors_torch()
    if safe_serialization and st is None:
        raise ImportError("safe_serialization=True needs the safetensors package (pip install safetensors)")
    use_st = st is not None if safe_serialization is None else safe_serialization
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)
    stale = os.path.join(path, BIN_NAME if use_st else SAFETENSORS_NAME)
    if os.path.exists(stale):  # load_lora prefers safetensors, so never leave an older file of the other kind
        os.remove(stale)
    if use_st:
        st.save_file(tensors, os.path.join(path, SAFETENSORS_NAME), metadata={"format": "pt"})
    else:
        torch.save(tensors, os.path.join(path, BIN_NAME))
    return config


def _read_lora_files(path: str) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    weights = path
    if os.path.isdir(path):
        weights = next((os.path.join(path, f) for f in (SAFETENSORS_NAME, BIN_NAME)
                        if os.path.exists(os.path.join(path, f))), None)
        if weights is None:
            raise FileNotFoundError(f"{path} holds neither {SAFETENSORS_NAME} nor {BIN_NAME}")
    if weights.endswith(".safetensors"):
        st = _safetensors_torch()
        if st is None:
            raise ImportError(f"reading {weights} needs the safetensors package (pip install safetensors)")
        state = st.load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    config = None
    if os.path.exists(os.path.join(folder, CONFIG_NAME)):
        with open(os.path.join(folder, CONFIG_NAME)) as f:
            config = json.load(f)
    return state, config


def _parse_key(key: str) -> tuple[str, str, str | None] | None:
    """``(module, "lora_A" or "lora_B", adapter or None)`` for ``<module>.lora_A[.<adapter>].weight`` keys."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part in _PARTS:
            tail = parts[i + 1:]
            if tail == ["weight"] or (len(tail) == 2 and tail[1] == "weight"):
                return ".".join(parts[:i]), part, (tail[0] if len(tail) == 2 else None)
            return None
    return None


def _pattern_value(patterns: Mapping[str, Any] | None, module: str, default: Any) -> Any:
    """PEFT's rank_pattern / alpha_pattern lookup (peft.utils.other.get_pattern_key): the first key, a regex,
    that matches the end of the name after a dot."""
    for key, value in (patterns or {}).items():
        if re.fullmatch(rf"(.*\.)?({key})", module):
            return value
    return default


def _check_peft_config(config: Mapping[str, Any]) -> None:
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"not a LoRA adapter (peft_type={config['peft_type']!r})")
    if config.get("bias", "none") not in (None, "none"):
        raise ValueError(f"adapters with trained biases (bias={config['bias']!r}) are not supported")
    if config.get("use_dora"):
        raise ValueError("DoRA adapters (use_dora=True) are not supported")
    if config.get("lora_bias"):
        raise ValueError("adapters with a bias on B (lora_bias=True) are not supported")


def load_lora(
    model: nn.Module,
    path_or_state_dict: str | os.PathLike | Mapping[str, Tensor],
    name: str = "default",
    strength: float = 1.0,
    *,
    config: Mapping[str, Any] | None = None,
    dtype: torch.dtype | None = None,
) -> list[str]:
    """Add the adapter stored at ``path_or_state_dict`` to ``model`` as adapter ``name``; returns its layers.

    Accepts a directory written by :func:`save_lora` or PEFT, a weights file (``.safetensors`` or ``.bin``,
    with ``adapter_config.json`` next to it), or a state dict. Keys may carry PEFT's ``base_model.model.``
    prefix and may or may not include an adapter name (``lora_A.weight`` or ``lora_A.<adapter>.weight``).
    ``config`` (PEFT's fields: ``r``, ``lora_alpha``, ``use_rslora``, ``lora_dropout``, ``rank_pattern``,
    ``alpha_pattern``, ``modules_to_save``) overrides the file's; without any, alpha equals the rank (scale 1).
    Modules listed in ``modules_to_save`` are overwritten in place. ``dtype`` is the adapters' dtype (default:
    each layer's). Everything is checked before the model changes.
    """
    _check_name(name)
    file_config = None
    if isinstance(path_or_state_dict, (str, os.PathLike)):
        state, file_config = _read_lora_files(os.fspath(path_or_state_dict))
    else:
        state = dict(path_or_state_dict)
    config = dict(config if config is not None else file_config or {})
    _check_peft_config(config)

    pairs: dict[str, dict[str, Tensor]] = {}
    others: dict[str, Tensor] = {}
    sources = set()
    for key, value in state.items():
        key = key.removeprefix(PEFT_PREFIX)
        parsed = _parse_key(key)
        if parsed is None:
            others[key] = value
            continue
        module, part, source = parsed
        sources.add(source)
        if part in pairs.setdefault(module, {}):
            raise ValueError(f"{module}.{part} appears more than once (the state dict holds several adapters)")
        pairs[module][part] = value
    if len(sources) > 1:
        raise ValueError(f"the state dict holds several adapters ({sorted(map(str, sources))}); pass one")

    modules = dict(model.named_modules())
    order = {n: i for i, n in enumerate(modules)}
    problems, plan = [], []
    for module in sorted(pairs, key=lambda n: order.get(n, len(order))):  # files may sort their keys
        ab, layer = pairs[module], modules.get(module)
        if set(ab) != set(_PARTS):
            problems.append(f"{module}: needs both lora_A and lora_B")
        elif layer is None or not supports_lora(layer):
            problems.append(f"{module}: " + ("no such layer" if layer is None else f"{type(layer).__name__} "
                                             "cannot take an adapter"))
        else:
            a, b = ab["lora_A"], ab["lora_B"]
            r = a.shape[0] if a.dim() == 2 else -1
            if a.dim() != 2 or b.dim() != 2 or tuple(a.shape) != (r, layer.in_features) \
                    or tuple(b.shape) != (layer.out_features, r):
                problems.append(f"{module}: lora_A {tuple(a.shape)} and lora_B {tuple(b.shape)} do not fit a "
                                f"{layer.in_features} -> {layer.out_features} layer")
                continue
            r_config = _pattern_value(config.get("rank_pattern"), module, config.get("r", r))
            if r_config != r:
                problems.append(f"{module}: the config says rank {r_config} but the weights have rank {r}")
            if name in _settings(layer):
                problems.append(f"{module}: already has an adapter named {name!r}")
            alpha = _pattern_value(config.get("alpha_pattern"), module, config.get("lora_alpha", r))
            plan.append((module, layer, a, b, r, float(alpha)))
    owners = list(config.get("modules_to_save") or [])
    tensors = {**dict(model.named_buffers()), **dict(model.named_parameters())} if others else {}
    copies = []
    for key, value in others.items():
        target = tensors.get(key) if any(key.startswith(o + ".") for o in owners) else None
        if target is None or target.shape != value.shape:
            problems.append(f"{key}: " + ("not a LoRA weight" if target is None else
                                          f"shape {tuple(value.shape)} does not fit {tuple(target.shape)}"))
        else:
            copies.append((target, value))
    if not plan and not problems:
        problems.append("no LoRA weights found")
    if problems:
        raise ValueError("cannot load the adapter:\n  " + "\n  ".join(problems))

    for _, layer, a, b, r, alpha in plan:
        s = add_lora_adapter(layer, name, r=r, alpha=alpha, dropout=float(config.get("lora_dropout") or 0.0),
                             use_rslora=bool(config.get("use_rslora", False)), dtype=dtype)
        s.strength = float(strength)
        with torch.no_grad():
            layer.lora_A[name].weight.copy_(a)
            layer.lora_B[name].weight.copy_(b)
    with torch.no_grad():
        for target, value in copies:
            target.copy_(value)
    return [p[0] for p in plan]


# ---------------------------------------------------------------------------------------------------------
# Merging


@torch.no_grad()
def merge_lora_into_weights(model: nn.Module, name: str = "default") -> list[str]:
    """Fold adapter ``name`` into the full-precision weights (``W += strength * scale * B @ A``) and delete it.

    Only where that is exact: plain ``nn.Linear`` layers and BitLinear / HBitLinear layers without an inner
    RMSNorm (``input_norm=False``; for HBitLinear the update is rotated like the weights). With the norm the
    weight sees the normalized input while the adapter sees the raw input, so no weight update reproduces the
    branch; packed layers have no full-precision weight. Such layers, and disabled adapters, are refused
    before anything changes. The ternary codes of a merged BitLinear are re-derived from the new weights, so
    a quantized layer no longer computes exactly what base plus adapter did: keeping the adapter as a side
    branch is the exact option (a warning says so). Returns the merged layer names.
    """
    found = _select(model, name)
    problems = []
    for n, m, _ in found:
        if not isinstance(getattr(m, "weight", None), Tensor):
            problems.append(f"{n}: packed layer without full-precision weights")
        elif getattr(m, "norm", None) is not None:
            problems.append(f"{n}: the layer normalizes its input inside (input_norm=True), the adapter does not")
        elif not m.lora_settings[name].enabled:
            problems.append(f"{n}: the adapter is disabled")
    if problems:
        raise ValueError(f"cannot merge adapter {name!r} exactly; keep it as a side branch:\n  "
                         + "\n  ".join(problems))
    quantized = []
    for n, m, _ in found:
        s = m.lora_settings[name]
        delta = (m.lora_B[name].weight.double() @ m.lora_A[name].weight.double()) * (s.strength * s.scale)
        rotate = getattr(m, "_weight_for_input_space", None)  # HBitLinear: weights act on rotated inputs
        if rotate is not None:
            delta = rotate(delta)
        m.weight.add_(delta.to(m.weight.dtype))
        _remove_adapter(m, name)
        if hasattr(m, "quant_strength"):
            quantized.append(n)
    if quantized:
        warnings.warn(f"merged adapter {name!r} into the latent weights of {len(quantized)} ternary layers: "
                      "quantizing them re-derives the ternary codes, so the quantized model only approximates "
                      "base plus adapter. Keeping the adapter as a side branch is exact.", stacklevel=2)
    return [n for n, _, _ in found]
