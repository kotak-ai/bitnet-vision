"""FLUX.2 [klein] with ternary weights, kept 2-bit in memory, and existing FLUX.2 LoRAs on top.

PrismML's Bonsai Image 4B is FLUX.2 [klein] 4B (diffusers' ``Flux2Transformer2DModel``) with the 100 matmul
linears of its 25 transformer blocks ternary: every group of 128 weights along the input axis holds only
``-s``, ``0`` and ``+s``. It is weight-only (activations stay 16-bit, no norm inside the layers). Everything
else (embedders, modulation, ``norm_out``, ``proj_out`` and the QK norms) stays 16-bit. Two layouts exist:

* ``prism-ml/bonsai-image-ternary-4B-mlx-2bit``, folder ``transformer-packed-mflux``: MLX's 2-bit affine
  format, group 128 (``<layer>.weight`` uint32 [out, in / 16], ``.scales`` and ``.biases`` bfloat16
  [out, in / 128], ``biases == -scales``, codes 0..2), 1.43 GB.
* ``prism-ml/bonsai-image-ternary-4B-unpacked``, folder ``transformer``: plain bfloat16 weights, 7.75 GB.

:func:`load_bonsai_transformer` reads either into a ``Flux2Transformer2DModel`` whose 100 ternary layers are
:class:`~bitnet_vision.weight_only.PackedTernaryLinear` (2 bits per weight plus the scales). The model is
built on PyTorch's meta device and every tensor is read on its own, so loading never holds the dense
weights: the packed transformer takes about 1.4 GB.

:func:`load_flux2_lora` loads an existing FLUX.2 klein 4B LoRA (PEFT, diffusers, ai-toolkit, BFL or kohya
files) onto that transformer as a full-precision side branch (``bitnet_vision.lora``). The formats are
normalized by diffusers' own ``Flux2KleinPipeline.lora_state_dict``, so module names, the split of fused qkv
LoRAs and the alpha scaling are exactly what ``pipe.load_lora_weights`` would use on the dense model.
Merging a LoRA into the ternary weights is not offered: re-rounding to ternary loses a small LoRA entirely.

diffusers is optional: this module imports it only inside the functions that need it.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from ..apple.mlx_ternary import decode_mlx_ternary
from ..lora import in_lora_adapter, load_lora
from ..weight_only import PackedTernaryLinear, absmean_ternary, scale_shape, split_ternary

__all__ = [
    "BONSAI_MLX",
    "BONSAI_UNPACKED",
    "BONSAI_GROUP_SIZE",
    "FLUX2_TERNARY_TARGETS",
    "ternarize_transformer",
    "round_linears_to_ternary",
    "load_bonsai_transformer",
    "Flux2LoRASummary",
    "load_flux2_lora",
]

BONSAI_MLX = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
BONSAI_UNPACKED = "prism-ml/bonsai-image-ternary-4B-unpacked"
BONSAI_GROUP_SIZE = 128
# The 100 ternary linears of Bonsai: 12 per double-stream block, 2 per single-stream block (diffusers names).
FLUX2_TERNARY_TARGETS = (
    r"transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)"
    r"|ff\.linear_in|ff\.linear_out|ff_context\.linear_in|ff_context\.linear_out)"
    r"|single_transformer_blocks\.\d+\.attn\.(to_qkv_mlp_proj|to_out)"
)
TRANSFORMER_SUBFOLDERS = ("transformer-packed-mflux", "transformer")  # MLX pack first: 1.43 GB instead of 7.75
WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
INDEX_NAME = WEIGHTS_NAME + ".index.json"


def _diffusers() -> Any:
    try:
        import diffusers
    except ImportError as err:
        raise ImportError("FLUX.2 support needs diffusers 0.41 or newer: pip install 'bitnet-vision[flux]'") from err
    if not hasattr(diffusers, "Flux2Transformer2DModel") or not hasattr(diffusers, "Flux2KleinPipeline"):
        raise ImportError(f"diffusers {diffusers.__version__} has no FLUX.2 klein support; install 0.41 or newer")
    return diffusers


def _matcher(targets: str | Iterable[str] | None) -> Any:
    """None: everything; a string: a regex matching the whole name; else names or dotted suffixes."""
    if targets is None:
        return lambda name: True
    if isinstance(targets, str):
        pattern = re.compile(targets)
        return lambda name: pattern.fullmatch(name) is not None
    names = list(targets)
    return lambda name: any(name == t or name.endswith("." + t) for t in names)


def _linears(model: nn.Module, targets: str | Iterable[str] | None) -> list[tuple[nn.Module, str, str, nn.Linear]]:
    """``(parent, child name, qualified name, layer)`` for the plain nn.Linear layers matching ``targets``."""
    match = _matcher(targets)
    found = []
    for parent_name, parent in model.named_modules(remove_duplicate=False):
        for child_name, child in parent._modules.items():  # includes repeated (shared) children
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if type(child) is nn.Linear and not in_lora_adapter(full) and match(full):
                found.append((parent, child_name, full, child))
    return found


# ---------------------------------------------------------------------------------------------------------
# Ternary layers in any diffusers transformer


def ternarize_transformer(
    model: nn.Module,
    group_size: int = BONSAI_GROUP_SIZE,
    *,
    targets: str | Iterable[str] | None = None,
    granularity: str = "group",
    cache_dense: bool = False,
) -> list[str]:
    """Replace the ternary-valued ``nn.Linear`` layers of ``model`` (in place) by packed ternary layers.

    A layer is ternary-valued when every group of ``group_size`` weights along the input axis (or every row,
    or the whole matrix, with ``granularity`` "row" or "tensor") holds only ``-s``, ``0`` and ``+s``. The
    conversion is exact. With ``targets=None`` every such layer is converted and the others are left alone;
    with ``targets`` (a regex matching the whole name, or names and dotted suffixes) exactly the matching
    layers are converted, and each one must be ternary-valued (otherwise nothing changes and ValueError names
    the layers). LoRA adapters on the layers are kept. Returns the converted names.
    """
    plan, problems, built = [], [], {}
    for parent, child_name, full, child in _linears(model, targets):
        if id(child) not in built:
            try:
                built[id(child)] = PackedTernaryLinear.from_linear(child, group_size, granularity, cache_dense,
                                                                   name=full)
            except ValueError as err:
                if targets is not None:
                    problems.append(str(err))
                continue
        plan.append((parent, child_name, full, built[id(child)]))
    if problems:
        raise ValueError("cannot convert these layers:\n  " + "\n  ".join(problems))
    if targets is not None and not plan:
        raise ValueError(f"no nn.Linear layer matches targets={targets!r}")
    for parent, child_name, _, new in plan:
        setattr(parent, child_name, new)
    return [full for _, _, full, _ in plan]


@torch.no_grad()
def round_linears_to_ternary(
    model: nn.Module,
    targets: str | Iterable[str] | None = FLUX2_TERNARY_TARGETS,
    group_size: int = BONSAI_GROUP_SIZE,
    granularity: str = "group",
    scale_dtype: torch.dtype | None = None,
) -> list[str]:
    """Round the weights of the matching ``nn.Linear`` layers to ternary values, in place; returns their names.

    Plain absmean rounding per group (``s = mean|W|``, ``W <- s * clip(round(W / s), -1, 1)``), with no
    training: a post-training baseline, and how the tests build small ternary models. The layers stay dense;
    :func:`ternarize_transformer` packs them afterwards. ``scale_dtype`` rounds the scales to that dtype first
    (bfloat16 makes the weights exactly representable in a bfloat16 file, like Bonsai's).
    """
    names = []
    for _, _, full, layer in _linears(model, targets):
        if full in names:
            continue
        codes, scale = absmean_ternary(layer.weight, granularity, group_size)
        if scale_dtype is not None:
            scale = scale.to(scale_dtype).float()
        out, k = layer.weight.shape
        rows, cols = scale.shape
        w = (codes.float().reshape(out, cols, k // cols) * scale.unsqueeze(-1)).reshape(out, k)
        layer.weight.copy_(w)
        names.append(full)
    if not names:
        raise ValueError(f"no nn.Linear layer matches targets={targets!r}")
    return names


# ---------------------------------------------------------------------------------------------------------
# Bonsai Image 4B


def _hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as err:
        raise ImportError("downloading from the Hugging Face Hub needs huggingface_hub: "
                          "pip install 'bitnet-vision[flux]'") from err
    return huggingface_hub


def _safe_open() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as err:
        raise ImportError("reading .safetensors files needs safetensors: pip install 'bitnet-vision[flux]'") from err
    return safe_open


def _repo_weight_files(repo: str, revision: str | None, token: Any) -> list[str]:
    """The weight files of a Hub repo; offline (``HF_HUB_OFFLINE=1``) or unreachable, those in the local cache."""
    hub = _hub()
    try:
        return hub.list_repo_files(repo, revision=revision, token=token)
    except Exception as err:  # noqa: BLE001 (connection and offline errors differ between huggingface_hub versions)
        try:
            root = hub.snapshot_download(repo, revision=revision, token=token, local_files_only=True,
                                         allow_patterns=["*.safetensors", "*.bin"])
        except Exception:  # noqa: BLE001
            raise err from None
        return [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                for d, _, names in os.walk(root) for f in names]


def _is_transformer_folder(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and any(
        os.path.isfile(os.path.join(path, n)) for n in (WEIGHTS_NAME, INDEX_NAME))


def _transformer_files(source: str, subfolder: str | None, revision: str | None, token: Any) -> tuple[str, list[str]]:
    """``(folder, weight files)``: the folder holds config.json (downloading only the transformer's files)."""
    source = os.fspath(source)
    if os.path.isfile(source):  # a weights file, with config.json next to it
        return os.path.dirname(os.path.abspath(source)), [source]
    folder = _transformer_folder(source, subfolder, revision, token)
    return folder, _weight_files(folder)


def _transformer_folder(source: str, subfolder: str | None, revision: str | None, token: Any) -> str:
    """The transformer folder of a local copy, or of a Hub repo (downloading only that folder)."""
    if os.path.isdir(source):
        for sub in ([subfolder] if subfolder else ["", *TRANSFORMER_SUBFOLDERS]):
            if _is_transformer_folder(os.path.join(source, sub)):
                return os.path.join(source, sub)
        raise FileNotFoundError(f"{source} holds no transformer (config.json and {WEIGHTS_NAME}), "
                                f"neither directly nor in {list(TRANSFORMER_SUBFOLDERS)}")
    hub = _hub()
    subs = [subfolder] if subfolder else list(TRANSFORMER_SUBFOLDERS)
    for sub in subs:  # a folder the repo does not have downloads nothing
        root = hub.snapshot_download(source, revision=revision, token=token,
                                     allow_patterns=[f"{sub}/*.json", f"{sub}/*.safetensors"])
        if _is_transformer_folder(os.path.join(root, sub)):
            return os.path.join(root, sub)
    raise FileNotFoundError(f"{source} has no transformer folder {subs} with config.json and {WEIGHTS_NAME}")


def _weight_files(folder: str) -> list[str]:
    index = os.path.join(folder, INDEX_NAME)
    if os.path.isfile(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(folder, s) for s in shards]
    return [os.path.join(folder, WEIGHTS_NAME)]


@dataclass
class _Ternary:
    """One ternary layer as stored: its file keys, shape and group size."""

    keys: dict[str, str]  # role ("weight", "scales", "biases", "bias") -> file key
    in_features: int
    out_features: int
    group_size: int


def _ternary_layers(
    header: Mapping[str, tuple[str, list[int]]], folder: str, group_size: int, ternary_modules: Any
) -> tuple[str, dict[str, _Ternary]]:
    """The file's layout ("mlx" or "unpacked") and its ternary layers, checked against the file's own claims."""
    found: dict[str, _Ternary] = {}
    if any(k.endswith(".scales") for k in header):
        for key in header:
            if not key.endswith(".scales"):
                continue
            module = key[: -len(".scales")]
            keys = {role: f"{module}.{role}" for role in ("weight", "scales", "biases")}
            missing = [k for k in keys.values() if k not in header]
            if missing:
                raise ValueError(f"{module}: the MLX pack lacks {missing}")
            (wdt, wshape), (_, sshape) = header[keys["weight"]], header[keys["scales"]]
            if wdt != "U32" or len(wshape) != 2 or len(sshape) != 2 or header[keys["biases"]][1] != sshape:
                raise ValueError(f"{module}: not MLX 2-bit layout (weight {wdt} {wshape}, scales {sshape})")
            k = wshape[1] * 16
            if sshape[0] != wshape[0] or k % sshape[1]:
                raise ValueError(f"{module}: weight {wshape} and scales {sshape} do not fit a 2-bit layout")
            if f"{module}.bias" in header:
                keys["bias"] = f"{module}.bias"
            found[module] = _Ternary(keys, k, wshape[0], k // sshape[1])
        qconfig = os.path.join(folder, "quantization_config.json")
        if os.path.isfile(qconfig):
            with open(qconfig) as f:
                q = json.load(f)
            if q.get("bits", 2) != 2 or q.get("solver", "ternary") != "ternary":
                raise ValueError(f"{qconfig}: bits={q.get('bits')}, solver={q.get('solver')}; expected 2-bit ternary")
            listed = set(q.get("quantized_modules") or found)
            if listed != set(found):
                raise ValueError(f"{qconfig} lists other quantized modules than the weights hold: "
                                 f"{sorted(listed ^ set(found))[:10]}")
            if any(t.group_size != q.get("group_size", t.group_size) for t in found.values()):
                raise ValueError(f"{qconfig}: group_size={q.get('group_size')} differs from the weights")
        return "mlx", found
    match = _matcher(ternary_modules)
    for key, (_, shape) in header.items():
        module = key[: -len(".weight")]
        if key.endswith(".weight") and len(shape) == 2 and match(module):
            keys = {"weight": key, **({"bias": f"{module}.bias"} if f"{module}.bias" in header else {})}
            scale_shape(shape[0], shape[1], "group", group_size)  # raises if group_size does not divide in
            found[module] = _Ternary(keys, shape[1], shape[0], group_size)
    if not found:
        raise ValueError(f"no weight matches ternary_modules={ternary_modules!r}")
    return "unpacked", found


def load_bonsai_transformer(
    source: str | os.PathLike = BONSAI_MLX,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    *,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    group_size: int = BONSAI_GROUP_SIZE,
    ternary_modules: str | Iterable[str] | None = FLUX2_TERNARY_TARGETS,
    cache_dense: bool = False,
) -> nn.Module:
    """Load Bonsai Image's transformer as a diffusers ``Flux2Transformer2DModel`` with packed ternary layers.

    ``source``: a Hugging Face repo id (:data:`BONSAI_MLX`, the default, or :data:`BONSAI_UNPACKED`), a local
    copy of either repo, a transformer folder (``config.json`` plus ``diffusion_pytorch_model.safetensors``),
    or that weights file. From the Hub only the transformer folder is downloaded (1.43 GB for the MLX pack).

    The layout is read from the file. In the MLX pack every layer with ``.scales`` is ternary; it is checked
    (``biases == -scales``, codes 0..2, ``quantization_config.json`` agrees) and decoded. In the unpacked
    layout the layers matching ``ternary_modules`` (default: Bonsai's 100 block linears) are checked to hold
    only ``-s``, ``0`` and ``+s`` in every group of ``group_size`` inputs, and loading fails loudly otherwise.
    Those layers become :class:`PackedTernaryLinear`; every other tensor is loaded as stored, cast to
    ``dtype`` (bfloat16, Bonsai's own dtype, keeps everything exact; the scales are bfloat16 too).

    The model is built on the meta device and the tensors are read one at a time, so the dense weights are
    never held: memory stays near the packed size. Returns the model in eval mode on ``device``.
    """
    diffusers = _diffusers()
    safe_open = _safe_open()
    folder, files = _transformer_files(source, subfolder, revision, token)
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    if config.get("_class_name", "Flux2Transformer2DModel") != "Flux2Transformer2DModel":
        raise ValueError(f"{folder} holds a {config['_class_name']}, not a Flux2Transformer2DModel")
    with ExitStack() as stack:
        handles, header = {}, {}  # tensor key -> open file, and -> (dtype, shape)
        for path in files:
            f = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for key in f.keys():
                sl = f.get_slice(key)
                header[key] = (sl.get_dtype(), list(sl.get_shape()))
                handles[key] = f
        layout, ternary = _ternary_layers(header, folder, group_size, ternary_modules)

        with torch.device("meta"):
            model = diffusers.Flux2Transformer2DModel.from_config(config)
            modules = dict(model.named_modules())
            problems = []
            for name, t in ternary.items():
                linear = modules.get(name)
                if type(linear) is not nn.Linear or (linear.out_features, linear.in_features) != (
                        t.out_features, t.in_features):
                    problems.append(f"{name}: the file holds a {t.out_features} x {t.in_features} ternary layer, the "
                                    f"model has {type(linear).__name__ if linear is not None else 'nothing'} there")
                    continue
                if (linear.bias is not None) != ("bias" in t.keys):
                    problems.append(f"{name}: bias in the model and in the file differ")
                    continue
                packed = PackedTernaryLinear(t.in_features, t.out_features, bias=linear.bias is not None,
                                             group_size=t.group_size, dtype=dtype, cache_dense=cache_dense)
                parent, _, child = name.rpartition(".")
                setattr(model.get_submodule(parent) if parent else model, child, packed)
        if getattr(model, "_keep_in_fp32_modules", None):
            raise NotImplementedError("this diffusers version keeps some FLUX.2 modules in float32; not handled here")
        nn.Module.to(model, dtype)  # diffusers' own .to() would warn about float32 modules even when there are none
        expected = model.state_dict()
        transient = [n for n, _ in model.named_buffers() if n not in expected]
        if transient:  # computed in __init__ and not in the files: to_empty() below would leave them uninitialized
            raise NotImplementedError(f"this diffusers version gives {type(model).__name__} buffers that its files do "
                                      f"not hold ({transient[:3]}); loading without dense weights cannot fill them")
        used = {k for t in ternary.values() for k in t.keys.values()}
        targets = {f"{name}.{b}" for name in ternary for b in ("weight_packed", "weight_scale", "bias")}
        missing = sorted(k for k in expected if k not in targets and k not in header)
        unexpected = sorted(k for k in header if k not in used and k not in expected)
        for k in sorted(set(header) - used - set(unexpected)):
            if list(expected[k].shape) != header[k][1]:
                problems.append(f"{k}: shape {header[k][1]} in the file, {list(expected[k].shape)} in the model")
        if missing:
            problems.append(f"the file lacks {len(missing)} tensors of the model, e.g. {missing[:5]}")
        if unexpected:
            problems.append(f"the file holds {len(unexpected)} tensors the model does not have, e.g. {unexpected[:5]}")
        if problems:
            raise ValueError(f"cannot load {folder}:\n  " + "\n  ".join(problems))

        model.to_empty(device=device)
        rounded = []
        with torch.no_grad():
            for name, t in ternary.items():
                layer = model.get_submodule(name)
                get = {role: handles[key].get_tensor(key) for role, key in t.keys.items()}
                if layout == "mlx":
                    codes, scale = decode_mlx_ternary(get["weight"], get["scales"], get["biases"], name=name)
                else:
                    codes, scale = split_ternary(get["weight"], "group", t.group_size, name=name)
                scaled = scale.to(dtype)
                if not torch.equal(scaled.to(scale.dtype), scale):
                    rounded.append(name)
                layer.set_ternary(codes, scaled, get.get("bias"))
                del get, codes, scale, scaled
            params = {**dict(model.named_parameters()), **dict(model.named_buffers())}
            for key in sorted(set(header) - used):
                params[key].copy_(handles[key].get_tensor(key))
    if rounded:
        warnings.warn(f"casting the stored scales to {dtype} rounds them in {len(rounded)} layers, e.g. {rounded[0]}; "
                      "the stored dtype (bfloat16 for Bonsai) is exact", stacklevel=2)
    return model.eval()


# ---------------------------------------------------------------------------------------------------------
# Existing FLUX.2 LoRAs

# Keys of text-encoder adapters in the formats diffusers and kohya write; skipped with a warning.
_TEXT_ENCODER_PREFIXES = ("text_encoder.", "text_encoder_2.", "text_encoders.", "lora_te_", "lora_te1_", "lora_te2_",
                          "lora_te3_")
# Adapter types this loader refuses, with what they are. Checked on the file's own keys, before diffusers sees
# them: diffusers drops DoRA's 'dora_scale' with only a log message, which would silently change the adapter.
_UNSUPPORTED = (
    (r"(^|\.)dora_scale$|lora_magnitude_vector", "DoRA magnitude vectors ('dora_scale' / 'lora_magnitude_vector'): "
     "DoRA rescales every output row by the norms of the weights it was trained on, and dropping them changes "
     "the adapter"),
    (r"\.hada_", "LyCORIS LoHa (the product of two low-rank pairs, a dense update per layer)"),
    (r"\.lokr_", "LyCORIS LoKr (a Kronecker-product update)"),
    (r"\.lora_mid\.", "LyCORIS LoCon with a middle (Tucker) factor"),
    (r"\.(oft_blocks|oft_R|oft_diag)", "OFT / BOFT (orthogonal fine-tuning)"),
    (r"\.(diff|diff_b)$", "full weight differences (LyCORIS 'full')"),
    (r"\.(w_norm|b_norm)$", "norm-layer updates (LyCORIS)"),
    (r"\.(a1|a2|b1|b2|bm)\.weight$", "GLoRA"),
    (r"\.(lora_B|lora_up)\.bias$", "a bias on LoRA's B matrix (lora_bias)"),
)
_LORA_KEY = re.compile(r"(.+)\.(lora_A|lora_B|lora_down|lora_up)(\.[^.]+)?\.weight$")
_FUSED_QKV = re.compile(r"(^|[._])double_blocks[._]\d+[._](img|txt)_attn[._]qkv$")
_SKIP_WEIGHTS = ("scheduler", "optimizer", "checkpoint")  # never LoRA weights (as in diffusers)


@dataclass
class Flux2LoRASummary:
    """What :func:`load_flux2_lora` attached. ``scales`` are alpha / rank per module, before ``strength``."""

    name: str
    source: str
    weight_file: str | None
    format: str
    alpha: str
    strength: float
    modules: list[str]
    ranks: dict[str, int]
    scales: dict[str, float]
    skipped: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        ranks = sorted(set(self.ranks.values()))
        scales = sorted(set(self.scales.values()))
        lines = [f"LoRA {self.name!r} from {self.weight_file or self.source} ({self.format})",
                 f"  {len(self.modules)} modules, rank {ranks if len(ranks) > 1 else ranks[0]}, "
                 f"scale {scales if len(scales) > 1 else scales[0]} ({self.alpha}), strength {self.strength}"]
        if self.skipped:
            lines.append(f"  skipped {len(self.skipped)} tensors: " + ", ".join(sorted(set(self.skipped.values()))))
        return "\n".join(lines)


def _pick(files: list[str], where: str) -> str:
    """The LoRA weights file among ``files``, as diffusers guesses it, but refusing to guess between several."""
    for ext in (".safetensors", ".bin"):
        found = [f for f in files if f.endswith(ext) and not any(s in f for s in _SKIP_WEIGHTS)]
        found = [f for f in found if f.endswith("pytorch_lora_weights" + ext)] or found
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"{where} holds several LoRA weight files {found}; choose one with weight_name=")
    raise FileNotFoundError(f"{where} holds no .safetensors or .bin LoRA weights")


def _fetch_lora(
    source: Any, weight_name: str | None, subfolder: str | None, revision: str | None, token: Any
) -> tuple[dict[str, Tensor], dict[str, Any] | None, str | None]:
    """``(state dict, lora_adapter_metadata or None, file path)``, read the way diffusers reads LoRA files."""
    if isinstance(source, Mapping):
        return dict(source), None, None
    source = os.fspath(source)
    if os.path.isfile(source):
        path = source
    elif os.path.isdir(source):
        folder = os.path.join(source, subfolder) if subfolder else source
        path = os.path.join(folder, weight_name or _pick(sorted(os.listdir(folder)), folder))
    else:
        hub = _hub()
        if weight_name is None:
            files = _repo_weight_files(source, revision, token)
            if subfolder:
                files = [f[len(subfolder) + 1:] for f in files if f.startswith(subfolder + "/")]
            weight_name = _pick(files, source)
        path = hub.hf_hub_download(source, weight_name, subfolder=subfolder, revision=revision, token=token)
    if path.endswith(".safetensors"):
        with _safe_open()(path, framework="pt", device="cpu") as f:
            raw = (f.metadata() or {}).get("lora_adapter_metadata")
            state = {k: f.get_tensor(k) for k in f.keys()}
        return state, (json.loads(raw) if raw else None), path
    return torch.load(path, map_location="cpu", weights_only=True), None, path


def _format_name(keys: Iterable[str]) -> str:
    """Which of diffusers' FLUX.2 code paths a state dict takes (Flux2LoraLoaderMixin.lora_state_dict)."""
    keys = list(keys)
    if any(".lora_down.weight" in k for k in keys):
        return "kohya"
    if any(k.startswith("base_model.model.") for k in keys):
        return "PEFT (base_model.model.)"
    if any(k.startswith("diffusion_model.") for k in keys):
        return "ai-toolkit / BFL (diffusion_model.)"
    return "diffusers (transformer.)"


def _refuse_unsupported(state: Mapping[str, Tensor], where: str) -> None:
    kinds: dict[str, list[str]] = {}
    for key in state:
        for pattern, kind in _UNSUPPORTED:
            if re.search(pattern, key):
                kinds.setdefault(kind, []).append(key)
                break
        else:
            if not _LORA_KEY.fullmatch(key) and not key.endswith(".alpha"):
                kinds.setdefault("tensors that are not LoRA A/B weights or kohya alphas", []).append(key)
    if kinds:
        raise ValueError(f"cannot load the LoRA in {where}; nothing was loaded. It holds:\n  " + "\n  ".join(
            f"{kind}: {len(keys)} tensors, e.g. {keys[:3]}" for kind, keys in kinds.items()))


def _peft_config(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """The LoRA config diffusers would build for the transformer (diffusers.utils.peft_utils._create_lora_config).

    Without ``lora_adapter_metadata`` alpha equals each module's rank (scale 1; kohya files arrive with
    alpha / rank already folded into A and B). With it, its ``transformer.`` entries form a PEFT ``LoraConfig``,
    whose defaults (r = lora_alpha = 8) fill any missing field.
    """
    if metadata is None:
        return {}
    return {"r": 8, "lora_alpha": 8, **{k[len("transformer."):]: v for k, v in metadata.items()
                                       if k.startswith("transformer.")}}


def load_flux2_lora(
    transformer: nn.Module,
    source: str | os.PathLike | Mapping[str, Tensor],
    name: str = "default",
    strength: float = 1.0,
    *,
    weight_name: str | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    adapter_dtype: torch.dtype | None = torch.float32,
) -> Flux2LoRASummary:
    """Attach an existing FLUX.2 LoRA to ``transformer`` as adapter ``name`` and return what was attached.

    ``source``: a ``.safetensors`` (or ``.bin``) file, a folder holding one, a Hugging Face repo id (``weight_name``
    picks the file when there are several), or a state dict. ``transformer`` is a ``Flux2Transformer2DModel``,
    packed (:func:`load_bonsai_transformer`) or dense: module names are diffusers' own, unchanged.

    Every format diffusers 0.41 reads for FLUX.2 works (PEFT ``base_model.model.``, diffusers ``transformer.``,
    ai-toolkit ``diffusion_model.``, BFL names, kohya ``lora_unet_``), because the keys go through diffusers'
    public ``Flux2KleinPipeline.lora_state_dict``. Scaling follows diffusers exactly: kohya's alpha / rank is
    folded into the weights, a ``lora_adapter_metadata`` header gives a PEFT config (rank and alpha patterns,
    rsLoRA), and a file with neither gets scale 1.

    Text-encoder parts are skipped with a warning (listed in the summary). Anything else that cannot be
    attached is refused before the model changes, with the reason: DoRA, LoHa, LoKr and other LyCORIS types,
    LoRA biases, tensors diffusers' converter would drop, and modules or shapes the transformer lacks.
    ``strength`` scales the adapter's output (``set_lora_strength`` changes it later); several adapters with
    different names add up. ``adapter_dtype`` is the dtype A and B are stored in: float32 by default, so float32
    files load unrounded (diffusers' own loader stores them in the model's dtype; None does the same here).
    The branch always computes in float32.
    """
    diffusers = _diffusers()
    if not isinstance(transformer, nn.Module) and isinstance(getattr(transformer, "transformer", None), nn.Module):
        transformer = transformer.transformer  # a pipeline was passed
    raw, metadata, path = _fetch_lora(source, weight_name, subfolder, revision, token)
    where = path or ("a state dict" if isinstance(source, Mapping) else os.fspath(source))
    skipped = {k: "text-encoder adapter (only the transformer is loaded)" for k in raw
               if k.startswith(_TEXT_ENCODER_PREFIXES)}
    if skipped:
        warnings.warn(f"{where}: skipped {len(skipped)} text-encoder LoRA tensors; only the transformer takes "
                      "adapters here", stacklevel=2)
    state = {k: v for k, v in raw.items() if k not in skipped}
    if not state:
        raise ValueError(f"{where} holds no transformer LoRA weights")
    _refuse_unsupported(state, where)
    fmt = _format_name(state)
    parts: dict[str, set[str]] = {}
    for m in filter(None, map(_LORA_KEY.fullmatch, state)):
        parts.setdefault(m.group(1), set()).add(m.group(2))
    unpaired = sorted(k for k, v in parts.items() if v not in ({"lora_A", "lora_B"}, {"lora_down", "lora_up"}))
    if unpaired:
        raise ValueError(f"cannot load the LoRA in {where}; nothing was loaded. {len(unpaired)} modules lack one of "
                         f"their two matrices, e.g. {unpaired[:3]}")
    alphas = [k for k in state if k.endswith(".alpha")]
    if alphas and fmt != "kohya":
        raise ValueError(f"cannot load the LoRA in {where}; nothing was loaded. It is in {fmt} format but holds "
                         f"{len(alphas)} '.alpha' tensors (e.g. {alphas[:2]}), which only kohya files carry; "
                         "diffusers 0.41 refuses such a FLUX.2 file too, rather than guess how to scale it")
    sources = set(parts)
    expected = len(sources) + 2 * sum(1 for s in sources if _FUSED_QKV.search(s))  # a fused qkv LoRA becomes 3

    work = dict(state)  # diffusers' kohya converter removes the keys it uses from the dict it gets
    try:
        converted = diffusers.Flux2KleinPipeline.lora_state_dict(work)
    except ValueError as err:
        raise ValueError(f"diffusers could not convert the LoRA in {where} ({fmt} format): {err}") from err
    lora, other = {}, []
    for key, value in converted.items():
        if key.startswith("transformer.") and re.search(r"\.lora_[AB]\.weight$", key):
            lora[key[len("transformer."):]] = value
        else:
            other.append(key)
    if other:
        raise ValueError(f"cannot load the LoRA in {where}: after conversion it holds {len(other)} tensors that "
                         f"are not transformer LoRA weights, e.g. {other[:5]} (alpha keys outside the kohya format "
                         "are not read by diffusers' FLUX.2 loader either)")
    modules = {k.rsplit(".lora_", 1)[0] for k in lora}
    if len(modules) != expected:
        left = sorted({k.rsplit(".", 2)[0] for k in work}) if fmt == "kohya" else []
        raise ValueError(f"cannot load the LoRA in {where}: it holds {len(sources)} LoRA modules, which should "
                         f"become {expected} transformer modules, but diffusers' converter returned {len(modules)}"
                         + (f"; it did not convert {left}" if left else ""))

    config = _peft_config(metadata)
    attached = load_lora(transformer, lora, name=name, strength=strength, config=config, dtype=adapter_dtype)
    layers = dict(transformer.named_modules())
    settings = {m: layers[m].lora_settings[name] for m in attached}
    alpha = ("lora_adapter_metadata in the file" if metadata is not None else
             "kohya alpha / rank, folded into A and B" if fmt == "kohya" else "no alpha in the file: scale 1")
    return Flux2LoRASummary(
        name=name, source="a state dict" if isinstance(source, Mapping) else os.fspath(source), weight_file=path,
        format=fmt, alpha=alpha, strength=float(strength), modules=attached,
        ranks={m: s.r for m, s in settings.items()}, scales={m: s.scale for m, s in settings.items()}, skipped=skipped)
