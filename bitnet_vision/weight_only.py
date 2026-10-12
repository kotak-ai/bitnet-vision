"""Weight-only ternary layers with per-group scales, for ternary models made elsewhere.

``BitLinear`` and ``BitLinearInference`` follow BitNet b1.58: an RMSNorm inside the layer, 8-bit activations
and one weight scale per matrix (or per row). Ternary models made by others can be different. PrismML's Bonsai
Image 4B, a ternary FLUX.2 klein, is weight-only: its activations stay in 16-bit floating point, its layers
have no inner norm, and every group of 128 weights along the input axis has its own scale. Such a layer
computes ``x @ W.T + bias`` with ``W = codes * scale``, codes in {-1, 0, +1}.

:class:`PackedTernaryLinear` holds such a layer in 2 bits per weight: ``weight_packed`` uses the repo's
:func:`~bitnet_vision.export.pack_ternary` layout (uint8 ``[ceil(out / 4), in]``) and ``weight_scale`` holds
one scale per group, shape ``[out, in / group_size]`` (``[out, 1]`` per row, ``[1, 1]`` per tensor). The forward
pass rebuilds the dense weight in the input's dtype (exact: each weight is a code times a scale) and calls
``F.linear``. That dense copy is dropped after the call unless ``cache_dense=True``, so the layer stays at 2 bits
per weight in memory, plus one dense layer while it runs.

Why a separate class and not a mode of ``BitLinearInference``: that layer's whole forward pass (inner RMSNorm,
Hadamard rotation, int8 activations, integer matmul, float32 rescaling) and its ports (Metal kernels, MLX) are
built for BitNet's W1.58A8 recipe. A weight-only layer shares only the 2-bit packing, so a mode flag would add
a branch to every method and every port, each of which would then have to refuse the other mode.

LoRA adapters (``bitnet_vision.lora``) work as on the other ternary layers: a full-precision side branch on the
raw input, added after the bias. Merging them is refused (there are no full-precision weights to merge into, and
re-rounding to ternary loses a small adapter entirely).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .export import VALUES_PER_BYTE, pack_ternary
from .lora import LoRASettings, copy_lora, lora_forward

__all__ = ["GRANULARITIES", "PackedTernaryLinear", "scale_shape", "split_ternary", "absmean_ternary"]

GRANULARITIES = ("group", "row", "tensor")
_CHUNK_ROWS = 4096  # rows checked at a time, so validating a large layer needs little extra memory


def scale_shape(out_features: int, in_features: int, granularity: str = "group", group_size: int = 128) -> tuple:
    """Shape of the scale tensor: ``(out, in / group_size)``, ``(out, 1)`` per row or ``(1, 1)`` per tensor."""
    if granularity == "group":
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise ValueError(f"group_size must be a positive int, got {group_size!r}")
        if in_features % group_size:
            raise ValueError(f"group_size={group_size} does not divide in_features={in_features}")
        return out_features, in_features // group_size
    if granularity == "row":
        return out_features, 1
    if granularity == "tensor":
        return 1, 1
    raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")


def _grouped(w: Tensor, rows: int, cols: int) -> Tensor:
    """``w`` [out, in] viewed as [out, cols, in / cols]; scales of shape [rows, cols] broadcast against it."""
    return w.reshape(w.shape[0], cols, w.shape[1] // cols)


def split_ternary(
    weight: Tensor, granularity: str = "group", group_size: int = 128, name: str = "weight"
) -> tuple[Tensor, Tensor]:
    """``(codes, scale)`` of a weight that already is ternary per group: ``weight == codes * scale`` exactly.

    Each group (``group_size`` weights along the input axis, a row, or the whole tensor) may hold only the
    values ``-s``, ``0`` and ``+s`` for one ``s >= 0``, which becomes its scale (an all-zero group gets scale 0).
    Raises ValueError, naming the first offending groups, otherwise. ``codes`` is int8 [out, in]; ``scale`` has
    the weight's dtype and shape :func:`scale_shape`.
    """
    if weight.dim() != 2:
        raise ValueError(f"{name}: expected a 2-D weight, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"{name}: expected a floating-point weight, got {weight.dtype}")
    out, k = weight.shape
    try:
        rows, cols = scale_shape(out, k, granularity, group_size)
    except ValueError as err:
        raise ValueError(f"{name}: {err}") from None
    w = weight.detach()
    if granularity == "group":
        scale = _grouped(w, rows, cols).abs().amax(dim=-1)
    elif granularity == "row":
        scale = w.abs().amax(dim=1, keepdim=True)
    else:
        scale = w.abs().amax().reshape(1, 1)
    if not torch.isfinite(scale).all():
        raise ValueError(f"{name}: the weight holds NaN or infinite values")
    codes = torch.empty(out, k, dtype=torch.int8, device=w.device)
    examples, n_bad = [], 0
    for r0 in range(0, out, _CHUNK_ROWS):
        r1 = min(out, r0 + _CHUNK_ROWS)
        g = _grouped(w[r0:r1], rows, cols)
        s = (scale[r0:r1] if rows == out else scale).unsqueeze(-1)
        ok = ((g == 0) | (g == s) | (g == -s)).all(dim=-1)  # [r1 - r0, cols]
        if not ok.all():
            where = (~ok).nonzero().tolist()
            n_bad += len(where)
            for i, j in where[: max(0, 3 - len(examples))]:
                values = torch.unique(g[i, j]).float().tolist()
                examples.append(f"row {r0 + i}, group {j}: {len(values)} distinct values, e.g. {values[:5]}")
        codes[r0:r1] = torch.sign(w[r0:r1]).to(torch.int8)
    if n_bad:
        what = {"group": f"groups of {group_size} weights", "row": "rows", "tensor": "tensor"}[granularity]
        raise ValueError(f"{name} is not ternary: {n_bad} {what} hold values other than -s, 0 and +s for one "
                         "scale s (" + "; ".join(examples) + ")")
    return codes, scale


def absmean_ternary(
    weight: Tensor, granularity: str = "group", group_size: int = 128, eps: float = 1e-5
) -> tuple[Tensor, Tensor]:
    """Round a full-precision weight to ternary: ``scale = mean|W|`` per group, ``codes = clip(round(W / scale))``.

    BitNet's absmean rounding (round to nearest, no training), per group. Returns int8 codes [out, in] and
    float32 scales of shape :func:`scale_shape`; ``codes * scale`` approximates the weight.
    """
    out, k = weight.shape
    rows, cols = scale_shape(out, k, granularity, group_size)
    w = weight.detach().float()
    if granularity == "group":
        scale = _grouped(w, rows, cols).abs().mean(dim=-1)
    elif granularity == "row":
        scale = w.abs().mean(dim=1, keepdim=True)
    else:
        scale = w.abs().mean().reshape(1, 1)
    scale = scale.clamp(min=eps)
    codes = (_grouped(w, rows, cols) / scale.unsqueeze(-1)).round().clamp(-1, 1).reshape(out, k)
    return codes.to(torch.int8), scale


class PackedTernaryLinear(nn.Module):
    """Weight-only ternary linear layer, 2 bits per weight, one scale per group of inputs.

    Args:
        in_features, out_features, bias: as for ``nn.Linear`` (``bias`` defaults to False).
        group_size: weights per scale along the input axis (any divisor of ``in_features``; 128 for Bonsai).
        granularity: "group" (default), "row" (one scale per output row) or "tensor" (one scale).
        dtype: dtype of ``weight_scale`` and ``bias`` (Bonsai stores bfloat16 scales).
        cache_dense: keep the dense weight after the first call, per dtype and device (off by default: the
            layer then holds only its 2-bit codes and scales between calls).

    The forward pass casts the scales to the input's dtype, so float32, float16 and bfloat16 inputs each
    see exactly ``codes * scale`` rounded once to their dtype, the weight a dense ``nn.Linear`` in that dtype
    would hold. Build one from an ``nn.Linear`` whose weights are already ternary with :meth:`from_linear`.
    """

    lora_settings: dict[str, LoRASettings] | None = None  # adapters, see bitnet_vision.lora

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        group_size: int = 128,
        granularity: str = "group",
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        cache_dense: bool = False,
    ) -> None:
        super().__init__()
        rows, cols = scale_shape(out_features, in_features, granularity, group_size)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.granularity = granularity
        self.group_size = in_features // cols
        self.cache_dense = bool(cache_dense)
        block = -(-out_features // VALUES_PER_BYTE)
        self.register_buffer("weight_packed", torch.full((block, in_features), 0b01010101, dtype=torch.uint8,
                                                         device=device))
        self.register_buffer("weight_scale", torch.ones(rows, cols, dtype=dtype, device=device))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.bias = None
        self._dense: dict[tuple, Tensor] = {}

    # The dense copy belongs to the current weights and device: forget it when either changes.
    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._dense = {}
        super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn: Any, *args: Any, **kwargs: Any) -> PackedTernaryLinear:
        self._dense = {}
        return super()._apply(fn, *args, **kwargs)

    @torch.no_grad()
    def set_ternary(self, codes: Tensor, scale: Tensor, bias: Tensor | None = None) -> None:
        """Load weights given as int codes [out, in] in {-1, 0, 1} and scales of shape ``weight_scale.shape``."""
        if tuple(codes.shape) != (self.out_features, self.in_features):
            raise ValueError(f"codes have shape {tuple(codes.shape)}, expected {(self.out_features, self.in_features)}")
        if tuple(scale.shape) != tuple(self.weight_scale.shape):
            raise ValueError(f"scale has shape {tuple(scale.shape)}, expected {tuple(self.weight_scale.shape)}")
        self.weight_packed.copy_(pack_ternary(codes))  # packed where the codes are, then moved
        self.weight_scale.copy_(scale)
        if bias is not None:
            if self.bias is None:
                raise ValueError("this layer was built without a bias")
            self.bias.copy_(bias)
        self._dense = {}

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls, linear: nn.Linear, group_size: int = 128, granularity: str = "group", cache_dense: bool = False,
        name: str = "weight",
    ) -> PackedTernaryLinear:
        """Pack an ``nn.Linear`` whose weight is already ternary per group (see :func:`split_ternary`).

        Exact: the new layer computes what ``linear`` computes. Scales and bias keep the weight's dtype, and LoRA
        adapters of ``linear`` are copied over.
        """
        codes, scale = split_ternary(linear.weight, granularity, group_size, name=name)
        new = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, group_size=group_size,
                  granularity=granularity, dtype=linear.weight.dtype, device=linear.weight.device,
                  cache_dense=cache_dense)
        new.set_ternary(codes, scale, None if linear.bias is None else linear.bias.detach().to(linear.weight.dtype))
        copy_lora(linear, new)
        return new.train(linear.training)

    def codes(self) -> Tensor:
        """The ternary codes, int8 [out, in]."""
        return self._unpack(torch.int8)

    def _unpack(self, dtype: torch.dtype) -> Tensor:
        """Codes in ``dtype``, written block by block into one tensor (no int16 or full-size uint8 copy)."""
        packed = self.weight_packed
        block = packed.shape[0]
        w = torch.empty(block * VALUES_PER_BYTE, self.in_features, dtype=dtype, device=packed.device)
        for i in range(VALUES_PER_BYTE):
            w[i * block : (i + 1) * block].copy_((packed >> (2 * i)) & 3)
        return w[: self.out_features].sub_(1)

    def dequantize(self, dtype: torch.dtype | None = None) -> Tensor:
        """The dense weight ``codes * scale`` [out, in] in ``dtype`` (default: the scales' dtype)."""
        dtype = self.weight_scale.dtype if dtype is None else dtype
        key = (dtype, self.weight_packed.device)
        w = self._dense.get(key)
        if w is None:
            w = self._unpack(dtype)
            rows, cols = self.weight_scale.shape
            _grouped(w, rows, cols).mul_(self.weight_scale.to(dtype).unsqueeze(-1))
            if self.cache_dense:
                self._dense[key] = w
        return w

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point():
            raise TypeError(f"PackedTernaryLinear needs a floating-point input, got {x.dtype}")
        bias = None if self.bias is None else self.bias.to(x.dtype)
        y = F.linear(x, self.dequantize(x.dtype), bias)
        return lora_forward(self, x, y)

    def extra_repr(self) -> str:
        detail = f"group_size={self.group_size}" if self.granularity == "group" else f"granularity={self.granularity}"
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"{detail}, scale_dtype={self.weight_scale.dtype}, cache_dense={self.cache_dense}")
