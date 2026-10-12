"""MLX port of the exported ternary layer (``BitLinearInference``) and of one BitViT block.

MLX is optional: nothing here imports it until a function or class that needs it is used, and the classes
(``MLXBitLinear``, ``MLXBlock``) are built on first access.

Weights. MLX's 2-bit affine format dequantizes ``w = scale * q + bias`` with codes ``q`` packed 16 per
uint32 along the input (K) axis, least significant bits first, and one scale and bias per group of K.
Storing ``q = code + 1`` with ``scale = g`` and ``bias = -g`` represents ``W = codes * g`` exactly.
``mx.quantize(bits=2)`` does not: its zero-preserving rounding turns ``-g`` into ``-g / 2``. PrismML's Bonsai
Image MLX pack stores its ternary layers in exactly this encoding, with one scale per group of 128 inputs.
:func:`encode_mlx_ternary` and :func:`decode_mlx_ternary` convert between it and ternary codes in plain PyTorch
(no MLX needed); :func:`pack_ternary_mlx` builds MLX arrays.

Arithmetic. ``MLXBitLinear`` computes exactly what the repo's layer computes: RMSNorm and per-token int8
absmax quantization in float32, an integer matmul (activation codes times ternary codes, weights stored
with unit scales), then ``/ activation_scale * weight_scale + bias`` in float32. In float32 every partial
sum is an integer below 2^24, so the matmul is exact in any summation order and, without the RMSNorm, the
output on MLX's CPU backend is bit-identical to ``BitLinearInference``. In float16 and bfloat16 the codes
enter the matmul divided by 128 (exact), so its output stays far below float16's maximum (raw sums reach
127 * K, 195,072 at K = 1536) and is rounded only once: Metal's matmul kernels accumulate in float32. MLX's
CPU quantized matmul accumulates half precision in half precision, so judge half precision on the GPU.

LoRA adapters are added as dense float32 matmuls on the raw input. Their sums are not integers, so they
match PyTorch to float32 rounding rather than bit for bit.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ..export import BitLinearInference, convert_for_inference, unpack_ternary
from ..layers import BitLinear

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

__all__ = [
    "GROUP_SIZES",
    "encode_mlx_ternary",
    "decode_mlx_ternary",
    "pack_ternary_mlx",
    "from_packed_repo",
    "from_packed_ternary_linear",
    "mlx_block_from_vit_block",
    "compiled",
    "to_mx",
    "to_torch",
]

GROUP_SIZES = (32, 64, 128)
_LAZY_CLASSES = ("MLXBitLinear", "MLXBlock")  # built on first access; left out of __all__ so * never needs MLX
# Activation codes enter the matmul divided by 128 in half precision (exact; see the module docstring).
_HALF_PRESCALE = 1.0 / 128.0


def _mlx() -> tuple[Any, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as mnn
        import numpy  # noqa: F401  (arrays move between torch and MLX through NumPy; neither package requires it)
    except ImportError as err:
        raise ImportError("bitnet_vision.apple.mlx_ternary needs MLX and NumPy (pip install mlx numpy)") from err
    return mx, mnn


def _mx_dtype(dtype: Any) -> Any:
    mx, _ = _mlx()
    if isinstance(dtype, str):
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"dtype must be 'float32', 'float16' or 'bfloat16', got {dtype!r}")
        return getattr(mx, dtype)
    if dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError(f"dtype must be float32, float16 or bfloat16, got {dtype}")
    return dtype


def _numpy(x: Any) -> np.ndarray:
    mx, _ = _mlx()  # first, so a missing NumPy gets the install hint
    import numpy as np

    if torch.is_tensor(x):
        x = x.detach().cpu()
        return (x.float() if x.dtype == torch.bfloat16 else x).numpy()
    if isinstance(x, mx.array):
        return np.array(x.astype(mx.float32) if x.dtype == mx.bfloat16 else x)
    return np.asarray(x)


def to_mx(x: Any, dtype: Any = None) -> mx.array:
    """Convert a torch tensor (any device), numpy array or MLX array to an MLX array, optionally cast."""
    mx, _ = _mlx()
    if isinstance(x, mx.array):
        out = x
    elif torch.is_tensor(x) and x.dtype == torch.bfloat16:
        out = mx.array(_numpy(x)).astype(mx.bfloat16)
    else:
        out = mx.array(_numpy(x))
    return out if dtype is None else out.astype(getattr(mx, dtype) if isinstance(dtype, str) else dtype)


def to_torch(x: Any, dtype: torch.dtype | None = None) -> Tensor:
    """Convert an MLX array to a CPU torch tensor (bfloat16 stays bfloat16), optionally cast."""
    mx, _ = _mlx()
    out = torch.from_numpy(_numpy(x))
    if isinstance(x, mx.array) and x.dtype == mx.bfloat16:
        out = out.to(torch.bfloat16)
    return out if dtype is None else out.to(dtype)


_CODES_PER_WORD = 16  # 2-bit codes per uint32
_DECODE_ROWS = 2048  # rows decoded at a time, so a large layer needs little extra memory


def encode_mlx_ternary(codes: Tensor, scales: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """MLX's 2-bit affine layout of ``W = codes * scales``, in PyTorch: ``(w_q, scales, -scales)``.

    ``codes`` is [N, K] in {-1, 0, 1} and ``scales`` is [N, K / g] (one scale per group of g inputs). ``w_q`` is
    [N, K / 16] with ``code + 1`` in 2 bits each, least significant bits first along K: uint32, or int32 with
    the same bits on PyTorch versions without ``torch.uint32`` (before 2.3). This is the layout of
    :func:`pack_ternary_mlx` and of the Bonsai Image MLX pack.
    """
    if codes.dim() != 2 or scales.dim() != 2 or scales.shape[0] != codes.shape[0]:
        raise ValueError(f"codes must be [N, K] and scales [N, K / group], got {tuple(codes.shape)} and "
                         f"{tuple(scales.shape)}")
    n, k = codes.shape
    if k % _CODES_PER_WORD or k % scales.shape[1]:
        raise ValueError(f"K={k} must be a multiple of 16 and of the number of scale groups ({scales.shape[1]})")
    if codes.numel() and (codes.min() < -1 or codes.max() > 1):
        raise ValueError("codes must hold only -1, 0 and 1")
    q = (codes.to(torch.int64) + 1).reshape(n, k // _CODES_PER_WORD, _CODES_PER_WORD)
    shifts = torch.arange(0, 32, 2, dtype=torch.int64, device=codes.device)
    words = (q << shifts).sum(dim=-1)  # disjoint bits, so the sum is the bitwise or
    w_q = words.to(torch.uint32) if hasattr(torch, "uint32") else (words - (words >> 31 << 32)).to(torch.int32)
    return w_q, scales, -scales


def decode_mlx_ternary(w_q: Tensor, scales: Tensor, biases: Tensor, name: str = "weight") -> tuple[Tensor, Tensor]:
    """Read an MLX 2-bit affine tensor that holds ternary weights: returns int8 codes [N, K] and ``scales``.

    Checks that it is ternary in the exact encoding above: ``biases == -scales`` everywhere and every code is
    0, 1 or 2 (so ``w = scale * q + bias = scale * (q - 1)``). Raises ValueError otherwise. ``w_q`` may be
    uint32 or int32 (the same bits).
    """
    if w_q.dim() != 2 or scales.shape != biases.shape or scales.dim() != 2 or scales.shape[0] != w_q.shape[0]:
        raise ValueError(f"{name}: expected weight [N, K / 16] and scales, biases [N, K / group], got "
                         f"{tuple(w_q.shape)}, {tuple(scales.shape)}, {tuple(biases.shape)}")
    if not torch.equal(biases, -scales):
        diff = (biases != -scales).sum().item()
        raise ValueError(f"{name}: biases differ from -scales in {diff} groups, so the layer is not ternary "
                         "(an ordinary 2-bit layer has four levels per group)")
    if getattr(torch, "uint32", None) is not None and w_q.dtype == torch.uint32:
        w_q = w_q.view(torch.int32)
    elif w_q.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name}: expected uint32 or int32 packed codes, got {w_q.dtype}")
    n, words = w_q.shape
    k = words * _CODES_PER_WORD
    if k % scales.shape[1]:
        raise ValueError(f"{name}: K={k} is not a multiple of the number of scale groups ({scales.shape[1]})")
    shifts = torch.arange(0, 32, 2, dtype=w_q.dtype, device=w_q.device)
    codes = torch.empty(n, k, dtype=torch.int8, device=w_q.device)
    for r0 in range(0, n, _DECODE_ROWS):
        q = (w_q[r0 : r0 + _DECODE_ROWS].unsqueeze(-1) >> shifts) & 3  # arithmetic shift keeps the low bits right
        if (q == 3).any():
            raise ValueError(f"{name}: holds 2-bit code 3, which a ternary layer never uses")
        codes[r0 : r0 + _DECODE_ROWS] = (q.reshape(q.shape[0], k) - 1).to(torch.int8)
    return codes, scales


def _scale_matrix(weight_scale: Any, rows: int, k: int, group_size: int | None) -> tuple[np.ndarray, int]:
    """float32 scales [rows, k / group_size] for MLX, from a scalar, one value per row, or [rows, groups]."""
    import numpy as np

    s = np.asarray(_numpy(weight_scale), dtype=np.float32)
    if s.ndim == 2 and s.shape[0] == rows and s.shape[1] > 1:  # one scale per group of inputs
        if k % s.shape[1]:
            raise ValueError(f"in_features={k} is not a multiple of the {s.shape[1]} scale groups")
        span = k // s.shape[1]
        valid = [g for g in GROUP_SIZES if span % g == 0]
        if group_size is None:
            if not valid:
                raise ValueError(f"scale groups of {span} inputs fit none of MLX's group sizes {GROUP_SIZES}")
            group_size = max(valid)
        elif group_size not in valid:
            raise ValueError(f"group_size={group_size} does not divide the scale groups of {span} inputs "
                             f"(valid: {valid})")
        return np.repeat(s, span // group_size, axis=1), group_size
    s = s.reshape(-1)
    if s.size == 1:
        s = np.full(rows, s[0], dtype=np.float32)
    elif s.size != rows:
        raise ValueError(f"weight_scale must be a scalar, one value per row ({rows}) or one per group "
                         f"[{rows}, groups], got shape {np.shape(weight_scale)}")
    group_size = 64 if group_size is None else group_size
    if group_size not in GROUP_SIZES:
        raise ValueError(f"group_size must be one of {GROUP_SIZES}, got {group_size}")
    if k % group_size:
        valid = [g for g in GROUP_SIZES if k % g == 0]
        raise ValueError(f"in_features={k} is not a multiple of group_size={group_size} (valid: {valid})")
    return np.repeat(s[:, None], k // group_size, axis=1), group_size


def pack_ternary_mlx(
    codes: Any, weight_scale: Any, group_size: int | None = None, dtype: Any = "float32"
) -> tuple[mx.array, mx.array, mx.array]:
    """Pack ternary ``codes`` [N, K] into MLX 2-bit affine arrays ``(w_q, scales, biases)``.

    ``weight_scale`` is the multiplier, so ``W = codes * weight_scale``: a scalar, one value per row, or one
    per group of inputs ([N, G], as in ``PackedTernaryLinear`` and Bonsai). The result represents W exactly
    (``scales`` and ``biases`` are rounded to ``dtype``) for
    ``mx.quantized_matmul(x, w_q, scales, biases, transpose=True, group_size=group_size, bits=2)``:
    ``w_q`` is uint32 [N, K / 16] holding ``code + 1`` in 2 bits each, least significant bits first along K;
    ``scales`` is ``weight_scale`` and ``biases`` is ``-weight_scale``, both [N, K / group_size]. ``group_size``
    defaults to 64 for scalar and per-row scales and to the largest MLX group size that fits inside one scale
    group for per-group scales (128 for Bonsai).
    """
    mx, _ = _mlx()
    import numpy as np

    c = _numpy(codes)
    if c.ndim != 2:
        raise ValueError(f"codes must have shape [out_features, in_features], got {c.shape}")
    rows, k = c.shape
    scales, group_size = _scale_matrix(weight_scale, rows, k, group_size)
    if not np.isin(c, (-1, 0, 1)).all():  # also rejects NaN and non-integer floats
        raise ValueError("codes must hold only -1, 0 and 1")
    q = (c.astype(np.int64) + 1).astype(np.uint32).reshape(rows, k // 16, 16)
    w_q = np.bitwise_or.reduce(q << np.arange(0, 32, 2, dtype=np.uint32), axis=-1)
    dt = _mx_dtype(dtype)
    return mx.array(w_q), mx.array(scales).astype(dt), mx.array(-scales).astype(dt)


def from_packed_repo(
    weight_packed: Any, out_features: int, weight_scale: Any, group_size: int = 64, dtype: Any = "float32"
) -> tuple[mx.array, mx.array, mx.array]:
    """:func:`pack_ternary_mlx` starting from the repo's packed uint8 layout (``BitLinearInference.weight_packed``)."""
    codes = unpack_ternary(torch.as_tensor(_numpy(weight_packed)), out_features)
    return pack_ternary_mlx(codes, weight_scale, group_size=group_size, dtype=dtype)


def from_packed_ternary_linear(layer: nn.Module, dtype: Any = None) -> tuple[mx.array, mx.array, mx.array, int]:
    """MLX arrays ``(w_q, scales, biases, group_size)`` of a weight-only ``PackedTernaryLinear``.

    Exact: ``mx.quantized_matmul(x, w_q, scales, biases, transpose=True, group_size=group_size, bits=2)``
    equals ``x @ (codes * scale).T``. ``dtype`` defaults to the layer's scale dtype (bfloat16 for Bonsai).
    Bias and LoRA adapters are not included.
    """
    from ..weight_only import PackedTernaryLinear

    if not isinstance(layer, PackedTernaryLinear):
        raise TypeError(f"expected a PackedTernaryLinear, got {type(layer).__name__}")
    if dtype is None:
        dtype = str(layer.weight_scale.dtype).removeprefix("torch.")
    span = layer.in_features // layer.weight_scale.shape[1]  # inputs per scale
    fits = [g for g in GROUP_SIZES if span % g == 0]
    if not fits:
        raise ValueError(f"scale groups of {span} inputs fit none of MLX's group sizes {GROUP_SIZES}")
    scale = layer.weight_scale.detach().float().cpu()
    w_q, scales, biases = pack_ternary_mlx(layer.codes().cpu(), scale, group_size=max(fits), dtype=dtype)
    return w_q, scales, biases, max(fits)


def compiled(fn: Callable, **kwargs: Any) -> Callable:
    """``mx.compile(fn, **kwargs)``: fuses the elementwise steps around each matmul into a few kernels.

    Arrays the function captures (a module's weights) are traced as constants. ``shapeless=True`` is unsafe
    for :class:`MLXBlock`, whose head split reads the input shape.
    """
    mx, _ = _mlx()
    return mx.compile(fn, **kwargs)


def _check_supported(layer: BitLinearInference) -> None:
    if layer.hadamard is not None:
        raise NotImplementedError("MLXBitLinear does not support Hadamard-rotated layers (HBitLinear)")
    if layer.act_bits != 8 or layer.act_method != "absmax":
        raise NotImplementedError(
            f"MLXBitLinear supports only 8-bit absmax activations, got act_bits={layer.act_bits}, "
            f"act_method={layer.act_method!r}"
        )


def _adapters(layer: BitLinearInference) -> list[tuple[Tensor, Tensor, float]]:
    """``(A, B, strength * scale)`` in float32 for every enabled LoRA adapter of ``layer``."""
    return [
        (layer.lora_A[k].weight.detach().float(), layer.lora_B[k].weight.detach().float(), s.strength * s.scale)
        for k, s in (layer.lora_settings or {}).items()
        if s.enabled and s.strength != 0
    ]


def _as_inference_layer(layer: nn.Module) -> BitLinearInference:
    if isinstance(layer, BitLinear):
        layer = convert_for_inference(layer)
    if not isinstance(layer, BitLinearInference):
        raise TypeError(f"expected a BitLinearInference (or BitLinear) layer, got {type(layer).__name__}")
    return layer


@functools.cache
def _classes() -> dict[str, type]:
    mx, mnn = _mlx()

    class MLXBitLinear(mnn.Module):
        """MLX version of ``BitLinearInference`` (8-bit absmax activations, no Hadamard rotation).

        Build it with :meth:`from_inference`. The constructor takes either packed unit-scale ternary weights
        (``w_q``, ``scales``, ``biases``, ``group_size`` as from ``pack_ternary_mlx(codes, 1.0)``) or a dense
        ``weight`` holding the codes, plus the repo's float32 ``weight_scale``, ``bias`` and RMSNorm settings.
        The dense variant runs a plain matmul instead of ``mx.quantized_matmul``; in float32 both are exact
        and give identical results. ``dtype`` is the matmul dtype; norm, quantization and the rescale run in
        float32, and the output has the input's dtype.

        ``adapters`` lists LoRA branches as ``(A [r, in], B [out, r], coefficient)``, coefficient being
        strength times scale; each adds ``coefficient * (x @ A.T) @ B.T`` on the raw input, in float32.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            weight_scale: Any,
            bias: Any = None,
            norm: bool = True,
            norm_weight: Any = None,
            norm_eps: float = 1e-6,
            w_q: mx.array | None = None,
            scales: mx.array | None = None,
            biases: mx.array | None = None,
            group_size: int | None = None,
            weight: mx.array | None = None,
            dtype: Any = "float32",
            adapters: Sequence[tuple[Any, Any, float]] = (),
        ) -> None:
            super().__init__()
            if (weight is None) == (w_q is None):
                raise ValueError("pass either packed weights (w_q, scales, biases) or a dense weight")
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.compute_dtype = _mx_dtype(dtype)
            self.packed = w_q is not None
            self.group_size = group_size
            self.w_q, self.scales, self.biases = w_q, scales, biases
            self.weight = weight
            self.weight_scale = to_mx(weight_scale, mx.float32)
            self.bias = None if bias is None else to_mx(bias, mx.float32)
            self.norm = bool(norm)
            self.norm_weight = None if norm_weight is None else to_mx(norm_weight, mx.float32)
            self.norm_eps = float(norm_eps)
            self.lora_A = [to_mx(a, mx.float32) for a, _, _ in adapters]
            self.lora_B = [to_mx(b, mx.float32) for _, b, _ in adapters]
            self.lora_scale = [mx.array(float(c), dtype=mx.float32) for _, _, c in adapters]
            self.freeze()

        @classmethod
        def from_inference(
            cls, layer: BitLinearInference, group_size: int = 64, dtype: Any = "float32", dense: bool = False
        ) -> MLXBitLinear:
            """Build from the repo's exported layer (a training ``BitLinear`` is exported first).

            Enabled LoRA adapters are carried over with their current strength; disabled ones are left out.
            """
            layer = _as_inference_layer(layer)
            _check_supported(layer)
            dt = _mx_dtype(dtype)
            n, k = layer.out_features, layer.in_features
            if dense:
                codes = unpack_ternary(layer.weight_packed.cpu(), n)
                weights: dict[str, Any] = {"weight": to_mx(codes).astype(dt)}
            else:
                w_q, scales, biases = from_packed_repo(layer.weight_packed, n, 1.0, group_size=group_size, dtype=dt)
                weights = {"w_q": w_q, "scales": scales, "biases": biases, "group_size": group_size}
            norm = layer.norm
            return cls(
                k,
                n,
                weight_scale=to_mx(layer.weight_scale.float()),
                bias=None if layer.bias is None else to_mx(layer.bias.float()),
                norm=norm is not None,
                norm_weight=None if norm is None or norm.weight is None else to_mx(norm.weight.float()),
                norm_eps=norm.eps if norm is not None else 1e-6,
                dtype=dt,
                adapters=_adapters(layer),
                **weights,
            )

        def quantize_input(self, x: mx.array) -> tuple[mx.array, mx.array]:
            """``(codes, scale)`` as in ``quantize_activations``: float32 codes in [-128, 127], ``x ~= codes / scale``.

            ``mx.round`` rounds half to even, like ``torch.round``.
            """
            x = x.astype(mx.float32)
            if self.norm:
                x = mx.fast.rms_norm(x, self.norm_weight, self.norm_eps)  # x * rsqrt(mean(x^2) + eps) * weight
            # 127 * (1 / absmax), not 127 / absmax: torch evaluates ``127 / tensor`` that way
            scale = mx.reciprocal(mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5)) * 127.0
            return mx.clip(mx.round(x * scale), -128, 127), scale

        def __call__(self, x: mx.array) -> mx.array:
            out_dtype = x.dtype
            codes, x_scale = self.quantize_input(x)
            half = self.compute_dtype != mx.float32
            a = (codes * _HALF_PRESCALE if half else codes).astype(self.compute_dtype)
            if self.packed:
                acc = mx.quantized_matmul(
                    a, self.w_q, self.scales, self.biases, transpose=True, group_size=self.group_size, bits=2
                )
            else:
                acc = a @ self.weight.T
            acc = acc.astype(mx.float32)
            if half:
                acc = acc * (1.0 / _HALF_PRESCALE)  # exact: back to the integer sums
            y = acc / x_scale * self.weight_scale  # same operation order as BitLinearInference
            if self.bias is not None:
                y = y + self.bias
            if self.lora_A:
                xf = x.astype(mx.float32)
                delta = None
                for a, b, c in zip(self.lora_A, self.lora_B, self.lora_scale):
                    d = ((xf @ a.T) @ b.T) * c
                    delta = d if delta is None else delta + d
                y = y + delta
            return y.astype(out_dtype)

        def _extra_repr(self) -> str:
            kind = f"packed 2-bit, group_size={self.group_size}" if self.packed else "dense"
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"norm={self.norm}, {kind}, dtype={self.compute_dtype}, adapters={len(self.lora_A)}"
            )

    class MLXBlock(mnn.Module):
        """MLX version of ``bitnet_vision.models.vit.Block`` at inference (no DropPath). Input (B, N, C).

        Build it with :func:`mlx_block_from_vit_block`.
        """

        def __init__(
            self,
            *,
            num_heads: int,
            qkv: MLXBitLinear,
            proj: MLXBitLinear,
            fc1: MLXBitLinear,
            fc2: MLXBitLinear,
            mlp_act: str = "gelu",
            norm1: tuple[mx.array | None, mx.array | None, float] | None = None,
            norm2: tuple[mx.array | None, mx.array | None, float] | None = None,
            ls1: mx.array | None = None,
            ls2: mx.array | None = None,
        ) -> None:
            super().__init__()
            if mlp_act not in ("gelu", "relu2"):
                raise ValueError(f"mlp_act must be 'gelu' or 'relu2', got {mlp_act!r}")
            self.num_heads = int(num_heads)
            self.qkv, self.proj, self.fc1, self.fc2 = qkv, proj, fc1, fc2
            self.mlp_act = mlp_act
            self.norm1_weight, self.norm1_bias, self.norm1_eps = norm1 if norm1 is not None else (None, None, None)
            self.norm2_weight, self.norm2_bias, self.norm2_eps = norm2 if norm2 is not None else (None, None, None)
            self.ls1, self.ls2 = ls1, ls2
            self.freeze()

        def _attention(self, x: mx.array) -> mx.array:
            b, n, c = x.shape
            hd = c // self.num_heads
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, hd).transpose(2, 0, 3, 1, 4)
            out = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=1.0 / math.sqrt(hd))
            return self.proj(out.transpose(0, 2, 1, 3).reshape(b, n, c))

        def _mlp(self, x: mx.array) -> mx.array:
            x = self.fc1(x)
            x = mnn.gelu(x) if self.mlp_act == "gelu" else mx.square(mx.maximum(x, 0))  # gelu: erf form
            return self.fc2(x)

        def __call__(self, x: mx.array) -> mx.array:
            h = x
            if self.norm1_eps is not None:
                h = mx.fast.layer_norm(h, self.norm1_weight, self.norm1_bias, self.norm1_eps)
            h = self._attention(h)
            x = x + (h * self.ls1 if self.ls1 is not None else h)
            h = x
            if self.norm2_eps is not None:
                h = mx.fast.layer_norm(h, self.norm2_weight, self.norm2_bias, self.norm2_eps)
            h = self._mlp(h)
            return x + (h * self.ls2 if self.ls2 is not None else h)

    for cls in (MLXBitLinear, MLXBlock):
        cls.__qualname__ = cls.__name__
    return {"MLXBitLinear": MLXBitLinear, "MLXBlock": MLXBlock}


def __getattr__(name: str) -> Any:
    if name in _LAZY_CLASSES:
        return _classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_CLASSES))


def _layer_norm_params(norm: nn.Module, dtype: Any) -> tuple[mx.array | None, mx.array | None, float] | None:
    if isinstance(norm, nn.Identity):
        return None
    if not isinstance(norm, nn.LayerNorm):
        raise NotImplementedError(f"unsupported block norm {type(norm).__name__}")
    if len(norm.normalized_shape) != 1:
        raise NotImplementedError("only LayerNorm over the last dimension is supported")
    weight = None if norm.weight is None else to_mx(norm.weight.float()).astype(dtype)
    bias = None if norm.bias is None else to_mx(norm.bias.float()).astype(dtype)
    return weight, bias, float(norm.eps)


def _layer_scale(ls: nn.Module, dtype: Any) -> mx.array | None:
    if isinstance(ls, nn.Identity):
        return None
    return to_mx(ls.gamma.detach().float()).astype(dtype)


def mlx_block_from_vit_block(
    block: nn.Module, group_size: int = 64, dtype: Any = "float32", dense: bool = False
) -> Any:
    """MLX version of one ``bitnet_vision.models.vit.Block``: an :class:`MLXBlock` to call on (B, N, C) arrays.

    Pass the block after ``convert_for_inference`` (training BitLinear layers are exported on the fly). Each
    linear layer becomes an :class:`MLXBitLinear`, with its enabled LoRA adapters; LayerNorm, softmax
    attention and GELU run in ``dtype``.
    """
    dt = _mx_dtype(dtype)
    linear = functools.partial(_classes()["MLXBitLinear"].from_inference, group_size=group_size, dtype=dt, dense=dense)
    attn, mlp = block.attn, block.mlp
    return _classes()["MLXBlock"](
        num_heads=attn.num_heads,
        qkv=linear(attn.qkv),
        proj=linear(attn.proj),
        fc1=linear(mlp.fc1),
        fc2=linear(mlp.fc2),
        mlp_act=mlp.act,
        norm1=_layer_norm_params(block.norm1, dt),
        norm2=_layer_norm_params(block.norm2, dt),
        ls1=_layer_scale(block.ls1, dt),
        ls2=_layer_scale(block.ls2, dt),
    )
