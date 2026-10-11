"""Fused Metal kernels for BitLinear inference on Apple GPUs (PyTorch's MPS backend).

On M1 to M4-class GPUs the ternary matmul is compute-bound and the GPU has no int8 matrix or sparsity hardware
to exploit, so the win is fusing the many small elementwise passes around it. Two kernels replace them:

* ``fused_front_end``: RMSNorm, per-token absmax and rounding to int8 codes in one pass over each token.
* ``fused_epilogue``: ``acc / x_scale * weight_scale + bias`` in one pass over the output.

The matmul in between stays ``F.linear`` in float32 on the integer-valued codes, which is exact (sums stay far
below 2^24) and as fast as float16 on these GPUs; float16 would overflow, since 127 * 1536 > 65504.

Kernel layout:

* Front end: one threadgroup per token row, a multiple of 32 threads with about 4 elements per thread, at most
  1024. Thread ``t`` handles elements ``t, t + T, t + 2T, ...`` of its row (coalesced reads; no remainder
  case). Each reduction (sum of squares, then absmax) is ``simd_sum`` / ``simd_max`` within each simdgroup,
  one slot per simdgroup in threadgroup memory, a barrier, and a serial pass over the <= 32 slots that every
  thread does in the same order, so all threads see the same value. The row is read again from memory for
  each pass instead of being kept in threadgroup memory, so any K works.
* Epilogue: one thread per output element on a 2D grid (N, M).

Off MPS, for other dtypes or bit widths, and when autograd has to track an input, the same functions run a
reference built from the repo's own ``RMSNorm`` and ``quantize_activations``, so on CPU
``FusedBitLinearInference`` returns exactly what ``BitLinearInference`` returns. Compared with that reference,
the Metal front end differs only in the order of the RMSNorm sum of squares: scales can move by a few ulps and,
rarely, a code by 1 (tests/test_apple_mps.py states the tolerance). Without the norm it uses the same correctly
rounded float32 steps and should agree exactly. That assumes PyTorch's default (safe) Metal math mode, in which
division and square root are correctly rounded (MSL specification, Table 8.1). With ``PYTORCH_MPS_FAST_MATH=1``
division is approximate and the compiler may assume there are no NaN or inf values (results for such tokens are
undefined).
"""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..export import BitLinearInference, convert_for_inference, unpack_ternary
from ..hadamard import autocast_disabled
from ..layers import BitLinear, RMSNorm
from ..quant import QUANT_EPS, quantize_activations

__all__ = [
    "METAL_SOURCE",
    "metal_available",
    "fused_front_end",
    "fused_epilogue",
    "FusedBitLinearInference",
    "convert_to_fused",
]

# Must stay ASCII: PyTorch hands it to Metal as an NSASCIIStringEncoding string.
METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Threadgroup-wide sum or max of one value per thread; every thread returns the same result. `partials` needs
// one slot per simdgroup. The second barrier lets the caller reuse `partials` right away.
inline float bitnet_group_sum(float v, threadgroup float* partials, uint lane, uint sg, uint n_sg) {
  v = simd_sum(v);
  if (lane == 0) {
    partials[sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < n_sg; ++i) {
    total += partials[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// max that propagates NaN like torch.amax (metal::max returns the other operand)
inline float bitnet_nan_max(float a, float b) {
  return isnan(a) ? a : (isnan(b) ? b : max(a, b));
}

inline float bitnet_group_max(float v, threadgroup float* partials, uint lane, uint sg, uint n_sg) {
  v = simd_any(isnan(v)) ? numeric_limits<float>::quiet_NaN() : simd_max(v);
  if (lane == 0) {
    partials[sg] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < n_sg; ++i) {
    total = bitnet_nan_max(total, partials[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// Element i of the RMSNorm output. Like the repo's RMSNorm it computes in float32, multiplies by the weight
// after the normalization, and rounds the result to the input type T.
template <typename T>
inline float bitnet_load_normed(device const T* x, device const float* norm_w, long i, float inv_rms,
                                bool use_norm) {
  float v = float(x[i]);
  if (use_norm) {
    v = float(T((v * inv_rms) * norm_w[i]));
  }
  return v;
}

// One token row: optional RMSNorm, then 8-bit absmax quantization exactly as quant.quantize_activations:
// scale = (1 / max(absmax, qeps)) * 127 (torch evaluates 127 / t as t.reciprocal() * 127), codes =
// clamp(rint(x * scale), -128, 127) with rint rounding half to even like torch.round.
template <typename T>
inline void bitnet_row_front_end(device const T* x, device const float* norm_w, device float* codes,
                                 device float* scales, long K, bool use_norm, float eps, float qeps,
                                 threadgroup float* partials, uint row, uint tid, uint tg_size, uint lane, uint sg,
                                 uint n_sg) {
  x += long(row) * K;
  codes += long(row) * K;
  float inv_rms = 1.0f;
  if (use_norm) {
    float ss = 0.0f;
    for (long i = tid; i < K; i += tg_size) {
      const float v = float(x[i]);
      ss += v * v;
    }
    ss = bitnet_group_sum(ss, partials, lane, sg, n_sg);
    // torch: mean = sum / K, then rsqrt, which its CPU kernels compute as 1 / sqrt (two roundings); a single
    // correctly rounded rsqrt can differ from that in the last bit
    inv_rms = 1.0f / precise::sqrt(ss / float(K) + eps);
  }
  float amax = 0.0f;
  for (long i = tid; i < K; i += tg_size) {
    amax = bitnet_nan_max(amax, fabs(bitnet_load_normed(x, norm_w, i, inv_rms, use_norm)));
  }
  amax = bitnet_group_max(amax, partials, lane, sg, n_sg);
  const float denom = isnan(amax) ? amax : max(amax, qeps);
  const float scale = (1.0f / denom) * 127.0f;
  for (long i = tid; i < K; i += tg_size) {
    const float q = bitnet_load_normed(x, norm_w, i, inv_rms, use_norm) * scale;
    codes[i] = isnan(q) ? q : clamp(rint(q), -128.0f, 127.0f);
  }
  if (tid == 0) {
    scales[row] = scale;
  }
}

kernel void bitnet_front_end_float(
    device const float* x [[buffer(0)]],
    device const float* norm_w [[buffer(1)]],
    device float* codes [[buffer(2)]],
    device float* scales [[buffer(3)]],
    constant long& K [[buffer(4)]],
    constant long& use_norm [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant float& qeps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]]) {
  threadgroup float partials[32];
  bitnet_row_front_end(x, norm_w, codes, scales, K, use_norm != 0, eps, qeps, partials, row, tid, tg_size, lane, sg,
                       n_sg);
}

kernel void bitnet_front_end_half(
    device const half* x [[buffer(0)]],
    device const float* norm_w [[buffer(1)]],
    device float* codes [[buffer(2)]],
    device float* scales [[buffer(3)]],
    constant long& K [[buffer(4)]],
    constant long& use_norm [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant float& qeps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]]) {
  threadgroup float partials[32];
  bitnet_row_front_end(x, norm_w, codes, scales, K, use_norm != 0, eps, qeps, partials, row, tid, tg_size, lane, sg,
                       n_sg);
}

// out[m, n] = acc[m, n] / x_scale[m] * w_scale[n * w_step] + bias[n], in the order BitLinearInference uses.
kernel void bitnet_epilogue(
    device const float* acc [[buffer(0)]],
    device const float* x_scale [[buffer(1)]],
    device const float* w_scale [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant long& M [[buffer(5)]],
    constant long& N [[buffer(6)]],
    constant long& w_step [[buffer(7)]],
    constant long& has_bias [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]) {
  const long n = pos.x;
  const long m = pos.y;
  if (n >= N || m >= M) {
    return;
  }
  const long i = m * N + n;
  float y = acc[i] / x_scale[m];
  y = y * w_scale[n * w_step];
  if (has_bias != 0) {
    y = y + bias[n];
  }
  out[i] = y;
}
"""

_FRONT_END_KERNELS = {torch.float32: "bitnet_front_end_float", torch.float16: "bitnet_front_end_half"}
_ELEMENTS_PER_THREAD = 4  # front end: about this many elements of a row per thread
_EPILOGUE_GROUP = 256
_SHARED_SLOTS = 32  # threadgroup slots in the kernels: one per simdgroup

_library: Any = None
_kernels: dict[str, Any] = {}


def metal_available() -> bool:
    """True if PyTorch can run the Metal kernels: an MPS device and ``torch.mps.compile_shader`` (PyTorch 2.7+)."""
    mps = getattr(torch, "mps", None)
    return bool(torch.backends.mps.is_available()) and hasattr(mps, "compile_shader")


def _kernel(name: str) -> Any:
    """A kernel from :data:`METAL_SOURCE`, compiled once per process."""
    global _library
    if name not in _kernels:
        if _library is None:
            try:
                _library = torch.mps.compile_shader(METAL_SOURCE)
            except Exception as e:  # compile errors arrive as SyntaxError; no MPS as RuntimeError or AttributeError
                raise RuntimeError(f"could not compile the bitnet_vision Metal kernels: {type(e).__name__}: {e}") from e
        _kernels[name] = getattr(_library, name)
    return _kernels[name]


def _group_size(kernel: Any, wanted: int) -> int:
    """Threads per threadgroup: ``wanted`` rounded up to whole simdgroups, within the pipeline's limits."""
    width = int(getattr(kernel, "thread_execution_width", 32))
    limit = min(int(getattr(kernel, "max_threads_per_threadgroup", 1024)), _SHARED_SLOTS * width)
    limit = max(width, limit // width * width)
    return min(limit, max(width, -(-wanted // width) * width))


def _use_metal(*tensors: Tensor | None) -> bool:
    """Metal kernels for MPS tensors, unless autograd has to record the op (the kernels have no backward)."""
    if tensors[0].device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
        return False
    return not (torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors))


class _NormParams(NamedTuple):
    weight: Tensor
    eps: float


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    # Runs RMSNorm.forward itself on the given parameters, so the reference cannot drift from the repo's norm.
    return RMSNorm.forward(_NormParams(weight, eps), x)


def _front_end_reference(x: Tensor, norm_weight: Tensor | None, eps: float, bits: int) -> tuple[Tensor, Tensor]:
    if norm_weight is not None:
        x = _rms_norm(x, norm_weight, eps)
    return quantize_activations(x, bits=bits, method="absmax")


def fused_front_end(x: Tensor, norm_weight: Tensor | None, eps: float, bits: int = 8) -> tuple[Tensor, Tensor]:
    """RMSNorm (if ``norm_weight`` is given) and per-token absmax quantization of ``x`` (shape ``[..., K]``).

    Returns ``(codes, scale)`` like :func:`bitnet_vision.quant.quantize_activations`: float32 codes in
    [-128, 127] of shape ``[..., K]`` and float32 scales of shape ``[..., 1]``, with ``x ~= codes / scale``.
    The norm is exactly the repo's RMSNorm, including its rounding back to the input dtype. On MPS, 8-bit
    float32 or float16 input runs as one Metal kernel; everything else runs the reference.
    """
    k = x.shape[-1]
    if norm_weight is not None and norm_weight.shape != (k,):
        raise ValueError(f"norm_weight must have shape ({k},), got {tuple(norm_weight.shape)}")
    if bits != 8 or x.dtype not in _FRONT_END_KERNELS or x.numel() == 0 or not _use_metal(x, norm_weight):
        return _front_end_reference(x, norm_weight, eps, bits)
    x2 = x.reshape(-1, k).contiguous()
    m = x2.shape[0]
    codes = torch.empty((m, k), dtype=torch.float32, device=x.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    # Without a norm the kernel never reads the weight buffer; codes holds its place (float32, at least K values).
    w = codes if norm_weight is None else norm_weight.to(device=x.device, dtype=torch.float32).contiguous()
    kernel = _kernel(_FRONT_END_KERNELS[x.dtype])
    group = _group_size(kernel, -(-k // _ELEMENTS_PER_THREAD))
    kernel(x2, w, codes, scale, k, int(norm_weight is not None), float(eps), float(QUANT_EPS),
           threads=m * group, group_size=group)
    return codes.reshape(x.shape), scale.reshape(*x.shape[:-1], 1)


def _epilogue_reference(acc: Tensor, x_scale: Tensor, weight_scale: Tensor, bias: Tensor | None) -> Tensor:
    y = acc.float() / x_scale * weight_scale.float()  # same expression as BitLinearInference.forward
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y


def fused_epilogue(acc: Tensor, x_scale: Tensor, weight_scale: Tensor, bias: Tensor | None) -> Tensor:
    """``acc / x_scale * weight_scale + bias`` in float32, in BitLinearInference's order of operations.

    ``acc`` is ``[..., N]``, ``x_scale`` ``[..., 1]``, ``weight_scale`` ``[1]`` or ``[N]`` and ``bias`` ``[N]``
    or None. One Metal kernel on MPS (float32 ``acc``); the reference everywhere else.
    """
    n = acc.shape[-1]
    fits = (
        acc.dtype == torch.float32
        and x_scale.shape == (*acc.shape[:-1], 1)
        and weight_scale.numel() in (1, n)
        and (bias is None or bias.numel() == n)
    )
    if not fits or acc.numel() == 0 or not _use_metal(acc, x_scale, weight_scale, bias):
        return _epilogue_reference(acc, x_scale, weight_scale, bias)
    a = acc.reshape(-1, n).contiguous()
    m = a.shape[0]
    xs = x_scale.reshape(m).to(device=acc.device, dtype=torch.float32).contiguous()
    ws = weight_scale.reshape(-1).to(device=acc.device, dtype=torch.float32).contiguous()
    b = ws if bias is None else bias.reshape(n).to(device=acc.device, dtype=torch.float32).contiguous()
    out = torch.empty_like(a)
    kernel = _kernel("bitnet_epilogue")
    group = min(n, _EPILOGUE_GROUP, int(getattr(kernel, "max_threads_per_threadgroup", 1024)))
    kernel(a, xs, ws, b, out, m, n, int(ws.numel() > 1), int(bias is not None), threads=[n, m], group_size=[group, 1])
    return out.reshape(acc.shape)


class FusedBitLinearInference(BitLinearInference):
    """BitLinearInference with a fused front end and epilogue (Metal kernels on Apple GPUs).

    Build it with :meth:`from_inference`. It keeps the packed weights, so its state dict is the same as
    BitLinearInference's, and also keeps the unpacked codes as a float32 ``[N, K]`` matrix on the weights'
    device (4 bytes per weight). The forward pass is front end, float32 ``F.linear`` with autocast off, then
    epilogue. Layers that are not 8-bit absmax (4-bit, or full-precision activations) and float64 inputs use
    BitLinearInference's forward. Run it under ``torch.no_grad()`` or ``torch.inference_mode()``: when
    autograd has to record an input, the reference ops run instead of the kernels.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._codes_f32: Tensor | None = None

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._codes_f32 = None  # new weights: forget the unpacked copy
        super()._load_from_state_dict(*args, **kwargs)

    @classmethod
    @torch.no_grad()
    def from_inference(cls, layer: BitLinearInference) -> FusedBitLinearInference:
        """A fused layer that shares ``layer``'s buffers, norm and Hadamard modules."""
        new = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            act_bits=layer.act_bits,
            act_method=layer.act_method,
            weight_granularity=layer.weight_granularity,
            input_norm=False,
            hadamard=False,
            int_matmul=layer.int_matmul,
            keep_unpacked=layer.keep_unpacked,
        )
        new.weight_packed = layer.weight_packed
        new.weight_scale = layer.weight_scale
        if layer.bias is not None:
            new.bias = layer.bias
        new.norm = layer.norm
        new.hadamard = layer.hadamard
        new.train(layer.training)
        if new.weight_packed.device.type != "meta":
            new._float_codes()
        return new

    def _float_codes(self) -> Tensor:
        """Ternary codes as float32 ``[N, K]`` on the weights' device, unpacked once."""
        device = self.weight_packed.device
        if self._codes_f32 is None or self._codes_f32.device != device:
            # a normal tensor even when first built under inference_mode, so later autograd calls can save it
            with torch.inference_mode(False), torch.no_grad():
                codes = unpack_ternary(self.weight_packed.cpu(), self.out_features, dtype=torch.float32)
                self._codes_f32 = codes.to(device)
        return self._codes_f32

    def forward(self, x: Tensor) -> Tensor:
        if self.act_bits != 8 or self.act_method != "absmax" or x.dtype == torch.float64:
            return super().forward(x)
        norm = self.norm
        if self.hadamard is None and type(norm) is RMSNorm and norm.weight is not None:
            codes, x_scale = fused_front_end(x, norm.weight, norm.eps)
        else:
            h = x if norm is None else norm(x)
            if self.hadamard is not None:
                h = self.hadamard(h)
            codes, x_scale = fused_front_end(h, None, 0.0)
        with autocast_disabled(codes.device.type):
            acc = F.linear(codes, self._float_codes())
        return fused_epilogue(acc, x_scale, self.weight_scale, self.bias).to(x.dtype)


def convert_to_fused(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Replace every BitLinearInference in ``model`` with a :class:`FusedBitLinearInference`.

    Use it on the output of :func:`bitnet_vision.export.convert_for_inference`; training BitLinear layers
    still in the model are exported first. ``model`` may itself be a single layer. A layer shared by several
    parents stays shared.
    """
    if isinstance(model, FusedBitLinearInference):
        return model
    if isinstance(model, BitLinear):
        return FusedBitLinearInference.from_inference(convert_for_inference(model))
    if isinstance(model, BitLinearInference):
        return FusedBitLinearInference.from_inference(model if inplace else copy.deepcopy(model))
    if not inplace:
        model = copy.deepcopy(model)
    if any(isinstance(m, BitLinear) for m in model.modules()):
        convert_for_inference(model, inplace=True)
    fused: dict[int, FusedBitLinearInference] = {}
    for module in list(model.modules()):
        for name, child in list(module._modules.items()):  # includes repeated (shared) children
            if isinstance(child, BitLinearInference) and not isinstance(child, FusedBitLinearInference):
                if id(child) not in fused:
                    fused[id(child)] = FusedBitLinearInference.from_inference(child)
                setattr(module, name, fused[id(child)])
    return model
