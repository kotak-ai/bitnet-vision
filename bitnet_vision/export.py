"""Inference export: 2-bit packed ternary weights, int8 activations, integer matmul.

Training-mode BitLinear keeps full-precision latent weights and only *simulates* quantization, so it is no
smaller or faster than nn.Linear. ``convert_for_inference`` swaps every BitLinear for a
``BitLinearInference`` whose state dict holds:

* ``weight_packed``: uint8 of shape ``(ceil(out_features / 4), in_features)``, 4 ternary weights per byte,
  each stored as ``code + 1`` in {0, 1, 2}. The output rows are split into 4 contiguous blocks of
  ``ceil(out_features / 4)`` rows and block ``i`` sits in bits ``2i`` and ``2i + 1`` (lowest bits first).
  This is the bit layout of Hugging Face transformers' BitNet integration and of the
  microsoft/bitnet-b1.58-2B-4T checkpoint. When ``out_features`` is not a multiple of 4 the last block is
  padded with code 0 (stored as 1) and the padding is dropped on unpack.
* ``weight_scale``: float32 multiplier ``mean|W|`` (shape ``(1,)``, or ``(out_features,)`` for per-channel
  scales), so ``W ~= codes * weight_scale``. Same convention as the 2B-4T checkpoint and bitnet.cpp.
* ``bias``: added after rescaling (Hugging Face's AutoBitLinear instead scales its bias), with Tequila's
  dead-zone term folded in when it was used.

The bit layout matches Hugging Face, but the module names and the bias convention do not, so these files
are not drop-in loadable by transformers. The conventions are recorded in ``PACKED_METADATA``.

At inference time each layer unpacks its weights once to int8 (1 byte per weight in memory; 2 bits on
disk), quantizes activations per token to int8, and multiplies in integer arithmetic. ``torch._int_mm`` is
used only where it is actually faster: on CUDA when the shapes allow it, and on CPU only if a one-time
timing check says so (from PyTorch 2.6, CPUs without AVX512-VNNI silently fall back to a very slow loop).
Otherwise an exact float matmul on the integer values is used. Both paths give identical results.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hadamard import HadamardTransform, autocast_disabled
from .layers import BitLinear, HBitLinear, RMSNorm
from .lora import LoRASettings, copy_lora, lora_forward
from .quant import quantize_activations, quantize_weights

__all__ = [
    "PACKED_METADATA",
    "pack_ternary",
    "unpack_ternary",
    "BitLinearInference",
    "convert_for_inference",
    "packed_state_dict_bytes",
]

VALUES_PER_BYTE = 4
PACKED_METADATA = {
    "weight_layout": "uint8 [ceil(out/4), in]; code+1 in 2 bits; 4 row blocks, block i in bits 2i..2i+1",
    "weight_scale": "multiplier: W ~= codes * weight_scale (weight_scale = mean|W|)",
    "bias": "added after rescaling",
    "activations": "per-token symmetric int8 (absmax) or absmean for < 8 bits, computed at runtime",
}


def pack_ternary(codes: Tensor) -> Tensor:
    """Pack ternary codes {-1, 0, 1} of shape (rows, ...) into uint8 of shape (ceil(rows / 4), ...)."""
    if codes.dim() == 0:
        raise ValueError("cannot pack a scalar")
    if codes.is_floating_point() and not torch.equal(codes, codes.round()):  # also catches NaN
        raise ValueError("pack_ternary expects integer values in {-1, 0, 1}")
    if codes.numel() and (codes.min() < -1 or codes.max() > 1):
        raise ValueError("pack_ternary expects values in {-1, 0, 1}")
    rows = codes.shape[0]
    block = -(-rows // VALUES_PER_BYTE)
    u = (codes.to(torch.int16) + 1).to(torch.uint8)
    pad = block * VALUES_PER_BYTE - rows
    if pad:
        u = torch.cat([u, u.new_ones((pad, *u.shape[1:]))], dim=0)  # pad with code 0 (stored as 1)
    packed = torch.zeros((block, *u.shape[1:]), dtype=torch.uint8, device=u.device)
    for i in range(VALUES_PER_BYTE):
        packed |= u[i * block : (i + 1) * block] << (2 * i)
    return packed


def unpack_ternary(packed: Tensor, rows: int, dtype: torch.dtype = torch.int8) -> Tensor:
    """Inverse of :func:`pack_ternary`: returns ``rows`` rows of codes in {-1, 0, 1} with ``dtype``."""
    if dtype in (torch.uint8, torch.bool):
        raise ValueError("unpack_ternary needs a signed or floating dtype to hold -1")
    if not 0 <= rows <= VALUES_PER_BYTE * packed.shape[0]:
        raise ValueError(f"rows={rows} does not fit in {packed.shape[0]} packed rows")
    parts = [((packed >> (2 * i)) & 3) for i in range(VALUES_PER_BYTE)]
    u = torch.cat(parts, dim=0)[:rows]
    return u.to(torch.int16).sub(1).to(dtype)


# One timing decision per (K, N, threads) for CPU int8 matmuls; True means torch._int_mm is faster.
_CPU_INT_MM_FAST: dict[tuple[int, int, int], bool] = {}


def _cpu_int_mm_is_fast(k: int, n: int) -> bool:
    key = (k, n, torch.get_num_threads())
    if key not in _CPU_INT_MM_FAST:
        fast = False
        try:
            a = torch.randint(-127, 128, (256, k), dtype=torch.int8)
            b = torch.randint(-1, 2, (k, n), dtype=torch.int8)
            af, bf = a.float(), b.float()

            def best(fn):
                fn()
                times = []
                for _ in range(3):
                    t = time.perf_counter()
                    fn()
                    times.append(time.perf_counter() - t)
                return min(times)

            fast = best(lambda: torch._int_mm(a, b)) < best(lambda: af @ bf)
        except (RuntimeError, AttributeError, NotImplementedError):
            fast = False  # no CPU int8 matmul in this PyTorch build (e.g. before 2.4)
        _CPU_INT_MM_FAST[key] = fast
    return _CPU_INT_MM_FAST[key]


class BitLinearInference(nn.Module):
    """Inference-only BitLinear with packed 2-bit weights. Build it with :meth:`from_bitlinear`.

    Args:
        int_matmul: "auto" (default: use ``torch._int_mm`` where it is faster, see the module docstring),
            "always", or "never" (exact float matmul on integer values).
        keep_unpacked: keep an int8 copy of the weights in memory after the first call (default; 1 byte per
            weight). With False the weights are unpacked on every call, which saves memory and costs time.
            The copy is dropped automatically when new weights are loaded.

    LoRA adapters run as in BitLinear: full precision, on the raw input, added after the bias.
    """

    lora_settings: dict[str, LoRASettings] | None = None  # adapters, see bitnet_vision.lora

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        act_bits: int | None = 8,
        act_method: str = "absmax",
        weight_granularity: str = "tensor",
        input_norm: bool = True,
        norm_eps: float = 1e-6,
        hadamard: bool = False,
        hadamard_random_signs: bool = True,
        int_matmul: str = "auto",
        keep_unpacked: bool = True,
    ) -> None:
        super().__init__()
        if int_matmul not in ("auto", "always", "never"):
            raise ValueError("int_matmul must be 'auto', 'always' or 'never'")
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.act_method = act_method
        self.weight_granularity = weight_granularity
        self.int_matmul = int_matmul
        self.keep_unpacked = keep_unpacked
        block = -(-out_features // VALUES_PER_BYTE)
        self.register_buffer("weight_packed", torch.full((block, in_features), 0b01010101, dtype=torch.uint8))
        n_scales = out_features if weight_granularity == "channel" else 1
        self.register_buffer("weight_scale", torch.ones(n_scales, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None
        self.norm = RMSNorm(in_features, eps=norm_eps) if input_norm else None
        self.hadamard = HadamardTransform(in_features, random_signs=hadamard_random_signs) if hadamard else None
        self._w_cache: dict[tuple, Tensor] = {}

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._w_cache = {}  # new weights: forget the unpacked copy
        super()._load_from_state_dict(*args, **kwargs)

    @classmethod
    @torch.no_grad()
    def from_bitlinear(cls, layer: BitLinear, **kwargs: Any) -> BitLinearInference:
        if layer.quant_strength != 1.0:
            raise ValueError("only fully quantized layers (quant_strength == 1) can be exported")
        is_h = isinstance(layer, HBitLinear)
        new = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None or bool(layer.deadzone_lambda),
            act_bits=layer.act_bits,
            act_method=layer.act_method,
            weight_granularity=layer.weight_granularity,
            input_norm=layer.norm is not None,
            norm_eps=layer.norm.eps if layer.norm is not None else 1e-6,
            hadamard=is_h,
            hadamard_random_signs=layer.hadamard.random_signs if is_h else True,
            **kwargs,
        )
        codes, scale = quantize_weights(layer.weight, granularity=layer.weight_granularity)
        new.weight_packed = pack_ternary(codes.to(torch.int8))
        # float32 scales, except that a float64 model keeps float64 so exports can be checked exactly
        scale_dtype = torch.float64 if layer.weight.dtype == torch.float64 else torch.float32
        new.weight_scale = (1.0 / scale).reshape(-1).to(scale_dtype).clone()  # multiplier: mean|W|
        if new.bias is not None:
            bias = torch.zeros(layer.out_features, dtype=layer.weight.dtype, device=layer.weight.device)
            if layer.bias is not None:
                bias = bias + layer.bias.detach()
            if layer.deadzone_lambda:
                bias = bias + layer.deadzone_lambda * layer.deadzone_rowsum().detach()
            new.bias = bias.clone()
        if layer.norm is not None:
            new.norm = copy.deepcopy(layer.norm)
            new.norm.weight.requires_grad_(False)
        copy_lora(layer, new)
        return new.to(layer.weight.device)

    def _codes(self, dtype: torch.dtype, transposed: bool = False) -> Tensor:
        """Unpacked codes as ``dtype``. Only int8 copies are cached (one per orientation)."""
        key = (transposed, self.weight_packed.device)
        w = self._w_cache.get(key)
        if w is None:
            w = unpack_ternary(self.weight_packed, self.out_features, dtype=torch.int8)
            if transposed:
                w = w.t().contiguous()  # (K, N) layout for torch._int_mm
            if self.keep_unpacked:
                self._w_cache[key] = w
        return w if dtype == torch.int8 else w.to(dtype)

    def _use_int_mm(self, m: int) -> bool:
        if self.int_matmul == "never" or self.act_bits is None or self.act_bits > 8:
            return False
        if self.int_matmul == "always":
            return True
        if self.weight_packed.is_cuda:  # cuBLASLt int8: M > 16 and K, N multiples of 8
            return m > 16 and self.in_features % 8 == 0 and self.out_features % 8 == 0
        return self.weight_packed.device.type == "cpu" and _cpu_int_mm_is_fast(self.in_features, self.out_features)

    def _int_matmul(self, x_codes: Tensor) -> Tensor:
        """Integer matmul of activation codes (any leading dims) with ternary codes, returned as float32."""
        lead = x_codes.shape[:-1]
        a = x_codes.reshape(-1, self.in_features)
        if self._use_int_mm(a.shape[0]):
            out = torch._int_mm(a.to(torch.int8), self._codes(torch.int8, transposed=True)).to(torch.float32)
        else:
            # Exact for integer values as long as in_features * 128 < 2^24, true for any ViT width.
            with autocast_disabled(a.device.type):
                out = F.linear(a.to(torch.float32), self._codes(torch.float32))
        return out.reshape(*lead, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        raw = x
        if self.norm is not None:
            x = self.norm(x)
        if self.hadamard is not None:
            x = self.hadamard(x)
        if self.act_bits is None:
            xf = x if x.dtype == torch.float64 else x.float()
            with autocast_disabled(xf.device.type):
                y = F.linear(xf, self._codes(xf.dtype)) * self.weight_scale.to(xf.dtype)
        else:
            codes, x_scale = quantize_activations(x, bits=self.act_bits, method=self.act_method)
            if codes.dtype == torch.float64:
                y = F.linear(codes, self._codes(torch.float64)) / x_scale * self.weight_scale.double()
            else:
                y = self._int_matmul(codes) / x_scale * self.weight_scale.float()
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return lora_forward(self, raw, y.to(x.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"act_bits={self.act_bits}, act_method={self.act_method}, weight_granularity={self.weight_granularity}, "
            f"hadamard={self.hadamard is not None}, int_matmul={self.int_matmul}"
        )


def convert_for_inference(model: nn.Module, inplace: bool = False, **kwargs: Any) -> nn.Module:
    """Replace every BitLinear / HBitLinear in ``model`` with a packed ``BitLinearInference``.

    ``model`` may itself be a BitLinear. Extra keyword arguments (``int_matmul``, ``keep_unpacked``) are
    passed to each new layer. A layer shared by several parents stays shared. LoRA adapters are copied over
    and stay in full precision.
    """
    if isinstance(model, BitLinear):
        return BitLinearInference.from_bitlinear(model, **kwargs).eval()
    if not inplace:
        model = copy.deepcopy(model)
    converted: dict[int, BitLinearInference] = {}
    for module in list(model.modules()):
        for name, child in list(module._modules.items()):  # includes repeated (shared) children
            if isinstance(child, BitLinear):
                if id(child) not in converted:
                    converted[id(child)] = BitLinearInference.from_bitlinear(child, **kwargs)
                setattr(module, name, converted[id(child)])
    model.eval()
    return model


def packed_state_dict_bytes(state_dict: dict[str, Any]) -> int:
    """Total tensor bytes in a state dict (a quick size measure that ignores pickle overhead)."""
    return sum(t.numel() * t.element_size() for t in state_dict.values() if torch.is_tensor(t))
