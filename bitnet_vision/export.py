"""Inference export: 2-bit packed ternary weights, int8 activations, integer matmul.

Training-mode BitLinear keeps full-precision latent weights and only *simulates* quantization, so it is no
smaller or faster than nn.Linear. ``convert_for_inference`` swaps every BitLinear for a
``BitLinearInference`` that stores:

* ``weight_packed``: uint8, 4 ternary weights per byte (2 bits each, value + 1 in {0, 1, 2}). The layout
  follows Hugging Face transformers' BitNet format: the output dimension is split into 4 contiguous
  blocks of ``ceil(out_features / 4)`` rows and block ``i`` lives in bits ``2i`` and ``2i + 1``. When
  ``out_features`` is not a multiple of 4, the last block is zero-padded and the padding is dropped on
  unpack.
* ``weight_scale``: float32, ``1 / mean|W|`` (shape ``(1,)``, or ``(out_features,)`` for per-channel scales)
  so that ``W ~= codes / weight_scale``. This matches HF's ``weight_scale`` convention.

Activations are quantized per token to int8 codes and multiplied with the ternary codes in integer
arithmetic (``torch._int_mm`` when the backend supports it, otherwise an exact float matmul on integer
values), then divided by both scales. With float64 inputs the result equals the training-mode layer's
output up to rounding; in float32 a few int8 roundings can flip, so compare accuracy, not bits.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hadamard import HadamardTransform
from .layers import BitLinear, HBitLinear, RMSNorm
from .quant import quantize_activations, quantize_weights

__all__ = [
    "pack_ternary",
    "unpack_ternary",
    "BitLinearInference",
    "convert_for_inference",
    "packed_state_dict_bytes",
]

VALUES_PER_BYTE = 4


def pack_ternary(codes: Tensor) -> Tensor:
    """Pack ternary codes {-1, 0, 1} of shape (rows, ...) into uint8 of shape (ceil(rows / 4), ...)."""
    if codes.dim() == 0:
        raise ValueError("cannot pack a scalar")
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
    parts = [((packed >> (2 * i)) & 3) for i in range(VALUES_PER_BYTE)]
    u = torch.cat(parts, dim=0)[:rows]
    return u.to(torch.int16).sub(1).to(dtype)


class BitLinearInference(nn.Module):
    """Inference-only BitLinear with packed 2-bit weights. Build it with :meth:`from_bitlinear`.

    Args:
        cache_unpacked: keep an unpacked int8 copy of the weights (4x smaller than float32 instead of 16x)
            to avoid unpacking on every call. A real deployment would use a kernel that reads packed
            weights directly (bitnet.cpp, T-MAC, or a Triton kernel); this module is the reference.
    """

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
        cache_unpacked: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.act_method = act_method
        self.weight_granularity = weight_granularity
        self.cache_unpacked = cache_unpacked
        block = -(-out_features // VALUES_PER_BYTE)
        self.register_buffer("weight_packed", torch.ones(block, in_features, dtype=torch.uint8) * 0b01010101)
        n_scales = out_features if weight_granularity == "channel" else 1
        self.register_buffer("weight_scale", torch.ones(n_scales, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None
        self.norm = RMSNorm(in_features, eps=norm_eps) if input_norm else None
        self.hadamard = HadamardTransform(in_features) if hadamard else None
        self._w_cache: Tensor | None = None
        self._int_mm_ok: bool | None = None

    @classmethod
    @torch.no_grad()
    def from_bitlinear(cls, layer: BitLinear, cache_unpacked: bool = False) -> BitLinearInference:
        if layer.quant_strength != 1.0:
            raise ValueError("only fully quantized layers (quant_strength == 1) can be exported")
        new = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            act_bits=layer.act_bits,
            act_method=layer.act_method,
            weight_granularity=layer.weight_granularity,
            input_norm=layer.norm is not None,
            norm_eps=layer.norm.eps if layer.norm is not None else 1e-6,
            hadamard=isinstance(layer, HBitLinear),
            cache_unpacked=cache_unpacked,
        )
        codes, scale = quantize_weights(layer.weight, granularity=layer.weight_granularity)
        new.weight_packed = pack_ternary(codes.to(torch.int8))
        # float32 scales, except that a float64 model keeps float64 so exports can be checked exactly
        scale_dtype = torch.float64 if layer.weight.dtype == torch.float64 else torch.float32
        new.weight_scale = scale.reshape(-1).to(scale_dtype).clone()
        if layer.bias is not None:
            new.bias = layer.bias.detach().clone()
        if layer.norm is not None:
            new.norm = copy.deepcopy(layer.norm)
            new.norm.weight.requires_grad_(False)
        return new.to(layer.weight.device)

    def _codes(self, dtype: torch.dtype) -> Tensor:
        cache = self._w_cache
        if cache is not None and cache.dtype == dtype and cache.device == self.weight_packed.device:
            return cache
        w = unpack_ternary(self.weight_packed, self.out_features, dtype=dtype)
        if self.cache_unpacked:
            self._w_cache = w
        return w

    def _int_matmul(self, x_codes: Tensor) -> Tensor:
        """Integer matmul of activation codes (any leading dims) with ternary codes, as float32."""
        lead = x_codes.shape[:-1]
        a = x_codes.reshape(-1, self.in_features)
        if self.act_bits is not None and self.act_bits <= 8 and self._int_mm_ok is not False and a.shape[0] > 16:
            try:
                out = torch._int_mm(a.to(torch.int8), self._codes(torch.int8).t().contiguous())
                self._int_mm_ok = True
                return out.to(torch.float32).reshape(*lead, self.out_features)
            except (RuntimeError, AttributeError, NotImplementedError):
                self._int_mm_ok = False  # backend has no int8 matmul for these shapes; use the float path
        # Exact for integer values as long as in_features * 128 < 2^24 (true for any ViT width).
        out = F.linear(a.to(torch.float32), self._codes(torch.float32))
        return out.reshape(*lead, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm is not None:
            x = self.norm(x)
        if self.hadamard is not None:
            x = self.hadamard(x)
        if self.act_bits is None:
            xf = x if x.dtype == torch.float64 else x.float()
            y = F.linear(xf, self._codes(xf.dtype)) / self.weight_scale.to(xf.dtype)
        else:
            codes, x_scale = quantize_activations(x, bits=self.act_bits, method=self.act_method)
            if codes.dtype == torch.float64:
                y = F.linear(codes, self._codes(torch.float64)) / x_scale / self.weight_scale.double()
            else:
                y = self._int_matmul(codes) / x_scale / self.weight_scale.float()
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"act_bits={self.act_bits}, act_method={self.act_method}, weight_granularity={self.weight_granularity}, "
            f"hadamard={self.hadamard is not None}"
        )


def convert_for_inference(model: nn.Module, inplace: bool = False, cache_unpacked: bool = False) -> nn.Module:
    """Replace every BitLinear / HBitLinear in ``model`` with a packed ``BitLinearInference``."""
    if not inplace:
        model = copy.deepcopy(model)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, BitLinear):
                setattr(module, name, BitLinearInference.from_bitlinear(child, cache_unpacked=cache_unpacked))
    model.eval()
    return model


def packed_state_dict_bytes(state_dict: dict[str, Any]) -> int:
    """Total tensor bytes in a state dict (a quick size measure that ignores pickle overhead)."""
    return sum(t.numel() * t.element_size() for t in state_dict.values() if torch.is_tensor(t))
