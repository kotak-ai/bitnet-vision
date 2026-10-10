"""Quantizers for BitNet b1.58-style quantization-aware training.

Every ``fake_quant_*`` function returns *dequantized* values with the same dtype as its input, so it can
be used inside a straight-through estimator (STE)::

    x_q = x + (fake_quant_activations(x) - x).detach()

Because the whole round/clip/rescale step sits inside ``detach()``, the scales act as constants during
backpropagation and the gradient passes through unchanged. This is exactly the structure of Microsoft's
reference code ("The Era of 1-bit LLMs: Training Tips, Code and FAQ", Fig. 3). The three earlier branches
of this repo each broke it in a different way (see docs/AUDIT.md), so keep it this way.

All scale and rounding math runs in float32 (float64 inputs stay float64): in float16, ``127 / absmax``
overflows for small activations and turns zero tokens into NaN.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "act_qrange",
    "quantize_activations",
    "fake_quant_activations",
    "quantize_weights",
    "fake_quant_weights",
    "ste",
]

# Smallest allowed scale denominator, as in the reference code (``clamp_(min=1e-5)``).
QUANT_EPS = 1e-5


def _compute_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def act_qrange(bits: int) -> tuple[int, int]:
    """Signed integer range for ``bits``-bit activations, e.g. (-128, 127) for 8 bits."""
    if not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError(f"activation bits must be an int in [2, 16], got {bits!r}")
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def quantize_activations(
    x: Tensor, bits: int = 8, method: str = "absmax", eps: float = QUANT_EPS
) -> tuple[Tensor, Tensor]:
    """Per-token symmetric quantization along the last dimension.

    Returns ``(codes, scale)`` with ``x ~= codes / scale``. ``codes`` holds integers in the signed
    ``bits``-bit range (stored as a float tensor in the compute dtype) and ``scale`` has shape
    ``x.shape[:-1] + (1,)``.

    method="absmax" (BitNet b1.58, 8-bit): ``scale = (2^(b-1) - 1) / max|x|``. With 127 for 8 bits the
        token's largest value maps exactly onto a code instead of being clipped (the paper text writes
        Q_b = 128; the released code uses 127).
    method="absmean" (BitNet a4.8 / BitNet v2, 4-bit): ``scale = sqrt(2^(b-1) - 1) / mean|x|``, which for
        4 bits is ``sqrt(7) / mean|x|``. It clips the rare large values to get finer steps for the bulk.
    """
    qmin, qmax = act_qrange(bits)
    xf = x.to(_compute_dtype(x))
    if method == "absmax":
        denom = xf.abs().amax(dim=-1, keepdim=True) if xf.shape[-1] > 0 else xf.new_ones(*xf.shape[:-1], 1)
        scale = qmax / denom.clamp(min=eps)
    elif method == "absmean":
        denom = xf.abs().mean(dim=-1, keepdim=True) if xf.shape[-1] > 0 else xf.new_ones(*xf.shape[:-1], 1)
        scale = (qmax**0.5) / denom.clamp(min=eps)
    else:
        raise ValueError(f"unknown activation quantization method {method!r} (use 'absmax' or 'absmean')")
    codes = (xf * scale).round().clamp(qmin, qmax)
    return codes, scale


def fake_quant_activations(x: Tensor, bits: int = 8, method: str = "absmax", eps: float = QUANT_EPS) -> Tensor:
    """Quantize then dequantize activations; same shape and dtype as ``x``. No gradient flows through it."""
    codes, scale = quantize_activations(x.detach(), bits=bits, method=method, eps=eps)
    out = codes / scale
    if x.dtype in (torch.float16, torch.bfloat16):  # absmean codes can overshoot a huge value by up to 2x
        limit = torch.finfo(x.dtype).max
        out = out.clamp(-limit, limit)
    return out.to(x.dtype)


def quantize_weights(w: Tensor, granularity: str = "tensor", eps: float = QUANT_EPS) -> tuple[Tensor, Tensor]:
    """Absmean ternary quantization (BitNet b1.58).

    Returns ``(codes, scale)`` with ``w ~= codes / scale`` and codes in {-1, 0, 1}.

    granularity="tensor": one scale for the whole matrix, ``scale = 1 / mean|W|`` (the reference recipe).
    granularity="channel": one scale per output row, ``scale[i] = 1 / mean|W[i, :]|``.
    """
    wf = w.to(_compute_dtype(w))
    if granularity == "tensor":
        gamma = wf.abs().mean() if wf.numel() > 0 else wf.new_ones(())
    elif granularity == "channel":
        if wf.dim() < 2:
            raise ValueError("per-channel weight quantization needs a weight with at least 2 dims")
        gamma = wf.abs().mean(dim=tuple(range(1, wf.dim())), keepdim=True)
    else:
        raise ValueError(f"unknown weight granularity {granularity!r} (use 'tensor' or 'channel')")
    scale = 1.0 / gamma.clamp(min=eps)
    codes = (wf * scale).round().clamp(-1, 1)
    return codes, scale


def fake_quant_weights(w: Tensor, granularity: str = "tensor", eps: float = QUANT_EPS) -> Tensor:
    """Ternarize then dequantize weights; same shape and dtype as ``w``. No gradient flows through it."""
    codes, scale = quantize_weights(w.detach(), granularity=granularity, eps=eps)
    return (codes / scale).to(w.dtype)


def ste(x: Tensor, x_quant: Tensor, strength: float = 1.0) -> Tensor:
    """Straight-through estimator: forward value ``x + strength * (x_quant - x)``, gradient of identity.

    ``strength`` < 1 blends in quantization gradually, which helps when starting from full-precision
    weights (Hugging Face, "Fine-tuning LLMs to 1.58bit", 2024).
    """
    if strength >= 1.0:
        return x + (x_quant - x).detach()
    return x + (strength * (x_quant - x)).detach()
