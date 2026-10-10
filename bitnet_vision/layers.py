"""BitNet layers: RMSNorm, BitLinear (BitNet b1.58) and HBitLinear (BitNet v2).

BitLinear follows Microsoft's reference training code ("The Era of 1-bit LLMs: Training Tips, Code and
FAQ", Fig. 3) exactly::

    x = RMSNorm(x)                                              # norm inside the layer
    x_q = x + (activation_quant(x) - x).detach()                # 8-bit per-token absmax, 127 scale
    w_q = w + (weight_quant(w) - w).detach()                    # ternary {-1, 0, 1}, absmean scale
    y = F.linear(x_q, w_q, bias)                                # bias is added after rescaling

The quantizers return dequantized values, so the matmul runs on already rescaled numbers. That keeps it
inside the float16 range and gives the reference gradients: the scales are constants in backward.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hadamard import HadamardTransform
from .quant import fake_quant_activations, fake_quant_weights, quantize_weights, ste

__all__ = [
    "RMSNorm",
    "BitLinear",
    "HBitLinear",
    "bitlinear_modules",
    "set_quant_strength",
    "convert_linear_layers",
]


class RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * weight``, computed in float32 for half-precision inputs.

    Squaring in float16 overflows for values >= 256 and wipes out the whole token, so the statistics are
    always computed in float32 (as in LLaMA's RMSNorm).
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        xf = x if x.dtype == torch.float64 else x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.to(y.dtype)
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}"


class BitLinear(nn.Linear):
    """Drop-in replacement for ``nn.Linear`` with ternary weights and low-bit activations.

    Args:
        in_features, out_features, bias, device, dtype: as for ``nn.Linear``.
        act_bits: activation bits (8 in BitNet b1.58). ``None`` keeps activations in full precision.
        act_method: "absmax" (default for 8 bits) or "absmean" (BitNet a4.8 / v2 recipe for 4 bits).
        weight_granularity: "tensor" (reference: one scale per matrix) or "channel" (one per output row).
        input_norm: put an RMSNorm in front of the quantizer, as the reference does. Turn it off only if
            the input is already normalized.
        norm_eps: epsilon of that RMSNorm.

    Attributes:
        quant_strength: 1.0 = fully quantized (default), 0.0 = full precision with the same architecture
            (used for full-precision baselines and teachers), values in between blend the two, which
            helps when converting a trained full-precision model.

    The full-precision ``weight`` stays the trainable parameter; it is quantized on the fly. Real memory
    and speed savings come from ``bitnet_vision.export.convert_for_inference``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        act_bits: int | None = 8,
        act_method: str = "absmax",
        weight_granularity: str = "tensor",
        input_norm: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if act_bits is not None:
            fake_quant_activations(torch.zeros(1, 1), bits=act_bits, method=act_method)  # validate early
        if weight_granularity not in ("tensor", "channel"):
            raise ValueError(f"unknown weight granularity {weight_granularity!r}")
        self.act_bits = act_bits
        self.act_method = act_method
        self.weight_granularity = weight_granularity
        self.norm = RMSNorm(in_features, eps=norm_eps, device=device, dtype=dtype) if input_norm else None
        self.quant_strength = 1.0

    def rotate(self, x: Tensor) -> Tensor:
        """Hook applied between the norm and the activation quantizer (identity here, Hadamard in v2)."""
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.norm is not None:
            x = self.norm(x)
        x = self.rotate(x)
        w = self.weight
        lam = self.quant_strength
        if lam > 0:
            if self.act_bits is not None:
                x = ste(x, fake_quant_activations(x, self.act_bits, self.act_method), lam)
            w = ste(w, fake_quant_weights(w, self.weight_granularity), lam)
        return F.linear(x, w, self.bias)

    @torch.no_grad()
    def ternary_weight(self) -> tuple[Tensor, Tensor]:
        """``(codes, scale)`` with codes in {-1, 0, 1} (int8) and ``weight ~= codes / scale``."""
        codes, scale = quantize_weights(self.weight, granularity=self.weight_granularity)
        return codes.to(torch.int8), scale.float()

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> BitLinear:
        """Build a layer with the same shape and copy weight and bias from a trained ``nn.Linear``.

        Note: with ``input_norm=True`` (the default) the new layer also normalizes its input, so it only
        computes the same function after (short) fine-tuning. HBitLinear rotates the copied weights so
        that, before quantization, it computes exactly what the original layer computed on its input.
        """
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **kwargs,
        )
        with torch.no_grad():
            layer.weight.copy_(layer._weight_for_input_space(linear.weight))
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def _weight_for_input_space(self, weight: Tensor) -> Tensor:
        return weight

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, act_bits={self.act_bits}, act_method={self.act_method}, "
            f"weight_granularity={self.weight_granularity}, input_norm={self.norm is not None}"
        )


class HBitLinear(BitLinear):
    """BitNet v2 H-BitLinear: norm, exact orthonormal Hadamard rotation, then BitLinear quantization.

    The rotation spreads activation outliers across channels, which is what makes 4-bit activations
    workable (BitNet v2, arXiv 2504.18415). BitNet v2 uses it for the attention output projection and the
    MLP down projection. ``act_method`` defaults to "absmax" for 8 bits and "absmean" below 8 bits.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        act_bits: int | None = 8,
        act_method: str | None = None,
        weight_granularity: str = "tensor",
        input_norm: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        if act_method is None:
            act_method = "absmax" if act_bits is None or act_bits >= 8 else "absmean"
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            act_bits=act_bits,
            act_method=act_method,
            weight_granularity=weight_granularity,
            input_norm=input_norm,
            norm_eps=norm_eps,
        )
        self.hadamard = HadamardTransform(in_features)
        if device is not None:
            self.hadamard.to(device)

    def rotate(self, x: Tensor) -> Tensor:
        return self.hadamard(x)

    def _weight_for_input_space(self, weight: Tensor) -> Tensor:
        # y = x W^T = (x H)(W H)^T because H H^T = I, so rotated inputs need weights W H.
        return self.hadamard(weight)


def bitlinear_modules(model: nn.Module) -> Iterator[BitLinear]:
    """All BitLinear / HBitLinear layers in ``model``."""
    return (m for m in model.modules() if isinstance(m, BitLinear))


def set_quant_strength(model: nn.Module, strength: float) -> None:
    """Set ``quant_strength`` on every BitLinear in ``model`` (0 = full precision, 1 = fully quantized)."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"quant strength must be in [0, 1], got {strength}")
    for m in bitlinear_modules(model):
        m.quant_strength = float(strength)


def convert_linear_layers(
    model: nn.Module,
    exclude: Iterable[str] = (),
    layer_cls: type[BitLinear] = BitLinear,
    **kwargs,
) -> nn.Module:
    """Replace every ``nn.Linear`` in ``model`` (in place) by ``layer_cls.from_linear``.

    ``exclude`` lists qualified module names (or prefixes) to keep in full precision, for example the
    classifier head. Returns ``model``.
    """
    exclude = tuple(exclude)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if type(child) is nn.Linear and not any(full == e or full.startswith(e + ".") for e in exclude):
                setattr(module, child_name, layer_cls.from_linear(child, **kwargs))
    return model
