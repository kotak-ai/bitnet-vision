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

import warnings
from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hadamard import HadamardTransform
from .lora import LoRASettings, copy_lora, in_lora_adapter, lora_forward
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
        act_method: "absmax" or "absmean". Default: absmax for 8 or more bits, absmean below 8 bits
            (the BitNet a4.8 / v2 recipe for 4-bit activations).
        weight_granularity: "tensor" (reference: one scale per matrix) or "channel" (one per output row).
        input_norm: put an RMSNorm in front of the quantizer, as the reference does. Turn it off only if
            the input is already normalized.
        norm_eps: epsilon of that RMSNorm.
        deadzone_lambda: Tequila (arXiv 2509.23809). Weights whose ternary code is 0 get stuck there,
            receiving only noisy straight-through gradients. A small value (1e-3 in the paper) adds
            ``lambda * sum(dead weights of each output row)`` to the output, which gives those weights a
            direct gradient and folds into the bias at export. 0 (default) is plain BitNet.

    Attributes:
        quant_strength: 1.0 = fully quantized (default), 0.0 = full precision with the same architecture
            (used for full-precision baselines and teachers), values in between blend the two, which
            helps when converting a trained full-precision model.

    The full-precision ``weight`` stays the trainable parameter; it is quantized on the fly. Real memory
    and speed savings come from ``bitnet_vision.export.convert_for_inference``.

    LoRA adapters (``bitnet_vision.lora``) are added to the output in full precision; they read the raw input,
    before the norm and rotation.
    """

    lora_settings: dict[str, LoRASettings] | None = None  # adapters, see bitnet_vision.lora

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
        deadzone_lambda: float = 0.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if act_method is None:
            act_method = "absmax" if act_bits is None or act_bits >= 8 else "absmean"
        if act_bits is not None:
            fake_quant_activations(torch.zeros(1, 1), bits=act_bits, method=act_method)  # validate early
        if weight_granularity not in ("tensor", "channel"):
            raise ValueError(f"unknown weight granularity {weight_granularity!r}")
        self.act_bits = act_bits
        self.act_method = act_method
        self.weight_granularity = weight_granularity
        self.norm = RMSNorm(in_features, eps=norm_eps, device=device, dtype=dtype) if input_norm else None
        self.deadzone_lambda = float(deadzone_lambda)
        self.quant_strength = 1.0

    def rotate(self, x: Tensor) -> Tensor:
        """Hook applied between the norm and the activation quantizer (identity here, Hadamard in v2)."""
        return x

    def forward(self, x: Tensor) -> Tensor:
        raw = x  # LoRA adapters read the input as given
        in_dtype = x.dtype
        if in_dtype in (torch.float16, torch.bfloat16):
            x = x.float()  # norm, rotation and rounding in float32, so codes match the exported model
        if self.norm is not None:
            x = self.norm(x)
        x = self.rotate(x)
        w = self.weight
        lam = self.quant_strength
        if lam > 0:
            if self.act_bits is not None:
                x = ste(x, fake_quant_activations(x, self.act_bits, self.act_method), lam)
            w = ste(w, fake_quant_weights(w, self.weight_granularity), lam)
        y = F.linear(x.to(in_dtype), w, self.bias)
        if self.deadzone_lambda and lam > 0:
            y = y + (lam * self.deadzone_lambda) * self.deadzone_rowsum().to(y.dtype)
        return lora_forward(self, raw, y)

    def deadzone_rowsum(self) -> Tensor:
        """Per output row, the sum of latent weights whose ternary code is 0 (differentiable in the weights)."""
        codes, _ = quantize_weights(self.weight.detach(), granularity=self.weight_granularity)
        return (self.weight * (codes == 0).to(self.weight.dtype)).sum(dim=1)

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
        LoRA adapters of ``linear`` are copied over unchanged.
        """
        kwargs.setdefault("device", linear.weight.device)
        kwargs.setdefault("dtype", linear.weight.dtype)
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, **kwargs)
        with torch.no_grad():
            layer.weight.copy_(layer._weight_for_input_space(linear.weight))
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        layer.weight.requires_grad_(linear.weight.requires_grad)
        if layer.bias is not None:
            layer.bias.requires_grad_(linear.bias.requires_grad)
        copy_lora(linear, layer)
        return layer.train(linear.training)

    def _weight_for_input_space(self, weight: Tensor) -> Tensor:
        return weight

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, act_bits={self.act_bits}, act_method={self.act_method}, "
            f"weight_granularity={self.weight_granularity}, input_norm={self.norm is not None}"
            + (f", deadzone_lambda={self.deadzone_lambda}" if self.deadzone_lambda else "")
        )


class HBitLinear(BitLinear):
    """BitNet v2 H-BitLinear: norm, exact orthonormal Hadamard rotation, then BitLinear quantization.

    The rotation spreads activation outliers across channels, which is what makes 4-bit activations
    workable (BitNet v2, arXiv 2504.18415). BitNet v2 uses it for the attention output projection and the
    MLP down projection. ``random_signs`` (default on) flips a fixed random set of input signs before the
    Hadamard matrix, so mostly-positive inputs such as GELU outputs are not piled onto one coordinate (see
    hadamard.py).
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
        deadzone_lambda: float = 0.0,
        random_signs: bool = True,
    ) -> None:
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
            deadzone_lambda=deadzone_lambda,
        )
        self.hadamard = HadamardTransform(in_features, random_signs=random_signs)
        if device is not None:
            self.hadamard.to(device)

    def rotate(self, x: Tensor) -> Tensor:
        return self.hadamard(x)

    def _weight_for_input_space(self, weight: Tensor) -> Tensor:
        # y = x W^T = (x R)(W R)^T because R R^T = I, so rotated inputs need weights W R.
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
    """Replace every plain ``nn.Linear`` in ``model`` (in place) by ``layer_cls.from_linear``.

    ``exclude`` lists qualified module names (or prefixes) to keep in full precision, for example the
    classifier head. A Linear shared by several parents stays shared. Subclasses of nn.Linear, the
    projections inside ``nn.MultiheadAttention`` (whose input projection is a raw parameter) and the layers
    of LoRA adapters are left alone, with a warning for attention. Returns ``model``.
    """
    exclude = tuple(exclude)
    replaced: dict[int, BitLinear] = {}
    if any(isinstance(m, nn.MultiheadAttention) for m in model.modules()):
        warnings.warn("nn.MultiheadAttention layers are not converted and stay in full precision", stacklevel=2)
    for name, module in list(model.named_modules(remove_duplicate=False)):
        for child_name, child in list(module._modules.items()):  # includes repeated (shared) children
            full = f"{name}.{child_name}" if name else child_name
            if in_lora_adapter(full):
                continue
            if type(child) is nn.Linear and not any(full == e or full.startswith(e + ".") for e in exclude):
                if id(child) not in replaced:
                    replaced[id(child)] = layer_cls.from_linear(child, **kwargs)
                setattr(module, child_name, replaced[id(child)])
    return model
