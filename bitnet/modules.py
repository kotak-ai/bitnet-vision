"""Core BitNet modules (PyTorch).

This file implements:
- `BitLinear`: a linear layer with ternary weights {-1, 0, +1} (absmean scaling)
  and 8-bit per-token activation quantization (absmax scaling), using an STE for
  backpropagation while keeping full-precision latent parameters.
- `RMSNorm`: Root Mean Square LayerNorm variant used in many transformer models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Tuple

import torch
from torch import Tensor, nn


def _ste(quantized: Tensor, latent: Tensor) -> Tensor:
    """Straight-Through Estimator (STE).

    Forward uses `quantized`, backward uses identity gradient as if output were `latent`.
    """

    # y = latent + (quantized - latent).detach()
    return latent + (quantized - latent).detach()


def _absmean_scale(w: Tensor, eps: float) -> Tensor:
    """absmean scale factor γ = mean(|W|) as a scalar tensor."""

    gamma = w.abs().mean()
    return gamma.clamp_min(eps)


def _ternarize_absmean(w: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
    """Quantize weights to ternary with absmean scaling (with STE).

    Mathematical form:
      γ = mean(|W|)
      W_normalized = W / (γ + ε)
      W_ternary = RoundClip(W_normalized, -1, 1)  # then mapped to {-1,0,+1}

    Returns:
      (w_ternary_ste, gamma)
        - w_ternary_ste: STE'd tensor in {-1,0,+1} (forward), with gradients
          flowing as if it were W_normalized.
        - gamma: scalar scale factor (clamped by eps).
    """

    gamma = _absmean_scale(w, eps=eps)
    w_norm = w / gamma

    # RoundClip then map to exact ternary using sign (sign(0) == 0).
    w_roundclip = torch.clamp(torch.round(w_norm), -1.0, 1.0)
    w_ternary = torch.sign(w_roundclip)

    # STE in normalized domain so gradients propagate to full-precision W.
    w_ternary_ste = _ste(w_ternary, w_norm)
    return w_ternary_ste, gamma


@dataclass(frozen=True)
class ActQuantConfig:
    """Activation quantization configuration."""

    bits: int = 8

    @property
    def q_b(self) -> int:
        """Quantization range constant Q_b = 2^(b-1)."""

        return 1 << (self.bits - 1)


def _absmax_per_token(x: Tensor, eps: float) -> Tensor:
    """absmax scale η = max(|x|) per token (last-dim)."""

    eta = x.abs().amax(dim=-1, keepdim=True)
    return eta.clamp_min(eps)


def _quantize_activations_int(
    x: Tensor,
    *,
    eps: float,
    cfg: ActQuantConfig,
) -> Tuple[Tensor, Tensor]:
    """Quantize activations per-token to int range using absmax (with STE).

    Mathematical form (b=8):
      η = max(|x|) per token
      Q_b = 2^(b-1)
      x_quant = Clip(round(x * Q_b / η), -Q_b, Q_b-1)

    Returns:
      (x_quant_ste, eta)
        - x_quant_ste: forward equals integer-like quantized values (float tensor),
          backward behaves like x_scaled = x * Q_b / η (STE).
        - eta: per-token scale (clamped by eps).
    """

    q_b: Final[int] = cfg.q_b
    eta = _absmax_per_token(x, eps=eps)

    # Work in "integer domain": x_scaled = x * Q_b / eta
    x_scaled = x * (float(q_b) / eta)
    x_quant = torch.clamp(torch.round(x_scaled), -float(q_b), float(q_b - 1))

    # STE in scaled domain: forward uses x_quant, backward flows through x_scaled.
    x_quant_ste = _ste(x_quant, x_scaled)
    return x_quant_ste, eta


class BitLinear(nn.Module):
    """BitNet-style linear layer: ternary weights + int8 per-token activations.

    Requirements satisfied:
    - Extends `nn.Module`
    - Weight ternarization uses absmean (γ) during forward pass
    - Activation quantization uses absmax (η) per token during forward pass (8-bit)
    - Uses STE for backprop through quantization
    - Maintains full-precision latent weights (and optional bias) as parameters

    Forward computation:
      y = (x_quant × W_ternary) × (γ × η / Q_b)

    Notes:
    - Supports inputs of shape (batch, in_features) or (batch, seq, in_features).
    - The quantized tensors are represented as float tensors holding integer values
      for broad device compatibility (int matmul kernels vary by backend).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        eps: float = 1e-5,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.eps = float(eps)
        self.act_cfg = ActQuantConfig(bits=int(activation_bits))

        # Full-precision latent parameters (gradient accumulates here).
        self.weight: nn.Parameter = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters similarly to nn.Linear."""

        # Kaiming uniform is standard for linear layers.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() not in (2, 3):
            raise ValueError(
                f"BitLinear expects 2D or 3D input (got shape {tuple(x.shape)})"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dim == in_features ({self.in_features}), got {x.shape[-1]}"
            )

        # --- Quantize activations (per token) ---
        x_quant_ste, eta = _quantize_activations_int(x, eps=self.eps, cfg=self.act_cfg)

        # --- Quantize weights (ternary) ---
        w_ternary_ste, gamma = _ternarize_absmean(self.weight, eps=self.eps)

        # --- Compute y = (x_quant × W_ternary) × (γ × η / Q_b) ---
        # x_quant_ste: (..., in_features)
        # w_ternary_ste: (out_features, in_features) -> transpose to (in_features, out_features)
        y_int = torch.matmul(x_quant_ste, w_ternary_ste.t())

        scale = (gamma * eta) / float(self.act_cfg.q_b)
        y = y_int * scale

        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, eps={self.eps}, activation_bits={self.act_cfg.bits}"
        )


class RMSNorm(nn.Module):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight

    - No bias term
    - Learnable `weight` parameter of shape (dim,)
    """

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight: nn.Parameter = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm expected last dim {self.dim}, got {x.shape[-1]}")
        rms = torch.mean(x * x, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

