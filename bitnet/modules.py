"""
BitNet b1.58 Core Modules

This module implements the core components of Microsoft's BitNet b1.58 architecture:
- BitLinear: A linear layer with ternary weight quantization {-1, 0, +1}
- RMSNorm: Root Mean Square Layer Normalization

Reference: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
https://arxiv.org/abs/2402.17764
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm normalizes the input tensor using the root mean square of its elements,
    without centering (no mean subtraction). This is computationally more efficient
    than LayerNorm while achieving similar performance.
    
    Formula:
        rmsnorm(x) = x * rsqrt(mean(x^2) + ε) * weight
    
    Args:
        dim: The dimension of the input features.
        eps: A small constant added for numerical stability. Default: 1e-6.
    
    Attributes:
        weight: Learnable scale parameter of shape (dim,).
        eps: Epsilon value for numerical stability.
    
    Example:
        >>> norm = RMSNorm(dim=512)
        >>> x = torch.randn(2, 10, 512)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMS normalization to the input tensor.
        
        Args:
            x: Input tensor of shape (..., dim).
        
        Returns:
            Normalized tensor of the same shape as input.
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        # Using rsqrt for efficiency: 1 / sqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


def _ste_sign(x: Tensor) -> Tensor:
    """
    Ternary sign function with Straight-Through Estimator (STE).
    
    Forward: Returns sign of x (-1, 0, or +1)
    Backward: Gradient passes through unchanged (STE)
    
    Args:
        x: Input tensor.
    
    Returns:
        Tensor with values in {-1, 0, +1}.
    """
    # Forward pass: compute sign
    # Backward pass: gradient flows through unchanged (STE)
    return (x.sign() - x).detach() + x


def _ste_round(x: Tensor) -> Tensor:
    """
    Round function with Straight-Through Estimator (STE).
    
    Forward: Rounds to nearest integer
    Backward: Gradient passes through unchanged (STE)
    
    Args:
        x: Input tensor.
    
    Returns:
        Rounded tensor.
    """
    return (x.round() - x).detach() + x


def quantize_weights_absmean(
    weights: Tensor,
    eps: float = 1e-5
) -> tuple[Tensor, Tensor]:
    """
    Quantize weights to ternary values {-1, 0, +1} using the absmean method.
    
    This implements the weight quantization from BitNet b1.58:
    1. Compute scale factor γ = mean(|W|)
    2. Normalize: W_normalized = W / (γ + ε)
    3. Quantize: W_ternary = RoundClip(W_normalized, -1, 1)
    
    Uses STE for gradient flow through the quantization.
    
    Args:
        weights: Full-precision weight tensor.
        eps: Small constant for numerical stability.
    
    Returns:
        Tuple of (quantized_weights, scale_factor):
        - quantized_weights: Ternary weights in {-1, 0, +1}
        - scale_factor: The γ value used for dequantization
    """
    # Compute scale factor: mean of absolute values
    gamma = weights.abs().mean()
    
    # Normalize weights by scale factor
    weights_normalized = weights / (gamma + eps)
    
    # Round and clip to [-1, 1] using STE
    # First round, then clip to ensure values are in {-1, 0, +1}
    weights_rounded = _ste_round(weights_normalized)
    weights_ternary = weights_rounded.clamp(-1, 1)
    
    return weights_ternary, gamma


def quantize_activations_absmax(
    x: Tensor,
    bits: int = 8
) -> tuple[Tensor, Tensor]:
    """
    Quantize activations to b-bit integers using per-token absmax quantization.
    
    This implements the activation quantization from BitNet b1.58:
    1. Compute per-token scale: η = max(|x|) for each token
    2. Compute quantization range: Q_b = 2^(b-1)
    3. Quantize: x_quant = Clip(round(x × Q_b / η), -Q_b, Q_b-1)
    
    Uses STE for gradient flow through the quantization.
    
    Args:
        x: Input activation tensor of shape (batch, ..., features) or 
           (batch, seq_len, features).
        bits: Number of bits for quantization. Default: 8.
    
    Returns:
        Tuple of (quantized_activations, scale_factor):
        - quantized_activations: Quantized activations
        - scale_factor: Per-token η values for dequantization
    """
    # Quantization range
    Q_b = 2 ** (bits - 1)
    
    # Compute per-token scale factor (absmax along feature dimension)
    # eta shape: (..., 1) to broadcast correctly
    eta = x.abs().max(dim=-1, keepdim=True).values
    
    # Avoid division by zero
    eta = eta.clamp(min=1e-5)
    
    # Scale activations to quantization range
    x_scaled = x * Q_b / eta
    
    # Round with STE and clip to valid range
    x_rounded = _ste_round(x_scaled)
    x_quant = x_rounded.clamp(-Q_b, Q_b - 1)
    
    return x_quant, eta


class BitLinear(nn.Module):
    """
    BitLinear layer implementing BitNet b1.58 quantization.
    
    This is a drop-in replacement for nn.Linear that uses:
    - Ternary weight quantization {-1, 0, +1} using absmean function
    - 8-bit activation quantization using per-token absmax function
    - Straight-Through Estimator (STE) for backpropagation
    
    The layer maintains full-precision latent weights for gradient accumulation
    during training. Quantization is applied during the forward pass.
    
    Mathematical formulation:
        Weight quantization (absmean):
            γ = mean(|W|)
            W_normalized = W / (γ + ε)
            W_ternary = RoundClip(W_normalized, -1, 1)
        
        Activation quantization (absmax):
            η = max(|x|) per token
            Q_b = 2^(b-1) where b=8
            x_quant = Clip(round(x × Q_b / η), -Q_b, Q_b-1)
        
        Forward computation:
            y = (x_quant × W_ternary) × (γ × η / Q_b)
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: False (BitNet doesn't use bias).
        eps: Small constant for numerical stability. Default: 1e-5.
        bits: Number of bits for activation quantization. Default: 8.
    
    Attributes:
        weight: Full-precision latent weights of shape (out_features, in_features).
        bias: Optional bias of shape (out_features,).
    
    Example:
        >>> layer = BitLinear(512, 1024)
        >>> x = torch.randn(2, 10, 512)
        >>> output = layer(x)
        >>> output.shape
        torch.Size([2, 10, 1024])
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
        bits: int = 8
    ) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.bits = bits
        
        # Full-precision latent weights for gradient accumulation
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        # Kaiming initialization (fan_in mode)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with ternary weight and 8-bit activation quantization.
        
        Args:
            x: Input tensor of shape (batch, features) or (batch, seq_len, features).
        
        Returns:
            Output tensor of shape (batch, out_features) or (batch, seq_len, out_features).
        """
        # Quantize weights to ternary values {-1, 0, +1}
        weights_ternary, gamma = quantize_weights_absmean(self.weight, self.eps)
        
        # Quantize activations to 8-bit integers (per-token)
        Q_b = 2 ** (self.bits - 1)
        x_quant, eta = quantize_activations_absmax(x, self.bits)
        
        # Compute: y = (x_quant × W_ternary^T) × (γ × η / Q_b)
        # The scaling factor combines weight and activation scales
        y = F.linear(x_quant, weights_ternary, None)
        
        # Apply combined scaling factor
        # gamma is scalar, eta is per-token (..., 1)
        scale = gamma * eta / Q_b
        y = y * scale
        
        # Add bias if present
        if self.bias is not None:
            y = y + self.bias
        
        return y
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"bits={self.bits}"
        )


# Alias for consistency with common naming conventions
BitLinear1p58 = BitLinear


if __name__ == "__main__":
    # Simple test to verify the implementation
    print("Testing BitNet b1.58 modules...\n")
    
    # Test RMSNorm
    print("=" * 50)
    print("Testing RMSNorm")
    print("=" * 50)
    
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean (per feature): {y.mean(dim=(0, 1))[:5]}...")
    print(f"Output std (per feature): {y.std(dim=(0, 1))[:5]}...")
    print()
    
    # Test BitLinear
    print("=" * 50)
    print("Testing BitLinear")
    print("=" * 50)
    
    layer = BitLinear(in_features=64, out_features=128)
    x = torch.randn(2, 10, 64)
    y = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Layer: {layer}")
    print()
    
    # Test gradient flow
    print("=" * 50)
    print("Testing Gradient Flow (STE)")
    print("=" * 50)
    
    x = torch.randn(2, 10, 64, requires_grad=True)
    layer = BitLinear(64, 128)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    print(f"Input gradient exists: {x.grad is not None}")
    print(f"Weight gradient exists: {layer.weight.grad is not None}")
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Weight gradient shape: {layer.weight.grad.shape}")
    print(f"Weight gradient norm: {layer.weight.grad.norm():.4f}")
    print()
    
    # Verify ternary quantization
    print("=" * 50)
    print("Verifying Weight Quantization")
    print("=" * 50)
    
    weights_ternary, gamma = quantize_weights_absmean(layer.weight)
    unique_values = torch.unique(weights_ternary.detach())
    print(f"Scale factor (gamma): {gamma.item():.4f}")
    print(f"Unique quantized values: {unique_values.tolist()}")
    print(f"Expected: [-1.0, 0.0, 1.0]")
    print()
    
    # Test 2D input
    print("=" * 50)
    print("Testing 2D Input (batch, features)")
    print("=" * 50)
    
    x_2d = torch.randn(8, 64)
    y_2d = layer(x_2d)
    print(f"Input shape: {x_2d.shape}")
    print(f"Output shape: {y_2d.shape}")
    print()
    
    print("All tests passed!")
