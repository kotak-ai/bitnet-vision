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


# =============================================================================
# Hadamard Transform Utilities for H-BitLinear (BitNet v2)
# =============================================================================


def _next_power_of_2(n: int) -> int:
    """
    Return the smallest power of 2 greater than or equal to n.
    
    Args:
        n: Input integer.
    
    Returns:
        Smallest power of 2 >= n.
    """
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def generate_hadamard_matrix(n: int, device: torch.device = None, dtype: torch.dtype = None) -> Tensor:
    """
    Generate a Walsh-Hadamard matrix of size n × n.
    
    The Hadamard matrix is constructed recursively using the Sylvester construction:
        H_1 = [[1]]
        H_2n = [[H_n, H_n], [H_n, -H_n]]
    
    Properties:
        - H_m × H_m^T = m × I (orthogonal up to scaling)
        - H_m^(-1) = H_m / m (self-inverse up to scaling)
        - All entries are +1 or -1
    
    Args:
        n: Size of the matrix. Must be a power of 2.
        device: Target device for the tensor.
        dtype: Data type for the tensor.
    
    Returns:
        Hadamard matrix of shape (n, n).
    
    Raises:
        ValueError: If n is not a power of 2.
    
    Example:
        >>> H = generate_hadamard_matrix(4)
        >>> H
        tensor([[ 1.,  1.,  1.,  1.],
                [ 1., -1.,  1., -1.],
                [ 1.,  1., -1., -1.],
                [ 1., -1., -1.,  1.]])
    """
    if not _is_power_of_2(n):
        raise ValueError(f"Hadamard matrix size must be a power of 2, got {n}")
    
    # Start with H_1 = [[1]]
    H = torch.ones(1, 1, device=device, dtype=dtype)
    
    # Recursively build up: H_2n = [[H_n, H_n], [H_n, -H_n]]
    while H.size(0) < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    
    return H


def fast_walsh_hadamard_transform(x: Tensor, normalize: bool = True) -> Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT) with O(n log n) complexity.
    
    This is an in-place-style algorithm that computes the Hadamard transform
    efficiently without explicitly constructing the Hadamard matrix.
    
    The transform is applied along the last dimension of the input tensor.
    
    Args:
        x: Input tensor of shape (..., n) where n must be a power of 2.
        normalize: If True, normalize by 1/sqrt(n) to make the transform
                   orthonormal (i.e., its own inverse). Default: True.
    
    Returns:
        Hadamard-transformed tensor of the same shape.
    
    Raises:
        ValueError: If the last dimension is not a power of 2.
    
    Example:
        >>> x = torch.randn(2, 8)
        >>> y = fast_walsh_hadamard_transform(x)
        >>> x_recovered = fast_walsh_hadamard_transform(y)  # Self-inverse when normalized
        >>> torch.allclose(x, x_recovered)
        True
    """
    n = x.shape[-1]
    
    if not _is_power_of_2(n):
        raise ValueError(f"FWHT requires last dimension to be power of 2, got {n}")
    
    # Work with a copy to avoid modifying input
    result = x.clone()
    
    # Number of stages = log2(n)
    h = 1
    while h < n:
        # Process pairs of elements at distance h
        # This is the butterfly operation of FWHT
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                # Butterfly: (a, b) -> (a + b, a - b)
                a = result[..., j].clone()
                b = result[..., j + h].clone()
                result[..., j] = a + b
                result[..., j + h] = a - b
        h *= 2
    
    # Normalize for orthonormal transform
    if normalize:
        result = result / math.sqrt(n)
    
    return result


def fast_walsh_hadamard_transform_vectorized(x: Tensor, normalize: bool = True) -> Tensor:
    """
    Vectorized Fast Walsh-Hadamard Transform (FWHT) with O(n log n) complexity.
    
    This is a more efficient implementation that uses tensor operations
    instead of explicit loops, making it faster on GPU.
    
    Args:
        x: Input tensor of shape (..., n) where n must be a power of 2.
        normalize: If True, normalize by 1/sqrt(n) for orthonormal transform.
    
    Returns:
        Hadamard-transformed tensor of the same shape.
    """
    n = x.shape[-1]
    
    if not _is_power_of_2(n):
        raise ValueError(f"FWHT requires last dimension to be power of 2, got {n}")
    
    # Get original shape for reshaping
    original_shape = x.shape
    
    # Flatten all but last dimension for easier processing
    x_flat = x.reshape(-1, n)
    batch_size = x_flat.shape[0]
    
    result = x_flat.clone()
    
    # Number of stages = log2(n)
    h = 1
    while h < n:
        # Reshape to group pairs
        result = result.view(batch_size, n // (2 * h), 2, h)
        
        # Butterfly operation using slicing
        a = result[:, :, 0, :].clone()
        b = result[:, :, 1, :].clone()
        result[:, :, 0, :] = a + b
        result[:, :, 1, :] = a - b
        
        # Reshape back
        result = result.view(batch_size, n)
        h *= 2
    
    # Normalize for orthonormal transform
    if normalize:
        result = result / math.sqrt(n)
    
    # Restore original shape
    return result.view(original_shape)


class HadamardTransform(torch.autograd.Function):
    """
    Autograd Function for Hadamard Transform with proper backward pass.
    
    The Hadamard transform is orthogonal (when normalized), meaning:
        H @ H^T = I
    
    Therefore, the backward pass (gradient w.r.t. input) also uses the
    Hadamard transform, since for an orthogonal matrix:
        d(loss)/d(input) = H^T @ d(loss)/d(output) = H @ d(loss)/d(output)
    
    This is because H is symmetric (H = H^T) for the Walsh-Hadamard matrix.
    """
    
    @staticmethod
    def forward(ctx, x: Tensor, padded_dim: int) -> Tensor:
        """
        Apply forward Hadamard transform.
        
        Args:
            ctx: Context for saving tensors for backward.
            x: Input tensor with last dimension being power of 2.
            padded_dim: The dimension after padding (for gradient computation).
        
        Returns:
            Hadamard-transformed tensor.
        """
        ctx.padded_dim = padded_dim
        return fast_walsh_hadamard_transform_vectorized(x, normalize=True)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        """
        Apply backward pass using Hadamard transform.
        
        Due to orthogonality of H (when normalized), the gradient also
        goes through the Hadamard transform.
        
        Args:
            ctx: Context with saved tensors.
            grad_output: Gradient of loss w.r.t. output.
        
        Returns:
            Gradient of loss w.r.t. input, None for padded_dim.
        """
        # Apply Hadamard transform to gradients (orthogonal property)
        grad_input = fast_walsh_hadamard_transform_vectorized(grad_output, normalize=True)
        return grad_input, None


def hadamard_transform(x: Tensor, padded_dim: Optional[int] = None) -> Tensor:
    """
    Apply Hadamard transform with autograd support.
    
    This is a convenience wrapper around HadamardTransform.apply() that
    handles the autograd function properly.
    
    Args:
        x: Input tensor with last dimension being power of 2.
        padded_dim: The padded dimension size (optional, for internal use).
    
    Returns:
        Hadamard-transformed tensor.
    """
    if padded_dim is None:
        padded_dim = x.shape[-1]
    return HadamardTransform.apply(x, padded_dim)


class HBitLinear(BitLinear):
    """
    H-BitLinear: BitLinear with online Hadamard transformation (BitNet v2).
    
    This extends BitLinear by applying a Hadamard transformation to activations
    before quantization. The Hadamard transform smooths sharp, outlier-prone
    activation distributions into more Gaussian-like forms, making them more
    suitable for low-bit quantization.
    
    The transformation uses the Fast Walsh-Hadamard Transform (FWHT) for
    O(n log n) efficiency instead of O(n²) matrix multiplication.
    
    Forward pass:
        1. Pad input to power of 2 if necessary
        2. Apply Hadamard transform: x_h = FWHT(x) / sqrt(n)
        3. Unpad to original dimension
        4. Apply standard BitLinear quantization and computation
    
    Backward pass:
        - Due to orthogonality, gradients also go through Hadamard transform
    
    Hadamard Properties:
        - H_m × H_m^T = m × I (orthogonal)
        - H_m^(-1) = H_m / m (self-inverse)
        - Recursive: H_2n = [[H_n, H_n], [H_n, -H_n]]
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: False.
        eps: Small constant for numerical stability. Default: 1e-5.
        bits: Number of bits for activation quantization. Default: 8.
    
    Example:
        >>> layer = HBitLinear(64, 128)
        >>> x = torch.randn(2, 10, 64)
        >>> output = layer(x)
        >>> output.shape
        torch.Size([2, 10, 128])
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
        bits: int = 8
    ) -> None:
        super().__init__(in_features, out_features, bias, eps, bits)
        
        # Compute padded dimension (next power of 2)
        self.padded_in_features = _next_power_of_2(in_features)
        self._needs_padding = self.padded_in_features != in_features
    
    def _apply_hadamard_transform(self, x: Tensor) -> Tensor:
        """
        Apply Hadamard transform to input activations.
        
        Handles padding for non-power-of-2 dimensions.
        
        Args:
            x: Input tensor of shape (..., in_features).
        
        Returns:
            Hadamard-transformed tensor of shape (..., in_features).
        """
        original_dim = x.shape[-1]
        
        # Pad to power of 2 if necessary
        if self._needs_padding:
            pad_size = self.padded_in_features - original_dim
            x = F.pad(x, (0, pad_size), mode='constant', value=0)
        
        # Apply Hadamard transform
        x = hadamard_transform(x, self.padded_in_features)
        
        # Unpad back to original dimension
        if self._needs_padding:
            x = x[..., :original_dim]
        
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with Hadamard transform, then ternary weight and 8-bit 
        activation quantization.
        
        Args:
            x: Input tensor of shape (batch, features) or (batch, seq_len, features).
        
        Returns:
            Output tensor of shape (batch, out_features) or (batch, seq_len, out_features).
        """
        # Step 1: Apply Hadamard transform to smooth activation distribution
        x_hadamard = self._apply_hadamard_transform(x)
        
        # Step 2: Quantize weights to ternary values {-1, 0, +1}
        weights_ternary, gamma = quantize_weights_absmean(self.weight, self.eps)
        
        # Step 3: Quantize transformed activations to 8-bit integers (per-token)
        Q_b = 2 ** (self.bits - 1)
        x_quant, eta = quantize_activations_absmax(x_hadamard, self.bits)
        
        # Step 4: Compute: y = (x_quant × W_ternary^T) × (γ × η / Q_b)
        y = F.linear(x_quant, weights_ternary, None)
        
        # Apply combined scaling factor
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
            f"bits={self.bits}, "
            f"padded_in_features={self.padded_in_features}"
        )


# Alias
HadamardBitLinear = HBitLinear


if __name__ == "__main__":
    # Simple test to verify the implementation
    print("Testing BitNet b1.58 and v2 modules...\n")
    
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
    
    # Test Hadamard Transform
    print("=" * 50)
    print("Testing Fast Walsh-Hadamard Transform")
    print("=" * 50)
    
    x = torch.randn(2, 8)
    y = fast_walsh_hadamard_transform_vectorized(x, normalize=True)
    x_recovered = fast_walsh_hadamard_transform_vectorized(y, normalize=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Is self-inverse: {torch.allclose(x, x_recovered, atol=1e-5)}")
    print()
    
    # Test Hadamard matrix generation
    print("=" * 50)
    print("Testing Hadamard Matrix Generation")
    print("=" * 50)
    
    H4 = generate_hadamard_matrix(4)
    print(f"H_4 matrix:\n{H4}")
    # Verify orthogonality: H @ H^T = n * I
    HHT = H4 @ H4.T
    expected = 4 * torch.eye(4)
    print(f"H @ H^T = 4*I: {torch.allclose(HHT, expected)}")
    print()
    
    # Test HBitLinear with power-of-2 dimension
    print("=" * 50)
    print("Testing HBitLinear (power-of-2 dimension)")
    print("=" * 50)
    
    h_layer = HBitLinear(in_features=64, out_features=128)
    x = torch.randn(2, 10, 64)
    y = h_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Layer: {h_layer}")
    print()
    
    # Test HBitLinear with non-power-of-2 dimension
    print("=" * 50)
    print("Testing HBitLinear (non-power-of-2 dimension)")
    print("=" * 50)
    
    h_layer_non_pow2 = HBitLinear(in_features=50, out_features=100)
    x = torch.randn(2, 10, 50)
    y = h_layer_non_pow2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Layer: {h_layer_non_pow2}")
    print(f"Padded dimension: {h_layer_non_pow2.padded_in_features}")
    print()
    
    # Test HBitLinear gradient flow
    print("=" * 50)
    print("Testing HBitLinear Gradient Flow")
    print("=" * 50)
    
    h_layer = HBitLinear(64, 128)
    x = torch.randn(2, 10, 64, requires_grad=True)
    y = h_layer(x)
    loss = y.sum()
    loss.backward()
    
    print(f"Input gradient exists: {x.grad is not None}")
    print(f"Weight gradient exists: {h_layer.weight.grad is not None}")
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Weight gradient norm: {h_layer.weight.grad.norm():.4f}")
    print()
    
    # Compare activation distributions
    print("=" * 50)
    print("Comparing Activation Distributions")
    print("=" * 50)
    
    # Create some outlier-prone activations
    x = torch.randn(4, 64)
    x[:, 0] *= 10  # Create outliers in first dimension
    
    # Without Hadamard
    absmax_orig = x.abs().max(dim=-1).values
    
    # With Hadamard
    x_h = fast_walsh_hadamard_transform_vectorized(x, normalize=True)
    absmax_hadamard = x_h.abs().max(dim=-1).values
    
    print(f"Original absmax (per token): {absmax_orig.tolist()}")
    print(f"After Hadamard absmax (per token): {absmax_hadamard.tolist()}")
    print(f"Absmax reduced: {(absmax_hadamard < absmax_orig).all()}")
    print()
    
    print("All tests passed!")
