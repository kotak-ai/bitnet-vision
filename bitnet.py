import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Formula:
        rmsnorm(x) = x * rsqrt(mean(x^2) + ε) * weight
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size: The size of the last dimension of the input.
            eps: Small value for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, ..., hidden_size)
        
        Returns:
            Normalized tensor of the same shape.
        """
        # Calculate mean of squared values
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Calculate inverse square root with epsilon
        norm = torch.rsqrt(variance + self.eps)
        # Normalize and scale
        return x * norm * self.weight


def activation_quant(x: torch.Tensor):
    """
    Per-token activation quantization to 8-bit integers.
    
    Formula:
        η = max(|x|) per token
        Q_b = 2^(b-1) where b=8 (128)
        x_quant = Clip(round(x × Q_b / η), -Q_b, Q_b-1)
    """
    # Scale factor η (eta): max(|x|) per token (dim=-1)
    # x shape: (batch, seq, features) or (batch, features)
    eta = x.abs().max(dim=-1, keepdim=True).values
    eta = eta.clamp(min=1e-5)  # Avoid division by zero

    Q_b = 128.0
    
    # Quantize
    x_scaled = x * Q_b / eta
    x_quant = torch.round(x_scaled)
    x_quant = torch.clamp(x_quant, -Q_b, Q_b - 1)

    # STE: detach the gradient for the quantization step
    # x_quant = x + (x_quant - x).detach()
    # However, we need to return both the quantized value and the scale for the linear layer
    # We apply STE here to x_quant so it flows back to x
    x_quant = x + (x_quant - x).detach()

    return x_quant, eta


def weight_quant(w: torch.Tensor, eps: float = 1e-5):
    """
    Weight quantization to ternary values {-1, 0, +1}.
    
    Formula:
        γ = mean(|W|)
        W_normalized = W / (γ + ε)
        W_ternary = RoundClip(W_normalized, -1, 1)
    """
    # Scale factor γ (gamma): mean(|W|)
    gamma = w.abs().mean()
    
    # Normalize
    # W_normalized = W / (γ + ε)
    w_normalized = w / (gamma + eps)
    
    # Quantize: RoundClip(W_normalized, -1, 1)
    w_ternary = torch.round(w_normalized)
    w_ternary = torch.clamp(w_ternary, -1, 1)
    
    # STE: detach gradient
    w_ternary = w + (w_ternary - w).detach()
    
    return w_ternary, gamma


class BitLinear(nn.Module):
    """
    BitLinear layer implementing BitNet b1.58 architecture.
    
    Quantizes weights to ternary {-1, 0, 1} and activations to 8-bit integers.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, eps: float = 1e-5):
        """
        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: Whether to add a bias term (default: False).
            eps: Epsilon for numerical stability in weight quantization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Maintain full-precision latent weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, in_features) or (batch, in_features)
            
        Returns:
            Output tensor of shape (batch, seq_len, out_features) or (batch, out_features)
        """
        # 1. Quantize activations (absmax)
        x_quant, eta = activation_quant(x)
        
        # 2. Quantize weights (absmean)
        w_ternary, gamma = weight_quant(self.weight, self.eps)
        
        # 3. Linear computation with quantized values
        # The actual matrix multiplication simulates the low-bit operation
        y = F.linear(x_quant, w_ternary, bias=self.bias)
        
        # 4. Rescale
        # y = (x_quant * W_ternary) * (gamma * eta / Q_b)
        # Q_b = 128
        Q_b = 128.0
        scale = (gamma * eta) / Q_b
        
        y = y * scale
        
        return y
