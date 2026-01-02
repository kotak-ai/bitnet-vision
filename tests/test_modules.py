"""
Unit tests for BitNet b1.58 modules.
"""

import math
import pytest
import torch
import torch.nn as nn

from bitnet import (
    BitLinear,
    RMSNorm,
    quantize_activations_absmax,
    quantize_weights_absmean,
)


class TestRMSNorm:
    """Tests for RMSNorm module."""
    
    def test_output_shape_2d(self):
        """Test RMSNorm preserves 2D input shape."""
        norm = RMSNorm(dim=64)
        x = torch.randn(8, 64)
        y = norm(x)
        assert y.shape == x.shape
    
    def test_output_shape_3d(self):
        """Test RMSNorm preserves 3D input shape."""
        norm = RMSNorm(dim=128)
        x = torch.randn(2, 10, 128)
        y = norm(x)
        assert y.shape == x.shape
    
    def test_output_shape_4d(self):
        """Test RMSNorm preserves 4D input shape."""
        norm = RMSNorm(dim=256)
        x = torch.randn(2, 4, 8, 256)
        y = norm(x)
        assert y.shape == x.shape
    
    def test_learnable_weight(self):
        """Test that weight parameter is learnable."""
        norm = RMSNorm(dim=64)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (64,)
    
    def test_weight_initialization(self):
        """Test weight is initialized to ones."""
        norm = RMSNorm(dim=64)
        assert torch.allclose(norm.weight, torch.ones(64))
    
    def test_gradient_flow(self):
        """Test gradients flow through RMSNorm."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert norm.weight.grad is not None
    
    def test_numerical_stability(self):
        """Test RMSNorm handles small values without NaN."""
        norm = RMSNorm(dim=64, eps=1e-6)
        x = torch.zeros(2, 10, 64)  # All zeros
        y = norm(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
    
    def test_normalization_property(self):
        """Test that RMS normalization is applied correctly."""
        dim = 64
        norm = RMSNorm(dim=dim)
        x = torch.randn(2, 10, dim)
        y = norm(x)
        
        # Manually compute expected output
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        expected = x / rms * norm.weight
        
        assert torch.allclose(y, expected, atol=1e-5)


class TestQuantizeWeightsAbsmean:
    """Tests for weight quantization function."""
    
    def test_output_is_ternary(self):
        """Test that quantized weights are in {-1, 0, +1}."""
        weights = torch.randn(128, 64)
        w_quant, gamma = quantize_weights_absmean(weights)
        
        unique_values = torch.unique(w_quant.detach())
        assert all(v in [-1, 0, 1] for v in unique_values.tolist())
    
    def test_scale_factor_is_positive(self):
        """Test that scale factor gamma is positive."""
        weights = torch.randn(128, 64)
        _, gamma = quantize_weights_absmean(weights)
        
        assert gamma.item() > 0
    
    def test_scale_factor_is_absmean(self):
        """Test that gamma equals mean of absolute values."""
        weights = torch.randn(128, 64)
        _, gamma = quantize_weights_absmean(weights)
        
        expected_gamma = weights.abs().mean()
        assert torch.allclose(gamma, expected_gamma, atol=1e-6)
    
    def test_gradient_flow_ste(self):
        """Test STE allows gradients to flow."""
        weights = torch.randn(128, 64, requires_grad=True)
        w_quant, gamma = quantize_weights_absmean(weights)
        
        loss = w_quant.sum() + gamma
        loss.backward()
        
        assert weights.grad is not None
        assert not torch.isnan(weights.grad).any()


class TestQuantizeActivationsAbsmax:
    """Tests for activation quantization function."""
    
    def test_output_range(self):
        """Test that quantized activations are in valid range."""
        x = torch.randn(2, 10, 64)
        x_quant, eta = quantize_activations_absmax(x, bits=8)
        
        Q_b = 2 ** 7  # 128 for 8-bit
        assert x_quant.min() >= -Q_b
        assert x_quant.max() <= Q_b - 1
    
    def test_per_token_scaling(self):
        """Test that each token has its own scale factor."""
        x = torch.randn(2, 10, 64)
        _, eta = quantize_activations_absmax(x, bits=8)
        
        # eta should have shape (..., 1) for broadcasting
        assert eta.shape == (2, 10, 1)
    
    def test_scale_factor_is_absmax(self):
        """Test that eta is the per-token absmax."""
        x = torch.randn(2, 10, 64)
        _, eta = quantize_activations_absmax(x, bits=8)
        
        expected_eta = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        assert torch.allclose(eta, expected_eta)
    
    def test_gradient_flow_ste(self):
        """Test STE allows gradients to flow."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        x_quant, eta = quantize_activations_absmax(x, bits=8)
        
        loss = x_quant.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_different_bit_widths(self):
        """Test quantization with different bit widths."""
        x = torch.randn(2, 10, 64)
        
        for bits in [4, 8, 16]:
            x_quant, _ = quantize_activations_absmax(x, bits=bits)
            Q_b = 2 ** (bits - 1)
            
            assert x_quant.min() >= -Q_b
            assert x_quant.max() <= Q_b - 1


class TestBitLinear:
    """Tests for BitLinear module."""
    
    def test_output_shape_2d(self):
        """Test BitLinear with 2D input (batch, features)."""
        layer = BitLinear(64, 128)
        x = torch.randn(8, 64)
        y = layer(x)
        
        assert y.shape == (8, 128)
    
    def test_output_shape_3d(self):
        """Test BitLinear with 3D input (batch, seq, features)."""
        layer = BitLinear(64, 128)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        
        assert y.shape == (2, 10, 128)
    
    def test_weight_shape(self):
        """Test weight tensor has correct shape."""
        layer = BitLinear(64, 128)
        assert layer.weight.shape == (128, 64)
    
    def test_no_bias_by_default(self):
        """Test that bias is disabled by default."""
        layer = BitLinear(64, 128)
        assert layer.bias is None
    
    def test_optional_bias(self):
        """Test that bias can be enabled."""
        layer = BitLinear(64, 128, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (128,)
    
    def test_bias_in_output(self):
        """Test that bias affects output."""
        layer_no_bias = BitLinear(64, 128, bias=False)
        layer_with_bias = BitLinear(64, 128, bias=True)
        
        # Copy weights
        layer_with_bias.weight.data.copy_(layer_no_bias.weight.data)
        layer_with_bias.bias.data.fill_(1.0)
        
        x = torch.randn(2, 10, 64)
        y_no_bias = layer_no_bias(x)
        y_with_bias = layer_with_bias(x)
        
        # Output should differ by approximately the bias value
        diff = (y_with_bias - y_no_bias).mean()
        assert diff.abs() > 0.5  # Should be close to 1.0
    
    def test_gradient_flow(self):
        """Test gradients flow through BitLinear."""
        layer = BitLinear(64, 128)
        x = torch.randn(2, 10, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weight.grad is not None
    
    def test_weight_update(self):
        """Test that weights can be updated via gradient descent."""
        layer = BitLinear(64, 128)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        
        initial_weight = layer.weight.clone()
        
        x = torch.randn(2, 10, 64)
        y = layer(x)
        loss = y.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        assert not torch.allclose(layer.weight, initial_weight)
    
    def test_kaiming_initialization(self):
        """Test that weights are initialized with Kaiming uniform."""
        torch.manual_seed(42)
        layer = BitLinear(64, 128)
        
        # Kaiming uniform bounds for fan_in mode
        fan_in = 64
        bound = math.sqrt(3.0 / fan_in)
        
        # Weights should be within bounds
        assert layer.weight.min() >= -bound * 1.1  # Small tolerance
        assert layer.weight.max() <= bound * 1.1
    
    def test_deterministic_forward(self):
        """Test that forward pass is deterministic."""
        layer = BitLinear(64, 128)
        x = torch.randn(2, 10, 64)
        
        y1 = layer(x)
        y2 = layer(x)
        
        assert torch.allclose(y1, y2)
    
    def test_different_bit_widths(self):
        """Test BitLinear with different activation bit widths."""
        for bits in [4, 8, 16]:
            layer = BitLinear(64, 128, bits=bits)
            x = torch.randn(2, 10, 64)
            y = layer(x)
            
            assert y.shape == (2, 10, 128)
            assert not torch.isnan(y).any()
    
    def test_extra_repr(self):
        """Test string representation includes all parameters."""
        layer = BitLinear(64, 128, bias=True, bits=4)
        repr_str = layer.extra_repr()
        
        assert "in_features=64" in repr_str
        assert "out_features=128" in repr_str
        assert "bias=True" in repr_str
        assert "bits=4" in repr_str


class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_rmsnorm_bitlinear_pipeline(self):
        """Test RMSNorm followed by BitLinear."""
        norm = RMSNorm(dim=64)
        linear = BitLinear(64, 128)
        
        x = torch.randn(2, 10, 64)
        x = norm(x)
        y = linear(x)
        
        assert y.shape == (2, 10, 128)
    
    def test_multiple_bitlinear_layers(self):
        """Test stacking multiple BitLinear layers."""
        layers = nn.Sequential(
            BitLinear(64, 128),
            nn.ReLU(),
            BitLinear(128, 256),
            nn.ReLU(),
            BitLinear(256, 64),
        )
        
        x = torch.randn(2, 10, 64)
        y = layers(x)
        
        assert y.shape == (2, 10, 64)
    
    def test_training_loop(self):
        """Test a simple training loop with BitLinear."""
        model = nn.Sequential(
            RMSNorm(dim=64),
            BitLinear(64, 32),
            nn.ReLU(),
            BitLinear(32, 10),
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Simulate a training step
        x = torch.randn(8, 64)
        targets = torch.randint(0, 10, (8,))
        
        initial_loss = None
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, targets)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Loss should generally decrease (not guaranteed but likely)
        # At minimum, training should complete without errors
        assert loss.item() < initial_loss * 2  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
