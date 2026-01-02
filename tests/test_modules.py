"""
Unit tests for BitNet b1.58 and BitNet v2 modules.
"""

import math
import pytest
import torch
import torch.nn as nn

from bitnet import (
    BitLinear,
    HBitLinear,
    RMSNorm,
    fast_walsh_hadamard_transform,
    fast_walsh_hadamard_transform_vectorized,
    generate_hadamard_matrix,
    hadamard_transform,
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


class TestHadamardMatrix:
    """Tests for Hadamard matrix generation."""
    
    def test_size_2(self):
        """Test H_2 matrix."""
        H = generate_hadamard_matrix(2)
        expected = torch.tensor([[1., 1.], [1., -1.]])
        assert torch.allclose(H, expected)
    
    def test_size_4(self):
        """Test H_4 matrix."""
        H = generate_hadamard_matrix(4)
        assert H.shape == (4, 4)
        # Check all entries are +1 or -1
        assert torch.allclose(H.abs(), torch.ones(4, 4))
    
    def test_orthogonality(self):
        """Test H @ H^T = n * I."""
        for n in [2, 4, 8, 16, 32]:
            H = generate_hadamard_matrix(n)
            HHT = H @ H.T
            expected = n * torch.eye(n)
            assert torch.allclose(HHT, expected)
    
    def test_symmetry(self):
        """Test that H = H^T (symmetric)."""
        for n in [2, 4, 8, 16]:
            H = generate_hadamard_matrix(n)
            assert torch.allclose(H, H.T)
    
    def test_non_power_of_2_raises(self):
        """Test that non-power-of-2 raises ValueError."""
        with pytest.raises(ValueError):
            generate_hadamard_matrix(3)
        with pytest.raises(ValueError):
            generate_hadamard_matrix(5)
        with pytest.raises(ValueError):
            generate_hadamard_matrix(7)
    
    def test_device_dtype(self):
        """Test matrix generation with specific device and dtype."""
        H = generate_hadamard_matrix(4, dtype=torch.float64)
        assert H.dtype == torch.float64


class TestFastWalshHadamardTransform:
    """Tests for Fast Walsh-Hadamard Transform."""
    
    def test_output_shape(self):
        """Test FWHT preserves shape."""
        x = torch.randn(2, 8)
        y = fast_walsh_hadamard_transform_vectorized(x)
        assert y.shape == x.shape
    
    def test_output_shape_3d(self):
        """Test FWHT with 3D input."""
        x = torch.randn(2, 10, 16)
        y = fast_walsh_hadamard_transform_vectorized(x)
        assert y.shape == x.shape
    
    def test_self_inverse(self):
        """Test normalized FWHT is its own inverse."""
        x = torch.randn(4, 32)
        y = fast_walsh_hadamard_transform_vectorized(x, normalize=True)
        x_recovered = fast_walsh_hadamard_transform_vectorized(y, normalize=True)
        assert torch.allclose(x, x_recovered, atol=1e-5)
    
    def test_equivalence_to_matrix(self):
        """Test FWHT gives same result as matrix multiplication."""
        n = 8
        x = torch.randn(4, n)
        H = generate_hadamard_matrix(n)
        
        # Matrix multiplication (normalized)
        y_matrix = (x @ H) / math.sqrt(n)
        
        # FWHT
        y_fwht = fast_walsh_hadamard_transform_vectorized(x, normalize=True)
        
        assert torch.allclose(y_matrix, y_fwht, atol=1e-5)
    
    def test_non_power_of_2_raises(self):
        """Test FWHT raises for non-power-of-2 dimensions."""
        x = torch.randn(4, 5)
        with pytest.raises(ValueError):
            fast_walsh_hadamard_transform_vectorized(x)
    
    def test_basic_vs_vectorized(self):
        """Test basic and vectorized implementations match."""
        x = torch.randn(2, 16)
        y_basic = fast_walsh_hadamard_transform(x, normalize=True)
        y_vec = fast_walsh_hadamard_transform_vectorized(x, normalize=True)
        assert torch.allclose(y_basic, y_vec, atol=1e-5)
    
    def test_gradient_flow(self):
        """Test gradients flow through Hadamard transform."""
        x = torch.randn(2, 8, requires_grad=True)
        y = hadamard_transform(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_gradient_correctness(self):
        """Test backward pass uses Hadamard transform correctly."""
        x = torch.randn(2, 8, requires_grad=True)
        y = hadamard_transform(x)
        
        # Manual gradient computation: for orthonormal H, grad_x = H @ grad_y
        grad_y = torch.ones_like(y)
        y.backward(grad_y)
        
        # grad_x should be H @ grad_y / sqrt(n), which equals
        # hadamard_transform(grad_y) since it's self-inverse
        expected_grad = fast_walsh_hadamard_transform_vectorized(grad_y, normalize=True)
        assert torch.allclose(x.grad, expected_grad, atol=1e-5)
    
    def test_reduces_outliers(self):
        """Test Hadamard transform reduces activation outliers."""
        # Create input with outliers
        x = torch.randn(4, 64)
        x[:, 0] *= 10  # Create outliers
        
        original_absmax = x.abs().max(dim=-1).values
        
        x_h = fast_walsh_hadamard_transform_vectorized(x, normalize=True)
        hadamard_absmax = x_h.abs().max(dim=-1).values
        
        # Hadamard should generally reduce the maximum value
        assert (hadamard_absmax < original_absmax).all()


class TestHBitLinear:
    """Tests for H-BitLinear (BitNet v2) module."""
    
    def test_output_shape_2d(self):
        """Test HBitLinear with 2D input."""
        layer = HBitLinear(64, 128)
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 128)
    
    def test_output_shape_3d(self):
        """Test HBitLinear with 3D input."""
        layer = HBitLinear(64, 128)
        x = torch.randn(2, 10, 64)
        y = layer(x)
        assert y.shape == (2, 10, 128)
    
    def test_power_of_2_no_padding(self):
        """Test power-of-2 dimension doesn't need padding."""
        layer = HBitLinear(64, 128)
        assert layer.padded_in_features == 64
        assert not layer._needs_padding
    
    def test_non_power_of_2_padding(self):
        """Test non-power-of-2 dimension is padded."""
        layer = HBitLinear(50, 100)
        assert layer.padded_in_features == 64  # Next power of 2
        assert layer._needs_padding
        
        layer = HBitLinear(100, 200)
        assert layer.padded_in_features == 128
    
    def test_inherits_from_bitlinear(self):
        """Test HBitLinear inherits from BitLinear."""
        layer = HBitLinear(64, 128)
        assert isinstance(layer, BitLinear)
    
    def test_weight_shape(self):
        """Test weight tensor has correct shape."""
        layer = HBitLinear(64, 128)
        assert layer.weight.shape == (128, 64)
    
    def test_no_bias_by_default(self):
        """Test bias is disabled by default."""
        layer = HBitLinear(64, 128)
        assert layer.bias is None
    
    def test_optional_bias(self):
        """Test bias can be enabled."""
        layer = HBitLinear(64, 128, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (128,)
    
    def test_gradient_flow(self):
        """Test gradients flow through HBitLinear."""
        layer = HBitLinear(64, 128)
        x = torch.randn(2, 10, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(layer.weight.grad).any()
    
    def test_gradient_flow_non_power_of_2(self):
        """Test gradients flow with non-power-of-2 dimensions."""
        layer = HBitLinear(50, 100)
        x = torch.randn(2, 10, 50, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == (2, 10, 50)
        assert not torch.isnan(x.grad).any()
    
    def test_weight_update(self):
        """Test weights can be updated via gradient descent."""
        layer = HBitLinear(64, 128)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        
        initial_weight = layer.weight.clone()
        
        x = torch.randn(2, 10, 64)
        y = layer(x)
        loss = y.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not torch.allclose(layer.weight, initial_weight)
    
    def test_deterministic_forward(self):
        """Test forward pass is deterministic."""
        layer = HBitLinear(64, 128)
        x = torch.randn(2, 10, 64)
        
        y1 = layer(x)
        y2 = layer(x)
        
        assert torch.allclose(y1, y2)
    
    def test_extra_repr(self):
        """Test string representation includes padded dimension."""
        layer = HBitLinear(50, 100, bits=4)
        repr_str = layer.extra_repr()
        
        assert "in_features=50" in repr_str
        assert "out_features=100" in repr_str
        assert "padded_in_features=64" in repr_str
        assert "bits=4" in repr_str
    
    def test_different_bit_widths(self):
        """Test HBitLinear with different activation bit widths."""
        for bits in [4, 8, 16]:
            layer = HBitLinear(64, 128, bits=bits)
            x = torch.randn(2, 10, 64)
            y = layer(x)
            
            assert y.shape == (2, 10, 128)
            assert not torch.isnan(y).any()


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
    
    def test_rmsnorm_hbitlinear_pipeline(self):
        """Test RMSNorm followed by HBitLinear."""
        norm = RMSNorm(dim=64)
        linear = HBitLinear(64, 128)
        
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
    
    def test_multiple_hbitlinear_layers(self):
        """Test stacking multiple HBitLinear layers."""
        layers = nn.Sequential(
            HBitLinear(64, 128),
            nn.SiLU(),
            HBitLinear(128, 256),
            nn.SiLU(),
            HBitLinear(256, 64),
        )
        
        x = torch.randn(2, 10, 64)
        y = layers(x)
        
        assert y.shape == (2, 10, 64)
    
    def test_mixed_layers(self):
        """Test mixing BitLinear and HBitLinear layers."""
        layers = nn.Sequential(
            BitLinear(64, 128),
            nn.ReLU(),
            HBitLinear(128, 128),
            nn.ReLU(),
            BitLinear(128, 64),
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
    
    def test_training_loop_hbitlinear(self):
        """Test training loop with HBitLinear."""
        model = nn.Sequential(
            RMSNorm(dim=64),
            HBitLinear(64, 32),
            nn.SiLU(),
            HBitLinear(32, 10),
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
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
        
        assert loss.item() < initial_loss * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
