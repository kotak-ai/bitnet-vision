import math

import torch

from bitnet.modules import (
    ActQuantConfig,
    BitLinear,
    RMSNorm,
    _quantize_activations_int,
    _ternarize_absmean,
)


def test_weight_ternary_values_and_scale_positive() -> None:
    torch.manual_seed(0)
    w = torch.randn(7, 11)
    wq, gamma = _ternarize_absmean(w, eps=1e-5)

    # Forward value is quantized (STE), so it should be exact ternary.
    uniques = set(torch.unique(wq).tolist())
    assert uniques.issubset({-1.0, 0.0, 1.0})
    assert gamma.item() > 0.0


def test_activation_quant_int8_range_per_token() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 5, 17)
    cfg = ActQuantConfig(bits=8)
    xq, eta = _quantize_activations_int(x, eps=1e-5, cfg=cfg)

    assert eta.shape == (3, 5, 1)
    assert float(eta.min()) > 0.0
    assert float(xq.min()) >= -float(cfg.q_b)
    assert float(xq.max()) <= float(cfg.q_b - 1)


def test_bitlinear_matches_reference_formula_2d() -> None:
    torch.manual_seed(0)
    layer = BitLinear(13, 7, bias=False, eps=1e-5, activation_bits=8)
    x = torch.randn(4, 13)

    # Reference uses the same internal quantizers to check the formula exactly.
    xq, eta = _quantize_activations_int(x, eps=layer.eps, cfg=layer.act_cfg)
    wq, gamma = _ternarize_absmean(layer.weight, eps=layer.eps)
    y_ref = torch.matmul(xq, wq.t()) * ((gamma * eta) / float(layer.act_cfg.q_b))

    y = layer(x)
    assert torch.allclose(y, y_ref)


def test_bitlinear_matches_reference_formula_3d() -> None:
    torch.manual_seed(0)
    layer = BitLinear(9, 5, bias=False, eps=1e-5, activation_bits=8)
    x = torch.randn(2, 3, 9)

    xq, eta = _quantize_activations_int(x, eps=layer.eps, cfg=layer.act_cfg)
    wq, gamma = _ternarize_absmean(layer.weight, eps=layer.eps)
    y_ref = torch.matmul(xq, wq.t()) * ((gamma * eta) / float(layer.act_cfg.q_b))

    y = layer(x)
    assert torch.allclose(y, y_ref)


def test_bitlinear_backward_produces_finite_grads() -> None:
    torch.manual_seed(0)
    layer = BitLinear(8, 6, bias=True)
    x = torch.randn(4, 8, requires_grad=True)
    y = layer(x).pow(2).mean()
    y.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None and layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()


def test_rmsnorm_matches_definition() -> None:
    torch.manual_seed(0)
    dim = 11
    eps = 1e-6
    n = RMSNorm(dim, eps=eps)
    x = torch.randn(2, 3, dim)

    rms = (x * x).mean(dim=-1, keepdim=True)
    y_ref = x * torch.rsqrt(rms + eps) * n.weight
    y = n(x)
    assert torch.allclose(y, y_ref)

