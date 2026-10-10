import copy
import io
import pickle

import pytest
import torch
from torch import nn

from bitnet_vision.hadamard import hadamard_matrix
from bitnet_vision.layers import (
    BitLinear,
    HBitLinear,
    RMSNorm,
    bitlinear_modules,
    convert_linear_layers,
    set_quant_strength,
)
from tests import reference


def _random_layer(cls=BitLinear, d_in=64, d_out=48, bias=True, **kw):
    layer = cls(d_in, d_out, bias=bias, **kw).double()
    with torch.no_grad():
        if bias:
            layer.bias.normal_()
        layer.norm.weight.uniform_(0.5, 1.5)
    return layer


@pytest.mark.parametrize("shape", [(5, 64), (2, 7, 64), (2, 3, 4, 64), (2, 1, 3, 4, 64)])
@pytest.mark.parametrize("bias", [True, False])
def test_bitlinear_matches_reference_outputs_and_gradients(shape, bias):
    """Forward values AND every gradient must equal Microsoft's reference recipe (float64, exact)."""
    torch.manual_seed(0)
    layer = _random_layer(bias=bias)
    w = layer.weight.detach().clone().requires_grad_()
    b = layer.bias.detach().clone().requires_grad_() if bias else None
    g = layer.norm.weight.detach().clone().requires_grad_()
    x1 = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()

    y1 = layer(x1)
    y2 = reference.bitlinear_forward(x2, w, b, g)
    up = torch.randn_like(y1)
    (y1 * up).sum().backward()
    (y2 * up).sum().backward()

    torch.testing.assert_close(y1, y2, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer.weight.grad, w.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer.norm.weight.grad, g.grad, rtol=1e-12, atol=1e-12)
    if bias:
        torch.testing.assert_close(layer.bias.grad, b.grad, rtol=1e-12, atol=1e-12)


def test_gradients_match_reference_after_training():
    """Bugs that leak gradient into the scales look fine at init but drift during training; train first."""
    torch.manual_seed(0)
    layer = _random_layer(d_in=32, d_out=8)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(100):
        x = torch.randn(64, 32, dtype=torch.float64)
        loss = (layer(x) - x[:, :8]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = layer.weight.detach().clone().requires_grad_()
    x = torch.randn(64, 32, dtype=torch.float64)
    layer.zero_grad()
    (layer(x) ** 2).sum().backward()
    (reference.bitlinear_forward(x, w, layer.bias.detach(), layer.norm.weight.detach()) ** 2).sum().backward()
    torch.testing.assert_close(layer.weight.grad, w.grad, rtol=1e-10, atol=1e-12)


def test_bias_is_added_after_rescaling():
    torch.manual_seed(0)
    layer = BitLinear(32, 16, bias=True)
    x = torch.randn(5, 32)
    with torch.no_grad():
        layer.bias.fill_(5.0)
    with_bias = layer(x)
    with torch.no_grad():
        layer.bias.zero_()
    torch.testing.assert_close(with_bias - layer(x), torch.full((5, 16), 5.0))


def test_every_weight_receives_gradient():
    """Clamping outside the straight-through estimator freezes weights at +-1 (old branch 6825)."""
    torch.manual_seed(0)
    layer = BitLinear(64, 32)
    layer(torch.randn(16, 64)).pow(2).sum().backward()
    assert (layer.weight.grad != 0).float().mean() > 0.99


def test_input_norm_makes_layer_scale_invariant():
    torch.manual_seed(0)
    layer = BitLinear(64, 32)
    x = torch.randn(3, 64)
    torch.testing.assert_close(layer(10 * x), layer(x))


def test_quant_strength_zero_is_full_precision():
    torch.manual_seed(0)
    layer = BitLinear(64, 32)
    layer.quant_strength = 0.0
    x = torch.randn(3, 64)
    torch.testing.assert_close(layer(x), nn.functional.linear(layer.norm(x), layer.weight, layer.bias))


def test_set_quant_strength_and_iteration():
    model = nn.Sequential(BitLinear(8, 8), nn.ReLU(), HBitLinear(8, 4))
    assert len(list(bitlinear_modules(model))) == 2
    set_quant_strength(model, 0.3)
    assert all(m.quant_strength == 0.3 for m in bitlinear_modules(model))
    with pytest.raises(ValueError):
        set_quant_strength(model, 1.5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_is_finite_for_zero_and_large_tokens(dtype):
    torch.manual_seed(0)
    layer = BitLinear(1024, 256).to(dtype)
    x = torch.randn(4, 1024).to(dtype)
    x[0] = 0
    x[1, 0] = 300.0  # >= 256: squaring overflows float16 unless the norm upcasts
    x[2] *= 50
    y = layer(x)
    assert y.dtype == dtype and torch.isfinite(y).all()
    ref = BitLinear(1024, 256)
    ref.load_state_dict({k: v.float() for k, v in layer.state_dict().items()})
    y32 = ref(x.float())
    assert (y.float() - y32).abs().max() < 0.05 * y32.abs().max()


def test_rmsnorm_float16_large_values():
    norm = RMSNorm(8).half()
    x = torch.full((1, 8), 400.0, dtype=torch.float16)
    torch.testing.assert_close(norm(x), torch.ones(1, 8, dtype=torch.float16))


def test_empty_and_noncontiguous_inputs():
    layer = BitLinear(64, 32)
    assert layer(torch.randn(0, 64)).shape == (0, 32)
    assert layer(torch.randn(2, 0, 64)).shape == (2, 0, 32)
    x = torch.randn(64, 10).t()
    torch.testing.assert_close(layer(x), layer(x.contiguous()))


def test_nn_linear_compatible_signature_and_state_dict():
    layer = BitLinear(16, 8, True, "cpu", torch.float64)
    assert layer.weight.dtype == torch.float64 and layer.bias is not None
    assert set(layer.state_dict()) == {"weight", "bias", "norm.weight"}
    buf = io.BytesIO()
    torch.save(layer.state_dict(), buf)
    buf.seek(0)
    other = BitLinear(16, 8).double()
    other.load_state_dict(torch.load(buf, weights_only=True))
    x = torch.randn(3, 16, dtype=torch.float64)
    torch.testing.assert_close(other(x), layer(x))
    clone = pickle.loads(pickle.dumps(copy.deepcopy(layer)))
    torch.testing.assert_close(clone(x), layer(x))


def test_from_linear_copies_weights():
    lin = nn.Linear(32, 16)
    layer = BitLinear.from_linear(lin)
    assert torch.equal(layer.weight, lin.weight) and torch.equal(layer.bias, lin.bias)


def test_hbitlinear_from_linear_preserves_function_before_quantization():
    torch.manual_seed(0)
    lin = nn.Linear(192, 50).double()
    layer = HBitLinear.from_linear(lin, input_norm=False)
    layer.quant_strength = 0.0
    x = torch.randn(4, 192, dtype=torch.float64)
    torch.testing.assert_close(layer(x), lin(x))


@pytest.mark.parametrize("width", [64, 192, 768])
@pytest.mark.parametrize("signs", [True, False])
def test_hbitlinear_matches_explicit_hadamard_then_reference(width, signs):
    torch.manual_seed(0)
    layer = _random_layer(HBitLinear, d_in=width, d_out=24, random_signs=signs)
    h = layer.hadamard.matrix()
    if not signs:
        torch.testing.assert_close(h, hadamard_matrix(width))
    x = torch.randn(3, 5, width, dtype=torch.float64)
    xn = reference.rms_norm(x, layer.norm.weight.detach()) @ h
    xq = reference.activation_quant(xn)
    wq = reference.weight_quant(layer.weight.detach().clone())
    expected = nn.functional.linear(xq, wq, layer.bias.detach())
    torch.testing.assert_close(layer(x), expected, rtol=1e-9, atol=1e-9)


def test_hbitlinear_4bit_defaults_to_absmean():
    layer = HBitLinear(64, 32, act_bits=4)
    assert layer.act_method == "absmean"
    assert HBitLinear(64, 32).act_method == "absmax"


def test_convert_linear_layers_respects_exclude():
    model = nn.Sequential()
    model.add_module("body", nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)))
    model.add_module("head", nn.Linear(8, 2))
    convert_linear_layers(model, exclude=["head"])
    assert isinstance(model.body[0], BitLinear) and isinstance(model.body[2], BitLinear)
    assert type(model.head) is nn.Linear


def test_invalid_layer_arguments():
    with pytest.raises(ValueError):
        BitLinear(8, 8, act_bits=1)
    with pytest.raises(ValueError):
        BitLinear(8, 8, weight_granularity="group")


def test_deadzone_term_adds_row_sums_of_zero_coded_weights():
    torch.manual_seed(0)
    layer = BitLinear(64, 16, bias=False, deadzone_lambda=1e-2).double()
    plain = BitLinear(64, 16, bias=False).double()
    plain.load_state_dict(layer.state_dict())
    x = torch.randn(5, 64, dtype=torch.float64)
    codes, _ = layer.ternary_weight()
    dead = (codes == 0).double()
    expected_extra = 1e-2 * (layer.weight.detach() * dead).sum(1)
    torch.testing.assert_close(layer(x) - plain(x), expected_extra.expand(5, -1))
    # dead weights get an extra direct gradient of lambda * sum(dL/dy) over tokens
    layer(x).sum().backward()
    plain(x).sum().backward()
    torch.testing.assert_close(layer.weight.grad - plain.weight.grad, 1e-2 * 5 * dead)
