import pytest
import torch

from bitnet_vision.quant import (
    act_qrange,
    fake_quant_activations,
    fake_quant_weights,
    quantize_activations,
    quantize_weights,
    ste,
)
from tests import reference


def test_weight_codes_are_ternary_and_use_absmean():
    torch.manual_seed(0)
    w = torch.randn(64, 48)
    codes, scale = quantize_weights(w)
    assert set(codes.unique().tolist()) == {-1.0, 0.0, 1.0}
    assert torch.allclose(scale, 1.0 / w.abs().mean())
    # a healthy share of zeros for Gaussian weights (|w| < 0.5 * mean|w| -> 0): roughly 30%
    zero_share = (codes == 0).float().mean().item()
    assert 0.2 < zero_share < 0.45


def test_weight_fake_quant_matches_reference():
    torch.manual_seed(1)
    w = torch.randn(32, 32, dtype=torch.float64)
    assert torch.equal(fake_quant_weights(w), reference.weight_quant(w.clone()))


def test_channel_granularity_uses_one_scale_per_row():
    w = torch.randn(8, 16)
    w[3] *= 100
    codes, scale = quantize_weights(w, granularity="channel")
    assert scale.shape == (8, 1)
    assert torch.allclose(scale[:, 0], 1.0 / w.abs().mean(dim=1))
    assert (codes[3] != 0).any() and (codes[0] != 0).any()


def test_activation_codes_fit_int8_and_max_is_not_clipped():
    torch.manual_seed(0)
    x = torch.randn(4, 10, 64)
    codes, scale = quantize_activations(x)
    lo, hi = act_qrange(8)
    assert (lo, hi) == (-128, 127)
    assert codes.min() >= lo and codes.max() <= hi
    assert torch.equal(codes, codes.round())
    assert scale.shape == (4, 10, 1)
    # the largest |x| of every token maps exactly onto +-127 (the 127 convention of the released code)
    assert torch.equal(codes.abs().amax(-1), torch.full((4, 10), 127.0))


def test_activation_fake_quant_matches_reference():
    torch.manual_seed(2)
    x = torch.randn(3, 7, 33, dtype=torch.float64)
    assert torch.equal(fake_quant_activations(x), reference.activation_quant(x.clone()))


def test_activation_scale_is_per_token():
    x = torch.randn(2, 5, 16)
    x[0, 0] *= 1000
    fq = fake_quant_activations(x)
    # the huge token does not affect the precision of the others
    assert torch.allclose(fq[1], x[1], atol=x[1].abs().max().item() / 127)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_and_extreme_tokens_stay_finite(dtype):
    x = torch.randn(4, 64).to(dtype)
    x[0] = 0
    x[1] = 1e-4
    x[2] *= 1000 if dtype != torch.float16 else 100
    for method, bits in (("absmax", 8), ("absmean", 4)):
        out = fake_quant_activations(x, bits=bits, method=method)
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
        assert torch.equal(out[0], torch.zeros_like(out[0]))


def test_absmean_4bit_range_and_formula():
    torch.manual_seed(0)
    x = torch.randn(6, 128)
    codes, scale = quantize_activations(x, bits=4, method="absmean")
    assert codes.min() >= -8 and codes.max() <= 7
    assert torch.allclose(scale, 7**0.5 / x.abs().mean(-1, keepdim=True))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        quantize_activations(torch.randn(2, 4), bits=1)
    with pytest.raises(ValueError):
        quantize_activations(torch.randn(2, 4), method="minmax")
    with pytest.raises(ValueError):
        quantize_weights(torch.randn(2, 4), granularity="group")


def test_zero_weight_matrix_is_finite_and_trainable():
    w = torch.zeros(4, 4, requires_grad=True)
    wq = ste(w, fake_quant_weights(w))
    assert torch.equal(wq, torch.zeros(4, 4))
    wq.sum().backward()
    assert torch.equal(w.grad, torch.ones(4, 4))  # identity gradient, so an all-zero init can still learn


def test_ste_is_identity_in_backward_and_blends_forward():
    x = torch.randn(5, requires_grad=True)
    q = torch.zeros(5)
    y = ste(x, q, strength=0.25)
    assert torch.allclose(y, 0.75 * x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(5))
