"""Weight-only packed ternary layer (bitnet_vision/weight_only.py) and the MLX 2-bit encoding of its weights."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bitnet_vision.apple import mlx_ternary as mt
from bitnet_vision.lora import add_lora, lora_layers, merge_lora_into_weights, save_lora, supports_lora
from bitnet_vision.weight_only import PackedTernaryLinear, absmean_ternary, scale_shape, split_ternary

CASES = [("group", 16), ("group", 48), ("row", None), ("tensor", None)]
requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")


def _ternary_linear(k=96, n=37, granularity="group", group_size=16, bias=True, seed=0, dtype=torch.float32):
    """An nn.Linear whose weight is codes * scale, with codes and one positive scale per group drawn at random."""
    g = torch.Generator().manual_seed(seed)
    rows, cols = scale_shape(n, k, granularity, group_size or 16)
    codes = torch.randint(-1, 2, (n, k), generator=g)
    scale = torch.rand(rows, cols, generator=g, dtype=torch.float64) + 0.05
    w = (codes.double().reshape(n, cols, k // cols) * scale.unsqueeze(-1)).reshape(n, k)
    layer = nn.Linear(k, n, bias=bias, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(w.to(dtype))
        if bias:
            layer.bias.copy_(torch.randn(n, generator=g, dtype=torch.float64))
    return layer, codes


@pytest.mark.parametrize("granularity,group_size", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_packed_layer_equals_the_dense_layer_exactly(granularity, group_size, dtype):
    linear, codes = _ternary_linear(granularity=granularity, group_size=group_size, dtype=dtype)
    packed = PackedTernaryLinear.from_linear(linear, group_size=group_size or 16, granularity=granularity)
    assert packed.weight_packed.dtype == torch.uint8 and packed.weight_packed.shape == (10, 96)  # 4 rows per byte
    assert torch.equal(packed.codes(), codes.to(torch.int8))
    assert torch.equal(packed.dequantize(), linear.weight)
    x = torch.randn(3, 5, 96, generator=torch.Generator().manual_seed(1)).to(dtype)
    assert torch.equal(packed(x), linear(x))  # same weight, same F.linear call, so bit for bit
    if dtype != torch.float32:  # a float32 input sees the exact weight too
        assert torch.equal(packed(x.float()), F.linear(x.float(), linear.weight.float(), linear.bias.float()))


def test_dense_cache_is_off_by_default_and_follows_new_weights():
    linear, _ = _ternary_linear(bias=False)
    packed = PackedTernaryLinear.from_linear(linear, group_size=16)
    x = torch.randn(4, 96)
    packed(x)
    assert packed._dense == {}
    cached = PackedTernaryLinear.from_linear(linear, group_size=16, cache_dense=True)
    y = cached(x)
    assert len(cached._dense) == 1 and torch.equal(cached(x), y)
    other, _ = _ternary_linear(bias=False, seed=5)
    cached.load_state_dict(PackedTernaryLinear.from_linear(other, group_size=16).state_dict())
    assert cached._dense == {} and torch.equal(cached(x), other(x))
    cached(x)
    cached.to(torch.float64)  # moving or casting the layer drops the cached copy
    assert cached._dense == {}


def test_split_ternary_refuses_weights_that_are_not_ternary():
    linear, _ = _ternary_linear(bias=False)
    w = linear.weight.detach().clone()
    w[3, 17] = w[3, 16:32].abs().max() * 0.5  # a fourth value in group 1 of row 3
    with pytest.raises(ValueError, match=r"1 groups of 16 weights.*row 3, group 1"):
        split_ternary(w, "group", 16, name="bad")
    with pytest.raises(ValueError, match="does not divide"):
        split_ternary(linear.weight, "group", 40)
    w[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        split_ternary(w, "group", 16)
    zero = torch.zeros(4, 32)  # all-zero groups are ternary with scale 0
    codes, scale = split_ternary(zero, "group", 16)
    assert not codes.any() and not scale.any()


def test_absmean_ternary_rounds_like_bitnet_per_group():
    w = torch.randn(8, 64, generator=torch.Generator().manual_seed(0))
    codes, scale = absmean_ternary(w, "group", 32)
    expected = w.reshape(8, 2, 32).abs().mean(-1)
    assert torch.equal(scale, expected)
    rounded = (w.reshape(8, 2, 32) / expected[..., None]).round().clamp(-1, 1).reshape(8, 64)
    assert torch.equal(codes, rounded.to(torch.int8))


def test_lora_runs_as_a_side_branch_and_merging_is_refused(tmp_path):
    linear, _ = _ternary_linear()
    model = nn.Sequential(PackedTernaryLinear.from_linear(linear, group_size=16))
    assert supports_lora(model[0])
    add_lora(model, r=4, alpha=8, targets=None, dtype=torch.float64)
    a, b = model[0].lora_A["default"].weight, model[0].lora_B["default"].weight
    assert a.dtype == torch.float64
    with torch.no_grad():
        b.normal_()
    x = torch.randn(2, 96, dtype=torch.float64)
    expected = F.linear(x, linear.weight.double(), linear.bias.double()) + 2.0 * (x @ a.T) @ b.T
    torch.testing.assert_close(model(x), expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="packed layer without full-precision weights"):
        merge_lora_into_weights(model)
    pytest.importorskip("safetensors")
    save_lora(model, str(tmp_path))
    assert [n for n, _ in lora_layers(model)] == ["0"]


@pytest.mark.parametrize("granularity,group_size", CASES)
def test_mlx_encoding_round_trips_and_checks_its_input(granularity, group_size):
    linear, codes = _ternary_linear(granularity=granularity, group_size=group_size, bias=False)
    packed = PackedTernaryLinear.from_linear(linear, group_size=group_size or 16, granularity=granularity)
    scale = packed.weight_scale.expand(37, -1) if granularity == "tensor" else packed.weight_scale
    w_q, scales, biases = mt.encode_mlx_ternary(packed.codes(), scale)
    assert w_q.shape == (37, 6) and torch.equal(biases, -scales)
    back, s = mt.decode_mlx_ternary(w_q, scales, biases)
    assert torch.equal(back, codes.to(torch.int8)) and torch.equal(s, scale)
    # word j of a row holds inputs 16j .. 16j + 15, least significant bits first, as code + 1
    q = codes[:, :16] + 1
    word = sum(int(v) << (2 * i) for i, v in enumerate(q[0].tolist()))
    assert int(w_q[0, 0].view(torch.int32) if w_q.dtype != torch.int32 else w_q[0, 0]) & 0xFFFFFFFF == word
    with pytest.raises(ValueError, match="biases differ from -scales"):
        mt.decode_mlx_ternary(w_q, scales, biases * 0.5)
    three = w_q.clone().view(torch.int32) if w_q.dtype != torch.int32 else w_q.clone()
    three[2, 1] |= 3 << 6
    with pytest.raises(ValueError, match="code 3"):
        mt.decode_mlx_ternary(three, scales, biases)


def test_mlx_arrays_of_a_packed_layer_are_exact():
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    k, n = 256, 24
    for granularity, group_size in [("group", 128), ("group", 64), ("group", 256), ("row", None), ("tensor", None)]:
        linear, _ = _ternary_linear(k, n, granularity, group_size, bias=False, dtype=torch.bfloat16)
        packed = PackedTernaryLinear.from_linear(linear, group_size=group_size or 16, granularity=granularity)
        w_q, scales, biases, gs = mt.from_packed_ternary_linear(packed)
        assert scales.dtype == mx.bfloat16 and gs == (64 if group_size == 64 else 128)
        if granularity == "group" and group_size <= 128:  # the same bits as the PyTorch encoder
            assert np.array_equal(np.array(w_q), mt.encode_mlx_ternary(packed.codes(), packed.weight_scale)[0].numpy())
        x = torch.randn(3, k, generator=torch.Generator().manual_seed(2))
        y = mx.quantized_matmul(mt.to_mx(x), w_q, scales.astype(mx.float32), biases.astype(mx.float32), transpose=True,
                                group_size=gs, bits=2)
        torch.testing.assert_close(mt.to_torch(y), x @ linear.weight.float().T, rtol=1e-5, atol=1e-4)


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_layer_on_an_apple_gpu(dtype):
    """On MPS: the 2-bit codes unpack (uint8 shifts) to the same codes, the rebuilt weight is exact, and the layer
    with a float32 LoRA branch gives the CPU's output up to the GPU's summation order."""
    linear, codes = _ternary_linear(dtype=dtype)
    model = nn.Sequential(PackedTernaryLinear.from_linear(linear, group_size=16))
    add_lora(model, r=4, alpha=8, targets=None, dtype=torch.float32)
    with torch.no_grad():
        model[0].lora_B["default"].weight.normal_(generator=torch.Generator().manual_seed(3))
    x = torch.randn(3, 5, 96, generator=torch.Generator().manual_seed(1)).to(dtype)
    expected = model(x)
    gpu = copy.deepcopy(model).to("mps")
    assert gpu[0].weight_packed.device.type == "mps"
    assert torch.equal(gpu[0].codes().cpu(), codes.to(torch.int8))
    assert torch.equal(gpu[0].dequantize(dtype).cpu(), linear.weight.detach())
    got = gpu(x.to("mps"))
    assert got.dtype == dtype
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(got.cpu(), expected, rtol=tol, atol=tol)
