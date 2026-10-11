"""Fused BitLinear front end and epilogue for Apple GPUs (bitnet_vision.apple.mps_kernels).

The reference paths run everywhere and must equal the repo's own layers exactly. Tests marked with
``requires_metal`` compare the Metal kernels with those references; they run only on a Mac with an MPS device
and PyTorch 2.7 or newer.

Tolerance for the Metal front end with RMSNorm: the kernel adds up the squares in a different order (strided
per-thread partial sums, then simd_sum) than the CPU, so the token's RMS can differ by a few float32 ulps.
That scale cancels between a value and its token's absmax, so codes almost never change; when one does, it
lands within 1 of the reference. We allow codes at most 1 apart from the reference values requantized with
the kernel's scale, in at most 1 in 1000 entries (at least 2 per tensor), and scales within 1e-6 relative. For
float16 input the normalized values are rounded to half precision (as RMSNorm does), and a few-ulp change can
flip that rounding for the token's largest value, so its scale may differ by up to one half-precision step
(2^-10 relative). Without the norm every step is a single correctly rounded float32 operation done in the same
order as on the CPU, so the result should be bit-exact; a dedicated test checks that.
"""

import math
import re

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bitnet_vision.apple import mps_kernels
from bitnet_vision.apple.mps_kernels import (
    METAL_SOURCE,
    FusedBitLinearInference,
    convert_to_fused,
    fused_epilogue,
    fused_front_end,
    metal_available,
)
from bitnet_vision.export import BitLinearInference, convert_for_inference, unpack_ternary
from bitnet_vision.layers import BitLinear, HBitLinear, RMSNorm
from bitnet_vision.models import create_model
from bitnet_vision.quant import quantize_activations

requires_metal = pytest.mark.skipif(not metal_available(), reason="needs an Apple GPU and torch.mps.compile_shader")


def _tokens(m: int, k: int, dtype: torch.dtype = torch.float32, seed: int = 0) -> torch.Tensor:
    """Random tokens with very different sizes per row, an all-zero row and a row with one huge value."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g) * torch.exp(torch.randn(m, 1, generator=g))
    if m >= 3:
        x[0] = 0
        x[1, k // 3] = 1e4
        if dtype == torch.float32:
            x[2] *= 1e-6  # absmax below the 1e-5 floor (in float16 these would be subnormal)
    return x.to(dtype)


def _norm_weight(k: int, seed: int = 1) -> torch.Tensor:
    return 1 + 0.3 * torch.randn(k, generator=torch.Generator().manual_seed(seed))


def _inference_layer(cls=BitLinear, k: int = 192, n: int = 50, seed: int = 0, **kwargs) -> BitLinearInference:
    torch.manual_seed(seed)
    layer = cls(k, n, **kwargs)
    with torch.no_grad():
        if layer.bias is not None:
            layer.bias.normal_()
        if layer.norm is not None:
            layer.norm.weight.uniform_(0.5, 1.5)
    return BitLinearInference.from_bitlinear(layer)


# ---------------------------------------------------------------------------------------------------------
# Reference paths (run everywhere)


def test_metal_source_is_ascii():
    assert METAL_SOURCE.isascii()  # PyTorch passes the source to Metal as an ASCII string


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_norm", [False, True])
def test_front_end_reference_equals_norm_then_quantize(dtype, use_norm):
    x = _tokens(10, 192, dtype).reshape(2, 5, 192)
    norm = RMSNorm(192, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(_norm_weight(192))
    codes, scale = fused_front_end(x, norm.weight if use_norm else None, norm.eps)
    expected_codes, expected_scale = quantize_activations(norm(x) if use_norm else x, bits=8, method="absmax")
    assert codes.dtype == scale.dtype == torch.float32
    assert codes.shape == (2, 5, 192) and scale.shape == (2, 5, 1)
    assert torch.equal(codes, expected_codes) and torch.equal(scale, expected_scale)


def test_front_end_other_bits_and_shapes():
    x = _tokens(6, 384)
    for bits in (4, 6):
        assert all(map(torch.equal, fused_front_end(x, None, 0.0, bits=bits), quantize_activations(x, bits=bits)))
    xt = x.t()[:, :4].t()  # non-contiguous rows
    assert all(map(torch.equal, fused_front_end(xt, None, 0.0), quantize_activations(xt)))
    codes, scale = fused_front_end(torch.zeros(0, 384), _norm_weight(384), 1e-6)
    assert codes.shape == (0, 384) and scale.shape == (0, 1)
    codes, scale = fused_front_end(x[0], None, 0.0)
    assert codes.shape == (384,) and scale.shape == (1,)
    with pytest.raises(ValueError):
        fused_front_end(x, torch.ones(383), 1e-6)


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_epilogue_reference_uses_inference_layer_order(per_channel, bias):
    g = torch.Generator().manual_seed(0)
    acc = torch.randint(-20000, 20000, (3, 7, 50), generator=g).float()
    x_scale = torch.rand(3, 7, 1, generator=g) * 50 + 1e-3
    w_scale = torch.rand(50 if per_channel else 1, generator=g) * 0.05
    b = torch.randn(50, generator=g) if bias else None
    expected = acc / x_scale * w_scale  # the expression in BitLinearInference.forward
    if bias:
        expected = expected + b
    assert torch.equal(fused_epilogue(acc, x_scale, w_scale, b), expected)


LAYER_VARIANTS = [
    (BitLinear, {}),
    (BitLinear, {"bias": False}),
    (BitLinear, {"weight_granularity": "channel"}),
    (BitLinear, {"input_norm": False}),
    (BitLinear, {"bias": False, "deadzone_lambda": 1e-2}),
    (HBitLinear, {}),
    (HBitLinear, {"input_norm": False, "weight_granularity": "channel"}),
    (HBitLinear, {"random_signs": False, "bias": False}),
]


@pytest.mark.parametrize("cls,kwargs", LAYER_VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_layer_equals_inference_layer(cls, kwargs, dtype):
    inf = _inference_layer(cls, **kwargs)
    fused = FusedBitLinearInference.from_inference(inf)
    x = (3 * torch.randn(3, 9, 192)).to(dtype)
    x[0, 0] = 0
    with torch.no_grad():
        y = fused(x)
        assert y.dtype == dtype and torch.equal(y, inf(x))
    assert torch.equal(fused(x), inf(x))  # autograd on: same reference ops


@pytest.mark.parametrize("cls,kwargs", [
    (HBitLinear, {"act_bits": 4}),
    (BitLinear, {"act_bits": None}),
    (BitLinear, {"act_bits": 6, "act_method": "absmax"}),
])
def test_fused_layer_falls_back_for_other_activation_settings(cls, kwargs):
    inf = _inference_layer(cls, **kwargs)
    fused = FusedBitLinearInference.from_inference(inf)
    x = torch.randn(4, 192, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(fused(x), inf(x))


def test_fused_layer_float64_and_odd_shapes():
    inf = _inference_layer()
    fused = FusedBitLinearInference.from_inference(inf)
    with torch.no_grad():
        for x in (torch.randn(5, 192, dtype=torch.float64), torch.randn(192), torch.zeros(0, 192),
                  torch.randn(2, 0, 192)):
            y = fused(x)
            assert y.shape == x.shape[:-1] + (50,) and y.dtype == x.dtype and torch.equal(y, inf(x))


def test_fused_layer_shares_weights_and_caches_float_codes():
    inf = _inference_layer(cls=HBitLinear)
    fused = FusedBitLinearInference.from_inference(inf)
    assert fused.weight_packed is inf.weight_packed and fused.norm is inf.norm and fused.hadamard is inf.hadamard
    assert list(fused.state_dict()) == list(inf.state_dict())
    codes = fused._float_codes()
    assert codes is fused._float_codes()  # unpacked once
    assert codes.dtype == torch.float32 and torch.equal(codes, unpack_ternary(inf.weight_packed, 50).float())

    other = _inference_layer(cls=HBitLinear, seed=1)
    fused.load_state_dict(other.state_dict())  # new weights must replace the cached codes
    x = torch.randn(4, 192, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(fused(x), other(x))


@pytest.mark.parametrize("overrides", [
    {}, {"hadamard": True}, {"weight_granularity": "channel"}, {"hadamard": True, "hadamard_act_bits": 4},
])
def test_convert_to_fused_matches_convert_for_inference(overrides):
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", **overrides).eval()
    packed = convert_for_inference(model)
    fused = convert_to_fused(packed)
    assert not any(isinstance(m, FusedBitLinearInference) for m in packed.modules())  # input left alone
    assert sum(isinstance(m, FusedBitLinearInference) for m in fused.modules()) == 24
    assert all(isinstance(m, FusedBitLinearInference) for m in fused.modules() if isinstance(m, BitLinearInference))
    assert list(fused.state_dict()) == list(packed.state_dict())
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(fused(x), packed(x))


def test_convert_to_fused_exports_training_layers_and_works_in_place():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", hadamard=True).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = convert_for_inference(model)(x)
        fused = convert_to_fused(model)
        assert any(isinstance(m, BitLinear) for m in model.modules())  # not in place: model unchanged
        assert torch.equal(fused(x), expected)
        packed = convert_for_inference(model)
        assert convert_to_fused(packed, inplace=True) is packed
        assert torch.equal(packed(x), expected)


def test_convert_to_fused_keeps_shared_layers_shared():
    shared = _inference_layer(k=16, n=16)
    model = nn.Sequential(shared, nn.Sequential(shared), shared)
    fused = convert_to_fused(model)
    assert isinstance(fused[0], FusedBitLinearInference)
    assert fused[0] is fused[1][0] is fused[2]
    assert model[0] is shared  # not in place
    single = convert_to_fused(shared)
    assert isinstance(single, FusedBitLinearInference) and single.weight_packed is not shared.weight_packed
    assert isinstance(convert_to_fused(BitLinear(16, 8)), FusedBitLinearInference)


def _buffer_args(kernel: str) -> list[tuple[str, str]]:
    """``(type, name)`` of the ``[[buffer(i)]]`` arguments of ``kernel`` in METAL_SOURCE, in index order."""
    signature = re.search(rf"kernel void {kernel}\((.*?)\)\s*\{{", METAL_SOURCE, re.S).group(1)
    args = re.findall(r"((?:device|constant)[\w\s]*?[*&])\s*(\w+)\s*\[\[buffer\((\d+)\)\]\]", signature)
    assert [int(i) for *_, i in args] == list(range(len(args)))
    return [(" ".join(ty.split()), name) for ty, name, _ in args]


class _FakeMetalKernel:
    """Stands in for a compiled kernel: checks a call against the Metal signature the way torch.mps binds it
    (tensor -> device buffer of that element type, Python int -> 8-byte ``long``, float -> ``float``), checks the
    dispatch geometry, then computes the kernel's result from the arguments it found under each Metal name."""

    thread_execution_width = 32
    max_threads_per_threadgroup = 1024

    def __init__(self, name: str) -> None:
        self.name, self.args = name, _buffer_args(name)

    def __call__(self, *args, threads, group_size):
        assert len(args) == len(self.args), self.name
        a = {}
        for (ty, name), value in zip(self.args, args):
            if ty.startswith("device"):
                dtype = {"float*": torch.float32, "half*": torch.float16}[ty.split()[-1]]
                assert isinstance(value, torch.Tensor) and value.dtype == dtype and value.is_contiguous(), name
            else:
                assert type(value) is {"constant long&": int, "constant float&": float}[ty], name
            a[name] = value
        if self.name == "bitnet_epilogue":
            m, n = a["M"], a["N"]
            assert threads == [n, m] and group_size[1] == 1 and 1 <= group_size[0] <= 1024
            y = a["acc"].view(m, n) / a["x_scale"].view(m, 1) * a["w_scale"][torch.arange(n) * a["w_step"]]
            a["out"].view(m, n).copy_(y + a["bias"].view(n) if a["has_bias"] else y)
            return
        k, m = a["K"], a["scales"].numel()
        assert a["x"].numel() == a["codes"].numel() == m * k
        assert group_size % 32 == 0 and 32 <= group_size <= 1024 and threads == m * group_size  # one group per row
        h = a["x"].view(m, k)
        if a["use_norm"]:
            h = mps_kernels._rms_norm(h, a["norm_w"].view(k), a["eps"])
        codes, scale = quantize_activations(h, eps=a["qeps"])
        a["codes"].view(m, k).copy_(codes)
        a["scales"].view(m, 1).copy_(scale)


def test_kernel_calls_match_metal_signatures(monkeypatch):
    # Runs everywhere: the Python wrappers drive fake kernels that check each call against METAL_SOURCE, so the
    # Metal path's argument order, types, shapes and dispatch geometry are tested without a Mac.
    used = set()

    class FakeLibrary:
        def __getattr__(self, name):
            used.add(name)
            return _FakeMetalKernel(name)

    monkeypatch.setattr(torch.mps, "compile_shader", lambda source: FakeLibrary(), raising=False)
    monkeypatch.setattr(mps_kernels, "_library", None)
    monkeypatch.setattr(mps_kernels, "_kernels", {})
    monkeypatch.setattr(mps_kernels, "_use_metal", lambda *tensors: True)
    for dtype in (torch.float32, torch.float16):
        x = _tokens(10, 100, dtype)  # an all-zero row and a tiny one: codes depend on eps and qeps
        for xi in (x.reshape(2, 5, 100), x.reshape(5, 2, 100).transpose(0, 1), x[3]):
            for w in (None, _norm_weight(100)):
                got, ref = fused_front_end(xi, w, 1e-6), mps_kernels._front_end_reference(xi, w, 1e-6, 8)
                assert all(map(torch.equal, got, ref))
    acc = torch.randint(-20000, 20000, (3, 7, 50), generator=torch.Generator().manual_seed(0)).float()
    x_scale = torch.rand(3, 7, 1) + 0.5
    for w_scale, bias in ((torch.rand(1), None), (torch.rand(50), torch.randn(50)), (torch.rand(50), None)):
        assert torch.equal(fused_epilogue(acc, x_scale, w_scale, bias),
                           mps_kernels._epilogue_reference(acc, x_scale, w_scale, bias))
    for cls, kwargs in LAYER_VARIANTS:
        inf = _inference_layer(cls, **kwargs)
        x = 3 * torch.randn(3, 9, 192)
        with torch.no_grad():
            assert torch.equal(FusedBitLinearInference.from_inference(inf)(x), inf(x))
    assert used == {"bitnet_front_end_float", "bitnet_front_end_half", "bitnet_epilogue"}


def test_compile_failure_is_reported_as_runtime_error(monkeypatch):
    def broken(source):
        raise SyntaxError("program_source:12:3: error: use of undeclared identifier 'shared'")

    monkeypatch.setattr(torch.mps, "compile_shader", broken, raising=False)
    monkeypatch.setattr(mps_kernels, "_library", None)
    monkeypatch.setattr(mps_kernels, "_kernels", {})
    with pytest.raises(RuntimeError, match="undeclared identifier 'shared'"):
        mps_kernels._kernel("bitnet_epilogue")


# ---------------------------------------------------------------------------------------------------------
# Metal kernels against the references (Apple GPUs only)


def _assert_front_end_close(codes, scale, x, norm_weight, eps):
    """Metal (codes, scale) against the CPU reference, with the tolerance in the module docstring."""
    codes, scale = codes.cpu(), scale.cpu()
    ref_codes, ref_scale = fused_front_end(x, norm_weight, eps)
    assert codes.shape == ref_codes.shape and scale.shape == ref_scale.shape
    if x.numel() == 0:
        return
    rtol = 2**-10 if x.dtype == torch.float16 and norm_weight is not None else 1e-6
    torch.testing.assert_close(scale, ref_scale, rtol=rtol, atol=0)
    normed = (mps_kernels._rms_norm(x, norm_weight, eps) if norm_weight is not None else x).float()
    expected = (normed * scale).round().clamp(-128, 127)
    diff = (codes - expected).abs()
    assert diff.max().item() <= 1
    assert (diff > 0).sum().item() <= max(2, diff.numel() // 1000)


@requires_metal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("use_norm", [False, True])
@pytest.mark.parametrize("k", [192, 384, 768, 1536, 4096])
@pytest.mark.parametrize("m", [0, 1, 74, 266, 1000])
def test_metal_front_end_matches_reference(m, k, dtype, use_norm):
    x = _tokens(m, k, dtype, seed=m + k)
    w = _norm_weight(k) if use_norm else None
    codes, scale = fused_front_end(x.to("mps"), None if w is None else w.to("mps"), 1e-6)
    _assert_front_end_close(codes, scale, x, w, 1e-6)


@requires_metal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("k", [33, 100, 192, 1536, 4096])
def test_metal_front_end_without_norm_is_bit_exact(k, dtype):
    # Relies on Metal's default (safe) math mode giving correctly rounded float32 division; see the module
    # docstring of bitnet_vision.apple.mps_kernels.
    x = _tokens(266, k, dtype, seed=k)
    codes, scale = fused_front_end(x.to("mps"), None, 0.0)
    ref_codes, ref_scale = quantize_activations(x)
    assert torch.equal(codes.cpu(), ref_codes) and torch.equal(scale.cpu(), ref_scale)


@requires_metal
def test_metal_front_end_odd_widths_layouts_and_non_finite_values():
    for k in (1, 33, 100, 1000, 5000):  # rows that do not fill whole simdgroups, and K above 4096
        x, w = _tokens(9, k, seed=k), _norm_weight(k)
        codes, scale = fused_front_end(x.to("mps"), w.to("mps"), 1e-6)
        _assert_front_end_close(codes, scale, x, w, 1e-6)
    x = _tokens(24, 384).reshape(2, 12, 384).transpose(0, 1)  # non-contiguous, 3D
    codes, scale = fused_front_end(x.to("mps"), None, 0.0)
    assert codes.shape == (12, 2, 384) and scale.shape == (12, 2, 1)
    _assert_front_end_close(codes, scale, x, None, 0.0)
    x = torch.randn(4, 192, generator=torch.Generator().manual_seed(0))
    x[0, 5], x[1, 7], x[2, 9] = float("nan"), float("inf"), -float("inf")
    for w in (None, torch.ones(192)):
        codes, scale = fused_front_end(x.to("mps"), None if w is None else w.to("mps"), 1e-6)
        ref_codes, ref_scale = fused_front_end(x, w, 1e-6)
        assert torch.equal(codes.cpu().isnan(), ref_codes.isnan())
        assert torch.equal(scale.cpu().isnan(), ref_scale.isnan())
        finite = ~ref_codes[:3].isnan()  # rows with NaN or inf: codes that are not NaN (0 here) must match
        assert torch.equal(codes.cpu()[:3][finite], ref_codes[:3][finite])
        _assert_front_end_close(codes[3:], scale[3:], x[3:], w, 1e-6)


@requires_metal
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("m,n", [(1, 192), (74, 576), (266, 1536), (1000, 384), (5, 7)])
def test_metal_epilogue_matches_reference(m, n, per_channel, bias):
    g = torch.Generator().manual_seed(m * n)
    acc = torch.randint(-200000, 200000, (m, n), generator=g).float()
    x_scale = torch.rand(m, 1, generator=g) * 50 + 1e-3
    w_scale = torch.rand(n if per_channel else 1, generator=g) * 0.05
    b = torch.randn(n, generator=g) if bias else None
    out = fused_epilogue(acc.to("mps"), x_scale.to("mps"), w_scale.to("mps"), None if b is None else b.to("mps"))
    ref = fused_epilogue(acc, x_scale, w_scale, b)
    # One rounding per step on both sides; Metal may contract the last multiply-add into an fma.
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-6, atol=1e-6 * ref.abs().max().item())


@requires_metal
def test_mps_float32_matmul_is_exact_on_integer_codes():
    k = 1536
    worst = F.linear(torch.full((4, k), 127.0, device="mps"), torch.ones(8, k, device="mps"))
    assert torch.all(worst.cpu() == 127.0 * k)  # 195,072: overflows float16, exact in float32
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-128, 128, (266, k), generator=g).float()
    w = torch.randint(-1, 2, (384, k), generator=g).float()
    assert torch.equal(F.linear(a.to("mps"), w.to("mps")).cpu(), F.linear(a, w))


@requires_metal
@pytest.mark.parametrize("cls,kwargs", LAYER_VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_metal_fused_layer_close_to_cpu_inference_layer(cls, kwargs, dtype):
    inf = _inference_layer(cls, k=384, n=1536, **kwargs)
    fused = FusedBitLinearInference.from_inference(_inference_layer(cls, k=384, n=1536, **kwargs)).to("mps")
    x = (3 * torch.randn(2, 266, 384)).to(dtype)
    with torch.no_grad():
        y = fused(x.to("mps")).cpu()
        y_ref = inf(x)
        h = inf.norm(x) if inf.norm is not None else x
        h = inf.hadamard(h) if inf.hadamard is not None else h
        _, x_scale = quantize_activations(h)
    # Allowed: a few codes per token 1 apart (MPS norm and rotation round differently from the CPU), a token scale
    # off by one half-precision step, and float16 output rounding.
    y_ref = y_ref.float()
    bias = inf.bias if inf.bias is not None else torch.zeros(1536)
    step = inf.weight_scale / x_scale  # one activation code times one weight, in output units
    bound = 2e-3 * (y_ref.abs() + (y_ref - bias).abs()) + 3 * step + 1e-5
    assert y.dtype == dtype and ((y.float() - y_ref).abs() <= bound).all()


@requires_metal
def test_metal_kernels_are_used_and_autocast_does_not_leak_in(monkeypatch):
    inf = _inference_layer(k=384, n=384)
    fused = FusedBitLinearInference.from_inference(inf).to("mps")
    x = torch.randn(266, 384, device="mps")
    with torch.no_grad():
        expected = fused(x)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            assert torch.equal(fused(x), expected)

    def unexpected(*args):
        raise AssertionError("reference path used on MPS")

    monkeypatch.setattr(mps_kernels, "_front_end_reference", unexpected)
    monkeypatch.setattr(mps_kernels, "_epilogue_reference", unexpected)
    with torch.inference_mode():
        assert torch.equal(fused(x), expected)
    with pytest.raises(AssertionError, match="reference path"):
        fused(x.requires_grad_())  # autograd has to see the ops, so the reference runs


@requires_metal
def test_metal_bitvit_close_to_cpu():
    # Quantized models amplify rounding: an ulp-sized change can flip a code, and that spreads through the blocks.
    # The yardstick is how far the CPU model's own logits move when every input value moves by one float32 ulp
    # (about 1% of the largest logit for this untrained model). Wrong kernels miss by the size of the logits.
    torch.manual_seed(0)
    packed = convert_for_inference(create_model("bitvit_cifar_small", hadamard=True).eval())
    fused = convert_to_fused(packed).to("mps")
    x = torch.randn(16, 3, 32, 32)
    with torch.no_grad():
        y, y_ref = fused(x.to("mps")).cpu(), packed(x)
        wiggle = (packed(x.nextafter(torch.full_like(x, math.inf))) - y_ref).abs().max().item()
    assert math.isfinite(y.abs().max().item())
    assert (y - y_ref).abs().max().item() <= 10 * wiggle + 1e-3 * y_ref.abs().max().item()
