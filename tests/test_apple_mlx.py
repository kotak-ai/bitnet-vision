"""MLX port of the ternary inference layer and ViT block (bitnet_vision/apple/mlx_ternary.py).

Runs on MLX's CPU backend on Linux and on the GPU (Metal) on a Mac, so no check relies on CPU-only
rounding except those pinned to ``mx.cpu``. RMSNorm and LayerNorm sum in a different order than PyTorch,
which can move an activation lying on a rounding tie (k + 0.5) to the neighbouring int8 code; the
comparisons allow exactly that and nothing else. Half precision is only smoke-tested: MLX's CPU quantized
matmul accumulates float16 in float16 (Metal's kernels accumulate in float32).
"""

import os
import subprocess
import sys

import pytest
import torch

from bitnet_vision import BitLinear, HBitLinear, convert_for_inference, pack_ternary
from bitnet_vision.apple import mlx_ternary as mt
from bitnet_vision.lora import DEFAULT_TARGETS, add_lora, disable_lora, lora_layers, set_lora_strength
from bitnet_vision.models.vit import BitViTConfig, Block
from bitnet_vision.quant import quantize_activations

try:
    import numpy as np
except ImportError:  # comes with the mlx extra; without it only the import test below runs
    np = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES = [(192, 576), (384, 1536), (768, 192), (1536, 384)]  # (in, out): T/4 qkv, S/2 fc1, T/4 fc2, S/2 fc2


@pytest.fixture(scope="module")
def mx():
    pytest.importorskip("numpy")
    return pytest.importorskip("mlx.core")


def _layer(k, n, bias=True, granularity="tensor", norm=True, seed=0):
    torch.manual_seed(seed)
    layer = BitLinear(k, n, bias=bias, weight_granularity=granularity, input_norm=norm)
    with torch.no_grad():
        layer.weight.normal_(0, 0.02)
        if granularity == "channel":
            layer.weight.mul_(torch.rand(n, 1) * 2 + 0.25)  # clearly different row scales
        if bias:
            layer.bias.normal_()
        if norm:
            layer.norm.weight.uniform_(0.5, 1.5)
    return convert_for_inference(layer, int_matmul="never")


def _add_adapters(model, seed=0):
    """Two non-zero LoRA adapters on every BitLinear-family layer (one at strength -0.5) and a disabled third."""
    for name, r in (("default", 4), ("b", 2), ("off", 2)):
        add_lora(model, r=r, name=name, targets=DEFAULT_TARGETS if hasattr(model, "attn") else None)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, m in lora_layers(model):
            for k in m.lora_settings:
                for w in (m.lora_A[k].weight, m.lora_B[k].weight):
                    w.copy_(0.1 * torch.randn(w.shape, generator=g))
    set_lora_strength(model, -0.5, "b")
    disable_lora(model, "off")
    return model


def _inputs(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 37, k, generator=g) * (3 * torch.rand(2, 37, 1, generator=g) + 0.1)
    x[0, 0] = 0  # absmax clamped to 1e-5, codes all 0
    x[0, 1] *= 1e-4 / x[0, 1].abs().max()  # mean(x^2) far below the norm's eps
    x[1, 0] *= 1e-7 / x[1, 0].abs().max()  # without the norm: absmax below the 1e-5 clamp
    x[1, 2, 5] = 50.0  # outlier
    return x


def _check_layer(m, ref, x, exact=False):
    """MLX layer ``m`` against the torch layer ``ref`` on ``x``.

    Codes must match except next to a rounding tie, and every token whose codes match must agree within
    float32 rounding. ``exact``: bit-identical codes and outputs.
    """
    xm = mt.to_mx(x)
    y = mt.to_torch(m(xm))
    codes = mt.to_torch(m.quantize_input(xm)[0])
    with torch.no_grad():
        y_ref = ref(x)
        codes_ref, scale_ref = quantize_activations(ref.norm(x) if ref.norm is not None else x)
        pre = (ref.norm(x) if ref.norm is not None else x) * scale_ref
    if exact:
        assert torch.equal(codes, codes_ref) and torch.equal(y, y_ref)
        return
    diff = codes - codes_ref
    near_tie = ((pre - pre.floor()) - 0.5).abs() < 1e-3
    assert diff.abs().max() <= 1 and not diff[~near_tie].any(), "codes differ away from a rounding tie"
    same = (diff == 0).all(dim=-1)
    assert same.float().mean() >= 0.95
    scale = y_ref.abs().max()
    torch.testing.assert_close(y[same], y_ref[same], rtol=1e-5, atol=1e-6 * scale)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=5e-3 * scale)


def _assert_block_close(y, y_ref):
    """Most tokens within float32 rounding of the reference.

    A code flipped at a rounding tie moves its token by about 1e-3 (relative to the largest output), and a
    flip in qkv reaches other tokens through attention. Over 8 random blocks of each kind here at most 12%
    of the tokens were past the tight bound; with a tanh instead of an erf GELU it was 93% to 100%.
    """
    scale = y_ref.abs().max()
    token_err = (y - y_ref).abs().amax(dim=-1) / scale
    assert (token_err <= 1e-5).float().mean() >= 0.75, f"worst token errors {token_err.flatten().topk(5).values}"
    assert token_err.max() <= 1e-2


@pytest.mark.parametrize("missing", ["mlx", "numpy"])  # neither MLX nor torch installs NumPy
def test_module_imports_without_mlx_or_numpy(missing):
    code = (
        f"import sys; sys.modules[{missing!r}] = None\n"
        "import bitnet_vision.apple.mlx_ternary as mt\n"
        "assert callable(mt.pack_ternary_mlx)\n"
        "for use in (lambda: mt.MLXBitLinear, lambda: mt.to_mx([1.0])):\n"
        "    try:\n        use()\n    except ImportError as e:\n        assert 'pip install mlx numpy' in str(e), e\n"
        "    else:\n        raise AssertionError('expected ImportError')\n"
    )
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_lazy_classes(mx):
    assert {"MLXBitLinear", "MLXBlock"} <= set(dir(mt))
    assert mt.MLXBitLinear is mt.MLXBitLinear
    with pytest.raises(AttributeError):
        mt.NotAThing  # noqa: B018


def test_to_mx_to_torch_roundtrip(mx):
    for t in [torch.randn(3, 5), torch.randn(3, 5).half(), torch.randn(3, 5).bfloat16(),
              torch.randint(-1, 2, (4, 6), dtype=torch.int8), torch.randn(5, 3).t()]:
        a = mt.to_mx(t)
        assert a.shape == tuple(t.shape)
        back = mt.to_torch(a)
        assert back.dtype == t.dtype and torch.equal(back, t)
    assert mt.to_mx(torch.ones(2), "float16").dtype == mx.float16
    assert mt.to_torch(mx.ones(2), torch.float64).dtype == torch.float64


def test_pack_layout(mx):
    codes = torch.zeros(2, 32, dtype=torch.int8)
    codes[0, :4] = torch.tensor([-1, 0, 1, 1])
    codes[1, 16] = 1
    w_q, scales, biases = mt.pack_ternary_mlx(codes, 1.0, group_size=32)
    w = np.array(w_q)
    assert w_q.dtype == mx.uint32 and w.shape == (2, 2)
    zeros = sum(1 << (2 * i) for i in range(16))  # code 0 is stored as 1 in every 2-bit slot
    assert w[0, 0] == (zeros & ~0xFF) | (0 << 0) | (1 << 2) | (2 << 4) | (2 << 6) and w[0, 1] == zeros  # -1, 0, 1, 1
    assert w[1, 0] == zeros and w[1, 1] == zeros + 1
    assert np.array_equal(np.array(scales), np.ones((2, 1))) and np.array_equal(np.array(biases), -np.ones((2, 1)))


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("k", [192, 384, 768, 1536])
@pytest.mark.parametrize("per_row", [False, True])
def test_pack_dequantizes_exactly(mx, group_size, k, per_row):
    if k % group_size:
        with pytest.raises(ValueError, match="valid"):
            mt.pack_ternary_mlx(torch.zeros(4, k, dtype=torch.int8), 1.0, group_size=group_size)
        return
    torch.manual_seed(k + group_size)
    n = 48
    codes = torch.randint(-1, 2, (n, k), dtype=torch.int8)
    scale = torch.rand(n) * 0.1 + 0.001 if per_row else torch.tensor([0.0372])
    for dtype, tdtype in [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        w_q, scales, biases = mt.pack_ternary_mlx(codes, scale, group_size=group_size, dtype=dtype)
        assert w_q.shape == (n, k // 16) and scales.shape == biases.shape == (n, k // group_size)
        assert scales.dtype == biases.dtype == getattr(mx, dtype)
        w = mt.to_torch(mx.dequantize(w_q, scales, biases, group_size=group_size, bits=2), torch.float32)
        expected = codes.float() * scale.to(tdtype).float().reshape(-1, 1)
        assert torch.equal(w, expected)


def test_pack_rejects_bad_input(mx):
    ok = torch.zeros(4, 64, dtype=torch.int8)
    for codes, scale, group, match in [
        (ok + 2, 1.0, 64, "only -1, 0 and 1"),
        (ok.float() + 0.5, 1.0, 64, "only -1, 0 and 1"),
        (torch.zeros(4, 96, dtype=torch.int8), 1.0, 64, "not a multiple"),
        (ok, 1.0, 16, "group_size"),
        (ok[0], 1.0, 64, "shape"),
        (ok, torch.ones(3), 64, "one value per row"),
    ]:
        with pytest.raises(ValueError, match=match):
            mt.pack_ternary_mlx(codes, scale, group_size=group)
    with pytest.raises(ValueError, match="dtype"):
        mt.pack_ternary_mlx(ok, 1.0, dtype="float64")


@pytest.mark.parametrize("rows", [48, 50])  # 50: the repo layout pads the last row block
def test_from_packed_repo(mx, rows):
    codes = torch.randint(-1, 2, (rows, 128), dtype=torch.int8)
    scale = torch.rand(rows) + 0.5
    ours = mt.from_packed_repo(pack_ternary(codes), rows, scale, group_size=128)
    direct = mt.pack_ternary_mlx(codes, scale, group_size=128)
    for a, b in zip(ours, direct):
        assert np.array_equal(np.array(a), np.array(b))


def test_quantized_matmul_on_packed_weights_is_exact(mx):
    """Integer activation codes times unit-scale ternary weights: exact integer sums, also on the GPU."""
    torch.manual_seed(0)
    for k, n in SHAPES:
        codes = torch.randint(-1, 2, (n, k), dtype=torch.int8)
        x = torch.randint(-127, 128, (74, k)).float()
        exact = x.double() @ codes.double().t()
        w_q, s, b = mt.pack_ternary_mlx(codes, 1.0)
        y = mt.to_torch(mx.quantized_matmul(mt.to_mx(x), w_q, s, b, transpose=True, group_size=64, bits=2))
        assert torch.equal(y.double(), exact)
        g = 0.0372  # folded scale: float32 rounding of each product, so close but not exact
        w_q, s, b = mt.pack_ternary_mlx(codes, g)
        y = mt.to_torch(mx.quantized_matmul(mt.to_mx(x), w_q, s, b, transpose=True, group_size=64, bits=2))
        bound = (x.abs().double() @ codes.abs().double().t()).max() * g
        torch.testing.assert_close(y.double(), exact * float(np.float32(g)), rtol=0, atol=1e-5 * bound)


def test_mx_quantize_is_not_exact_for_ternary(mx):
    """Documents the trap that pack_ternary_mlx avoids: MLX's own 2-bit quantizer maps -g to -g / 2."""
    torch.manual_seed(0)
    codes = torch.randint(-1, 2, (64, 256), dtype=torch.int8)
    g = 0.0372
    w = mt.to_mx(codes.float() * g)
    deq = mt.to_torch(mx.dequantize(*mx.quantize(w, group_size=64, bits=2), group_size=64, bits=2))
    if torch.equal(deq, mt.to_torch(w)):
        pytest.skip("mx.quantize(bits=2) is exact for ternary weights in this MLX version")
    torch.testing.assert_close(deq[codes == -1], torch.full_like(deq[codes == -1], -g / 2), rtol=1e-3, atol=0)
    exact = mx.dequantize(*mt.pack_ternary_mlx(codes, g), group_size=64, bits=2)
    assert torch.equal(mt.to_torch(exact), mt.to_torch(w))


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("granularity", ["tensor", "channel"])
@pytest.mark.parametrize("norm", [True, False])
def test_layer_matches_inference_layer(mx, k, n, bias, granularity, norm):
    ref = _layer(k, n, bias=bias, granularity=granularity, norm=norm)
    m = mt.MLXBitLinear.from_inference(ref)
    assert m.packed and m.group_size == 64
    _check_layer(m, ref, _inputs(k))


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("granularity", ["tensor", "channel"])
def test_layer_without_norm_is_bit_identical_on_cpu(mx, k, n, granularity):
    """Same quantizer arithmetic (127 * (1 / absmax), round half to even) and exact integer matmul."""
    ref = _layer(k, n, granularity=granularity, norm=False)
    with mx.stream(mx.cpu):
        _check_layer(mt.MLXBitLinear.from_inference(ref), ref, _inputs(k), exact=True)


def test_quantizer_rounds_half_to_even(mx):
    """127 * (1 / 127) is exactly 1 in float32, so these values sit exactly on rounding ties."""
    ref = _layer(192, 64, norm=False)
    x = torch.zeros(1, 1, 192)
    x[0, 0, :8] = torch.tensor([127.0, 0.5, 1.5, 2.5, -0.5, -2.5, 126.5, -127.0])
    with mx.stream(mx.cpu):
        codes = mt.to_torch(mt.MLXBitLinear.from_inference(ref).quantize_input(mt.to_mx(x))[0])
        assert codes[0, 0, :8].tolist() == [127, 0, 2, 2, 0, -2, 126, -127]
        _check_layer(mt.MLXBitLinear.from_inference(ref), ref, x, exact=True)


def test_rms_norm_matches(mx):
    ref = _layer(384, 64)
    m = mt.MLXBitLinear.from_inference(ref)
    x = _inputs(384)
    y = mt.to_torch(mx.fast.rms_norm(mt.to_mx(x), m.norm_weight, m.norm_eps))
    torch.testing.assert_close(y, ref.norm(x).detach(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_group_size_does_not_change_the_result(mx, group_size):
    ref = _layer(384, 96)
    x = mt.to_mx(_inputs(384))
    y = mt.MLXBitLinear.from_inference(ref, group_size=group_size)(x)
    assert np.array_equal(np.array(y), np.array(mt.MLXBitLinear.from_inference(ref, group_size=32)(x)))


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("norm", [True, False])
def test_dense_and_packed_agree(mx, k, n, norm):
    ref = _layer(k, n, granularity="channel", norm=norm)
    x = mt.to_mx(_inputs(k))
    dense = mt.MLXBitLinear.from_inference(ref, dense=True)
    assert not dense.packed and dense.weight.shape == (n, k)
    packed = mt.MLXBitLinear.from_inference(ref)
    assert np.array_equal(np.array(dense(x)), np.array(packed(x)))


@pytest.mark.parametrize("k,n", SHAPES[:2])
@pytest.mark.parametrize("dense", [False, True])
def test_layer_with_lora_adapters_matches_inference_layer(mx, k, n, dense):
    ref = _add_adapters(_layer(k, n, granularity="channel"))
    m = mt.MLXBitLinear.from_inference(ref, dense=dense)
    assert len(m.lora_A) == len(m.lora_B) == len(m.lora_scale) == 2  # the disabled adapter is left out
    x = _inputs(k)
    _check_layer(m, ref, x)
    with torch.no_grad():
        y = ref(x)
        disable_lora(ref)
        assert (y - ref(x)).abs().max() > 0.1 * y.abs().max()  # the adapters matter at this tolerance


def test_training_layer_with_adapters_is_exported_on_the_fly(mx):
    torch.manual_seed(0)
    layer = _add_adapters(BitLinear(192, 64))
    x = mt.to_mx(_inputs(192))
    a = mt.MLXBitLinear.from_inference(layer)(x)
    b = mt.MLXBitLinear.from_inference(convert_for_inference(layer))(x)
    assert np.array_equal(np.array(a), np.array(b))


def test_training_layer_is_exported_on_the_fly(mx):
    torch.manual_seed(0)
    layer = BitLinear(192, 64)
    x = mt.to_mx(_inputs(192))
    a = mt.MLXBitLinear.from_inference(layer)(x)
    b = mt.MLXBitLinear.from_inference(convert_for_inference(layer))(x)
    assert np.array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("make", [
    lambda: HBitLinear(192, 64),
    lambda: BitLinear(192, 64, act_bits=4),
    lambda: BitLinear(192, 64, act_bits=8, act_method="absmean"),
    lambda: BitLinear(192, 64, act_bits=None),
])
def test_unsupported_layers_raise(mx, make):
    with pytest.raises(NotImplementedError):
        mt.MLXBitLinear.from_inference(convert_for_inference(make()))


def test_rejects_other_modules(mx):
    with pytest.raises(TypeError):
        mt.MLXBitLinear.from_inference(torch.nn.Linear(4, 4))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("dense", [False, True])
def test_half_precision_smoke(mx, dtype, dense):
    ref = _layer(1536, 384)
    x = mt.to_mx(_inputs(1536), dtype)
    y32 = mt.to_torch(mt.MLXBitLinear.from_inference(ref)(x.astype(mx.float32)))
    y = mt.MLXBitLinear.from_inference(ref, dtype=dtype, dense=dense)(x)
    assert y.dtype == getattr(mx, dtype)
    y = mt.to_torch(y, torch.float32)
    assert torch.isfinite(y).all()
    assert torch.nn.functional.cosine_similarity(y.flatten(), y32.flatten(), dim=0) > 0.999


@pytest.mark.parametrize("dense", [False, True])
def test_float16_matmul_does_not_overflow(mx, dense):
    """All codes at 127 and all weights at +1: raw sums of 127 * 1536 = 195,072 exceed float16's 65,504."""
    layer = BitLinear(1536, 8)
    with torch.no_grad():
        layer.weight.fill_(0.01)
        layer.bias.zero_()
    ref = convert_for_inference(layer, int_matmul="never")
    x = torch.ones(1, 3, 1536)
    m = mt.MLXBitLinear.from_inference(ref, dtype="float16", dense=dense)
    y = mt.to_torch(m(mt.to_mx(x, "float16")), torch.float32)
    torch.testing.assert_close(y, ref(x), rtol=1e-2, atol=0)  # 15.36; CPU float16 accumulation is coarse


def test_compiled_layer_matches_eager(mx):
    for dense in (False, True):
        ref = _layer(768, 192, granularity="channel")
        m = mt.MLXBitLinear.from_inference(ref, dense=dense)
        x = mt.to_mx(_inputs(768))
        eager, comp = mt.to_torch(m(x)), mt.to_torch(mt.compiled(m)(x))
        torch.testing.assert_close(comp, eager, rtol=1e-6, atol=1e-6 * eager.abs().max())


def _block(dim, heads, seed=0, **cfg):
    torch.manual_seed(seed)
    block = Block(BitViTConfig(embed_dim=dim, num_heads=heads, mlp_ratio=4.0, **cfg))
    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.endswith("bias") or name.endswith("gamma"):
                p.normal_(0, 0.5)
            elif "norm" in name:  # LayerNorm and RMSNorm weights
                p.uniform_(0.5, 1.5)
    return convert_for_inference(block, int_matmul="never")


BLOCKS = {
    "T4": dict(dim=192, heads=3, tokens=74),
    "S2": dict(dim=384, heads=6, tokens=266, group_size=128),
    "T4_layerscale_relu2_channel": dict(dim=192, heads=3, tokens=74, group_size=32, layer_scale_init=0.1,
                                        mlp_act="relu2", weight_granularity="channel"),
    "T4_no_block_norm": dict(dim=192, heads=3, tokens=74, block_norm=False),
    "T4_large_eps": dict(dim=192, heads=3, tokens=74, eps=0.25),  # every LayerNorm and RMSNorm
}


@pytest.mark.parametrize("name", list(BLOCKS))
def test_block_matches_torch(mx, name):
    cfg = dict(BLOCKS[name])
    dim, heads, tokens, group_size = cfg.pop("dim"), cfg.pop("heads"), cfg.pop("tokens"), cfg.pop("group_size", 64)
    eps = cfg.pop("eps", None)
    block = _block(dim, heads, **cfg)
    for module in block.modules():
        if eps is not None and hasattr(module, "eps"):
            module.eps = eps
    x = torch.randn(2, tokens, dim, generator=torch.Generator().manual_seed(1))
    x[0, 1:4] *= 1e-3  # variance near the LayerNorm eps
    with torch.no_grad():
        y_ref = block(x)
    m = mt.mlx_block_from_vit_block(block, group_size=group_size)
    y = m(mt.to_mx(x))
    _assert_block_close(mt.to_torch(y), y_ref)
    dense = mt.mlx_block_from_vit_block(block, dense=True)(mt.to_mx(x))
    assert np.array_equal(np.array(dense), np.array(y))
    _assert_block_close(mt.to_torch(mt.compiled(m)(mt.to_mx(x))), mt.to_torch(y))


def test_block_with_lora_adapters_matches_torch(mx):
    block = _add_adapters(_block(192, 3, layer_scale_init=0.1))
    x = torch.randn(2, 74, 192, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        y_ref = block(x)
    _assert_block_close(mt.to_torch(mt.mlx_block_from_vit_block(block)(mt.to_mx(x))), y_ref)
    disable_lora(block)
    with torch.no_grad():
        assert (block(x) - y_ref).abs().max() > 0.05 * y_ref.abs().max()


def test_block_float16_smoke(mx):
    block = _block(192, 3)
    x = mt.to_mx(torch.randn(2, 74, 192))
    y32 = mt.to_torch(mt.mlx_block_from_vit_block(block)(x))
    y16 = mt.mlx_block_from_vit_block(block, dtype="float16")(x.astype(mx.float16))
    assert y16.dtype == mx.float16
    y16 = mt.to_torch(y16, torch.float32)
    assert torch.isfinite(y16).all()
    assert torch.nn.functional.cosine_similarity(y16.flatten(), y32.flatten(), dim=0) > 0.999


def test_block_with_hadamard_layers_raises(mx):
    with pytest.raises(NotImplementedError, match="Hadamard"):
        mt.mlx_block_from_vit_block(_block(192, 3, hadamard=True))
