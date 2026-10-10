import os

import pytest
import torch

from bitnet_vision.checkpoint import load_packed, save_packed
from bitnet_vision.export import (
    BitLinearInference,
    convert_for_inference,
    pack_ternary,
    packed_state_dict_bytes,
    unpack_ternary,
)
from bitnet_vision.layers import BitLinear, HBitLinear
from bitnet_vision.models import create_model


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 8, 13, 192])
def test_pack_unpack_roundtrip(rows):
    codes = torch.randint(-1, 2, (rows, 7), dtype=torch.int8)
    packed = pack_ternary(codes)
    assert packed.dtype == torch.uint8 and packed.shape == (-(-rows // 4), 7)
    assert torch.equal(unpack_ternary(packed, rows), codes)


def test_pack_layout_matches_hugging_face_bitnet():
    # HF transformers packs block i of ceil(rows/4) rows into bits 2i..2i+1, storing value + 1.
    codes = torch.tensor([[-1], [0], [1], [1], [0], [-1], [-1], [1]], dtype=torch.int8)
    packed = pack_ternary(codes)
    expected = torch.tensor(
        [[(0) | (2 << 2) | (1 << 4) | (0 << 6)], [(1) | (2 << 2) | (0 << 4) | (2 << 6)]], dtype=torch.uint8
    )
    assert torch.equal(packed, expected)


def test_pack_rejects_non_ternary():
    with pytest.raises(ValueError):
        pack_ternary(torch.tensor([[2]]))


@pytest.mark.parametrize("cls,granularity,act_bits,extra", [
    (BitLinear, "tensor", 8, {}), (BitLinear, "channel", 8, {}), (HBitLinear, "tensor", 8, {}),
    (HBitLinear, "tensor", 4, {}), (HBitLinear, "tensor", 8, {"random_signs": False}), (BitLinear, "tensor", None, {}),
    (BitLinear, "tensor", 8, {"deadzone_lambda": 1e-2}),
])
def test_inference_layer_equals_training_layer_in_float64(cls, granularity, act_bits, extra):
    torch.manual_seed(0)
    layer = cls(192, 50, bias=True, weight_granularity=granularity, act_bits=act_bits, **extra).double()
    with torch.no_grad():
        layer.bias.normal_()
    inf = BitLinearInference.from_bitlinear(layer)
    x = torch.randn(3, 9, 192, dtype=torch.float64)
    torch.testing.assert_close(inf(x), layer(x), rtol=1e-9, atol=1e-9)


def test_inference_float32_close_to_training_layer():
    torch.manual_seed(0)
    layer = BitLinear(768, 3072)
    inf = BitLinearInference.from_bitlinear(layer, keep_unpacked=True)
    x = torch.randn(64, 768)
    y, y_ref = inf(x), layer(x)
    assert (y - y_ref).abs().max() < 1e-3 * y_ref.abs().max()
    inf(x)  # second call reuses the unpacked int8 weights


def test_cannot_export_partially_quantized_layer():
    layer = BitLinear(8, 8)
    layer.quant_strength = 0.5
    with pytest.raises(ValueError):
        BitLinearInference.from_bitlinear(layer)


def test_packed_model_is_much_smaller_and_agrees(tmp_path):
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", hadamard=True).eval()
    packed = convert_for_inference(model)
    assert not any(isinstance(m, BitLinear) for m in packed.modules())
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        agree = (packed(x).argmax(-1) == model(x).argmax(-1)).float().mean().item()
    assert agree >= 0.75  # float32 rounding can flip a few int8 codes on a random (untrained) model
    fp_bytes = packed_state_dict_bytes(model.state_dict())
    packed_bytes = packed_state_dict_bytes(packed.state_dict())
    assert fp_bytes / packed_bytes > 10

    path = os.path.join(tmp_path, "packed.pt")
    save_packed(model, path)
    reloaded = load_packed(path)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), packed(x))
    assert os.path.getsize(path) < fp_bytes / 8


def test_packed_model_equals_training_model_in_float64():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2).double().eval()
    packed = convert_for_inference(model)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(packed(x), model(x), rtol=1e-8, atol=1e-8)


def test_weight_scale_is_a_mean_abs_multiplier():
    torch.manual_seed(0)
    layer = BitLinear(64, 32)
    inf = BitLinearInference.from_bitlinear(layer)
    torch.testing.assert_close(inf.weight_scale, layer.weight.detach().abs().mean().reshape(1))
    codes = unpack_ternary(inf.weight_packed, 32, torch.float32)
    torch.testing.assert_close(codes * inf.weight_scale, layer.ternary_weight()[0].float() / layer.ternary_weight()[1])


def test_deadzone_without_bias_is_folded_into_a_new_bias():
    torch.manual_seed(0)
    layer = BitLinear(64, 16, bias=False, deadzone_lambda=1e-2).double()
    inf = BitLinearInference.from_bitlinear(layer)
    assert inf.bias is not None
    x = torch.randn(4, 64, dtype=torch.float64)
    torch.testing.assert_close(inf(x), layer(x), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mode", ["never", "auto", "always"])
def test_int_matmul_modes_agree(mode):
    torch.manual_seed(0)
    layer = BitLinear(192, 64)
    x = torch.randn(40, 192)
    ref = BitLinearInference.from_bitlinear(layer, int_matmul="never")(x)
    try:
        out = BitLinearInference.from_bitlinear(layer, int_matmul=mode, keep_unpacked=False)(x)
    except RuntimeError:
        pytest.skip("torch._int_mm is not available on this backend")
    torch.testing.assert_close(out, ref)


def test_packed_checkpoint_records_conventions(tmp_path):
    from bitnet_vision.export import PACKED_METADATA

    model = create_model("bitvit_cifar_tiny", depth=1)
    payload = save_packed(model, str(tmp_path / "p.pt"))
    assert payload["conventions"] == PACKED_METADATA
    assert "multiplier" in payload["conventions"]["weight_scale"]
