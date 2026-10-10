"""Regression tests for problems found in code review of this rewrite."""

import os
import warnings

import pytest
import torch
from torch import nn

from bitnet_vision.checkpoint import load_model, load_packed, save_model, save_packed, transfer_weights
from bitnet_vision.export import BitLinearInference, convert_for_inference, pack_ternary, unpack_ternary
from bitnet_vision.hadamard import HadamardTransform
from bitnet_vision.layers import BitLinear, HBitLinear, convert_linear_layers, set_quant_strength
from bitnet_vision.models import create_model
from bitnet_vision.quant import fake_quant_activations
from bitnet_vision.training import TensorDataset, TrainConfig, TwoStageSchedule, load_cifar10, train


def _colors(n, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n,), generator=g)
    images = torch.randint(40, 160, (n, 3, 32, 32), generator=g)
    tint = torch.tensor([[60, 0, 0], [0, 60, 0], [0, 0, 60], [-30, -30, -30]])
    return TensorDataset((images + tint[labels][:, :, None, None]).clamp(0, 255).to(torch.uint8), labels)


def _small(**kw):
    kw.setdefault("depth", 1)
    return create_model("bitvit_cifar_tiny", embed_dim=96, num_heads=2, num_classes=4, **kw)


# ---------------------------------------------------------------- export and checkpoints


def test_unpacked_weights_are_refreshed_after_loading_new_weights():
    torch.manual_seed(0)
    a, b = BitLinear(64, 32), BitLinear(64, 32)
    inf = BitLinearInference.from_bitlinear(a, keep_unpacked=True)
    x = torch.randn(40, 64)
    inf(x)  # fills the unpacked cache with a's weights
    inf.load_state_dict(BitLinearInference.from_bitlinear(b).state_dict())
    torch.testing.assert_close(inf(x), BitLinearInference.from_bitlinear(b)(x))


@pytest.mark.parametrize("src_kw,dst_kw", [
    ({"hadamard": True}, {}),
    ({}, {"hadamard": True}),
    ({"hadamard": True}, {"hadamard": True, "hadamard_random_signs": False}),
])
def test_transfer_weights_preserves_the_function_in_every_direction(src_kw, dst_kw):
    torch.manual_seed(0)
    src = _small(quantize=False, **src_kw).double().eval()
    dst = _small(quantize=False, **dst_kw).double().eval()
    transfer_weights(dst, src)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(dst(x), src(x))


def test_load_model_refuses_to_toggle_hadamard(tmp_path):
    save_model(_small(), str(tmp_path / "m.pt"))
    with pytest.raises(ValueError, match="transfer_weights"):
        load_model(str(tmp_path / "m.pt"), hadamard=True)


def test_convert_a_bare_layer_and_keep_shared_layers_shared():
    layer = BitLinear(16, 8)
    assert isinstance(convert_for_inference(layer), BitLinearInference)
    shared = BitLinear(8, 8)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    packed = convert_for_inference(model)
    assert packed[0] is packed[2]


def test_pack_and_unpack_reject_invalid_input():
    with pytest.raises(ValueError):
        pack_ternary(torch.tensor([[0.6]]))
    with pytest.raises(ValueError):
        pack_ternary(torch.tensor([[float("nan")]]))
    packed = pack_ternary(torch.zeros(4, 3, dtype=torch.int8))
    with pytest.raises(ValueError):
        unpack_ternary(packed, 4, dtype=torch.uint8)
    with pytest.raises(ValueError):
        unpack_ternary(packed, 5)


def test_packed_save_of_a_manually_quantized_full_precision_model_loads(tmp_path):
    model = _small(quantize=False)
    set_quant_strength(model, 1.0)
    save_packed(model, str(tmp_path / "p.pt"))
    reloaded = load_packed(str(tmp_path / "p.pt"))
    assert reloaded(torch.randn(2, 3, 32, 32)).shape == (2, 4)


def test_packed_float_path_ignores_autocast():
    torch.manual_seed(0)
    inf = BitLinearInference.from_bitlinear(BitLinear(64, 32), int_matmul="never")
    x = torch.randn(4, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = inf(x)
    torch.testing.assert_close(y, inf(x))


# ---------------------------------------------------------------- core math


def test_hadamard_stays_float32_under_autocast():
    t = HadamardTransform(192, random_signs=True)
    x = torch.randn(8, 192)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = t(x)
    torch.testing.assert_close(y, t(x), rtol=0, atol=0)


def test_hadamard_buffers_stay_exact_after_half_precision_round_trip():
    t = HadamardTransform(768, random_signs=True)
    exact = t.factor0.clone()
    t.half().float()
    assert torch.equal(t.factor0, exact) and t.factor0.dtype == torch.float32
    t.bfloat16()
    assert torch.equal(t.factor0, exact)


def test_hadamard_survives_meta_construction_and_to_empty():
    with torch.device("meta"):
        layer = HBitLinear(192, 32)
    layer.to_empty(device="cpu")
    ref = HadamardTransform(192, random_signs=True)
    assert torch.equal(layer.hadamard.factor0, ref.factor0) and torch.equal(layer.hadamard.signs, ref.signs)


def test_plain_bitlinear_uses_absmean_below_8_bits():
    assert BitLinear(8, 8, act_bits=4).act_method == "absmean"
    assert BitLinear(8, 8).act_method == "absmax"


def test_float16_absmean_does_not_overflow():
    x = torch.full((2, 64), 3.0e4, dtype=torch.float16)
    x[:, 0] = 6.0e4
    assert torch.isfinite(fake_quant_activations(x, bits=4, method="absmean")).all()


def test_convert_linear_layers_keeps_sharing_frozen_state_and_warns_on_attention():
    shared = nn.Linear(8, 8)
    shared.weight.requires_grad_(False)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    convert_linear_layers(model)
    assert model[0] is model[2] and isinstance(model[0], BitLinear)
    assert not model[0].weight.requires_grad
    with pytest.warns(UserWarning, match="MultiheadAttention"):
        convert_linear_layers(nn.Sequential(nn.MultiheadAttention(8, 2)))


# ---------------------------------------------------------------- training


@pytest.mark.slow
def test_resume_reproduces_an_uninterrupted_run_with_drop_path(tmp_path):
    """DropPath draws from the global RNG, so resuming must restore it, not reseed it."""
    cfg = TrainConfig(epochs=3, batch_size=32, warmup_epochs=0.5, log_every=0)
    data = _colors(64, 0)
    torch.manual_seed(0)
    init = _small(depth=2, drop_path_rate=0.2).state_dict()

    def fresh():
        model = _small(depth=2, drop_path_rate=0.2)
        model.load_state_dict(init)
        return model

    assert fresh().blocks[-1].drop_path.p > 0  # stochastic depth really draws from the global RNG

    full = fresh()
    train(full, data, None, cfg, log=lambda s: None)

    class Interrupt(Exception):
        pass

    def interrupt_at_epoch_2(msg):
        if msg.startswith('{"epoch": 2'):
            raise Interrupt  # before epoch 2 is saved, so last.pt holds epoch 1

    with pytest.raises(Interrupt):
        train(fresh(), data, None, cfg, out_dir=str(tmp_path), log=interrupt_at_epoch_2)
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    assert state["epoch"] == 1
    resumed = _small(depth=2, drop_path_rate=0.2)
    torch.manual_seed(999)  # a fresh process would have some other global RNG state
    train(resumed, data, None, cfg, resume=state, log=lambda s: None)
    for (name, a), b in zip(full.state_dict().items(), resumed.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


@pytest.mark.slow
def test_resuming_a_finished_run_restores_low_bit_activations(tmp_path):
    cfg = TrainConfig(epochs=2, batch_size=32, warmup_epochs=0.5, final_act_bits=4, final_act_bits_fraction=0.25,
                      log_every=0)
    model = _small(hadamard=True)
    train(model, _colors(64, 0), None, cfg, out_dir=str(tmp_path), log=lambda s: None)
    fresh = _small(hadamard=True)  # rebuilt from 8-bit command-line settings
    train(fresh, _colors(64, 0), None, cfg, resume=torch.load(tmp_path / "last.pt", weights_only=True),
          log=lambda s: None)
    assert all(m.act_bits == 4 for m in fresh.modules() if isinstance(m, BitLinear))
    assert fresh.config.act_bits == 4 and fresh.config.hadamard_act_bits == 4


def test_unfinished_quantization_ramp_is_rejected():
    cfg = TrainConfig(epochs=2, batch_size=32, warmup_epochs=0.5, fp_fraction=0.5, quant_ramp_epochs=2, log_every=0)
    with pytest.raises(ValueError, match="no fully ternary"):
        train(_small(), _colors(64, 0), None, cfg, log=lambda s: None)


def test_warmup_must_end_before_the_stage_switch():
    with pytest.raises(ValueError, match="warmup"):
        TwoStageSchedule(total_steps=10, peak_lr=1e-3, kind="bitnet", warmup_steps=5)
    with pytest.raises(ValueError, match="warmup"):
        TwoStageSchedule(total_steps=10, peak_lr=1e-3, kind="cosine", warmup_steps=10)


def test_rotated_scope_needs_hadamard_layers():
    cfg = TrainConfig(epochs=1, batch_size=32, schedule="cosine", warmup_epochs=0, final_act_bits=4,
                      final_act_bits_fraction=0.5, final_act_bits_scope="rotated", log_every=0)
    with pytest.raises(ValueError, match="HBitLinear"):
        train(_small(), _colors(64, 0), None, cfg, log=lambda s: None)


def test_truncated_cifar_file_is_detected(tmp_path):
    folder = tmp_path / "cifar-10-batches-bin"
    folder.mkdir()
    for i in range(1, 6):
        (folder / f"data_batch_{i}.bin").write_bytes(bytes(3073))
    (folder / "test_batch.bin").write_bytes(bytes(3073 + 100))  # cut in the middle of a record
    with pytest.raises(ValueError, match="records"):
        load_cifar10(str(tmp_path), download=False)


def test_checkpoints_are_written_atomically(tmp_path):
    train(_small(), _colors(64, 0), None, TrainConfig(epochs=1, batch_size=32, schedule="cosine", warmup_epochs=0,
                                                       log_every=0), out_dir=str(tmp_path), log=lambda s: None)
    assert sorted(os.listdir(tmp_path)) == ["last.pt", "metrics.json"]


def test_no_warnings_for_a_standard_model():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        create_model("bitvit_cifar_tiny", hadamard=True)


# ---------------------------------------------------------------- coverage gaps found in review


@pytest.mark.parametrize("cls,kw", [
    (BitLinear, {}), (BitLinear, {"weight_granularity": "channel"}), (HBitLinear, {}),
    (HBitLinear, {"act_bits": 4}), (BitLinear, {"act_bits": None}),
])
def test_float32_packed_path_matches_float64(cls, kw):
    """The deployment path (float32, integer matmul) against the exact float64 evaluation of the same file."""
    torch.manual_seed(0)
    layer = cls(192, 96, **kw)
    inf32 = BitLinearInference.from_bitlinear(layer)
    inf64 = BitLinearInference.from_bitlinear(layer.double())
    x = torch.randn(64, 192)
    y32, y64 = inf32(x), inf64(x.double())
    diff = (y32.double() - y64).abs()
    assert diff.mean() < 1e-4 * y64.abs().mean()  # nearly all entries agree to float32 rounding
    assert diff.max() < 0.05 * y64.abs().max()  # a rare flipped activation code moves an entry by one step


@pytest.mark.parametrize("cls", [BitLinear, HBitLinear])
def test_partial_quant_strength_blends_inside_the_layer(cls):
    torch.manual_seed(0)
    layer = cls(64, 32).double()
    layer.quant_strength = 0.3
    x = torch.randn(4, 64, dtype=torch.float64)
    xn = layer.rotate(layer.norm(x))
    xq = fake_quant_activations(xn)
    from bitnet_vision.quant import fake_quant_weights

    w = layer.weight.detach()
    expected = nn.functional.linear(xn + 0.3 * (xq - xn), w + 0.3 * (fake_quant_weights(w) - w), layer.bias)
    torch.testing.assert_close(layer(x), expected)


def test_4bit_absmean_reaches_the_most_negative_code():
    from bitnet_vision.quant import quantize_activations

    x = torch.tensor([[-3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    codes, _ = quantize_activations(x, bits=4, method="absmean")
    assert codes.min() == -8


def test_hadamard_on_bf16_input_equals_float32_math():
    t = HadamardTransform(192, random_signs=True)
    x = torch.randn(4, 192, dtype=torch.bfloat16)
    expected = t(x.float()).to(torch.bfloat16)
    assert torch.equal(t(x), expected)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(t(x), expected)


def test_bf16_autocast_quantizes_the_same_codes_as_float32():
    torch.manual_seed(0)
    layer = HBitLinear(192, 64)
    x = torch.randn(16, 192).to(torch.bfloat16)  # e.g. the bf16 output of attention under autocast
    seen = {}
    original = layer.rotate

    def spy(v):
        out = original(v)
        seen.setdefault("dtype", out.dtype)
        seen.setdefault("values", out.detach().clone())
        return out

    layer.rotate = spy
    with torch.autocast("cpu", dtype=torch.bfloat16):
        layer(x)
    assert seen["dtype"] == torch.float32
    torch.testing.assert_close(seen["values"], original(layer.norm(x.float())), rtol=0, atol=0)


def test_old_hadamard_checkpoints_keep_their_unsigned_rotation(tmp_path):
    model = _small(hadamard=True, hadamard_random_signs=False)
    cfg = model.config.to_dict()
    del cfg["hadamard_random_signs"]  # as written before the field existed
    torch.save({"model_config": cfg, "model": model.state_dict()}, tmp_path / "old.pt")
    loaded = load_model(str(tmp_path / "old.pt"))
    assert loaded.config.hadamard_random_signs is False
    assert not loaded.blocks[0].mlp.fc2.hadamard.random_signs


def test_distillation_temperature_scaling():
    from bitnet_vision.training import distillation_loss

    s, t = torch.tensor([[2.0, 0.0, -1.0]]), torch.tensor([[0.0, 1.0, 0.5]])
    p_t = torch.softmax(t / 2, -1)
    manual = (p_t * (torch.log_softmax(t / 2, -1) - torch.log_softmax(s / 2, -1))).sum() * 4
    torch.testing.assert_close(distillation_loss(s, t, temperature=2.0), manual)


def test_bitnet_cosine_schedule_values():
    s = TwoStageSchedule(total_steps=100, peak_lr=1e-3, kind="bitnet_cosine", weight_decay=0.1)
    assert s(0)[0] == pytest.approx(1e-3)
    assert s(25)[0] == pytest.approx(1e-3 * 0.5 * (1 + __import__("math").cos(__import__("math").pi * 0.25)))
    assert s(50) == (pytest.approx(1e-3 * 0.5 * 2 / 3), 0.0)


def test_drop_path_keeps_the_expected_value():
    from bitnet_vision.models.vit import DropPath

    torch.manual_seed(0)
    dp = DropPath(0.3).train()
    x = torch.ones(20000, 4)
    out = dp(x)
    assert all(v == 0.0 or abs(v - 1 / 0.7) < 1e-5 for v in out.unique().tolist())
    assert abs(out.mean().item() - 1.0) < 0.02
    assert torch.equal(dp.eval()(x), x)


@pytest.mark.slow
def test_quant_ramp_values_and_code_flip_metric():
    cfg = TrainConfig(epochs=4, batch_size=32, warmup_epochs=0.5, fp_fraction=0.25, quant_ramp_epochs=2, log_every=0)
    hist = train(_small(), _colors(64, 0), None, cfg, log=lambda s: None)
    assert [e["quant_strength"] for e in hist["epochs"]] == [0.0, 0.5, 1.0, 1.0]
    assert hist["epochs"][1]["code_flip_rate"] > 0


def test_cifar_download_path_extracts_and_checks_the_checksum(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile

    from bitnet_vision.training import data

    record = bytes([3]) + bytes(3072)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in [f"data_batch_{i}.bin" for i in range(1, 6)] + ["test_batch.bin", "../evil.bin"]:
            info = tarfile.TarInfo(f"cifar-10-batches-bin/{name}")
            info.size = len(record)
            tar.addfile(info, io.BytesIO(record))
    archive = tmp_path / "cifar-10-binary.tar.gz"
    archive.write_bytes(buf.getvalue())
    monkeypatch.setattr(data, "CIFAR10_MD5", "0" * 32)
    with pytest.raises(RuntimeError, match="checksum"):
        load_cifar10(str(tmp_path), download=False)
    monkeypatch.setattr(data, "CIFAR10_MD5", hashlib.md5(buf.getvalue()).hexdigest())
    train_set, test_set = load_cifar10(str(tmp_path), download=False)
    assert len(train_set) == 5 and test_set.labels.tolist() == [3]
    assert not (tmp_path / "evil.bin").exists()


def test_full_precision_fraction_of_one_is_rejected():
    cfg = TrainConfig(epochs=1, batch_size=32, schedule="cosine", warmup_epochs=0, fp_fraction=1.0, log_every=0)
    with pytest.raises(ValueError, match="no fully ternary"):
        train(_small(), _colors(64, 0), None, cfg, log=lambda s: None)


def test_low_bit_phase_that_rounds_to_zero_steps_is_rejected():
    cfg = TrainConfig(epochs=1, batch_size=32, schedule="cosine", warmup_epochs=0, final_act_bits=4,
                      final_act_bits_fraction=0.05, log_every=0)  # 5% of 2 steps rounds to 0
    with pytest.raises(ValueError, match="no low-bit steps"):
        train(_small(), _colors(64, 0), None, cfg, log=lambda s: None)


def test_resume_refuses_changed_training_settings(tmp_path):
    cfg = TrainConfig(epochs=2, batch_size=32, warmup_epochs=0.5, log_every=0)
    train(_small(), _colors(64, 0), None, cfg, out_dir=str(tmp_path), log=lambda s: None)
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    with pytest.raises(ValueError, match="epochs"):
        train(_small(), _colors(64, 0), None, TrainConfig(epochs=3, batch_size=32, warmup_epochs=0.5, log_every=0),
              resume=state, log=lambda s: None)


def test_gradient_clipping_is_applied_every_step(monkeypatch):
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *args, **kwargs):
        params = list(params)
        calls.append(max_norm)
        out = original(params, max_norm, *args, **kwargs)
        total = torch.sqrt(sum(p.grad.pow(2).sum() for p in params if p.grad is not None))
        assert total <= max_norm * 1.001
        return out

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    cfg = TrainConfig(epochs=1, batch_size=16, schedule="cosine", warmup_epochs=0, grad_clip=0.01, log_every=0)
    train(_small(), _colors(64, 0), None, cfg, log=lambda s: None)
    assert calls == [0.01] * 4


def test_label_smoothing_changes_the_loss():
    def first_loss(smoothing):
        torch.manual_seed(0)
        cfg = TrainConfig(epochs=1, batch_size=64, schedule="cosine", warmup_epochs=0, label_smoothing=smoothing,
                          augment=False, log_every=0)
        return train(_small(), _colors(64, 0), None, cfg, log=lambda s: None)["epochs"][0]["train_loss"]

    assert first_loss(0.0) != pytest.approx(first_loss(0.3))


def test_config_differences_ignore_settings_that_cannot_matter():
    from bitnet_vision.checkpoint import config_differences, upgrade_config
    from bitnet_vision.models import BitViTConfig

    old = BitViTConfig().to_dict()
    del old["hadamard_random_signs"]  # written before the field existed
    assert config_differences(BitViTConfig(**upgrade_config(old)), BitViTConfig()) == []
    old_h = {**old, "hadamard": True}
    assert config_differences(BitViTConfig(**upgrade_config(old_h)), BitViTConfig(hadamard=True)) == [
        "hadamard_random_signs"]
    assert config_differences(BitViTConfig(), BitViTConfig(act_bits=4, hadamard=True)) == ["hadamard"]
    assert config_differences(BitViTConfig(), BitViTConfig(embed_dim=96, num_heads=2)) == ["embed_dim", "num_heads"]
