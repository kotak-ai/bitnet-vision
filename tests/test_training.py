import pytest
import torch
from torch import nn

from bitnet_vision.layers import BitLinear
from bitnet_vision.models import create_model
from bitnet_vision.training import (
    TensorDataset,
    TrainConfig,
    TwoStageSchedule,
    augment,
    distillation_loss,
    evaluate,
    iterate_batches,
    load_cifar10,
    normalize,
    ternary_stats,
    train,
)


def test_bitnet_schedule_matches_the_faq_recipe():
    """Linear decay to 0; at the midpoint lr x 2/3 and weight decay 0.1 -> 0 (BitNet b1.58 FAQ, Table 2)."""
    s = TwoStageSchedule(total_steps=110, peak_lr=1.5e-3, kind="bitnet", warmup_steps=10, weight_decay=0.1)
    assert s(0) == pytest.approx((1.5e-4, 0.1))
    assert s(10)[0] == pytest.approx(1.5e-3)
    for t in (20, 54, 90, 109):
        lr, wd = s(t)
        base = 1.5e-3 * (110 - t) / 100
        if t < 55:
            assert (lr, wd) == (pytest.approx(base), 0.1)
        else:
            assert (lr, wd) == (pytest.approx(base * 2 / 3), 0.0)


def test_restart_schedule_has_short_low_lr_second_phase():
    s = TwoStageSchedule(total_steps=100, peak_lr=1e-3, kind="restart", weight_decay=0.05, stage2_weight_decay=0.01)
    assert s.split_step == 90
    assert s(89)[0] < 1e-5 and s(89)[1] == 0.05
    assert s(90) == (pytest.approx(1e-4), 0.01)
    assert 1e-5 <= s(99)[0] < 1e-4


def test_wsd_schedule_holds_then_cools_down():
    s = TwoStageSchedule(total_steps=100, peak_lr=1e-3, kind="wsd", warmup_steps=5, weight_decay=0.05)
    assert [s(t)[0] for t in (5, 50, 89)] == [pytest.approx(1e-3)] * 3
    assert s(89)[1] == 0.05 and s(90)[1] == 0.0
    assert s(95)[0] == pytest.approx(5e-4) and s(99)[0] < 5e-5  # cosine over the last 10 steps


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_single_stage_schedules(kind):
    s = TwoStageSchedule(total_steps=50, peak_lr=1e-3, kind=kind, weight_decay=0.05)
    assert s(0)[0] == pytest.approx(1e-3)
    assert all(s(i)[1] == 0.05 for i in range(50))
    assert s(49)[0] < 1e-4


def test_schedule_rejects_bad_arguments():
    with pytest.raises(ValueError):
        TwoStageSchedule(total_steps=10, peak_lr=1e-3, kind="step")
    with pytest.raises(ValueError):
        TwoStageSchedule(total_steps=10, peak_lr=1e-3, split=1.5)


def test_distillation_loss():
    s = torch.randn(4, 10)
    assert distillation_loss(s, s.clone()).abs() < 1e-6
    t = torch.randn(4, 10)
    y = torch.randint(0, 10, (4,))
    pure = distillation_loss(s, t)
    mixed = distillation_loss(s, t, y, alpha=0.5)
    ce = nn.functional.cross_entropy(s, y)
    torch.testing.assert_close(mixed, 0.5 * pure + 0.5 * ce)
    s.requires_grad_()
    t.requires_grad_()
    distillation_loss(s, t, temperature=2.0).backward()
    assert t.grad is None and s.grad is not None


def test_augment_shapes_dtype_and_determinism():
    x = torch.randint(0, 256, (6, 3, 32, 32), dtype=torch.uint8)
    a = augment(x, generator=torch.Generator().manual_seed(0))
    b = augment(x, generator=torch.Generator().manual_seed(0))
    assert a.shape == x.shape and a.dtype == torch.uint8 and torch.equal(a, b)
    assert torch.equal(augment(x, pad=0, flip=False), x)
    flipped = augment(x, pad=0, flip=True, generator=torch.Generator().manual_seed(1))
    assert all(torch.equal(f, o) or torch.equal(f, o.flip(-1)) for f, o in zip(flipped, x))


def test_normalize_and_batches():
    x = torch.full((2, 3, 4, 4), 255, dtype=torch.uint8)
    assert normalize(x).dtype == torch.float32
    data = TensorDataset(torch.zeros(10, 3, 4, 4, dtype=torch.uint8), torch.arange(10))
    sizes = [len(y) for _, y in iterate_batches(data, 4, shuffle=False)]
    assert sizes == [4, 4, 2]
    assert [len(y) for _, y in iterate_batches(data, 4, shuffle=True, drop_last=True)] == [4, 4]


def _bytes(t):
    return bytes(t.flatten().tolist())


def test_load_cifar10_binary_format(tmp_path):
    folder = tmp_path / "cifar-10-batches-bin"
    folder.mkdir()
    labels = torch.tensor([3, 7], dtype=torch.uint8)
    pixels = torch.randint(0, 256, (2, 3072), dtype=torch.uint8)
    rec = torch.cat([labels[:, None], pixels], 1)
    for i in range(1, 6):
        (folder / f"data_batch_{i}.bin").write_bytes(_bytes(rec))
    (folder / "test_batch.bin").write_bytes(_bytes(rec))
    train_set, test_set = load_cifar10(str(tmp_path), download=False)
    assert len(train_set) == 10 and len(test_set) == 2
    assert test_set.labels.tolist() == [3, 7]
    assert torch.equal(test_set.images[1], pixels[1].view(3, 32, 32))
    with pytest.raises(FileNotFoundError):
        load_cifar10(str(tmp_path / "missing"), download=False)


def _blobs(n, seed):
    """Synthetic 4-class images: each class tints the image a different color (learnable in a few steps)."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n,), generator=g)
    images = torch.randint(40, 160, (n, 3, 32, 32), generator=g)
    tint = torch.tensor([[60, 0, 0], [0, 60, 0], [0, 0, 60], [-30, -30, -30]])
    images = images + tint[labels][:, :, None, None]
    return TensorDataset(images.clamp(0, 255).to(torch.uint8), labels)


@pytest.mark.slow
def test_train_loop_learns_and_keeps_ternary_codes_healthy(tmp_path):
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2, embed_dim=96, num_heads=2, num_classes=4)
    cfg = TrainConfig(epochs=3, batch_size=32, lr=2e-3, warmup_epochs=0.5, log_every=0)
    hist = train(model, _blobs(256, 0), _blobs(128, 1), cfg, out_dir=str(tmp_path), log=lambda s: None)
    last = hist["epochs"][-1]
    assert last["test_acc"] > 0.9
    assert 0.1 < last["ternary"]["zero"] < 0.6  # no collapse of the codes to zero
    assert last["weight_decay"] == 0.0  # stage 2 switched weight decay off
    assert (tmp_path / "last.pt").exists() and (tmp_path / "metrics.json").exists()


@pytest.mark.slow
def test_distilled_training_from_full_precision_init():
    torch.manual_seed(0)
    teacher = create_model("bitvit_cifar_tiny", depth=2, embed_dim=96, num_heads=2, num_classes=4, quantize=False)
    train(teacher, _blobs(256, 0), None, TrainConfig(epochs=2, batch_size=32, warmup_epochs=0.5, log_every=0),
          log=lambda s: None)
    student = create_model("bitvit_cifar_tiny", depth=2, embed_dim=96, num_heads=2, num_classes=4)
    student.load_state_dict(teacher.state_dict())
    cfg = TrainConfig(epochs=2, batch_size=32, lr=1e-3, quant_ramp_epochs=1.0, warmup_epochs=0.5, log_every=0)
    hist = train(student, _blobs(256, 0), _blobs(128, 1), cfg, teacher=teacher, log=lambda s: None)
    assert hist["epochs"][-1]["test_acc"] > 0.9
    assert all(m.quant_strength == 1.0 for m in student.modules() if isinstance(m, BitLinear))


def test_evaluate_and_ternary_stats():
    model = create_model("bitvit_cifar_tiny", depth=1, num_classes=4)
    res = evaluate(model, _blobs(20, 2), batch_size=8)
    assert 0.0 <= res["acc"] <= 1.0 and res["loss"] > 0
    stats = ternary_stats(model)
    assert sum(stats.values()) == pytest.approx(1.0)


def test_bf16_autocast_training_step():
    model = create_model("bitvit_cifar_tiny", depth=1, embed_dim=96, num_heads=2, num_classes=4)
    cfg = TrainConfig(epochs=1, batch_size=16, amp="bf16", schedule="cosine", warmup_epochs=0.5, log_every=0)
    hist = train(model, _blobs(32, 0), _blobs(16, 1), cfg, log=lambda s: None)
    assert hist["epochs"][0]["train_loss"] > 0


@pytest.mark.slow
def test_fp_first_then_ternary_and_late_4bit_switch():
    from bitnet_vision.layers import HBitLinear

    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=1, embed_dim=96, num_heads=2, num_classes=4, hadamard=True)
    cfg = TrainConfig(epochs=4, batch_size=32, fp_fraction=0.5, final_act_bits=4, final_act_bits_scope="rotated",
                      final_act_bits_fraction=0.25, log_every=0)
    hist = train(model, _blobs(64, 0), _blobs(32, 1), cfg, log=lambda s: None)
    strengths = [e["quant_strength"] for e in hist["epochs"]]
    assert strengths == [0.0, 0.0, 1.0, 1.0]
    h_layers = [m for m in model.modules() if isinstance(m, HBitLinear)]
    assert h_layers and all(m.act_bits == 4 and m.act_method == "absmean" for m in h_layers)
    assert model.config.hadamard_act_bits == 4 and hist["model_config"]["hadamard_act_bits"] == 4
    assert all(m.act_bits == 8 for m in model.modules() if isinstance(m, BitLinear) and not isinstance(m, HBitLinear))


@pytest.mark.slow
def test_final_low_bit_switch_scope_all_changes_every_layer_and_the_config():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=1, embed_dim=96, num_heads=2, num_classes=4, hadamard=True)
    cfg = TrainConfig(epochs=2, batch_size=32, final_act_bits=4, final_act_bits_fraction=0.5, warmup_epochs=0.5,
                      log_every=0)
    train(model, _blobs(64, 0), None, cfg, log=lambda s: None)
    layers = [m for m in model.modules() if isinstance(m, BitLinear)]
    assert all(m.act_bits == 4 and m.act_method == "absmean" for m in layers)
    assert model.config.act_bits == 4 and model.config.hadamard_act_bits == 4
