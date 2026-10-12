"""LoRA adapters (bitnet_vision/lora.py): the branch, mixing, training, export, checkpoints and PEFT files."""

import copy
import json
import os
import pickle
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bitnet_vision.apple.mps_kernels import FusedBitLinearInference, convert_to_fused
from bitnet_vision.checkpoint import load_model, load_packed, save_model, save_packed, transfer_weights
from bitnet_vision.export import BitLinearInference, convert_for_inference
from bitnet_vision.layers import BitLinear, HBitLinear, convert_linear_layers, set_quant_strength
from bitnet_vision.lora import (
    DEFAULT_TARGETS,
    adapter_names,
    add_lora,
    add_lora_adapter,
    add_lora_from_config,
    delete_lora,
    disable_lora,
    enable_lora,
    load_lora,
    lora_config,
    lora_layers,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora_into_weights,
    save_lora,
    set_lora_strength,
)
from bitnet_vision.models import create_model
from bitnet_vision.training import TensorDataset, TrainConfig, build_optimizer, evaluate, train
from tests.test_scripts import _fake_cifar, _load_script

ALL = DEFAULT_TARGETS + ("head",)


def _randomize(model, seed=0, scale=0.05):
    """Non-zero A and B for every adapter (a new adapter adds exactly 0)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, m in lora_layers(model):
            for k in m.lora_settings:
                for w in (m.lora_A[k].weight, m.lora_B[k].weight):
                    w.copy_(torch.randn(w.shape, generator=g, dtype=torch.float64) * scale)


def _tokens(k, dtype=torch.float32, seed=0):
    """Tokens whose sizes differ by orders of magnitude, so the raw and the normalized input differ a lot."""
    g = torch.Generator().manual_seed(seed)
    sizes = torch.exp(2 * torch.randn(3, 7, 1, generator=g, dtype=torch.float64))
    return (torch.randn(3, 7, k, generator=g, dtype=torch.float64) * sizes).to(dtype)


def _images(n=2, dtype=torch.float32):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(1), dtype=torch.float64).to(dtype)


def _weights(layer, name="default"):
    return layer.lora_A[name].weight, layer.lora_B[name].weight


LAYERS = {
    "bitlinear": lambda: BitLinear(64, 48),
    "bitlinear_channel_nobias": lambda: BitLinear(64, 48, bias=False, weight_granularity="channel"),
    "hbitlinear": lambda: HBitLinear(64, 48),
    "linear": lambda: nn.Linear(64, 48),
    "packed": lambda: convert_for_inference(BitLinear(64, 48)),
    "fused": lambda: convert_to_fused(HBitLinear(64, 48)),
}


@pytest.mark.parametrize("make", LAYERS.values(), ids=list(LAYERS))
def test_a_new_adapter_changes_nothing(make):
    torch.manual_seed(0)
    layer = make()
    x = _tokens(64)
    with torch.no_grad():
        before = layer(x)
    assert add_lora(layer, r=4, targets=None) == [""]
    s = layer.lora_settings["default"]
    assert (s.r, s.alpha, s.scale, s.strength, s.enabled) == (4, 16.0, 4.0, 1.0, True)
    a, b = _weights(layer)
    assert a.shape == (4, 64) and b.shape == (48, 4) and a.dtype == b.dtype == torch.float32
    assert not b.any()
    assert a.abs().max() <= 1 / 8 and a.std() > 0.05  # PEFT's kaiming_uniform_(a=sqrt(5)): U(-1/sqrt(in), 1/sqrt(in))
    with torch.no_grad():
        assert torch.equal(layer(x), before)


def test_a_new_adapter_leaves_a_model_unchanged():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2).eval()
    x = _images()
    with torch.no_grad():
        before = model(x)
        add_lora(model, targets=ALL)
        assert torch.equal(model(x), before)


@pytest.mark.parametrize("kind", ["bitlinear", "hbitlinear", "linear"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("granularity", ["tensor", "channel"])
@pytest.mark.parametrize("rslora", [False, True])
def test_branch_equals_the_formula_in_float64(kind, bias, granularity, rslora):
    """y = base(x) + strength * scale * B A x on the RAW input, also for the gradients."""
    if kind == "linear" and granularity == "channel":
        pytest.skip("nn.Linear has no weight scales")
    torch.manual_seed(0)
    cls = {"bitlinear": BitLinear, "hbitlinear": HBitLinear}.get(kind)
    layer = nn.Linear(64, 48, bias=bias) if cls is None else cls(64, 48, bias=bias, weight_granularity=granularity)
    layer = layer.double()
    if bias:
        with torch.no_grad():
            layer.bias.normal_()
    add_lora(layer, r=4, alpha=6.0, targets=None, use_rslora=rslora)
    _randomize(layer, scale=0.3)
    set_lora_strength(layer, 0.75)
    scale = 6.0 / (2.0 if rslora else 4.0)
    assert layer.lora_settings["default"].scale == scale
    a, b = _weights(layer)
    x = _tokens(64, torch.float64).requires_grad_()

    y = layer(x)
    disable_lora(layer)
    expected = layer(x) + 0.75 * scale * ((x @ a.T) @ b.T)
    enable_lora(layer)
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)

    up = torch.randn_like(y)
    inputs = [a, b, x, layer.weight]
    grads = torch.autograd.grad((y * up).sum(), inputs)
    expected_grads = torch.autograd.grad((expected * up).sum(), inputs)
    for g, e in zip(grads, expected_grads):
        torch.testing.assert_close(g, e, rtol=1e-12, atol=1e-12)


def test_dropout_acts_only_in_training_mode():
    torch.manual_seed(0)
    layer = BitLinear(64, 48).double()
    add_lora(layer, targets=None, dropout=0.5)
    _randomize(layer)
    x = _tokens(64, torch.float64)
    a, b = _weights(layer)
    with torch.no_grad():
        layer.eval()
        y = layer(x)
        disable_lora(layer)
        torch.testing.assert_close(y, layer(x) + 2.0 * ((x @ a.T) @ b.T), rtol=1e-12, atol=1e-12)
        enable_lora(layer)
        layer.train()
        assert not torch.equal(layer(x), y)


def test_strength_enabling_and_several_adapters_combine_linearly():
    torch.manual_seed(0)
    layer = BitLinear(64, 48).double()
    add_lora(layer, r=4, name="a", targets=None)
    add_lora(layer, r=2, alpha=3, name="b", targets=None)
    assert adapter_names(layer) == ["a", "b"]
    _randomize(layer)
    x = _tokens(64, torch.float64)
    with torch.no_grad():
        disable_lora(layer)
        base = layer(x)
        enable_lora(layer, "a")
        da = layer(x) - base
        disable_lora(layer, "a")
        enable_lora(layer, "b")
        db = layer(x) - base
        enable_lora(layer)
        assert da.abs().max() > 1e-2 and db.abs().max() > 1e-2
        torch.testing.assert_close(layer(x), base + da + db, rtol=1e-12, atol=1e-12)
        set_lora_strength(layer, 0.5, "a")
        set_lora_strength(layer, -2.0, "b")
        torch.testing.assert_close(layer(x), base + 0.5 * da - 2.0 * db, rtol=1e-12, atol=1e-12)
        set_lora_strength(layer, 0.0, None)
        assert torch.equal(layer(x), base)
    with pytest.raises(ValueError, match="no adapter named 'c'"):
        set_lora_strength(layer, 1.0, "c")


def test_disabling_or_deleting_adapters_restores_the_base_model_exactly():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2).eval()
    x = _images()
    keys, modules = list(model.state_dict()), [n for n, _ in model.named_modules()]
    with torch.no_grad():
        before = model(x)
        add_lora(model, targets=ALL)
        assert add_lora(model, name="b", r=2, targets=("fc1",)) == ["blocks.0.mlp.fc1", "blocks.1.mlp.fc1"]
        _randomize(model)
        assert not torch.allclose(model(x), before)
        disable_lora(model)
        assert torch.equal(model(x), before)
        enable_lora(model)
        assert delete_lora(model, "b") == ["blocks.0.mlp.fc1", "blocks.1.mlp.fc1"]
        assert adapter_names(model) == ["default"]
        assert len(delete_lora(model, None)) == 9
        assert torch.equal(model(x), before)
    assert list(model.state_dict()) == keys and [n for n, _ in model.named_modules()] == modules
    assert adapter_names(model) == []
    with pytest.raises(ValueError, match="no LoRA adapters"):
        delete_lora(model, None)


def test_targets_suffixes_regex_and_errors():
    model = create_model("bitvit_cifar_tiny", depth=2)
    # patch_embed.proj also ends in "proj", but it is a Conv2d and is skipped
    assert add_lora(model, targets=("proj",)) == ["blocks.0.attn.proj", "blocks.1.attn.proj"]
    assert add_lora(model, name="re", targets=r"blocks\.1\..*fc\d") == ["blocks.1.mlp.fc1", "blocks.1.mlp.fc2"]
    every = add_lora(model, name="all", targets=None)  # not the adapters' own Linear layers
    assert len(every) == 9 and every[-1] == "head"
    with pytest.raises(ValueError, match="already exists"):
        add_lora(model, targets=("proj",))
    with pytest.raises(ValueError, match="no supported layer"):
        add_lora(model, name="x", targets=("nothing",))
    for bad in ("a.b", "", "keys"):
        with pytest.raises(ValueError, match="invalid adapter name"):
            add_lora(model, name=bad)
    with pytest.raises(ValueError, match="rank"):
        add_lora(model, name="z", r=0)
    assert "z" not in adapter_names(model)
    with pytest.raises(TypeError):
        add_lora_adapter(nn.Conv2d(3, 3, 1))
    with pytest.raises(TypeError):
        add_lora_adapter(nn.MultiheadAttention(8, 2).out_proj)  # an nn.Linear subclass that is never called


def test_lora_state_dict_and_config():
    model = create_model("bitvit_cifar_tiny", depth=1)
    add_lora(model, r=4, targets=("qkv", "head"))
    add_lora(model, r=2, name="b", targets=("fc2",))
    sd = lora_state_dict(model)
    assert list(sd) == ["blocks.0.attn.qkv.lora_A.default.weight", "blocks.0.attn.qkv.lora_B.default.weight",
                        "head.lora_A.default.weight", "head.lora_B.default.weight"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in sd.items())
    assert set(lora_state_dict(model, "b")) == {"blocks.0.mlp.fc2.lora_A.b.weight", "blocks.0.mlp.fc2.lora_B.b.weight"}
    config = lora_config(model)
    assert list(config) == ["blocks.0.attn.qkv", "blocks.0.mlp.fc2", "head"]
    assert config["blocks.0.mlp.fc2"] == {"b": {"r": 2, "alpha": 16.0, "dropout": 0.0, "use_rslora": False,
                                                "strength": 1.0, "enabled": True}}


def test_mark_only_lora_trainable():
    model = create_model("bitvit_cifar_tiny", depth=1)
    add_lora(model, name="a")
    add_lora(model, name="b", targets=("fc1",))
    names = mark_only_lora_trainable(model, names="a", extra=("head",))
    expected = {k for k in model.state_dict() if ".lora_A.a." in k or ".lora_B.a." in k} | {"head.weight", "head.bias"}
    assert set(names) == expected and len(expected) == 10
    assert {n for n, p in model.named_parameters() if p.requires_grad} == expected
    assert len(mark_only_lora_trainable(model)) == 10  # both adapters, nothing else
    with pytest.raises(ValueError, match="unknown adapters"):
        mark_only_lora_trainable(model, names=["c"])
    with pytest.raises(ValueError, match="match no parameter"):
        mark_only_lora_trainable(model, extra=("heads",))
    with pytest.raises(ValueError, match="no LoRA adapters"):
        mark_only_lora_trainable(create_model("bitvit_cifar_tiny", depth=1))


def _blobs(n, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n,), generator=g)
    images = torch.randint(40, 160, (n, 3, 32, 32), generator=g)
    tint = torch.tensor([[60, 0, 0], [0, 60, 0], [0, 0, 60], [-30, -30, -30]])
    return TensorDataset((images + tint[labels][:, :, None, None]).clamp(0, 255).to(torch.uint8), labels)


def test_training_only_the_adapter_changes_nothing_else():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=1, embed_dim=96, num_heads=2, num_classes=4)
    add_lora(model, r=4, targets=ALL)
    trainable = mark_only_lora_trainable(model)
    assert trainable and all("lora_A" in n or "lora_B" in n for n in trainable)
    optimizer = build_optimizer(model, 1e-3, 0.1)
    in_optimizer = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in in_optimizer} == {id(p) for p in model.parameters() if p.requires_grad}
    assert len(in_optimizer) == len(trainable)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    data = _blobs(128, 0)
    before = evaluate(model, data)["loss"]
    cfg = TrainConfig(epochs=3, batch_size=32, lr=1e-2, schedule="cosine", warmup_epochs=0.5, log_every=0)
    train(model, data, None, cfg, log=lambda s: None)
    after = evaluate(model, data)
    assert after["loss"] < before / 2 and after["acc"] > 0.9
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    assert all(model.state_dict()[n].any() for n in trainable if "lora_B" in n)


@pytest.mark.parametrize("overrides", [{}, {"hadamard": True, "weight_granularity": "channel"}])
def test_export_keeps_adapters_and_equals_the_training_model_in_float64(overrides):
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2, **overrides).double().eval()
    add_lora(model, targets=ALL)
    add_lora(model, r=2, name="b", targets=("fc1",))
    set_lora_strength(model, 0.7, "b")
    _randomize(model, scale=0.1)
    x = _images(dtype=torch.float64)
    packed = convert_for_inference(model, int_matmul="never")
    assert lora_config(packed) == lora_config(model)
    assert sum(isinstance(m, BitLinearInference) for _, m in lora_layers(packed)) == 8
    with torch.no_grad():
        y = model(x)
        torch.testing.assert_close(packed(x), y, rtol=1e-8, atol=1e-8)
        fused = convert_to_fused(packed)
        assert sum(isinstance(m, FusedBitLinearInference) for _, m in lora_layers(fused)) == 8
        assert torch.equal(fused(x), packed(x))  # float64 runs BitLinearInference's forward: adapters once
        disable_lora(packed)
        assert (packed(x) - y).abs().max() > 1e-3  # the adapters matter


def test_fused_layers_apply_adapters_like_the_packed_layers():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2, hadamard=True).eval()
    add_lora(model, targets=ALL)
    _randomize(model, scale=0.1)
    packed = convert_for_inference(model)
    fused = convert_to_fused(packed)
    x = _images()
    with torch.no_grad():
        assert torch.equal(fused(x), packed(x))
        layer = fused.blocks[0].attn.qkv
        set_lora_strength(layer, 0.0)
        assert not torch.equal(fused(x), packed(x))
    single = FusedBitLinearInference.from_inference(packed.blocks[0].mlp.fc1)
    assert single.lora_A is packed.blocks[0].mlp.fc1.lora_A  # shares the adapters like the buffers


def test_training_checkpoint_round_trip_keeps_adapters(tmp_path):
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2).eval()
    add_lora(model, targets=ALL)
    add_lora(model, r=2, name="b", targets=("fc2",), use_rslora=True)
    add_lora(model, r=3, name="c", targets=("qkv",))
    _randomize(model)
    set_lora_strength(model, 0.25, "b")
    disable_lora(model, "c")
    path = str(tmp_path / "model.pt")
    save_model(model, path)
    assert torch.load(path, weights_only=True)["lora"] == lora_config(model)
    loaded = load_model(path).eval()
    assert lora_config(loaded) == lora_config(model)
    x = _images()
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))
    plain = str(tmp_path / "plain.pt")
    save_model(create_model("bitvit_cifar_tiny", depth=1), plain)
    assert "lora" not in torch.load(plain, weights_only=True)  # unchanged format without adapters


def test_packed_checkpoint_round_trip_keeps_adapters(tmp_path):
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2, hadamard=True).eval()
    add_lora(model, targets=ALL)
    add_lora(model, r=2, name="b", targets=("proj",))
    _randomize(model)
    set_lora_strength(model, -0.5, "b")
    path = str(tmp_path / "packed.pt")
    payload = save_packed(model, path)
    assert payload["lora"] == lora_config(model)
    loaded = load_packed(path, int_matmul="never")
    x = _images()
    with torch.no_grad():
        assert torch.equal(loaded(x), convert_for_inference(model, int_matmul="never")(x))
    assert lora_config(loaded) == lora_config(model)


def test_transfer_weights_copies_adapters_into_a_rotated_model():
    torch.manual_seed(0)
    src = create_model("bitvit_cifar_tiny", depth=1).double().eval()
    add_lora(src, targets=ALL)
    _randomize(src, scale=0.2)
    dst = create_model("bitvit_cifar_tiny", depth=1, hadamard=True).double().eval()
    transfer_weights(dst, src)
    assert lora_config(dst) == lora_config(src)
    set_quant_strength(src, 0.0)
    set_quant_strength(dst, 0.0)
    x = _images(dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(dst(x), src(x), rtol=1e-9, atol=1e-9)


def test_adapters_survive_deepcopy_pickle_and_state_dict_round_trips():
    torch.manual_seed(0)
    model = nn.Sequential(BitLinear(16, 16), nn.GELU(), nn.Linear(16, 8))
    add_lora(model, r=2, targets=None)
    _randomize(model, scale=0.3)
    x = torch.randn(5, 16)
    with torch.no_grad():
        y = model(x)
        for clone in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
            assert torch.equal(clone(x), y)
            set_lora_strength(clone[2], 0.0)  # the plain Linear's hook reads the clone's own settings
            assert not torch.equal(clone(x), y) and torch.equal(model(x), y)
        fresh = nn.Sequential(BitLinear(16, 16), nn.GELU(), nn.Linear(16, 8))
        add_lora_from_config(fresh, lora_config(model))
        fresh.load_state_dict(model.state_dict())
        assert torch.equal(fresh(x), y)
        h = torch.randn(3, 16)
        assert torch.equal(model[2](input=h), model[2](h))  # keyword call reaches the hook too
        delete_lora(model)
        assert torch.equal(model[2](h), F.linear(h, model[2].weight, model[2].bias))
        add_lora(model, r=2, targets=None)  # re-adding must not stack a second hook
        _randomize(model, scale=0.3)
        assert torch.equal(model(x), y)


def test_converting_linear_layers_carries_adapters_and_keeps_adapter_layers_full_precision():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 8)).double()
    add_lora(model, r=2, targets=None)
    _randomize(model, scale=0.3)
    x = torch.randn(5, 16, dtype=torch.float64)
    with torch.no_grad():
        delta = model[0](x) - F.linear(x, model[0].weight, model[0].bias)
        convert_linear_layers(model, exclude=("2",))
        assert isinstance(model[0], BitLinear) and type(model[2]) is nn.Linear
        assert type(model[0].lora_A["default"]) is nn.Linear  # the adapter is not ternarized
        y = model[0](x)
        disable_lora(model)
        torch.testing.assert_close(y - model[0](x), delta, rtol=1e-12, atol=1e-12)
    # converting again, or a model whose ternary layers already carry adapters, leaves the adapters alone
    convert_linear_layers(model)
    assert isinstance(model[2], BitLinear)
    assert all(type(m) is nn.Linear for n, m in model.named_modules() if n.endswith(".default"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_branch_runs_in_float32_for_half_inputs(dtype):
    torch.manual_seed(0)
    layer = BitLinear(64, 48).to(dtype)
    add_lora(layer, targets=None)
    _randomize(layer)
    a, b = _weights(layer)
    assert a.dtype == dtype
    x = _tokens(64).to(dtype)
    with torch.no_grad():
        y = layer(x)
        disable_lora(layer)
        base = layer(x)
        expected = (base.float() + F.linear(F.linear(x.float(), a.float()), b.float()) * 2.0).to(dtype)
    assert y.dtype == dtype and torch.equal(y, expected)


def test_branch_stays_float32_under_bf16_autocast():
    torch.manual_seed(0)
    model = nn.Sequential(BitLinear(64, 48), nn.Linear(48, 8))
    add_lora(model, targets=None)
    _randomize(model)
    x = _tokens(64)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        h = model[0](x)
        y = model[1](h)
        disable_lora(model)
        h_base, y_base = model[0](x), model[1](h)
    a0, b0 = _weights(model[0])
    a1, b1 = _weights(model[1])
    assert h.dtype == y.dtype == torch.bfloat16
    assert torch.equal(h, (h_base.float() + F.linear(F.linear(x, a0), b0) * 2.0).bfloat16())
    assert torch.equal(y, (y_base.float() + F.linear(F.linear(h.float(), a1), b1) * 2.0).bfloat16())


@pytest.mark.slow
def test_torch_compile_matches_eager():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=1, embed_dim=96, num_heads=2).eval()
    add_lora(model, targets=ALL)
    _randomize(model, scale=0.1)
    x = _images()
    torch._dynamo.reset()
    with torch.no_grad():
        torch.testing.assert_close(torch.compile(model, backend="aot_eager")(x), model(x), rtol=1e-5, atol=1e-5)
        packed = convert_for_inference(model, int_matmul="never")
        torch.testing.assert_close(torch.compile(packed, backend="aot_eager")(x), packed(x), rtol=1e-5, atol=1e-5)
    model.train()
    compiled = torch.compile(model, backend="aot_eager")
    compiled(x).square().sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if "lora" in n}
    model.zero_grad()
    model(x).square().sum().backward()
    for n, g in grads.items():
        torch.testing.assert_close(g, model.get_parameter(n).grad, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------------------------------------
# PEFT-compatible files


@pytest.mark.parametrize("safe", [True, False])
def test_save_and_load_lora_round_trip(tmp_path, safe):
    if safe:
        pytest.importorskip("safetensors")
    torch.manual_seed(0)
    base = create_model("bitvit_cifar_tiny", depth=2).eval()
    tuned = copy.deepcopy(base)
    add_lora(tuned, r=4, alpha=8, targets=ALL, use_rslora=True, dropout=0.1)
    _randomize(tuned)
    folder = str(tmp_path / "adapter")
    save_lora(tuned, folder, safe_serialization=safe)
    weights = "adapter_model.safetensors" if safe else "adapter_model.bin"
    assert sorted(os.listdir(folder)) == ["adapter_config.json", weights]
    x = _images()
    assert load_lora(base, folder) == [n for n, _ in lora_layers(tuned)]
    assert lora_config(base) == lora_config(tuned)
    with torch.no_grad():
        assert torch.equal(base(x), tuned(x))
        # the weights file itself also works (config read from next to it), and the strength is chosen on load
        load_lora(base, os.path.join(folder, weights), name="second", strength=0.5)
        set_lora_strength(tuned, 1.5)
        assert torch.allclose(base(x), tuned(x), rtol=1e-4, atol=1e-5)


def test_saving_one_kind_of_weights_file_removes_the_other(tmp_path):
    pytest.importorskip("safetensors")
    model = create_model("bitvit_cifar_tiny", depth=1)
    add_lora(model)
    folder = str(tmp_path / "adapter")
    for safe, kept in [(True, "adapter_model.safetensors"), (False, "adapter_model.bin"),
                       (None, "adapter_model.safetensors")]:
        save_lora(model, folder, safe_serialization=safe)
        assert sorted(os.listdir(folder)) == ["adapter_config.json", kept]


def test_without_safetensors_the_bin_file_is_written(tmp_path, monkeypatch):
    import bitnet_vision.lora as lora_module

    monkeypatch.setattr(lora_module, "_safetensors_torch", lambda: None)
    model = create_model("bitvit_cifar_tiny", depth=1)
    add_lora(model)
    save_lora(model, str(tmp_path / "a"))
    assert sorted(os.listdir(tmp_path / "a")) == ["adapter_config.json", "adapter_model.bin"]
    with pytest.raises(ImportError, match="safetensors"):
        save_lora(model, str(tmp_path / "b"), safe_serialization=True)


def _read_weights(folder):
    if os.path.exists(os.path.join(folder, "adapter_model.safetensors")):
        from safetensors.torch import load_file

        return load_file(os.path.join(folder, "adapter_model.safetensors"))
    return torch.load(os.path.join(folder, "adapter_model.bin"), weights_only=True)


def test_saved_files_have_peft_layout(tmp_path):
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2)
    add_lora(model, r=4, alpha=8, targets=("qkv", "proj", "fc1"))
    add_lora(model, r=2, alpha=4, targets=("fc2",))  # same adapter, other rank: PEFT's rank_pattern
    add_lora(model, r=3, name="other", targets=("qkv",))  # not saved
    _randomize(model)
    folder = str(tmp_path / "adapter")
    config = save_lora(model, folder)
    with open(os.path.join(folder, "adapter_config.json")) as f:
        assert json.load(f) == config
    layers = [n for n, _ in lora_layers(model, "default")]
    fc2 = ["blocks.0.mlp.fc2", "blocks.1.mlp.fc2"]
    assert {k: config[k] for k in ("peft_type", "r", "lora_alpha", "target_modules", "lora_dropout", "use_rslora",
                                   "bias", "rank_pattern", "alpha_pattern", "modules_to_save")} == {
        "peft_type": "LORA", "r": 4, "lora_alpha": 8.0, "target_modules": layers, "lora_dropout": 0.0,
        "use_rslora": False, "bias": "none", "rank_pattern": {n: 2 for n in fc2},
        "alpha_pattern": {n: 4.0 for n in fc2}, "modules_to_save": None}
    state = _read_weights(folder)
    expected = {f"base_model.model.{n}.lora_{p}.weight" for n in layers for p in "AB"}
    assert set(state) == expected  # no adapter name in the keys, as PEFT saves them
    for n, m in lora_layers(model, "default"):
        assert torch.equal(state[f"base_model.model.{n}.lora_A.weight"], m.lora_A["default"].weight)
        assert torch.equal(state[f"base_model.model.{n}.lora_B.weight"], m.lora_B["default"].weight)
    fresh = create_model("bitvit_cifar_tiny", depth=2)
    load_lora(fresh, folder)
    assert {n: s["default"]["r"] for n, s in lora_config(fresh).items()} == {n: 2 if n in fc2 else 4 for n in layers}
    assert lora_config(fresh) == {n: {"default": s["default"]} for n, s in lora_config(model).items()
                                  if "default" in s}


def test_loads_a_peft_style_state_dict_with_patterns():
    """What PEFT writes for target_modules=["qkv", "fc2"], rank_pattern={"fc2": 2}, alpha_pattern={"fc2": 2}."""
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=1).double()
    g = torch.Generator().manual_seed(0)
    state = {
        "base_model.model.blocks.0.attn.qkv.lora_A.weight": torch.randn(4, 192, generator=g),
        "base_model.model.blocks.0.attn.qkv.lora_B.weight": torch.randn(576, 4, generator=g),
        "base_model.model.blocks.0.mlp.fc2.lora_A.weight": torch.randn(2, 384, generator=g),
        "base_model.model.blocks.0.mlp.fc2.lora_B.weight": torch.randn(192, 2, generator=g),
    }
    config = {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": ["qkv", "fc2"], "bias": "none",
              "rank_pattern": {"fc2": 2}, "alpha_pattern": {"fc2": 2}, "use_dora": False, "revision": None}
    assert load_lora(model, state, config=config) == ["blocks.0.attn.qkv", "blocks.0.mlp.fc2"]
    qkv, fc2 = model.blocks[0].attn.qkv, model.blocks[0].mlp.fc2
    assert qkv.lora_settings["default"].scale == 2.0 and fc2.lora_settings["default"].scale == 1.0
    assert qkv.lora_A["default"].weight.dtype == torch.float64  # cast to the layer's dtype
    assert torch.equal(fc2.lora_B["default"].weight, state["base_model.model.blocks.0.mlp.fc2.lora_B.weight"].double())
    # in-model keys (with an adapter name) and no config: alpha = rank, so scale 1
    other = create_model("bitvit_cifar_tiny", depth=1)
    load_lora(other, lora_state_dict(model), name="copy")
    assert other.blocks[0].attn.qkv.lora_settings["copy"].alpha == 4.0


def test_files_match_the_peft_library(tmp_path):
    """Runs only where PEFT is installed: its LoRA layers wrap BitLinear like an nn.Linear, so both directions
    must give identical outputs."""
    peft = pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    torch.manual_seed(0)
    base = create_model("bitvit_cifar_tiny", depth=2).eval()
    x = _images()
    for extra in ((), ("head",)):
        ours = copy.deepcopy(base)
        add_lora(ours, r=4, alpha=8, targets=("qkv", "proj", "fc1") + ("head",) * (not extra), use_rslora=True)
        add_lora(ours, r=2, alpha=3, targets=("fc2",), use_rslora=True)
        _randomize(ours)
        with torch.no_grad():
            ours.head.bias.add_(1.0 if extra else 0.0)  # a trained head travels only as modules_to_save
        save_lora(ours, str(tmp_path / f"ours{len(extra)}"), modules_to_save=extra)
        theirs = peft.PeftModel.from_pretrained(copy.deepcopy(base), str(tmp_path / f"ours{len(extra)}")).eval()
        with torch.no_grad():
            assert torch.equal(theirs(x), ours(x))

    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=r"blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])",
                             rank_pattern={"fc2": 2}, alpha_pattern={"fc2": 3}, modules_to_save=["head"])
    theirs = peft.get_peft_model(copy.deepcopy(base), config)
    _randomize(theirs)
    with torch.no_grad():
        for n, p in theirs.named_parameters():
            if "modules_to_save" in n:
                p.add_(1.0)
    theirs.save_pretrained(str(tmp_path / "theirs"))
    mine = copy.deepcopy(base)
    assert len(load_lora(mine, str(tmp_path / "theirs"))) == 8
    with torch.no_grad():
        assert torch.equal(mine(x), theirs.eval()(x))


def test_modules_to_save_travel_with_the_adapter(tmp_path):
    torch.manual_seed(0)
    base = create_model("bitvit_cifar_tiny", depth=1).eval()
    tuned = copy.deepcopy(base)
    add_lora(tuned, targets=DEFAULT_TARGETS)
    _randomize(tuned)
    with torch.no_grad():
        tuned.head.weight.mul_(-2.0)
    config = save_lora(tuned, str(tmp_path / "a"), modules_to_save=("head",), safe_serialization=False)
    assert config["modules_to_save"] == ["head"]
    state = _read_weights(str(tmp_path / "a"))
    assert {k for k in state if "lora_" not in k} == {"base_model.model.head.weight", "base_model.model.head.bias"}
    load_lora(base, str(tmp_path / "a"))
    x = _images()
    with torch.no_grad():
        assert torch.equal(base(x), tuned(x))
    add_lora(tuned, name="h", targets=("head",))  # PEFT cannot hold a LoRA inside a module it saves whole
    with pytest.raises(ValueError, match="modules_to_save"):
        save_lora(tuned, str(tmp_path / "b"), name="h", modules_to_save=("head",))


def test_load_lora_checks_everything_before_changing_the_model():
    model = create_model("bitvit_cifar_tiny", depth=1)
    p = "base_model.model.blocks.0.attn.qkv."
    good = {p + "lora_A.weight": torch.zeros(4, 192), p + "lora_B.weight": torch.zeros(576, 4)}
    cases = [
        ({p + "lora_A.weight": torch.zeros(4, 192)}, None, "needs both"),
        ({**good, p + "lora_B.weight": torch.zeros(575, 4)}, None, "do not fit"),
        ({k.replace("blocks.0", "blocks.7"): v for k, v in good.items()}, None, "no such layer"),
        ({"patch_embed.proj.lora_A.weight": torch.zeros(4, 3), "patch_embed.proj.lora_B.weight": torch.zeros(192, 4)},
         None, "cannot take an adapter"),
        ({**good, p + "bias": torch.zeros(576)}, None, "not a LoRA weight"),
        ({**good, "head.weight": torch.zeros(10, 192)}, None, "not a LoRA weight"),  # not in modules_to_save
        ({**good, "head.weight": torch.zeros(9, 192)}, {"modules_to_save": ["head"]}, "does not fit"),
        ({"x.lora_A.a.weight": torch.zeros(1, 1), "x.lora_A.b.weight": torch.zeros(1, 1)}, None, "several adapters"),
        (good, {"r": 8}, "rank 8"),
        (good, {"peft_type": "IA3"}, "not a LoRA"),
        (good, {"bias": "all"}, "trained biases"),
        (good, {"use_dora": True}, "DoRA"),
        (good, {"lora_bias": True}, "bias on B"),
        ({}, None, "no LoRA weights"),
    ]
    for state, config, match in cases:
        with pytest.raises(ValueError, match=match):
            load_lora(model, state, config=config)
        assert adapter_names(model) == []
    load_lora(model, good)
    with pytest.raises(ValueError, match="already has an adapter"):
        load_lora(model, good)
    with pytest.raises(FileNotFoundError):
        load_lora(model, os.path.dirname(os.path.abspath(__file__)), name="x")


# ---------------------------------------------------------------------------------------------------------
# Merging


@pytest.mark.parametrize("kind", ["linear", "bitlinear", "hbitlinear"])
def test_merge_matches_the_side_branch_in_full_precision(kind):
    torch.manual_seed(0)
    if kind == "linear":
        layer = nn.Linear(64, 48)
    else:
        layer = (BitLinear if kind == "bitlinear" else HBitLinear)(64, 48, input_norm=False)
        layer.quant_strength = 0.0  # the full-precision twin
    layer = layer.double()
    add_lora(layer, r=4, targets=None)
    _randomize(layer, scale=0.3)
    set_lora_strength(layer, 0.7)
    x = _tokens(64, torch.float64)
    with torch.no_grad():
        expected = layer(x)
    if kind == "linear":
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert merge_lora_into_weights(layer) == [""]
    else:
        with pytest.warns(UserWarning, match="side branch is exact"):
            merge_lora_into_weights(layer)
    assert layer.lora_settings is None
    with torch.no_grad():
        torch.testing.assert_close(layer(x), expected, rtol=1e-10, atol=1e-10)


def test_merge_refuses_inexact_cases_without_changing_anything():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=1, quantize=False).double().eval()
    add_lora(model, targets=("qkv", "head"))
    _randomize(model, scale=0.2)
    state = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="normalizes its input"):
        merge_lora_into_weights(model)  # BitLinear's inner RMSNorm sees what the adapter does not
    assert adapter_names(model) == ["default"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    packed = convert_for_inference(create_model("bitvit_cifar_tiny", depth=1))
    add_lora(packed, targets=("qkv",))
    with pytest.raises(ValueError, match="packed"):
        merge_lora_into_weights(packed)
    with pytest.raises(ValueError, match="no adapter named"):
        merge_lora_into_weights(model, "nothing")

    delete_lora(model)
    add_lora(model, targets=("head",))  # the full-precision head merges exactly
    _randomize(model, scale=0.2)
    x = _images(dtype=torch.float64)
    with torch.no_grad():
        expected = model(x)
        disable_lora(model)
        with pytest.raises(ValueError, match="disabled"):
            merge_lora_into_weights(model)
        enable_lora(model)
        assert merge_lora_into_weights(model) == ["head"]
        torch.testing.assert_close(model(x), expected, rtol=1e-10, atol=1e-10)
    assert adapter_names(model) == []


# ---------------------------------------------------------------------------------------------------------
# scripts/train.py and scripts/eval.py


@pytest.mark.slow
def test_train_script_trains_and_saves_a_lora_adapter(tmp_path):
    data = str(tmp_path / "data")
    _fake_cifar(data)
    train_script, eval_script = _load_script("train"), _load_script("eval")
    common = ["--data-dir", data, "--epochs", "1", "--batch-size", "50", "--log-every", "0", "--warmup-epochs", "0",
              "--schedule", "cosine"]
    base = str(tmp_path / "base")
    train_script.main(common + ["--out", base])
    with pytest.raises(SystemExit, match="--init"):
        train_script.main(common + ["--out", str(tmp_path / "x"), "--lora-r", "4"])

    out = str(tmp_path / "lora")
    args = common + ["--out", out, "--init", f"{base}/model.pt", "--lora-r", "4", "--lora-train-head", "--lr", "3e-3"]
    summary = train_script.main(args)
    assert summary["lora"]["layers"] == 24 and summary["lora"]["dir"] == os.path.join(out, "lora")
    with open(os.path.join(out, "lora", "adapter_config.json")) as f:
        config = json.load(f)
    assert (config["r"], config["lora_alpha"], config["modules_to_save"]) == (4, 8.0, ["head"])

    init = torch.load(f"{base}/model.pt", weights_only=True)["model"]
    tuned = torch.load(f"{out}/model.pt", weights_only=True)
    assert len(tuned["lora"]) == 24
    for k, v in init.items():  # the base stayed frozen, except the head that was asked for
        assert torch.equal(tuned["model"][k], v) != k.startswith("head."), k
    assert all(tuned["model"][k].any() for k in tuned["model"] if "lora_B" in k)

    # the adapter file on top of the --init model reproduces the run; so does the packed checkpoint
    info = eval_script.main([f"{base}/model.pt", "--data-dir", data, "--lora", os.path.join(out, "lora")])
    assert info["lora_layers"] == 24 and info["acc"] == summary["final_test_acc"]
    info = eval_script.main([os.path.join(out, "packed.pt"), "--data-dir", data])
    assert info["acc"] == summary["packed_test_acc"]
    res = _load_script("bench").main(["--checkpoint", f"{out}/model.pt", "--batch-sizes", "2", "--threads", "1",
                                      "--repeats", "1"])
    assert res["runs"][0]["packed_vs_training_argmax_agreement"] >= 0.5

    again = train_script.main(args + ["--resume"])
    assert again["final_test_acc"] == summary["final_test_acc"]
    rank8 = common + ["--out", out, "--init", f"{base}/model.pt", "--lora-r", "8", "--lora-train-head", "--lr", "3e-3"]
    with pytest.raises(ValueError, match="LoRA adapters differ"):
        train_script.main(rank8 + ["--resume"])
    with pytest.raises(ValueError, match="trainable"):
        train_script.main([a for a in args if a != "--lora-train-head"] + ["--resume"])
    with pytest.raises(SystemExit, match="already has LoRA adapters"):
        train_script.main(common + ["--out", str(tmp_path / "y"), "--init", f"{out}/model.pt", "--lora-r", "2"])
