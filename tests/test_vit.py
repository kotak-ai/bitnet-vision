import pytest
import torch
from torch import nn

from bitnet_vision.checkpoint import load_model, save_model, transfer_weights
from bitnet_vision.layers import BitLinear, HBitLinear
from bitnet_vision.models import MODEL_PRESETS, BitViT, BitViTConfig, create_model
from bitnet_vision.training.optim import param_groups


def test_cifar_forward_shape_and_param_count():
    model = create_model("bitvit_cifar_tiny")
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    n = sum(p.numel() for p in model.parameters())
    assert 1.7e6 < n < 1.9e6


@pytest.mark.parametrize("name", sorted(MODEL_PRESETS))
def test_presets_build(name):
    cfg = BitViTConfig(**MODEL_PRESETS[name])
    model = BitViT(cfg, depth=1)  # one block keeps the test fast
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    assert model(x).shape == (1, cfg.num_classes)


def test_only_transformer_linears_are_ternary():
    model = create_model("bitvit_cifar_tiny")
    ternary = {n for n, m in model.named_modules() if isinstance(m, BitLinear)}
    assert ternary and all(n.startswith("blocks.") for n in ternary)
    assert len(ternary) == 4 * model.config.depth
    assert isinstance(model.patch_embed.proj, nn.Conv2d)
    assert type(model.head) is nn.Linear
    assert isinstance(model.norm, nn.LayerNorm)


def test_hadamard_option_uses_hbitlinear_for_proj_and_fc2_only():
    model = create_model("bitvit_cifar_tiny", hadamard=True, hadamard_act_bits=4)
    for blk in model.blocks:
        assert isinstance(blk.attn.proj, HBitLinear) and isinstance(blk.mlp.fc2, HBitLinear)
        assert not isinstance(blk.attn.qkv, HBitLinear) and not isinstance(blk.mlp.fc1, HBitLinear)
        assert blk.mlp.fc2.act_bits == 4 and blk.attn.qkv.act_bits == 8
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_full_precision_twin_has_identical_parameter_names():
    q = create_model("bitvit_cifar_tiny")
    fp = create_model("bitvit_cifar_tiny", quantize=False)
    assert q.state_dict().keys() == fp.state_dict().keys()
    assert all(m.quant_strength == 0.0 for m in fp.modules() if isinstance(m, BitLinear))


def test_transfer_weights_rotates_into_hadamard_layers():
    torch.manual_seed(0)
    src = create_model("bitvit_cifar_tiny", depth=2, quantize=False).double().eval()
    dst = create_model("bitvit_cifar_tiny", depth=2, quantize=False, hadamard=True).double().eval()
    transfer_weights(dst, src)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(dst(x), src(x))


def test_save_and_load_roundtrip(tmp_path):
    model = create_model("bitvit_cifar_tiny", depth=2, hadamard=True).eval()
    path = tmp_path / "m.pt"
    save_model(model, str(path))
    loaded = load_model(str(path)).eval()
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(loaded(x), model(x))
    as_fp = load_model(str(path), quantize=False)
    assert all(m.quant_strength == 0.0 for m in as_fp.modules() if isinstance(m, BitLinear))


def test_weight_decay_groups_skip_norms_biases_and_embeddings():
    model = create_model("bitvit_cifar_tiny", depth=1)
    decay, no_decay = param_groups(model, 0.05)
    names = {id(p): n for n, p in model.named_parameters()}
    decayed = {names[id(p)] for p in decay["params"]}
    assert "pos_embed" not in decayed and "cls_token" not in decayed
    assert not any(n.endswith("bias") or "norm" in n for n in decayed)
    assert "blocks.0.attn.qkv.weight" in decayed and "patch_embed.proj.weight" in decayed
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0


def test_block_norms_are_on_by_default():
    model = create_model("bitvit_cifar_tiny", depth=1)
    assert isinstance(model.blocks[0].norm1, nn.LayerNorm) and isinstance(model.blocks[0].norm2, nn.LayerNorm)


def test_mean_pool_relu2_layer_scale_and_no_block_norm_options():
    model = create_model("bitvit_cifar_tiny", depth=1, pool="mean", mlp_act="relu2", block_norm=False,
                         layer_scale_init=1e-4)
    assert isinstance(model.blocks[0].norm1, nn.Identity)
    assert torch.allclose(model.blocks[0].ls1.gamma, torch.full((192,), 1e-4))
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_invalid_config():
    with pytest.raises(ValueError):
        BitViTConfig(embed_dim=100, num_heads=3)
    with pytest.raises(KeyError):
        create_model("nope")


def test_gradients_reach_every_parameter():
    model = create_model("bitvit_cifar_tiny", depth=2, hadamard=True)
    model(torch.randn(4, 3, 32, 32)).logsumexp(-1).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, missing


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    model = create_model("bitvit_cifar_tiny", depth=2)
    x, y = torch.randn(16, 3, 32, 32), torch.arange(16) % 10
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(60):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (model(x).argmax(-1) == y).float().mean() >= 0.9
