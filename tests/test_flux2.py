"""FLUX.2 / Bonsai Image support (bitnet_vision/generators/flux2.py) on a tiny Flux2Transformer2DModel.

Covers packing a ternary transformer, loading both Bonsai file layouts (written here in the real files' format)
and loading LoRA files of every kind diffusers reads, each compared with diffusers' own result on the dense
model in float32. Needs the 'flux' extra (diffusers, peft, safetensors); skipped without it. No network.
"""

import copy
import json
import os
import subprocess
import sys
import warnings

import pytest
import torch
from torch import nn

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict  # noqa: E402

from bitnet_vision.apple.mlx_ternary import encode_mlx_ternary  # noqa: E402
from bitnet_vision.generators import (  # noqa: E402
    FLUX2_TERNARY_TARGETS,
    load_bonsai_transformer,
    load_flux2_lora,
    round_linears_to_ternary,
    ternarize_transformer,
)
from bitnet_vision.lora import (  # noqa: E402
    delete_lora,
    load_lora,
    lora_layers,
    merge_lora_into_weights,
    save_lora,
    set_lora_strength,
)
from bitnet_vision.weight_only import PackedTernaryLinear  # noqa: E402

GROUP = 16  # divides every input width of the tiny model (32, 96, 128)
CONFIG = dict(patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=16,
              num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=32,
              axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)  # FLUX.2 klein shape, scaled down
N_TERNARY = 2 * 12 + 2 * 2  # 12 linears per double-stream block, 2 per single-stream block
# Module names of the original BFL checkpoint (and kohya's, with "_"), per diffusers module.
DOUBLE = {"attn.to_out.0": "img_attn.proj", "attn.to_add_out": "txt_attn.proj", "ff.linear_in": "img_mlp.0",
          "ff.linear_out": "img_mlp.2", "ff_context.linear_in": "txt_mlp.0", "ff_context.linear_out": "txt_mlp.2"}
FUSED = {"img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
         "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")}
SINGLE = {"attn.to_qkv_mlp_proj": "linear1", "attn.to_out": "linear2"}


@pytest.fixture(scope="module")
def dense():
    """A tiny FLUX.2 transformer whose block linears hold ternary values (group 16, bfloat16 scales) and whose
    other weights are bfloat16 values, so a bfloat16 file stores the whole model exactly."""
    torch.manual_seed(0)
    model = Flux2Transformer2DModel(**CONFIG).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.bfloat16().float())
    names = round_linears_to_ternary(model, group_size=GROUP, scale_dtype=torch.bfloat16)
    assert len(names) == N_TERNARY
    return model


@pytest.fixture(scope="module")
def inputs():
    g = torch.Generator().manual_seed(1)
    n_img, n_txt = 12, 5
    return dict(hidden_states=torch.randn(2, n_img, 8, generator=g),
                encoder_hidden_states=torch.randn(2, n_txt, 24, generator=g), timestep=torch.tensor([0.3, 0.7]),
                img_ids=torch.randint(0, 4, (n_img, 4), generator=g).float(), txt_ids=torch.zeros(n_txt, 4))


def _run(model, inputs):
    with torch.no_grad():
        return model(**inputs).sample


def _packed(dense):
    model = copy.deepcopy(dense)
    ternarize_transformer(model, group_size=GROUP)
    return model


def _ternary_names(model):
    return sorted(n for n, m in model.named_modules() if isinstance(m, PackedTernaryLinear))


# ---------------------------------------------------------------------------------------------------------
# Packing and Bonsai files


def test_ternarize_converts_exactly_the_ternary_layers_and_is_exact(dense, inputs):
    packed = copy.deepcopy(dense)
    names = ternarize_transformer(packed, group_size=GROUP)
    assert len(names) == N_TERNARY and _ternary_names(packed) == sorted(names)
    assert type(packed.single_stream_modulation.linear) is nn.Linear  # not ternary: left alone
    assert torch.equal(_run(packed, inputs), _run(dense, inputs))
    with pytest.raises(ValueError, match=r"x_embedder: group_size=16 does not divide in_features=8(.|\n)*"
                       r"single_stream_modulation\.linear is not ternary"):
        ternarize_transformer(copy.deepcopy(dense), group_size=GROUP,
                              targets=["single_stream_modulation.linear", "x_embedder", "attn.to_q"])
    assert ternarize_transformer(copy.deepcopy(dense), group_size=GROUP, targets=FLUX2_TERNARY_TARGETS) == names


def _config_file(model, folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump({**dict(model.config), "_class_name": "Flux2Transformer2DModel"}, f)


def _write_unpacked(model, root):
    """Bonsai's unpacked layout: transformer/ with config.json and bfloat16 weights under diffusers' names."""
    folder = os.path.join(root, "transformer")
    _config_file(model, folder)
    safetensors_torch.save_file({k: v.bfloat16().contiguous() for k, v in model.state_dict().items()},
                                os.path.join(folder, "diffusion_pytorch_model.safetensors"))
    return folder


def _write_mlx(model, root, mutate=None):
    """Bonsai's MLX pack: transformer-packed-mflux/ with uint32 codes, bfloat16 scales and biases == -scales."""
    folder = os.path.join(root, "transformer-packed-mflux")
    _config_file(model, folder)
    packed = _packed(model)
    tensors, quantized = {}, []
    for name, m in packed.named_modules():
        if isinstance(m, PackedTernaryLinear):
            w_q, scales, biases = encode_mlx_ternary(m.codes(), m.weight_scale.bfloat16())
            tensors.update({f"{name}.weight": w_q, f"{name}.scales": scales, f"{name}.biases": biases})
            quantized.append(name)
    for k, v in model.state_dict().items():
        if k.rsplit(".", 1)[0] not in quantized:
            tensors[k] = v.bfloat16().contiguous()
    if mutate:
        mutate(tensors)
    safetensors_torch.save_file(tensors, os.path.join(folder, "diffusion_pytorch_model.safetensors"))
    with open(os.path.join(folder, "quantization_config.json"), "w") as f:
        json.dump({"format": "mlx-packed-affine", "solver": "ternary", "bits": 2, "group_size": GROUP,
                   "scale_dtype": "bfloat16", "quantized_modules": sorted(quantized)}, f)
    return folder


@pytest.mark.parametrize("layout", ["unpacked", "mlx"])
def test_bonsai_files_load_packed_and_equal_the_dense_model(dense, inputs, tmp_path, layout):
    folder = (_write_unpacked if layout == "unpacked" else _write_mlx)(dense, str(tmp_path))
    model = load_bonsai_transformer(str(tmp_path), dtype=torch.float32, group_size=GROUP)  # the repo's root
    assert _ternary_names(model) == _ternary_names(_packed(dense))
    assert all(m.weight_scale.dtype == torch.float32 for m in model.modules() if isinstance(m, PackedTernaryLinear))
    assert not model.training and all(p.device.type == "cpu" for p in model.parameters())
    assert torch.equal(_run(model, inputs), _run(dense, inputs))
    weights = os.path.join(folder, "diffusion_pytorch_model.safetensors")  # or the weights file itself
    again = load_bonsai_transformer(weights, dtype=torch.float32, group_size=GROUP)
    assert all(torch.equal(a, b) for a, b in zip(again.state_dict().values(), model.state_dict().values()))


def test_bfloat16_load_matches_diffusers_own_loader_bit_for_bit(dense, inputs, tmp_path):
    folder = _write_unpacked(dense, str(tmp_path))
    reference = Flux2Transformer2DModel.from_pretrained(folder, torch_dtype=torch.bfloat16).eval()
    _write_mlx(dense, str(tmp_path))
    bf16_inputs = {k: v.bfloat16() if k in ("hidden_states", "encoder_hidden_states") else v for k, v in inputs.items()}
    for sub in ("transformer", "transformer-packed-mflux"):
        model = load_bonsai_transformer(os.path.join(str(tmp_path), sub), group_size=GROUP)  # bfloat16 default
        assert next(model.parameters()).dtype == torch.bfloat16
        assert torch.equal(_run(model, bf16_inputs), _run(reference, bf16_inputs))


def test_loader_refuses_groups_that_are_not_ternary(dense, tmp_path):
    broken = copy.deepcopy(dense)
    with torch.no_grad():
        w = broken.transformer_blocks[1].attn.to_v.weight
        w[5, 20] = w[5, 16:32].abs().max() * 0.5
    _write_unpacked(broken, str(tmp_path))
    with pytest.raises(ValueError, match=r"transformer_blocks\.1\.attn\.to_v is not ternary.*row 5, group 1"):
        load_bonsai_transformer(str(tmp_path), dtype=torch.float32, group_size=GROUP)
    with pytest.raises(ValueError, match="does not divide"):
        load_bonsai_transformer(str(tmp_path), dtype=torch.float32, group_size=128)


def test_loader_refuses_mlx_packs_that_are_not_ternary(dense, tmp_path):
    def biases(t):
        t["transformer_blocks.0.attn.to_k.biases"] = t["transformer_blocks.0.attn.to_k.biases"] * 0.5

    def code3(t):
        w = t["single_transformer_blocks.1.attn.to_out.weight"].view(torch.int32)
        w[0, 0] |= 3

    for mutate, message in [(biases, "biases differ from -scales"), (code3, "code 3")]:
        _write_mlx(dense, str(tmp_path), mutate)
        with pytest.raises(ValueError, match=message):
            load_bonsai_transformer(str(tmp_path), dtype=torch.float32)
    folder = _write_mlx(dense, str(tmp_path))
    with open(os.path.join(folder, "quantization_config.json")) as f:
        q = json.load(f)
    q["quantized_modules"] = q["quantized_modules"][1:]
    with open(os.path.join(folder, "quantization_config.json"), "w") as f:
        json.dump(q, f)
    with pytest.raises(ValueError, match="other quantized modules"):
        load_bonsai_transformer(str(tmp_path), dtype=torch.float32)


def test_loader_refuses_models_with_buffers_the_files_do_not_hold(dense, tmp_path, monkeypatch):
    """The model is built on the meta device and filled from the file; a buffer computed in __init__ and not saved
    (persistent=False) would stay uninitialized, so such a model is refused instead."""

    build = Flux2Transformer2DModel.from_config

    def with_buffer(config, **kwargs):
        model = build(config, **kwargs)
        model.register_buffer("freqs", torch.arange(4.0), persistent=False)
        return model

    folder = _write_unpacked(dense, str(tmp_path))
    monkeypatch.setattr(Flux2Transformer2DModel, "from_config", staticmethod(with_buffer))
    with pytest.raises(NotImplementedError, match=r"buffers that its files do not hold \(\['freqs'\]\)"):
        load_bonsai_transformer(folder, dtype=torch.float32, group_size=GROUP)


def test_loader_refuses_missing_and_unexpected_tensors(dense, tmp_path):
    folder = _write_unpacked(dense, str(tmp_path))
    path = os.path.join(folder, "diffusion_pytorch_model.safetensors")
    tensors = safetensors_torch.load_file(path)
    tensors["extra.weight"] = tensors.pop("proj_out.weight")
    safetensors_torch.save_file(tensors, path)
    with pytest.raises(ValueError, match=r"lacks 1 tensors.*proj_out\.weight(.|\n)*does not have.*extra\.weight"):
        load_bonsai_transformer(folder, dtype=torch.float32, group_size=GROUP)


# ---------------------------------------------------------------------------------------------------------
# LoRA files, against diffusers' own loading on the dense model

LORA_TARGETS = (r".*(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out"
                r"|ff\.linear_in|ff\.linear_out|ff_context\.linear_in|ff_context\.linear_out|to_qkv_mlp_proj"
                r"|single_transformer_blocks\.\d+\.attn\.to_out|proj_out|x_embedder)")


def _peft_lora(dense, seed, r=4, alpha=8, **config):
    """A trained-looking PEFT LoRA on the dense model: ``(diffusers-named state dict, LoraConfig)``."""
    model = copy.deepcopy(dense)
    config = LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS, **config)
    model.add_adapter(config, adapter_name="x")
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator=g) * (0.3 if "lora_A" in name else 0.05))
    return get_peft_model_state_dict(model, adapter_name="x"), config


def _save_diffusers(folder, state, config=None):
    Flux2KleinPipeline.save_lora_weights(folder, transformer_lora_layers=state,
                                         transformer_lora_adapter_metadata=config.to_dict() if config else None)
    return os.path.join(folder, "pytorch_lora_weights.safetensors")


def _diffusers_reference(dense, loras, inputs):
    """Output of the dense model with LoRAs loaded the way ``pipe.load_lora_weights`` loads them."""
    model = copy.deepcopy(dense)
    for name, (source, _) in loras.items():
        source = dict(source) if isinstance(source, dict) else source  # diffusers' kohya converter empties dicts
        state, metadata = Flux2KleinPipeline.lora_state_dict(source, return_lora_metadata=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Already found a `peft_config`")  # PEFT, on the second adapter
            Flux2KleinPipeline.load_lora_into_transformer(state, transformer=model, adapter_name=name,
                                                          metadata=metadata)
    model.set_adapters(list(loras), [strength for _, strength in loras.values()])
    return _run(model, inputs)


def _kohya(state, scale):
    """kohya's FLUX.2 file for a PEFT LoRA (diffusers names), by kohya's rules: ``lora_unet_`` plus the BFL name
    with '_' for '.', lora_down = A, lora_up = B, alpha = scale * rank. q, k and v share one fused module:
    lora_down stacks the three A's, lora_up holds the three B's block-diagonally (kohya's split_dims)."""
    out = {}

    def put(name, a, b):
        out[f"lora_unet_{name}.lora_down.weight"] = a
        out[f"lora_unet_{name}.lora_up.weight"] = b
        out[f"lora_unet_{name}.alpha"] = torch.tensor(scale * a.shape[0])

    def ab(module):
        return state[f"{module}.lora_A.weight"], state[f"{module}.lora_B.weight"]

    for i in range(CONFIG["num_layers"]):
        prefix = f"transformer_blocks.{i}."
        for fused, parts in FUSED.items():
            a, b = zip(*(ab(prefix + p) for p in parts))
            put(f"double_blocks_{i}_{fused.replace('.', '_')}", torch.cat(a), torch.block_diag(*b))
        for diff, bfl in DOUBLE.items():
            put(f"double_blocks_{i}_{bfl.replace('.', '_')}", *ab(prefix + diff))
    for i in range(CONFIG["num_single_layers"]):
        for diff, bfl in SINGLE.items():
            put(f"single_blocks_{i}_{bfl}", *ab(f"single_transformer_blocks.{i}.{diff}"))
    put("img_in", *ab("x_embedder"))
    put("final_layer_linear", *ab("proj_out"))
    return out


def _bfl(prefix, seed, rank=4):
    """A LoRA in BFL names (fused qkv with one A, as fal's FLUX.2 klein LoRAs are stored), no alpha."""
    g = torch.Generator().manual_seed(seed)
    dim, mlp = 32, 96
    out = {}

    def put(name, n_in, n_out):
        out[f"{prefix}{name}.lora_A.weight"] = torch.randn(rank, n_in, generator=g) * 0.3
        out[f"{prefix}{name}.lora_B.weight"] = torch.randn(n_out, rank, generator=g) * 0.05

    for i in range(CONFIG["num_layers"]):
        for kind in ("img", "txt"):
            put(f"double_blocks.{i}.{kind}_attn.qkv", dim, 3 * dim)
            put(f"double_blocks.{i}.{kind}_attn.proj", dim, dim)
            put(f"double_blocks.{i}.{kind}_mlp.0", dim, 2 * mlp)
            put(f"double_blocks.{i}.{kind}_mlp.2", mlp, dim)
    for i in range(CONFIG["num_single_layers"]):
        put(f"single_blocks.{i}.linear1", dim, 3 * dim + 2 * mlp)
        put(f"single_blocks.{i}.linear2", dim + mlp, dim)
    return out


def _close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", ["patterns", "rslora", "no_metadata"])
def test_diffusers_files_load_as_diffusers_loads_them(dense, inputs, tmp_path, variant):
    kwargs = {"patterns": dict(rank_pattern={"to_qkv_mlp_proj": 2}, alpha_pattern={"proj_out": 3, "to_k": 1}),
              "rslora": dict(use_rslora=True), "no_metadata": {}}[variant]
    state, config = _peft_lora(dense, 0, **kwargs)
    path = _save_diffusers(str(tmp_path), state, None if variant == "no_metadata" else config)
    packed = _packed(dense)
    summary = load_flux2_lora(packed, str(tmp_path), name="style", strength=0.7)
    assert summary.weight_file == path and summary.format == "diffusers (transformer.)"
    assert len(summary.modules) == len(state) // 2 == N_TERNARY + 2 and not summary.skipped
    scales = set(summary.scales.values())
    assert scales == {"patterns": {2.0, 4.0, 0.75, 0.25}, "rslora": {4.0}, "no_metadata": {1.0}}[variant]
    assert summary.alpha.startswith("no alpha" if variant == "no_metadata" else "lora_adapter_metadata")
    reference = _diffusers_reference(dense, {"style": (path, 0.7)}, inputs)
    assert (reference - _run(dense, inputs)).abs().max() > 0.05  # the LoRA does something
    _close(_run(packed, inputs), reference)


def test_kohya_and_bfl_files_load_as_diffusers_loads_them(dense, inputs, tmp_path):
    state, _ = _peft_lora(dense, 1, r=4, alpha=2)
    kohya = _kohya(state, scale=0.5)
    path = str(tmp_path / "kohya.safetensors")
    safetensors_torch.save_file(kohya, path)
    packed = _packed(dense)
    summary = load_flux2_lora(packed, path, name="k")
    assert summary.format == "kohya" and set(summary.scales.values()) == {1.0}  # alpha / rank folded into A, B
    assert len(summary.modules) == N_TERNARY + 2
    reference = _diffusers_reference(dense, {"k": (path, 1.0)}, inputs)
    _close(_run(packed, inputs), reference)
    # kohya's file is the same LoRA as the PEFT one it was built from
    peft_path = _save_diffusers(str(tmp_path / "peft"), state, LoraConfig(r=4, lora_alpha=2,
                                                                          target_modules=LORA_TARGETS))
    _close(_diffusers_reference(dense, {"p": (peft_path, 1.0)}, inputs), reference)

    for prefix, fmt in [("base_model.model.", "PEFT"), ("diffusion_model.", "ai-toolkit")]:
        bfl = _bfl(prefix, seed=2)
        packed = _packed(dense)
        summary = load_flux2_lora(packed, bfl, name="b", strength=1.3)
        assert summary.format.startswith(fmt) and summary.source == "a state dict"
        assert len(summary.modules) == 2 * 12 + 2 * 2  # each fused qkv becomes q, k and v
        _close(_run(packed, inputs), _diffusers_reference(dense, {"b": (bfl, 1.3)}, inputs))


def test_several_loras_with_their_own_strengths(dense, inputs, tmp_path):
    state, config = _peft_lora(dense, 3)
    path = _save_diffusers(str(tmp_path), state, config)
    bfl = _bfl("base_model.model.", seed=4)
    packed = _packed(dense)
    load_flux2_lora(packed, path, name="style", strength=0.8)
    load_flux2_lora(packed, bfl, name="sprites", strength=-0.5)
    _close(_run(packed, inputs), _diffusers_reference(dense, {"style": (path, 0.8), "sprites": (bfl, -0.5)}, inputs))
    set_lora_strength(packed, 1.2, "sprites")
    _close(_run(packed, inputs), _diffusers_reference(dense, {"style": (path, 0.8), "sprites": (bfl, 1.2)}, inputs))
    delete_lora(packed, None)
    assert torch.equal(_run(packed, inputs), _run(dense, inputs))
    load_flux2_lora(packed, path, name="style")
    with pytest.raises(ValueError, match="already has an adapter"):
        load_flux2_lora(packed, path, name="style")


def test_text_encoder_parts_are_skipped_with_a_warning(dense, inputs):
    bfl = _bfl("base_model.model.", seed=5)
    with_te = {**{f"text_encoder.layers.0.q_proj.lora_{p}.weight": torch.ones(2, 2) for p in "AB"}, **bfl}
    packed = _packed(dense)
    with pytest.warns(UserWarning, match="skipped 2 text-encoder"):
        summary = load_flux2_lora(packed, with_te, name="b")
    assert sorted(summary.skipped) == sorted(k for k in with_te if k.startswith("text_encoder."))
    _close(_run(packed, inputs), _diffusers_reference(dense, {"b": (bfl, 1.0)}, inputs))


@pytest.mark.parametrize("extra,message", [
    ({"base_model.model.single_blocks.0.linear2.lora_magnitude_vector": torch.ones(32)}, "DoRA"),
    ({"lora_unet_single_blocks_0_linear2.dora_scale": torch.ones(1, 128)}, "DoRA"),
    ({"lora_unet_single_blocks_0_linear2.hada_w1_a": torch.ones(32, 2)}, "LoHa"),
    ({"lora_unet_single_blocks_0_linear2.lokr_w1": torch.ones(4, 4)}, "LoKr"),
    ({"base_model.model.single_blocks.0.linear2.lora_B.bias": torch.ones(32)}, "bias on LoRA's B"),
    ({"base_model.model.single_blocks.0.linear2.weird": torch.ones(3)}, "not LoRA A/B weights"),
    ({"base_model.model.single_blocks.0.linear2.alpha": torch.tensor(4.0)}, "only kohya files carry"),
    ({"base_model.model.single_blocks.1.linear9.lora_A.weight": torch.ones(4, 32)}, "lack one of their two"),
    ({"lora_unet_single_blocks_0_linear2.lora_down.weight": torch.ones(4, 128)}, "lack one of their two"),
])
def test_unsupported_adapters_are_refused_before_anything_changes(dense, extra, message):
    packed = _packed(dense)
    with pytest.raises(ValueError, match=message):
        load_flux2_lora(packed, {**_bfl("base_model.model.", seed=6), **extra})
    assert not lora_layers(packed)


def test_unmatched_modules_are_refused(dense, tmp_path):
    packed = _packed(dense)
    kohya = _kohya(_peft_lora(dense, 7)[0], scale=1.0)
    kohya["lora_unet_double_blocks_0_img_mod_lin.lora_down.weight"] = torch.ones(4, 32)  # no FLUX.2 rule for this
    kohya["lora_unet_double_blocks_0_img_mod_lin.lora_up.weight"] = torch.ones(192, 4)
    with pytest.raises(ValueError, match="did not convert.*double_blocks_0_img_mod_lin"):
        load_flux2_lora(packed, kohya)
    wrong = _bfl("base_model.model.", seed=8, rank=4)
    wrong["base_model.model.single_blocks.1.linear2.lora_A.weight"] = torch.ones(4, 64)  # in_features is 128
    with pytest.raises(ValueError, match=r"single_transformer_blocks\.1\.attn\.to_out: lora_A \(4, 64\)"):
        load_flux2_lora(packed, wrong)
    with pytest.raises(ValueError, match="no transformer LoRA weights"), pytest.warns(UserWarning, match="skipped"):
        load_flux2_lora(packed, {"text_encoder.a.lora_A.weight": torch.ones(1, 1)})
    assert not lora_layers(packed)
    for name in ("a.safetensors", "b.safetensors"):
        safetensors_torch.save_file(_bfl("base_model.model.", seed=9), str(tmp_path / name))
    with pytest.raises(ValueError, match="several LoRA weight files"):
        load_flux2_lora(packed, str(tmp_path))
    assert load_flux2_lora(packed, str(tmp_path), weight_name="b.safetensors").weight_file.endswith("b.safetensors")


def test_adapters_on_the_packed_transformer_are_float32_side_branches(dense, tmp_path):
    packed = _packed(dense).to(torch.bfloat16)
    load_flux2_lora(packed, _bfl("base_model.model.", seed=10), name="b")
    layer = packed.single_transformer_blocks[0].attn.to_out
    assert layer.weight_scale.dtype == torch.bfloat16 and layer.lora_A["b"].weight.dtype == torch.float32
    with pytest.raises(ValueError, match="packed layer without full-precision weights"):
        merge_lora_into_weights(packed, "b")
    save_lora(packed, str(tmp_path), "b")  # bitnet_vision's own PEFT-format copy loads back unchanged
    again = _packed(dense)
    load_lora(again, str(tmp_path), name="b", dtype=torch.float32)
    assert torch.equal(again.single_transformer_blocks[0].attn.to_out.lora_B["b"].weight, layer.lora_B["b"].weight)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")
def test_packed_transformer_with_a_lora_on_an_apple_gpu(dense, inputs, tmp_path):
    """Loading straight onto MPS (device="mps"), then a LoRA on the GPU model: the CPU's output, up to the GPU's
    summation order."""
    _write_mlx(dense, str(tmp_path))
    bfl = _bfl("base_model.model.", seed=11)
    cpu = load_bonsai_transformer(str(tmp_path), dtype=torch.float32, group_size=GROUP)
    gpu = load_bonsai_transformer(str(tmp_path), dtype=torch.float32, group_size=GROUP, device="mps")
    assert all(t.device.type == "mps" for t in gpu.state_dict().values())
    load_flux2_lora(cpu, bfl, name="b")
    load_flux2_lora(gpu, bfl, name="b")
    assert gpu.single_transformer_blocks[0].attn.to_out.lora_A["b"].weight.device.type == "mps"
    got = _run(gpu, {k: v.to("mps") for k, v in inputs.items()}).cpu()
    torch.testing.assert_close(got, _run(cpu, inputs), rtol=1e-3, atol=1e-3)


def test_the_package_imports_without_touching_diffusers():
    code = ("import sys, bitnet_vision.generators, bitnet_vision.weight_only; "
            "assert 'diffusers' not in sys.modules, 'diffusers was imported'")
    subprocess.run([sys.executable, "-c", code], check=True)
