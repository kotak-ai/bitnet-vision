"""scripts/flux2_lora.py (generate, compare, realign) on the tiny pipeline its --tiny option builds.

The tiny models have the real ones' structure and file formats (a FLUX.2 klein-shaped transformer, its ternary copy
in Bonsai's MLX layout, a Qwen3 text encoder read at layers 9, 18 and 27, the FLUX.2 VAE, a LoRA in fal's format),
so every command runs the code it runs on the real models, on a CPU in seconds. Needs the 'flux' extra; no network.
"""

import argparse
import copy
import importlib.util
import json
import os
import sys

import pytest
import torch
from torch import nn

pytest.importorskip("diffusers")
pytest.importorskip("peft")
pytest.importorskip("PIL")
safetensors = pytest.importorskip("safetensors")

from diffusers import Flux2KleinPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from bitnet_vision.generators import load_bonsai_transformer, load_flux2_lora  # noqa: E402
from bitnet_vision.lora import load_lora  # noqa: E402
from bitnet_vision.weight_only import PackedTernaryLinear  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    spec = importlib.util.spec_from_file_location("script_flux2_lora", os.path.join(ROOT, "scripts", "flux2_lora.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the file runs
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture(scope="module")
def tiny(script, tmp_path_factory):
    root = str(tmp_path_factory.mktemp("tiny"))
    sources, lora = script.build_tiny(root)
    return root, sources, lora


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(hidden_states=torch.randn(2, 64, 16, generator=g),
                encoder_hidden_states=torch.randn(2, 32, 24, generator=g), timestep=torch.tensor([0.3, 0.8]),
                img_ids=torch.cartesian_prod(torch.arange(1), torch.arange(8), torch.arange(8), torch.arange(1)),
                txt_ids=torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(32)))


def _run(model, inputs):
    with torch.no_grad():
        return model(**inputs).sample


def _dense_copy(packed):
    """The packed model with every PackedTernaryLinear back as an nn.Linear holding its exact weights."""
    dense = copy.deepcopy(packed)
    for name, layer in list(dense.named_modules()):
        if isinstance(layer, PackedTernaryLinear):
            linear = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
            with torch.no_grad():
                linear.weight.copy_(layer.dequantize(torch.float32))
            parent, _, child = name.rpartition(".")
            setattr(dense.get_submodule(parent), child, linear)
    return dense


# ---------------------------------------------------------------------------------------------------------------
# Pieces


def test_lora_argument_and_size_parsing(script, tmp_path):
    spec = script.parse_lora("fal/flux-2-klein-4b-spritesheet-lora/flux-spritesheet-lora.safetensors:0.7")
    assert (spec.source, spec.weight_name, spec.subfolder, spec.strength) == (
        "fal/flux-2-klein-4b-spritesheet-lora", "flux-spritesheet-lora.safetensors", None, 0.7)
    spec = script.parse_lora("owner/repo/some/dir/x.safetensors")
    assert (spec.source, spec.subfolder, spec.weight_name, spec.strength) == ("owner/repo", "some/dir", "x.safetensors",
                                                                             1.0)
    assert script.parse_lora("owner/repo:1.5").strength == 1.5
    assert script.parse_lora("owner/repo:latest").source == "owner/repo:latest"  # not a number: part of the name
    local = tmp_path / "a" / "b" / "lora.safetensors"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"")
    spec = script.parse_lora(f"{local}:-0.5")
    assert (spec.source, spec.weight_name, spec.strength) == (str(local), None, -0.5)  # existing file: kept whole
    assert script.pick_size("1024", 512, 16) == (1024, 1024)
    assert script.pick_size("768x512", 512, 16) == (768, 512)
    assert script.pick_size(None, 512, 16) == (512, 512)
    with pytest.raises(SystemExit, match="multiples of 16"):
        script.pick_size("1000", 512, 16)


def test_retention_metric(script):
    g = torch.Generator().manual_seed(0)
    a, effect, gap = (torch.randn(1, 64, 16, generator=g) for _ in range(3))
    same = script.retention(a, a + effect, a + gap, a + gap + effect)
    assert same["retention_cos"] == pytest.approx(1.0, abs=1e-12) and same["retention_norm"] == pytest.approx(1.0)
    half = script.retention(a, a + effect, a + gap, a + gap - 0.5 * effect)
    assert half["retention_cos"] == pytest.approx(-1.0, abs=1e-12) and half["retention_norm"] == pytest.approx(0.5)
    assert script.psnr(a, a) == float("inf") and script.psnr(torch.zeros(3), torch.full((3,), 0.1)) == pytest.approx(20)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sampler_and_decoder_equal_the_pipeline(script, tiny, tmp_path, dtype):
    """The script's own loop (needed to reuse prompts, swap transformers and record trajectories) produces exactly
    the image Flux2KleinPipeline produces for the same prompt and generator seed."""
    root, sources, _ = tiny
    pipe = Flux2KleinPipeline.from_pretrained(sources.fp, dtype=dtype)
    reference = pipe(prompt="tiny style, a red fox", height=32, width=48, num_inference_steps=4, guidance_scale=1.0,
                     max_sequence_length=32, generator=torch.Generator().manual_seed(3), output_type="pt").images[0]
    run = script.Run("test", argparse.Namespace(out=str(tmp_path), quiet=True), torch.device("cpu"), dtype)
    transformer = pipe.transformer
    conds = script.encode_prompts(pipe, ["tiny style, a red fox"], torch.device("cpu"), 32, run)
    assert pipe.text_encoder is None  # freed after encoding
    noise = script.make_noise(3, pipe, 48, 32, dtype)
    latents, trajectory = script.sample(transformer, pipe.scheduler, noise, conds["tiny style, a red fox"], 4, dtype,
                                        torch.device("cpu"), record=True)
    image = script.decode(pipe, latents, 48, 32, torch.device("cpu"))
    assert image.shape == (3, 32, 48) and torch.equal(image, reference.float())
    # the recorded trajectory: 4 states, their timesteps, and the model's outputs there
    assert len(trajectory["x"]) == len(trajectory["v"]) == 4 and len(trajectory["sigmas"]) == 5
    assert trajectory["t"][0] == pytest.approx(1000.0) and trajectory["sigmas"][-1] == 0.0
    assert torch.equal(trajectory["x"][0], pipe._pack_latents(noise))


def test_sampler_with_guidance_equals_the_pipeline(script, tiny, tmp_path):
    """Classifier-free guidance, which diffusers applies to klein models that are not step-distilled (such as
    klein-base 4B), with the empty prompt as the negative one."""
    root, sources, _ = tiny
    pipe = Flux2KleinPipeline.from_pretrained(sources.fp, is_distilled=False)
    assert not pipe.config.is_distilled
    reference = pipe(prompt="a boat", height=32, width=32, num_inference_steps=3, guidance_scale=4.0,
                     max_sequence_length=32, generator=torch.Generator().manual_seed(1), output_type="pt").images[0]
    run = script.Run("test", argparse.Namespace(out=str(tmp_path), quiet=True), torch.device("cpu"), torch.float32)
    conds = script.encode_prompts(pipe, ["a boat", ""], torch.device("cpu"), 32, run)
    noise = script.make_noise(1, pipe, 32, 32, torch.float32)
    latents, _ = script.sample(pipe.transformer, pipe.scheduler, noise, conds["a boat"], 3, torch.float32,
                               torch.device("cpu"), negative=conds[""], guidance=4.0)
    assert torch.equal(script.decode(pipe, latents, 32, 32, torch.device("cpu")), reference.float())


# ---------------------------------------------------------------------------------------------------------------
# The three commands


@pytest.mark.slow
def test_generate(script, tmp_path):
    out = tmp_path / "gen"
    result = script.main(["generate", "--tiny", "--out", str(out), "--prompt", "a red fox", "--prompt", "a cat",
                          "--seed", "0", "1", "--quiet"])
    files = sorted(p for p in os.listdir(out) if p.endswith(".png"))
    assert files == ["p00_s0_a-red-fox.png", "p00_s1_a-red-fox.png", "p01_s0_a-cat.png", "p01_s1_a-cat.png"]
    assert Image.open(out / files[0]).size == (32, 32)
    with open(out / "settings.json") as f:
        settings = json.load(f)
    assert settings["base"] == "ternary" and settings["width"] == settings["height"] == 32
    assert settings["loras"][0]["modules"] == 32 and settings["loras"][0]["format"].startswith("PEFT")
    assert [im["file"] for im in settings["images"]] == [im["file"] for im in result["images"]]
    assert settings["sample_steps"] == 4 and settings["guidance"] == 1.0
    for key in ("environment", "seconds_by_phase", "memory", "sources"):
        assert key in settings
    # the full-precision base and a second LoRA (the same file at another strength) also work
    lora = os.path.join(out, "tiny-models", "lora", "tiny-lora.safetensors")
    result = script.main(["generate", "--tiny", "--out", str(tmp_path / "fp"), "--base", "fp", "--lora", lora,
                          "--lora", f"{lora}:-0.5", "--prompt", "a cat", "--quiet"])
    assert [entry["strength"] for entry in result["loras"]] == [1.0, -0.5] and len(result["images"]) == 1


@pytest.mark.slow
def test_compare_retention_is_one_when_the_bases_are_equal(script, tmp_path):
    """Sanity check of the metric: with --tiny-same-base the full-precision model holds the ternary weights, so the
    LoRA must do exactly the same on both (both bases compute bit-identical outputs in float32)."""
    out = tmp_path / "cmp"
    result = script.main(["compare", "--tiny", "--tiny-same-base", "--out", str(out), "--seed", "0", "5", "--quiet"])
    assert len(result["rows"]) == 4  # 2 default prompts (with the LoRA's trigger words) x 2 seeds
    assert all(row["prompt"].startswith("tiny style, ") for row in result["rows"])
    for row in result["rows"]:
        assert row["retention_cos"] == pytest.approx(1.0, abs=1e-12)
        assert row["retention_norm"] == pytest.approx(1.0, abs=1e-12)
        assert row["first_step_cos"] == pytest.approx(1.0, abs=1e-12)  # identical inputs, identical bases
        assert row["first_step_norm"] == pytest.approx(1.0, abs=1e-12)
        assert row["psnr_d_vs_b"] is None and row["psnr_c_vs_a"] is None  # identical images
        assert row["lora_effect"] > 0.01  # and the LoRA does change the image
    assert result["mean"]["retention_cos"] == pytest.approx(1.0, abs=1e-12)
    grid = Image.open(out / "grid.png")
    assert grid.size[0] > 4 * 320 and set(result["files"]) <= set(os.listdir(out))
    with open(out / "metrics.json") as f:
        assert json.load(f)["mean"]["retention_norm"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.slow
def test_realign_lowers_the_loss_and_saves_a_lora_both_loaders_read(script, tmp_path):
    out = tmp_path / "realign"
    result = script.main(["realign", "--tiny", "--out", str(out), "--steps", "40", "--eval-every", "20", "--lr", "1e-3",
                          "--batch-size", "2", "--trajectories", "1", "--quiet"])
    before, after = result["before"], result["after"]
    assert result["trigger"] == "tiny style" and len(result["train_prompts"]) == 16 and len(result["eval_rows"]) == 4
    assert result["training_points"] == 16 * 4 and [h["step"] for h in result["history"]] == [0, 20, 40]
    assert after["eval_loss"] < 0.9 * before["eval_loss"]  # held-out prompts
    assert after["train_probe_loss"] < 0.9 * before["train_probe_loss"]
    assert after["psnr_vs_b"] > before["psnr_vs_b"]  # closer to klein + LoRA
    assert 0 < before["retention_cos"] < 0.99  # the tiny ternary base does weaken the LoRA

    # diffusers' file: 'transformer.' keys and the rank and alpha in its metadata
    path = out / "pytorch_lora_weights.safetensors"
    with safetensors.safe_open(str(path), framework="pt") as f:
        keys, metadata = list(f.keys()), json.loads(f.metadata()["lora_adapter_metadata"])
    assert len(keys) == 2 * 32 and all(k.startswith("transformer.") for k in keys)
    assert metadata["transformer.r"] == 4 and metadata["transformer.lora_alpha"] == 4.0

    # It loads in diffusers' own LoRA loader on a dense copy of the ternary transformer, and in load_flux2_lora
    # on the packed one, with the same result; the repo's PEFT copy gives the same adapter.
    tiny = out / "tiny-models"
    packed = load_bonsai_transformer(str(tiny / "bonsai"), dtype=torch.float32)
    dense = _dense_copy(packed)
    pipe = Flux2KleinPipeline.from_pretrained(str(tiny / "fp"), transformer=dense)
    pipe.load_lora_weights(str(out))
    inputs = _inputs()
    reference = _run(pipe.transformer, inputs)
    summary = load_flux2_lora(packed, str(out), name="realigned")
    assert summary.format == "diffusers (transformer.)" and len(summary.modules) == 32
    torch.testing.assert_close(_run(packed, inputs), reference, rtol=1e-5, atol=1e-6)
    again = load_bonsai_transformer(str(tiny / "bonsai"), dtype=torch.float32)
    load_lora(again, str(out / "peft"), name="realigned", dtype=torch.float32)
    assert torch.equal(_run(again, inputs), _run(packed, inputs))
    original = load_bonsai_transformer(str(tiny / "bonsai"), dtype=torch.float32)
    load_flux2_lora(original, str(tiny / "lora" / "tiny-lora.safetensors"), name="original")
    assert not torch.allclose(_run(original, inputs), reference)  # training changed the adapter

    # compare --realigned reproduces realign's last evaluation exactly (same prompt, seed and noise)
    row = after["rows"][0]
    cmp = script.main(["compare", "--tiny", "--out", str(tmp_path / "cmp"), "--prompt", row["prompt"], "--seed",
                       str(row["seed"]), "--realigned", str(out), "--quiet"])
    e = cmp["rows"][0]
    assert e["retention_cos_e"] == pytest.approx(row["retention_cos"], abs=1e-9)
    assert e["retention_norm_e"] == pytest.approx(row["retention_norm"], abs=1e-9)
    assert e["psnr_e_vs_b"] == pytest.approx(row["psnr_vs_b"], abs=1e-6)
    assert e["first_step_cos_e"] == pytest.approx(row["first_step_cos"], abs=1e-9)
    assert e["first_step_norm_e"] == pytest.approx(row["first_step_norm"], abs=1e-9)
    assert e["first_step_cos"] == pytest.approx(before["rows"][0]["first_step_cos"], abs=1e-9)
    assert -1 <= before["first_step_cos"] <= 1 and before["first_step_norm"] > 0
    assert e["retention_cos"] == pytest.approx(before["rows"][0]["retention_cos"], abs=1e-9)
    assert Image.open(tmp_path / "cmp" / "grid.png").size[0] > 5 * 320
    for name in ("realign.json", "realign_info.json", "eval_grid.png", "peft/adapter_config.json"):
        assert (out / name).exists(), name


@pytest.mark.slow
def test_realign_effect_matching_raises_retention(script, tmp_path):
    """--match effect trains the LoRA's effect (with minus without the LoRA) instead of the output, which is what
    the retention metric measures."""
    result = script.main(["realign", "--tiny", "--out", str(tmp_path / "r"), "--steps", "40", "--eval-every", "40",
                          "--lr", "1e-3", "--batch-size", "2", "--trajectories", "1", "--match", "effect", "--quiet"])
    before, after = result["before"], result["after"]
    assert result["match"] == "effect" and result["training_points"] == 64
    assert after["eval_loss"] < 0.9 * before["eval_loss"]
    assert after["retention_cos"] > before["retention_cos"] + 0.02
    assert after["retention_norm"] > before["retention_norm"]


@pytest.mark.slow
def test_realign_with_continuous_timesteps_keeps_the_teacher(script, tmp_path):
    """--timesteps continuous trains at random points between the sampler's steps (on the straight Euler segments
    of the teacher's trajectories), running the teacher on each of them."""
    result = script.main(["realign", "--tiny", "--out", str(tmp_path / "r"), "--steps", "30", "--eval-every", "30",
                          "--lr", "1e-3", "--batch-size", "2", "--trajectories", "1", "--timesteps", "continuous",
                          "--prompt", "tiny style, a red fox", "--prompt", "tiny style, a house", "--eval-prompt",
                          "tiny style, a boat", "--quiet"])
    assert result["timesteps"] == "continuous" and result["training_points"] == 8 and len(result["eval_rows"]) == 1
    assert result["after"]["eval_loss"] < result["before"]["eval_loss"]


def test_realign_refuses_several_loras(script, tmp_path):
    with pytest.raises(SystemExit, match="exactly one --lora"):
        script.main(["realign", "--tiny", "--out", str(tmp_path), "--lora", "a", "--lora", "b", "--quiet"])


def test_default_size_follows_gpu_memory(monkeypatch):
    # a 16 GB Mac (11.8 GB GPU working set) gets 512 x 512 by default, a 96 GB one 1024 x 1024
    m = _load_script()
    mps = torch.device("mps")
    monkeypatch.setattr(m, "device_memory_gb", lambda device: 11.8)
    assert m.default_size(mps, tiny=False) == 512
    monkeypatch.setattr(m, "device_memory_gb", lambda device: 72.0)
    assert m.default_size(mps, tiny=False) == 1024
    monkeypatch.setattr(m, "device_memory_gb", lambda device: None)
    assert m.default_size(mps, tiny=False) == 1024
    assert m.default_size(torch.device("cpu"), tiny=False) == 512 and m.default_size(mps, tiny=True) == 32
