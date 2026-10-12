#!/usr/bin/env python
"""Existing FLUX.2 [klein] 4B LoRAs on Bonsai Image 4B, the ternary FLUX.2 klein, with its weights kept 2-bit.

Three commands:

  generate  images from Bonsai (100 ternary layers packed at 2 bits per weight) with any number of LoRAs
  compare   how much of a LoRA's effect survives on the ternary base, measured against full-precision klein 4B
  realign   re-train a LoRA's A and B so that Bonsai + LoRA matches klein 4B + LoRA (data-free distillation)

On a Mac with Apple silicon (device mps, bfloat16), after `pip install -e ".[flux]"`:

  LORA=Sawata97/flux2_4b_koni_animestyle   # a text-to-image style LoRA for klein 4B, trigger words anime_style
  python scripts/flux2_lora.py generate --lora $LORA --prompt "anime_style, a red fox in a snowy forest" --seed 0 1
  python scripts/flux2_lora.py compare --lora $LORA --prompt "anime_style, a red fox in a snowy forest" --seed 0 1
  python scripts/flux2_lora.py realign --lora $LORA --out realigned-anime
  python scripts/flux2_lora.py compare --lora $LORA --realigned realigned-anime --prompt "anime_style, ..." --seed 0

Every command takes --tiny: it builds a tiny random pipeline on the spot (random klein-shaped transformer, its
ternary copy in Bonsai's MLX file layout, Qwen3 text encoder, byte-level tokenizer, VAE, a LoRA in fal's file
format) and runs the same code on it, in seconds on a CPU and without downloads. The tests use it.

Models. The ternary transformer is read from prism-ml/bonsai-image-ternary-4B-mlx-2bit (1.43 GB download) by
bitnet_vision.generators.load_bonsai_transformer and stays packed. The text encoder (Qwen3-4B), tokenizer, VAE and
scheduler come from prism-ml/bonsai-image-ternary-4B-unpacked, without downloading its 7.75 GB transformer. Those
files are byte-identical to black-forest-labs/FLUX.2-klein-4B's (same SHA-256 on the Hub, checked 2026-10-12), so
one text encoder serves both bases and every prompt is encoded once. The full-precision transformer is
FLUX.2-klein-4B's (7.75 GB, Apache-2.0). Sampling follows the model cards: 4 steps, guidance 1.0 (both models are
step-distilled; diffusers ignores guidance for them). The sampler repeats Flux2KleinPipeline's loop exactly
(the tests check this bit for bit), so images match `pipe(...)` for the same noise.

LoRAs: --lora SOURCE[:STRENGTH], repeatable. SOURCE is a .safetensors file, a folder, a Hub repo id, or
"owner/repo/file.safetensors" for repos that hold several files (fal's spritesheet repo holds two different LoRAs).
Every FLUX.2 format diffusers reads works (PEFT, diffusers, ai-toolkit, BFL names, kohya; see load_flux2_lora).
Each LoRA is a float32 side branch, computed the same way on both bases; text-encoder parts are skipped. These
commands make images from text only: LoRAs trained for image editing (fal's spritesheet LoRA, for one) need an
input image and say little here.

compare. For the same prompts, seeds and initial noise:
  (a) klein 4B   (b) klein 4B + LoRA   (c) Bonsai   (d) Bonsai + LoRA   (e) Bonsai + re-aligned LoRA (--realigned)
With D = final latents with the LoRA minus final latents without it (the transformer's latents after the last
step, in float32):
  retention_cos    cos(D_bonsai, D_klein): 1 when the LoRA moves the image the same way on both bases, 0 when
                   the two changes are unrelated
  retention_norm   |D_bonsai| / |D_klein|: how far it moves it (1 = as far)
  first_step_cos   the same two numbers for D = the transformer's output (velocity) at the first step with the
  first_step_norm  LoRA minus without. Both bases see identical inputs there (the same noise, prompt and timestep),
                   so this isolates the LoRA's effect; the final latents also carry the drift between the two
                   bases' different trajectories, which lowers retention_cos even when the images look alike
  psnr_d_vs_b      PSNR of image (d) against image (b) in dB (pixels in [0, 1]); psnr_c_vs_a is the same for the
                   bases alone, the gap the LoRA cannot be blamed for. null means identical images.
The (e) columns repeat these for the re-aligned LoRA. Output: grid.png (labeled), one PNG per image, metrics.json.

realign. Teacher: klein 4B + the LoRA, frozen. Student: packed Bonsai + the same LoRA, initialized from it; only its
A and B matrices train (a strength other than 1 is folded into B first, so the result is used at strength 1).
Training inputs come from the teacher, as in EfficientDM (arXiv 2310.03270): for each prompt the teacher runs its
4-step sampler from random noise, and the student trains on the latents of those trajectories at their timesteps.
The objective (--match):
  output  (default) mean squared error between the student's and the teacher's transformer outputs (velocities)
          at identical inputs, as in EfficientDM. The LoRA then also learns to pull Bonsai toward klein 4B: images
          get closer to (b) while retention_cos can drop, so judge it by psnr_e_vs_b and the held-out loss. Its
          strength setting scales that correction too.
  effect  the LoRA's effect on the student, student(x) - student_without_LoRA(x), against its effect on the
          teacher. The LoRA stays a pure add-on (strength 0 is plain Bonsai) and retention is what it raises.
          Costs one more teacher pass per recorded step and one more student pass per training step.
--timesteps trajectory (default) trains at the sampler's own 4 timesteps, whose teacher outputs are recorded while
it samples, so the teacher is freed before training. --timesteps continuous trains at random points between those
steps (the Euler path is a straight line there) and runs the teacher on each, so both transformers stay in memory;
the 4-step sampler never evaluates those points, so keep the default unless you sample with other step counts.
The schedule depends on the image size: the 4 steps sit at noise levels 1.0, 0.958, 0.884 and 0.718 at 512 x 512
and 1.0, 0.967, 0.908 and 0.767 at 1024 x 1024, so re-align at the size you generate at when time allows.
AdamW, lr 1e-4 (the literature suggests 1e-4 to 5e-4), 300 steps of batch 1, gradient clipping at 1, gradient
checkpointing on the student, the model's own dtype (no autocast: the student trains with exactly the arithmetic it
samples with; the LoRA branch computes in float32), 512 x 512 by default (--size). Prompts: --prompts FILE, otherwise
16 built-in prompts with the LoRA's trigger words in front (--trigger, or the model card's instance_prompt); 4 more
are held out. Every --eval-every steps it logs the held-out loss and the compare metrics on the held-out prompts,
and saves the LoRA twice: pytorch_lora_weights.safetensors in diffusers' format ('transformer.' keys, rank and alpha
in lora_adapter_metadata), which diffusers' load_lora_weights and load_flux2_lora both read, and peft/ in this
repo's PEFT format (bitnet_vision.lora.load_lora). realign.json holds the log; eval_grid.png shows the held-out
prompts before and after.

Memory plan for the M2 Max with 96 GB, in bfloat16 (weight sizes measured on the real files; the rest estimated,
as nothing has run on a Mac yet):
  * text encoder 8.0 GB, loaded once to encode every prompt (7.9 MB each), then freed; VAE 0.17 GB, kept;
  * one transformer at a time: klein 4B 7.75 GB or packed Bonsai 1.37 GB (compare runs every klein arm, frees
    it, then runs every Bonsai arm); LoRAs in float32 (rank 16 on 88 modules: about 0.1 GB);
  * activations: a few GB at 1024 x 1024 (4608 tokens per pass), much less at 512 x 512 (1536);
  * realign: recorded trajectories take 0.5 MB per step at 512 x 512 (64 MB with the defaults); the student
    trains with gradient checkpointing, so it holds one block's activations and one rebuilt dense layer at a
    time, plus the LoRA's gradients and AdamW's two float32 copies. --timesteps continuous adds the teacher.
  Peak, estimated: 10 to 16 GB for every command (the text encoder phase is the largest), so all of it fits
  easily; time is the limit on this Mac.

Time on an M2 Max, estimated from arithmetic (2 x 3.7 G weights x tokens per pass, about 7 TFLOPS in bfloat16 on
the GPU), not measured: one transformer pass takes about 1.7 s at 512 x 512 and 6 s at 1024 x 1024, so an image
(4 passes plus the VAE) takes about 8 s and 25 to 30 s. Bonsai is not faster than klein 4B there: the GPU has no
faster low-bit arithmetic, and the packed layers rebuild each dense weight on every call. On a 4-thread CPU
(measured, bfloat16, 512 x 512) a packed Bonsai step took 10.6 to 12.7 s against 6.7 to 8.9 s for dense klein 4B;
--cache-dense removes that cost and the memory saving with it. compare with 4 prompts at 1024 x 1024 should take
about 10 minutes and realign with the defaults about 45 minutes (an hour with --match effect), both estimates.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import gc
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import textwrap
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bitnet_vision.generators import (  # noqa: E402
    BONSAI_MLX,
    BONSAI_UNPACKED,
    FLUX2_TERNARY_TARGETS,
    load_bonsai_transformer,
    load_flux2_lora,
    round_linears_to_ternary,
    ternarize_transformer,
)
from bitnet_vision.lora import disable_lora, enable_lora, lora_layers, mark_only_lora_trainable, save_lora  # noqa: E402

FP_MODEL = "black-forest-labs/FLUX.2-klein-4B"
STEPS = 4  # model cards of klein 4B and Bonsai
GUIDANCE = 1.0
ARMS = {"a": "klein 4B", "b": "klein 4B + LoRA", "c": "Bonsai", "d": "Bonsai + LoRA", "e": "Bonsai + re-aligned LoRA"}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
TRAIN_PROMPTS = (
    "a red fox sitting in a snowy forest at dawn",
    "a bowl of ramen on a wooden table, steam rising",
    "a lighthouse on a rocky coast during a storm",
    "a portrait of an old fisherman with a knitted cap",
    "a futuristic city skyline at night with neon lights",
    "a vintage bicycle leaning against a brick wall",
    "a cup of coffee and an open book by a rainy window",
    "a mountain lake reflecting pine trees and clouds",
    "a small robot watering plants in a greenhouse",
    "a medieval castle on a hill under a starry sky",
    "a golden retriever puppy playing with a ball in the grass",
    "a sailing ship on a calm sea at sunset",
    "a desert road with a lone car and distant mesas",
    "a stack of colorful macarons on a white plate",
    "an astronaut floating above the earth",
    "a cozy cabin interior with a fireplace and a sleeping cat",
)
EVAL_PROMPTS = (
    "a hot air balloon over green hills",
    "a samurai standing in a bamboo forest",
    "a tram on a city street in the rain",
    "a teapot and two cups on a garden table",
)


# ---------------------------------------------------------------------------------------------------------------
# Small helpers


def _diffusers() -> Any:
    try:
        import diffusers
    except ImportError as err:
        raise SystemExit("this script needs diffusers 0.41 or newer: pip install -e '.[flux]'") from err
    if not hasattr(diffusers, "Flux2KleinPipeline"):
        raise SystemExit(f"diffusers {diffusers.__version__} has no Flux2KleinPipeline; install 0.41 or newer")
    return diffusers


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str, tiny: bool) -> torch.dtype:
    if name == "auto":
        return torch.float32 if tiny else torch.bfloat16  # bfloat16 is the models' own dtype
    return DTYPES[name]


def pick_size(text: str | None, default: int, multiple: int) -> tuple[int, int]:
    """``(width, height)`` from "1024" or "1024x768" (width x height)."""
    if text is None:
        width = height = default
    else:
        m = re.fullmatch(r"(\d+)(?:x(\d+))?", text.strip())
        if not m:
            raise SystemExit(f"--size: expected N or WxH, got {text!r}")
        width = int(m.group(1))
        height = int(m.group(2) or m.group(1))
    if width % multiple or height % multiple or not width or not height:
        raise SystemExit(f"--size: width and height must be positive multiples of {multiple}, got {width}x{height}")
    return width, height


def default_size(device: torch.device, tiny: bool) -> int:
    return 32 if tiny else 512 if device.type == "cpu" else 1024


def release(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def _fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _db(x: float | None) -> str:
    """A PSNR for printing: None stands for identical images (stored as null in the JSON files)."""
    return "inf" if x is None else f"{x:.1f}"


def _peak_rss_gb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e9 if sys.platform == "darwin" else 1e6)  # bytes on macOS, KiB elsewhere


def _device_gb(device: torch.device) -> float | None:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode:
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def environment(device: torch.device) -> dict[str, Any]:
    versions = {}
    for module in ("diffusers", "transformers", "peft", "safetensors"):
        try:
            versions[module] = __import__(module).__version__
        except ImportError:
            versions[module] = None
    return {"python": platform.python_version(), "torch": torch.__version__, **versions,
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "device": str(device),
            "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(), "git_commit": _git_commit()}


class Run:
    """The output folder, log, phase timings and memory readings of one command.

    Memory after each phase: ``peak_rss_gb``, the process's peak resident memory so far; ``device_gb``, on CUDA the
    peak allocated so far, on MPS the Metal driver's current allocation (MPS has no peak counter).
    """

    def __init__(self, command: str, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
        self.command = command
        self.out = args.out or os.path.join("flux2_results", f"{command}-{datetime.datetime.now():%Y%m%d-%H%M%S}")
        os.makedirs(self.out, exist_ok=True)
        self.device, self.dtype = device, dtype
        self.start = time.perf_counter()
        self.timings: dict[str, float] = {}
        self.memory: list[dict[str, Any]] = []
        self.quiet = bool(getattr(args, "quiet", False))
        self.log(f"{command}: device {device}, dtype {str(dtype).replace('torch.', '')}, output in {self.out}")

    def log(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    @contextlib.contextmanager
    def phase(self, name: str, detail: str = "", announce: bool = True) -> Any:
        """Time a phase (``name`` keys the timings, so repeated phases add up) and read memory after it."""
        if announce:
            self.log(f"- {name}{': ' + detail if detail else ''}")
        t = time.perf_counter()
        yield
        self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - t, 3)
        self.memory.append({"after": name, "peak_rss_gb": _finite(_peak_rss_gb()),
                            "device_gb": _finite(_device_gb(self.device))})

    def path(self, *parts: str) -> str:
        return os.path.join(self.out, *parts)

    def summary(self) -> dict[str, Any]:
        return {"seconds_total": round(time.perf_counter() - self.start, 3), "seconds_by_phase": self.timings,
                "memory": self.memory, "peak_rss_gb": _finite(_peak_rss_gb())}

    def write_json(self, name: str, data: dict[str, Any]) -> str:
        path = self.path(name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path


# ---------------------------------------------------------------------------------------------------------------
# Where the models come from


@dataclass
class Sources:
    fp: str  # full-precision transformer: repo id or folder (with transformer/ inside, or the transformer itself)
    ternary: str  # Bonsai: repo id or folder (MLX pack or unpacked layout)
    components: str  # text encoder, tokenizer, VAE and scheduler: repo id or folder


@dataclass
class LoRASpec:
    text: str  # as given on the command line
    source: str
    strength: float = 1.0
    weight_name: str | None = None
    subfolder: str | None = None


def parse_lora(text: str) -> LoRASpec:
    """``SOURCE[:STRENGTH]``; a Hub "owner/repo/[folder/]file.safetensors" names one file of a repo."""
    source, strength = text, 1.0
    head, sep, tail = text.rpartition(":")
    if sep and head:
        try:
            strength = float(tail)
        except ValueError:
            pass
        else:
            source = head
    weight_name = subfolder = None
    if not os.path.exists(source) and source.endswith((".safetensors", ".bin")) and source.count("/") >= 2:
        parts = source.split("/")
        source, weight_name, subfolder = "/".join(parts[:2]), parts[-1], "/".join(parts[2:-1]) or None
    return LoRASpec(text, source, strength, weight_name, subfolder)


def resolve(args: argparse.Namespace, run: Run) -> tuple[Sources, list[LoRASpec]]:
    """The model sources and LoRAs; with --tiny the tiny models are built first (and their LoRA is the default)."""
    loras = [parse_lora(t) for t in args.lora or []]
    if not args.tiny:
        return Sources(args.fp_model, args.ternary_model, args.components), loras
    with run.phase("build tiny models"):
        sources, lora = build_tiny(run.path("tiny-models"), same_base=args.tiny_same_base, seed=args.tiny_seed)
    return sources, loras or [parse_lora(lora)]


def read_prompts(prompts: list[str] | None, path: str | None) -> list[str]:
    out = list(prompts or [])
    if path:
        with open(path, encoding="utf-8") as f:
            out += [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    return out


def with_trigger(prompts: list[str] | tuple[str, ...], trigger: str | None) -> list[str]:
    if not trigger:
        return list(prompts)
    return [p if trigger.lower() in p.lower() else f"{trigger}, {p}" for p in prompts]


def _card_trigger(source: str) -> str | None:
    """``instance_prompt`` (or the most common widget prompt) of the model card: README.md next to a local file or
    folder, or the Hub repo's card. None when there is none or it cannot be read (offline, for example)."""
    try:
        from huggingface_hub import ModelCard
    except ImportError:
        return None
    if os.path.exists(source):
        folder = source if os.path.isdir(source) else os.path.dirname(os.path.abspath(source))
        readme = os.path.join(folder, "README.md")
        if not os.path.isfile(readme):
            return None
        source = readme
    try:
        data = ModelCard.load(source).data.to_dict()
    except Exception:  # noqa: BLE001 (offline, no card, a card that does not parse)
        return None
    instance = data.get("instance_prompt")
    if isinstance(instance, list):
        instance = next((x for x in instance if isinstance(x, str) and x.strip()), None)
    if isinstance(instance, str) and instance.strip():
        return instance.strip()
    texts = [w.get("text") for w in data.get("widget") or [] if isinstance(w, dict)]
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    return Counter(texts).most_common(1)[0][0] if texts else None


def find_trigger(spec: LoRASpec) -> str | None:
    """The LoRA's trigger words: its model card's ``instance_prompt`` (or most common widget prompt), else the
    file's ``modelspec.trigger_phrase`` or ``instance_prompt`` metadata; None when nothing is found."""
    trigger = _card_trigger(spec.source)
    if trigger or not (os.path.isfile(spec.source) and spec.source.endswith(".safetensors")):
        return trigger
    try:
        from safetensors import safe_open

        with safe_open(spec.source, framework="pt") as f:
            meta = f.metadata() or {}
    except Exception:  # noqa: BLE001 (not readable as safetensors: load_flux2_lora reports that)
        return None
    return next((meta[k].strip() for k in ("modelspec.trigger_phrase", "instance_prompt", "trigger_phrase")
                 if meta.get(k, "").strip()), None)


# ---------------------------------------------------------------------------------------------------------------
# Loading


def load_components(sources: Sources, dtype: torch.dtype, run: Run) -> Any:
    """Text encoder, tokenizer, VAE and scheduler as a Flux2KleinPipeline without a transformer.

    Passing ``transformer=None`` makes diffusers skip the transformer folder: nothing of the 7.75 GB transformer
    of the unpacked Bonsai repo is downloaded or read.
    """
    diffusers = _diffusers()
    with run.phase("load text encoder, tokenizer, VAE, scheduler", sources.components):
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(sources.components, transformer=None, dtype=dtype)
    return pipe


@torch.no_grad()
def encode_prompts(pipe: Any, prompts: list[str], device: torch.device, max_sequence_length: int,
                   run: Run) -> dict[str, tuple[Tensor, Tensor]]:
    """``{prompt: (embeddings [1, L, D], text ids [1, L, 4])}`` on the CPU; then the text encoder is freed."""
    out = {}
    with run.phase("encode prompts", f"{len(set(prompts))} prompts, then free the text encoder"):
        pipe.text_encoder.to(device)
        for prompt in dict.fromkeys(prompts):
            embeds, ids = pipe.encode_prompt(prompt=prompt, device=device, max_sequence_length=max_sequence_length)
            out[prompt] = (embeds.cpu(), ids.cpu())
    pipe.text_encoder = None
    release(device)
    return out


def load_fp_transformer(sources: Sources, dtype: torch.dtype, device: torch.device, run: Run) -> nn.Module:
    diffusers = _diffusers()
    subfolder = "transformer"
    if os.path.isdir(sources.fp) and not os.path.isdir(os.path.join(sources.fp, "transformer")):
        subfolder = None
    with run.phase("load full-precision transformer", sources.fp):
        model = diffusers.Flux2Transformer2DModel.from_pretrained(sources.fp, subfolder=subfolder, dtype=dtype)
        model = model.to(device).eval().requires_grad_(False)
    return model


def load_ternary_transformer(sources: Sources, dtype: torch.dtype, device: torch.device, cache_dense: bool,
                             run: Run) -> nn.Module:
    with run.phase("load packed ternary transformer", sources.ternary):
        # read on the CPU, then moved: every tensor is checked and decoded there
        model = load_bonsai_transformer(sources.ternary, dtype=dtype, device="cpu", cache_dense=cache_dense)
        model = model.to(device).eval().requires_grad_(False)
    return model


def attach_loras(model: nn.Module, specs: list[LoRASpec], prefix: str, run: Run) -> list[dict[str, Any]]:
    """Load each LoRA as adapter ``<prefix><i>`` (float32 side branches); returns what was attached."""
    attached = []
    for i, spec in enumerate(specs):
        name = f"{prefix}{i}"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = load_flux2_lora(model, spec.source, name=name, strength=spec.strength,
                                      weight_name=spec.weight_name, subfolder=spec.subfolder)
        for w in caught:
            run.log(f"  warning: {w.message}")
        run.log("  " + str(summary).replace("\n", "\n  "))
        attached.append({"adapter": name, "lora": spec.text, "strength": spec.strength, "format": summary.format,
                         "weight_file": summary.weight_file, "alpha": summary.alpha, "modules": len(summary.modules),
                         "ranks": sorted(set(summary.ranks.values())), "scales": sorted(set(summary.scales.values())),
                         "skipped_tensors": len(summary.skipped)})
    return attached


def fold_strength(model: nn.Module, name: str) -> float:
    """Multiply adapter ``name``'s B by its strength and set the strength to 1 (same output); returns it."""
    strengths = set()
    with torch.no_grad():
        for _, m in lora_layers(model, name):
            s = m.lora_settings[name]
            strengths.add(s.strength)
            m.lora_B[name].weight.mul_(s.strength)
            s.strength = 1.0
    return strengths.pop() if len(strengths) == 1 else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# Sampling, exactly as Flux2KleinPipeline.__call__ does it (text-to-image)


def make_noise(seed: int, pipe: Any, width: int, height: int, dtype: torch.dtype) -> Tensor:
    """The pipeline's initial noise for ``generator=torch.Generator("cpu").manual_seed(seed)``: [1, C, h, w] on the
    CPU, so every device and every arm starts from the same numbers."""
    from diffusers.utils.torch_utils import randn_tensor

    scale = pipe.vae_scale_factor
    h, w = 2 * (height // (scale * 2)), 2 * (width // (scale * 2))
    shape = (1, pipe.vae.config.latent_channels * 4, h // 2, w // 2)
    return randn_tensor(shape, generator=torch.Generator("cpu").manual_seed(seed), device=torch.device("cpu"),
                        dtype=dtype)


def set_schedule(scheduler: Any, steps: int, seq_len: int, device: torch.device) -> Tensor:
    import numpy as np
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

    sigmas = np.linspace(1.0, 1 / steps, steps)
    if getattr(scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=seq_len, num_steps=steps)
    timesteps, _ = retrieve_timesteps(scheduler, steps, device, sigmas=sigmas, mu=mu)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    return timesteps


def velocity(model: nn.Module, x: Tensor, timestep: Tensor, cond: tuple[Tensor, Tensor], latent_ids: Tensor,
             dtype: torch.dtype) -> Tensor:
    """The transformer's output for latents ``x`` [B, N, C] at ``timestep`` [B] (already in x's dtype, / 1000)."""
    embeds, text_ids = cond
    out = model(hidden_states=x.to(dtype), timestep=timestep, guidance=None, encoder_hidden_states=embeds,
                txt_ids=text_ids, img_ids=latent_ids, return_dict=False)[0]
    return out[:, : x.size(1)]


def _cond(cond: tuple[Tensor, Tensor], device: torch.device, batch: int = 1) -> tuple[Tensor, Tensor]:
    embeds, ids = cond
    return embeds.to(device).expand(batch, -1, -1), ids.to(device).expand(batch, -1, -1)


@torch.no_grad()
def sample(model: nn.Module, scheduler: Any, noise: Tensor, cond: tuple[Tensor, Tensor], steps: int,
           dtype: torch.dtype, device: torch.device, *, negative: tuple[Tensor, Tensor] | None = None,
           guidance: float = 1.0, record: bool = False) -> tuple[Tensor, dict[str, Any] | None]:
    """Final latents [1, N, C] (CPU) from ``noise``; with ``record`` also the trajectory: the latents fed to each
    step, its timestep and the model's output there (the teacher's targets for realign) and the sigmas."""
    from diffusers import Flux2KleinPipeline as Pipe

    noise = noise.to(device)
    latents = Pipe._pack_latents(noise)
    latent_ids = Pipe._prepare_latent_ids(noise).to(device)
    timesteps = set_schedule(scheduler, steps, latents.shape[1], device)
    cond = _cond(cond, device)
    negative = _cond(negative, device) if negative is not None else None
    trajectory = {"x": [], "t": [], "v": [], "sigmas": scheduler.sigmas.float().cpu().tolist()} if record else None
    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        v = velocity(model, latents, timestep / 1000, cond, latent_ids, dtype)
        if negative is not None:
            v_neg = velocity(model, latents, timestep / 1000, negative, latent_ids, dtype)
            v = v_neg + guidance * (v - v_neg)
        if record:
            trajectory["x"].append(latents.cpu())
            trajectory["t"].append(float(t))
            trajectory["v"].append(v.cpu())
        latents_dtype = latents.dtype
        latents = scheduler.step(v, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype and torch.backends.mps.is_available():
            latents = latents.to(latents_dtype)  # as the pipeline does (pytorch#99272)
    return latents.cpu(), trajectory


@torch.no_grad()
def decode(pipe: Any, latents: Tensor, width: int, height: int, device: torch.device) -> Tensor:
    """Image [3, H, W] in [0, 1] (float32, CPU) from final latents, as the pipeline decodes them."""
    from diffusers import Flux2KleinPipeline as Pipe

    scale = pipe.vae_scale_factor
    lh, lw = 2 * (height // (scale * 2)), 2 * (width // (scale * 2))
    vae = pipe.vae
    latents = latents.to(device)
    ids = Pipe._prepare_latent_ids(torch.empty(1, latents.shape[-1], lh // 2, lw // 2)).to(device)
    x = Pipe._unpack_latents_with_ids(latents, ids, lh // 2, lw // 2)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(x.device, x.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(x.device, x.dtype)
    x = Pipe._unpatchify_latents(x * std + mean)
    image = vae.decode(x.to(vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pt")[0].float().cpu()


def to_pil(image: Tensor) -> Any:
    """PIL image from [3, H, W] in [0, 1], rounded as diffusers' numpy_to_pil rounds."""
    from PIL import Image

    return Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))


def psnr(x: Tensor, y: Tensor) -> float:
    mse = (x.double() - y.double()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def retention(without_fp: Tensor, with_fp: Tensor, without_t: Tensor, with_t: Tensor) -> dict[str, float | None]:
    """cos(D_t, D_fp) and |D_t| / |D_fp| for D = latents with the LoRA minus latents without it."""
    d_fp = (with_fp.double() - without_fp.double()).flatten()
    d_t = (with_t.double() - without_t.double()).flatten()
    n_fp, n_t = d_fp.norm().item(), d_t.norm().item()
    cos = float(d_fp @ d_t) / (n_fp * n_t) if n_fp > 0 and n_t > 0 else None
    return {"retention_cos": cos, "retention_norm": n_t / n_fp if n_fp > 0 else None,
            "lora_effect": n_fp / max(without_fp.double().norm().item(), 1e-30)}


def make_grid(rows: list[list[Any]], col_labels: list[str], row_labels: list[str],
              captions: list[list[str]] | None = None, cell: int = 320) -> Any:
    """A white sheet with one labeled column per arm and one labeled row per prompt and seed."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow before 10.1
        font = ImageFont.load_default()
    first = next(img for row in rows for img in row if img is not None)
    cell_h = max(1, round(cell * first.height / first.width))
    pad, head, label_w, cap_h = 8, 26, 220, 20 if captions else 0
    width = label_w + len(col_labels) * (cell + pad) + pad
    height = head + len(rows) * (cell_h + cap_h + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for j, label in enumerate(col_labels):
        draw.text((label_w + pad + j * (cell + pad), 6), label, fill="black", font=font)
    for i, row in enumerate(rows):
        y = head + i * (cell_h + cap_h + pad)
        lines = textwrap.wrap(row_labels[i], 26)[: max(1, cell_h // 18)]
        draw.multiline_text((6, y + 2), "\n".join(lines), fill="black", font=font)
        for j, img in enumerate(row):
            x = label_w + pad + j * (cell + pad)
            if img is not None:
                small = img.width >= cell
                sheet.paste(img.resize((cell, cell_h), Image.LANCZOS if small else Image.NEAREST), (x, y))
            if captions and captions[i][j]:
                draw.text((x, y + cell_h + 2), captions[i][j], fill="black", font=font)
    return sheet


# ---------------------------------------------------------------------------------------------------------------
# Tiny models for --tiny (same code path, seconds on a CPU)

TINY_TRANSFORMER = dict(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=16,
                        num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=32,
                        axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)
TINY_GROUP = 16  # divides every input width of the tiny transformer (16, 24, 32, 96, 128)
# scheduler_config.json of FLUX.2-klein-4B and of both Bonsai repos (the same file)
SCHEDULER_CONFIG = dict(base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=4096,
                        max_shift=1.15, num_train_timesteps=1000, shift=3.0, shift_terminal=None,
                        stochastic_sampling=False, time_shift_type="exponential", use_beta_sigmas=False,
                        use_dynamic_shifting=True, use_exponential_sigmas=False, use_karras_sigmas=False)
CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def _tiny_tokenizer() -> Any:
    """A byte-level BPE tokenizer with Qwen's chat tokens and a chat template: any text, one token per byte, no
    download. Built through the generic fast-tokenizer class, which transformers 4 and 5 both have."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    for s in specials:
        vocab[s] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=specials[0], pad_token=specials[0],
                                   additional_special_tokens=specials[1:], chat_template=CHAT_TEMPLATE)


def _bfl_names(model: nn.Module) -> dict[str, tuple[int, int]]:
    """BFL module names of a FLUX.2 transformer (as fal's LoRAs use them) with (in, out) of each LoRA target."""
    mods = dict(model.named_modules())
    out = {}
    for i, _ in enumerate(model.transformer_blocks):
        p = f"transformer_blocks.{i}."
        for kind, q, proj, ff in (("img", "attn.to_q", "attn.to_out.0", "ff"),
                                  ("txt", "attn.add_q_proj", "attn.to_add_out", "ff_context")):
            q_layer = mods[p + q]
            out[f"double_blocks.{i}.{kind}_attn.qkv"] = (q_layer.in_features, 3 * q_layer.out_features)
            for bfl, diff in ((f"{kind}_attn.proj", proj), (f"{kind}_mlp.0", f"{ff}.linear_in"),
                              (f"{kind}_mlp.2", f"{ff}.linear_out")):
                out[f"double_blocks.{i}.{bfl}"] = (mods[p + diff].in_features, mods[p + diff].out_features)
    for i, _ in enumerate(model.single_transformer_blocks):
        for bfl, diff in (("linear1", "attn.to_qkv_mlp_proj"), ("linear2", "attn.to_out")):
            layer = mods[f"single_transformer_blocks.{i}.{diff}"]
            out[f"single_blocks.{i}.{bfl}"] = (layer.in_features, layer.out_features)
    for bfl, diff in (("img_in", "x_embedder"), ("txt_in", "context_embedder"), ("final_layer.linear", "proj_out"),
                      ("single_stream_modulation.lin", "single_stream_modulation.linear")):
        out[bfl] = (mods[diff].in_features, mods[diff].out_features)  # 16-bit layers, as in real LoRAs
    return out


def _write_mlx_pack(model: nn.Module, folder: str, group: int) -> None:
    """Bonsai's MLX layout: uint32 codes, bfloat16 scales, biases == -scales, quantization_config.json."""
    from safetensors.torch import save_file

    from bitnet_vision.apple.mlx_ternary import encode_mlx_ternary

    os.makedirs(folder, exist_ok=True)
    packed = copy.deepcopy(model)
    names = ternarize_transformer(packed, group_size=group, targets=FLUX2_TERNARY_TARGETS)
    tensors = {}
    for name in names:
        layer = packed.get_submodule(name)
        w_q, scales, biases = encode_mlx_ternary(layer.codes(), layer.weight_scale.bfloat16())
        tensors.update({f"{name}.weight": w_q, f"{name}.scales": scales, f"{name}.biases": biases})
    for key, value in model.state_dict().items():
        if key.rsplit(".", 1)[0] not in names:
            tensors[key] = value.bfloat16().contiguous()
    save_file(tensors, os.path.join(folder, "diffusion_pytorch_model.safetensors"))
    model.save_config(folder)
    with open(os.path.join(folder, "quantization_config.json"), "w") as f:
        json.dump({"format": "mlx-packed-affine", "solver": "ternary", "bits": 2, "group_size": group,
                   "scale_dtype": "bfloat16", "quantized_modules": sorted(names)}, f, indent=2)


def build_tiny(root: str, same_base: bool = False, seed: int = 0) -> tuple[Sources, str]:
    """Write tiny stand-ins of the real models under ``root``; returns their sources and the tiny LoRA's path.

    * ``fp``: a whole Flux2KleinPipeline folder (as FLUX.2-klein-4B): random transformer with bfloat16-exact
      weights, a 28-layer Qwen3 text encoder of width 8 (layers 9, 18 and 27 feed the transformer, as in the real
      one), a byte-level tokenizer, a 2-level VAE with random batch-norm statistics, the real scheduler config.
    * ``bonsai/transformer-packed-mflux``: the transformer's 28 block linears rounded to ternary (absmean, one
      bfloat16 scale per 16 inputs) in Bonsai's MLX file layout. With ``same_base`` the full-precision transformer
      holds those same ternary weights, so both bases compute the same function.
    * ``lora/tiny-lora.safetensors``: a rank-4 LoRA in fal's format (BFL names, 'base_model.model.' prefix, fused
      qkv, no alpha) on every block linear and four 16-bit layers, plus a README.md with an instance_prompt.
    """
    diffusers = _diffusers()
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fp = diffusers.Flux2Transformer2DModel(**TINY_TRANSFORMER).eval()
        with torch.no_grad():
            for name, p in fp.named_parameters():
                # Larger modulation and block weights than the default init, so that the blocks (whose linears
                # become ternary) shape the output as in the real model, instead of the embedders around them.
                gain = 4.0 if "modulation" in name else 2.0 if "blocks" in name and p.dim() == 2 else 1.0
                p.copy_((p * gain).bfloat16().float())  # exact in the bfloat16 files
        ternary = copy.deepcopy(fp)
        round_linears_to_ternary(ternary, group_size=TINY_GROUP, scale_dtype=torch.bfloat16)
        if same_base:
            fp = copy.deepcopy(ternary)
        text_config = Qwen3Config(vocab_size=259, hidden_size=8, intermediate_size=16, num_hidden_layers=28,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=128,
                                  tie_word_embeddings=True)
        text_encoder = Qwen3ForCausalLM(text_config).eval()
        vae = diffusers.AutoencoderKLFlux2(block_out_channels=(8, 8), down_block_types=("DownEncoderBlock2D",) * 2,
                                           up_block_types=("UpDecoderBlock2D",) * 2, layers_per_block=1,
                                           latent_channels=4, norm_num_groups=4, sample_size=32).eval()
        with torch.no_grad():
            vae.bn.running_mean.copy_(0.1 * torch.randn_like(vae.bn.running_mean))
            vae.bn.running_var.copy_(0.5 + torch.rand_like(vae.bn.running_var))
        g = torch.Generator().manual_seed(seed + 1)
        lora = {}
        for name, (n_in, n_out) in _bfl_names(fp).items():
            small = 0.2 if "blocks" not in name else 1.0  # 16-bit layers: a smaller share of the effect
            lora[f"base_model.model.{name}.lora_A.weight"] = torch.randn(4, n_in, generator=g) * 0.3
            lora[f"base_model.model.{name}.lora_B.weight"] = torch.randn(n_out, 4, generator=g) * 0.05 * small
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG)
    pipe = diffusers.Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                                        tokenizer=_tiny_tokenizer(), transformer=fp, is_distilled=True)
    pipe.save_pretrained(os.path.join(root, "fp"))
    _write_mlx_pack(ternary, os.path.join(root, "bonsai", "transformer-packed-mflux"), TINY_GROUP)
    os.makedirs(os.path.join(root, "lora"), exist_ok=True)
    lora_path = os.path.join(root, "lora", "tiny-lora.safetensors")
    save_file(lora, lora_path)
    with open(os.path.join(root, "lora", "README.md"), "w") as f:
        f.write("---\nbase_model: tiny\ninstance_prompt: tiny style\n---\n\nA random LoRA for tests.\n")
    fp_dir = os.path.join(root, "fp")
    return Sources(fp=fp_dir, ternary=os.path.join(root, "bonsai"), components=fp_dir), lora_path


# ---------------------------------------------------------------------------------------------------------------
# Shared command setup


@dataclass
class Setup:
    run: Run
    sources: Sources
    loras: list[LoRASpec]
    pipe: Any
    width: int
    height: int
    prompts: list[str]
    conds: dict[str, tuple[Tensor, Tensor]]
    negative: tuple[Tensor, Tensor] | None
    guidance: float


def setup(command: str, args: argparse.Namespace, prompts_for: Any, default_size_px: int | None = None) -> Setup:
    """Device, output folder, models, prompts encoded once (text encoder freed), VAE on the device."""
    _diffusers()  # before anything is created
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, args.tiny)
    size = default_size_px if default_size_px is not None else default_size(device, args.tiny)
    pick_size(args.size, size, 4 if args.tiny else 16)  # fail before loading anything (checked again below)
    run = Run(command, args, device, dtype)
    sources, loras = resolve(args, run)
    prompts = prompts_for(loras)
    if not prompts:
        raise SystemExit("no prompts: give --prompt or --prompts")
    pipe = load_components(sources, dtype, run)
    width, height = pick_size(args.size, size, pipe.vae_scale_factor * 2)
    cfg = args.guidance > 1 and not pipe.config.is_distilled
    if args.guidance > 1 and pipe.config.is_distilled:
        run.log(f"  note: guidance {args.guidance} is ignored, the model is step-distilled (as in diffusers)")
    conds = encode_prompts(pipe, prompts + ([""] if cfg else []), device, args.max_sequence_length, run)
    pipe.vae.to(device)
    return Setup(run, sources, loras, pipe, width, height, list(dict.fromkeys(prompts)), conds,
                 conds[""] if cfg else None, args.guidance)


def _sample(s: Setup, model: nn.Module, prompt: str, noise: Tensor, steps: int, record: bool = False):
    return sample(model, s.pipe.scheduler, noise, s.conds[prompt], steps, s.run.dtype, s.run.device,
                  negative=s.negative, guidance=s.guidance, record=record)


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:n] or "prompt"


# ---------------------------------------------------------------------------------------------------------------
# generate


def cmd_generate(args: argparse.Namespace) -> dict[str, Any]:
    def prompts_for(loras: list[LoRASpec]) -> list[str]:
        prompts = read_prompts(args.prompt, args.prompts)
        if not prompts:
            trigger = next((t for t in (find_trigger(spec) for spec in loras) if t), None)
            prompts = with_trigger(EVAL_PROMPTS[:1], trigger)
        return prompts

    s = setup("generate", args, prompts_for)
    run = s.run
    if args.base == "fp":
        model = load_fp_transformer(s.sources, run.dtype, run.device, run)
    else:
        model = load_ternary_transformer(s.sources, run.dtype, run.device, args.cache_dense, run)
    loras = attach_loras(model, s.loras, "lora", run) if s.loras else []
    images = []
    for i, prompt in enumerate(s.prompts):
        for seed in args.seed:
            noise = make_noise(seed, s.pipe, s.width, s.height, run.dtype)
            t = time.perf_counter()
            with run.phase("sample", announce=False):
                latents, _ = _sample(s, model, prompt, noise, args.sample_steps)
            with run.phase("decode", announce=False):
                image = decode(s.pipe, latents, s.width, s.height, run.device)
            path = run.path(f"p{i:02d}_s{seed}_{_slug(prompt)}.png")
            to_pil(image).save(path)
            images.append({"file": os.path.basename(path), "prompt": prompt, "seed": seed})
            run.log(f"  {path} ({time.perf_counter() - t:.1f} s)")
    result = {"command": "generate", "argv": args.argv, "base": args.base, "sources": vars(s.sources),
              "loras": loras, "width": s.width, "height": s.height, "sample_steps": args.sample_steps,
              "guidance": args.guidance, "max_sequence_length": args.max_sequence_length,
              "dtype": str(run.dtype).replace("torch.", ""), "cache_dense": args.cache_dense, "images": images,
              "environment": environment(run.device), **run.summary()}
    run.write_json("settings.json", result)
    return result


# ---------------------------------------------------------------------------------------------------------------
# compare


def first_step(v: dict[str, Tensor], arm: str = "d") -> dict[str, float | None]:
    """cos and norm ratio of the LoRA's change to the first step's output (velocity) on both bases.

    At the first step both transformers get the same input (the initial noise, the same prompt and timestep), so
    this compares the LoRA's effect on identical inputs; the final-latent retention also carries the drift of the
    two bases' different trajectories."""
    m = retention(v["a"], v["b"], v["c"], v[arm])
    return {"first_step_cos": m["retention_cos"], "first_step_norm": m["retention_norm"]}


def compare_metrics(lat: dict[str, Tensor], img: dict[str, Tensor], vel: dict[str, Tensor]) -> dict[str, Any]:
    """Metrics of one prompt and seed from final latents, images and first-step outputs keyed by arm letter."""
    m = retention(lat["a"], lat["b"], lat["c"], lat["d"])
    m.update(first_step(vel))
    m.update(psnr_d_vs_b=_finite(psnr(img["d"], img["b"])), psnr_c_vs_a=_finite(psnr(img["c"], img["a"])))
    if "e" in lat:
        e = retention(lat["a"], lat["b"], lat["c"], lat["e"])
        v = first_step(vel, "e")
        m.update(retention_cos_e=e["retention_cos"], retention_norm_e=e["retention_norm"],
                 first_step_cos_e=v["first_step_cos"], first_step_norm_e=v["first_step_norm"],
                 psnr_e_vs_b=_finite(psnr(img["e"], img["b"])))
    return m


def cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    def prompts_for(loras: list[LoRASpec]) -> list[str]:
        prompts = read_prompts(args.prompt, args.prompts)
        if not prompts:
            trigger = next((t for t in (find_trigger(spec) for spec in loras) if t), None)
            prompts = with_trigger(EVAL_PROMPTS[:2], trigger)
        return prompts

    if not args.lora and not args.tiny:
        raise SystemExit("compare needs at least one --lora")
    realigned = parse_lora(args.realigned) if args.realigned else None
    if realigned and args.lora and len(args.lora) != 1:
        raise SystemExit("--realigned replaces one LoRA: give exactly one --lora with it")
    s = setup("compare", args, prompts_for)
    run = s.run
    rows = [(i, p, seed) for i, p in enumerate(s.prompts) for seed in args.seed]
    noises = {seed: make_noise(seed, s.pipe, s.width, s.height, run.dtype) for seed in args.seed}
    arms = ["a", "b", "c", "d"] + (["e"] if realigned else [])
    latents: dict[str, dict[int, Tensor]] = {arm: {} for arm in arms}
    first: dict[str, dict[int, Tensor]] = {arm: {} for arm in arms}  # the transformer's output at the first step

    def run_arm(arm: str, model: nn.Module) -> None:
        with run.phase(f"sample ({arm})", f"{ARMS[arm]}, {len(rows)} images"):
            for r, (_, prompt, seed) in enumerate(rows):
                latents[arm][r], traj = _sample(s, model, prompt, noises[seed], args.sample_steps, record=True)
                first[arm][r] = traj["v"][0]

    model = load_fp_transformer(s.sources, run.dtype, run.device, run)
    fp_loras = attach_loras(model, s.loras, "lora", run)
    disable_lora(model)
    run_arm("a", model)
    enable_lora(model)
    run_arm("b", model)
    del model
    release(run.device)
    model = load_ternary_transformer(s.sources, run.dtype, run.device, args.cache_dense, run)
    ternary_loras = attach_loras(model, s.loras, "lora", run)
    disable_lora(model)
    run_arm("c", model)
    enable_lora(model)
    run_arm("d", model)
    realigned_info = None
    if realigned:
        disable_lora(model)
        realigned_info = attach_loras(model, [realigned], "realigned", run)
        run_arm("e", model)
    del model
    release(run.device)

    images: dict[str, dict[int, Tensor]] = {arm: {} for arm in arms}
    files = []
    with run.phase("decode", f"{len(arms) * len(rows)} images"):
        for arm in arms:
            for r, (i, _, seed) in enumerate(rows):
                images[arm][r] = decode(s.pipe, latents[arm][r], s.width, s.height, run.device)
                name = f"{arm}_p{i:02d}_s{seed}.png"
                to_pil(images[arm][r]).save(run.path(name))
                files.append(name)
    per_row = []
    for r, (i, prompt, seed) in enumerate(rows):
        m = compare_metrics({a: latents[a][r] for a in arms}, {a: images[a][r] for a in arms},
                            {a: first[a][r] for a in arms})
        per_row.append({"prompt": prompt, "seed": seed, "prompt_index": i, **m})
    keys = [k for k in per_row[0] if k not in ("prompt", "seed", "prompt_index")]
    mean = {k: _mean([row[k] for row in per_row]) for k in keys}

    captions = []
    for row in per_row:
        cap = {a: "" for a in arms}
        cap["c"] = f"PSNR vs (a) {_db(row['psnr_c_vs_a'])} dB"
        cap["d"] = (f"cos {_fmt(row['retention_cos'], 2)} (step 1: {_fmt(row['first_step_cos'], 2)})  "
                    f"vs (b) {_db(row['psnr_d_vs_b'])} dB")
        if "e" in arms:
            cap["e"] = (f"cos {_fmt(row['retention_cos_e'], 2)} (step 1: {_fmt(row['first_step_cos_e'], 2)})  "
                        f"vs (b) {_db(row['psnr_e_vs_b'])} dB")
        captions.append([cap[a] for a in arms])
    grid = make_grid([[to_pil(images[a][r]) for a in arms] for r in range(len(rows))],
                     [f"({a}) {ARMS[a]}" for a in arms], [f"{p} (seed {seed})" for _, p, seed in rows],
                     captions, cell=args.grid_cell)
    grid.save(run.path("grid.png"))

    run.log("\nLoRA effect on the ternary base (D = with LoRA - without; final latents, and the first step's output):")
    run.log(f"  {'prompt':40s} {'seed':>5s} {'cos':>7s} {'|D|':>7s} {'cos 1':>7s} {'|D| 1':>7s} {'PSNR d-b':>9s} "
            f"{'PSNR c-a':>9s}" + (f" {'cos e':>7s} {'cos 1 e':>7s} {'PSNR e-b':>9s}" if realigned else ""))
    for row in per_row + [{"prompt": "mean", "seed": "", **mean}]:
        line = (f"  {row['prompt'][:40]:40s} {row['seed']!s:>5s} {_fmt(row['retention_cos']):>7s} "
                f"{_fmt(row['retention_norm']):>7s} {_fmt(row['first_step_cos']):>7s} "
                f"{_fmt(row['first_step_norm']):>7s} {_db(row['psnr_d_vs_b']):>9s} {_db(row['psnr_c_vs_a']):>9s}")
        if realigned:
            line += (f" {_fmt(row['retention_cos_e']):>7s} {_fmt(row['first_step_cos_e']):>7s}"
                     f" {_db(row['psnr_e_vs_b']):>9s}")
        run.log(line)
    result = {"command": "compare", "argv": args.argv, "arms": {a: ARMS[a] for a in arms}, "sources": vars(s.sources),
              "loras_fp": fp_loras, "loras_ternary": ternary_loras, "realigned": realigned_info,
              "width": s.width, "height": s.height, "sample_steps": args.sample_steps, "guidance": args.guidance,
              "max_sequence_length": args.max_sequence_length, "dtype": str(run.dtype).replace("torch.", ""),
              "metrics_definition": {
                  "retention_cos": "cos(D_bonsai, D_klein), D = final latents with LoRA - without (float32)",
                  "retention_norm": "|D_bonsai| / |D_klein|",
                  "first_step_cos": "cos(D_bonsai, D_klein) for D = the transformer's output at the first step with "
                                    "LoRA - without; both bases see identical inputs there (the initial noise)",
                  "first_step_norm": "|D_bonsai| / |D_klein| at the first step",
                  "lora_effect": "|D_klein| / |latents of (a)|",
                  "psnr_d_vs_b": "PSNR of image (d) against (b), pixels in [0, 1]; null = identical",
                  "psnr_c_vs_a": "PSNR of image (c) against (a): the gap of the bases without LoRA"},
              "rows": per_row, "mean": mean, "files": files + ["grid.png"],
              "environment": environment(run.device), **run.summary()}
    run.write_json("metrics.json", result)
    return result


# ---------------------------------------------------------------------------------------------------------------
# realign


@dataclass
class Point:
    """One training input: latents on a teacher trajectory, its timestep, and the teacher's output there."""

    prompt: str
    x: Tensor  # [1, N, C], latent dtype, CPU
    t: float  # scheduler timestep (sigma * 1000)
    v: Tensor | None  # the teacher's output at (x, t); None: compute it with the teacher now
    v0: Tensor | None = None  # the same without the LoRA (--match effect)


def trajectory_point(traj: dict[str, Any], k: int, u: float | None, num_train_timesteps: int) -> Point:
    """Step ``k`` of a recorded trajectory (``u`` None), or the point a fraction ``u`` of the way to step k + 1.

    The Euler sampler moves in a straight line between steps (x_k + (sigma - sigma_k) v_k), so every such point
    lies on the trajectory the teacher itself followed.
    """
    if u is None:
        return Point(traj["prompt"], traj["x"][k], traj["t"][k], traj["v"][k], (traj.get("v0") or [None] * (k + 1))[k])
    sigma_k, sigma_next = traj["sigmas"][k], traj["sigmas"][k + 1]
    sigma = sigma_k + u * (sigma_next - sigma_k)
    x = traj["x"][k]
    x = (x.float() + (sigma - sigma_k) * traj["v"][k].float()).to(x.dtype)
    return Point(traj["prompt"], x, sigma * num_train_timesteps, None)


def save_realigned(model: nn.Module, name: str, folder: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Write adapter ``name`` as diffusers' pytorch_lora_weights.safetensors and as this repo's PEFT folder."""
    diffusers = _diffusers()
    state, ranks, alphas, rslora = {}, {}, {}, set()
    for module, layer in lora_layers(model, name):
        s = layer.lora_settings[name]
        state[f"{module}.lora_A.weight"] = layer.lora_A[name].weight.detach().float().cpu().contiguous()
        state[f"{module}.lora_B.weight"] = (layer.lora_B[name].weight.detach().float() * s.strength).cpu().contiguous()
        ranks[module], alphas[module] = s.r, s.alpha
        rslora.add(s.use_rslora)
    r = Counter(ranks.values()).most_common(1)[0][0]
    alpha = Counter(alphas.values()).most_common(1)[0][0]
    metadata = {"r": r, "lora_alpha": alpha, "target_modules": sorted(ranks), "lora_dropout": 0.0, "bias": "none",
                "use_rslora": rslora.pop() if len(rslora) == 1 else False,
                "rank_pattern": {m: v for m, v in ranks.items() if v != r},
                "alpha_pattern": {m: v for m, v in alphas.items() if v != alpha}}
    os.makedirs(folder, exist_ok=True)
    diffusers.Flux2KleinPipeline.save_lora_weights(folder, transformer_lora_layers=state,
                                                   transformer_lora_adapter_metadata=metadata)
    save_lora(model, os.path.join(folder, "peft"), name)  # strength is 1 here (folded into B at the start)
    with open(os.path.join(folder, "realign_info.json"), "w") as f:
        json.dump({**extra, "modules": len(ranks), "rank": r, "lora_alpha": alpha}, f, indent=2, default=str)
    return metadata


def cmd_realign(args: argparse.Namespace) -> dict[str, Any]:
    if args.lora is not None and len(args.lora) != 1:
        raise SystemExit("realign takes exactly one --lora")
    if not args.lora and not args.tiny:
        raise SystemExit("realign needs --lora")
    info: dict[str, Any] = {}

    def prompts_for(loras: list[LoRASpec]) -> list[str]:
        trigger = args.trigger if args.trigger is not None else find_trigger(loras[0])
        train = read_prompts(args.prompt, args.prompts) or with_trigger(TRAIN_PROMPTS, trigger)
        held_out = read_prompts(args.eval_prompt, args.eval_prompts) or with_trigger(EVAL_PROMPTS, trigger)
        info.update(trigger=trigger, train_prompts=train, eval_prompts=held_out)
        return train + held_out

    s = setup("realign", args, prompts_for, default_size_px=32 if args.tiny else 512)
    run, dtype, device = s.run, s.run.dtype, s.run.device
    spec, name, steps = s.loras[0], "lora0", args.sample_steps
    n_train = s.pipe.scheduler.config.num_train_timesteps
    eval_rows = [(p, args.seed) for p in info["eval_prompts"]]
    run.log(f"  {len(info['train_prompts'])} training prompts x {args.trajectories} trajectories, {len(eval_rows)} "
            f"held out; trigger words {info['trigger']!r}; {s.width}x{s.height} pixels")

    from diffusers import Flux2KleinPipeline as Pipe

    scale = s.pipe.vae_scale_factor * 2
    latent_ids = Pipe._prepare_latent_ids(torch.empty(1, 1, s.height // scale, s.width // scale)).to(device)
    effect = args.match == "effect"

    def without_lora(model: nn.Module, x: Tensor, timestep: Tensor, cond: tuple[Tensor, Tensor]) -> Tensor:
        disable_lora(model, name)
        try:
            with torch.no_grad():
                return velocity(model, x, timestep, cond, latent_ids, dtype)
        finally:
            enable_lora(model, name)

    def record(prompt: str, noise: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """The teacher's trajectory from ``noise`` (and, for --match effect, its outputs there without LoRA)."""
        final, traj = _sample(s, teacher, prompt, noise, steps, record=True)
        traj["prompt"] = prompt
        if effect:
            cond = _cond(s.conds[prompt], device)
            traj["v0"] = [without_lora(teacher, x.to(device), torch.tensor([t], device=device).to(x.dtype) / 1000,
                                       cond).cpu() for x, t in zip(traj["x"], traj["t"])]
        return final, traj

    # Teacher: klein 4B + LoRA. Record its trajectories (training inputs and targets) and the held-out references.
    student = None
    teacher = load_fp_transformer(s.sources, dtype, device, run)
    teacher_lora = attach_loras(teacher, [spec], "lora", run)
    folded = fold_strength(teacher, name)
    trajectories, eval_trajectories = [], []
    with run.phase("teacher trajectories"):
        for i, prompt in enumerate(info["train_prompts"]):
            for j in range(args.trajectories):
                noise = make_noise(args.seed + 1000 * (j + 1) + i, s.pipe, s.width, s.height, dtype)
                trajectories.append(record(prompt, noise)[1])
    lat: dict[str, dict[int, Tensor]] = {arm: {} for arm in "abcde"}
    first: dict[str, dict[int, Tensor]] = {arm: {} for arm in "abcde"}  # outputs at the first step (same input)
    noises = {r: make_noise(seed, s.pipe, s.width, s.height, dtype) for r, (_, seed) in enumerate(eval_rows)}
    with run.phase("teacher on held-out prompts", "with and without LoRA"):
        for r, (prompt, _) in enumerate(eval_rows):
            lat["b"][r], traj = record(prompt, noises[r])
            first["b"][r] = traj["v"][0]
            eval_trajectories.append(traj)
        disable_lora(teacher)
        for r, (prompt, _) in enumerate(eval_rows):
            lat["a"][r], traj = _sample(s, teacher, prompt, noises[r], steps, record=True)
            first["a"][r] = traj["v"][0]
        enable_lora(teacher)
    if args.timesteps == "trajectory":  # every target is recorded: free the teacher before the student loads
        teacher = None
        release(device)

    # Student: packed Bonsai + the same LoRA; only its A and B train.
    student = load_ternary_transformer(s.sources, dtype, device, args.cache_dense, run)
    student_lora = attach_loras(student, [spec], "lora", run)
    fold_strength(student, name)
    with run.phase("student without LoRA on held-out prompts"):
        disable_lora(student)
        for r, (prompt, _) in enumerate(eval_rows):
            lat["c"][r], traj = _sample(s, student, prompt, noises[r], steps, record=True)
            first["c"][r] = traj["v"][0]
        enable_lora(student)
    images: dict[str, dict[int, Tensor]] = {arm: {} for arm in lat}
    with run.phase("decode references"):
        for arm in "abc":
            for r in lat[arm]:
                images[arm][r] = decode(s.pipe, lat[arm][r], s.width, s.height, device)
    trainable = mark_only_lora_trainable(student, name)
    params = [p for p in student.parameters() if p.requires_grad]
    run.log(f"  training {len(trainable)} tensors, {sum(p.numel() for p in params) / 1e6:.2f} M parameters")
    if not args.no_gradient_checkpointing:
        student.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    points = [(n, k) for n in range(len(trajectories)) for k in range(steps)]
    probe = [trajectory_point(trajectories[n], k, None, n_train) for n, k in rng.sample(points, min(args.probe,
                                                                                                  len(points)))]
    eval_points = [trajectory_point(traj, k, None, n_train) for traj in eval_trajectories for k in range(steps)]

    def loss_of(batch: list[Point]) -> Tensor:
        """--match output: |student(x) - teacher(x)|^2, both with the LoRA. --match effect: the LoRA's effect
        student(x) - student_without_lora(x) against teacher(x) - teacher_without_lora(x)."""
        x = torch.cat([p.x for p in batch]).to(device)
        timestep = torch.tensor([p.t for p in batch], dtype=torch.float32, device=device).to(x.dtype) / 1000
        cond = (torch.cat([s.conds[p.prompt][0] for p in batch]).to(device), s.conds[batch[0].prompt][1].to(device))
        with torch.no_grad():
            if all(p.v is not None for p in batch):  # recorded while the teacher sampled
                target = torch.cat([p.v for p in batch]).to(device).float()
                if effect:
                    target -= torch.cat([p.v0 for p in batch]).to(device).float()
            else:  # --timesteps continuous: run the teacher here
                target = velocity(teacher, x, timestep, cond, latent_ids, dtype).float()
                if effect:
                    target -= without_lora(teacher, x, timestep, cond).float()
            if effect:
                target += without_lora(student, x, timestep, cond).float()
        # No autocast: the model already is in its dtype and the LoRA branch runs in float32 by itself, so the
        # student trains with exactly the arithmetic it samples with.
        pred = velocity(student, x, timestep, cond, latent_ids, dtype)
        return F.mse_loss(pred.float(), target.float())

    @torch.no_grad()
    def mean_loss(items: list[Point]) -> float:
        return sum(loss_of([p]).item() for p in items) / max(1, len(items))

    def draw(n: int) -> list[Point]:
        chosen = rng.sample(points, n) if n <= len(points) else [rng.choice(points) for _ in range(n)]
        live = args.timesteps == "continuous"
        return [trajectory_point(trajectories[i], k, rng.random() if live else None, n_train) for i, k in chosen]

    history: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []

    def evaluate(step: int) -> dict[str, Any]:
        arm = "d" if step == 0 else "e"
        with run.phase("evaluate", announce=False):
            entry: dict[str, Any] = {"step": step, "eval_loss": mean_loss(eval_points),
                                     "train_probe_loss": mean_loss(probe)}
            rows = []
            for r, (prompt, seed) in enumerate(eval_rows):
                lat[arm][r], traj = _sample(s, student, prompt, noises[r], steps, record=True)
                first[arm][r] = traj["v"][0]
                images[arm][r] = decode(s.pipe, lat[arm][r], s.width, s.height, device)
                m = retention(lat["a"][r], lat["b"][r], lat["c"][r], lat[arm][r])
                m.update(first_step({a: first[a][r] for a in ("a", "b", "c", arm)}, arm))
                rows.append({"prompt": prompt, "seed": seed, **m,
                             "psnr_vs_b": _finite(psnr(images[arm][r], images["b"][r]))})
            for key in ("retention_cos", "retention_norm", "first_step_cos", "first_step_norm", "psnr_vs_b"):
                entry[key] = _mean([row[key] for row in rows])
            entry["rows"] = rows
        history.append(entry)
        run.log(f"  step {step}: held-out loss {entry['eval_loss']:.5f}, train-probe loss "
                f"{entry['train_probe_loss']:.5f}, retention cos {_fmt(entry['retention_cos'])}, "
                f"|D| {_fmt(entry['retention_norm'])}, first-step cos {_fmt(entry['first_step_cos'])}, "
                f"PSNR against klein + LoRA {_db(entry['psnr_vs_b'])} dB")
        return entry

    def result(done: int) -> dict[str, Any]:
        return {"command": "realign", "argv": args.argv, "lora": spec.text, "strength_folded_into_B": folded,
                "sources": vars(s.sources), "teacher_lora": teacher_lora, "student_lora": student_lora,
                "trigger": info["trigger"], "train_prompts": info["train_prompts"],
                "eval_rows": [{"prompt": p, "seed": seed} for p, seed in eval_rows],
                "width": s.width, "height": s.height, "sample_steps": steps, "guidance": args.guidance,
                "max_sequence_length": args.max_sequence_length, "dtype": str(dtype).replace("torch.", ""),
                "timesteps": args.timesteps, "match": args.match, "trajectories": len(trajectories),
                "training_points": len(points),
                "steps_done": done, "steps": args.steps, "lr": args.lr, "batch_size": args.batch_size,
                "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
                "gradient_checkpointing": not args.no_gradient_checkpointing, "trainable_tensors": len(trainable),
                "trainable_parameters": sum(p.numel() for p in params),
                "before": history[0], "after": history[-1], "history": history, "log": log,
                "files": {"diffusers": "pytorch_lora_weights.safetensors", "peft": "peft/", "grid": "eval_grid.png"},
                "environment": environment(device), **run.summary()}

    def save(done: int) -> None:
        save_realigned(student, name, run.out, {"from_lora": spec.text, "step": done,
                                                "strength_folded_into_B": folded})
        run.write_json("realign.json", result(done))

    before = evaluate(0)
    window: list[float] = []
    train_seconds = 0.0  # optimizer steps only, without the evaluations
    for step in range(1, args.steps + 1):
        t = time.perf_counter()
        loss = loss_of(draw(args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)) if args.max_grad_norm else None
        optimizer.step()
        window.append(loss.item())
        train_seconds += time.perf_counter() - t
        run.timings["train"] = round(train_seconds, 3)
        if step % args.log_every == 0 or step == args.steps:
            log.append({"step": step, "loss": sum(window) / len(window), "train_seconds": round(train_seconds, 2),
                        "grad_norm": grad_norm})
            run.log(f"  step {step}/{args.steps}: loss {log[-1]['loss']:.5f}, {train_seconds / step:.2f} s per step, "
                    f"about {(args.steps - step) * train_seconds / step / 60:.1f} min of training left")
            window = []
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            save(step)
    if not args.steps:
        save(0)
    after = history[-1]
    arms = ["a", "b", "c", "d"] + (["e"] if args.steps else [])
    labels = [f"({a}) {ARMS[a]}" for a in arms]
    labels[3] = "(d) Bonsai + LoRA (before)"
    make_grid([[to_pil(images[a][r]) for a in arms] for r in range(len(eval_rows))], labels,
              [f"{p} (seed {seed})" for p, seed in eval_rows], cell=args.grid_cell).save(run.path("eval_grid.png"))
    out = result(args.steps)
    run.write_json("realign.json", out)
    run.log(f"held-out loss {before['eval_loss']:.5f} -> {after['eval_loss']:.5f}; retention cos "
            f"{_fmt(before['retention_cos'])} -> {_fmt(after['retention_cos'])}, |D| {_fmt(before['retention_norm'])} "
            f"-> {_fmt(after['retention_norm'])}, first-step cos {_fmt(before['first_step_cos'])} -> "
            f"{_fmt(after['first_step_cos'])}; LoRA saved in {run.out}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Command line


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("models and device")
    g.add_argument("--lora", action="append", metavar="SOURCE[:STRENGTH]",
                   help="LoRA file, folder, Hub repo id or owner/repo/file.safetensors, with an optional strength "
                        "(default 1.0); repeatable")
    g.add_argument("--fp-model", default=FP_MODEL, help=f"full-precision FLUX.2 klein (default {FP_MODEL})")
    g.add_argument("--ternary-model", default=BONSAI_MLX, help=f"ternary Bonsai transformer (default {BONSAI_MLX})")
    g.add_argument("--components", default=BONSAI_UNPACKED,
                   help=f"text encoder, tokenizer, VAE and scheduler (default {BONSAI_UNPACKED}; its transformer "
                        "is not downloaded)")
    g.add_argument("--device", default="auto", help="auto (cuda, then mps, then cpu), mps, cuda or cpu")
    g.add_argument("--dtype", default="auto", choices=["auto", *DTYPES],
                   help="model dtype (auto: bfloat16, the models' own, float32 with --tiny). bfloat16 on an Apple "
                        "GPU needs macOS 14 or newer; float16 can overflow in FLUX.2's activations")
    g.add_argument("--cache-dense", action="store_true",
                   help="keep each packed layer's dense weight after its first call (faster, but as much memory "
                        "as the full-precision model)")
    g.add_argument("--tiny", action="store_true", help="build and use a tiny random pipeline (tests, dry runs)")
    g.add_argument("--tiny-same-base", action="store_true",
                   help="with --tiny: the full-precision model holds the ternary weights (both bases are equal)")
    g.add_argument("--tiny-seed", type=int, default=0, help=argparse.SUPPRESS)
    s = common.add_argument_group("sampling")
    s.add_argument("--prompt", action="append", help="a prompt (repeatable)")
    s.add_argument("--prompts", metavar="FILE", help="a text file with one prompt per line")
    s.add_argument("--size", help="N or WxH in pixels (default 1024 on a GPU, 512 on the CPU; realign: 512)")
    s.add_argument("--sample-steps", type=int, default=STEPS, help=f"sampler steps (default {STEPS})")
    s.add_argument("--guidance", type=float, default=GUIDANCE,
                   help=f"guidance scale (default {GUIDANCE}; ignored by step-distilled models)")
    s.add_argument("--max-sequence-length", type=int, default=None,
                   help="text tokens per prompt (default 512 as in diffusers; 32 with --tiny)")
    common.add_argument("--out", help="output folder (default flux2_results/<command>-<time>)")
    common.add_argument("--quiet", action="store_true", help="print nothing")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", parents=[common], help="generate images with Bonsai and LoRAs")
    gen.add_argument("--seed", type=int, nargs="+", default=[0])
    gen.add_argument("--base", choices=["ternary", "fp"], default="ternary",
                     help="ternary: packed Bonsai (default); fp: full-precision klein 4B")
    cmp_ = sub.add_parser("compare", parents=[common], help="LoRA effect on klein 4B against Bonsai")
    cmp_.add_argument("--seed", type=int, nargs="+", default=[0])
    cmp_.add_argument("--realigned", metavar="SOURCE[:STRENGTH]", help="also run Bonsai with this re-aligned LoRA (e)")
    cmp_.add_argument("--grid-cell", type=int, default=320, help="pixels per image in grid.png")
    rea = sub.add_parser("realign", parents=[common], help="re-align a LoRA to Bonsai by distillation")
    rea.add_argument("--steps", type=int, default=300, help="optimizer steps (default 300)")
    rea.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default 1e-4)")
    rea.add_argument("--batch-size", type=int, default=1)
    rea.add_argument("--weight-decay", type=float, default=0.0)
    rea.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping (0: off)")
    rea.add_argument("--trajectories", type=int, default=2, help="teacher trajectories per training prompt")
    rea.add_argument("--timesteps", choices=["trajectory", "continuous"], default="trajectory",
                     help="trajectory: the sampler's own steps, teacher outputs recorded (teacher freed); "
                          "continuous: any point of the trajectories, teacher run live (both models in memory)")
    rea.add_argument("--match", choices=["output", "effect"], default="output",
                     help="output: student + LoRA matches teacher + LoRA (EfficientDM; the LoRA also absorbs part "
                          "of the gap between the bases); effect: the LoRA's effect on the student matches its "
                          "effect on the teacher (keeps strength 0 = plain Bonsai; aims at retention 1)")
    rea.add_argument("--eval-prompt", action="append", help="held-out prompt (repeatable)")
    rea.add_argument("--eval-prompts", metavar="FILE", help="held-out prompts, one per line")
    rea.add_argument("--trigger", help="trigger words put in front of the built-in prompts (default: from the "
                                       "LoRA's model card; '' for none)")
    rea.add_argument("--eval-every", type=int, default=100, help="evaluate and save every N steps")
    rea.add_argument("--log-every", type=int, default=10)
    rea.add_argument("--probe", type=int, default=16, help="training points in the train-probe loss")
    rea.add_argument("--seed", type=int, default=0, help="seed of the noise and of the sampling of training points")
    rea.add_argument("--no-gradient-checkpointing", action="store_true")
    rea.add_argument("--grid-cell", type=int, default=256, help="pixels per image in eval_grid.png")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    if args.max_sequence_length is None:
        args.max_sequence_length = 32 if args.tiny else 512
    if args.command == "realign":
        if args.steps < 0 or args.batch_size < 1 or args.trajectories < 1 or args.eval_every < 1 or args.log_every < 1:
            raise SystemExit("--steps must be >= 0; --batch-size, --trajectories, --eval-every, --log-every >= 1")
    command = {"generate": cmd_generate, "compare": cmd_compare, "realign": cmd_realign}[args.command]
    threads = torch.get_num_threads()
    if args.tiny:
        torch.set_num_threads(1)  # tiny matrices: threads only add overhead, a lot of it on a busy machine
    try:
        return command(args)
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    main()
