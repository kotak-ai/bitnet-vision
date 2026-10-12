#!/usr/bin/env python
"""Correctness and training readiness of bitnet_vision on an Apple GPU (PyTorch MPS, the Metal kernels, MLX).

  python scripts/apple_check.py                       # on the Mac: MPS float32 against CPU float64
  python scripts/apple_check.py --only 2,6            # some items only
  python scripts/apple_check.py --device cpu --quick  # dry run anywhere: CPU float32 against CPU float64

Every item compares the device, in float32, with a float64 reference on the CPU (MPS has no float64). The limits:

* Plain float math (full-precision layers, gradients that do not depend on int8 codes): within 10 times the
  error of the same computation in float32 on the CPU, plus 1e-5 relative.
* One quantized layer: its int8 activation codes may differ from float64's only next to a rounding tie (by 1, in
  at most 1 of 1000 entries), and its output, recomputed in float64 from the device's own codes, must agree to
  float32 rounding. Comparing raw outputs would be flaky: one flipped code moves a token by about 1e-3. Its
  ternary weight codes must match exactly: the test weights are first moved at least 1e-5 away from the weight
  rounding ties (|w| / mean|w| = 0.5), where a float32 mean|w| a few ulps off could flip a code.
* A whole quantized model: an ulp-sized change flips codes that spread through the blocks, so the yardstick is
  how far the CPU float32 model lands from float64, with and without every input nudged by one ulp; the device
  passes within 10 times that (plus 1e-3). This catches broken kernels, not small biases.

Items: 1 torch version and device, 2 BitLinear and HBitLinear forward and backward, 3 matmul gradients that reduce
over more than 32768 rows (pytorch#177116, M1/M2 GPUs), 4 forward-mode AD (torch.func.jvp) through attention and an
iMF-style step, 5 the packed model (convert_for_inference) and autocast, 6 the fused Metal kernels, 7 MLX, 8 one
training step of a T/4-sized BitViT. Items 5, 6 and 8 also run with LoRA adapters (bitnet_vision.lora). Each row
reports ok, FAILED (with the error), known (a PyTorch bug this repo does not hit) or skipped (with the reason).
The table is printed and everything, with the machine's details, goes to apple_results/apple_check-<time>.json.

Shapes: TernFlow's T/4 (width 192, MLP 768, 74 tokens per image) and S/2 (384, 1536, 266 tokens). The BitViT
stand-ins built with create_model have 65 and 257 tokens per 32x32 image.
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
import re
import subprocess
import sys
import textwrap
import time
import traceback
import types
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import jvp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bitnet_vision.apple import mlx_ternary as mt  # noqa: E402
from bitnet_vision.apple import mps_kernels  # noqa: E402
from bitnet_vision.apple.mps_kernels import convert_to_fused, fused_front_end, metal_available  # noqa: E402
from bitnet_vision.export import BitLinearInference, convert_for_inference, unpack_ternary  # noqa: E402
from bitnet_vision.layers import BitLinear, HBitLinear, RMSNorm, bitlinear_modules, set_quant_strength  # noqa: E402
from bitnet_vision.lora import add_lora, mark_only_lora_trainable, set_lora_strength  # noqa: E402
from bitnet_vision.models import BitViT, create_model  # noqa: E402
from bitnet_vision.models.vit import Attention, BitViTConfig, Block  # noqa: E402
from bitnet_vision.quant import fake_quant_weights, quantize_activations, quantize_weights  # noqa: E402

# torch.func.jvp trips an internal torch.jit deprecation warning on recent PyTorch; it says nothing about this code.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)

# ---------------------------------------------------------------------------------------------------------------
# Shared with scripts/apple_bench.py

STATUSES = ("ok", "FAILED", "known", "skipped")
# create_model overrides for the BitViT stand-ins of TernFlow's two sizes; the quick ones are for dry runs.
MODELS = {
    "T/4": dict(embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0, patch_size=4, img_size=32, num_classes=10),
    "S/2": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=2, img_size=32, num_classes=10),
}
QUICK_MODELS = {
    "T/4": dict(embed_dim=64, depth=1, num_heads=2, mlp_ratio=4.0, patch_size=4, img_size=16, num_classes=10),
    "S/2": dict(embed_dim=128, depth=1, num_heads=2, mlp_ratio=4.0, patch_size=4, img_size=16, num_classes=10),
}
TOKENS = {"T/4": 74, "S/2": 266}  # TernFlow tokens per image
QUICK_TOKENS = {"T/4": 9, "S/2": 33}
ENV_VARS = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_DISABLE_PREFILL_ATTENTION",
    "OMP_NUM_THREADS",
)


class Skip(Exception):
    """Raised by a check or benchmark to report it as skipped; the message is the reason."""


def model_config(size: str, quick: bool) -> dict[str, Any]:
    return dict((QUICK_MODELS if quick else MODELS)[size])


def build_model(size: str, quick: bool, **overrides: Any) -> BitViT:
    """The BitViT stand-in for TernFlow's T/4 or S/2 (random weights)."""
    return create_model("bitvit_cifar_tiny", **{**model_config(size, quick), **overrides})


def add_lora_for_check(model: nn.Module) -> None:
    """Two LoRA adapters (one at half strength) on every ternary layer and the head, with non-zero B."""
    add_lora(model, r=8, targets=("qkv", "proj", "fc1", "fc2", "head"))
    add_lora(model, r=4, name="second", targets=("fc1",))
    set_lora_strength(model, 0.5, "second")
    g = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))


def linear_shapes(quick: bool) -> list[tuple[str, int, int, int]]:
    """(name, in_features, out_features, TernFlow tokens per image) of each ternary layer of T/4 and S/2."""
    shapes = []
    for size in ("T/4", "S/2"):
        cfg = model_config(size, quick)
        d, m = cfg["embed_dim"], (QUICK_TOKENS if quick else TOKENS)[size]
        h = int(d * cfg["mlp_ratio"])
        shapes += [(f"{size} qkv", d, 3 * d, m), (f"{size} proj", d, d, m), (f"{size} fc1", d, h, m),
                   (f"{size} fc2", h, d, m)]
    return shapes


def _command(*cmd: str, timeout: float = 30.0) -> str | None:
    """Output of a command, or None if it is missing, fails or times out."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def parse_gpu_info(text: str) -> dict[str, Any]:
    """GPU name and core count from the output of ``system_profiler SPDisplaysDataType``."""
    info: dict[str, Any] = {}
    if m := re.search(r"Chipset Model:\s*(.+)", text):
        info["gpu"] = m.group(1).strip()
    if m := re.search(r"Total Number of Cores:\s*(\d+)", text):
        info["gpu_cores"] = int(m.group(1))
    return info


def chip_generation(chip: str | None) -> int | None:
    """N of an "Apple MN ..." chip name (2 for "Apple M2 Max"), or None."""
    m = re.search(r"\bM(\d+)\b", chip or "")
    return int(m.group(1)) if m else None


def torch_version_tuple() -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def mlx_info() -> dict[str, Any] | None:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    info: dict[str, Any] = {"version": getattr(mx, "__version__", "unknown"), "metal": bool(mx.metal.is_available()),
                            "default_device": str(mx.default_device())}
    device_info = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
    with contextlib.suppress(Exception):
        info["device_info"] = {k: v if isinstance(v, (int, float, str, bool)) else str(v)
                               for k, v in device_info().items()}
    return info


def environment() -> dict[str, Any]:
    """Machine, software and machine-state details that change the numbers (as scripts/bench.py records)."""
    env: dict[str, Any] = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "metal_kernels_available": metal_available(),
        "env": {k: os.environ[k] for k in ENV_VARS if k in os.environ},
    }
    with contextlib.suppress(AttributeError):
        env["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    if hasattr(os, "getloadavg"):
        env["load_average_1m"] = round(os.getloadavg()[0], 2)
    env["git_commit"] = _command("git", "-C", ROOT, "rev-parse", "HEAD")
    if env["git_commit"]:
        env["git_dirty"] = bool(_command("git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"))
    if sys.platform == "darwin":
        env["chip"] = _command("sysctl", "-n", "machdep.cpu.brand_string")
        env["macos"] = platform.mac_ver()[0] or _command("sw_vers", "-productVersion")
        mem = _command("sysctl", "-n", "hw.memsize")
        if mem and mem.isdigit():
            env["memory_gb"] = round(int(mem) / 2**30, 1)
        env.update(parse_gpu_info(_command("system_profiler", "SPDisplaysDataType", timeout=60) or ""))
        batt = _command("pmset", "-g", "batt")
        if batt:
            env["power_source"] = batt.splitlines()[0]  # e.g. "Now drawing from 'AC Power'"
        if m := re.search(r"lowpowermode\s+(\d)", _command("pmset", "-g") or ""):
            env["low_power_mode"] = m.group(1) == "1"
    if env["mps_available"]:
        with contextlib.suppress(Exception):
            env["mps_recommended_max_memory_gb"] = round(torch.mps.recommended_max_memory() / 2**30, 1)
    try:
        env["mlx"] = mlx_info()
    except Exception as e:  # noqa: BLE001 - a broken MLX install is reported, not fatal
        env["mlx"] = {"error": f"{type(e).__name__}: {e}"}
    return env


def describe(env: dict[str, Any]) -> str:
    """One line about the machine and the software versions."""
    hw = [env.get("chip") or env["machine"]]
    if env.get("gpu_cores"):
        hw.append(f"{env['gpu_cores']} GPU cores")
    if env.get("memory_gb"):
        hw.append(f"{env['memory_gb']:g} GB")
    if env.get("macos"):
        hw.append(f"macOS {env['macos']}")
    mlx = env.get("mlx")
    sw = [f"torch {env['torch']}", f"MLX {mlx.get('version', '(broken)')}" if mlx else "no MLX",
          f"commit {(env.get('git_commit') or 'unknown')[:10]}" + (" (modified)" if env.get("git_dirty") else "")]
    return ", ".join(hw) + "; " + ", ".join(sw)


def environment_notes(env: dict[str, Any]) -> list[str]:
    notes = []
    if env["env"].get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        notes.append("PYTORCH_ENABLE_MPS_FALLBACK=1: ops without an MPS kernel silently run on the CPU")
    if env["env"].get("PYTORCH_MPS_FAST_MATH") == "1":
        notes.append("PYTORCH_MPS_FAST_MATH=1: the Metal kernels are not expected to match the CPU exactly")
    if env.get("low_power_mode"):
        notes.append("Low Power Mode is on: speeds are lower than normal")
    if "Battery" in (env.get("power_source") or ""):
        notes.append("running on battery: speeds can be lower than on AC power")
    return notes


def free_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
        with contextlib.suppress(Exception):
            clear()


def need_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise Skip("no MPS device (torch.backends.mps.is_available() is False)")


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def mlx_device(device: str) -> tuple[Any, Any]:
    """(mlx.core, the MLX device that matches ``device``: the GPU for "mps", the CPU for "cpu"), or Skip."""
    try:
        import mlx.core as mx
    except ImportError:
        raise Skip("MLX is not installed (pip install mlx)") from None
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise Skip("NumPy is not installed (pip install numpy); bitnet_vision's MLX port needs it") from None
    if device == "mps":
        if not mx.metal.is_available():
            raise Skip("MLX has no Metal GPU here")
        return mx, mx.gpu
    return mx, mx.cpu


def explicit_attention_forward(self: Attention, x: Tensor) -> Tensor:
    """``Attention.forward`` with softmax(q k^T / sqrt(d)) v written out instead of F.scaled_dot_product_attention.

    Forward-mode AD (torch.func.jvp) works through these ops on MPS; F.scaled_dot_product_attention there routes to
    an MPS-only kernel without a forward-AD formula whenever no input requires grad.
    """
    b, n, c = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    attn = torch.softmax(q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5, dim=-1)
    return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


def use_explicit_attention(model: nn.Module) -> nn.Module:
    """Bind :func:`explicit_attention_forward` as the forward of every Attention module in ``model`` (in place)."""
    for m in model.modules():
        if isinstance(m, Attention):
            m.forward = types.MethodType(explicit_attention_forward, m)
    return model


def _first_line(text: str, width: int = 220) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= width else line[: width - 3] + "..."


def _json_default(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.tolist()
    return str(obj)


class Report:
    """The rows of one run. Each row is printed when it finishes and the JSON file is rewritten after every row,
    so an interrupted run keeps what it measured."""

    def __init__(self, script: str, args: argparse.Namespace, line: Callable[[dict], str],
                 free_each_row: bool = True) -> None:
        name = f"{script}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out = args.out or "apple_results"
        self.path = os.path.join(out, name) if os.path.isdir(out) or not out.endswith(".json") else out
        self.line, self.free_each_row = line, free_each_row
        self.data: dict[str, Any] = {"script": script, "args": vars(args), "environment": environment(), "rows": []}
        self.t0 = time.perf_counter()

    @property
    def rows(self) -> list[dict]:
        return self.data["rows"]

    @property
    def env(self) -> dict[str, Any]:
        return self.data["environment"]

    def run(self, row_id: str, name: str, fn: Callable[[], dict | None], **fields: Any) -> dict:
        """Run ``fn`` and record its row: ok (FAILED if it returns ok=False, or known when it also returns
        known=True: a PyTorch bug this repo does not hit), FAILED if it raises, skipped on Skip."""
        row: dict[str, Any] = {"id": row_id, "name": name, **fields}
        t = time.perf_counter()
        try:
            result = dict(fn() or {})
        except Skip as e:
            row.update(status="skipped", message=str(e))
        except Exception as e:  # noqa: BLE001 - a failure is a result: record it and go on
            row.update(status="FAILED", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        else:
            ok, known = result.pop("ok", True), result.pop("known", False)
            row.update(status="ok" if ok else "known" if known else "FAILED", message=result.pop("message", ""),
                       **result)
        row["seconds"] = round(time.perf_counter() - t, 3)
        self.rows.append(row)
        print(self.line(row), flush=True)
        self.save()
        if self.free_each_row and row["status"] != "skipped":
            free_memory()
        return row

    def counts(self) -> dict[str, int]:
        return {s: sum(r["status"] == s for r in self.rows) for s in STATUSES}

    def save(self) -> None:
        self.data["summary"] = self.counts()
        self.data["elapsed_s"] = round(time.perf_counter() - self.t0, 1)
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, default=_json_default)
        os.replace(tmp, self.path)

    def finish(self) -> None:
        self.data["environment"]["load_average_1m_end"] = round(os.getloadavg()[0], 2) if hasattr(
            os, "getloadavg") else None
        self.save()
        c = self.counts()
        print(f"\n{c['ok']} ok, {c['FAILED']} FAILED, {c['known']} known PyTorch issues, {c['skipped']} skipped in "
              f"{self.data['elapsed_s']:.0f} s. "
              f"Results: {self.path}")


def torch_warning(env: dict[str, Any], device: str) -> str | None:
    """A loud warning if this PyTorch silently corrupts gradients on this Mac's GPU, else None."""
    gen = chip_generation(env.get("chip"))
    if device != "mps" or torch_version_tuple() >= (2, 14) or (gen is not None and gen > 2):
        return None
    chip = env.get("chip") or "this Mac"
    return (f"torch {torch.__version__} on {chip}: PyTorch before 2.14 silently corrupts MPS matmul gradients on M1/M2 "
            "GPUs once a reduction dimension exceeds 32768 (pytorch#177116), e.g. the weight gradient of an S/2 "
            "layer at batch 124 or more. Upgrade before training: pip install -U \"torch>=2.14\"")


def banner(text: str) -> str:
    bar = "!" * 100
    return f"{bar}\n{textwrap.fill('WARNING: ' + text, 100)}\n{bar}"


# ---------------------------------------------------------------------------------------------------------------
# Comparison helpers

FACTOR = 10.0  # the device passes within FACTOR times the CPU float32 error (plus a floor)


def to(t: Tensor, device: str, dtype: torch.dtype) -> Tensor:
    """A copy of ``t`` on ``device``; floating tensors in ``dtype``. Always a copy, so a run that sets requires_grad
    or accumulates gradients never touches the shared inputs."""
    if t.is_floating_point():
        return t.to(device=device, dtype=dtype, copy=True)
    return t.to(device=device, copy=True)


def nudge(t: Tensor) -> Tensor:
    """Every value moved by one ulp toward +inf (the perturbation behind the chaotic-model yardstick)."""
    return t.nextafter(torch.full_like(t, math.inf)) if t.is_floating_point() else t


def on_cpu(d: dict[str, Tensor | None]) -> dict[str, Tensor]:
    """Detached CPU copies (never views of tensors that a later step could change in place)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in d.items() if v is not None}


def rel_err(a: Tensor, ref: Tensor) -> float:
    """max|a - ref| / max|ref| in float64 (inf if ``a`` has NaN or inf where the difference is taken)."""
    a, ref = a.detach().cpu().double(), ref.detach().cpu().double()
    if a.shape != ref.shape:
        raise ValueError(f"shape {tuple(a.shape)} differs from the reference {tuple(ref.shape)}")
    if a.numel() == 0:
        return 0.0
    diff = (a - ref).abs().max().item()
    if not math.isfinite(diff):
        return math.inf
    scale = ref.abs().max().item()
    return diff / scale if scale > 0 else diff


def entry(name: str, value: float, limit: float, yardstick: float | None = None, count: bool = False) -> dict[str, Any]:
    """One limit of a check. ``count`` marks whole-number limits (such as codes at most 1 apart), which are not named
    as the worst entry of a check that passes."""
    e: dict[str, Any] = {"name": name, "value": float(value), "limit": float(limit)}
    if yardstick is not None:
        e["yardstick"] = float(yardstick)
    if count:
        e["count"] = True
    return e


def _ratio(e: dict[str, Any]) -> float:
    value, limit = e["value"], e["limit"]
    if not math.isfinite(value):
        return math.inf
    if limit <= 0:
        return 0.0 if value <= 0 else math.inf
    return value / limit


def _fmt(v: float) -> str:
    return f"{v:.3g}" if v == 0 or 1e-3 <= abs(v) < 1e4 else f"{v:.2e}"


def verdict(entries: list[dict[str, Any]], note: str = "") -> dict[str, Any]:
    """ok if every entry's value is within its limit; the message names the entry nearest to (or over) its limit."""
    ok = all(e["value"] <= e["limit"] for e in entries)  # NaN fails
    worst = max([e for e in entries if not (ok and e.get("count"))] or entries, key=_ratio)
    msg = f"{'worst' if ok else 'OVER LIMIT'}: {worst['name']} {_fmt(worst['value'])} (limit {_fmt(worst['limit'])}"
    if "yardstick" in worst:
        msg += f", CPU float32 {_fmt(worst['yardstick'])}"
    msg += ")"
    return {"ok": ok, "message": f"{note}; {msg}" if note else msg, "checks": entries}


def run_three(run: Callable[..., dict], inputs: tuple[Tensor, ...], device: str, nudged: bool = False,
              device_run: Callable[..., dict] | None = None) -> tuple[dict, dict, list[dict]]:
    """``run(device, dtype, *inputs)`` on the device in float32, on the CPU in float64 (the reference) and on the CPU in
    float32 (the yardstick; with ``nudged`` also on inputs moved by one ulp). Returns (device, reference, [cpu runs]).
    ``device_run`` replaces ``run`` for the device only."""
    ref = run("cpu", torch.float64, *inputs)
    cpu = [run("cpu", torch.float32, *inputs)]
    if device == "cpu" and device_run is None:
        dev = cpu[0]
    else:
        dev = (device_run or run)(device, torch.float32, *inputs)
    if nudged:
        cpu.append(run("cpu", torch.float32, *[nudge(t) for t in inputs]))
    return dev, ref, cpu


def compare(dev: dict, ref: dict, cpu: list[dict], floor: float, keys: list[str] | None = None,
            prefix: str = "") -> list[dict[str, Any]]:
    """Per key: relative error of the device against float64, within FACTOR x the worst CPU float32 error + floor."""
    entries = []
    for key in keys or list(ref):
        yard = max(rel_err(c[key], ref[key]) for c in cpu)
        entries.append(entry(prefix + key, rel_err(dev[key], ref[key]), FACTOR * yard + floor, yard))
    return entries


def code_entries(codes: Tensor, ref_codes: Tensor, pre: Tensor, prefix: str = "",
                 tie_check: bool = True) -> tuple[list[dict[str, Any]], float]:
    """Device int8 codes against float64's: at most 1 apart, in at most 1 of 1000 entries (at least 2 per tensor),
    and (with ``tie_check``) only where float64's value before rounding is within 1e-3 of a rounding tie.
    Returns the entries and the share of codes that differ."""
    codes, ref_codes, pre = codes.detach().cpu().double(), ref_codes.detach().cpu().double(), pre.detach().cpu()
    if codes.shape != ref_codes.shape:
        raise ValueError(f"codes have shape {tuple(codes.shape)}, expected {tuple(ref_codes.shape)}")
    n = max(codes.numel(), 1)
    diff = (codes - ref_codes).abs()
    differ = ~(diff == 0)  # NaN counts as different
    share = differ.sum().item() / n
    entries = [entry(prefix + "codes: largest difference", diff.max().item() if codes.numel() else 0.0, 1, count=True),
               entry(prefix + "codes: share that differ", share, max(2, codes.numel() // 1000) / n)]
    if tie_check:
        pre = pre.double()
        away = differ & ((pre - pre.floor() - 0.5).abs() > 1e-3)
        entries.append(entry(prefix + "codes: differences away from a rounding tie", away.sum().item(), 0, count=True))
    return entries, share


def reference_given_codes(layer64: BitLinearInference, x: Tensor, in_dtype: torch.dtype = torch.float32):
    """Float64 reference of an exported layer on input ``x``: (output as a function of activation codes, float64's own
    codes, float64's values before rounding, float64's activation scales). Normalized and rotated values are rounded
    to ``in_dtype`` when it is half precision, as RMSNorm and HadamardTransform do."""
    h = x.detach().cpu().double()
    half = in_dtype in (torch.float16, torch.bfloat16)
    if layer64.norm is not None:
        h = layer64.norm(h)
        h = h.to(in_dtype).double() if half else h
    if layer64.hadamard is not None:
        h = layer64.hadamard(h)
        h = h.to(in_dtype).double() if half else h
    codes, scale = quantize_activations(h, bits=layer64.act_bits, method=layer64.act_method)
    w = unpack_ternary(layer64.weight_packed.cpu(), layer64.out_features, dtype=torch.float64)
    ws = layer64.weight_scale.detach().cpu().double()
    bias = layer64.bias.detach().cpu().double() if layer64.bias is not None else None

    def output(c: Tensor) -> tuple[Tensor, float]:
        term = (c.detach().cpu().double() @ w.T) / scale * ws
        size = term.abs().max().item() if term.numel() else 0.0
        return (term + bias if bias is not None else term), size

    return output, codes, h * scale, scale


def given_codes_err(output: Callable, y: Tensor, codes: Tensor) -> float:
    """Error of ``y`` against the float64 output recomputed from ``codes``, relative to the larger of the output and
    the product term (so a bias that cancels the product does not hide an error)."""
    ref, size = output(codes)
    if ref.numel() == 0:
        return 0.0
    diff = (y.detach().cpu().double() - ref).abs().max().item()
    if not math.isfinite(diff):
        return math.inf
    return diff / max(ref.abs().max().item(), size, 1e-30)


def inference_entries(layer64: BitLinearInference, x: Tensor, dev: tuple[Tensor, ...], cpu: tuple[Tensor, ...],
                      prefix: str = "", in_dtype: torch.dtype = torch.float32,
                      floor: float = 1e-5) -> tuple[list[dict[str, Any]], float]:
    """Entries for one exported layer: device codes against float64 (``code_entries``) and the device output against
    float64 recomputed from the device's own codes, within FACTOR x CPU float32's error + floor. ``dev`` and ``cpu``
    are (output, codes) or (output, codes, activation scales); scales are checked when given."""
    output, codes64, pre, scale64 = reference_given_codes(layer64, x, in_dtype)
    half = in_dtype in (torch.float16, torch.bfloat16)
    entries, share = code_entries(dev[1], codes64, pre, prefix, tie_check=not half)
    yard = given_codes_err(output, cpu[0], cpu[1])
    entries.append(entry(prefix + "output given its codes", given_codes_err(output, dev[0], dev[1]),
                         FACTOR * yard + floor, yard))
    if len(dev) > 2 and len(cpu) > 2:
        # For half input the rounding of a token's largest normalized value can flip by one half-precision step.
        yard = rel_err(cpu[2], scale64)
        entries.append(entry(prefix + "activation scales", rel_err(dev[2], scale64),
                             FACTOR * yard + (2**-10 if half else 1e-6), yard))
    return entries, share


def tokens(shape: tuple[int, ...], seed: int = 0) -> Tensor:
    """Random float32 tokens whose sizes differ a lot from row to row, as real activations do."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * torch.exp(torch.randn(*shape[:-1], 1, generator=g))


def init_layer(layer: BitLinear, seed: int = 0) -> BitLinear:
    """Non-trivial bias and RMSNorm weights, so mistakes in either show up."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if layer.bias is not None:
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=g) * 0.5)
        if layer.norm is not None:
            layer.norm.weight.copy_(torch.rand(layer.norm.weight.shape, generator=g) + 0.5)
    return layer


def clear_of_weight_ties(model: nn.Module, margin: float = 1e-5) -> nn.Module:
    """Move every ternary latent weight with |w| / mean|w| within ``margin`` of 0.5 (a rounding tie) out to
    0.5 + 2 * margin, in place. There a float32 mean|w| a few ulps off float64's can flip the weight's code (the
    random T/4 model has one weight 4e-8 from a tie), so weight codes can then be compared exactly."""
    with torch.no_grad():
        for layer in bitlinear_modules(model):
            w = layer.weight
            for _ in range(3):  # moving weights changes mean|w| slightly; settles at once in practice
                a = w.detach().double().abs()
                gamma = a.mean() if layer.weight_granularity == "tensor" else a.mean(dim=1, keepdim=True)
                gamma = gamma.expand_as(a)
                near = (a / gamma - 0.5).abs() < margin
                if not near.any():
                    break
                w[near] = (w[near].double().sign() * (0.5 + 2 * margin) * gamma[near]).to(w.dtype)
    return model


def exported(cls: type[BitLinear], k: int, n: int, seed: int = 0, **kwargs: Any) -> BitLinearInference:
    torch.manual_seed(seed)
    return convert_for_inference(init_layer(cls(k, n, **kwargs), seed), int_matmul="never")


@contextlib.contextmanager
def count_fallbacks():
    """Count calls into mps_kernels' reference ops (the non-Metal path) while the block runs."""
    calls = {"n": 0}
    originals = {name: getattr(mps_kernels, name) for name in ("_front_end_reference", "_epilogue_reference")}

    def counted(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return fn(*args, **kwargs)

        return wrapper

    for name, fn in originals.items():
        setattr(mps_kernels, name, counted(fn))
    try:
        yield calls
    finally:
        for name, fn in originals.items():
            setattr(mps_kernels, name, fn)


def inference_codes(layer: BitLinearInference, x: Tensor) -> tuple[Tensor, Tensor]:
    """(activation codes, scales) of an exported layer, computed with its own ops on x's device."""
    h = layer.norm(x) if layer.norm is not None else x
    if layer.hadamard is not None:
        h = layer.hadamard(h)
    return quantize_activations(h, bits=layer.act_bits, method=layer.act_method)


# ---------------------------------------------------------------------------------------------------------------
# Items


class Checks:
    def __init__(self, report: Report, device: str, quick: bool) -> None:
        self.report, self.device, self.quick = report, device, quick
        self.env = report.env

    # 1 -------------------------------------------------------------------------------------------------------
    def torch_version(self) -> dict:
        v = torch_version_tuple()
        msg = f"torch {torch.__version__}"
        warning = torch_warning(self.env, self.device)
        if warning:
            return {"ok": False, "message": warning}
        if v >= (2, 14):
            msg += " >= 2.14, which fixes MPS matmul gradients on M1/M2 GPUs (pytorch#177116)"
        elif self.device == "mps":
            msg += f" < 2.14: {self.env.get('chip')} is not affected by pytorch#177116, but 2.14 is recommended"
        else:
            msg += " < 2.14: fine for this CPU dry run, too old for training on an M1/M2 GPU (pytorch#177116)"
        if v < (2, 7):
            msg += "; the fused Metal kernels need torch >= 2.7"
        return {"message": msg}

    def device_info(self) -> dict:
        notes = environment_notes(self.env)
        if self.device == "cpu":
            return {"message": "; ".join(["dry run: the 'device' side is CPU float32 (use --device mps on the Mac)",
                                          *notes])}
        if not torch.backends.mps.is_available():
            why = "torch.backends.mps.is_available() is False"
            if not torch.backends.mps.is_built():
                why += ": this PyTorch build has no MPS support"
            elif platform.machine() != "arm64":
                why += ": Python runs as x86_64 (Rosetta); install an arm64 Python"
            else:
                why += " (MPS needs macOS 14 or newer)"
            return {"ok": False, "message": why}
        x = torch.arange(16.0, device="mps").reshape(4, 4)
        if not torch.equal((x @ x).cpu(), x.cpu() @ x.cpu()):
            return {"ok": False, "message": "a 4x4 matmul on MPS gives the wrong result"}
        parts = [describe(self.env)]
        if "mps_recommended_max_memory_gb" in self.env:
            parts.append(f"MPS working set up to {self.env['mps_recommended_max_memory_gb']:g} GB")
        parts.append("Metal kernels available" if metal_available() else "no torch.mps.compile_shader (torch < 2.7)")
        return {"message": "; ".join(parts + notes)}

    # 2 -------------------------------------------------------------------------------------------------------
    def bitlinear(self, cls: type[BitLinear], k: int, n: int, m: int) -> dict:
        need_device(self.device)
        torch.manual_seed(k + n)
        master = clear_of_weight_ties(init_layer(cls(k, n), seed=k + n))
        x = tokens((m, k), seed=1)
        gy = torch.randn(m, n, generator=torch.Generator().manual_seed(2))

        def run(device: str, dtype: torch.dtype, x: Tensor, gy: Tensor) -> dict:
            layer = copy.deepcopy(master).to(device=device, dtype=dtype)
            x = to(x, device, dtype).requires_grad_()
            y = layer(x)
            y.backward(to(gy, device, dtype))
            with torch.no_grad():
                h = layer.rotate(layer.norm(x.detach()))
                codes, scale = quantize_activations(h)
            return on_cpu(dict(y=y, codes=codes, pre=h * scale, scale=scale, grad_x=x.grad, grad_b=layer.bias.grad,
                               grad_norm=layer.norm.weight.grad, grad_w=layer.weight.grad,
                               w_codes=quantize_weights(layer.weight.detach())[0],
                               wq=fake_quant_weights(layer.weight.detach()), b=layer.bias.detach()))

        dev, ref, cpu = run_three(run, (x, gy), self.device)
        entries, share = code_entries(dev["codes"], ref["codes"], ref["pre"])
        w_diff = (dev["w_codes"] != ref["w_codes"]).sum().item()
        entries.append(entry("weight codes that differ", w_diff, 0, count=True))

        def given(r: dict, key: str) -> Tensor:  # float64 y or weight gradient computed from r's activation codes
            xq = r["codes"].double() / ref["scale"]
            return xq @ ref["wq"].T + ref["b"] if key == "y" else gy.double().T @ xq

        for key in ("y", "grad_w"):
            yard = rel_err(cpu[0][key], given(cpu[0], key))
            entries.append(entry(f"{key} given its codes", rel_err(dev[key], given(dev, key)), FACTOR * yard + 1e-5,
                                 yard))
        entries += compare(dev, ref, cpu, 1e-5, keys=["grad_x", "grad_b", "grad_norm"])
        return verdict(entries, note=f"{share:.3%} of codes differ")

    # 3 -------------------------------------------------------------------------------------------------------
    def large_reduction(self, kind: str) -> dict:
        need_device(self.device)
        g = torch.Generator().manual_seed(0)
        d = 32 if self.quick else 384
        if kind == "bmm":
            rows = 32800 if self.quick else 40000
            inputs = (torch.randn(2, d, rows, generator=g), torch.randn(2, rows, d, generator=g))

            def run(device: str, dtype: torch.dtype, a: Tensor, b: Tensor) -> dict:
                return on_cpu({"bmm": torch.bmm(to(a, device, dtype), to(b, device, dtype))})
        else:
            if kind == "2d":
                shape = (32800,) if self.quick else (40000,)
            else:
                shape = (2, 16400) if self.quick else (150, 266)  # S/2 tokens at batch 150
            rows = math.prod(shape)
            inputs = (torch.randn(*shape, d, generator=g), torch.randn(*shape, d, generator=g),
                      torch.randn(d, d, generator=g) / d**0.5)

            def run(device: str, dtype: torch.dtype, x: Tensor, gy: Tensor, w: Tensor) -> dict:
                x, w = to(x, device, dtype).requires_grad_(), to(w, device, dtype).requires_grad_()
                F.linear(x, w).backward(to(gy, device, dtype))
                return on_cpu({"grad_w": w.grad, "grad_x": x.grad})

        dev, ref, cpu = run_three(run, inputs, self.device)
        result = verdict(compare(dev, ref, cpu, 1e-5), note=f"reduces over {rows:,} rows")
        if kind == "bmm" and not result["ok"]:
            # a real PyTorch bug on M1/M2 GPUs, but nothing in this repo runs a bmm that long: attention reduces over
            # head_dim and the token count (at most a few thousand), so report it without failing the run
            result["known"] = True
            result["message"] += ("; known PyTorch bug on M1/M2 GPUs: the 2.14 fix covers mm and linear, not bmm "
                                  "(pytorch#183535). Nothing in this repo runs a bmm that long")
        return result

    # 4 -------------------------------------------------------------------------------------------------------
    def _block(self, quantize: bool, explicit: bool) -> tuple[Block, Tensor, Tensor]:
        cfg = model_config("T/4", self.quick)
        torch.manual_seed(0)
        block = Block(BitViTConfig(**cfg))
        set_quant_strength(block, 1.0 if quantize else 0.0)
        clear_of_weight_ties(block)
        if explicit:
            use_explicit_attention(block)
        n = (QUICK_TOKENS if self.quick else TOKENS)["T/4"]
        x = tokens((2, n, cfg["embed_dim"]), seed=3)
        v = torch.randn(x.shape, generator=torch.Generator().manual_seed(4))
        return block, x, v

    @staticmethod
    def _jvp_run(block: Block) -> Callable[..., dict]:
        def run(device: str, dtype: torch.dtype, x: Tensor, v: Tensor) -> dict:
            b = copy.deepcopy(block).to(device=device, dtype=dtype)
            out, tangent = jvp(b, (to(x, device, dtype),), (to(v, device, dtype),))
            return on_cpu({"out": out, "tangent": tangent})

        return run

    def jvp_explicit(self, quantize: bool) -> dict:
        need_device(self.device)
        block, x, v = self._block(quantize, explicit=True)
        dev, ref, cpu = run_three(self._jvp_run(block), (x, v), self.device, nudged=quantize)
        # A quantized block is chaotic: a flipped int8 code moves a token by about 1e-3 (see the module docstring).
        return verdict(compare(dev, ref, cpu, 1e-2 if quantize else 1e-5))

    def imf_step(self) -> dict:
        need_device(self.device)
        block, x, v = self._block(quantize=False, explicit=True)

        def run(device: str, dtype: torch.dtype, x: Tensor, v: Tensor) -> dict:
            b = copy.deepcopy(block).to(device=device, dtype=dtype)
            x, v = to(x, device, dtype), to(v, device, dtype)
            out, tangent = jvp(b, (x,), (v,))
            loss = (out - (v - tangent.detach())).pow(2).mean()  # MeanFlow-style target with a stop-gradient
            loss.backward()
            return on_cpu({"loss": loss.reshape(1), "tangent": tangent,
                           **{f"grad {name}": p.grad for name, p in b.named_parameters()}})

        dev, ref, cpu = run_three(run, (x, v), self.device)
        return verdict(compare(dev, ref, cpu, 1e-5))

    def sdpa_forward_ad(self, how: str) -> dict:
        """F.scaled_dot_product_attention under forward-mode AD on the device, against explicit attention in float64.
        An error is the expected outcome on MPS for torch.func.jvp and is reported, not failed; wrong values fail."""
        need_device(self.device)
        block, x, v = self._block(quantize=False, explicit=False)
        b = copy.deepcopy(block).to(self.device)
        xd, vd = x.to(self.device), v.to(self.device)
        try:
            if how == "func.jvp":
                out, tangent = jvp(b, (xd,), (vd,))
            else:  # dual tensors with grad mode on and weights that require grad, as inside a training step
                with fwAD.dual_level():
                    out, tangent = fwAD.unpack_dual(b(fwAD.make_dual(xd, vd)))
        except Exception as e:  # noqa: BLE001 - what happens is the result
            hint = " (expected on MPS: use explicit attention under torch.func.jvp)" if how == "func.jvp" else ""
            return {"message": f"raises {type(e).__name__}: {_first_line(str(e), 120)}{hint}",
                    "outcome": type(e).__name__}
        dev = on_cpu({"out": out, "tangent": tangent})
        _, ref, cpu = run_three(self._jvp_run(use_explicit_attention(copy.deepcopy(block))), (x, v), "cpu")
        result = verdict(compare(dev, ref, cpu, 1e-5))
        result["message"] = ("works: " if result["ok"] else "runs but WRONG: ") + result["message"]
        result["outcome"] = "works" if result["ok"] else "wrong values"
        return result

    # 5 -------------------------------------------------------------------------------------------------------
    def _packed_model(self, lora: bool = False) -> tuple[BitViT, Tensor]:
        torch.manual_seed(0)
        model = build_model("T/4", self.quick, hadamard=True).eval()
        if lora:
            add_lora_for_check(model)
        cfg = model.config
        g = torch.Generator().manual_seed(5)
        x = torch.randn(2 if self.quick else 4, 3, cfg.img_size, cfg.img_size, generator=g)
        return convert_for_inference(model), x

    def _captured_inputs(self, packed: nn.Module, x: Tensor) -> tuple[nn.Module, dict[str, Tensor]]:
        """The float64 model and the input of each of its exported layers on ``x``."""
        ref = copy.deepcopy(packed).double()
        inputs: dict[str, Tensor] = {}
        hooks = [m.register_forward_pre_hook(lambda mod, args, name=name: inputs.__setitem__(name, args[0].detach()))
                 for name, m in ref.named_modules() if isinstance(m, BitLinearInference)]
        with torch.no_grad():
            ref(x.double())
        for h in hooks:
            h.remove()
        return ref, inputs

    def packed_layers(self) -> dict:
        need_device(self.device)
        packed, x = self._packed_model()
        ref, inputs = self._captured_inputs(packed, x)
        dev_model = copy.deepcopy(packed).to(self.device)
        entries, shares = [], []
        with torch.no_grad():
            for name, x64 in inputs.items():
                x32 = x64.float()
                layer, dev_layer = packed.get_submodule(name), dev_model.get_submodule(name)
                xd = x32.to(self.device)
                dev = (dev_layer(xd), *inference_codes(dev_layer, xd))
                cpu = (layer(x32), *inference_codes(layer, x32))
                e, share = inference_entries(ref.get_submodule(name), x32, dev, cpu, prefix=f"{name} ")
                entries += e
                shares.append(share)
        return verdict(entries, note=f"{len(inputs)} layers, up to {max(shares):.3%} of codes differ")

    @staticmethod
    def _model_run(model: nn.Module) -> Callable[..., dict]:
        def run(device: str, dtype: torch.dtype, x: Tensor) -> dict:
            m = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
            with torch.no_grad():
                return on_cpu({"logits": m(to(x, device, dtype))})

        return run

    def model_logits(self, fused: bool = False, lora: bool = False) -> dict:
        need_device(self.device)
        packed, x = self._packed_model(lora)
        run = self._model_run(packed)
        device_run = None
        if fused:
            fused_model = convert_to_fused(packed)

            def device_run(device: str, dtype: torch.dtype, x: Tensor) -> dict:
                with torch.inference_mode():
                    return on_cpu({"logits": copy.deepcopy(fused_model).to(device)(x.to(device))})

        dev, ref, cpu = run_three(run, (x,), self.device, nudged=True, device_run=device_run)
        agree = (dev["logits"].argmax(-1) == ref["logits"].argmax(-1)).float().mean().item()
        result = verdict(compare(dev, ref, cpu, 1e-3), note=f"argmax agrees on {agree:.0%} of {len(x)} images")
        result["argmax_agreement"] = agree
        return result

    def _autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.device == "mps" else torch.bfloat16  # CPU autocast is bfloat16

    def autocast_layers(self) -> dict:
        need_device(self.device)
        packed, x = self._packed_model()
        _, inputs = self._captured_inputs(packed, x)
        model = copy.deepcopy(packed).to(self.device)
        dtype = self._autocast_dtype()
        worst, worst_name = 0.0, ""
        with torch.no_grad():
            for name, x64 in inputs.items():
                layer, xd = model.get_submodule(name), x64.float().to(self.device)
                y = layer(xd)
                with torch.autocast(device_type=self.device, dtype=dtype):
                    y_ac = layer(xd)
                d = (y_ac.float() - y).abs().max().item() if y_ac.dtype == y.dtype else math.inf
                if not d <= worst:
                    worst, worst_name = d, name
            # Largest possible sum: codes 127 everywhere times weights +1, 127 * K (195,072 for K = 1536) > 65,504.
            k = 256 if self.quick else 1536
            layer = BitLinear(k, 64, bias=False)
            with torch.no_grad():
                layer.weight.fill_(0.01)
            layer = convert_for_inference(layer).to(self.device)
            ones = torch.ones(4, k, device=self.device)
            y = layer(ones)
            with torch.autocast(device_type=self.device, dtype=dtype):
                y_ac = layer(ones)
        # 127 K / 127 * ws; .item() first, since MPS has no float64
        expected = torch.full(y.shape, k * layer.weight_scale.item(), dtype=torch.float64)
        change = (y_ac.float() - y).abs().max().item()
        entries = [entry(f"largest change of a layer under autocast ({worst_name or 'none'})", worst, 0.0),
                   entry(f"worst-case sum 127 x {k}: change under autocast", change, 0.0),
                   entry(f"worst-case sum 127 x {k}: error", rel_err(y_ac.float(), expected), 1e-6)]
        return verdict(entries, note=f"torch.autocast({self.device!r}, {str(dtype).replace('torch.', '')}), "
                                     f"{len(inputs)} layers")

    def autocast_model(self) -> dict:
        need_device(self.device)
        packed, x = self._packed_model()
        model = copy.deepcopy(packed).to(self.device)
        xd = x.to(self.device)
        with torch.no_grad():
            y = model(xd)
            with torch.autocast(device_type=self.device, dtype=self._autocast_dtype()):
                y_ac = model(xd)
        finite = bool(torch.isfinite(y_ac).all())
        agree = (y_ac.argmax(-1) == y.argmax(-1)).float().mean().item()
        err = rel_err(y_ac, y)
        # Autocast also lowers the patch embedding, attention and head, so only finiteness is required here.
        return {"ok": finite, "message": f"logits {'finite' if finite else 'NOT FINITE'}, {_fmt(err)} from float32, "
                                         f"argmax agrees on {agree:.0%} (information only)",
                "rel_err_vs_float32": err, "argmax_agreement": agree}

    # 6 -------------------------------------------------------------------------------------------------------
    def metal_compile(self) -> dict:
        if self.device != "mps":
            raise Skip("dry run on the CPU: the Metal kernels need an Apple GPU")
        need_device(self.device)
        if not hasattr(torch.mps, "compile_shader"):
            raise Skip(f"torch {torch.__version__} has no torch.mps.compile_shader (needs torch >= 2.7)")
        info = {}
        for name in ("bitnet_front_end_float", "bitnet_front_end_half", "bitnet_epilogue"):
            kernel = mps_kernels._kernel(name)
            info[name] = {attr: int(getattr(kernel, attr)) for attr in
                          ("max_threads_per_threadgroup", "thread_execution_width") if hasattr(kernel, attr)}
        widths = {v.get("thread_execution_width") for v in info.values()}
        limits = {v.get("max_threads_per_threadgroup") for v in info.values()}
        return {"message": f"compiled 3 kernels; simdgroup width {widths}, up to {limits} threads per threadgroup",
                "kernels": info}

    def _fused_path(self) -> str:
        return "Metal kernels" if self.device == "mps" and metal_available() else "reference ops"

    @staticmethod
    def _front_end(layer: BitLinearInference, x: Tensor) -> tuple[Tensor, Tensor]:
        """The front end exactly as FusedBitLinearInference.forward runs it: (codes, scale)."""
        norm = layer.norm
        if layer.hadamard is None and type(norm) is RMSNorm and norm.weight is not None:
            return fused_front_end(x, norm.weight, norm.eps)
        h = x if norm is None else norm(x)
        if layer.hadamard is not None:
            h = layer.hadamard(h)
        return fused_front_end(h, None, 0.0)

    def fused(self, k: int, n: int, lead: tuple[int, ...], dtype: torch.dtype = torch.float32,
              cls: type[BitLinear] = BitLinear, x: Tensor | None = None,
              init: Callable[[BitLinear], None] | None = None, **kwargs: Any) -> dict:
        need_device(self.device)
        torch.manual_seed(k + n)
        layer = init_layer(cls(k, n, **kwargs), seed=k + n)
        if init is not None:
            init(layer)
        inf32 = convert_for_inference(layer, int_matmul="never")
        inf64 = copy.deepcopy(inf32).double()
        cpu_fused, dev_fused = convert_to_fused(inf32), convert_to_fused(inf32).to(self.device)
        x = (tokens((*lead, k), seed=k) if x is None else x).to(dtype)
        with torch.inference_mode():
            xd = x.to(self.device)
            with count_fallbacks() as fallbacks:
                y = dev_fused(xd)
                codes, scale = self._front_end(dev_fused, xd)
            y_cpu = inf32(x)
            codes_cpu, scale_cpu = self._front_end(cpu_fused, x)
        entries = [entry("output shape or dtype differs from BitLinearInference",
                         float(y.shape != y_cpu.shape or y.dtype != y_cpu.dtype), 0, count=True)]
        note = self._fused_path()
        if x.numel():
            # float16 output: one more rounding, 2^-11 relative
            e, share = inference_entries(inf64, x, (y, codes, scale), (y_cpu, codes_cpu, scale_cpu), in_dtype=dtype,
                                         floor=2e-3 if dtype == torch.float16 else 1e-5)
            entries += e + [entry("non-finite outputs", (~torch.isfinite(y.float())).sum().item(), 0, count=True)]
            if note == "Metal kernels":
                entries.append(entry("calls that fell back to the reference ops", fallbacks["n"], 0, count=True))
            note += f", {share:.3%} of codes differ"
        return verdict(entries, note=note)

    def fused_edge_init(self, what: str) -> Callable[[BitLinear], None]:
        def init(layer: BitLinear) -> None:
            with torch.no_grad():
                if what == "worst":  # every weight code +1 and every activation code 127: the largest sums
                    layer.weight.fill_(0.01)
                    if layer.norm is not None:
                        layer.norm.weight.fill_(1.0)

        return init

    # 7 -------------------------------------------------------------------------------------------------------
    def mlx_about(self) -> dict:
        mx, dev = mlx_device(self.device)
        info = mlx_info() or {}
        arch = (info.get("device_info") or {}).get("architecture")
        return {"message": f"MLX {info.get('version')} on {dev}" + (f" ({arch})" if arch else ""), "mlx": info}

    def mlx_layer(self, k: int, n: int, m: int) -> dict:
        mx, dev = mlx_device(self.device)
        inf32 = exported(BitLinear, k, n, seed=k + n)
        inf64 = copy.deepcopy(inf32).double()
        x = tokens((2, m, k), seed=k)
        group = 64 if k % 64 == 0 else 32
        with mx.stream(dev):
            packed = mt.MLXBitLinear.from_inference(inf32, group_size=group)
            dense = mt.MLXBitLinear.from_inference(inf32, dense=True)
            xm = mt.to_mx(x)
            codes_m, scale_m = packed.quantize_input(xm)
            y, codes, scale = mt.to_torch(packed(xm)), mt.to_torch(codes_m), mt.to_torch(scale_m)
            y_dense = mt.to_torch(dense(xm))
            y_compiled = mt.to_torch(mt.compiled(packed)(xm))
        with torch.no_grad():
            cpu = (inf32(x), *inference_codes(inf32, x))
        entries, share = inference_entries(inf64, x, (y, codes, scale), cpu)
        entries += [entry("dense against packed weights (largest difference)", (y_dense - y).abs().max().item(), 0.0),
                    entry("mx.compile against eager", rel_err(y_compiled, y), 1e-6)]
        return verdict(entries, note=f"{share:.3%} of codes differ")

    def mlx_quantize_trap(self) -> dict:
        mx, dev = mlx_device(self.device)
        g = torch.Generator().manual_seed(0)
        w = torch.randint(-1, 2, (64, 128), generator=g).float() * 0.0372
        with mx.stream(dev):
            wq, scales, biases = mx.quantize(mt.to_mx(w), group_size=64, bits=2)
            back = mt.to_torch(mx.dequantize(wq, scales, biases, group_size=64, bits=2))
        err = (back - w).abs().max().item()
        if err == 0:
            return {"message": "mx.quantize(bits=2) is now exact for ternary weights (the hand packing in "
                               "mlx_ternary is still exact too)", "max_error": err}
        return {"message": f"mx.quantize(bits=2) is not exact for ternary weights (max error {err:.3g} for g = "
                           "0.0372, -g becomes -g/2): keep the hand packing in mlx_ternary", "max_error": err}

    def mlx_half(self) -> dict:
        mx, dev = mlx_device(self.device)
        _, k, n, m = linear_shapes(self.quick)[6]  # S/2 fc1
        inf32 = exported(BitLinear, k, n, seed=1)
        x = tokens((2, m, k), seed=2)
        worst_k = 256 if self.quick else 1536
        worst = BitLinear(worst_k, 64, bias=False)
        with torch.no_grad():
            worst.weight.fill_(0.01)
        with mx.stream(dev):
            y = mt.to_torch(mt.MLXBitLinear.from_inference(inf32, dtype="float16")(mt.to_mx(x, "float16")))
            y_worst = mt.to_torch(mt.MLXBitLinear.from_inference(worst, dtype="float16")(
                mt.to_mx(torch.ones(4, worst_k), "float16")))
        with torch.no_grad():
            ref = copy.deepcopy(inf32).double()(x.double())
        expected = torch.full((4, 64), worst_k * convert_for_inference(worst).weight_scale.item())
        # Metal's quantized matmul accumulates in float32; MLX's CPU backend accumulates float16 in float16.
        gpu = self.device == "mps"
        entries = [entry("error against float64", rel_err(y.float(), ref), 1e-2 if gpu else 5e-2),
                   entry(f"worst-case sum 127 x {worst_k}: error", rel_err(y_worst.float(), expected),
                         2e-3 if gpu else 5e-2)]
        note = "float16 on the GPU, about 1e-3 expected" if gpu else "float16 on MLX's CPU backend (loose limits)"
        return verdict(entries, note=note)

    def mlx_block(self, size: str, dtype: str = "float32") -> dict:
        mx, dev = mlx_device(self.device)
        torch.manual_seed(0)
        model = build_model(size, self.quick).eval()
        block = convert_for_inference(model.blocks[0])
        d, n = model.config.embed_dim, model.config.num_patches + 1
        x = tokens((2, n, d), seed=6)
        group = 64 if d % 64 == 0 else 32

        def device_run(device: str, dt: torch.dtype, x: Tensor) -> dict:
            with mx.stream(dev):
                mb = mt.mlx_block_from_vit_block(block, group_size=group, dtype=dtype)
                xm = mt.to_mx(x, dtype)
                out = {"out": mt.to_torch(mb(xm)).float()}
                if dtype == "float32":
                    out["compiled"] = mt.to_torch(mt.compiled(mb)(xm))
            return out

        def run(device: str, dt: torch.dtype, x: Tensor) -> dict:
            with torch.no_grad():
                return on_cpu({"out": copy.deepcopy(block).to(dt)(x.to(dt))})

        dev_out, ref, cpu = run_three(run, (x,), "mlx", device_run=device_run)
        y_ref = ref["out"]
        scale = y_ref.abs().max().item()
        token_err = (dev_out["out"].double() - y_ref).abs().amax(dim=-1) / scale
        close = (token_err <= 1e-4).float().mean().item()
        cpu_close = ((cpu[0]["out"].double() - y_ref).abs().amax(dim=-1) / scale <= 1e-4).float().mean().item()
        y = dev_out["out"].double()
        if dtype != "float32":
            cos = F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0).item()
            return verdict([entry("1 - cosine similarity to float64", 1 - cos, 1e-3),
                            entry("largest token error", token_err.max().item(), 5e-2)],
                           note=f"{dtype} smoke test (information only)")
        # One block: a code flipped at a rounding tie moves its token by about 1e-3 of the largest output, and a flip in
        # qkv reaches other tokens through attention, so most tokens (not all) must be within float32 rounding.
        entries = [entry("share of tokens off by more than 1e-4", 1 - close, 0.25),
                   entry("largest token error", token_err.max().item(), 1e-2),
                   entry("mx.compile against eager", rel_err(dev_out["compiled"], dev_out["out"]), 1e-5)]
        return verdict(entries, note=f"{n} tokens x 2, {close:.0%} within 1e-4 (CPU float32 {cpu_close:.0%})")

    # 8 -------------------------------------------------------------------------------------------------------
    def train_step(self, quantize: bool, large: bool = False, lora: bool = False) -> dict:
        need_device(self.device)
        cfg = model_config("T/4", self.quick)
        if large:
            cfg["depth"] = min(cfg["depth"], 2)
        torch.manual_seed(0)
        master = clear_of_weight_ties(create_model("bitvit_cifar_tiny", quantize=quantize, **cfg))
        if lora:  # adapters only, on a frozen base
            add_lora_for_check(master)
            mark_only_lora_trainable(master)
        tokens_per_image = master.config.num_patches + 1
        batch = -(-32769 // tokens_per_image) if large else (2 if self.quick else 8)
        g = torch.Generator().manual_seed(7)
        x = torch.randn(batch, 3, cfg["img_size"], cfg["img_size"], generator=g)
        labels = torch.randint(0, cfg["num_classes"], (batch,), generator=g)
        lr = 1e-3

        def run(device: str, dtype: torch.dtype, x: Tensor, labels: Tensor) -> dict:
            model = copy.deepcopy(master).to(device=device, dtype=dtype).train()
            params = {name: p for name, p in model.named_parameters() if p.requires_grad}
            before = {name: p.detach().clone() for name, p in params.items()}
            opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=0.05)
            logits = model(to(x, device, dtype))
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            out = {"loss": loss.detach().reshape(1), "logits": logits.detach(),
                   **{f"grad {name}": p.grad for name, p in params.items()}}
            opt.step()
            out.update({f"update {name}": p.detach() - before[name] for name, p in params.items()})
            return on_cpu(out)

        dev, ref, cpu = run_three(run, (x, labels), self.device, nudged=quantize)
        keys = [k for k in ref if not k.startswith("update ")]
        # Quantized models are chaotic (see the module docstring); the full-precision twin is plain float math.
        result = verdict(compare(dev, ref, cpu, 1e-3 if quantize else 1e-5, keys=keys),
                         note=f"batch {batch}, {batch * tokens_per_image:,} token rows")
        # AdamW's first step is about lr * sign(gradient), so an update can differ by 2 lr where a gradient is near 0.
        changed = sum(((dev[k] - ref[k]).abs() > 0.1 * lr).sum().item() for k in ref if k.startswith("update "))
        total = sum(ref[k].numel() for k in ref if k.startswith("update "))
        result["adamw_updates_differing"] = changed / total
        result["message"] += f"; AdamW updates differ by > 0.1 lr in {changed / total:.3%} of weights (information)"
        return result


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps" if sys.platform == "darwin" else "cpu",
                   help="mps (default on a Mac) or cpu (dry run)")
    p.add_argument("--quick", action="store_true", help="tiny shapes (for dry runs)")
    p.add_argument("--only", default=None, help="comma-separated item numbers, e.g. 2,6")
    p.add_argument("--out", default=None,
                   help="JSON file or folder (default: apple_results/apple_check-<time>.json)")
    a = p.parse_args(argv)
    only = None if a.only is None else {s.strip() for s in a.only.split(",") if s.strip()}
    if only is not None and not only <= {str(i) for i in range(1, 9)}:
        p.error(f"--only takes item numbers 1 to 8, got {a.only!r}")

    report = Report("apple_check", a, check_line)
    c = Checks(report, a.device, a.quick)
    print(f"bitnet_vision Apple check, device {a.device}{' (quick)' if a.quick else ''}: {describe(report.env)}")
    warning = torch_warning(report.env, a.device)
    if warning:
        print(banner(warning))
    for note in environment_notes(report.env):
        print(f"note: {note}")
    print()

    q = a.quick
    shapes = linear_shapes(q)
    batch = {"T/4": 4, "S/2": 2} if not q else {"T/4": 2, "S/2": 2}
    items: list[tuple[str, str, Callable[[], dict]]] = [
        ("1.1", "torch version", c.torch_version),
        ("1.2", "device", c.device_info),
    ]
    for i, (name, k, n, m) in enumerate(s for s in shapes if not q or s[0].endswith(("qkv", "fc2"))):
        rows = m * batch[name[:3]]
        items.append((f"2.{i + 1}", f"BitLinear {name} {k}->{n}, {rows} tokens: fwd+bwd",
                      lambda k=k, n=n, rows=rows: c.bitlinear(BitLinear, k, n, rows)))
    offset = len(items) - 2
    rotated = [s for s in shapes if s[0].endswith(("proj", "fc2")) and (not q or "fc2" in s[0])]
    for i, (name, k, n, m) in enumerate(rotated):
        rows = m * batch[name[:3]]
        items.append((f"2.{offset + i + 1}", f"HBitLinear {name} {k}->{n}, {rows} tokens: fwd+bwd",
                      lambda k=k, n=n, rows=rows: c.bitlinear(HBitLinear, k, n, rows)))
    items += [
        ("3.1", "linear weight grad over > 32768 rows (2D input)", lambda: c.large_reduction("2d")),
        ("3.2", "linear weight grad over > 32768 rows (3D input)", lambda: c.large_reduction("3d")),
        ("3.3", "bmm reducing over > 32768", lambda: c.large_reduction("bmm")),
        ("4.1", "jvp: explicit attention block, full precision", lambda: c.jvp_explicit(quantize=False)),
        ("4.2", "jvp: explicit attention block, ternary", lambda: c.jvp_explicit(quantize=True)),
        ("4.3", "iMF-style step: jvp, then backward into weights", c.imf_step),
        ("4.4", "F.scaled_dot_product_attention under torch.func.jvp", lambda: c.sdpa_forward_ad("func.jvp")),
        ("4.5", "F.scaled_dot_product_attention, forward_ad + grad", lambda: c.sdpa_forward_ad("forward_ad")),
        ("5.1", "packed T/4 model: each layer on its own input", c.packed_layers),
        ("5.2", "packed T/4 model: logits", c.model_logits),
        ("5.3", "packed layers under autocast: unchanged", c.autocast_layers),
        ("5.4", "packed T/4 model under autocast", c.autocast_model),
        ("5.5", "packed T/4 model with LoRA adapters: logits", lambda: c.model_logits(lora=True)),
        ("6.1", "Metal kernels compile", c.metal_compile),
    ]
    for i, (name, k, n, m) in enumerate(shapes):
        items.append((f"6.{i + 2}", f"fused {name} {k}->{n}, 2x{m} tokens",
                      lambda k=k, n=n, m=m: c.fused(k, n, (2, m))))
    d, big = (128, 512) if q else (384, 1536)
    special = tokens((3, 74, d), seed=9)
    special[0, 0] = 0
    special[0, 1] *= 1e-6
    special[1, 0, 7] = 1e4
    edge = [
        ("zero, tiny and huge tokens", lambda: c.fused(d, d, (3, 74), x=special)),
        (f"worst-case sums 127 x {big}", lambda: c.fused(big, 64, (2, 4), x=torch.ones(2, 4, big),
                                                         init=c.fused_edge_init("worst"))),
        ("per-channel scales, no bias", lambda: c.fused(d, 3 * d, (2, 74), weight_granularity="channel", bias=False)),
        ("no input norm", lambda: c.fused(d, d, (2, 74), input_norm=False)),
        ("HBitLinear (Hadamard rotation)", lambda: c.fused(d, d, (2, 74), cls=HBitLinear)),
        ("float16 input", lambda: c.fused(d, 4 * d, (2, 74), dtype=torch.float16)),
        (f"K={512 if q else 4096}, one token", lambda: c.fused(512 if q else 4096, 64, (1,))),
        ("1000 tokens", lambda: c.fused(2 * d, d, (1000,))),
        ("empty batch", lambda: c.fused(d, d, (0,))),
    ]
    for i, (name, fn) in enumerate(edge):
        items.append((f"6.{len(shapes) + i + 2}", f"fused edge case: {name}", fn))
    items.append((f"6.{len(shapes) + len(edge) + 2}", "fused T/4 model (Hadamard): logits",
                  lambda: c.model_logits(fused=True)))
    items.append((f"6.{len(shapes) + len(edge) + 3}", "fused T/4 model with LoRA adapters: logits",
                  lambda: c.model_logits(fused=True, lora=True)))
    items.append(("7.1", "MLX", c.mlx_about))
    for i, (name, k, n, m) in enumerate(shapes):
        items.append((f"7.{i + 2}", f"MLXBitLinear {name} {k}->{n}, 2x{m} tokens",
                      lambda k=k, n=n, m=m: c.mlx_layer(k, n, m)))
    j = len(shapes) + 2
    items += [
        (f"7.{j}", "mx.quantize(bits=2) on ternary weights", c.mlx_quantize_trap),
        (f"7.{j + 1}", "MLXBitLinear float16", c.mlx_half),
        (f"7.{j + 2}", "MLX block T/4", lambda: c.mlx_block("T/4")),
        (f"7.{j + 3}", "MLX block S/2", lambda: c.mlx_block("S/2")),
        (f"7.{j + 4}", "MLX block T/4 float16", lambda: c.mlx_block("T/4", "float16")),
        ("8.1", "training step, ternary T/4", lambda: c.train_step(quantize=True)),
        ("8.2", "training step, full-precision T/4", lambda: c.train_step(quantize=False)),
        ("8.3", "training step, > 32768 token rows", lambda: c.train_step(quantize=False, large=True)),
        ("8.4", "training step, LoRA adapters on a frozen ternary T/4", lambda: c.train_step(quantize=True, lora=True)),
    ]
    try:
        for row_id, name, fn in items:
            if only is None or row_id.split(".")[0] in only:
                report.run(row_id, name, fn)
    except KeyboardInterrupt:
        report.data["interrupted"] = True
        print("\ninterrupted")
    report.finish()
    failed = [r for r in report.rows if r["status"] == "FAILED"]
    if failed:
        print("FAILED: " + ", ".join(f"{r['id']} {r['name']}" for r in failed))
    if warning:
        print(banner(warning))
    return report.data


def check_line(row: dict) -> str:
    return f"{row['id']:<6}{row['status']:<9}{row['name'][:56]:<56} {_first_line(row.get('message', ''))}"


if __name__ == "__main__":
    sys.exit(1 if main()["summary"]["FAILED"] else 0)
