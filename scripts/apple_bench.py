#!/usr/bin/env python
"""Speed of bitnet_vision on an Apple GPU: single layers, whole models and training steps (PyTorch MPS, Metal, MLX).

  python scripts/apple_bench.py                       # on the Mac
  python scripts/apple_bench.py --only A --repeats 20
  python scripts/apple_bench.py --device cpu --quick  # dry run anywhere

Protocol (as scripts/bench.py): all variants of one shape are built, warmed up, and then timed in turn, interleaved
over --repeats rounds, so a change in machine load or temperature hits all of them. A timed block runs a variant
often enough to take about 30 ms and ends with torch.mps.synchronize() or mx.synchronize(), so queued GPU work is
counted; the median and 10th/90th percentiles of the time per call are reported. Variants slower than 1 s per call
get at most 3 rounds. The power source, Low Power Mode, load average and versions are recorded; compare numbers
only from a quiet machine on AC power.

A  Layers: every ternary layer of T/4 (width 192, MLP 768) and S/2 (384, 1536) at M = 74, 266, 4736 and 17024
   tokens (one T/4 image, one S/2 image, 64 of each). Time, TFLOPS (2 M K N / t, also for the ternary variants) and
   speed relative to float32 F.linear for: F.linear float32 and float16 (the roofline), BitLinear's training-mode
   forward (fake quantization, the unfused cost), BitLinearInference (the packed export; a float32 matmul on MPS),
   FusedBitLinearInference (the Metal kernels), torch._weight_int8pack_mm on int8 ternary codes (matmul only), MLX
   float32 and float16 matmul, and MLXBitLinear with packed 2-bit weights (eager, mx.compile, mx.compile in float16).
   Ternary variants also report their error against BitLinearInference on the CPU.
B  Models: the T/4-like and S/2-like BitViT (create_model overrides; 32x32 images, 10 classes) at batch 1, 8, 64 and
   256: images per second for the full-precision twin (float32, float16), the ternary model in training mode,
   packed (convert_for_inference), fused (convert_to_fused), and the MLX block stack (blocks only: the patch
   embedding and head, about 1% of the work, are left out).
C  Training at batch 128 in float32: forward, backward and AdamW for the full-precision twin (TernFlow stages 1 and 2)
   and the ternary model, with F.scaled_dot_product_attention and with explicit attention, plus an iMF-style step:
   torch.func.jvp of the model along a random input tangent, a loss on the output against a target built from the
   stop-gradient tangent, backward, AdamW. The jvp needs explicit attention, because F.scaled_dot_product_attention has
   no forward-mode AD on MPS: apple_check.use_explicit_attention binds a softmax(q k^T / sqrt(d)) v forward to each
   Attention module of a copy of the model. S/2 is also timed at batch 127, just under 32768 token rows, where
   PyTorch 2.14 sends the weight-gradient matmuls of M1/M2 GPUs to different kernels.

BitViT has 65 (T/4) and 257 (S/2) tokens per image where TernFlow has 74 and 266. Results print as they come and go
to apple_results/apple_bench-<time>.json, with summary tables at the end.
"""

from __future__ import annotations

import argparse
import copy
import functools
import math
import os
import statistics
import sys
import textwrap
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apple_check import (  # noqa: E402
    Report,
    Skip,
    banner,
    build_model,
    describe,
    environment_notes,
    free_memory,
    linear_shapes,
    mlx_device,
    need_device,
    rel_err,
    sync,
    torch_warning,
    use_explicit_attention,
)

from bitnet_vision.apple import mlx_ternary as mt  # noqa: E402
from bitnet_vision.apple.mps_kernels import convert_to_fused, metal_available  # noqa: E402
from bitnet_vision.export import convert_for_inference, unpack_ternary  # noqa: E402
from bitnet_vision.layers import BitLinear  # noqa: E402
from bitnet_vision.quant import quantize_activations  # noqa: E402

TARGET_S = 0.03  # length of one timed block
WARMUP = 2
MAX_INNER = 2000
SLOW_S, SLOW_REPEATS = 1.0, 3
TOKENS = (74, 266, 4736, 17024)
QUICK_TOKENS = (8, 33)
BATCHES = (1, 8, 64, 256)
QUICK_BATCHES = (2,)
TRAIN_BATCH, QUICK_TRAIN_BATCH = 128, 2
MAX_LAYER_BYTES = 8 * 2**30  # layer cases estimated above this are skipped


class Timed:
    """What to time: ``fn`` launches one call, ``sync`` waits for the device, ``output`` (optional) returns a CPU
    tensor for the correctness check after timing, ``info`` goes into the row."""

    def __init__(self, fn: Callable[[], Any], sync: Callable[[], None], output: Callable[[], Tensor] | None = None,
                 info: dict[str, Any] | None = None) -> None:
        self.fn, self.sync, self.output, self.info = fn, sync, output, info or {}


def _sig(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def _percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def _block(t: Timed, inner: int) -> float:
    t.sync()
    start = time.perf_counter()
    for _ in range(inner):
        t.fn()
    t.sync()
    return (time.perf_counter() - start) / inner


def time_group(builders: list[tuple[str, Callable[[], Timed]]], repeats: int, quick: bool) -> dict[str, Any]:
    """Build, warm up and time every variant, interleaved over the rounds. Returns, per variant name, either a dict
    of timings (with the Timed under "_timed") or the exception it raised (Skip included)."""
    results: dict[str, Any] = {}
    state: dict[str, tuple[Timed, int, int, list[float]]] = {}
    for name, build in builders:
        try:
            t = build()
            for _ in range(1 if quick else WARMUP):
                t.fn()
            t.sync()
            first = _block(t, 1)
            inner = 1 if quick or first >= TARGET_S else min(MAX_INNER, math.ceil(TARGET_S / max(first, 1e-7)))
            state[name] = (t, inner, repeats if first < SLOW_S else min(repeats, SLOW_REPEATS), [])
        except Exception as e:  # noqa: BLE001 - recorded as the variant's FAILED (or skipped) row
            results[name] = e
    for r in range(max((s[2] for s in state.values()), default=0)):
        for name in list(state):
            t, inner, reps, times = state[name]
            if r < reps:
                try:
                    times.append(_block(t, inner))
                except Exception as e:  # noqa: BLE001
                    results[name] = e
                    del state[name]
    for name, (t, inner, reps, times) in state.items():
        results[name] = {"ms": _sig(statistics.median(times) * 1e3),
                         "p10_ms": _sig(_percentile(times, 0.1) * 1e3),
                         "p90_ms": _sig(_percentile(times, 0.9) * 1e3),
                         "calls_per_block": inner, "rounds": reps, **t.info, "_timed": t}
    return {name: results[name] for name, _ in builders}


def recorded(result: Any, derived: Callable[[dict], dict],
             check: Callable[[dict, Timed], dict] | None = None) -> Callable[[], dict]:
    """A row function for Report.run: re-raises a variant's exception, or returns its timings with ``derived``'s and
    ``check``'s fields."""

    def fn() -> dict:
        if isinstance(result, BaseException):
            raise result
        out = {k: v for k, v in result.items() if k != "_timed"}
        out.update(derived(out))
        if check is not None:
            with torch.inference_mode():  # the variant was built under inference mode
                out.update(check(out, result["_timed"]))
        return out

    return fn


def torch_sync(device: str) -> Callable[[], None]:
    return functools.partial(sync, device)


def mlx_timed(mx: Any, f: Callable[..., Any], *args: Any, info: dict | None = None) -> Timed:
    """Time an MLX function: each call is queued with mx.async_eval, the block waits with mx.synchronize."""
    if hasattr(mx, "async_eval") and hasattr(mx, "synchronize"):
        return Timed(lambda: mx.async_eval(f(*args)), mx.synchronize, lambda: mt.to_torch(f(*args)).float(), info or {})
    return Timed(lambda: mx.eval(f(*args)), lambda: None, lambda: mt.to_torch(f(*args)).float(), info or {})


def use_mlx(device: str) -> Any:
    mx, dev = mlx_device(device)
    mx.set_default_device(dev)
    return mx


def need_metal(device: str) -> str:
    """The path FusedBitLinearInference takes; Skip on MPS without the Metal kernels, which are what is timed."""
    need_device(device)
    if device == "mps" and not metal_available():
        raise Skip(f"torch {torch.__version__} has no torch.mps.compile_shader (needs >= 2.7)")
    return "Metal kernels" if device == "mps" else "reference ops (no Metal on this device)"


class Bench:
    def __init__(self, report: Report, device: str, quick: bool, repeats: int) -> None:
        self.report, self.device, self.quick, self.repeats = report, device, quick, repeats
        self.tables: dict[str, list[dict]] = {"A": [], "B": [], "C": []}

    def record(self, section: str, header: str, results: dict[str, Any], fields: dict[str, Any],
               check: Callable[[str], Callable[[dict, Timed], dict] | None], derived: Callable[[dict], dict]) -> None:
        print(f"\n{section}  {header}")
        for name, result in results.items():
            row_id = f"{section}{len(self.tables[section]) + 1}"
            row = self.report.run(row_id, name, recorded(result, derived, check(name)), section=section,
                                  variant=name, **fields)
            self.tables[section].append(row)

    def case(self, section: str, label: str, fn: Callable[[], None]) -> None:
        """Run one case; an error outside its rows becomes a FAILED row instead of ending the run."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            err = e

            def failed() -> dict:
                raise err

            row = self.report.run(f"{section}{len(self.tables[section]) + 1}", label, failed, section=section)
            self.tables[section].append(row)

    # A -------------------------------------------------------------------------------------------------------
    def layers(self) -> None:
        shapes = linear_shapes(self.quick)
        for layer, k, n, _ in [shapes[0], shapes[-1]] if self.quick else shapes:  # quick: T/4 qkv and S/2 fc2
            for m in QUICK_TOKENS if self.quick else TOKENS:
                self.case("A", f"{layer} M={m}", functools.partial(self.layer_case, layer, k, n, m))

    def layer_case(self, layer: str, k: int, n: int, m: int) -> None:
        dev = self.device
        g = torch.Generator().manual_seed(k * n + m)
        x = torch.randn(m, k, generator=g) * torch.exp(torch.randn(m, 1, generator=g))
        torch.manual_seed(0)
        train = BitLinear(k, n)
        with torch.no_grad():
            train.bias.normal_(0, 0.5)
        inf = convert_for_inference(train)
        with torch.inference_mode():
            ref = inf(x)
        too_big = 4 * (8 * m * k + 6 * m * n + 4 * k * n) > MAX_LAYER_BYTES

        def torch_case(build: Callable[[], Timed]) -> Callable[[], Timed]:
            def wrapped() -> Timed:
                need_device(dev)
                if too_big:
                    raise Skip("estimated working set above 8 GB")
                return build()

            return wrapped

        def linear(dtype: torch.dtype) -> Timed:
            xd = x.to(dev, dtype)
            w = (torch.randn(n, k, generator=g) / k**0.5).to(dev, dtype)
            b = torch.zeros(n, device=dev, dtype=dtype)
            return Timed(lambda: F.linear(xd, w, b), torch_sync(dev))

        def module(mod: torch.nn.Module, info: dict | None = None) -> Timed:
            xd = x.to(dev)
            return Timed(lambda: mod(xd), torch_sync(dev), lambda: mod(xd).float().cpu(), info or {})

        def fused() -> Timed:
            path = need_metal(dev)
            return module(convert_to_fused(inf).to(dev), {"path": path})

        codes_w = unpack_ternary(inf.weight_packed, n, dtype=torch.int8)
        x_codes, x_scale = quantize_activations(inf.norm(x).detach())
        x_deq = x_codes / x_scale  # the activations BitLinearInference multiplies, already quantized
        int8_ref = F.linear(x_deq, codes_w.float() * inf.weight_scale)

        def int8pack(dtype: torch.dtype) -> Timed:
            op = getattr(torch, "_weight_int8pack_mm", None)
            if op is None:
                raise Skip(f"torch {torch.__version__} has no torch._weight_int8pack_mm")
            a, w = x_deq.to(dev, dtype).contiguous(), codes_w.to(dev).contiguous()
            s = inf.weight_scale.expand(n).to(dev, dtype).contiguous()
            return Timed(lambda: op(a, w, s), torch_sync(dev), lambda: op(a, w, s).float().cpu(),
                         {"note": "matmul only, on quantized activations"})

        def mlx_matmul(dtype: str) -> Timed:
            mx = use_mlx(dev)
            xm, wm = mt.to_mx(x, dtype), mt.to_mx(torch.randn(n, k, generator=g) / k**0.5, dtype)
            bm = mx.zeros((n,), dtype=getattr(mx, dtype))
            if hasattr(mx, "addmm"):
                return mlx_timed(mx, lambda: mx.addmm(bm, xm, wm.T))
            return mlx_timed(mx, lambda: xm @ wm.T + bm)

        def mlx_ternary(compile_: bool, dtype: str = "float32") -> Timed:
            mx = use_mlx(dev)
            group = 64 if k % 64 == 0 else 32
            layer_m = mt.MLXBitLinear.from_inference(inf, group_size=group, dtype=dtype)
            f = mt.compiled(layer_m) if compile_ else layer_m
            return mlx_timed(mx, f, mt.to_mx(x, dtype), info={"group_size": group})

        builders = [
            ("F.linear fp32", torch_case(lambda: linear(torch.float32))),
            ("F.linear fp16", torch_case(lambda: linear(torch.float16))),
            ("BitLinear training fwd", torch_case(lambda: module(copy.deepcopy(train).to(dev)))),
            ("BitLinearInference", torch_case(lambda: module(copy.deepcopy(inf).to(dev)))),
            ("Fused (Metal)", torch_case(fused)),
            ("int8pack_mm fp32", torch_case(lambda: int8pack(torch.float32))),
            ("int8pack_mm fp16", torch_case(lambda: int8pack(torch.float16))),
            ("MLX matmul fp32", lambda: mlx_matmul("float32")),
            ("MLX matmul fp16", lambda: mlx_matmul("float16")),
            ("MLX 2-bit eager", lambda: mlx_ternary(False)),
            ("MLX 2-bit compiled", lambda: mlx_ternary(True)),
            ("MLX 2-bit fp16 compiled", lambda: mlx_ternary(True, "float16")),
        ]
        with torch.inference_mode():
            results = time_group(builders, self.repeats, self.quick)
        flops = 2 * m * k * n
        base = results.get("F.linear fp32")
        base_ms = base["ms"] if isinstance(base, dict) else None

        def check(name: str) -> Callable[[dict, Timed], dict] | None:
            if name.startswith(("F.linear", "MLX matmul")):
                return None
            reference = int8_ref if name.startswith("int8pack") else ref
            limit = 5e-2  # fp16 variants: about 1e-3; float32 ones: 1e-3 where a code flips at a rounding tie

            def fn(row: dict, t: Timed) -> dict:
                err = rel_err(t.output(), reference)
                out = {"err_vs_cpu": err}
                if not err <= limit:
                    out.update(ok=False, message=f"wrong output: error {err:.3g} against BitLinearInference on the CPU")
                return out

            return fn

        def derived(row: dict) -> dict:
            out = {"tflops": _sig(flops / (row["ms"] * 1e-3) / 1e12)}
            if base_ms:
                out["vs_fp32_linear"] = _sig(base_ms / row["ms"], 3)
            return out

        self.record("A", f"{layer} {k}->{n}, M={m} tokens ({flops / 1e9:.3g} GFLOP)", results,
                    {"layer": layer, "k": k, "n": n, "m": m}, check, derived)
        del results, builders
        free_memory()

    # B -------------------------------------------------------------------------------------------------------
    def models(self) -> None:
        for size in ("T/4", "S/2"):
            for batch in QUICK_BATCHES if self.quick else BATCHES:
                self.case("B", f"{size} batch {batch}", functools.partial(self.model_case, size, batch))

    def model_case(self, size: str, batch: int) -> None:
        dev = self.device
        torch.manual_seed(0)
        ternary = build_model(size, self.quick).eval()
        cfg = ternary.config
        fp = build_model(size, self.quick, quantize=False).eval()
        packed = convert_for_inference(ternary)
        x = torch.randn(batch, 3, cfg.img_size, cfg.img_size, generator=torch.Generator().manual_seed(batch))
        tokens = cfg.num_patches + 1

        def torch_model(make: Callable[[], torch.nn.Module], dtype: torch.dtype = torch.float32,
                        info: dict | None = None) -> Callable[[], Timed]:
            def build() -> Timed:
                need_device(dev)
                model = make().to(dev, dtype)
                xd = x.to(dev, dtype)
                return Timed(lambda: model(xd), torch_sync(dev), lambda: model(xd).float().cpu(), info or {})

            return build

        def fused() -> Timed:
            return torch_model(lambda: convert_to_fused(packed), info={"path": need_metal(dev)})()

        def mlx_trunk(compile_: bool, dtype: str = "float32") -> Timed:
            mx = use_mlx(dev)
            group = 64 if cfg.embed_dim % 64 == 0 else 32
            blocks = [mt.mlx_block_from_vit_block(b, group_size=group, dtype=dtype) for b in packed.blocks]

            def trunk(h: Any) -> Any:
                for b in blocks:
                    h = b(h)
                return h

            h0 = mt.to_mx(torch.randn(batch, tokens, cfg.embed_dim, generator=torch.Generator().manual_seed(1)), dtype)
            return mlx_timed(mx, mt.compiled(trunk) if compile_ else trunk, h0, info={"group_size": group})

        builders = [
            ("full precision fp32", torch_model(lambda: copy.deepcopy(fp))),
            ("full precision fp16", torch_model(lambda: copy.deepcopy(fp), torch.float16)),
            ("ternary training mode", torch_model(lambda: copy.deepcopy(ternary))),
            ("packed", torch_model(lambda: copy.deepcopy(packed))),
            ("fused (Metal)", fused),
            ("MLX blocks 2-bit eager", lambda: mlx_trunk(False)),
            ("MLX blocks 2-bit compiled", lambda: mlx_trunk(True)),
            ("MLX blocks 2-bit fp16 compiled", lambda: mlx_trunk(True, "float16")),
        ]
        with torch.inference_mode():
            results = time_group(builders, self.repeats, self.quick)

        def check(name: str) -> Callable[[dict, Timed], dict]:
            def fn(row: dict, t: Timed) -> dict:
                finite = bool(torch.isfinite(t.output()).all())
                return {} if finite else {"ok": False, "message": "output is not finite"}

            return fn

        def derived(row: dict) -> dict:
            return {"images_per_s": _sig(batch / (row["ms"] * 1e-3))}

        self.record("B", f"{size}-like BitViT, batch {batch} ({tokens} tokens per image)", results,
                    {"model": size, "batch": batch}, check, derived)
        del results, builders
        free_memory()

    # C -------------------------------------------------------------------------------------------------------
    def training(self) -> None:
        batch = QUICK_TRAIN_BATCH if self.quick else TRAIN_BATCH
        for size in ("T/4", "S/2"):
            self.case("C", f"{size} training", functools.partial(self.train_case, size, batch, full=True))
        if not self.quick:
            small = 32768 // (build_model("S/2", False).config.num_patches + 1)
            self.case("C", "S/2 training", functools.partial(self.train_case, "S/2", small, full=False))

    def train_case(self, size: str, batch: int, full: bool) -> None:
        dev = self.device
        torch.manual_seed(0)
        cfg = build_model(size, self.quick).config
        g = torch.Generator().manual_seed(2)
        x = torch.randn(batch, 3, cfg.img_size, cfg.img_size, generator=g)
        labels = torch.randint(0, cfg.num_classes, (batch,), generator=g)
        target = torch.randn(batch, cfg.num_classes, generator=g)
        tangent = torch.randn(x.shape, generator=g)

        def step(quantize: bool, attention: str, kind: str) -> Callable[[], Timed]:
            def build() -> Timed:
                need_device(dev)
                torch.manual_seed(0)
                model = build_model(size, self.quick, quantize=quantize).to(dev).train()
                if attention == "explicit":
                    use_explicit_attention(model)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
                xd, yd, td, vd = x.to(dev), labels.to(dev), target.to(dev), tangent.to(dev)
                last: dict[str, Tensor] = {}

                def plain() -> None:
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(xd), yd)
                    loss.backward()
                    opt.step()
                    last["loss"] = loss.detach()

                def jvp_step() -> None:
                    opt.zero_grad(set_to_none=True)
                    out, du = jvp(model, (xd,), (vd,))
                    loss = (out - (td - du.detach())).pow(2).mean()  # MeanFlow-style target, stop-gradient tangent
                    loss.backward()
                    opt.step()
                    last["loss"] = loss.detach()

                info = {"attention": attention, "lr": 1e-4}
                return Timed(jvp_step if kind == "jvp" else plain, torch_sync(dev),
                             lambda: last["loss"].float().cpu().reshape(1), info)

            return build

        rows = batch * (cfg.num_patches + 1)
        if full:
            builders = [
                ("full precision, F.sdpa", step(False, "sdpa", "plain")),
                ("full precision, explicit attention", step(False, "explicit", "plain")),
                ("full precision, jvp step", step(False, "explicit", "jvp")),
                ("ternary, F.sdpa", step(True, "sdpa", "plain")),
                ("ternary, jvp step", step(True, "explicit", "jvp")),
            ]
            header = f"{size}-like BitViT, training at batch {batch} ({rows:,} token rows)"
        else:
            builders = [("full precision, F.sdpa", step(False, "sdpa", "plain"))]
            header = f"{size}-like BitViT, training at batch {batch} ({rows:,} token rows, just under 32768)"
        results = time_group(builders, self.repeats, self.quick)
        memory = None
        if dev == "mps" and torch.backends.mps.is_available() and hasattr(torch.mps, "driver_allocated_memory"):
            memory = round(torch.mps.driver_allocated_memory() / 2**30, 2)

        def check(name: str) -> Callable[[dict, Timed], dict]:
            def fn(row: dict, t: Timed) -> dict:
                loss = t.output().item()
                return {"loss": loss} if math.isfinite(loss) else {"ok": False, "message": f"loss became {loss}"}

            return fn

        def derived(row: dict) -> dict:
            ips = batch / (row["ms"] * 1e-3)
            return {"images_per_s": _sig(ips), "million_images_per_day": _sig(ips * 86400 / 1e6, 3)}

        self.record("C", header, results, {"model": size, "batch": batch, "token_rows": rows,
                                           "mps_driver_memory_gb_after_group": memory}, check, derived)
        del results, builders
        free_memory()

    # Summary tables -----------------------------------------------------------------------------------------
    def summary(self) -> None:
        a = [r for r in self.tables["A"] if "variant" in r]
        if a:
            print("\nA  speed relative to float32 F.linear on the same device (higher is faster); first column TFLOPS")
            variants = list(dict.fromkeys(r["variant"] for r in a))
            cases = list(dict.fromkeys((r["layer"], r["m"]) for r in a))
            cells = {(r["layer"], r["m"], r["variant"]): r for r in a}
            header = ["layer", "M", "fp32 TF"] + [_short(v) for v in variants]
            lines = []
            for layer, m in cases:
                base = cells.get((layer, m, "F.linear fp32"), {})
                line = [layer, str(m), f"{base['tflops']:.2f}" if "tflops" in base else "-"]
                for v in variants:
                    r = cells.get((layer, m, v), {})
                    line.append(f"{r['vs_fp32_linear']:.2f}" if "vs_fp32_linear" in r else _mark(r))
                lines.append(line)
            _table(header, lines)
            _legend(variants)
        for section, what in (("B", "images per second"), ("C", "training throughput")):
            rows = [r for r in self.tables[section] if "variant" in r]
            if not rows:
                continue
            print(f"\n{section}  {what}")
            if section == "B":
                variants = list(dict.fromkeys(r["variant"] for r in rows))
                cases = list(dict.fromkeys((r["model"], r["batch"]) for r in rows))
                cells = {(r["model"], r["batch"], r["variant"]): r for r in rows}
                lines = [[mdl, str(b)] + [f"{cells[(mdl, b, v)]['images_per_s']:.1f}"
                                          if "images_per_s" in cells.get((mdl, b, v), {}) else _mark(cells.get(
                                              (mdl, b, v), {})) for v in variants] for mdl, b in cases]
                _table(["model", "batch"] + [_short(v) for v in variants], lines)
                _legend(variants)
            else:
                lines = [[r["model"], str(r["batch"]), r["variant"],
                          f"{r['images_per_s']:.1f}" if "images_per_s" in r else _mark(r),
                          f"{r['ms']:.1f}" if "ms" in r else "-",
                          f"{r['million_images_per_day']:.2f}" if "million_images_per_day" in r else "-"]
                         for r in rows]
                _table(["model", "batch", "step", "img/s", "ms/step", "M img/day"], lines)


SHORT = {  # column names of the summary tables
    "F.linear fp32": "lin32", "F.linear fp16": "lin16", "BitLinear training fwd": "train",
    "BitLinearInference": "packed", "Fused (Metal)": "fused", "int8pack_mm fp32": "i8mm32",
    "int8pack_mm fp16": "i8mm16", "MLX matmul fp32": "mlx32", "MLX matmul fp16": "mlx16", "MLX 2-bit eager": "q2",
    "MLX 2-bit compiled": "q2c", "MLX 2-bit fp16 compiled": "q2c16", "full precision fp32": "fp32",
    "full precision fp16": "fp16", "ternary training mode": "train", "packed": "packed", "fused (Metal)": "fused",
    "MLX blocks 2-bit eager": "mlx", "MLX blocks 2-bit compiled": "mlxc", "MLX blocks 2-bit fp16 compiled": "mlxc16",
}


def _short(name: str) -> str:
    return SHORT.get(name, name)


def _mark(row: dict) -> str:
    return {"FAILED": "FAIL", "skipped": "-"}.get(row.get("status", ""), "-")


def _legend(variants: list[str]) -> None:
    text = "; ".join(f"{_short(v)} = {v}" for v in variants)
    print(textwrap.fill(text, 117, initial_indent="   ", subsequent_indent="   "))


def _table(header: list[str], lines: list[list[str]]) -> None:
    widths = [max([len(h)] + [len(line[i]) for line in lines]) for i, h in enumerate(header)]
    print("   " + "  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for line in lines:
        print("   " + "  ".join(c.rjust(w) for c, w in zip(line, widths)))


def bench_line(row: dict) -> str:
    name = f"   {row['name'][:34]:<34} "
    if row["status"] != "ok":
        message = row.get("message", "").strip()
        return f"{name}{row['status']:<8} {message.splitlines()[0][:160] if message else ''}"
    spread = f"[{row['p10_ms']:.3f}-{row['p90_ms']:.3f}]"
    extra = []
    if "tflops" in row:
        extra.append(f"{row['tflops']:6.2f} TFLOPS")
    if "vs_fp32_linear" in row:
        extra.append(f"{row['vs_fp32_linear']:5.2f}x")
    if "images_per_s" in row:
        extra.append(f"{row['images_per_s']:9.1f} img/s")
    if "err_vs_cpu" in row:
        extra.append(f"err {row['err_vs_cpu']:.1e}")
    if "path" in row:
        extra.append(row["path"])
    return f"{name}{row['ms']:>10.3f} ms {spread:<22} {' '.join(extra)}"


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps" if sys.platform == "darwin" else "cpu",
                   help="mps (default on a Mac) or cpu (dry run)")
    p.add_argument("--quick", action="store_true", help="tiny shapes and 1 round (for dry runs)")
    p.add_argument("--only", default="A,B,C", help="comma-separated sections: A (layers), B (models), C (training)")
    p.add_argument("--repeats", type=int, default=None, help="timed rounds per variant (default 10, 1 with --quick)")
    p.add_argument("--out", default=None,
                   help="JSON file or folder (default: apple_results/apple_bench-<time>.json)")
    a = p.parse_args(argv)
    sections = [s.strip().upper() for s in a.only.split(",") if s.strip()]
    if not sections or not set(sections) <= {"A", "B", "C"}:
        p.error(f"--only takes sections A, B and C, got {a.only!r}")
    if a.repeats is None:
        a.repeats = 1 if a.quick else 10
    if a.repeats < 1:
        p.error("--repeats must be at least 1")

    report = Report("apple_bench", a, bench_line, free_each_row=False)  # memory is freed after each case
    bench = Bench(report, a.device, a.quick, a.repeats)
    print(f"bitnet_vision Apple bench, device {a.device}{' (quick)' if a.quick else ''}: {describe(report.env)}")
    warning = torch_warning(report.env, a.device)
    if warning:
        print(banner(warning))
    for note in environment_notes(report.env):
        print(f"note: {note}")
    run = {"A": bench.layers, "B": bench.models, "C": bench.training}
    try:
        for s in ("A", "B", "C"):
            if s in sections:
                run[s]()
    except KeyboardInterrupt:
        report.data["interrupted"] = True
        print("\ninterrupted")
    bench.summary()
    report.finish()
    if warning:
        print(banner(warning))
    return report.data


if __name__ == "__main__":
    main()
