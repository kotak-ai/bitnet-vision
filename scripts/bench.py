#!/usr/bin/env python
"""Forward-pass latency of the same BitViT in full precision, training mode, and packed inference mode.

  python scripts/bench.py --batch-sizes 1,64 --threads 1,4
  python scripts/bench.py --checkpoint runs/ternary_kd/model.pt --compile

Protocol: every variant is the same architecture; runs are interleaved and repeated, and the median and
10th/90th percentiles are reported; thread count, CPU features, PyTorch version and machine load are
recorded because they change the numbers a lot. Compare runs only from a quiet machine. The packed model
also reports how often its predictions agree with the training-mode model on the benchmark input.
Without a dedicated ternary kernel the main win is memory; int8 matmuls give a modest speedup on CPUs
with AVX512-VNNI, mostly when the model is compiled.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bitnet_vision.checkpoint import load_model  # noqa: E402
from bitnet_vision.export import convert_for_inference, packed_state_dict_bytes  # noqa: E402
from bitnet_vision.models import MODEL_PRESETS, create_model  # noqa: E402


def _percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def environment() -> dict:
    env = {"torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine()}
    try:
        env["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except AttributeError:
        pass
    if hasattr(os, "getloadavg"):
        env["load_average_1m"] = round(os.getloadavg()[0], 2)
    env["cpu_count"] = os.cpu_count()
    return env


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bitvit_cifar_tiny", choices=sorted(MODEL_PRESETS))
    p.add_argument("--checkpoint", default=None, help="trained ternary model.pt (default: random weights)")
    p.add_argument("--batch-sizes", default="1,64")
    p.add_argument("--threads", default=str(min(4, os.cpu_count() or 1)), help="comma-separated thread counts")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--compile", action="store_true", help="also time torch.compile'd variants")
    a = p.parse_args(argv)

    torch.manual_seed(0)
    tern = load_model(a.checkpoint).eval() if a.checkpoint else create_model(a.model).eval()
    fp = create_model(a.model, **{**tern.config.to_dict(), "quantize": False}).eval()
    fp.load_state_dict(tern.state_dict())  # same weights, quantization off
    variants = {
        "fp32": fp,
        "ternary_training_mode": tern,
        "packed_int8": convert_for_inference(tern, int_matmul="auto"),
        "packed_float_matmul": convert_for_inference(tern, int_matmul="never"),
    }
    if a.compile:
        for name in ("fp32", "packed_int8"):
            try:
                compiled = torch.compile(variants[name])
                with torch.inference_mode():
                    cfg = tern.config
                    compiled(torch.randn(2, cfg.in_chans, cfg.img_size, cfg.img_size))
                variants[f"{name}_compiled"] = compiled
            except Exception as e:  # noqa: BLE001 - compilation support varies by machine
                print(f"torch.compile failed for {name}: {type(e).__name__}: {e}", file=sys.stderr)

    cfg = tern.config
    results = {"model": a.model, "checkpoint": a.checkpoint, "environment": environment(), "runs": [],
               "bytes_fp32_state": packed_state_dict_bytes(fp.state_dict()),
               "bytes_packed_state": packed_state_dict_bytes(variants["packed_int8"].state_dict())}
    for threads in (int(t) for t in a.threads.split(",")):
        torch.set_num_threads(threads)
        for bs in (int(b) for b in a.batch_sizes.split(",")):
            x = torch.randn(bs, cfg.in_chans, cfg.img_size, cfg.img_size)
            times = {name: [] for name in variants}
            with torch.inference_mode():
                for m in variants.values():  # warmup (also unpacks weights and runs int8 timing checks)
                    m(x)
                    m(x)
                for _ in range(a.repeats):
                    for name, m in variants.items():  # interleaved, so load changes hit every variant
                        t = time.perf_counter()
                        m(x)
                        times[name].append((time.perf_counter() - t) * 1000)
                ref_logits = tern(x)
                packed_logits = variants["packed_int8"](x)
            run = {"threads": threads, "batch_size": bs, "ms": {
                name: {"median": round(statistics.median(ts), 3), "p10": round(_percentile(ts, 0.1), 3),
                       "p90": round(_percentile(ts, 0.9), 3)} for name, ts in times.items()},
                   "packed_vs_training_argmax_agreement": (packed_logits.argmax(-1) == ref_logits.argmax(-1))
                   .float().mean().item(),
                   "packed_vs_training_max_logit_diff": (packed_logits - ref_logits).abs().max().item()}
            results["runs"].append(run)
    results["environment"]["load_average_1m_end"] = environment().get("load_average_1m")
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
