#!/usr/bin/env python
"""Latency of full precision vs training-mode BitViT vs packed inference BitViT.

Reports the median over repeated runs after warmup. Honest caveat: without a dedicated ternary kernel
(bitnet.cpp, T-MAC, a Triton kernel), packed inference here saves memory, not time.

  python scripts/bench.py --model bitvit_cifar_tiny --batch-size 64 --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bitnet_vision.export import convert_for_inference, packed_state_dict_bytes  # noqa: E402
from bitnet_vision.models import MODEL_PRESETS, create_model  # noqa: E402


@torch.inference_mode()
def time_model(model, x, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bitvit_cifar_tiny", choices=sorted(MODEL_PRESETS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    fp = create_model(a.model, quantize=False).eval()
    tern = create_model(a.model).eval()
    packed = convert_for_inference(tern)
    packed_cached = convert_for_inference(tern, cache_unpacked=True)
    cfg = fp.config
    x = torch.randn(a.batch_size, cfg.in_chans, cfg.img_size, cfg.img_size)
    res = {
        "model": a.model, "batch_size": a.batch_size, "threads": torch.get_num_threads(),
        "ms_full_precision": time_model(fp, x, a.repeats),
        "ms_training_mode_ternary": time_model(tern, x, a.repeats),
        "ms_packed": time_model(packed, x, a.repeats),
        "ms_packed_cached_int8": time_model(packed_cached, x, a.repeats),
        "bytes_fp32": packed_state_dict_bytes(fp.state_dict()),
        "bytes_packed": packed_state_dict_bytes(packed.state_dict()),
    }
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
