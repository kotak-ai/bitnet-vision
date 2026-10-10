#!/usr/bin/env python
"""Evaluate a BitViT checkpoint on the CIFAR-10 test set.

  python scripts/eval.py runs/ternary/model.pt      # training checkpoint (fake quantization)
  python scripts/eval.py runs/ternary/packed.pt     # packed 2-bit inference checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bitnet_vision.checkpoint import PACKED_FORMAT, load_model, load_packed  # noqa: E402
from bitnet_vision.export import packed_state_dict_bytes  # noqa: E402
from bitnet_vision.training import evaluate, load_cifar10, ternary_stats  # noqa: E402


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    _, test_set = load_cifar10(a.data_dir)
    test_set = test_set.subset(a.test_subset)
    fmt = torch.load(a.checkpoint, map_location="cpu", weights_only=True).get("format")
    if fmt == PACKED_FORMAT:
        model = load_packed(a.checkpoint)
        info = {"kind": "packed"}
    else:
        model = load_model(a.checkpoint)
        info = {"kind": "training", "ternary": ternary_stats(model) if model.config.quantize else None}
    info.update(evaluate(model, test_set))
    info["state_bytes"] = packed_state_dict_bytes(model.state_dict())
    info["file_bytes"] = os.path.getsize(a.checkpoint)
    print(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    main()
