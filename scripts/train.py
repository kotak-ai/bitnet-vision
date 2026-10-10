#!/usr/bin/env python
"""Train a BitViT (or its full-precision twin) on CIFAR-10.

Examples:
  # full-precision baseline (also the teacher and initialization for the ternary model)
  python scripts/train.py --fp --schedule cosine --wd 0.05 --epochs 30 --out runs/fp
  # ternary from scratch with BitNet's two-stage schedule (lr x2/3 and weight decay off at the midpoint)
  python scripts/train.py --schedule bitnet --lr 1e-3 --epochs 30 --out runs/ternary
  # ternary initialized from and distilled by the full-precision run, FTerViT-style low-lr restart at the end
  python scripts/train.py --init runs/fp/model.pt --teacher runs/fp/model.pt --schedule restart \
      --lr 2e-4 --wd 0.05 --stage2-wd 0.01 --epochs 30 --out runs/ternary_kd
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bitnet_vision.checkpoint import (  # noqa: E402
    load_model,
    load_packed,
    save_model,
    save_packed,
    transfer_weights,
)
from bitnet_vision.export import packed_state_dict_bytes  # noqa: E402
from bitnet_vision.models import MODEL_PRESETS, create_model  # noqa: E402
from bitnet_vision.training import TrainConfig, evaluate, load_cifar10, train  # noqa: E402
from bitnet_vision.training.schedule import SCHEDULE_KINDS  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", default="bitvit_cifar_tiny", choices=sorted(MODEL_PRESETS))
    p.add_argument("--fp", action="store_true", help="train the full-precision twin (no quantization)")
    p.add_argument("--hadamard", action="store_true", help="BitNet v2 HBitLinear for attn proj and MLP fc2")
    p.add_argument("--act-bits", type=int, default=8)
    p.add_argument("--hadamard-act-bits", type=int, default=8)
    p.add_argument("--weight-granularity", default="tensor", choices=["tensor", "channel"])
    p.add_argument("--no-block-norm", action="store_true", help="drop the block LayerNorms (keep only the "
                   "RMSNorm inside each BitLinear, as in the BitNet LLM recipe)")
    p.add_argument("--layer-scale", type=float, default=None, help="LayerScale init, e.g. 1e-4")
    p.add_argument("--mlp-act", default="gelu", choices=["gelu", "relu2"])
    p.add_argument("--pool", default="cls", choices=["cls", "mean"])
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--schedule", default="bitnet", choices=list(SCHEDULE_KINDS))
    p.add_argument("--split", type=float, default=None, help="stage boundary (default 0.5 bitnet, 0.9 restart)")
    p.add_argument("--stage2-factor", type=float, default=2.0 / 3.0, help="bitnet: lr multiplier after split")
    p.add_argument("--restart-lr-ratio", type=float, default=0.1, help="restart: stage-2 lr as fraction of lr")
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--stage2-wd", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--init", default=None, help="checkpoint to initialize weights from (e.g. the fp run)")
    p.add_argument("--teacher", default=None, help="checkpoint of a full-precision teacher for distillation")
    p.add_argument("--distill-alpha", type=float, default=1.0)
    p.add_argument("--distill-temperature", type=float, default=1.0)
    p.add_argument("--fp-fraction", type=float, default=0.0,
                   help="train this share of steps in full precision, then switch to ternary")
    p.add_argument("--quant-ramp-epochs", type=float, default=0.0)
    p.add_argument("--final-hadamard-act-bits", type=int, default=None,
                   help="BitNet v2: switch HBitLinear activations to this many bits near the end (e.g. 4)")
    p.add_argument("--final-act-bits-fraction", type=float, default=0.05)
    p.add_argument("--amp", default=None, choices=["bf16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    train_set, test_set = load_cifar10(a.data_dir)
    train_set, test_set = train_set.subset(a.train_subset), test_set.subset(a.test_subset)

    overrides = dict(quantize=not a.fp, hadamard=a.hadamard, act_bits=a.act_bits,
                     hadamard_act_bits=a.hadamard_act_bits, weight_granularity=a.weight_granularity,
                     block_norm=not a.no_block_norm, layer_scale_init=a.layer_scale, mlp_act=a.mlp_act, pool=a.pool,
                     drop_path_rate=a.drop_path)
    model = create_model(a.model, **overrides)
    if a.init:
        transfer_weights(model, load_model(a.init))
        print(f"initialized from {a.init}")
    teacher = load_model(a.teacher, quantize=False) if a.teacher else None

    cfg = TrainConfig(
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, schedule=a.schedule, split=a.split,
        stage2_factor=a.stage2_factor, restart_lr_ratio=a.restart_lr_ratio, warmup_epochs=a.warmup_epochs,
        weight_decay=a.wd, stage2_weight_decay=a.stage2_wd, label_smoothing=a.label_smoothing,
        augment=not a.no_augment, distill_alpha=a.distill_alpha, distill_temperature=a.distill_temperature,
        fp_fraction=a.fp_fraction, quant_ramp_epochs=a.quant_ramp_epochs,
        final_hadamard_act_bits=a.final_hadamard_act_bits, final_act_bits_fraction=a.final_act_bits_fraction,
        amp=a.amp, device=a.device, seed=a.seed, log_every=a.log_every,
    )
    resume = None
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        resume = torch.load(last, map_location="cpu", weights_only=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    history = train(model, train_set, test_set, cfg, teacher=teacher, out_dir=a.out, resume=resume,
                    log=lambda s: print(s, flush=True))
    model.cpu()
    save_model(model, os.path.join(a.out, "model.pt"), train_config=cfg.to_dict())

    summary = {"model": a.model, "quantized": not a.fp, "params": sum(p.numel() for p in model.parameters()),
               "final_test_acc": history["epochs"][-1].get("test_acc"),
               "fp32_state_bytes": packed_state_dict_bytes(model.state_dict())}
    if not a.fp:
        payload = save_packed(model, os.path.join(a.out, "packed.pt"))
        packed = load_packed(os.path.join(a.out, "packed.pt"))
        summary["packed_state_bytes"] = packed_state_dict_bytes(payload["model"])
        summary["packed_test_acc"] = evaluate(packed, test_set)["acc"]
        summary["compression"] = summary["fp32_state_bytes"] / summary["packed_state_bytes"]
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
