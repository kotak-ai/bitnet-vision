"""Training and evaluation loop for BitViT on in-memory image tensors."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..layers import HBitLinear, bitlinear_modules, set_quant_strength
from .data import TensorDataset, augment, iterate_batches, normalize
from .distill import distillation_loss
from .optim import apply_schedule, build_optimizer
from .schedule import TwoStageSchedule

__all__ = ["TrainConfig", "evaluate", "ternary_stats", "switch_activation_bits", "train"]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    schedule: str = "bitnet"  # see TwoStageSchedule: bitnet, bitnet_cosine, restart, wsd, cosine, linear
    split: float | None = None  # stage boundary as a fraction of training (default 0.5 bitnet, 0.9 restart)
    stage2_factor: float = 2.0 / 3.0  # bitnet: lr multiplier after the split
    restart_lr_ratio: float = 0.1  # restart: stage-2 peak lr as a fraction of lr
    warmup_epochs: float = 1.0
    weight_decay: float = 0.1
    stage2_weight_decay: float = 0.0
    min_lr: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float | None = 1.0
    label_smoothing: float = 0.1
    augment: bool = True
    # Only used with a teacher. 0.9 keeps a little cross-entropy as an anchor, which matters with a weak
    # CPU-trained teacher; 1.0 is FTerViT's pure-KL recipe (it used a strong ImageNet-pretrained teacher).
    distill_alpha: float = 0.9
    distill_temperature: float = 1.0
    # Full precision first, then ternary, in one run with one optimizer (continual QAT, arXiv 2502.11895).
    fp_fraction: float = 0.0
    quant_ramp_epochs: float = 0.0  # >0: blend quantization in gradually after the full-precision phase
    # BitNet v2: train with 8-bit activations, then switch to 4 bits for the last ~5% of steps, reusing the
    # optimizer state. scope "all" switches every BitLinear (the paper's recipe); "rotated" only HBitLinear.
    final_act_bits: int | None = None
    final_act_bits_fraction: float = 0.05
    final_act_bits_scope: str = "all"
    amp: str | None = None  # "bf16" for bfloat16 autocast
    device: str = "cpu"
    seed: int = 0
    eval_batch_size: int = 500
    log_every: int = 100

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _autocast(device: str, amp: str | None):
    if amp is None:
        return contextlib.nullcontext()
    if amp != "bf16":
        raise ValueError("amp must be None or 'bf16' (float16 needs loss scaling, which is not wired in)")
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model: nn.Module, data: TensorDataset, batch_size: int = 500, device: str = "cpu",
             amp: str | None = None) -> dict[str, float]:
    """Top-1 accuracy and mean cross-entropy on ``data``."""
    was_training = model.training
    model.eval()
    correct, loss_sum = 0, 0.0
    for images, labels in iterate_batches(data, batch_size, shuffle=False):
        x, y = normalize(images.to(device)), labels.to(device)
        with _autocast(device, amp):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    model.train(was_training)
    n = max(1, len(data))
    return {"acc": correct / n, "loss": loss_sum / n}


@torch.no_grad()
def ternary_stats(model: nn.Module) -> dict[str, float]:
    """Share of ternary weights at -1 / 0 / +1 across all BitLinear layers.

    A healthy run keeps a sizable share of zeros (around a third). A share that drifts toward 1.0 means
    weights are collapsing to zero, the failure mode of letting gradients flow through the scales.
    """
    counts = torch.zeros(3, dtype=torch.float64)
    for m in bitlinear_modules(model):
        codes, _ = m.ternary_weight()
        counts += torch.stack([(codes == v).sum() for v in (-1, 0, 1)]).double().cpu()
    total = counts.sum().clamp(min=1)
    return {"neg": (counts[0] / total).item(), "zero": (counts[1] / total).item(), "pos": (counts[2] / total).item()}


def _codes_snapshot(model: nn.Module) -> list[Tensor]:
    return [m.ternary_weight()[0].cpu() for m in bitlinear_modules(model)]


def train(
    model: nn.Module,
    train_data: TensorDataset,
    test_data: TensorDataset | None,
    cfg: TrainConfig,
    teacher: nn.Module | None = None,
    out_dir: str | None = None,
    log: Callable[[str], None] = print,
    resume: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train ``model`` and return a history dict. Saves ``last.pt`` and ``metrics.json`` in ``out_dir``.

    ``resume`` is a checkpoint dict written by a previous run (see ``last.pt``).
    """
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    device = cfg.device
    model.to(device).train()
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    steps_per_epoch = len(train_data) // cfg.batch_size
    if steps_per_epoch == 0:
        raise ValueError("batch_size is larger than the training set")
    total_steps = steps_per_epoch * cfg.epochs
    schedule = TwoStageSchedule(
        total_steps=total_steps, peak_lr=cfg.lr, kind=cfg.schedule,
        warmup_steps=int(cfg.warmup_epochs * steps_per_epoch), weight_decay=cfg.weight_decay,
        stage2_weight_decay=cfg.stage2_weight_decay, split=cfg.split, stage2_factor=cfg.stage2_factor,
        restart_lr_ratio=cfg.restart_lr_ratio, min_lr=cfg.min_lr,
    )
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay, betas=tuple(cfg.betas))
    config = getattr(model, "config", None)
    quantized = bool(getattr(config, "quantize", True))
    fp_steps = int(round(cfg.fp_fraction * total_steps)) if quantized else 0
    ramp_steps = int(cfg.quant_ramp_epochs * steps_per_epoch) if quantized else 0
    if fp_steps + ramp_steps > total_steps:
        raise ValueError(f"the full-precision phase ({fp_steps} steps) plus the quantization ramp ({ramp_steps} "
                         f"steps) is longer than training ({total_steps} steps); the model would never be fully "
                         "ternary")
    if cfg.final_act_bits_scope not in ("all", "rotated"):
        raise ValueError("final_act_bits_scope must be 'all' or 'rotated'")
    low_bit_step = (int(round((1.0 - cfg.final_act_bits_fraction) * total_steps))
                    if cfg.final_act_bits is not None else None)
    if (low_bit_step is not None and cfg.final_act_bits_scope == "rotated"
            and not any(isinstance(m, HBitLinear) for m in model.modules())):
        raise ValueError("final_act_bits_scope='rotated' but the model has no HBitLinear layers (use hadamard=True)")

    def quant_strength_at(s: int) -> float:
        if s < fp_steps:
            return 0.0
        return min(1.0, (s - fp_steps + 1) / ramp_steps) if ramp_steps else 1.0

    history: dict[str, Any] = {"train_config": cfg.to_dict(), "epochs": []}
    start_epoch, step = 0, 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch, step = resume["epoch"], resume["step"]
        history = resume.get("history", history)
        gen.set_state(resume["rng"])
        if "torch_rng" in resume:  # global generator (DropPath masks)
            torch.set_rng_state(resume["torch_rng"])
        if "cuda_rng" in resume and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        # Restore the phase the run was in (also when it had already finished).
        if step > 0 and (fp_steps or ramp_steps):
            set_quant_strength(model, quant_strength_at(step - 1))
        if low_bit_step is not None and step > low_bit_step:
            switch_activation_bits(model, cfg.final_act_bits, cfg.final_act_bits_scope)
        log(f"resumed from epoch {start_epoch}, step {step}")

    prev_codes = _codes_snapshot(model) if quantized else None
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        run_loss, run_correct, seen = 0.0, 0, 0
        for images, labels in iterate_batches(train_data, cfg.batch_size, shuffle=True, generator=gen,
                                              drop_last=True):
            lr, wd = schedule(step)
            apply_schedule(optimizer, lr, wd)
            if fp_steps or ramp_steps:
                set_quant_strength(model, quant_strength_at(step))
            if low_bit_step is not None and step >= low_bit_step:
                switch_activation_bits(model, cfg.final_act_bits, cfg.final_act_bits_scope)
            if cfg.augment:
                images = augment(images, generator=gen)
            x, y = normalize(images.to(device)), labels.to(device)
            with _autocast(device, cfg.amp):
                logits = model(x)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
            if teacher is not None:
                loss = distillation_loss(logits, t_logits, y, alpha=cfg.distill_alpha,
                                         temperature=cfg.distill_temperature, label_smoothing=cfg.label_smoothing)
            else:
                loss = F.cross_entropy(logits.float(), y, label_smoothing=cfg.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            step += 1
            run_loss += loss.item() * y.shape[0]
            run_correct += (logits.argmax(-1) == y).sum().item()
            seen += y.shape[0]
            if cfg.log_every and step % cfg.log_every == 0:
                log(f"epoch {epoch + 1} step {step}/{total_steps} loss {run_loss / seen:.4f} lr {lr:.2e} wd {wd:g}")

        record: dict[str, Any] = {
            "epoch": epoch + 1, "step": step, "train_loss": run_loss / max(1, seen),
            "train_acc": run_correct / max(1, seen), "lr": lr, "weight_decay": wd,
            "minutes": (time.time() - t0) / 60,
        }
        if test_data is not None:
            ev = evaluate(model, test_data, cfg.eval_batch_size, device, cfg.amp)
            record.update(test_acc=ev["acc"], test_loss=ev["loss"])
        if quantized:
            record["ternary"] = ternary_stats(model)
            zero = record["ternary"]["zero"]
            if quant_strength_at(step - 1) > 0 and not 0.1 <= zero <= 0.7:
                log(f"warning: {zero:.0%} of ternary weights are 0 (healthy runs stay roughly between 10% and 70%; "
                    "a drift toward 100% usually means gradient is leaking into the quantization scales)")
            codes = _codes_snapshot(model)
            changed = sum((a != b).sum().item() for a, b in zip(codes, prev_codes))
            record["code_flip_rate"] = changed / max(1, sum(c.numel() for c in codes))
            prev_codes = codes
        if quantized:
            record["quant_strength"] = quant_strength_at(step - 1)
        history["epochs"].append(record)
        if config is not None:
            history["model_config"] = config.to_dict()  # current config (activation bits may have switched)
        log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in record.items()}))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                     "step": step, "rng": gen.get_state(), "torch_rng": torch.get_rng_state(),
                     "history": history, "model_config": history.get("model_config")}
            if torch.cuda.is_available():
                state["cuda_rng"] = torch.cuda.get_rng_state_all()
            _atomic_save(lambda f, s=state: torch.save(s, f), os.path.join(out_dir, "last.pt"))
            _atomic_save(lambda f: f.write(json.dumps(history, indent=1).encode()),
                         os.path.join(out_dir, "metrics.json"))
    return history


def _atomic_save(write: Callable[[Any], Any], path: str) -> None:
    """Write to a temporary file, then rename, so an interrupted save never leaves a truncated file."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def switch_activation_bits(model: nn.Module, bits: int, scope: str = "all") -> None:
    """Move BitLinear layers to ``bits``-bit activations (absmean below 8 bits) and record it in the config.

    scope "all" changes every BitLinear (BitNet v2's recipe), "rotated" only the HBitLinear layers.
    """
    method = "absmax" if bits >= 8 else "absmean"
    for m in bitlinear_modules(model):
        if scope == "all" or isinstance(m, HBitLinear):
            m.act_bits, m.act_method = bits, method
    config = getattr(model, "config", None)
    if config is not None:
        if getattr(config, "hadamard", False):
            config.hadamard_act_bits = bits
        if scope == "all" and hasattr(config, "act_bits"):
            config.act_bits = bits
