"""Learning-rate and weight-decay schedules for ternary training.

Kinds:

* ``"bitnet"``: the exact two-stage recipe from Microsoft's BitNet b1.58 training FAQ (Table 2, Fig. 1).
  After a linear warmup the learning rate decays linearly to 0 over the whole run; at ``split`` (the
  midpoint) it is multiplied by ``stage2_factor`` (2/3, e.g. 1.5e-3 -> 1e-3 peaks) and weight decay
  switches from ``weight_decay`` to ``stage2_weight_decay`` (0). Weight decay pulls latent weights toward
  zero, where they keep flipping codes, so removing it lets the codes settle.
* ``"bitnet_cosine"``: the same with cosine curves, an approximation inspired by BitNet b1.58 2B4T
  (arXiv 2504.12285), which also uses a cosine weight-decay schedule in stage 1 and a separately chosen
  stage-2 peak.
* ``"restart"``: FTerViT (arXiv 2605.21171): a cosine main phase until ``split`` (0.9), then a short restart
  at ``restart_lr_ratio * peak_lr`` (0.1) that cosine-decays to a tenth of that, with
  ``stage2_weight_decay``. FTerViT's 10-epoch restart added 0 to 3.8 points (0 on CIFAR-10, 1.6 on
  CIFAR-100); the gain shrank the longer the main phase ran, so it behaves much like a final annealing.
* ``"wsd"``: warmup, hold ``peak_lr`` until ``split`` (0.9), then a cosine cooldown to ``min_lr`` with
  ``stage2_weight_decay``. "Linger and Lose" (arXiv 2609.32902) found that small ternary transformers
  lose capacity while lingering at mid-range learning rates under cosine decay, and a 10% WSD cooldown
  avoided it (shown on language models; untested on vision).
* ``"cosine"`` / ``"linear"``: one decay from ``peak_lr`` to ``min_lr`` with constant weight decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["TwoStageSchedule", "SCHEDULE_KINDS"]

SCHEDULE_KINDS = ("bitnet", "bitnet_cosine", "restart", "wsd", "cosine", "linear")
_DEFAULT_SPLIT = {"bitnet": 0.5, "bitnet_cosine": 0.5, "restart": 0.9, "wsd": 0.9}


def _decay(start: float, end: float, frac: float, shape: str) -> float:
    frac = min(max(frac, 0.0), 1.0)
    if shape == "linear":
        return start + (end - start) * frac
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * frac))


@dataclass
class TwoStageSchedule:
    """Callable returning ``(lr, weight_decay)`` for a step. See the module docstring for the kinds."""

    total_steps: int
    peak_lr: float
    kind: str = "bitnet"
    warmup_steps: int = 0
    weight_decay: float = 0.1
    stage2_weight_decay: float = 0.0
    split: float | None = None
    stage2_factor: float = 2.0 / 3.0
    restart_lr_ratio: float = 0.1
    min_lr: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(f"unknown schedule kind {self.kind!r}; choose from {SCHEDULE_KINDS}")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.split is None:
            self.split = _DEFAULT_SPLIT.get(self.kind, 1.0)
        if not 0.0 < self.split <= 1.0:
            raise ValueError("split must be in (0, 1]")
        limit = self.split_step if self.two_stage else self.total_steps
        if self.warmup_steps >= limit:
            raise ValueError(f"warmup ({self.warmup_steps} steps) must end before "
                             f"{'the stage switch' if self.two_stage else 'the end of training'} ({limit} steps)")

    @property
    def two_stage(self) -> bool:
        return self.kind in _DEFAULT_SPLIT

    @property
    def split_step(self) -> int:
        return int(round(self.split * self.total_steps)) if self.two_stage else self.total_steps

    def __call__(self, step: int) -> tuple[float, float]:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps, self.weight_decay
        frac = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        if self.kind in ("cosine", "linear"):
            return _decay(self.peak_lr, self.min_lr, frac, self.kind), self.weight_decay
        stage2 = step >= self.split_step
        wd = self.stage2_weight_decay if stage2 else self.weight_decay
        if self.kind in ("bitnet", "bitnet_cosine"):
            shape = "linear" if self.kind == "bitnet" else "cosine"
            lr = _decay(self.peak_lr, self.min_lr, frac, shape)
            return (lr * self.stage2_factor if stage2 else lr), wd
        split = self.split_step
        if self.kind == "wsd":
            if not stage2:
                return self.peak_lr, wd
            f = (step - split) / max(1, self.total_steps - split)
            return _decay(self.peak_lr, self.min_lr, f, "cosine"), wd
        # restart
        if not stage2:
            f = (step - self.warmup_steps) / max(1, split - self.warmup_steps)
            return _decay(self.peak_lr, self.min_lr, f, "cosine"), wd
        f = (step - split) / max(1, self.total_steps - split)
        top = self.peak_lr * self.restart_lr_ratio
        return _decay(top, top * 0.1, f, "cosine"), wd
