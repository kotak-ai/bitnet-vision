"""Knowledge distillation from a full-precision teacher.

Distillation and starting from full-precision weights are the two biggest accuracy levers for ternary
ViTs: TerViT (arXiv 2201.08050) gains 5.7 points on DeiT-T just from the full-precision start, and
FTerViT (arXiv 2605.21171) trains with a pure KL loss at temperature 1, noting that mixing in
cross-entropy conflicts with KL for low-bit students.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

__all__ = ["distillation_loss"]


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor | None = None,
    alpha: float = 1.0,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """``alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, labels)``.

    ``alpha=1`` (default) is the pure-KL recipe. The teacher logits are treated as constants.
    """
    t = temperature
    kd = F.kl_div(
        F.log_softmax(student_logits.float() / t, dim=-1),
        F.log_softmax(teacher_logits.detach().float() / t, dim=-1),
        log_target=True,
        reduction="batchmean",
    ) * (t * t)
    if alpha >= 1.0 or labels is None:
        return kd
    ce = F.cross_entropy(student_logits.float(), labels, label_smoothing=label_smoothing)
    return alpha * kd + (1.0 - alpha) * ce
