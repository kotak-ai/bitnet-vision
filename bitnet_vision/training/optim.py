"""Optimizer helpers: weight decay only on weight matrices, never on norms, biases or embeddings."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["param_groups", "build_optimizer", "apply_schedule"]


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Two AdamW groups: matrices (decayed) and everything else (biases, norm weights, pos/cls embeddings)."""
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and name not in skip:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "apply_decay": True},
        {"params": no_decay, "weight_decay": 0.0, "apply_decay": False},
    ]


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8
) -> torch.optim.AdamW:
    """AdamW with betas (0.9, 0.95), as in the BitNet b1.58 training tips."""
    return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps)


def apply_schedule(optimizer: torch.optim.Optimizer, lr: float, weight_decay: float) -> None:
    """Set the learning rate on every group and the weight decay on the decayed group only."""
    for group in optimizer.param_groups:
        group["lr"] = lr
        if group.get("apply_decay", False):
            group["weight_decay"] = weight_decay
