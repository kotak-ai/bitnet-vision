"""Microsoft's published BitNet b1.58 reference recipe, used as the ground truth in tests.

activation_quant, weight_quant and the straight-through structure of bitlinear_forward are copied verbatim
from "The Era of 1-bit LLMs: Training Tips, Code and FAQ" (Fig. 3,
https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf),
Copyright (c) Microsoft Corporation, MIT License (https://github.com/microsoft/unilm/blob/master/LICENSE).
The PDF calls RMSNorm without defining it; rms_norm below is a standard LLaMA-style RMSNorm (eps 1e-6,
learnable weight). An optional bias is added by F.linear after both quantizers (the reference has none).
"""

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def bitlinear_forward(x, weight, bias=None, norm_weight=None):
    if norm_weight is None:
        norm_weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    x_norm = rms_norm(x, norm_weight)
    x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
    w_quant = weight + (weight_quant(weight) - weight).detach()
    return F.linear(x_quant, w_quant, bias)
