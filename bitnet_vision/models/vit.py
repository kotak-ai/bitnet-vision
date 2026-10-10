"""BitViT: a Vision Transformer whose transformer linear layers are ternary BitLinear layers.

Kept in full precision, following TerViT (arXiv 2201.08050), ViT-1.58b and BitVLA: the convolutional
patch embedding, cls token, position embeddings, final norm and classifier head. TerViT measured that
quantizing the first and last layers to 2 bits costs DeiT-T 22 points of ImageNet accuracy, while they
are under 4% of the parameters.

Ternary: qkv, attention output projection, MLP fc1 and fc2. Softmax attention runs in full precision.
Norms sit in both places, as in FTerViT (arXiv 2605.21171), the strongest ternary ViT so far: the usual
pre-norm LayerNorm of each block (``block_norm=True``) and an RMSNorm inside every BitLinear (the BitNet
recipe). TerDiT (arXiv 2405.14854) also found that dropping the block norms hurts.

With ``quantize=False`` the very same architecture runs in full precision. That model is the baseline,
and its weights can initialize or teach the ternary model because every parameter name matches.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..layers import BitLinear, HBitLinear, set_quant_strength

__all__ = ["BitViTConfig", "BitViT", "MODEL_PRESETS", "create_model"]


@dataclass
class BitViTConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 10
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 2.0
    qkv_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    mlp_act: str = "gelu"  # "gelu" or "relu2" (squared ReLU, used by BitNet b1.58 2B4T)
    quantize: bool = True  # False: same model in full precision (baseline / teacher)
    act_bits: int | None = 8
    weight_granularity: str = "tensor"
    hadamard: bool = False  # BitNet v2: HBitLinear for the attention output proj and MLP fc2
    hadamard_act_bits: int | None = 8  # activation bits of those HBitLinear layers (4 in BitNet v2 stage 2)
    block_norm: bool = True
    layer_scale_init: float | None = None  # e.g. 1e-4: learnable per-channel scale on each residual branch
    pool: str = "cls"  # "cls" token or "mean" over patch tokens
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.img_size % self.patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        if self.pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {self.pool!r}")
        if self.mlp_act not in ("gelu", "relu2"):
            raise ValueError(f"mlp_act must be 'gelu' or 'relu2', got {self.mlp_act!r}")

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _act_method(bits: int | None) -> str:
    return "absmax" if bits is None or bits >= 8 else "absmean"  # BitNet v2: absmean for 4-bit activations


def _bit_linear(cfg: BitViTConfig, d_in: int, d_out: int, bias: bool, rotated: bool = False) -> BitLinear:
    if rotated and cfg.hadamard:
        bits = cfg.hadamard_act_bits
        return HBitLinear(d_in, d_out, bias=bias, act_bits=bits, act_method=_act_method(bits),
                          weight_granularity=cfg.weight_granularity)
    return BitLinear(d_in, d_out, bias=bias, act_bits=cfg.act_bits, act_method=_act_method(cfg.act_bits),
                     weight_granularity=cfg.weight_granularity)


class DropPath(nn.Module):
    """Stochastic depth: randomly drops the residual branch per sample during training."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    """Full-precision convolutional patchifier: (B, C, H, W) -> (B, num_patches, embed_dim)."""

    def __init__(self, cfg: BitViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_chans, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, cfg: BitViTConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.qkv = _bit_linear(cfg, cfg.embed_dim, 3 * cfg.embed_dim, cfg.qkv_bias)
        self.proj = _bit_linear(cfg, cfg.embed_dim, cfg.embed_dim, cfg.proj_bias, rotated=True)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, cfg: BitViTConfig) -> None:
        super().__init__()
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.fc1 = _bit_linear(cfg, cfg.embed_dim, hidden, cfg.mlp_bias)
        self.fc2 = _bit_linear(cfg, hidden, cfg.embed_dim, cfg.mlp_bias, rotated=True)
        self.act = cfg.mlp_act

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x) if self.act == "gelu" else F.relu(x).square()
        return self.fc2(x)


class LayerScale(nn.Module):
    """Learnable per-channel scale on a residual branch (CaiT); BinaryViT found a similar affine helps."""

    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, cfg: BitViTConfig, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim) if cfg.block_norm else nn.Identity()
        self.attn = Attention(cfg)
        self.ls1 = LayerScale(cfg.embed_dim, cfg.layer_scale_init) if cfg.layer_scale_init else nn.Identity()
        self.norm2 = nn.LayerNorm(cfg.embed_dim) if cfg.block_norm else nn.Identity()
        self.mlp = Mlp(cfg)
        self.ls2 = LayerScale(cfg.embed_dim, cfg.layer_scale_init) if cfg.layer_scale_init else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class BitViT(nn.Module):
    """Vision Transformer with ternary transformer layers. ``BitViT(config)`` or ``BitViT(**fields)``."""

    def __init__(self, config: BitViTConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        cfg = dataclasses.replace(config or BitViTConfig(), **overrides)
        self.config = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, cfg.embed_dim))
        rates = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist() if cfg.depth else []
        self.blocks = nn.ModuleList([Block(cfg, rates[i]) for i in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.reset_parameters()
        set_quant_strength(self, 1.0 if cfg.quantize else 0.0)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):  # includes BitLinear
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}

    def forward_features(self, x: Tensor) -> Tensor:
        """(B, C, H, W) -> pooled features (B, embed_dim), after the final norm."""
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = x[:, 0] if self.config.pool == "cls" else x[:, 1:].mean(dim=1)
        return self.norm(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    # CIFAR-sized (32x32, 4x4 patches -> 64 tokens). About 1.8M and 7.1M parameters.
    "bitvit_cifar_tiny": dict(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=2.0),
    "bitvit_cifar_small": dict(img_size=32, patch_size=4, embed_dim=384, depth=6, num_heads=6, mlp_ratio=2.0),
    # ImageNet-sized, DeiT shapes.
    "bitvit_tiny_patch16_224": dict(img_size=224, patch_size=16, num_classes=1000, embed_dim=192, depth=12,
                                    num_heads=3, mlp_ratio=4.0),
    "bitvit_small_patch16_224": dict(img_size=224, patch_size=16, num_classes=1000, embed_dim=384, depth=12,
                                     num_heads=6, mlp_ratio=4.0),
    "bitvit_base_patch16_224": dict(img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12,
                                    num_heads=12, mlp_ratio=4.0),
}


def create_model(name: str, **overrides: Any) -> BitViT:
    """Build a preset, e.g. ``create_model("bitvit_cifar_tiny", quantize=False)``."""
    if name not in MODEL_PRESETS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODEL_PRESETS)}")
    return BitViT(BitViTConfig(**{**MODEL_PRESETS[name], **overrides}))
