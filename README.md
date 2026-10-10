# bitnet-vision

Ternary ("1.58-bit") Vision Transformers in PyTorch. Every weight in the transformer layers is -1, 0 or
+1, following Microsoft's BitNet b1.58 recipe, with optional BitNet v2 Hadamard layers for 4-bit
activations. The repo covers the whole path: layers, a ViT model, CIFAR-10 training with distillation, and
export to packed 2-bit weights that make checkpoints about 12x smaller.

## What is in here

| Piece | What it does |
|---|---|
| `BitLinear` | Drop-in replacement for `nn.Linear`. Ternary weights (one absmean scale per matrix), 8-bit per-token activations, an RMSNorm on its input. Matches Microsoft's reference code exactly, forward and backward. |
| `HBitLinear` | BitNet v2 layer: rotates its input with an exact Hadamard matrix before quantizing, which spreads outliers and makes 4-bit activations workable. Works at every standard ViT width (192, 384, 768, 1536, 3072, ...). |
| `BitViT` | A Vision Transformer whose qkv, projection and MLP layers are ternary. The patch embedding, position embeddings, final norm and classifier head stay in full precision (quantizing those costs a lot of accuracy for very little size). |
| Training | CIFAR-10 loader, BitNet's two-stage schedule, FTerViT's low learning-rate restart, distillation from a full-precision teacher, "full precision first, then ternary" switching, bf16 autocast, resumable runs. |
| Export | `convert_for_inference` packs 4 weights per byte (Hugging Face BitNet layout) and runs int8 activations with an integer matmul. |

The design choices and the papers behind them are in [docs/RESEARCH.md](docs/RESEARCH.md). What was wrong
with the earlier attempts in this repo is in [docs/AUDIT.md](docs/AUDIT.md).

## Install

```bash
pip install torch  # any recent version, CPU or CUDA
pip install -e ".[dev]"
pytest -q          # about 30 seconds on a laptop CPU
```

## Quick start

```python
import torch
from bitnet_vision import BitLinear, create_model, convert_for_inference

# A ternary layer, used like nn.Linear
layer = BitLinear(768, 3072)
y = layer(torch.randn(8, 197, 768))

# A ternary ViT for 32x32 images, and its full-precision twin (same parameter names)
model = create_model("bitvit_cifar_tiny")
teacher = create_model("bitvit_cifar_tiny", quantize=False)
logits = model(torch.randn(4, 3, 32, 32))

# After training: pack the weights for inference (about 12x smaller state dict)
packed = convert_for_inference(model)
```

Presets: `bitvit_cifar_tiny` (1.8M parameters), `bitvit_cifar_small`, and the ImageNet-sized
`bitvit_tiny_patch16_224`, `bitvit_small_patch16_224`, `bitvit_base_patch16_224`. Options include
`hadamard=True` (BitNet v2 layers for the attention output and MLP down projections),
`hadamard_act_bits=4`, `weight_granularity="channel"`, `pool="mean"` and `layer_scale_init`.

## Training on CIFAR-10

CIFAR-10 is downloaded automatically (checksum verified) into `--data-dir`.

```bash
# 1) full-precision baseline, also the teacher
python scripts/train.py --fp --schedule cosine --wd 0.05 --epochs 15 --out runs/fp

# 2) ternary from scratch with BitNet's two-stage schedule
python scripts/train.py --schedule bitnet --lr 1e-3 --epochs 15 --out runs/ternary_scratch

# 3) ternary, started from the baseline and distilled from it, with a short low learning-rate restart
python scripts/train.py --init runs/fp/model.pt --teacher runs/fp/model.pt --schedule restart \
    --lr 2e-4 --wd 0.05 --stage2-wd 0.01 --epochs 10 --out runs/ternary_kd

# evaluate the packed 2-bit checkpoint, and time it
python scripts/eval.py runs/ternary_kd/packed.pt
python scripts/bench.py --threads 4
```

`scripts/cifar_experiments.sh` runs all three in sequence. Each run writes `metrics.json` (per-epoch
accuracy, the share of weights at -1/0/+1, and how many ternary codes flipped), `model.pt`, `packed.pt` and
`summary.json`. Interrupted runs continue with `--resume`.

Things worth knowing:

* **The share of zero weights is the health check.** A healthy run keeps roughly a third of the ternary
  weights at 0. If it drifts toward 100%, gradients are leaking into the quantization scales (that is
  exactly how one of the old branches failed).
* **Do not copy LLM learning rates.** Small vision models do best around 1e-3 from scratch and about
  1e-4 to 2e-4 when starting from full-precision weights.
* **Starting from a full-precision model helps most.** In published results it is worth about 5 points
  of ImageNet accuracy for small ViTs, and distillation adds more.
* **BitNet v2 4-bit activations:** train with `--hadamard`, then add `--final-hadamard-act-bits 4` to
  switch the Hadamard layers to 4-bit activations for the last 5% of training, as BitNet v2 does.

## Memory and speed, honestly

During training `BitLinear` keeps full-precision weights and only simulates quantization, so it is a bit
slower than `nn.Linear` and no smaller. The savings come from `convert_for_inference`: checkpoints shrink
about 12x and the weights stay ternary in memory. Speed needs a kernel that reads packed weights
directly (bitnet.cpp, T-MAC, or a custom Triton kernel); the reference inference path here unpacks
weights and uses `torch._int_mm` where the backend supports it, so expect memory savings rather than
speedups. `scripts/bench.py` measures both.

## Layout

```
bitnet_vision/
  quant.py        quantizers and the straight-through estimator
  layers.py       RMSNorm, BitLinear, HBitLinear, helpers to convert models
  hadamard.py     exact Hadamard rotations for 2^k, 12*2^k, 20*2^k, ... widths
  export.py       2-bit packing and the inference layer
  checkpoint.py   save/load for training and packed checkpoints
  models/vit.py   BitViT and presets
  training/       data, schedules, optimizer groups, distillation, training loop
scripts/          train.py, eval.py, bench.py, cifar_experiments.sh
tests/            unit tests, including exact comparisons with Microsoft's reference code
docs/             RESEARCH.md (papers and design choices), AUDIT.md (problems in the old branches)
```

## Main references

* BitNet b1.58: Ma et al., "The Era of 1-bit LLMs", arXiv 2402.17764, and Microsoft's "Training Tips,
  Code and FAQ" (the reference code this repo matches).
* BitNet v2: Wang et al., "Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs",
  arXiv 2504.18415.
* FTerViT: "Fully Ternary Vision Transformer", arXiv 2605.21171.
* TerViT: Xu et al., "An Efficient Ternary Vision Transformer", arXiv 2201.08050.

The full list, with what each paper changed in this code, is in [docs/RESEARCH.md](docs/RESEARCH.md).
