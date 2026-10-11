# Research behind the design

This page lists the papers that shaped the code and what each one changed. It was compiled in October 2026
from about 100 papers; every paper below was opened and its numbers checked by a second, independent
reader, who also corrected several claims (noted where it matters). The first column says how an idea is
used: **built in** (on by default), **option** (a flag), or **noted** (documented, not implemented).
Paper numbers are arXiv ids.

## The core layer

| Use | Idea | Source | Evidence |
|---|---|---|---|
| built in | Ternary weights with one absmean scale per matrix, 8-bit per-token activations with a 127/absmax scale, straight-through gradients with the scales held constant, RMSNorm inside every BitLinear | BitNet b1.58 (2402.17764) and Microsoft's "Training Tips, Code and FAQ" | `BitLinear` matches the published reference code exactly, forward and backward (`tests/test_layers.py`). The paper writes the activation range as 128; the released code uses 127, which this repo follows. |
| built in | Keep the scales as plain averages, not learned parameters | Tequila (2509.23809, Table 1), Ternary Mamba (2606.18114) | In a controlled 1B-parameter comparison, absmean scored 0.445 against 0.354 for a learned step size (LSQ) and 0.370 for ParetoQ's learned scales; Ternary Mamba saw weights collapse to zero with learned scales. ParetoQ (2502.02631) reports the opposite with its own setup, so the evidence is mixed but favors absmean for short training. |
| option | One scale per output row (`weight_granularity="channel"`) | TerViT (2201.08050), Tequila (Table 3) | Small gains: +0.6 points on DeiT-T in TerViT, +0.008 average in Tequila. Ternary Mamba saw no effect. |
| option | Tequila's dead-zone term (`deadzone_lambda`, paper default 1e-3) | Tequila (2509.23809) | Weights stuck at code 0 get a direct gradient through a small bias term that folds into the bias at export. +0.026 average on a 1B language model; untested on vision. |

## The vision model

| Use | Idea | Source | Evidence |
|---|---|---|---|
| built in | Keep patch embedding, position embeddings, norms and head in full precision | TerViT (2201.08050) | Ternarizing the first and last layers dropped DeiT-T from 66.9% to 44.2%; 8-bit cost only 0.3 points. FTerViT (2605.21171) can ternarize them, but only with 260 epochs of distillation, and its fully ternary DeiT-Tiny loses 9 points on ImageNet. |
| built in | Keep the softmax attention math in floating point | Q-ViT (2210.06707), FTerViT code | Quantizing the attention matrices to 2 bits cost up to 10 points on DeiT-S. |
| built in | Norms in both places: the usual block LayerNorms plus an RMSNorm inside every BitLinear | FTerViT's released code, "An Extra RMSNorm is All You Need" (2505.08823) | FTerViT, the strongest ternary ViT so far, keeps both. TerDiT (2405.14854, FID 4.34) also beat a BitNet-style model that drops the block norms (FID 6.60), though the two differ in other ways too. |
| option | `pool="seqpool"` (learned attention pooling) or `"mean"` | CCT (2104.05704, Table 8) | On CIFAR-10 a small full-precision ViT gained 4.5 points from SeqPool over the cls token; not yet measured for ternary models. |
| option | `layer_scale_init` | BinaryViT (2306.16678) | A per-channel residual scale helped binary ViTs a little; unverified for ternary ones. |

## Training

| Use | Idea | Source | Evidence |
|---|---|---|---|
| built in | `--schedule bitnet`: linear decay to 0; at the midpoint the learning rate is multiplied by 2/3 and weight decay goes from 0.1 to 0 | BitNet b1.58 FAQ (Table 2, Fig. 1) | The loss drops sharply at the switch and ends below single-stage runs. These are language-model settings; the fact-checker suggests weight decay 0.05 and other values are equally defensible for small vision models. |
| option | `--schedule wsd`: hold the learning rate, cool down over the last 10% | "Linger and Lose" (2609.32902) | Small ternary transformers trained with cosine decay lost much of what they had learned while lingering at mid-range learning rates; a 10% cooldown avoided it. Shown on language models only. |
| option | `--schedule restart`: a short low learning-rate phase at the end | FTerViT (2605.21171) | Adds 0 points on CIFAR-10, 1.6 on CIFAR-100 and 2.4 to 3.8 on ImageNet. The gain shrank the longer the main phase ran, so it behaves much like a final annealing. |
| option | Start from a trained full-precision model (`--init`) and distill from it (`--teacher`) | TerViT, FTerViT, BitVLA (2506.07530) | Starting from full precision was worth 5.7 points on DeiT-T. Note that TerViT's teacher was a free ImageNet-pretrained model, not budget-matched. |
| built in | Distillation mixes 90% KL with 10% cross-entropy (`--distill-alpha 0.9`); 1.0 is pure KL | FTerViT, Ternary Mamba, BHViT, DIDB-ViT | FTerViT uses pure KL with a strong pretrained teacher but never compares against a mix. Ternary Mamba found KL-only much worse than KL plus cross-entropy, and BHViT and DIDB-ViT found 0.8 to 0.9 best. With a small CPU-trained teacher, keeping some cross-entropy is the safer default. |
| option | `--fp-fraction`: train in full precision first, then switch to ternary in the same run | Continual QAT (2502.11895) | At equal budget, switching at 20% of training beat ternary-from-scratch (loss 3.088 vs 3.15). A gradual ramp only shortened a brief loss spike, so the ramp is optional and off by default. |
| guidance | Small learning rates for small vision models | BitNet b1.58 Reloaded (2407.09527), FTerViT | Weak evidence: Reloaded's small CIFAR-10 CNN did best at 1e-4 and collapsed at 5e-2, but in 10-epoch constant-rate runs. Sweep 5e-4 to 2e-3 from scratch; 1e-4 to 2e-4 from full-precision weights. |
| guidance | Light augmentation (random crop and flip) for short runs | OFQ (2302.02210), CCT | Heavier augmentation helps long runs (+3.7 points for AutoAugment at 300 epochs in CCT), not short CPU runs. |

## BitNet v2 and 4-bit activations

| Use | Idea | Source | Evidence |
|---|---|---|---|
| option | `hadamard=True`: rotate the inputs of the attention output projection and the MLP down projection before quantizing | BitNet v2 (2504.18415) | The rotation spreads outliers so 4-bit activations work. These two layers see the most outlier-heavy inputs. |
| built in | Exact Hadamard matrices for non-power-of-two widths (Paley construction times a power of two) | QuaRot (2404.00456), Dao-AILab fast-hadamard-transform | Covers 192, 384, 768, 1536, 3072, 640, 1280 exactly. The old pad-then-cut approach lost a quarter of the signal. |
| built in | Fixed random sign flips before the rotation (`hadamard_random_signs`) | QuaRot (randomized Hadamard), checked here | An unsigned Hadamard maps the mean of a vector onto one coordinate. After GELU, ViT MLP inputs are mostly positive, so the plain rotation creates an outlier instead of removing one. The fact-checker measured 4-bit error 0.31 without signs vs 0.11 with them at width 768; `tests/test_hadamard.py` checks the effect. BitNet v2 does not need signs because its language-model inputs are roughly zero-mean. |
| option | 4-bit activations use per-token absmean with a sqrt(7) scale and the range -8 to 7, switched on for the last 5% of training for every layer (`--final-act-bits 4`) | BitNet v2 (Eq. 7 and training setup) | BitNet v2 trained 95% of tokens at 8 bits, then all linear layers at 4 bits, reusing the optimizer state. |
| noted | Rotating all four ternary layers at 4 bits | RobuQ (2509.23582) | Helped a ternary diffusion transformer at 4-bit activations; worth trying if 4-bit accuracy disappoints. |

## Inference

| Use | Idea | Source |
|---|---|---|
| built in | Pack 4 ternary weights per byte in the bit layout of Hugging Face transformers and Microsoft's BitNet 2B4T checkpoint, with the scale stored as a multiplier (`mean(abs(W))`) | BitNet b1.58 2B4T (2504.12285), transformers BitNet integration, bitnet.cpp converter |
| built in | Unpack once to int8 in memory; use `torch._int_mm` only where a timing check shows it is faster | Checked here: from PyTorch 2.6, CPUs without AVX512-VNNI run it as a slow loop |
| noted | Real speedups need kernels that read packed weights (bitnet.cpp 2502.11880, T-MAC, Litespark 2605.06485); at ViT-Tiny sizes the activation quantization costs more than the matmul itself, so it must be fused | RobuQ (2509.23582, Table 9) |
| noted | Other formats: bitnet.cpp's I2_S (128-weight blocks, x86), base-3 packing at 1.6 bits per weight, sparse ternary formats | bitnet.cpp, BITCOS (2609.16338), Sherry (2601.07892) |

## Ideas looked at and left out

* ParetoQ's learned "SEQ" ternary quantizer: zeros 70 to 80% of weights at the start and scored far below
  absmean in Tequila's controlled comparison.
* Learned step sizes (LSQ) and threshold multipliers (TernaryCLIP's beta): see the core layer table.
* QuEST's trust mask and RMS-based 4-bit scale (2502.05003): gains are bundled with other changes in the
  paper; a possible future option.
* Hidden-state distillation (BitVLA), cached multi-crop teacher logits (Q-Var, 2307.00331), oscillation
  regularizers (OFQ, Q-Var): reasonable options for longer runs, not needed for the first version.

## What to expect on CIFAR-10

Published ternary ViT results on CIFAR-10 either fine-tune large pretrained models at 224x224 (FTerViT:
97.43% vs 97.52% in full precision, with an ImageNet-pretrained teacher) or train weak models from scratch
(ViT-1.58b, 2406.18051: 72.27% vs 76.28%). There is no published number for a short CPU run at 32x32, so
this repo always trains and reports a full-precision twin of the same size under the same budget. The
numbers from this repo's own runs are in the README, under "Results on CIFAR-10".
