# Research behind the design

This page lists the papers that shaped the code and what each one changed. The first column says how the
idea is used: **built in** (on by default), **option** (a flag), or **noted** (documented, not
implemented). Paper numbers are arXiv ids.

## The core layer

| Use | Idea | Source | Evidence |
|---|---|---|---|
| built in | Ternary weights with one absmean scale per matrix, 8-bit per-token activations with a 127/absmax scale, straight-through gradients with the scales held constant, RMSNorm inside every BitLinear | BitNet b1.58 (2402.17764) and Microsoft's "Training Tips, Code and FAQ" | `BitLinear` matches the published reference code exactly, forward and backward (`tests/test_layers.py`). |
| built in | Keep the quantization scales as a plain statistic, not a learned parameter | OFQ (2302.02210), Ternary Mamba (2606.18114) | A statistic scale beat a learned step size by 6.3 points at 2 bits on DeiT-S; learned scales caused weights to collapse to zero in Ternary Mamba. |
| option | One scale per output row (`weight_granularity="channel"`) | TerViT (2201.08050), Tequila (2509.23809) | +0.6 points on DeiT-T ImageNet in TerViT. |

## The vision model

| Use | Idea | Source | Evidence |
|---|---|---|---|
| built in | Keep patch embedding, position embeddings, final norm and head in full precision | TerViT (2201.08050) | Ternarizing the first and last layers dropped DeiT-T from 66.9% to 44.2%; 8-bit cost only 0.3 points. |
| built in | Keep the softmax attention math in floating point | Q-ViT (2210.06707), FTerViT (2605.21171) | Quantizing the attention matrices to 2 bits cost up to 10 points on DeiT-S. |
| built in | Norms in both places: the usual block LayerNorms plus an RMSNorm inside every BitLinear | FTerViT (2605.21171) code, "An Extra RMSNorm is All You Need" (2505.08823), TerDiT (2405.14854) | FTerViT, the strongest ternary ViT so far, keeps both. TerDiT (FID 4.34) also beat a BitNet-style model that drops the block norms (FID 6.60), though the two differ in other ways too. |
| option | `pool="mean"`, `layer_scale_init` | BinaryViT (2306.16678) | Average pooling and a per-channel residual scale helped binary ViTs; unverified for ternary ones, so both are off by default. |

## Training

| Use | Idea | Source | Evidence |
|---|---|---|---|
| built in | `--schedule bitnet`: linear decay to 0; at the midpoint the learning rate is multiplied by 2/3 and weight decay goes from 0.1 to 0 | BitNet b1.58 FAQ (Table 2, Fig. 1) | The loss drops sharply at the switch and ends below single-stage runs. |
| option | `--schedule restart`: a short low learning-rate restart for the last 10% of training | FTerViT (2605.21171) | A 10-epoch restart at a tenth of the learning rate added 1.6 to 3.5 points and beat 150 extra epochs of normal training. |
| option | Start from a trained full-precision model (`--init`) and distill from it (`--teacher`, KL loss at temperature 1) | TerViT, FTerViT, BitVLA (2506.07530) | Starting from full precision was worth 5.7 points on DeiT-T and 5.2 on DeiT-S. FTerViT's ternary DeiT-Tiny reaches 97.43% on CIFAR-10 vs 97.52% in full precision. |
| option | `--fp-fraction`: train in full precision first, then switch to ternary in the same run | Continual QAT (2502.11895), ParetoQ (2502.02631) | Switching at about 20% of training beat ternary-from-scratch. |
| guidance | Use small learning rates for small vision models (about 1e-3 from scratch, 1e-4 to 2e-4 from full-precision weights) | BitNet b1.58 Reloaded (2407.09527), FTerViT | On CIFAR-10 a small ternary network did best at 1e-4 to 1e-3 and collapsed at 5e-2. LLM learning rates do not transfer. |
| guidance | Light augmentation (random crop and flip) for short runs | OFQ, TernaryCLIP (2510.21879) | Heavy augmentation slowed convergence of low-bit models. |

## BitNet v2 and 4-bit activations

| Use | Idea | Source | Evidence |
|---|---|---|---|
| option | `hadamard=True`: rotate the inputs of the attention output projection and the MLP down projection with a Hadamard matrix before quantizing | BitNet v2 (2504.18415) | The rotation spreads outliers so 4-bit activations work. |
| built in | Exact Hadamard matrices for non-power-of-two widths (12 x 2^k and similar, Paley construction) | QuaRot, Dao-AILab fast-hadamard-transform | Covers 192, 384, 768, 1536, 3072 exactly. The old pad-then-cut approach lost a quarter of the signal. |
| option | 4-bit activations use per-token absmean with a sqrt(7) scale, and are switched on only for the last 5% of training (`--final-hadamard-act-bits 4`) | BitNet v2 (Eq. 7 and training setup) | BitNet v2 trained at 8 bits and switched to 4 bits for the final 5% of tokens. |

## Inference

| Use | Idea | Source |
|---|---|---|
| built in | Pack 4 ternary weights per byte in the Hugging Face transformers BitNet layout, with a weight scale of 1 / mean(abs(W)) | BitNet b1.58 2B4T (2504.12285), transformers BitNet integration |
| noted | Real speedups need kernels that read packed weights directly | bitnet.cpp (2502.11880), T-MAC |

## What to expect on CIFAR-10

Published ternary ViT results on CIFAR-10 either fine-tune large pretrained models at 224x224 (FTerViT:
within 0.1 points of full precision) or train weak models from scratch (ViT-1.58b, 2406.18051: about 4
points below full precision). There is no published number for a short CPU run at 32x32, so this repo
always trains and reports a full-precision twin of the same size under the same budget.
