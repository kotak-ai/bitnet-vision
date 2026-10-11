# bitnet-vision

Ternary ("1.58-bit") Vision Transformers in PyTorch. Every weight in the transformer layers is -1, 0 or
+1, following Microsoft's BitNet b1.58 recipe, with optional BitNet v2 Hadamard layers for 4-bit
activations. The repo covers the whole path: layers, a ViT model, CIFAR-10 training with distillation, and
export to packed 2-bit weights that make checkpoints 8 to 14 times smaller.

## Words used here

* **Ternary weights**: each weight is stored as -1, 0 or +1 (about 1.58 bits of information), plus one
  floating-point scale per weight matrix. That is what makes the model small.
* **Absmean / absmax scale**: how a block of numbers is shrunk onto the small integer grid. Weights are
  divided by their average size (absmean); activations by their largest value per token (absmax).
* **Per-token**: each image patch (token) gets its own activation scale, so one loud patch does not
  drown out the others.
* **Straight-through estimator (STE)**: rounding has no useful slope, so during training the gradient is
  passed through the rounding step as if it were not there. Getting this exactly right is what the old
  branches of this repo failed at (see [docs/AUDIT.md](docs/AUDIT.md)).
* **RMSNorm**: rescales each token to a standard size. BitNet puts one inside every quantized layer.
* **Hadamard rotation**: a fixed, lossless mixing of the channels. It spreads a few very large values
  across all channels, so they can be stored with fewer bits (BitNet v2's trick for 4-bit activations).
* **Distillation**: the ternary "student" learns to copy the predictions of a full-precision "teacher".

## What is in here

| Piece | What it does |
|---|---|
| `BitLinear` | Drop-in replacement for `nn.Linear` with ternary weights and 8-bit activations. Matches Microsoft's reference code exactly, in outputs and gradients. |
| `HBitLinear` | BitNet v2 layer: rotates its input with an exact Hadamard matrix (with fixed random sign flips) before quantizing. Works at every standard ViT width (192, 384, 768, 1536, 3072, ...). |
| `BitViT` | A Vision Transformer whose qkv, projection and MLP layers are ternary. The patch embedding, position embeddings, norms and classifier head stay in full precision: quantizing those costs a lot of accuracy for very little size. |
| Training | CIFAR-10 loader, BitNet's two-stage schedule and other schedules, distillation, "full precision first, then ternary" switching, BitNet v2's late switch to 4-bit activations, bf16 autocast, safe resumable checkpoints. |
| Export | `convert_for_inference` stores 4 weights per byte and runs int8 activations through an integer matmul. |

Why each design choice was made, with the papers behind it: [docs/RESEARCH.md](docs/RESEARCH.md).
What was wrong with the earlier attempts in this repo: [docs/AUDIT.md](docs/AUDIT.md).

## Install

```bash
pip install torch          # any version from 2.2, CPU or CUDA
pip install -e ".[dev]"
pytest -q                  # about a minute on a laptop CPU; add -m "not slow" to skip training checks
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

# After training: pack the weights for inference
packed = convert_for_inference(model)
```

Presets: `bitvit_cifar_tiny` (1.8M parameters), `bitvit_cifar_small` (7.2M), and the ImageNet-sized
`bitvit_tiny_patch16_224`, `bitvit_small_patch16_224`, `bitvit_base_patch16_224`.

Model options (keyword arguments to `create_model`, or `scripts/train.py` flags):

| Option | Default | Meaning |
|---|---|---|
| `quantize` | True | False gives the full-precision twin (`--fp`) |
| `hadamard` | False | BitNet v2 layers for the attention output and MLP down projections (`--hadamard`) |
| `hadamard_random_signs` | True | sign flips before the rotation; turn off for BitNet v2's plain rotation (`--no-hadamard-signs`) |
| `act_bits`, `hadamard_act_bits` | 8, 8 | activation bits; 4 uses BitNet v2's absmean quantizer |
| `weight_granularity` | "tensor" | "channel" gives one weight scale per output row |
| `block_norm` | True | the usual ViT LayerNorms in each block, on top of the RMSNorm inside each layer (`--no-block-norm`) |
| `pool` | "cls" | "mean" or "seqpool" (learned attention pooling) |
| `deadzone_lambda` | 0 | Tequila's fix for weights stuck at 0; the paper uses 1e-3 |
| `layer_scale_init`, `drop_path_rate`, `mlp_act` | off, 0, "gelu" | LayerScale, stochastic depth, squared-ReLU MLP |

## Training on CIFAR-10

CIFAR-10 is downloaded automatically (checksum verified) into `--data-dir`.

```bash
# 1) full-precision baseline, also the teacher
python scripts/train.py --fp --schedule cosine --wd 0.05 --epochs 15 --out runs/fp

# 2) ternary from scratch with BitNet's two-stage schedule
python scripts/train.py --schedule bitnet --lr 1e-3 --epochs 15 --out runs/ternary_scratch

# 3) ternary, started from the baseline and distilled from it
python scripts/train.py --init runs/fp/model.pt --teacher runs/fp/model.pt --schedule restart \
    --lr 2e-4 --wd 0.05 --stage2-wd 0.01 --warmup-epochs 0.5 --epochs 10 --out runs/ternary_kd

# evaluate the packed 2-bit checkpoint, and benchmark the trained model
python scripts/eval.py runs/ternary_kd/packed.pt
python scripts/bench.py --checkpoint runs/ternary_kd/model.pt
```

`scripts/cifar_experiments.sh` runs all three in sequence, plus a fourth full-precision run with the
same schedule as run 2, so the two differ only in quantization. Every run writes `metrics.json` (per-epoch
accuracy and loss), `model.pt` and `summary.json`; ternary runs also record the share of weights at -1, 0
and +1 and how many changed each epoch, and write the packed `packed.pt`. Interrupted runs continue with
`--resume` and the original command line; the model is rebuilt from the checkpoint, and changed training
settings are refused rather than silently bending the schedule.

Schedules (`--schedule`): `bitnet` (Microsoft's recipe: learning rate times 2/3 and weight decay off at the
midpoint), `bitnet_cosine`, `restart` (a short low learning-rate phase at the end, from FTerViT), `wsd`
(hold the learning rate, then cool down over the last 10%), `cosine` and `linear`.

Things worth knowing:

* **The share of zero weights is the health check.** Healthy runs keep roughly 25 to 40% of the ternary
  weights at 0, and training logs a warning outside 10 to 70%. Drifting toward 100% means gradient is
  leaking into the quantization scales (exactly how one of the old branches failed).
* **Do not copy learning rates from language models.** Small vision models do best around 1e-3 from
  scratch and 1e-4 to 2e-4 when starting from full-precision weights.
* **Starting from a full-precision model helps most.** Published results put it at about 5 points of
  ImageNet accuracy for small ViTs. `--fp-fraction 0.2` does the same inside a single run.
* **Distillation** defaults to 90% teacher-matching (KL) and 10% ordinary cross-entropy, which helps when
  the teacher is a small CPU-trained model. `--distill-alpha 1.0` is FTerViT's pure-KL setting.
* **BitNet v2 4-bit activations:** train with `--hadamard` and add `--final-act-bits 4`. Like BitNet v2,
  this trains with 8-bit activations and switches every layer to 4 bits for the last 5% of steps
  (`--final-act-bits-scope rotated` switches only the Hadamard layers).
* `torch.compile` works, but on PyTorch older than 2.5 a `--quant-ramp-epochs` ramp makes it recompile
  every step; compile only after the ramp or use a newer PyTorch.

## Results on CIFAR-10

One run of each line in `scripts/cifar_experiments.sh` with `bitvit_cifar_tiny` (1.8M parameters), on 4
CPU threads with no GPU, at commit 7ef581c (the training code has not changed since). Per-epoch numbers
are in [docs/cifar10_results.json](docs/cifar10_results.json).

| Run | Epochs | Test accuracy | Packed 2-bit model | Weights at 0 | Saved size |
|---|---|---|---|---|---|
| Full precision, cosine schedule | 15 | 63.4% |  |  | 7.2 MB |
| Full precision, BitNet schedule | 15 | 65.1% |  |  | 7.2 MB |
| Ternary, from scratch, BitNet schedule | 15 | 67.1% | 67.1% | 32% | 0.61 MB |
| Ternary, from the cosine full-precision model + distillation | 10 | 64.0% | 64.0% | 33% | 0.61 MB |

What this shows:

* **The 2-bit export is faithful.** The packed model scores the same as the model it came from (within
  0.02 points) and is 11.8 times smaller.
* **Training is healthy.** Each of the 24 ternary layers keeps 31 to 37% of its weights at 0 (no layer
  collapsed toward all zeros), and the accuracy curves rise steadily.
* **Ternary kept up with full precision on this small budget.** With the identical schedule, the ternary
  model was ahead at every epoch from epoch 3 on and finished 2.0 points higher. It also reached a lower
  training loss, so the lead comes from learning faster, not only from doing better on new images.
* **The schedule matters too.** The same full-precision model scored 1.7 points higher with BitNet's
  two-stage schedule than with a cosine schedule.

What it does not show:

* **These are short runs.** 15 epochs on a CPU, and every model was still improving at the end. In the
  runs trained from scratch, accuracy on the training images never rose above accuracy on the test images
  (training sees randomly cropped and flipped images), so all of them were far from finished. CCT reports
  about 91% for ViTs of 3 to 4M parameters trained for 300 epochs.
* **One seed each.** Gaps of a point or two between runs can come from the random seed alone.
* **"Ternary beats full precision" is not the general rule.** Published results have ternary ViTs a few
  points behind full precision (ViT-1.58b on CIFAR-10: 72.3% vs 76.3%), so do not expect this lead to
  hold with longer training or bigger models.
* **Distillation was limited by its teacher.** The distilled model started from the 63.4% cosine
  baseline, which was also its teacher, and ended 0.5 points above it. A stronger teacher (longer
  training, or a pretrained model) is what would make this route pay off.

## Memory and speed, honestly

During training `BitLinear` keeps full-precision weights and only simulates quantization, so it is a bit
slower than `nn.Linear` and no smaller. The savings come from `convert_for_inference`:

| Preset | Full precision | Packed | Smaller by |
|---|---|---|---|
| `bitvit_cifar_tiny` | 7.2 MB | 0.61 MB | 11.8x |
| `bitvit_cifar_small` | 28.7 MB | 2.1 MB | 13.6x |
| `bitvit_tiny_patch16_224` | 22.9 MB | 3.0 MB | 7.6x |
| `bitvit_small_patch16_224` | 88.3 MB | 8.7 MB | 10.1x |
| `bitvit_base_patch16_224` | 346.5 MB | 28.0 MB | 12.4x |

The smaller ImageNet models compress less because their full-precision parts (1000-class head, 197
position embeddings) are a bigger share. On disk weights take 2 bits each; in memory each layer keeps an
int8 copy (1 byte per weight) to avoid unpacking on every call.

Speed: `torch._int_mm` is used where it is faster: on CUDA when the shapes allow it, and on CPU only if
a one-time timing check says so (from PyTorch 2.6, CPUs without AVX512-VNNI fall back to a very slow
loop). Expect modest gains at best from pure PyTorch; real speedups need a kernel that reads packed
weights directly (bitnet.cpp, T-MAC, or a custom kernel). `scripts/bench.py` measures every variant and
records the machine state, so compare numbers only from a quiet machine.

## Checkpoint format

`save_packed` writes `{"format": "bitnet_vision.packed.v1", "conventions": ..., "model_config": ...,
"model": state_dict}`. Each ternary layer stores `weight_packed` (uint8, shape `[ceil(out/4), in]`, code + 1
in 2 bits, 4 row blocks per byte in the bit layout used by Hugging Face transformers and Microsoft's
BitNet 2B4T checkpoint), `weight_scale` (a multiplier, `W ~= codes * weight_scale`), and `bias`, which is
added after rescaling. Module names and the bias convention differ from transformers, so these files are
not drop-in loadable there. All checkpoints load with `torch.load(..., weights_only=True)`.

## Tests

`pytest` runs about 180 tests, including exact comparisons with Microsoft's reference code (forward and
backward), float16/bfloat16 behavior, Hadamard exactness at every ViT width, export equivalence, resume
exactness, and short end-to-end training runs. `python tools/mutation_test.py` plants 43 known bugs one at
a time (including every bug found in the old branches) and checks that the tests catch each one.

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
tools/            mutation_test.py
tests/            unit and regression tests; reference.py holds Microsoft's reference code
docs/             RESEARCH.md (papers and design choices), AUDIT.md (problems in the old branches)
```

## License

No license has been chosen for this repository yet, so by default others may not reuse the code.
`tests/reference.py` contains code from Microsoft's MIT-licensed unilm repository, with its notice.

## Main references

* BitNet b1.58: Ma et al., "The Era of 1-bit LLMs", arXiv 2402.17764, and Microsoft's "Training Tips,
  Code and FAQ" (the reference code this repo matches).
* BitNet v2: Wang et al., "Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs",
  arXiv 2504.18415.
* FTerViT: "Fully Ternary Vision Transformer", arXiv 2605.21171.
* TerViT: Xu et al., "An Efficient Ternary Vision Transformer", arXiv 2201.08050.
* QuaRot: Ashkboos et al., arXiv 2404.00456 (randomized Hadamard rotations).

The full list, with what each paper changed in this code, is in [docs/RESEARCH.md](docs/RESEARCH.md).
