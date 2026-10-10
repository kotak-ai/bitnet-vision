# Audit of the earlier branches

Before this rebuild the repository had three unmerged branches, each an independent attempt by an AI
coding agent at the same task (BitLinear, RMSNorm, and a BitNet v2 HBitLinear). None contained any vision
code. This rewrite supersedes them and they are meant to be deleted. Their last commits, in case anything
needs to be recovered after deletion (`git fetch origin <sha>` works while GitHub still keeps the objects):

| Branch | Last commit |
|---|---|
| `cursor/bitnet-core-modules-implementation-1bef` | `9758640669aee5f033398d870b77abb8093402a5` |
| `cursor/bitnet-core-modules-implementation-6825` | `08f911c89732195c9b38f42179a4c4a6ffeba5d0` |
| `cursor/bitnet-core-modules-implementation-8216` | `d16bbb28d6077f88448100175575c9a40cee543e` |

Each finding below was reproduced by one agent and then independently re-tested by a second agent that
tried to disprove it. The test that now guards against each problem is listed.

## Training was broken on every branch

Small 3-layer ternary MLPs trained on the same synthetic task, mean of 3 seeds:

| Version | Test accuracy |
|---|---|
| Microsoft reference recipe | 0.83 |
| Any branch after the fixes below | 0.80 |
| 1bef as written | 0.68 (its HBitLinear stayed at chance level) |
| 6825 as written (PyTorch >= 2.14) | 0.68 |
| 8216 as written | 0.63 |

1. **1bef let gradient flow through the weight scale** (`gamma = mean|W|`). At initialization this is
   invisible (gradient cosine 1.0000 with the reference), but during training it pushes every weight
   toward zero on almost every step, and 93-95% of the last layer's codes ended up at 0. Fixed by doing all
   quantization inside `.detach()`. Guarded by `test_gradients_match_reference_after_training`.
2. **6825 clamped after the straight-through rounding.** `clamp` passes no gradient outside its range, and
   since PyTorch 2.14 none at the range limits either, so every weight at +1 or -1 (about 75%) stopped
   learning. Guarded by `test_every_weight_receives_gradient` and the reference-gradient tests.
3. **8216 applied the straight-through trick in the wrong units.** Weight gradients came out `gamma` times
   too small and input gradients `eta/128` times too small per layer; in an 8-layer stack the first four
   layers got exactly zero gradient. Guarded by `test_bitlinear_matches_reference_outputs_and_gradients`.
4. **8216 multiplied the bias by the tiny dequantization factor**, so a bias of 1.0 added about 0.001.
   Guarded by `test_bias_is_added_after_rescaling`.
5. **The scale side channel existed on all three branches.** Fixing only the headline bug of 6825 or 8216
   made them train exactly as badly as 1bef.

## Numerical problems shared by all branches

6. **No norm inside BitLinear.** Microsoft's code normalizes the input of every BitLinear. Now built in
   (`input_norm=True`), guarded by `test_input_norm_makes_layer_scale_invariant`.
7. **Float16 failures.** RMSNorm squared values in float16, so any value of 256 or more zeroed the whole
   token; activation scaling overflowed into NaN for zero or tiny tokens; the integer-domain matmul could
   overflow. Now: norm and quantizer math in float32, and the matmul runs on dequantized values as in the
   reference. Guarded by `test_half_precision_is_finite_for_zero_and_large_tokens` and
   `test_zero_and_extreme_tokens_stay_finite`.
8. **Q_b = 128 instead of 127.** The paper text says 128 but the released code uses 127; with 128 the
   largest value of each token was always clipped. Guarded by
   `test_activation_codes_fit_int8_and_max_is_not_clipped`.

## Hadamard (BitNet v2) problems

9. **Pad-then-slice is not a rotation.** For widths like 768 (not a power of two), 1bef and 6825 padded to
   1024, transformed, and cut back to 768. That shrinks a third of the directions to half their length and
   keeps only 75% of the signal energy. 8216 kept the padded width instead (33% more weights). Now:
   exact Hadamard matrices for `m * 2^k` widths via the Paley construction (`m` = 12, 20, 24, 44, 48, 60, 72,
   80, ...), which covers every standard ViT width. Guarded by
   `test_vit_widths_get_exact_orthonormal_rotation`.
10. **4-bit activations**, the main point of BitNet v2, were missing (8216) or used a different recipe
    (1bef, 6825). Now `HBitLinear(act_bits=4)` uses the absmean 4-bit quantizer.
11. **6825's README cited an astronomy paper** as BitNet v2 (the correct id is arXiv 2504.18415).

## Engineering problems

12. **Weak tests.** Planting plausible bugs in a copy of each branch, 8 of 13 survived 1bef's tests and 9 of
    14 survived 6825's (removing the straight-through estimator entirely still passed). 8216 had no tests.
    One 6825 test was unseeded and failed about 41% of the time on its own. 1bef's tests failed under a
    plain `pytest` command. The same exercise on the rebuilt suite planted 25 bugs, including every bug
    listed on this page, and all 25 were caught.
13. **1bef rejected 4D inputs** such as channels-last image tensors, and crashed on empty batches.
14. **No real savings.** All branches only simulated quantization: checkpoints were the same size as
    `nn.Linear` and inference was slower. Now `convert_for_inference` stores 2-bit packed weights
    (about 12x smaller checkpoints) and uses int8 matmuls where available.
15. **Hygiene.** Compiled `__pycache__` files were committed, and there was no `.gitignore`, no packaging,
    no CI, and three incompatible layouts and APIs.
