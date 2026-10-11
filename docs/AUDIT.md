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

Each finding below was reproduced by one reviewer and then re-tested by a second reviewer who tried to
disprove it. The numbers were measured during that audit with scripts that ran the old branches directly.
The test that now guards against each problem is listed.

A few terms: the **scale** is the number a weight matrix (or a token's activations) is divided by before
rounding. The **straight-through estimator** is the trick that lets training see past the rounding step;
its job is to pass the learning signal (the gradient) through unchanged. **Gradient cosine** compares the
direction of two learning signals: 1.0 means identical.

## Training was broken on every branch

Small 3-layer ternary networks trained on the same synthetic task, average of 3 seeds:

| Version | Test accuracy |
|---|---|
| Microsoft reference recipe | 0.83 |
| Any branch after the fixes below | 0.80 |
| 1bef as written | 0.68 (its Hadamard layer never learned at all) |
| 6825 as written (PyTorch 2.14 and later) | 0.68 |
| 8216 as written | 0.63 |

1. **1bef let the learning signal flow into the weight scale** (the average weight size). At the start of
   training this looks perfect (gradient cosine 1.0000 with the reference), but during training it pushes
   every weight toward zero on almost every step; 93 to 95% of the last layer's weights ended up at 0.
   Fixed by doing all rounding inside `.detach()`, as the reference does. Guarded by
   `test_gradients_match_reference_after_training`.
2. **6825 clipped values after the rounding trick instead of before.** Clipping blocks the learning signal
   at its limits, so every weight sitting at +1 or -1 (about 75% of them on recent PyTorch) stopped
   learning. Guarded by `test_every_weight_receives_gradient` and the reference-gradient tests.
3. **8216 applied the rounding trick to unscaled values.** Its learning signals came out far too small:
   weight gradients shrank by the average weight size, and the signal passed to earlier layers shrank by
   each token's largest value divided by 128, again at every layer. In an 8-layer stack the first four
   layers got exactly zero gradient and stopped learning. Guarded by
   `test_bitlinear_matches_reference_outputs_and_gradients`.
4. **8216 multiplied the bias by the tiny rescaling factor**, so a bias of 1.0 added about 0.001. Guarded
   by `test_bias_is_added_after_rescaling`.
5. **The scale leak from item 1 existed on all three branches.** Fixing only the headline bug of 6825 or
   8216 made them train exactly as badly as 1bef.

## Numerical problems shared by all branches

6. **No norm inside BitLinear.** Microsoft's code normalizes the input of every BitLinear. Now built in,
   guarded by `test_input_norm_makes_layer_scale_invariant`.
7. **Float16 failures.** RMSNorm squared values in float16, so any value of 256 or more wiped out the whole
   token; activation scaling overflowed into NaN for zero or tiny tokens; the matrix multiply could
   overflow. Now norm and rounding run in float32, and the multiply works on already-rescaled values as in
   the reference. Guarded by `test_half_precision_is_finite_for_zero_and_large_tokens` and
   `test_zero_and_extreme_tokens_stay_finite`.
8. **Activation range 128 instead of 127.** The paper text says 128 but the released code uses 127; with
   128 the largest value of each token was always clipped. Guarded by
   `test_activation_codes_fit_int8_and_max_is_not_clipped`.

## Hadamard (BitNet v2) problems

9. **Padding, rotating and cutting back is not a rotation.** For widths like 768 (not a power of two),
   1bef and 6825 padded to 1024, applied the Hadamard transform and cut the result back to 768. That
   shrinks a third of the signal's directions to half their length and keeps only 75% of its energy. 8216
   kept the padded width instead (33% more weights). Now an exact Hadamard matrix is built for widths like
   12 x 2^k (using the Paley construction for the factor 12), which covers every standard ViT width.
   Guarded by `test_vit_widths_get_exact_orthonormal_rotation`.
10. **4-bit activations**, the main point of BitNet v2, were missing (8216) or used a different recipe
    (1bef, 6825). Now `act_bits=4` uses BitNet v2's absmean quantizer, and training can switch to it for
    the final 5% of steps as the paper does.
11. **6825's README cited an astronomy paper** as BitNet v2 (the correct id is arXiv 2504.18415).

## Engineering problems

12. **Weak tests.** Planting plausible bugs in a copy of each branch, 8 of 13 survived 1bef's tests and 9 of
    14 survived 6825's (removing the rounding trick entirely still passed). 8216 had no tests. One 6825
    test was unseeded and failed about 41% of the time on its own. 1bef's tests failed under a plain
    `pytest` command. For the rebuilt suite, `tools/mutation_test.py` plants 44 bugs one at a time,
    including a version of each code bug in items 1 to 10 and 13, and checks that the tests catch each one.
13. **1bef rejected 4D inputs** such as channels-last image tensors, and crashed on empty batches. Guarded
    by `test_bitlinear_matches_reference_outputs_and_gradients` (2D to 5D inputs) and
    `test_empty_and_noncontiguous_inputs`.
14. **No real savings.** All branches only simulated quantization: checkpoints were the same size as
    `nn.Linear` and inference was slower. Now `convert_for_inference` stores 2-bit packed weights
    (checkpoints 8 to 14 times smaller depending on the model).
15. **Hygiene.** Compiled `__pycache__` files were committed, and there was no `.gitignore`, no packaging,
    no CI, and three incompatible layouts and APIs.
