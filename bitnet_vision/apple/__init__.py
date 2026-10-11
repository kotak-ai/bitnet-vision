"""Experimental Apple Silicon support: fused Metal kernels for PyTorch's MPS backend and an MLX port of the
ternary inference layer.

Nothing here has run on Apple hardware yet: everything was written and checked on Linux, where only the
CPU fallbacks and MLX's CPU backend run. ``scripts/apple_check.py`` and ``scripts/apple_bench.py`` check
correctness and measure speed on a Mac.
"""
