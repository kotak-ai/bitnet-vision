"""Exact orthonormal Hadamard rotations for any common vision-transformer width.

BitNet v2 (arXiv 2504.18415) rotates activations with a Hadamard matrix before quantizing them, which
spreads outliers across channels. Sylvester Hadamard matrices only exist for powers of two, but ViT widths
are usually 12 * 2^k (192, 384, 768, 1536, 3072) or 20 * 2^k (640, 1280).

The earlier branches handled such widths by zero-padding to the next power of two, transforming, then
cutting back to the original width. That map is not a rotation: it shrinks a third of the input directions
to half their length (see docs/AUDIT.md). Here we instead build an exact n x n Hadamard matrix as a
Kronecker product ``H_m (x) H_{2^a} (x) H_{2^b} ...`` where ``H_m`` comes from the Paley construction
(m = q + 1 for a prime q = 3 mod 4, e.g. m = 12, 20, 24, 44, 48, 60, 72, 80) and the power-of-two part is
split into small Sylvester factors. Each factor is applied with one small matmul, so the cost is
O(n * sum(factor sizes)) per token, and autograd handles the backward pass.
This is the same idea as QuaRot's and Dao-AILab fast-hadamard-transform's 12N/20N kernels.

Random signs: an unsigned Hadamard matrix maps a vector's mean onto a single coordinate (its first column
is constant for Sylvester matrices and nearly so for Paley ones). After GELU, ViT MLP activations are
mostly positive, so a plain rotation *creates* one huge outlier per token instead of removing them; a
fact-check measured 4-bit error 0.31 vs 0.11 with signs on a 768-wide GELU input. Flipping a fixed random
set of input signs first (``random_signs=True``, the randomized Hadamard of QuaRot, arXiv 2404.00456)
removes the problem and keeps the map an exact rotation. BitNet v2 uses an unsigned matrix because its
LLM inputs are roughly zero-mean.

Widths with no such decomposition fall back to a fixed, seeded random orthogonal matrix (with a warning).
"""

from __future__ import annotations

import contextlib
import functools
import math
import random
import warnings

import torch
from torch import Tensor, nn

__all__ = [
    "sylvester_hadamard",
    "paley_hadamard",
    "hadamard_factors",
    "hadamard_matrix",
    "fixed_random_signs",
    "HadamardTransform",
]

# Largest Sylvester factor applied as one matmul. Small factors keep the cost near O(n log n).
_MAX_POW2_FACTOR = 32


def sylvester_hadamard(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Unnormalized Sylvester Hadamard matrix of size n (a power of two): H_2n = [[H, H], [H, -H]]."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Sylvester Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def _is_prime(q: int) -> bool:
    if q < 2:
        return False
    return all(q % d for d in range(2, int(q**0.5) + 1))


def paley_hadamard(m: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Unnormalized Hadamard matrix of order m = q + 1 for a prime q = 3 (mod 4) (Paley construction I).

    With the quadratic character chi mod q and Jacobsthal matrix ``Q[i, j] = chi(j - i)``, the matrix
    ``H = I + [[0, 1^T], [-1, Q]]`` satisfies ``H @ H.T = m * I``.
    """
    q = m - 1
    if not (_is_prime(q) and q % 4 == 3):
        raise ValueError(f"Paley construction I needs m - 1 to be a prime = 3 mod 4, got m={m}")
    residues = {(k * k) % q for k in range(1, q)}
    chi = [0] + [1 if a in residues else -1 for a in range(1, q)]
    jac = torch.tensor([[chi[(j - i) % q] for j in range(q)] for i in range(q)], dtype=dtype)
    s = torch.zeros(m, m, dtype=dtype)
    s[0, 1:] = 1
    s[1:, 0] = -1
    s[1:, 1:] = jac
    return torch.eye(m, dtype=dtype) + s


def _pow2_split(p: int) -> list[int]:
    """Split 2^k into as few, as equal as possible power-of-two factors no larger than _MAX_POW2_FACTOR."""
    k = p.bit_length() - 1
    if k == 0:
        return []
    max_bits = _MAX_POW2_FACTOR.bit_length() - 1
    chunks = -(-k // max_bits)
    bits = [k // chunks + (1 if i < k % chunks else 0) for i in range(chunks)]
    return [1 << b for b in bits]


@functools.cache
def hadamard_factors(n: int) -> tuple[int, ...] | None:
    """Factor sizes whose Kronecker product is an exact n x n Hadamard matrix, or None if unsupported.

    The first factor may be a Paley order (e.g. 12); the rest are powers of two.
    """
    if n < 1:
        raise ValueError(f"width must be positive, got {n}")
    pow2 = n & -n  # largest power of two dividing n
    # Try the largest power of two first so the non-power-of-two Paley factor stays as small as possible.
    p = pow2
    while p >= 1:
        m = n // p
        if m == 1:
            return tuple(_pow2_split(p)) or (1,)
        try:
            paley_hadamard(m)
        except ValueError:
            pass
        else:
            return (m, *_pow2_split(p))
        p //= 2
    return None


def _factor_matrix(size: int) -> Tensor:
    if size & (size - 1) == 0:
        return sylvester_hadamard(size)
    return paley_hadamard(size)


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Dense *orthonormal* n x n Hadamard matrix (``H @ H.T = I``). Mainly for tests and reference."""
    factors = hadamard_factors(n)
    if factors is None:
        raise ValueError(f"no exact Hadamard construction for width {n}")
    h = torch.ones(1, 1, dtype=torch.float64)
    for f in factors:
        h = torch.kron(h, _factor_matrix(f))
    return (h / math.sqrt(n)).to(dtype)


def autocast_disabled(device_type: str):
    """Context manager that turns autocast off on devices that support it (no-op elsewhere, e.g. meta)."""
    if device_type in ("cpu", "cuda", "xpu"):
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def fixed_random_signs(n: int, seed: int = 0) -> Tensor:
    """A fixed vector of n random +-1 values (float64). Uses Python's ``random`` so it never depends on the
    torch version or device, which keeps exported models reproducible."""
    rng = random.Random(seed)
    return torch.tensor([1.0 if rng.getrandbits(1) else -1.0 for _ in range(n)], dtype=torch.float64)


class HadamardTransform(nn.Module):
    """Computes ``x @ R`` along the last dimension, where R is an exact orthonormal n x n rotation.

    ``R = H`` (a Hadamard matrix scaled by 1/sqrt(n)), or ``R = diag(s) @ H`` with fixed random signs ``s``
    when ``random_signs=True``. The transform always runs in float32 (or float64 for float64 inputs), also
    under autocast, and returns the input dtype: running the matmuls in bfloat16 adds error comparable to
    the 8-bit rounding that follows (and flips many 4-bit codes).

    The buffers are not saved in checkpoints; they are rebuilt from exact float64 copies whenever the
    module is created, moved or cast, so half-precision casts and ``to_empty`` never degrade the rotation.
    """

    def __init__(self, n: int, seed: int = 0, random_signs: bool = False) -> None:
        super().__init__()
        self.n = int(n)
        self.seed = int(seed)
        self.random_signs = bool(random_signs)
        with torch.device("cpu"):  # never let a default or meta device context capture the exact values
            factors = hadamard_factors(self.n)
            if factors is None:
                warnings.warn(
                    f"width {self.n} has no exact Hadamard construction; using a fixed random orthogonal "
                    "matrix instead (dense, O(n^2) per token). Prefer widths of the form 2^k, 12*2^k or "
                    "20*2^k.",
                    stacklevel=2,
                )
                g = torch.Generator().manual_seed(seed)
                q, r = torch.linalg.qr(torch.randn(self.n, self.n, generator=g, dtype=torch.float64))
                mats = [q * torch.sign(torch.diagonal(r)).unsqueeze(0)]
                self.exact = False
            else:
                mats = [_factor_matrix(f) / math.sqrt(f) for f in factors]
                self.exact = True
            self._exact: dict[str, Tensor] = {f"factor{i}": m for i, m in enumerate(mats)}
            if self.random_signs:
                self._exact["signs"] = fixed_random_signs(self.n, self.seed)
        self.factor_sizes = tuple(m.shape[0] for m in mats)
        self._num_factors = len(mats)
        for name, exact in self._exact.items():
            self.register_buffer(name, exact.to(torch.float32), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # Rewrite every buffer from its exact copy on the buffer's new device: float64 if the module was cast
        # to float64, float32 otherwise (never the rounded half-precision values).
        for name, exact in self._exact.items():
            buf = getattr(self, name)
            if buf.device.type == "meta":
                continue
            if buf.dtype in (torch.float32, torch.float64):
                buf.copy_(exact)  # in place, so shared or pinned memory stays shared or pinned
            else:
                setattr(self, name, exact.to(device=buf.device, dtype=torch.float32))
        return self

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n:
            raise ValueError(f"expected last dim {self.n}, got {x.shape[-1]}")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        def buffer(name: str) -> Tensor:
            buf = getattr(self, name)
            if dtype == torch.float64 and buf.dtype != torch.float64:
                return self._exact[name].to(x.device)  # exact values for float64 inputs (e.g. gradchecks)
            return buf.to(device=x.device, dtype=dtype)

        lead = x.shape[:-1]
        k = self._num_factors
        with autocast_disabled(x.device.type):
            y = x.to(dtype)
            if self.random_signs:
                y = y * buffer("signs")
            y = y.reshape(*lead, *self.factor_sizes)
            for i in range(k):
                # Contract axis i of the factor grid: together this computes x @ kron(m_0, ..., m_{k-1}).
                y = torch.movedim(torch.movedim(y, -k + i, -1) @ buffer(f"factor{i}"), -1, -k + i)
        return y.reshape(*lead, self.n).to(x.dtype)

    @torch.no_grad()
    def matrix(self) -> Tensor:
        """The dense float64 rotation R (``forward(x) == x @ R``). For tests and reference."""
        return self(torch.eye(self.n, dtype=torch.float64))

    def extra_repr(self) -> str:
        return f"n={self.n}, factors={self.factor_sizes}, exact={self.exact}, random_signs={self.random_signs}"
