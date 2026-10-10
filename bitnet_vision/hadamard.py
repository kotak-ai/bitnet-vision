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

Widths with no such decomposition fall back to a fixed, seeded random orthogonal matrix (with a warning).
"""

from __future__ import annotations

import functools
import math
import warnings

import torch
from torch import Tensor, nn

__all__ = [
    "sylvester_hadamard",
    "paley_hadamard",
    "hadamard_factors",
    "hadamard_matrix",
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


class HadamardTransform(nn.Module):
    """Computes ``x @ H`` along the last dimension, where H is an exact orthonormal n x n rotation.

    The transform always runs in float32 (or float64 for float64 inputs) and returns the input dtype:
    running each matmul in bfloat16 adds error comparable to the 8-bit rounding that follows.
    """

    def __init__(self, n: int, seed: int = 0) -> None:
        super().__init__()
        self.n = int(n)
        factors = hadamard_factors(self.n)
        if factors is None:
            warnings.warn(
                f"width {self.n} has no exact Hadamard construction; using a fixed random orthogonal "
                "matrix instead (dense, O(n^2) per token). Prefer widths of the form 2^k, 12*2^k or 20*2^k.",
                stacklevel=2,
            )
            g = torch.Generator().manual_seed(seed)
            q, r = torch.linalg.qr(torch.randn(self.n, self.n, generator=g, dtype=torch.float64))
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
            mats = [q]
            self.exact = False
        else:
            mats = [_factor_matrix(f) / math.sqrt(f) for f in factors]
            self.exact = True
        self.factor_sizes = tuple(m.shape[0] for m in mats)
        # Non-persistent: rebuilt deterministically, so checkpoints stay small and backward compatible.
        for i, m in enumerate(mats):
            self.register_buffer(f"factor{i}", m.to(torch.float32), persistent=False)
        self._num_factors = len(mats)
        self._mats64 = mats  # exact float64 copies (CPU), used for float64 inputs such as gradchecks

    def _apply(self, fn, recurse=True):  # keep factors in float32 when the module is cast to half precision
        super()._apply(fn, recurse)
        for i in range(self._num_factors):
            buf = getattr(self, f"factor{i}")
            if buf.dtype not in (torch.float32, torch.float64):
                setattr(self, f"factor{i}", buf.float())
        return self

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n:
            raise ValueError(f"expected last dim {self.n}, got {x.shape[-1]}")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        lead = x.shape[:-1]
        y = x.to(dtype).reshape(*lead, *self.factor_sizes)
        k = self._num_factors
        for i in range(k):
            if dtype == torch.float64:
                m = self._mats64[i].to(x.device)
            else:
                m = getattr(self, f"factor{i}").to(dtype)
            # Contract axis i of the factor grid with m: together this computes x @ kron(m_0, ..., m_{k-1}).
            y = torch.movedim(torch.movedim(y, -k + i, -1) @ m, -1, -k + i)
        return y.reshape(*lead, self.n).to(x.dtype)

    def extra_repr(self) -> str:
        return f"n={self.n}, factors={self.factor_sizes}, exact={self.exact}"
