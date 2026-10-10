import warnings

import pytest
import torch

from bitnet_vision.hadamard import (
    HadamardTransform,
    fixed_random_signs,
    hadamard_factors,
    hadamard_matrix,
    paley_hadamard,
    sylvester_hadamard,
)

VIT_WIDTHS = [192, 384, 768, 1536, 3072, 640, 1280, 64, 1024]


@pytest.mark.parametrize("m", [4, 12, 20, 24, 44, 72])
def test_paley_matrices_are_hadamard(m):
    h = paley_hadamard(m)
    assert set(h.unique().tolist()) == {-1.0, 1.0}
    torch.testing.assert_close(h @ h.T, m * torch.eye(m, dtype=h.dtype))


def test_sylvester_matches_known_h4():
    expected = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float64)
    assert torch.equal(sylvester_hadamard(4), expected)
    with pytest.raises(ValueError):
        sylvester_hadamard(12)


@pytest.mark.parametrize("n", VIT_WIDTHS)
def test_vit_widths_get_exact_orthonormal_rotation(n):
    assert hadamard_factors(n) is not None
    h = hadamard_matrix(n)
    torch.testing.assert_close(h @ h.T, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(h.abs(), torch.full_like(h, n**-0.5))  # a true Hadamard matrix, entries +-1/sqrt(n)


@pytest.mark.parametrize("n", VIT_WIDTHS)
def test_fast_transform_matches_dense_matrix_and_preserves_norm(n):
    t = HadamardTransform(n)
    assert t.exact
    x = torch.randn(2, 3, n, dtype=torch.float64)
    y = t(x)
    torch.testing.assert_close(y, x @ hadamard_matrix(n), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_transform_is_invertible_by_its_transpose():
    n = 768
    t = HadamardTransform(n)
    x = torch.randn(4, n, dtype=torch.float64)
    torch.testing.assert_close(t(x) @ hadamard_matrix(n).T, x)


def test_shapes_dtypes_and_noncontiguous():
    t = HadamardTransform(192)
    assert t(torch.randn(192)).shape == (192,)
    assert t(torch.randn(2, 3, 4, 192)).shape == (2, 3, 4, 192)
    assert t(torch.randn(0, 192)).shape == (0, 192)
    x = torch.randn(192, 7).t()
    torch.testing.assert_close(t(x), t(x.contiguous()))
    tb = HadamardTransform(192).to(torch.bfloat16)
    assert tb.factor0.dtype == torch.float32  # factors stay float32 under half casting
    xb = torch.randn(5, 192, dtype=torch.bfloat16)
    assert tb(xb).dtype == torch.bfloat16
    torch.testing.assert_close(tb(xb).float(), t(xb.float()), atol=2e-2, rtol=2e-2)
    with pytest.raises(ValueError):
        t(torch.randn(3, 100))


def test_gradcheck():
    x = torch.randn(3, 96, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(HadamardTransform(96), (x,))


def test_unsupported_width_falls_back_to_orthogonal_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = HadamardTransform(100)
    assert not t.exact and any("no exact Hadamard" in str(w.message) for w in caught)
    x = torch.randn(4, 100, dtype=torch.float64)
    torch.testing.assert_close(t(x).norm(dim=-1), x.norm(dim=-1))


def test_transform_spreads_an_outlier():
    x = torch.zeros(1, 768)
    x[0, 5] = 100.0
    y = HadamardTransform(768)(x)
    assert y.abs().max() < 100.0 / 768**0.5 + 1e-3


@pytest.mark.parametrize("n", [192, 768, 3072])
def test_random_signs_keep_an_exact_rotation(n):
    t = HadamardTransform(n, random_signs=True)
    r = t.matrix()
    torch.testing.assert_close(r @ r.T, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=0)
    expected = torch.diag(fixed_random_signs(n)) @ hadamard_matrix(n)
    torch.testing.assert_close(r, expected, atol=1e-12, rtol=0)
    assert torch.equal(HadamardTransform(n, random_signs=True).signs, t.signs)  # deterministic


def test_random_signs_stop_positive_inputs_from_piling_onto_one_coordinate():
    """GELU outputs are mostly positive; an unsigned Hadamard turns their mean into one huge outlier."""
    torch.manual_seed(0)
    x = torch.nn.functional.gelu(torch.randn(256, 768))

    def crest(y):  # largest value relative to the RMS, per token (what absmax quantization pays for)
        return (y.abs().amax(-1) / y.pow(2).mean(-1).sqrt()).mean().item()

    plain, signed = crest(HadamardTransform(768)(x)), crest(HadamardTransform(768, random_signs=True)(x))
    assert plain > 2 * signed
    assert signed < crest(x)
