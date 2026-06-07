"""
Tests for HodgeMessagePassing (item 8).

Marked with @pytest.mark.torch; skipped if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

# Imports below require torch — gated by importorskip.
from topogeoml.core.complexes import SimplicialComplex, hodge_laplacian
from topogeoml.nn.hodge import (
    HodgeMessagePassing,
    build_hodge_layer_from_complex,
    normalize_hodge_laplacian,
    sparse_scipy_to_torch,
)

pytestmark = pytest.mark.torch


def _make_triangle() -> SimplicialComplex:
    """Filled triangle: 3 vertices, 3 edges, 1 face."""
    return SimplicialComplex(facets=[(0, 1, 2)])


def test_normalize_hodge_laplacian_is_symmetric() -> None:
    sc = _make_triangle()
    L = hodge_laplacian(sc, 0)
    L_norm = normalize_hodge_laplacian(L).toarray()
    np.testing.assert_allclose(L_norm, L_norm.T, atol=1e-12)


def test_sparse_scipy_to_torch_round_trip() -> None:
    """Converted torch sparse tensor should preserve values.

    The float32 path round-trips to float32 precision; the float64 path must
    preserve full precision with NO intermediate float32 truncation. The float64
    case uses a value not representable in float32 (1.0 + 1e-12) and asserts
    exact equality (atol=0), so it fails if a float32 intermediate is reintroduced.
    """
    arr = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float64)
    sparse = sp.csr_matrix(arr)
    t32 = sparse_scipy_to_torch(sparse, dtype=torch.float32)
    np.testing.assert_allclose(t32.to_dense().numpy(), arr.astype(np.float32))

    # float64: full precision preserved, exactly.
    val = 1.0 + 1e-12  # not representable in float32; survives only in float64
    assert np.float32(val) == np.float32(1.0)  # confirms float32 would truncate
    arr64 = np.array([[val, 0.0], [0.0, np.pi]], dtype=np.float64)
    t64 = sparse_scipy_to_torch(sp.csr_matrix(arr64), dtype=torch.float64)
    assert t64.dtype == torch.float64
    np.testing.assert_array_equal(t64.to_dense().numpy(), arr64)


def test_hodge_layer_forward_shape() -> None:
    sc = _make_triangle()
    layer = build_hodge_layer_from_complex(
        sc, k=0, in_features=4, out_features=8
    )
    x = torch.randn(sc.n_simplices(0), 4)
    out = layer(x)
    assert out.shape == (3, 8)


def test_hodge_layer_forward_relu_nonnegative() -> None:
    """Default activation is ReLU; outputs should be non-negative."""
    sc = _make_triangle()
    layer = build_hodge_layer_from_complex(
        sc, k=0, in_features=4, out_features=4
    )
    x = torch.randn(sc.n_simplices(0), 4)
    out = layer(x)
    assert (out >= 0).all()


def test_hodge_layer_backward_propagates_gradient() -> None:
    """Gradient should flow back through the sparse Laplacian to inputs and weights."""
    sc = _make_triangle()
    layer = build_hodge_layer_from_complex(
        sc, k=0, in_features=4, out_features=4
    )
    x = torch.randn(sc.n_simplices(0), 4, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    # Gradient should be non-trivial (not all zero).
    assert torch.any(layer.weight.grad != 0)


def test_hodge_layer_rejects_wrong_input_shape() -> None:
    sc = _make_triangle()
    layer = build_hodge_layer_from_complex(sc, k=0, in_features=4, out_features=4)
    # Wrong number of simplices.
    with pytest.raises(ValueError, match="shape mismatch"):
        layer(torch.randn(5, 4))
    # Wrong feature dim.
    with pytest.raises(ValueError, match="features"):
        layer(torch.randn(sc.n_simplices(0), 7))
    # Wrong ndim.
    with pytest.raises(ValueError, match="2D"):
        layer(torch.randn(3))


def test_hodge_layer_no_bias_option() -> None:
    sc = _make_triangle()
    L = normalize_hodge_laplacian(hodge_laplacian(sc, 0))
    layer = HodgeMessagePassing(in_features=4, out_features=4, laplacian=L, bias=False)
    assert layer.bias is None
    x = torch.randn(sc.n_simplices(0), 4)
    out = layer(x)
    assert out.shape == (3, 4)


def test_hodge_layer_rejects_non_positive_features() -> None:
    sc = _make_triangle()
    L = normalize_hodge_laplacian(hodge_laplacian(sc, 0))
    with pytest.raises(ValueError, match="positive"):
        HodgeMessagePassing(in_features=0, out_features=4, laplacian=L)
    with pytest.raises(ValueError, match="positive"):
        HodgeMessagePassing(in_features=4, out_features=0, laplacian=L)


def test_hodge_layer_stack_two_layers() -> None:
    """Stack two layers to verify chaining works."""
    sc = _make_triangle()
    L = normalize_hodge_laplacian(hodge_laplacian(sc, 0))
    layer1 = HodgeMessagePassing(in_features=4, out_features=8, laplacian=L)
    layer2 = HodgeMessagePassing(in_features=8, out_features=2, laplacian=L)
    x = torch.randn(sc.n_simplices(0), 4)
    out = layer2(layer1(x))
    assert out.shape == (3, 2)
