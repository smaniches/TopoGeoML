"""
Tests for SimplicialComplex, boundary operators, Hodge Laplacian, Betti numbers.

Validates against canonical examples with known homology:
  - Triangle boundary (S^1): β_0=1, β_1=1
  - Filled triangle (D^2): β_0=1, β_1=0
  - Two disjoint vertices: β_0=2
  - Tetrahedron boundary (S^2): β_0=1, β_1=0, β_2=1
"""

from __future__ import annotations

import numpy as np
import pytest

from topogeoml.core.complexes import (
    SimplicialComplex,
    betti_numbers,
    hodge_laplacian,
    is_chain_complex,
)


# ---------- Construction ----------


def test_simplicial_complex_from_single_triangle() -> None:
    """Filled triangle {0,1,2}: 3 vertices, 3 edges, 1 face."""
    sc = SimplicialComplex(facets=[(0, 1, 2)])
    assert sc.n_simplices(0) == 3
    assert sc.n_simplices(1) == 3
    assert sc.n_simplices(2) == 1
    assert sc.max_dim == 2


def test_simplicial_complex_triangle_boundary() -> None:
    """S^1 as triangle boundary: 3 vertices, 3 edges, no 2-face."""
    sc = SimplicialComplex(facets=[(0, 1), (1, 2), (0, 2)])
    assert sc.n_simplices(0) == 3
    assert sc.n_simplices(1) == 3
    assert sc.n_simplices(2) == 0


def test_simplicial_complex_tetrahedron() -> None:
    """Filled tetrahedron: 4 + 6 + 4 + 1 simplices."""
    sc = SimplicialComplex(facets=[(0, 1, 2, 3)])
    assert sc.n_simplices(0) == 4
    assert sc.n_simplices(1) == 6
    assert sc.n_simplices(2) == 4
    assert sc.n_simplices(3) == 1


def test_simplicial_complex_rejects_negative_vertices() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SimplicialComplex(facets=[(-1, 0, 1)])


def test_simplicial_complex_index_lookup() -> None:
    sc = SimplicialComplex(facets=[(0, 1, 2)])
    # Vertex indices are 0, 1, 2 in sorted order.
    assert sc.index_of((0,)) == 0
    assert sc.index_of((1,)) == 1
    assert sc.index_of((2,)) == 2
    # Edge (0,1) should be the first edge in lex order.
    assert sc.index_of((0, 1)) == 0
    with pytest.raises(KeyError):
        sc.index_of((5,))


# ---------- Boundary operators ----------


def test_boundary_d0_is_empty() -> None:
    """∂_0 maps C_0 → 0 by convention."""
    sc = SimplicialComplex(facets=[(0, 1, 2)])
    d0 = sc.boundary_matrix(0)
    assert d0.shape == (0, 3)


def test_boundary_d1_correct_signs() -> None:
    """∂(v_0, v_1) = v_1 - v_0."""
    sc = SimplicialComplex(facets=[(0, 1)])
    d1 = sc.boundary_matrix(1).toarray()
    # Edge (0,1) has column. Vertices 0, 1 are rows.
    # ∂[0,1] = -[0] + [1].
    assert d1.shape == (2, 1)
    np.testing.assert_allclose(d1[:, 0], [-1.0, 1.0])


def test_boundary_d2_correct_signs() -> None:
    """∂(v_0, v_1, v_2) = (v_1,v_2) - (v_0,v_2) + (v_0,v_1)."""
    sc = SimplicialComplex(facets=[(0, 1, 2)])
    d2 = sc.boundary_matrix(2).toarray()
    # Edges in lex order: (0,1)=0, (0,2)=1, (1,2)=2.
    # Triangle (0,1,2): ∂ = +(1,2) - (0,2) + (0,1).
    assert d2.shape == (3, 1)
    np.testing.assert_allclose(d2[:, 0], [1.0, -1.0, 1.0])


def test_chain_complex_identity_filled_triangle() -> None:
    """∂_1 ∂_2 = 0 for the filled triangle."""
    sc = SimplicialComplex(facets=[(0, 1, 2)])
    assert is_chain_complex(sc)


def test_chain_complex_identity_tetrahedron() -> None:
    """∂² = 0 in all dimensions for a tetrahedron."""
    sc = SimplicialComplex(facets=[(0, 1, 2, 3)])
    assert is_chain_complex(sc)


def test_chain_complex_identity_two_triangles_sharing_edge() -> None:
    """Two filled triangles glued along an edge."""
    sc = SimplicialComplex(facets=[(0, 1, 2), (1, 2, 3)])
    assert is_chain_complex(sc)


# ---------- Hodge Laplacian ----------


def test_hodge_laplacian_symmetric_psd() -> None:
    """L_k is symmetric positive semi-definite."""
    sc = SimplicialComplex(facets=[(0, 1, 2)])
    for k in range(3):
        L = hodge_laplacian(sc, k).toarray()
        # Symmetric.
        np.testing.assert_allclose(L, L.T, atol=1e-12)
        # PSD: all eigenvalues >= -tol.
        eigvals = np.linalg.eigvalsh(L)
        assert eigvals.min() >= -1e-10, f"L_{k} has negative eigenvalue {eigvals.min()}"


def test_hodge_laplacian_l0_equals_graph_laplacian_on_path() -> None:
    """L_0 of a 3-vertex path graph is the standard graph Laplacian."""
    # Path 0-1-2: edges (0,1), (1,2).
    sc = SimplicialComplex(facets=[(0, 1), (1, 2)])
    L0 = hodge_laplacian(sc, 0).toarray()
    expected = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(L0, expected)


# ---------- Betti numbers via Hodge theorem ----------


def test_betti_numbers_filled_triangle() -> None:
    """Filled triangle ≃ disk: β = (1, 0, 0)."""
    sc = SimplicialComplex(facets=[(0, 1, 2)])
    b = betti_numbers(sc, max_dim=2)
    assert b[0] == 1
    assert b[1] == 0
    assert b[2] == 0


def test_betti_numbers_triangle_boundary() -> None:
    """Triangle boundary ≃ S^1: β = (1, 1)."""
    sc = SimplicialComplex(facets=[(0, 1), (1, 2), (0, 2)])
    b = betti_numbers(sc, max_dim=1)
    assert b[0] == 1
    assert b[1] == 1


def test_betti_numbers_tetrahedron_boundary() -> None:
    """Tetrahedron boundary ≃ S^2: β = (1, 0, 1)."""
    # All four 2-faces but no 3-cell.
    sc = SimplicialComplex(
        facets=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    )
    b = betti_numbers(sc, max_dim=2)
    assert b[0] == 1
    assert b[1] == 0
    assert b[2] == 1


def test_betti_numbers_disjoint_vertices() -> None:
    """Three disjoint vertices: β_0 = 3."""
    sc = SimplicialComplex(facets=[(0,), (1,), (2,)])
    b = betti_numbers(sc, max_dim=0)
    assert b[0] == 3


def test_betti_numbers_two_disjoint_triangles() -> None:
    """Two disjoint filled triangles: β_0 = 2, β_1 = 0."""
    sc = SimplicialComplex(facets=[(0, 1, 2), (3, 4, 5)])
    b = betti_numbers(sc, max_dim=1)
    assert b[0] == 2
    assert b[1] == 0
