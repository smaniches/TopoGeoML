"""Tests for graph_to_clique_complex (item 5)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from topogeoml.core.complexes import betti_numbers, is_chain_complex
from topogeoml.data.graph_to_complex import graph_to_clique_complex


def test_triangle_graph_to_clique_complex_dim_2() -> None:
    """K_3 with max_dim=2: 3 vertices, 3 edges, 1 triangle."""
    G = nx.complete_graph(3)
    sc = graph_to_clique_complex(G, max_dim=2)
    assert sc.n_simplices(0) == 3
    assert sc.n_simplices(1) == 3
    assert sc.n_simplices(2) == 1


def test_triangle_graph_with_max_dim_1() -> None:
    """K_3 with max_dim=1: triangle face is excluded."""
    G = nx.complete_graph(3)
    sc = graph_to_clique_complex(G, max_dim=1)
    assert sc.n_simplices(0) == 3
    assert sc.n_simplices(1) == 3
    assert sc.n_simplices(2) == 0


def test_cycle_graph_clique_complex() -> None:
    """C_4 (4-cycle): 4 vertices, 4 edges, no triangles."""
    G = nx.cycle_graph(4)
    sc = graph_to_clique_complex(G, max_dim=2)
    assert sc.n_simplices(0) == 4
    assert sc.n_simplices(1) == 4
    assert sc.n_simplices(2) == 0
    # Betti: connected (β_0=1) and one loop (β_1=1).
    b = betti_numbers(sc, max_dim=1)
    assert b[0] == 1
    assert b[1] == 1


def test_complete_graph_k4_clique_complex_dim_3() -> None:
    """K_4 with max_dim=3: 4 vertices, 6 edges, 4 triangles, 1 tetrahedron."""
    G = nx.complete_graph(4)
    sc = graph_to_clique_complex(G, max_dim=3)
    assert sc.n_simplices(0) == 4
    assert sc.n_simplices(1) == 6
    assert sc.n_simplices(2) == 4
    assert sc.n_simplices(3) == 1


def test_complete_graph_k4_capped_at_2() -> None:
    """K_4 with max_dim=2: tetrahedron excluded, but all 4 triangles included."""
    G = nx.complete_graph(4)
    sc = graph_to_clique_complex(G, max_dim=2)
    assert sc.n_simplices(0) == 4
    assert sc.n_simplices(1) == 6
    assert sc.n_simplices(2) == 4
    assert sc.n_simplices(3) == 0
    # K_4 with capped dimension is the boundary of a 3-simplex ≃ S^2: β=(1,0,1).
    b = betti_numbers(sc, max_dim=2)
    assert b[0] == 1
    assert b[1] == 0
    assert b[2] == 1


def test_adjacency_matrix_input() -> None:
    """Lift from a (n,n) adjacency matrix."""
    # Triangle adjacency.
    adj = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )
    sc = graph_to_clique_complex(adj, max_dim=2)
    assert sc.n_simplices(0) == 3
    assert sc.n_simplices(1) == 3
    assert sc.n_simplices(2) == 1


def test_isolated_vertices_included() -> None:
    """Two-vertex graph with no edge: two 0-simplices."""
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    sc = graph_to_clique_complex(G, max_dim=1)
    assert sc.n_simplices(0) == 2
    assert sc.n_simplices(1) == 0


def test_isolated_vertices_excluded() -> None:
    """include_isolated_vertices=False drops isolated 0-simplices."""
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(2, 3)
    sc = graph_to_clique_complex(G, max_dim=1, include_isolated_vertices=False)
    # Only the edge (2,3) and its vertices.
    assert sc.n_simplices(0) == 2
    assert sc.n_simplices(1) == 1


def test_clique_complex_satisfies_chain_identity() -> None:
    """Any clique complex must satisfy ∂² = 0."""
    G = nx.complete_graph(5)
    sc = graph_to_clique_complex(G, max_dim=3)
    assert is_chain_complex(sc)


def test_rejects_invalid_max_dim() -> None:
    with pytest.raises(ValueError, match="max_dim must be >= 0"):
        graph_to_clique_complex(nx.complete_graph(3), max_dim=-1)


def test_rejects_non_square_adjacency() -> None:
    with pytest.raises(ValueError, match="square"):
        graph_to_clique_complex(np.zeros((3, 4)), max_dim=1)
    # Regression: typing.get_type_hints must resolve without raising. The
    # GraphLike alias references nx.Graph[Any]; the runtime networkx.Graph
    # class is not subscriptable in current releases (e.g. 3.6.1), so a naive
    # string alias makes get_type_hints raise "TypeError: type 'Graph' is not
    # subscriptable". The TYPE_CHECKING / runtime alias split keeps mypy strict
    # happy while staying runtime-introspectable. Folded into this existing
    # test rather than a new one to preserve the documented test-function count.
    import typing

    from topogeoml.data import graph_to_complex as mod

    for fn in (mod.graph_to_clique_complex, mod._coerce_to_graph):
        assert "graph" in typing.get_type_hints(fn)
