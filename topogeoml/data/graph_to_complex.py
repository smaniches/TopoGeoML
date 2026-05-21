"""
Graph → clique-complex lift (item 5 of the v0.0.1 scope lock).

Given a `networkx.Graph` or a square adjacency matrix, build the simplicial
complex whose simplices are the cliques of the graph (the *clique complex*,
sometimes called the flag complex). Clique dimension is capped at
`max_dim` to bound enumeration cost — `networkx.find_cliques` (Bron–Kerbosch)
is exponential in the worst case and must always be capped in production.

Mathematical contract
---------------------
The clique complex X(G) of a graph G is the abstract simplicial complex whose
k-simplices are the (k+1)-cliques of G. Capped at `max_dim`, we keep cliques
of size up to `max_dim + 1` vertices.

Output complexes always satisfy the chain identity ∂_{k-1} ∘ ∂_k = 0
(verified by `is_chain_complex`).
"""

from __future__ import annotations

import itertools
from typing import TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from topogeoml.core.complexes import SimplicialComplex

GraphLike: TypeAlias = nx.Graph | NDArray[np.floating]


def graph_to_clique_complex(
    graph: GraphLike,
    max_dim: int,
    include_isolated_vertices: bool = True,
) -> SimplicialComplex:
    """
    Lift a graph to its clique complex, capped at `max_dim`.

    Parameters
    ----------
    graph : networkx.Graph or square ndarray
        Input graph. If an ndarray is passed, it is interpreted as an
        adjacency matrix; any nonzero entry counts as an edge.
    max_dim : int
        Maximum simplex dimension. Cliques with more than ``max_dim + 1``
        vertices are decomposed into their ``(max_dim + 1)``-vertex sub-cliques
        before closure.
    include_isolated_vertices : bool
        If True (default), vertices with no incident edges still appear as
        0-simplices. If False, they are dropped from the complex.

    Returns
    -------
    SimplicialComplex
        Auto-closed under faces; satisfies ∂² = 0.

    Raises
    ------
    ValueError
        If ``max_dim < 0`` or if `graph` is an ndarray that is not square.
    TypeError
        If `graph` is neither an `nx.Graph` nor an `ndarray`.
    """
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")

    g = _coerce_to_graph(graph)

    # ``networkx.find_cliques`` emits a 1-clique iff the vertex is isolated
    # (degree 0). The pre-built ``isolated`` set + post-loop filter were
    # therefore both redundant work; the same decision is made inline by
    # checking ``len(clique) == 1``. Caught by Gemini PR #1 review.
    facets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for clique in nx.find_cliques(g):
        if not include_isolated_vertices and len(clique) == 1:
            continue
        verts = sorted(int(v) for v in clique)
        if len(verts) <= max_dim + 1:
            key = tuple(verts)
            if key not in seen:
                facets.append(key)
                seen.add(key)
        else:
            for sub in itertools.combinations(verts, max_dim + 1):
                if sub not in seen:
                    facets.append(sub)
                    seen.add(sub)

    return SimplicialComplex(facets=facets)


def _coerce_to_graph(graph: GraphLike) -> nx.Graph:
    """Accept either a networkx graph or a square adjacency ndarray."""
    if isinstance(graph, nx.Graph):
        return graph
    if isinstance(graph, np.ndarray):
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError(
                f"adjacency matrix must be square, got shape {graph.shape}"
            )
        return nx.from_numpy_array(graph)
    raise TypeError(
        f"expected networkx.Graph or square ndarray, got {type(graph).__name__}"
    )
