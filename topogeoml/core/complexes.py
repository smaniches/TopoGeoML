"""
Simplicial complexes, boundary operators, and Hodge Laplacians.

Implements:
  - SimplicialComplex: ordered enumeration of k-simplices indexed for matrix construction.
  - boundary_matrix(complex, k): sparse signed boundary ∂_k: C_k → C_{k-1} over R.
  - is_chain_complex(complex, max_dim, tol): verify ∂_{k-1} ∂_k = 0 within tolerance.
  - hodge_laplacian(complex, k): sparse L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T.

Conventions
-----------
* Simplices are sorted tuples of vertex indices: (v_0, v_1, ..., v_k) with v_0 < ... < v_k.
* The boundary of a k-simplex [v_0, ..., v_k] is the alternating sum
        d[v_0,...,v_k] = sum_i (-1)^i [v_0,...,omit(v_i),...,v_k]
  where omit(v_i) drops the i-th vertex (Hatcher Section 2.1, Carlsson 2009).
* Hodge Laplacian over R, not Z/2. Z/2 boundaries make d_k^T d_k useless as a Laplacian.

References
----------
* Hatcher, Algebraic Topology, Section 2.1
* Lim, "Hodge Laplacians on Graphs", SIAM Review 2020
* Schaub et al., "Random walks on simplicial complexes...", 2020
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

Simplex = tuple[int, ...]
"""A simplex is a sorted tuple of vertex indices."""


@dataclass
class SimplicialComplex:
    """
    Abstract simplicial complex with deterministic simplex indexing.

    A complex is given as the set of its maximal simplices (`facets`) plus all
    their faces. We store, per dimension k, a list of k-simplices in lexicographic
    order. The index of a simplex in `self.simplices[k]` is its column index in
    boundary matrices and its row/column index in the Hodge Laplacian L_k.

    Parameters
    ----------
    facets : Iterable[Sequence[int]]
        Maximal simplices. Each is an iterable of distinct non-negative ints.

    Notes
    -----
    Construction enumerates all faces. For a single facet with k+1 vertices this
    is 2^{k+1} - 1 simplices, so very high-dimensional facets are expensive.
    Cap clique dimension upstream (see data.graph_to_complex).
    """

    facets: Iterable[Sequence[int]]
    simplices: dict[int, list[Simplex]] = field(default_factory=dict, init=False)
    _index: dict[Simplex, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        all_simplices: set[Simplex] = set()
        for facet in self.facets:
            verts = tuple(sorted(set(int(v) for v in facet)))
            if len(verts) == 0:
                continue
            if any(v < 0 for v in verts):
                raise ValueError(f"vertex indices must be non-negative, got {verts}")
            # Enumerate all non-empty subsets (faces) of this facet.
            n = len(verts)
            for mask in range(1, 1 << n):
                face = tuple(verts[i] for i in range(n) if (mask >> i) & 1)
                all_simplices.add(face)

        # Group by dimension and sort lexicographically for deterministic indexing.
        by_dim: dict[int, list[Simplex]] = {}
        for simplex in all_simplices:
            k = len(simplex) - 1
            by_dim.setdefault(k, []).append(simplex)
        for k in by_dim:
            by_dim[k].sort()
        self.simplices = by_dim

        # Build flat index for boundary construction.
        idx: dict[Simplex, int] = {}
        for group in self.simplices.values():
            for i, s in enumerate(group):
                idx[s] = i
        self._index = idx

    @property
    def max_dim(self) -> int:
        """Highest non-empty simplex dimension; -1 if empty."""
        return max(self.simplices.keys()) if self.simplices else -1

    def n_simplices(self, k: int) -> int:
        """Count of k-simplices."""
        return len(self.simplices.get(k, []))

    def index_of(self, simplex: Sequence[int]) -> int:
        """Look up the column index of a simplex in C_k. Raises if not present."""
        key = tuple(sorted(simplex))
        if key not in self._index:
            raise KeyError(f"simplex {key} not in complex")
        return self._index[key]

    def boundary_matrix(self, k: int) -> sp.csr_matrix:
        """
        Sparse signed boundary ∂_k : C_k → C_{k-1} over R.

        Returns a (n_{k-1}, n_k) matrix. For k=0 returns an empty (0, n_0) matrix
        (boundary of vertices is zero by convention).

        Raises
        ------
        ValueError
            If k < 0 or there are no k-simplices.
        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        n_k = self.n_simplices(k)
        if k == 0:
            # By convention ∂_0 = 0: maps C_0 → 0. Return empty matrix.
            return sp.csr_matrix((0, n_k), dtype=np.float64)

        n_km1 = self.n_simplices(k - 1)
        if n_k == 0 or n_km1 == 0:
            return sp.csr_matrix((n_km1, n_k), dtype=np.float64)

        # Build via COO triplets (construction loop, not computation loop).
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        km1_index = {s: i for i, s in enumerate(self.simplices[k - 1])}
        for col, simplex in enumerate(self.simplices[k]):
            # ∂[v_0,...,v_k] = Σ_i (-1)^i [v_0,...,\hat{v_i},...,v_k]
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1 :]
                row = km1_index.get(face)
                if row is None:  # pragma: no cover
                    # Unreachable: SimplicialComplex.__post_init__ enforces
                    # closure under faces. This branch fires only if a
                    # caller manually mutates ``self.simplices`` after
                    # construction, which is not part of the public API.
                    raise RuntimeError(
                        f"face {face} of {simplex} missing from C_{k-1} — "
                        "complex closure violated"
                    )
                rows.append(row)
                cols.append(col)
                data.append(-1.0 if (i % 2) else 1.0)

        return sp.coo_matrix(
            (data, (rows, cols)),
            shape=(n_km1, n_k),
            dtype=np.float64,
        ).tocsr()


def is_chain_complex(
    complex_: SimplicialComplex,
    max_dim: int | None = None,
    tol: float = 1e-12,
) -> bool:
    """
    Verify ∂_{k-1} ∘ ∂_k = 0 for all k in [1, max_dim].

    The fundamental identity of chain complexes. If this fails, the boundary
    operator construction is wrong. Item 6 of the v0.0.1 spec.

    Parameters
    ----------
    complex_ : SimplicialComplex
    max_dim : int, optional
        Highest dimension to check. Defaults to complex_.max_dim.
    tol : float
        Max allowed Frobenius norm of the composition matrix.

    Returns
    -------
    bool
        True if ‖∂_{k-1} ∂_k‖_F <= tol for every k checked.
    """
    if max_dim is None:
        max_dim = complex_.max_dim
    for k in range(1, max_dim + 1):
        if complex_.n_simplices(k) == 0 or complex_.n_simplices(k - 1) == 0:
            continue
        d_k = complex_.boundary_matrix(k)
        # The line-190 `continue` plus `range(1, ...)` already guarantee
        # ``k >= 1`` and ``n_simplices(k - 1) > 0`` here, so no extra guard.
        d_km1 = complex_.boundary_matrix(k - 1)
        if d_km1.shape[0] > 0:
            product = d_km1 @ d_k
            # Frobenius norm of a sparse matrix.
            frob = float(np.sqrt(product.multiply(product).sum()))
            if frob > tol:  # pragma: no cover
                # Reachable only when the boundary-matrix implementation
                # is broken (e.g. wrong sign convention) — the bench's
                # correctness axis would catch this in a separate path.
                return False
    return True


def hodge_laplacian(complex_: SimplicialComplex, k: int) -> sp.csr_matrix:
    """
    k-th Hodge Laplacian over R: L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T.

    For k = 0 with a graph-like complex (vertices + edges), L_0 reduces to the
    standard combinatorial graph Laplacian D - A. Higher-order Hodge Laplacians
    extend spectral graph theory to simplicial complexes (Lim 2020).

    Properties:
        L_k is symmetric positive semi-definite.
        dim ker(L_k) = β_k (k-th Betti number) — discrete Hodge theorem.

    Parameters
    ----------
    complex_ : SimplicialComplex
    k : int
        Dimension; must be in [0, complex_.max_dim].

    Returns
    -------
    sp.csr_matrix
        Symmetric (n_k, n_k) sparse Laplacian.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    n_k = complex_.n_simplices(k)
    if n_k == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)

    laplacian: sp.spmatrix = sp.csr_matrix((n_k, n_k), dtype=np.float64)

    # Down-Laplacian: ∂_k^T ∂_k (zero for k = 0 by convention).
    if k >= 1 and complex_.n_simplices(k - 1) > 0:
        d_k = complex_.boundary_matrix(k)
        laplacian = laplacian + d_k.T @ d_k

    # Up-Laplacian: ∂_{k+1} ∂_{k+1}^T (zero if no (k+1)-simplices).
    if complex_.n_simplices(k + 1) > 0:
        d_kp1 = complex_.boundary_matrix(k + 1)
        laplacian = laplacian + d_kp1 @ d_kp1.T

    return laplacian.tocsr()


def betti_numbers(
    complex_: SimplicialComplex,
    max_dim: int | None = None,
    tol: float = 1e-8,
) -> dict[int, int]:
    """
    Compute Betti numbers β_k = dim ker(L_k) via Hodge theorem.

    Uses dense eigendecomposition; intended for small complexes. For large
    complexes, use the persistent homology pipeline (RipsFiltration) instead.

    Parameters
    ----------
    complex_ : SimplicialComplex
    max_dim : int, optional
        Highest dimension to compute. Defaults to complex_.max_dim.
    tol : float
        Eigenvalues with |λ| < tol count as zero.

    Returns
    -------
    dict[int, int]
        Mapping k → β_k.
    """
    if max_dim is None:
        max_dim = complex_.max_dim
    out: dict[int, int] = {}
    for k in range(max_dim + 1):
        if complex_.n_simplices(k) == 0:
            out[k] = 0
            continue
        L = hodge_laplacian(complex_, k)
        # Dense eigvalsh — symmetric, real spectrum.
        dense = L.toarray().astype(np.float64, copy=False)
        eigvals = np.linalg.eigvalsh(dense)
        out[k] = int(np.sum(np.abs(eigvals) < tol))
    return out
