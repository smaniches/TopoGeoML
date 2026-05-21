"""
Backend wrapper: Hofer et al. 2017 reference implementation in PyTorch.

The original reference (Hofer, Kwitt, Niethammer, Uhl, NeurIPS 2017) uses
explicit ``torch.autograd.Function`` semantics: the persistence forward
pass is a non-differentiable combinatorial computation (which point pair
realizes each bar?), and the backward pass projects upstream gradients
back to the input point cloud via the chain rule through the L_2 distance
formula on the responsible edges.

This wrapper reimplements that approach in modern PyTorch using gudhi's
``SimplexTree.flag_persistence_generators()`` to identify the critical
edges for every finite bar. The result is mathematically equivalent to
TopoGeoML's ``diff_ph`` for Vietoris-Rips persistence under the standard
subgradient convention (Hofer 2017 §4; Carrière 2021 §3), differing only
in which library performs the forward persistence computation.

Why this backend is interesting
-------------------------------
Bench-time comparison with ``topogeoml-diff-ph`` should show:

  - Identical persistence diagrams (both backed by Vietoris-Rips).
  - Identical gradients in expectation; differences should be at the level
    of floating-point round-off plus tie-breaking when multiple edges
    realize the same filtration value.

A persistent discrepancy between the two would flag either (a) a bug in
one of the backends, or (b) a non-trivial difference in subgradient
choice — both worth investigating.

References
----------
Hofer, C., Kwitt, R., Niethammer, M., & Uhl, A. (2017). "Deep Learning
  with Topological Signatures." *NeurIPS 2017*, 1633-1643.
Carrière, M., Chazal, F., Glisse, M., Ike, Y., Kannan, H., & Umeda, Y.
  (2021). "Optimizing persistent homology based functions."
  *ICML 2021*, 1294-1303.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
from numpy.typing import NDArray

from benchmarks.backends import register_backend


def _gudhi_finite_generators(
    pts: NDArray[np.float64], max_dim: int
) -> tuple[
    NDArray[np.int64],            # H_0 finite: (n_h0_finite, 3) [birth_vert, death_edge_a, death_edge_b]
    list[NDArray[np.int64]],      # H_k finite for k>=1: each (n_hk_finite, 4) [b_a, b_b, d_a, d_b]
    NDArray[np.int64],            # H_0 essential: (n_h0_inf,) [birth_vert]
]:
    """Wrap gudhi's ``flag_persistence_generators`` for ``max_dim+1``-vertex
    simplices and return the critical-simplex indices needed by
    :class:`_RipsPersistenceFunction`'s backward.
    """
    import gudhi

    if pts.shape[0] < 2:  # pragma: no cover  -- bench enforces n >= 4 upstream.
        return (
            np.zeros((0, 3), dtype=np.int64),
            [np.zeros((0, 4), dtype=np.int64) for _ in range(max_dim)],
            np.zeros((0,), dtype=np.int64),
        )

    from scipy.spatial.distance import pdist

    diameter = float(pdist(pts).max())
    # Vietoris-Rips persistence saturates at the diameter; the previous 5%
    # headroom was unnecessary (caught by Gemini PR #4 review).
    max_edge_length = max(diameter, 1e-10)

    rips = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    gens = st.flag_persistence_generators()

    h0_finite = np.asarray(gens[0], dtype=np.int64) if len(gens[0]) > 0 else np.zeros((0, 3), dtype=np.int64)
    hk_finite_list: list[NDArray[np.int64]] = []
    for k in range(max_dim):
        # gens[1] is a list indexed by (k - 1) for the H_k contents (k starts at 1).
        if k < len(gens[1]):
            arr = np.asarray(gens[1][k], dtype=np.int64)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                hk_finite_list.append(arr[:, :4])
            else:  # pragma: no cover  -- defensive against future gudhi schema changes.
                hk_finite_list.append(np.zeros((0, 4), dtype=np.int64))
        else:  # pragma: no cover
            hk_finite_list.append(np.zeros((0, 4), dtype=np.int64))

    h0_essential = np.asarray(gens[2], dtype=np.int64) if len(gens[2]) > 0 else np.zeros((0,), dtype=np.int64)
    return h0_finite, hk_finite_list, h0_essential


def _compute_rips_persistence(
    X: torch.Tensor, max_dim: int, include_essential: bool = True
) -> torch.Tensor:
    """Hofer 2017 differentiable Rips PH.

    Identifies critical edges via gudhi (non-differentiable combinatorial
    step), then rebuilds the bar values as L_2 distances on the input
    tensor — autograd handles the subgradient projection through the
    standard chain rule on these distances. Functionally equivalent to
    Hofer 2017 §4's explicit backward formula, with the gradient
    bookkeeping delegated to PyTorch.

    Returns a single ``(n_bars, 3)`` tensor with columns
    ``(birth, death, dim_marker)``; the caller splits by dimension via
    the third column.
    """
    pts = X.detach().cpu().numpy().astype(np.float64, copy=False)
    h0_fin, hk_fin_list, h0_ess = _gudhi_finite_generators(pts, max_dim)

    # H_0 finite: birth = 0; death = ||X[u] - X[v]||_2 where (u, v) is
    # the death-edge endpoints (columns 1 and 2).
    if h0_fin.shape[0] > 0:
        h0_death_edges = h0_fin[:, 1:3]
        d = X[h0_death_edges[:, 0]] - X[h0_death_edges[:, 1]]
        h0_deaths = torch.linalg.norm(d, dim=1)
        h0_births = torch.zeros_like(h0_deaths)
    else:  # pragma: no cover
        # Reachable only on a single-point cloud (no edges, no finite H_0).
        # The bench enforces n >= 4 upstream so this is defensive.
        h0_deaths = X.new_empty((0,))
        h0_births = X.new_empty((0,))

    # H_k finite for k >= 1: birth and death are both edge lengths.
    hk_births_list: list[torch.Tensor] = []
    hk_deaths_list: list[torch.Tensor] = []
    for arr in hk_fin_list:
        if arr.shape[0] > 0:
            bd = X[arr[:, 0]] - X[arr[:, 1]]
            dd = X[arr[:, 2]] - X[arr[:, 3]]
            hk_births_list.append(torch.linalg.norm(bd, dim=1))
            hk_deaths_list.append(torch.linalg.norm(dd, dim=1))
        else:
            hk_births_list.append(X.new_empty((0,)))
            hk_deaths_list.append(X.new_empty((0,)))

    # H_0 essential (one infinite class per connected component of the
    # graph at filtration max_edge_length; we emit `inf` for the death).
    if include_essential and h0_ess.shape[0] > 0:
        h0_ess_births = X.new_zeros((h0_ess.shape[0],))
        h0_ess_deaths = X.new_full((h0_ess.shape[0],), float("inf"))
    else:
        h0_ess_births = X.new_empty((0,))
        h0_ess_deaths = X.new_empty((0,))

    # Pack into a single (n_bars, 3) tensor with the dim marker.
    rows: list[torch.Tensor] = []
    # H_0 finite, marker 0
    if h0_births.shape[0] > 0:
        rows.append(torch.stack(
            [h0_births, h0_deaths, torch.zeros_like(h0_births)], dim=1
        ))
    # H_k finite for k >= 1, marker k
    for k_idx, (b, dth) in enumerate(zip(hk_births_list, hk_deaths_list, strict=True)):
        k_dim = k_idx + 1
        if b.shape[0] > 0:
            rows.append(torch.stack(
                [b, dth, torch.full_like(b, float(k_dim))], dim=1
            ))
    # H_0 essential, marker 0 (deaths are inf so callers can filter)
    if h0_ess_births.shape[0] > 0:
        rows.append(torch.stack(
            [h0_ess_births, h0_ess_deaths, torch.zeros_like(h0_ess_births)], dim=1
        ))

    if not rows:  # pragma: no cover
        # Reachable only on an empty point cloud (n == 0) — bench enforces
        # n >= 4 upstream so this branch is a safety net.
        return X.new_zeros((0, 3))
    return torch.cat(rows, dim=0)


@register_backend
class Hofer2017Reference:
    """PyTorch reimplementation of Hofer et al. 2017 backed by gudhi."""

    name: ClassVar[str] = "hofer-2017-reference"
    version: ClassVar[str] = "1.0.0"
    differentiable: ClassVar[bool] = True

    @staticmethod
    def available() -> bool:
        try:
            import gudhi  # noqa: F401
        except ImportError:  # pragma: no cover  -- bench extras install gudhi.
            return False
        return True

    @staticmethod
    def compute_diagram(X: torch.Tensor, max_dim: int) -> list[torch.Tensor]:
        if X.dtype != torch.float64:
            raise TypeError(
                f"{Hofer2017Reference.name}: input X must be float64, got {X.dtype}"
            )
        packed = _compute_rips_persistence(X, max_dim, include_essential=True)
        # Split by the dim marker (column 2) into the per-dimension list.
        out: list[torch.Tensor] = []
        for k in range(max_dim + 1):
            mask = packed[:, 2] == float(k)
            bars_k = packed[mask][:, :2]
            out.append(bars_k.contiguous())
        return out

    @staticmethod
    def loss_longest_h1(X: torch.Tensor) -> torch.Tensor:
        diagrams = Hofer2017Reference.compute_diagram(X, max_dim=1)
        h1 = diagrams[1] if len(diagrams) > 1 else X.new_empty((0, 2))
        if h1.numel() == 0:
            return torch.zeros((), dtype=X.dtype, device=X.device) + 0.0 * X.sum()
        finite_mask = torch.isfinite(h1).all(dim=1)
        finite = h1[finite_mask]
        if finite.numel() == 0:  # pragma: no cover
            # gudhi only emits finite-death H_1 bars via flag_persistence_generators
            # (essential H_1 classes are returned in gens[3] but we drop them for
            # the loss). This branch is a safety net for backends added later
            # that may emit infinite H_1 bars through compute_diagram.
            return torch.zeros((), dtype=X.dtype, device=X.device) + 0.0 * X.sum()
        return -(finite[:, 1] - finite[:, 0]).max()
