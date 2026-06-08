"""
Embedding Topology Audit — prototype.

Diagnose the topology of a learned embedding by computing Rips persistence on
a subsample of the embedding matrix and reporting:

  - β_0 estimate (significant 0-dim bars) — number of cluster-like components
  - β_1 estimate (significant 1-dim bars) — loop / hole structure
  - L^1 total persistence per dimension — concentration of topological signal
  - longest H_1 bar lifetime — single most significant loop
  - mean / median nearest-neighbor distance — local density indicator

This is a v0.0.1 prototype. The full audit (drift-tensor correction, intrinsic
dimension calibration, PH metric cascade selection, marketplace-graph aware
audits) lands in v0.1+.

Item 9 of the v0.0.1 scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from topogeoml.core.filtrations import RipsFiltration


@dataclass(frozen=True)
class EmbeddingTopologyAudit:
    """Structured report on the topology of an embedding."""

    n_points_audited: int
    """Number of embedding rows used for the audit (≤ original size after subsampling)."""

    ambient_dim: int
    """Embedding dimension."""

    beta_0_estimate: int
    """Count of 0-dim bars with lifetime ≥ persistence_threshold."""

    beta_1_estimate: int
    """Count of 1-dim bars with lifetime ≥ persistence_threshold."""

    total_persistence_h0: float
    """Sum of finite 0-dim bar lifetimes (L^1)."""

    total_persistence_h1: float
    """Sum of finite 1-dim bar lifetimes (L^1)."""

    longest_h1_lifetime: float
    """Lifetime of the most persistent 1-dim bar (0.0 if none)."""

    mean_nn_distance: float
    """Mean nearest-neighbor Euclidean distance — local density proxy."""

    median_nn_distance: float
    """Median nearest-neighbor distance — robust density proxy."""

    persistence_threshold: float
    """Threshold used for significant-bar counting."""

    provenance: dict[str, Any] = field(default_factory=dict)
    """Audit configuration record."""

    def summary(self) -> str:
        """Human-readable one-paragraph audit summary."""
        return (
            f"Audit of {self.n_points_audited} points in R^{self.ambient_dim}: "
            f"β_0 ≈ {self.beta_0_estimate} clusters, "
            f"β_1 ≈ {self.beta_1_estimate} loops "
            f"(longest H_1 lifetime {self.longest_h1_lifetime:.4f}). "
            f"Mean NN distance {self.mean_nn_distance:.4f}, "
            f"median {self.median_nn_distance:.4f}. "
            f"Total persistence: H_0 {self.total_persistence_h0:.4f}, "
            f"H_1 {self.total_persistence_h1:.4f}."
        )


def audit_embedding(
    embeddings: NDArray[np.floating[Any]],
    max_points: int = 1000,
    max_homology_dim: int = 1,
    persistence_threshold: float | None = None,
    seed: int = 42,
) -> EmbeddingTopologyAudit:
    """
    Compute a topology audit of an embedding matrix.

    Parameters
    ----------
    embeddings : NDArray
        Shape (n_points, ambient_dim).
    max_points : int
        If n_points > max_points, uniformly subsample to control Rips cost.
        Rips is O(n^d * polylog) where d = max_homology_dim+1, so n=1000 is
        a reasonable upper bound for H_1.
    max_homology_dim : int
        Highest homology dimension to compute (default 1).
    persistence_threshold : float, optional
        Minimum lifetime to count a bar as "significant". If None, defaults to
        2× median nearest-neighbor distance (noise-floor heuristic).
    seed : int
        RNG seed for subsampling.

    Returns
    -------
    EmbeddingTopologyAudit

    Notes
    -----
    Persistence-threshold defaulting is a v0.0.1 heuristic. v0.1 will use the
    PH metric cascade (Euclidean → Spectral → Fermat) with intrinsic-dimension
    calibration to pick the right metric *before* deciding what's significant.
    """
    X = np.ascontiguousarray(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D (n_points, dim); got shape {X.shape}")
    n_orig, dim = X.shape
    if n_orig < 2:
        raise ValueError(f"need at least 2 points for audit; got {n_orig}")

    # Subsample if needed (elite-code-standards §6.5: seeded RNG).
    rng = np.random.default_rng(seed)
    if n_orig > max_points:
        idx = rng.choice(n_orig, size=max_points, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    n_aud = X_sub.shape[0]

    # Nearest-neighbor diagnostics (k=2 because index 0 is the point itself).
    nn = NearestNeighbors(n_neighbors=2).fit(X_sub)
    dists, _ = nn.kneighbors(X_sub)
    nn_d = dists[:, 1]  # distance to first true neighbor
    mean_nn = float(np.mean(nn_d))
    median_nn = float(np.median(nn_d))

    # Threshold defaults to 2× median NN distance — bars shorter than the
    # local data scale are noise. Replace with cascade-calibrated value in v0.1.
    if persistence_threshold is None:
        persistence_threshold = 2.0 * median_nn

    # Rips persistence.
    rips = RipsFiltration(max_homology_dim=max_homology_dim)
    diagram = rips.compute(X_sub)

    def _count_significant(bars: NDArray[np.float64], threshold: float) -> int:
        if bars.size == 0:
            return 0
        finite_births = bars[:, 0]
        deaths = bars[:, 1]
        # Infinite deaths always count as significant.
        inf_mask = ~np.isfinite(deaths)
        finite_mask = ~inf_mask
        significant = int(inf_mask.sum())
        if finite_mask.any():  # pragma: no branch
            # Real Rips persistence on n >= 2 points always yields at least one
            # finite-death bar, so the false edge fires only for a manually
            # constructed all-infinite diagram.
            lifetimes = deaths[finite_mask] - finite_births[finite_mask]
            significant += int((lifetimes >= threshold).sum())
        return significant

    beta_0 = _count_significant(diagram.bars.get(0, np.empty((0, 2))), persistence_threshold)
    beta_1 = (
        _count_significant(diagram.bars.get(1, np.empty((0, 2))), persistence_threshold)
        if max_homology_dim >= 1
        else 0
    )

    tp_h0 = diagram.total_persistence(0, p=1.0) if 0 in diagram.bars else 0.0
    tp_h1 = diagram.total_persistence(1, p=1.0) if 1 in diagram.bars else 0.0

    h1_bars = diagram.bars.get(1)
    if h1_bars is not None and h1_bars.size > 0:
        finite_h1 = h1_bars[np.isfinite(h1_bars[:, 1])]
        longest_h1 = float((finite_h1[:, 1] - finite_h1[:, 0]).max()) if finite_h1.size > 0 else 0.0
    else:
        longest_h1 = 0.0

    return EmbeddingTopologyAudit(
        n_points_audited=n_aud,
        ambient_dim=dim,
        beta_0_estimate=beta_0,
        beta_1_estimate=beta_1,
        total_persistence_h0=float(tp_h0),
        total_persistence_h1=float(tp_h1),
        longest_h1_lifetime=longest_h1,
        mean_nn_distance=mean_nn,
        median_nn_distance=median_nn,
        persistence_threshold=float(persistence_threshold),
        provenance={
            "n_points_original": n_orig,
            "subsampled": n_orig > max_points,
            "max_homology_dim": max_homology_dim,
            "seed": seed,
            "filtration_backend": "ripser",
        },
    )
