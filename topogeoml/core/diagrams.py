"""
Persistence diagrams with full provenance.

A PersistenceDiagram wraps the multi-dimensional output of a filtration with
metadata identifying how it was produced. Provenance is mandatory per the
verification gate: every reported number must carry its origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DiagramProvenance:
    """
    Origin metadata for a persistence diagram.

    Recorded at compute time so that downstream features can be traced back
    to the exact filtration, metric, and parameters that produced them.
    """

    filtration: str
    """Name of the filtration backend, e.g. 'rips', 'cubical', 'alpha'."""

    metric: str
    """Distance metric used, e.g. 'euclidean', 'spectral', 'fermat'."""

    max_homology_dim: int
    """Highest homology dimension computed."""

    max_edge_length: float | None
    """Maximum edge length cutoff (None = no cutoff)."""

    n_points: int
    """Number of input points the diagram was computed from."""

    ambient_dim: int
    """Ambient dimension of the input point cloud."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Backend-specific extras (e.g. intrinsic-dimension estimate, coeff field)."""


@dataclass(frozen=True)
class PersistenceDiagram:
    """
    Multi-dimensional persistence diagram.

    Each dimension k stores an (n_k, 2) float array of (birth, death) pairs.
    Infinite death values use np.inf and should be handled explicitly by
    downstream vectorizers (typically replaced with max filtration value).

    Parameters
    ----------
    bars : dict[int, NDArray[np.float64]]
        Mapping from homology dimension to (birth, death) array.
    provenance : DiagramProvenance
        How the diagram was produced.

    Notes
    -----
    Frozen dataclass — diagrams are immutable. To modify, construct a new one.
    All birth/death arrays must be float64 and 2D with shape (n, 2). This is
    enforced at construction time.
    """

    bars: dict[int, NDArray[np.float64]]
    provenance: DiagramProvenance

    def __post_init__(self) -> None:
        # Validate every dimension's array shape and dtype.
        for dim, arr in self.bars.items():
            if not isinstance(dim, int) or dim < 0:
                raise ValueError(f"homology dimension must be non-negative int, got {dim!r}")
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"bars[{dim}] must be np.ndarray, got {type(arr).__name__}")
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    f"bars[{dim}] must have shape (n, 2), got {arr.shape}"
                )
            if arr.dtype != np.float64:
                raise TypeError(
                    f"bars[{dim}] must be float64, got {arr.dtype} — "
                    "explicit dtype is required (elite-code-standards §1.3)"
                )

    @property
    def max_dim(self) -> int:
        """Highest homology dimension present."""
        return max(self.bars.keys()) if self.bars else -1

    def n_bars(self, dim: int) -> int:
        """Number of (birth, death) pairs in dimension `dim`."""
        return int(self.bars.get(dim, np.empty((0, 2), dtype=np.float64)).shape[0])

    def lifetimes(self, dim: int, finite_only: bool = True) -> NDArray[np.float64]:
        """
        Lifetime (death - birth) values for dimension `dim`.

        Parameters
        ----------
        dim : int
            Homology dimension.
        finite_only : bool
            If True (default), drop bars with infinite death.

        Returns
        -------
        NDArray[np.float64]
            1D array of lifetimes.
        """
        arr = self.bars.get(dim)
        if arr is None or arr.size == 0:
            return np.empty(0, dtype=np.float64)
        births = arr[:, 0]
        deaths = arr[:, 1]
        lifetimes = deaths - births
        if finite_only:
            mask = np.isfinite(lifetimes)
            lifetimes = lifetimes[mask]
        return lifetimes.astype(np.float64, copy=False)

    def total_persistence(self, dim: int, p: float = 1.0) -> float:
        """
        L^p total persistence: sum of finite lifetimes raised to p.

        Standard topological summary statistic. p=1 is l1 total persistence,
        p=2 is the squared l2 norm of the diagram (as a discrete measure).
        """
        lifetimes = self.lifetimes(dim, finite_only=True)
        if lifetimes.size == 0:
            return 0.0
        return float(np.sum(np.power(lifetimes, p)))

    def __repr__(self) -> str:
        summary = ", ".join(
            f"H{d}={self.n_bars(d)}" for d in sorted(self.bars.keys())
        )
        return (
            f"PersistenceDiagram({summary}; "
            f"filtration={self.provenance.filtration}, "
            f"metric={self.provenance.metric})"
        )
