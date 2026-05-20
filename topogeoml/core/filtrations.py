"""
Filtrations: maps from data to persistence diagrams.

v0.1 ships Vietoris-Rips via ripser. Alpha, cubical, lower-star, and the
PH metric cascade (Euclidean → Spectral → Fermat with d_int/d_amb selection)
land in v0.2.

The contract: filtration.fit(X) is a no-op for stateless filtrations;
filtration.compute(X) returns a PersistenceDiagram with provenance attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ripser import ripser

from topogeoml.core.diagrams import DiagramProvenance, PersistenceDiagram


@dataclass(frozen=True)
class RipsFiltration:
    """
    Vietoris-Rips filtration via ripser.

    Computes persistent homology of the Rips complex on a point cloud or
    pre-computed distance matrix. Wraps ripser's output into a typed
    PersistenceDiagram with provenance metadata.

    Parameters
    ----------
    max_homology_dim : int
        Highest homology dimension to compute (default 1: H_0 and H_1).
    max_edge_length : float, optional
        Maximum edge length (also called 'thresh' in ripser). If None,
        ripser uses its default (∞), which can be expensive for large clouds.
    metric : str
        Distance metric. 'euclidean' (default), 'precomputed' for distance
        matrix input, or any scipy-compatible metric string.
    coeff : int
        Coefficient field for homology (default 2 = Z/2Z, standard for PH).

    Notes
    -----
    For n > ~2000 points, set max_edge_length to a finite value to avoid
    explosive complex sizes. Use intrinsic-dimension diagnostics from
    metric_space_diagnostics (v0.2) to pick this principled.
    """

    max_homology_dim: int = 1
    max_edge_length: float | None = None
    metric: str = "euclidean"
    coeff: int = 2

    def __post_init__(self) -> None:
        if self.max_homology_dim < 0:
            raise ValueError(
                f"max_homology_dim must be >= 0, got {self.max_homology_dim}"
            )
        if self.max_edge_length is not None and self.max_edge_length <= 0:
            raise ValueError(
                f"max_edge_length must be positive, got {self.max_edge_length}"
            )
        if self.coeff < 2:
            raise ValueError(f"coeff must be >= 2 (prime), got {self.coeff}")

    def compute(self, X: NDArray[np.floating[Any]]) -> PersistenceDiagram:
        """
        Compute the Rips persistence diagram on point cloud X.

        Parameters
        ----------
        X : NDArray
            Point cloud of shape (n_points, ambient_dim), or a square distance
            matrix of shape (n_points, n_points) if metric='precomputed'.

        Returns
        -------
        PersistenceDiagram
            Diagram with bars for dimensions 0..max_homology_dim and
            full provenance.
        """
        # Explicit dtype (elite-code-standards §1.3).
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_points, dim) or (n_points, n_points); "
                f"got shape {X_arr.shape}"
            )

        n_points = X_arr.shape[0]
        if self.metric == "precomputed":
            if X_arr.shape[0] != X_arr.shape[1]:
                raise ValueError(
                    f"precomputed distance matrix must be square, "
                    f"got {X_arr.shape}"
                )
            ambient_dim = -1  # unknown for precomputed
        else:
            ambient_dim = X_arr.shape[1]

        # Build ripser kwargs. ripser uses np.inf when thresh is unset.
        ripser_kwargs: dict[str, Any] = {
            "maxdim": int(self.max_homology_dim),
            "coeff": int(self.coeff),
            "distance_matrix": self.metric == "precomputed",
        }
        if self.max_edge_length is not None:
            ripser_kwargs["thresh"] = float(self.max_edge_length)
        if self.metric not in ("euclidean", "precomputed"):
            ripser_kwargs["metric"] = self.metric

        result = ripser(X_arr, **ripser_kwargs)

        # ripser returns dgms as list of (n_k, 2) arrays, one per dim 0..maxdim.
        # Coerce to float64 dict keyed by dimension.
        bars: dict[int, NDArray[np.float64]] = {}
        for dim, arr in enumerate(result["dgms"]):
            bars[dim] = np.ascontiguousarray(arr, dtype=np.float64)

        provenance = DiagramProvenance(
            filtration="rips",
            metric=self.metric,
            max_homology_dim=self.max_homology_dim,
            max_edge_length=self.max_edge_length,
            n_points=n_points,
            ambient_dim=ambient_dim,
            extra={"coeff": self.coeff, "backend": "ripser"},
        )

        return PersistenceDiagram(bars=bars, provenance=provenance)
