"""
TopologyFeaturePipeline — Module 1 of the TopoGeomML MVP.

Flow: point clouds → Rips filtration → persistence diagram → vectorization → feature matrix.

This is the topology-as-features path: produces a fixed-length feature vector
per input sample that drops into sklearn / XGBoost / PyTorch as ordinary
tabular features. Use it as a baseline before reaching for differentiable
layers or higher-order architectures.

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from topogeoml.core.filtrations import RipsFiltration
from topogeoml.core.vectorizers import (
    BettiCurveVectorizer,
    PersistenceImageVectorizer,
)

# Type alias: a batch of point clouds. Either a 3D array (n_samples, n_points, dim)
# when point clouds share size, or a sequence of 2D arrays with varying sizes.
PointCloudBatch = NDArray[np.floating[Any]] | Sequence[NDArray[np.floating[Any]]]


@dataclass
class FitProvenance:
    """
    Records what was fit and how. Required by the verification gate.

    Every reported number must carry its provenance: this is the
    pipeline-level record of how features were produced.
    """

    n_samples_seen: int
    ambient_dim: int
    max_homology_dim: int
    max_edge_length: float | None
    vectorizer: str
    output_dim: int
    metric: str
    pipeline_version: str
    extras: dict[str, Any] = field(default_factory=dict)


class TopologyFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Compute topological features from a batch of point clouds.

    Each input sample is an (n_points, ambient_dim) array. Output is a flat
    feature vector per sample of length determined by the vectorizer config.

    Parameters
    ----------
    max_homology_dim : int
        Highest homology dimension to compute (default 1: H_0 + H_1).
    max_edge_length : float, optional
        Rips edge length cutoff. None lets ripser pick its default (∞);
        set a finite value for large point clouds.
    vectorizer : {"persistence_image", "betti_curve"}
        Diagram vectorization method.
    resolution : int
        Vectorization resolution. PI: side length of square image. Betti: # samples.
    sigma : float
        PI Gaussian bandwidth (ignored for Betti curves).
    metric : str
        Rips distance metric ('euclidean' default; 'precomputed' for
        distance-matrix input).
    n_jobs : int
        Reserved for future parallel transform (currently 1).

    Attributes
    ----------
    fit_provenance_ : FitProvenance
        Set after fit(). Records the fitting context for downstream auditing.
    vectorizer_ : PersistenceImageVectorizer | BettiCurveVectorizer
        Vectorizer instance built during fit().
    filtration_ : RipsFiltration
        Filtration instance built during fit().

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # Two samples: a noisy circle and a noisy line.
    >>> theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    >>> circle = np.stack([np.cos(theta), np.sin(theta)], axis=1) + 0.05 * rng.standard_normal((50, 2))
    >>> line = np.stack([np.linspace(-1, 1, 50), np.zeros(50)], axis=1) + 0.05 * rng.standard_normal((50, 2))
    >>> pipe = TopologyFeaturePipeline(max_homology_dim=1, resolution=10)
    >>> X = [circle, line]
    >>> features = pipe.fit_transform(X)
    >>> features.shape
    (2, 200)
    """

    def __init__(
        self,
        max_homology_dim: int = 1,
        max_edge_length: float | None = None,
        vectorizer: Literal["persistence_image", "betti_curve"] = "persistence_image",
        resolution: int = 20,
        sigma: float = 0.1,
        metric: str = "euclidean",
        n_jobs: int = 1,
    ) -> None:
        self.max_homology_dim = max_homology_dim
        self.max_edge_length = max_edge_length
        self.vectorizer = vectorizer
        self.resolution = resolution
        self.sigma = sigma
        self.metric = metric
        self.n_jobs = n_jobs

    # ---------- sklearn API ----------

    def fit(
        self,
        X: PointCloudBatch,
        y: NDArray[np.floating[Any]] | None = None,
    ) -> TopologyFeaturePipeline:
        """
        Fit the pipeline.

        For v0.1 fitting is mostly metadata capture: validate the input shape,
        compute the fallback_max for vectorization from training data scale,
        and freeze the vectorizer configuration. No global state is leaked
        from training to transform beyond `fallback_max`, so per-fold use
        inside CV is safe (verification-gate compliant).
        """
        clouds = self._coerce_batch(X)
        n_samples = len(clouds)
        if n_samples == 0:
            raise ValueError("fit received an empty batch")

        ambient_dim = clouds[0].shape[1] if self.metric != "precomputed" else -1

        # Estimate filtration-value scale: max pairwise distance across the
        # first few samples (capped to bound fit cost). Used as fallback_max
        # for the vectorizer (substituted for infinite deaths and grid limits).
        fallback_max = self._estimate_filtration_scale(clouds, n_probe=min(8, n_samples))

        self.filtration_: RipsFiltration = RipsFiltration(
            max_homology_dim=self.max_homology_dim,
            max_edge_length=self.max_edge_length,
            metric=self.metric,
        )

        vec: PersistenceImageVectorizer | BettiCurveVectorizer
        if self.vectorizer == "persistence_image":
            vec = PersistenceImageVectorizer(
                homology_dims=tuple(range(self.max_homology_dim + 1)),
                resolution=self.resolution,
                sigma=self.sigma,
                fallback_max=fallback_max,
            )
        elif self.vectorizer == "betti_curve":
            vec = BettiCurveVectorizer(
                homology_dims=tuple(range(self.max_homology_dim + 1)),
                resolution=self.resolution,
                fallback_max=fallback_max,
            )
        else:
            raise ValueError(
                f"vectorizer must be 'persistence_image' or 'betti_curve', "
                f"got {self.vectorizer!r}"
            )
        self.vectorizer_ = vec

        from topogeoml._version import __version__

        self.fit_provenance_: FitProvenance = FitProvenance(
            n_samples_seen=n_samples,
            ambient_dim=ambient_dim,
            max_homology_dim=self.max_homology_dim,
            max_edge_length=self.max_edge_length,
            vectorizer=self.vectorizer,
            output_dim=vec.output_dim,
            metric=self.metric,
            pipeline_version=__version__,
            extras={"fallback_max": fallback_max},
        )

        return self

    def transform(self, X: PointCloudBatch) -> NDArray[np.float64]:
        """
        Compute topological feature vectors for each point cloud in X.

        Returns
        -------
        NDArray[np.float64]
            Shape (n_samples, self.fit_provenance_.output_dim).
        """
        check_is_fitted(self, ["filtration_", "vectorizer_", "fit_provenance_"])
        clouds = self._coerce_batch(X)
        if not clouds:
            return np.empty((0, self.fit_provenance_.output_dim), dtype=np.float64)

        # Pre-allocate output (elite-code-standards §3.1).
        n = len(clouds)
        out = np.empty((n, self.fit_provenance_.output_dim), dtype=np.float64)

        # Construction loop over samples is acceptable: each iteration calls
        # into ripser (C++) for the heavy work. No Python-level numeric
        # computation here.
        for i, cloud in enumerate(clouds):
            diagram = self.filtration_.compute(cloud)
            out[i] = self.vectorizer_.transform_one(diagram)

        return out

    # ---------- helpers ----------

    @staticmethod
    def _coerce_batch(X: PointCloudBatch) -> list[NDArray[np.float64]]:
        """
        Normalize input to a list of (n_points, dim) float64 arrays.

        Accepts a 3D array (n_samples, n_points, dim) or any sequence of 2D
        arrays. Empty sequences are permitted; per-sample arrays must be 2D.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 3:
                return [np.ascontiguousarray(X[i], dtype=np.float64) for i in range(X.shape[0])]
            if X.ndim == 2:
                # Single point cloud — wrap as a batch of 1.
                return [np.ascontiguousarray(X, dtype=np.float64)]
            raise ValueError(
                f"ndarray input must be 2D (single cloud) or 3D (batch); "
                f"got shape {X.shape}"
            )
        out: list[NDArray[np.float64]] = []
        for i, cloud in enumerate(X):
            arr = np.asarray(cloud, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"sample {i} must be 2D (n_points, dim); got shape {arr.shape}"
                )
            out.append(np.ascontiguousarray(arr))
        return out

    @staticmethod
    def _estimate_filtration_scale(
        clouds: list[NDArray[np.float64]],
        n_probe: int = 8,
    ) -> float:
        """
        Estimate a representative max filtration value from the training batch.

        Uses the max pairwise Euclidean distance across the first `n_probe`
        clouds. Provides a finite cap for infinite deaths and vectorizer grids.
        For precomputed distance-matrix input, uses the max off-diagonal entry.

        Returns 1.0 as a safe default if estimation yields a non-positive value.
        """
        if not clouds:
            return 1.0
        probe = clouds[:n_probe]
        max_d = 0.0
        for cloud in probe:
            if cloud.shape[0] < 2:
                continue
            if cloud.shape[0] == cloud.shape[1] and np.allclose(np.diag(cloud), 0.0):
                # Likely precomputed distance matrix: take max off-diagonal.
                off_diag = cloud[~np.eye(cloud.shape[0], dtype=bool)]
                if off_diag.size > 0:
                    d = float(np.max(off_diag))
                    max_d = max(max_d, d)
            else:
                # Bounding box diagonal as a cheap O(n*d) upper bound on max pair.
                bbox_diag = float(np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0)))
                max_d = max(max_d, bbox_diag)
        return max_d if max_d > 0.0 else 1.0
