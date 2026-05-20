"""
Diagram vectorizers: PersistenceDiagram → fixed-length NDArray.

Vectorization is what makes topological features usable by downstream ML.
Each vectorizer produces a deterministic feature vector of known length so
the pipeline output shape is predictable for sklearn / XGBoost / PyTorch.

v0.1 ships persistence images (Adams et al. 2017) and Betti curves.
Landscapes (Bubenik 2015) and persistence entropy land in v0.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from topogeoml.core.diagrams import PersistenceDiagram


def _bars_for_pi(
    diagram: PersistenceDiagram,
    dim: int,
    fallback_max: float,
) -> NDArray[np.float64]:
    """
    Extract bars for dimension `dim` and replace infinite deaths with
    `fallback_max`. PersistenceImager requires finite values.
    """
    arr = diagram.bars.get(dim)
    if arr is None or arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    out = np.asarray(arr, dtype=np.float64).copy()
    inf_mask = ~np.isfinite(out[:, 1])
    if inf_mask.any():
        out[inf_mask, 1] = fallback_max
    return out


def _persistence_image_one(
    bars: NDArray[np.float64],
    resolution: int,
    sigma: float,
    fallback_max: float,
    weight_power: float,
) -> NDArray[np.float64]:
    """
    Persistence image for one (birth, death) bar set on a fixed birth-persistence grid.

    Algorithm (Adams et al. 2017, "Persistence Images: A Stable Vector
    Representation of Persistent Homology", JMLR):
        1. Map (birth, death) → (birth, persistence = death − birth).
        2. Each point contributes an isotropic 2D Gaussian (bandwidth ``sigma``)
           weighted by ``persistence ** weight_power``.
        3. Discretize on a ``resolution × resolution`` grid spanning
           ``[0, fallback_max]`` on each axis.

    Returns a flat float64 array of length ``resolution ** 2`` (row-major).
    """
    if bars.shape[0] == 0:
        return np.zeros(resolution * resolution, dtype=np.float64)

    births = bars[:, 0].astype(np.float64, copy=False)
    pers = (bars[:, 1] - bars[:, 0]).astype(np.float64, copy=False)
    # Negative persistences would indicate malformed bars; clip defensively.
    pers = np.clip(pers, a_min=0.0, a_max=None)
    weights = pers ** weight_power

    edges = np.linspace(0.0, fallback_max, resolution + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])  # (resolution,)
    bx, py = np.meshgrid(centers, centers, indexing="xy")  # both (resolution, resolution)

    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    dx = bx[..., None] - births[None, None, :]  # (R, R, n)
    dy = py[..., None] - pers[None, None, :]  # (R, R, n)
    gauss = np.exp(-(dx * dx + dy * dy) * inv_two_sigma_sq)
    img = (gauss * weights[None, None, :]).sum(axis=-1)  # (R, R)
    return np.ascontiguousarray(img, dtype=np.float64).ravel()


@dataclass
class PersistenceImageVectorizer:
    """
    Persistence Images (Adams et al. 2017), in-house NumPy implementation.

    Discretizes a persistence diagram into a (resolution × resolution) image
    by smoothing each (birth, persistence) point with a Gaussian kernel,
    weighted by a function of persistence. Each homology dimension produces
    one image; outputs are concatenated and flattened.

    Parameters
    ----------
    homology_dims : tuple[int, ...]
        Dimensions to vectorize (default (0, 1)).
    resolution : int
        Pixel resolution per side. Output length = len(homology_dims) * resolution^2.
    sigma : float
        Gaussian kernel bandwidth in birth-persistence coordinates.
    fallback_max : float
        Finite value substituted for infinite deaths and the upper bound of the
        birth and persistence axes (typically the max observed birth in your
        dataset; choose calibrated to data scale).
    weight_power : float
        Persistence weight exponent (1.0 = linear in persistence, standard).

    Notes
    -----
    The grid is fixed to ``[0, fallback_max]`` on both axes. For batch use,
    callers should fit ``fallback_max`` across the full training set first;
    see TopologyFeaturePipeline for the fitted flow.
    """

    homology_dims: tuple[int, ...] = (0, 1)
    resolution: int = 20
    sigma: float = 0.1
    fallback_max: float = 1.0
    weight_power: float = 1.0

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {self.resolution}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if not self.homology_dims:
            raise ValueError("homology_dims must be non-empty")
        if self.fallback_max <= 0:
            raise ValueError(f"fallback_max must be positive, got {self.fallback_max}")

    @property
    def output_dim(self) -> int:
        """Length of the feature vector produced by transform()."""
        return len(self.homology_dims) * self.resolution * self.resolution

    def transform_one(self, diagram: PersistenceDiagram) -> NDArray[np.float64]:
        """
        Vectorize a single diagram into a 1D feature vector.

        Parameters
        ----------
        diagram : PersistenceDiagram
            Input diagram.

        Returns
        -------
        NDArray[np.float64]
            1D array of length self.output_dim.
        """
        feats: list[NDArray[np.float64]] = []
        for dim in self.homology_dims:
            bars = _bars_for_pi(diagram, dim, self.fallback_max)
            feats.append(
                _persistence_image_one(
                    bars,
                    resolution=self.resolution,
                    sigma=self.sigma,
                    fallback_max=self.fallback_max,
                    weight_power=self.weight_power,
                )
            )
        return np.concatenate(feats).astype(np.float64, copy=False)


@dataclass
class BettiCurveVectorizer:
    """
    Betti curve: function t ↦ β_k(t) = # bars containing t, sampled at fixed grid.

    Cheaper than persistence images, often surprisingly competitive as a
    baseline topological feature. Output length = len(homology_dims) * resolution.

    Parameters
    ----------
    homology_dims : tuple[int, ...]
        Dimensions to vectorize.
    resolution : int
        Number of sample points along the filtration parameter axis.
    fallback_max : float
        Finite value substituted for infinite deaths. Also the upper limit
        of the sampling grid (lower limit is 0).
    """

    homology_dims: tuple[int, ...] = (0, 1)
    resolution: int = 100
    fallback_max: float = 1.0

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {self.resolution}")
        if not self.homology_dims:
            raise ValueError("homology_dims must be non-empty")

    @property
    def output_dim(self) -> int:
        return len(self.homology_dims) * self.resolution

    def transform_one(self, diagram: PersistenceDiagram) -> NDArray[np.float64]:
        """
        Vectorize via Betti curves sampled at a uniform grid in [0, fallback_max].
        """
        grid = np.linspace(0.0, self.fallback_max, self.resolution, dtype=np.float64)
        feats: list[NDArray[np.float64]] = []
        for dim in self.homology_dims:
            bars = _bars_for_pi(diagram, dim, self.fallback_max)
            if bars.shape[0] == 0:
                feats.append(np.zeros(self.resolution, dtype=np.float64))
                continue
            # Vectorized: β_k(t) = #{i : birth_i <= t < death_i}.
            births = bars[:, 0:1]  # (n_bars, 1)
            deaths = bars[:, 1:2]  # (n_bars, 1)
            grid_row = grid[np.newaxis, :]  # (1, resolution)
            alive = (births <= grid_row) & (grid_row < deaths)  # (n_bars, resolution)
            curve = alive.sum(axis=0).astype(np.float64)  # (resolution,)
            feats.append(curve)
        return np.concatenate(feats).astype(np.float64, copy=False)
