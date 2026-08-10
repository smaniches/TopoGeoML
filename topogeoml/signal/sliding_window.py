"""
Sliding-window persistent homology feature extraction.

Implements DEFINITION 5.5 and the algorithm specification of §6.2 in
``docs/mathematics/foundations.md``.

The extracted feature vector is invariant under the isometry group of
:math:`(\\mathbb{R}^k, \\|\\cdot\\|_2)` acting on the point cloud
(PROPOSITION 4.2), and under temporal translation of the input signal
(COROLLARY 4.3). Scale normalization (REMARK 4.4) is optional via the
``scale_normalize`` argument.

Each window contributes :math:`5 (K + 1)` statistics: for each homology
dimension :math:`k \\in \\{0, 1, \\ldots, K\\}` the statistics are
``count_finite``, :math:`\\operatorname{Pers}_1`, :math:`\\operatorname{Pers}_2`,
persistence entropy :math:`E`, and longest lifetime
:math:`\\max(d_i - b_i)`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from ripser import ripser

# ---------------------------------------------------------------------------
# Statistics extracted from a single persistence diagram
# ---------------------------------------------------------------------------

# Order matters: this dictates the feature-vector column order.
_DIAGRAM_STATISTIC_NAMES: tuple[str, ...] = (
    "count_finite",
    "total_persistence_p1",
    "total_persistence_p2",
    "persistence_entropy",
    "longest_lifetime",
)
_N_STATS_PER_DIM = len(_DIAGRAM_STATISTIC_NAMES)


def _diagram_statistics(diagram: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the 5 statistics of DEFINITION 5.5 step 4 from a single
    persistence diagram.

    Parameters
    ----------
    diagram : numpy.ndarray, shape (n_bars, 2)
        Persistence diagram. Bars with ``death == inf`` are excluded
        from all statistics (REMARK in §5.1).

    Returns
    -------
    numpy.ndarray, shape (5,), dtype float64
        Statistics in order: count_finite, Pers_1, Pers_2, entropy,
        longest_lifetime.
    """
    if diagram.shape[0] == 0:
        return np.zeros(_N_STATS_PER_DIM, dtype=np.float64)

    finite_mask = np.isfinite(diagram[:, 1])
    finite_bars = diagram[finite_mask]
    if finite_bars.shape[0] == 0:
        return np.zeros(_N_STATS_PER_DIM, dtype=np.float64)

    lifetimes = (finite_bars[:, 1] - finite_bars[:, 0]).astype(
        np.float64, copy=False
    )
    # §1.2: lifetimes are differences of nonneg quantities; they can be
    # negative under numerical noise. Clip to zero.
    lifetimes = np.maximum(lifetimes, 0.0)

    count_finite = float(finite_bars.shape[0])
    pers_1 = float(lifetimes.sum())
    pers_2 = float((lifetimes * lifetimes).sum())

    # §1.4: division safety. Entropy convention 0 log 0 := 0 (Definition 5.2).
    total = pers_1 + 1e-300
    if pers_1 == 0.0:
        entropy = 0.0
    else:
        p = lifetimes / total
        p_safe = np.maximum(p, 1e-300)
        entropy = float(-(p * np.log(p_safe)).sum())

    longest = float(lifetimes.max()) if lifetimes.size > 0 else 0.0

    return np.array(
        [count_finite, pers_1, pers_2, entropy, longest],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Sliding-window topology features
# ---------------------------------------------------------------------------

@dataclass
class TopologyFeatureConfig:
    """
    Configuration for sliding-window topology feature extraction.

    Attributes
    ----------
    window_length : int
        :math:`W \\ge 2`. Length of each sliding window in points.
    stride : int
        :math:`\\Delta \\ge 1`. Stride between consecutive windows.
    max_homology_dim : int
        :math:`K \\ge 0`. Maximum homology dimension passed to ripser.
    edge_threshold : float, optional
        Vietoris-Rips edge length cutoff (passed to ripser as ``thresh``).
        If ``None``, ripser uses its default (``inf``); the empirical
        choice is the 95th percentile of pairwise distances.
    scale_normalize : bool
        If True, divide all persistence lifetimes by the window
        diameter before computing statistics (REMARK 4.4). Provides
        invariance under signal scaling.
    pooling : tuple[str, ...]
        Pooling functions across windows. Each entry must be one of
        ``"mean"``, ``"max"``, ``"min"``, ``"std"``. Order determines
        the feature-vector layout.
    """

    window_length: int = 32
    stride: int = 16
    max_homology_dim: int = 1
    edge_threshold: float | None = None
    scale_normalize: bool = True
    pooling: tuple[str, ...] = ("mean", "max")


def _pool(values: NDArray[np.float64], how: str) -> NDArray[np.float64]:
    """Single pooling operation across the window axis (axis=0).

    The cast to ``NDArray[np.float64]`` on each branch is necessary because
    numpy's reductions return ``Any`` under strict typing (the return dtype
    depends on the input dtype at runtime); the inputs here are statically
    constrained to ``np.float64`` and the outputs preserve that, so the
    cast is a safe annotation correction rather than a runtime change.
    """
    if values.shape[0] == 0:
        return np.zeros(values.shape[1:], dtype=np.float64)
    if how == "mean":
        return np.asarray(values.mean(axis=0), dtype=np.float64)
    if how == "max":
        return np.asarray(values.max(axis=0), dtype=np.float64)
    if how == "min":
        return np.asarray(values.min(axis=0), dtype=np.float64)
    if how == "std":
        # ddof=0: biased estimator, consistent across single-window edge cases.
        return np.asarray(values.std(axis=0), dtype=np.float64)
    raise ValueError(f"Unknown pooling function: {how!r}")


def sliding_window_topology_features(
    point_cloud: NDArray[np.floating],
    config: TopologyFeatureConfig | None = None,
) -> NDArray[np.float64]:
    """
    Extract pooled topology feature vector from a point cloud.

    Algorithm specification (§6.2 of foundations.md)
    -----------------------------------------------
    Input space:     ``point_cloud`` :math:`\\in \\mathbb{R}^{N \\times k}`,
                     ``config``.
    Output space:    :math:`\\mathbb{R}^{|\\text{pooling}| \\cdot (K+1) \\cdot 5}`.
    Correctness:     For each window offset
                     :math:`t_0 \\in \\{0, \\Delta, 2\\Delta, \\ldots\\}` with
                     :math:`t_0 + W \\le N`, compute
                     :math:`\\operatorname{Dgm}_k(P_W(\\text{point\\_cloud},
                     t_0))` via ``ripser`` and extract the 5 statistics of
                     ``_diagram_statistics``. Pool across windows using each
                     function listed in ``config.pooling``. Concatenate in
                     row-major order
                     ``[pool_0, pool_1, \\ldots] \\times [dim_0, \\ldots,
                     dim_K] \\times [\\text{stat}_0, \\ldots, \\text{stat}_4]``.
                     Determinism: identical inputs produce bitwise identical
                     outputs.
    Complexity:      :math:`O(w \\cdot W^{2K+2})` worst case, where
                     :math:`w = \\lfloor (N - W) / \\Delta\\rfloor + 1`
                     (Bauer, 2021, §4, on ripser's complexity).

    Parameters
    ----------
    point_cloud : numpy.ndarray, shape (N, k)
        Input point cloud (the rows of the signal matrix, or the rows
        of a delay-embedded signal).
    config : TopologyFeatureConfig, optional
        Configuration. Defaults to ``TopologyFeatureConfig()``.

    Returns
    -------
    numpy.ndarray, shape (n_pool * (K+1) * 5,), dtype float64
        Pooled topology feature vector.
    """
    if point_cloud.ndim != 2:
        raise ValueError(
            f"point_cloud must be 2D (N, k); got shape {point_cloud.shape}"
        )
    cfg = config if config is not None else TopologyFeatureConfig()
    if cfg.window_length < 2:
        raise ValueError(f"window_length must be >= 2; got {cfg.window_length}")
    if cfg.stride < 1:
        raise ValueError(f"stride must be >= 1; got {cfg.stride}")
    if cfg.max_homology_dim < 0:
        raise ValueError(
            f"max_homology_dim must be >= 0; got {cfg.max_homology_dim}"
        )
    if not cfg.pooling:
        raise ValueError("pooling must contain at least one entry")

    X = np.ascontiguousarray(point_cloud, dtype=np.float64)  # §1.3
    N, _ = X.shape
    W = cfg.window_length
    if W > N:
        # Single short window: use the whole cloud as one window.
        W = N
    stride = cfg.stride
    K = cfg.max_homology_dim
    n_dims = K + 1

    window_starts = np.arange(0, N - W + 1, stride, dtype=np.intp)
    n_windows = window_starts.shape[0]
    if n_windows == 0:  # pragma: no cover
        # Reached only if `window_starts` is empty. Since W is capped above
        # to ``min(W, N)`` and stride >= 1 is validated, ``N - W + 1 >= 1``
        # always holds, so np.arange yields at least one start. Defensive.
        n_pool = len(cfg.pooling)
        return np.zeros(n_pool * n_dims * _N_STATS_PER_DIM, dtype=np.float64)

    # Per-window per-dimension stats:  (n_windows, n_dims, 5).
    per_window_stats = np.zeros(
        (n_windows, n_dims, _N_STATS_PER_DIM), dtype=np.float64
    )

    ripser_kwargs: dict[str, float | int | bool] = {
        "maxdim": K,
    }
    if cfg.edge_threshold is not None:
        ripser_kwargs["thresh"] = float(cfg.edge_threshold)

    # Loop over windows. This is NOT a Python sample loop (§3.1) — windows
    # are a parameter sweep, not the inner per-sample iteration; the inner
    # computation (ripser) is C++.
    for w_idx in range(n_windows):
        t0 = int(window_starts[w_idx])
        window = X[t0 : t0 + W]
        # Avoid degenerate ripser calls when the window has duplicate points.
        if np.unique(window, axis=0).shape[0] < 2:
            continue
        # The public contract above fixes rows as points. ripser's heuristic
        # warning for k > N therefore cannot indicate a transposed input here.
        # Silence only that exact false positive; all other warnings propagate.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"The input point cloud has more columns than rows; "
                    r"did you mean to transpose\?"
                ),
                category=UserWarning,
                module=r"ripser\.ripser",
            )
            result = ripser(window, **ripser_kwargs)
        diagrams: list[NDArray[np.float64]] = result["dgms"]

        # Scale normalization (REMARK 4.4): divide lifetimes by window diameter.
        if cfg.scale_normalize:
            diff = window[:, None, :] - window[None, :, :]
            window_diam = float(
                np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0)).max()
            )
            denom = window_diam + 1e-300  # §1.4
        else:
            denom = 1.0

        for k in range(n_dims):
            if k < len(diagrams):  # pragma: no branch
                # ripser(maxdim=K) returns exactly K + 1 == n_dims diagrams, so
                # k is always in range; the false edge is a defensive guard.
                dgm = diagrams[k]
                if cfg.scale_normalize and dgm.shape[0] > 0:
                    dgm = dgm.copy()
                    finite_rows = np.isfinite(dgm[:, 1])
                    dgm[finite_rows, 0] /= denom
                    dgm[finite_rows, 1] /= denom
                per_window_stats[w_idx, k] = _diagram_statistics(dgm)

    # Pool across windows for each pooling function.
    pooled_blocks: list[NDArray[np.float64]] = []
    for how in cfg.pooling:
        pooled = _pool(per_window_stats, how)  # shape (n_dims, 5)
        pooled_blocks.append(pooled.ravel())
    feature_vector = np.concatenate(pooled_blocks, axis=0)

    # §1.3: enforce dtype on return.
    return np.ascontiguousarray(feature_vector, dtype=np.float64)


def topology_feature_names(config: TopologyFeatureConfig) -> list[str]:
    """
    Return the human-readable name of each entry in the feature vector
    produced by ``sliding_window_topology_features`` under ``config``.

    The order matches the row-major layout of the output vector:
    ``[pool] × [homology_dim] × [statistic]``.
    """
    n_dims = config.max_homology_dim + 1
    names: list[str] = []
    for pool in config.pooling:
        for k in range(n_dims):
            for stat in _DIAGRAM_STATISTIC_NAMES:
                names.append(f"{pool}__H{k}__{stat}")
    return names
