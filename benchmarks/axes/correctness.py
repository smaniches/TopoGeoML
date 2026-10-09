"""
Correctness axis — persistence-diagram agreement with ripser and dtype
preservation, with both bottleneck-distance verdicts and legacy sorted
elementwise differences retained as diagnostics.

This axis is a *pass/fail* axis: finite persistence diagrams are compared
as multisets under bottleneck distance (including diagonal matching) rather
than by lexicographically aligned rows. A tiny change in a repeated birth
time can otherwise reorder bars with different deaths and create a false
failure. Near-diagonal bars can be matched to the diagonal, as required by
the persistence-diagram metric.

The default numerical tolerance is 1e-6. That is an evaluation convention,
not a mathematical bound on ripser's numerical error; the original sorted
elementwise maxima are preserved for historical comparisons. Passing the
forward-diagram test does not establish correctness of the chosen gradients.

References
----------
Bauer, U. (2021). "Ripser: efficient computation of Vietoris-Rips
  persistence barcodes." *Journal of Applied and Computational Topology*,
  5(3), 391-423.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from benchmarks.axes.stability import _bottleneck_distance_finite
from benchmarks.backends import PHBackend
from benchmarks.datasets import Dataset


@dataclass(frozen=True)
class CorrectnessSeedResult:
    seed: int
    n_finite_bars_h0_backend: int
    n_finite_bars_h0_ripser: int
    n_finite_bars_h1_backend: int
    n_finite_bars_h1_ripser: int
    max_abs_diff_h0: float  # legacy sorted-elementwise diagnostic; not the verdict
    max_abs_diff_h1: float
    bottleneck_h0: float
    bottleneck_h1: float
    dtype_preserved: bool
    diagram_match_pass: bool


@dataclass(frozen=True)
class CorrectnessReport:
    backend_name: str
    backend_version: str
    dataset_name: str
    dataset_version: str
    n_points: int
    atol: float
    per_seed: list[CorrectnessSeedResult]
    overall_pass: bool  # True iff every per-seed bottleneck verdict and dtype pass

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "n_points": self.n_points,
            "atol": self.atol,
            "per_seed": [asdict(s) for s in self.per_seed],
            "overall_pass": self.overall_pass,
        }


def _sorted_finite(bars: np.ndarray) -> NDArray[np.float64]:
    """Return bars with infinite-death rows removed and rows sorted lexicographically.

    Sorting is required because ripser and backend may emit bars in
    different orders even when the set of bars is identical.
    """
    if bars.size == 0:
        return np.asarray(bars.reshape(0, 2), dtype=np.float64)
    finite = bars[np.isfinite(bars).all(axis=1)]
    if finite.size == 0:
        return np.asarray(finite.reshape(0, 2), dtype=np.float64)
    order = np.lexsort((finite[:, 1], finite[:, 0]))
    return np.asarray(finite[order], dtype=np.float64)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a_i - b_i| after sorted-elementwise alignment.

    Returns ``+inf`` when the two arrays have different lengths, signaling
    a mismatch in *number* of finite bars — which the diagram-match-pass
    check then flags as a failure.
    """
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def measure_correctness(
    backend: type[PHBackend],
    dataset: Dataset,
    *,
    n_points: int = 50,
    seeds: list[int] | None = None,
    atol: float = 1e-6,
) -> CorrectnessReport:
    """Compare backend output to the ripser reference, per seed.

    ripser is the *de facto* gold standard for Vietoris-Rips persistence
    (Bauer 2021) and is one of TopoGeoML's own core dependencies. Both
    backends in Phase 1 wrap ripser-compatible computations under the
    hood; this axis exists to catch silent regressions when that contract
    is violated.
    """
    from ripser import ripser

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    per_seed: list[CorrectnessSeedResult] = []
    overall_pass = True

    for seed in seeds:
        torch.manual_seed(seed)
        X = dataset.generate(seed=seed, n_points=n_points)
        ref = ripser(X.numpy(), maxdim=1)["dgms"]
        ref_h0 = _sorted_finite(ref[0])
        ref_h1 = _sorted_finite(ref[1])

        dgms = backend.compute_diagram(X, max_dim=1)
        # Per PHBackend contract, ``dgms`` has exactly ``max_dim + 1`` entries;
        # guard the indexing anyway so a non-conforming backend produces a
        # clean failure rather than an IndexError-during-iteration that hides
        # the real defect. (PR #3 review: gemini-code-assist.)
        if len(dgms) > 0 and dgms[0].numel():
            backend_h0 = _sorted_finite(dgms[0].detach().numpy())
        else:  # pragma: no cover
            # Defensive: PHBackend.compute_diagram is contractually required
            # to return ``max_dim + 1`` non-None tensors. All 4 registered
            # backends satisfy this; the empty fallback exists for backends
            # added later whose contract may be looser.
            backend_h0 = np.empty((0, 2))
        if len(dgms) > 1 and dgms[1].numel():
            backend_h1 = _sorted_finite(dgms[1].detach().numpy())
        else:  # pragma: no cover
            # Same defensive rationale as the H_0 path above.
            backend_h1 = np.empty((0, 2))

        max_h0 = _max_abs_diff(backend_h0, ref_h0)
        max_h1 = _max_abs_diff(backend_h1, ref_h1)
        # Compare diagram multisets, not row ordering. GUDHI's bottleneck
        # matching allows diagonal matches for zero/near-zero finite bars.
        bottleneck_h0 = _bottleneck_distance_finite(backend_h0, ref_h0)
        bottleneck_h1 = _bottleneck_distance_finite(backend_h1, ref_h1)

        dtype_preserved = all(
            d.dtype == torch.float64 if isinstance(d, torch.Tensor) else True
            for d in dgms
        )

        diagram_match = bottleneck_h0 <= atol and bottleneck_h1 <= atol
        if not (diagram_match and dtype_preserved):
            overall_pass = False

        per_seed.append(CorrectnessSeedResult(
            seed=seed,
            n_finite_bars_h0_backend=backend_h0.shape[0],
            n_finite_bars_h0_ripser=ref_h0.shape[0],
            n_finite_bars_h1_backend=backend_h1.shape[0],
            n_finite_bars_h1_ripser=ref_h1.shape[0],
            max_abs_diff_h0=max_h0,
            max_abs_diff_h1=max_h1,
            bottleneck_h0=bottleneck_h0,
            bottleneck_h1=bottleneck_h1,
            dtype_preserved=dtype_preserved,
            diagram_match_pass=diagram_match,
        ))

    return CorrectnessReport(
        backend_name=backend.name,
        backend_version=getattr(backend, "version", "") or "",
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        n_points=n_points,
        atol=atol,
        per_seed=per_seed,
        overall_pass=overall_pass,
    )


__all__ = ["CorrectnessReport", "CorrectnessSeedResult", "measure_correctness"]
