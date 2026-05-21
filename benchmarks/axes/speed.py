"""
Speed axis — forward and forward+backward latency, measured rigorously.

Microbenchmarking is notoriously easy to do wrong. The protocol here
follows the principles of the ``pyperf`` reference (Stinner 2017) and
the SPEC CPU methodology:

  - **Warm-up**: discard the first call entirely; JIT, allocator, and
    cache effects bias the first invocation.
  - **Measurement window**: ``timeit.repeat(repeat=R, number=N)`` runs
    R independent measurement passes, each timing N calls. Inside the
    measurement window we **disable garbage collection** so a stray
    GC pause cannot dominate a microsecond-scale measurement
    (Python ``gc`` docs; recommended pattern for any benchmark).
  - **Aggregation**: per-pass median is the central tendency; the
    min-of-medians across passes is the reported point estimate
    (resistant to system noise; recommended by pyperf).
  - **Confidence**: 95% percentile bootstrap CI on the raw per-pass
    median samples, with a clear NaN fallback when ``repeat < 2``.
  - **Pairwise statistics**: when two backends are run on the same
    seed set, we use Wilcoxon signed-rank (paired) via the stats
    module; otherwise Mann–Whitney U (independent).

We do **not** attempt CPU pinning at the Python layer — that's the
runner's job (the benchmark CI workflow uses a single-job pool to avoid
multi-tenant interference, which is the practical equivalent on hosted
runners).

References
----------
Stinner, V. (2017). *pyperf* — Python performance benchmarking toolkit.
  https://pyperf.readthedocs.io/
"""

from __future__ import annotations

import gc
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from benchmarks.backends import PHBackend
from benchmarks.datasets import Dataset
from benchmarks.stats import BootstrapCI, bootstrap_ci


@dataclass(frozen=True)
class TimingRow:
    """One measurement row for a specific (backend, dataset, n, seed)."""

    backend_name: str
    dataset_name: str
    n_points: int
    seed: int
    operation: str  # "forward" or "forward+backward"
    # Raw per-pass median ms — the bootstrap input.
    per_pass_medians_ms: list[float]
    point_estimate_ms: float  # min of per_pass_medians
    ci95_low_ms: float
    ci95_high_ms: float


@dataclass(frozen=True)
class SpeedReport:
    backend_name: str
    backend_version: str
    dataset_name: str
    dataset_version: str
    rows: list[TimingRow]
    n_outer_repeats: int
    n_inner_calls: int
    warmup_calls: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "n_outer_repeats": self.n_outer_repeats,
            "n_inner_calls": self.n_inner_calls,
            "warmup_calls": self.warmup_calls,
            "rows": [asdict(r) for r in self.rows],
        }


def _ci_from_pass_medians(medians_ms: list[float]) -> tuple[float, float, float]:
    """Return (point_estimate_ms, ci95_low_ms, ci95_high_ms) for the pass-median sample."""
    if len(medians_ms) == 0:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(medians_ms, dtype=np.float64)
    if arr.size < 2:
        # bootstrap_ci requires n >= 2; degrade to point estimate only.
        return float(arr.min()), float("nan"), float("nan")
    ci: BootstrapCI = bootstrap_ci(arr, statistic="min", confidence_level=0.95, n_resamples=2000, seed=0)
    return float(arr.min()), ci.ci_low, ci.ci_high


def _measure(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    number: int,
) -> list[float]:
    """Time ``fn`` repeatedly, returning per-pass median millisecond values.

    Each "pass" times ``number`` calls and divides by ``number``. We
    return ``repeat`` such per-call medians. GC is disabled during the
    measurement window only.
    """
    for _ in range(warmup):
        fn()

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        per_pass_total_seconds = timeit.repeat(fn, repeat=repeat, number=number)
    finally:
        if gc_was_enabled:
            gc.enable()

    return [(t / number) * 1000.0 for t in per_pass_total_seconds]


def measure_speed(
    backend: type[PHBackend],
    dataset: Dataset,
    *,
    n_points_list: list[int] | None = None,
    seeds: list[int] | None = None,
    warmup: int = 2,
    repeat: int = 5,
    number: int = 20,
) -> SpeedReport:
    """Time forward and forward+backward calls across (n, seed) cells.

    Each cell yields ``repeat`` per-pass medians; the report stores the
    raw vector so downstream reports can bootstrap or run pairwise tests.
    """
    if n_points_list is None:
        n_points_list = [30, 100, 300]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    rows: list[TimingRow] = []

    for n in n_points_list:
        for seed in seeds:
            torch.manual_seed(seed)
            X_base = dataset.generate(seed=seed, n_points=n)
            X_np = X_base.detach().numpy().copy()

            # Forward only.
            def _fwd(X_np: np.ndarray = X_np) -> None:
                X = torch.from_numpy(X_np).to(torch.float64)
                _ = backend.compute_diagram(X, max_dim=1)

            fwd_medians = _measure(_fwd, warmup=warmup, repeat=repeat, number=number)
            pe, lo, hi = _ci_from_pass_medians(fwd_medians)
            rows.append(TimingRow(
                backend_name=backend.name, dataset_name=dataset.name,
                n_points=n, seed=seed, operation="forward",
                per_pass_medians_ms=fwd_medians,
                point_estimate_ms=pe, ci95_low_ms=lo, ci95_high_ms=hi,
            ))

            # Forward + backward on the loss.
            def _fwd_bwd(X_np: np.ndarray = X_np) -> None:
                X = torch.from_numpy(X_np).to(torch.float64).requires_grad_(True)
                loss = backend.loss_longest_h1(X)
                loss.backward()  # type: ignore[no-untyped-call]

            bwd_medians = _measure(_fwd_bwd, warmup=warmup, repeat=repeat, number=number)
            pe, lo, hi = _ci_from_pass_medians(bwd_medians)
            rows.append(TimingRow(
                backend_name=backend.name, dataset_name=dataset.name,
                n_points=n, seed=seed, operation="forward+backward",
                per_pass_medians_ms=bwd_medians,
                point_estimate_ms=pe, ci95_low_ms=lo, ci95_high_ms=hi,
            ))

    return SpeedReport(
        backend_name=backend.name,
        backend_version=getattr(backend, "version", "") or "",
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        rows=rows,
        n_outer_repeats=repeat,
        n_inner_calls=number,
        warmup_calls=warmup,
    )


__all__ = ["SpeedReport", "TimingRow", "measure_speed"]
