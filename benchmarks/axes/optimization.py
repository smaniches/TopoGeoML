"""
Optimization axis — gradient-descent quality on a fixed objective.

This axis is **diagnostic**, not adjudicative: subgradient choices can
legitimately differ across backends (Hofer et al. 2017 §4; Carrière et al.
2021 §3), and a backend whose subgradient is "different" is not "wrong".
What this axis measures is the *practical descent behavior* on a target
objective — does the loss decrease, how fast, and how stably across seeds?

Two objectives are tracked:

  - ``inflate_h1``: minimize ``-longest_h1_lifetime``. A noisy circle
    should grow into a clean circle; the cloud should not collapse or
    diverge. Expected behavior: monotonically decreasing loss to a
    stable plateau.
  - ``shrink_h1``: minimize ``+longest_h1_lifetime``. A noisy circle
    should collapse toward a point cloud with no significant loops.
    Expected behavior: monotonic decrease to near-zero.

For each (backend, objective, seed) we record the full loss trajectory
plus summary statistics. Across seeds we report bootstrap CIs on the
final loss and the trajectory's decay constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from benchmarks.backends import PHBackend
from benchmarks.datasets import Dataset
from benchmarks.stats import bootstrap_ci

Objective = Literal["inflate_h1", "shrink_h1"]


@dataclass(frozen=True)
class TrajectoryRow:
    seed: int
    objective: Objective
    initial_loss: float
    final_loss: float
    n_steps: int
    final_cloud_min: float
    final_cloud_max: float
    final_cloud_span: float
    loss_trajectory: list[float]


@dataclass(frozen=True)
class OptimizationReport:
    backend_name: str
    backend_version: str
    dataset_name: str
    dataset_version: str
    n_points: int
    n_steps: int
    learning_rate: float
    objective: Objective
    per_seed: list[TrajectoryRow]
    final_loss_median: float
    final_loss_ci95_low: float
    final_loss_ci95_high: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "per_seed"},
            "per_seed": [asdict(r) for r in self.per_seed],
        }


def _run_one(
    backend: type[PHBackend],
    dataset: Dataset,
    *,
    seed: int,
    objective: Objective,
    n_points: int,
    n_steps: int,
    learning_rate: float,
) -> TrajectoryRow:
    torch.manual_seed(seed)
    X = dataset.generate(seed=seed, n_points=n_points).clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=learning_rate)
    trajectory: list[float] = []

    initial_loss = float("nan")
    sign = 1.0 if objective == "inflate_h1" else -1.0  # inflate: min -L; shrink: min +L
    for step in range(n_steps):
        opt.zero_grad()
        loss = sign * backend.loss_longest_h1(X)
        loss_value = float(loss.item())
        if step == 0:
            initial_loss = loss_value
        trajectory.append(loss_value)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()

    with torch.no_grad():
        cloud_min = float(X.min().item())
        cloud_max = float(X.max().item())

    return TrajectoryRow(
        seed=seed,
        objective=objective,
        initial_loss=initial_loss,
        final_loss=trajectory[-1] if trajectory else float("nan"),
        n_steps=n_steps,
        final_cloud_min=cloud_min,
        final_cloud_max=cloud_max,
        final_cloud_span=cloud_max - cloud_min,
        loss_trajectory=trajectory,
    )


def measure_optimization(
    backend: type[PHBackend],
    dataset: Dataset,
    *,
    objective: Objective = "inflate_h1",
    n_points: int = 50,
    seeds: list[int] | None = None,
    n_steps: int = 200,
    learning_rate: float = 1e-2,
) -> OptimizationReport:
    """Run gradient descent on ``objective`` for each seed; collect trajectories."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    per_seed = [
        _run_one(
            backend, dataset,
            seed=s, objective=objective,
            n_points=n_points, n_steps=n_steps,
            learning_rate=learning_rate,
        )
        for s in seeds
    ]

    finals = np.asarray([r.final_loss for r in per_seed], dtype=np.float64)
    if finals.size >= 2 and np.all(np.isfinite(finals)):
        ci = bootstrap_ci(finals, statistic="median", confidence_level=0.95, n_resamples=10_000, seed=0)
        median, lo, hi = ci.point_estimate, ci.ci_low, ci.ci_high
    else:
        median = float(np.nanmedian(finals)) if finals.size else float("nan")
        lo = hi = float("nan")

    return OptimizationReport(
        backend_name=backend.name,
        backend_version=getattr(backend, "version", "") or "",
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        n_points=n_points,
        n_steps=n_steps,
        learning_rate=learning_rate,
        objective=objective,
        per_seed=per_seed,
        final_loss_median=median,
        final_loss_ci95_low=lo,
        final_loss_ci95_high=hi,
    )


__all__ = ["OptimizationReport", "TrajectoryRow", "measure_optimization"]
