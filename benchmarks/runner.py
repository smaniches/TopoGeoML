"""
Benchmark runner — orchestrate (backend, dataset, axis) cells with full
provenance, isolation, and reproducibility.

Design invariants:

  - One failing cell does not crash the run. Errors are captured per-cell
    so a partial run still yields useful results.
  - Every result carries the provenance needed to reproduce it: git SHA
    (with a dirty flag), Python and library versions, the host OS / CPU /
    RAM fingerprint, and the seed list. Without that, results are not
    citable.
  - The runner does not mutate global state visibly. Seeds are scoped
    to axes that need them; ``torch.use_deterministic_algorithms(True)``
    is set at runner entry, restored at exit.

The output format is JSON with a schema versioned in ``leaderboard/schema.json``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from benchmarks.backends import PHBackend, available_backends, get_backend
from benchmarks.datasets import Dataset, get_dataset

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class Provenance:
    """All the metadata needed to re-create a benchmark result."""

    schema_version: str
    timestamp_utc: str
    git_sha: str  # empty string if not a git checkout
    git_dirty: bool
    python_version: str
    torch_version: str
    numpy_version: str
    scipy_version: str
    topogeoml_version: str
    torch_topological_version: str
    platform_string: str
    cpu_count: int
    process_memory_total_mb: int
    deterministic_algorithms_set: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CellResult:
    backend_name: str
    dataset_name: str
    axis_name: str
    success: bool
    payload: dict[str, Any] | None
    error_kind: str | None
    error_message: str | None
    error_traceback: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    provenance: Provenance
    config: dict[str, Any]
    cells: list[CellResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.as_dict(),
            "config": self.config,
            "cells": [c.as_dict() for c in self.cells],
        }


def _git_state() -> tuple[str, bool]:
    """Return (sha, dirty) for the current working tree, or ('', False) if not git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        return sha, bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):  # pragma: no cover
        # Defensive: this PR's CI runs inside a git checkout. Exercised when
        # the bench is invoked from an unpacked tarball with no git binary.
        return "", False


def _version_or_missing(pkg: str) -> str:
    from importlib import metadata
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:  # pragma: no cover
        # Defensive: bench extras pin every package we look up.
        return ""


def _process_memory_total_mb() -> int:
    """Total system memory in MiB; cross-platform best-effort."""
    try:
        # POSIX.
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (AttributeError, ValueError, OSError):  # pragma: no cover
        # Non-POSIX platforms (Windows) lack sysconf; bench supports it but
        # the fallback would only fire there.
        return 0


def _provenance() -> Provenance:
    sha, dirty = _git_state()
    return Provenance(
        schema_version=SCHEMA_VERSION,
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=sha,
        git_dirty=dirty,
        python_version=platform.python_version(),
        torch_version=_version_or_missing("torch"),
        numpy_version=_version_or_missing("numpy"),
        scipy_version=_version_or_missing("scipy"),
        topogeoml_version=_version_or_missing("topogeoml"),
        torch_topological_version=_version_or_missing("torch-topological"),
        platform_string=platform.platform(),
        cpu_count=os.cpu_count() or 0,
        process_memory_total_mb=_process_memory_total_mb(),
        deterministic_algorithms_set=True,
    )


# ---------------------------------------------------------------------------
# Axis dispatch table
# ---------------------------------------------------------------------------

AxisFn = Callable[..., Any]


def _axis_correctness(backend: type[PHBackend], dataset: Dataset, **kwargs: Any) -> dict[str, Any]:
    from benchmarks.axes.correctness import measure_correctness
    return measure_correctness(backend, dataset, **kwargs).as_dict()


def _axis_stability(backend: type[PHBackend], dataset: Dataset, **kwargs: Any) -> dict[str, Any]:
    from benchmarks.axes.stability import measure_stability
    return measure_stability(backend, dataset, **kwargs).as_dict()


def _axis_speed(backend: type[PHBackend], dataset: Dataset, **kwargs: Any) -> dict[str, Any]:
    from benchmarks.axes.speed import measure_speed
    return measure_speed(backend, dataset, **kwargs).as_dict()


def _axis_optimization(backend: type[PHBackend], dataset: Dataset, **kwargs: Any) -> dict[str, Any]:
    from benchmarks.axes.optimization import measure_optimization
    return measure_optimization(backend, dataset, **kwargs).as_dict()


AXES: dict[str, AxisFn] = {
    "correctness": _axis_correctness,
    "stability": _axis_stability,
    "speed": _axis_speed,
    "optimization": _axis_optimization,
}

#: Axes that invoke ``backend.loss_longest_h1`` and therefore require
#: ``backend.differentiable == True``. The runner skips non-differentiable
#: backends on these axes (with an explicit ``SkippedNonDifferentiable``
#: cell so the skip is visible in the report).
_DIFFERENTIABLE_AXES: frozenset[str] = frozenset({"stability", "speed", "optimization"})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    *,
    backend_names: list[str] | None = None,
    dataset_names: list[str] | None = None,
    axis_names: list[str] | None = None,
    axis_kwargs: dict[str, dict[str, Any]] | None = None,
) -> RunResult:
    """Execute every (backend × dataset × axis) cell that's available.

    Parameters
    ----------
    backend_names, dataset_names, axis_names
        Optional explicit selection. When ``None``, includes every
        available backend, every registered dataset, and every axis.
    axis_kwargs
        Optional per-axis keyword overrides forwarded to each
        ``measure_*`` function. Use this to thin the bench for CI
        budgets — e.g. ``{"speed": {"n_points_list": [30, 100],
        "repeat": 3, "number": 10}}``. Defaults to ``{}`` (each axis
        uses its statistical-rigor defaults).
    """
    # Default selections.
    if backend_names is None:
        backend_names = [cls.name for cls in available_backends()]
    if dataset_names is None:
        from benchmarks.datasets import all_dataset_names
        dataset_names = list(all_dataset_names())
    if axis_names is None:
        axis_names = list(AXES)

    # Validate axis names early — silent typos here would skip measurements
    # without a clear signal.
    unknown_axes = [a for a in axis_names if a not in AXES]
    if unknown_axes:
        raise KeyError(
            f"unknown axes: {unknown_axes}; available: {list(AXES)}"
        )

    # Determinism guarantees, scoped to the run.
    prior_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        prov = _provenance()
        run_result = RunResult(
            provenance=prov,
            config={
                "backend_names": backend_names,
                "dataset_names": dataset_names,
                "axis_names": axis_names,
            },
        )

        for backend_name in backend_names:
            backend_cls = get_backend(backend_name)
            if not backend_cls.available():
                run_result.cells.append(CellResult(
                    backend_name=backend_name,
                    dataset_name="*",
                    axis_name="*",
                    success=False,
                    payload=None,
                    error_kind="UnavailableBackend",
                    error_message=f"backend {backend_name!r} is registered but its dependencies are not importable",
                    error_traceback=None,
                ))
                continue

            for dataset_name in dataset_names:
                dataset = get_dataset(dataset_name)

                for axis_name in axis_names:
                    axis_fn = AXES[axis_name]

                    # Skip differentiability-requiring axes for non-diff
                    # backends — but record the skip so reports surface it.
                    if (
                        axis_name in _DIFFERENTIABLE_AXES
                        and not getattr(backend_cls, "differentiable", True)
                    ):
                        run_result.cells.append(CellResult(
                            backend_name=backend_name,
                            dataset_name=dataset_name,
                            axis_name=axis_name,
                            success=False,
                            payload=None,
                            error_kind="SkippedNonDifferentiable",
                            error_message=(
                                f"backend {backend_name!r} is non-differentiable; "
                                f"axis {axis_name!r} requires autograd through loss_longest_h1"
                            ),
                            error_traceback=None,
                        ))
                        continue

                    try:
                        kwargs = (axis_kwargs or {}).get(axis_name, {})
                        payload = axis_fn(backend_cls, dataset, **kwargs)
                        run_result.cells.append(CellResult(
                            backend_name=backend_name,
                            dataset_name=dataset_name,
                            axis_name=axis_name,
                            success=True,
                            payload=payload,
                            error_kind=None,
                            error_message=None,
                            error_traceback=None,
                        ))
                    except Exception as exc:
                        run_result.cells.append(CellResult(
                            backend_name=backend_name,
                            dataset_name=dataset_name,
                            axis_name=axis_name,
                            success=False,
                            payload=None,
                            error_kind=type(exc).__name__,
                            error_message=str(exc),
                            error_traceback=traceback.format_exc(),
                        ))

        return run_result
    finally:
        # Restore the prior determinism setting; never leave a flag set that
        # the caller didn't ask for.
        torch.use_deterministic_algorithms(prior_determinism, warn_only=True)


def write_result(result: RunResult, path: Path) -> None:
    """Atomically write the run result as JSON. UTF-8, sorted keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = result.as_dict()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


__all__ = [
    "AXES",
    "SCHEMA_VERSION",
    "CellResult",
    "Provenance",
    "RunResult",
    "run",
    "write_result",
]
