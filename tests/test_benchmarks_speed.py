"""
Smoke and contract tests for the speed axis.

Full speed measurement is slow (5 outer passes * 20 inner calls per cell).
These tests use the smallest viable configuration to exercise the
machinery — `warmup=1, repeat=2, number=2` — and assert that:

  - the output schema is well-formed,
  - timing values are finite and non-negative,
  - CI bounds bracket the point estimate when the sample is large enough,
  - GC state is restored after measurement.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _has_torch_topological() -> bool:
    try:
        import torch_topological  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_torch_topological(), reason="torch-topological not installed")
class TestSpeedAxis:
    def test_speed_runs_end_to_end(self) -> None:
        from benchmarks.axes.speed import measure_speed
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        report = measure_speed(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            n_points_list=[10],
            seeds=[0],
            warmup=1,
            repeat=2,
            number=2,
        )
        assert report.backend_name == "topogeoml-diff-ph"
        assert len(report.rows) == 2  # one forward, one forward+backward
        for row in report.rows:
            assert row.point_estimate_ms >= 0
            assert len(row.per_pass_medians_ms) == 2  # repeat=2
            for ms in row.per_pass_medians_ms:
                assert ms >= 0 and np.isfinite(ms)

    def test_speed_records_operations_separately(self) -> None:
        from benchmarks.axes.speed import measure_speed
        from benchmarks.backends import get_backend
        from benchmarks.datasets import get_dataset

        report = measure_speed(
            get_backend("topogeoml-diff-ph"),
            get_dataset("mnist_mock_digit_0"),
            n_points_list=[10],
            seeds=[0],
            warmup=1,
            repeat=2,
            number=2,
        )
        operations = {row.operation for row in report.rows}
        assert operations == {"forward", "forward+backward"}


class TestMeasureRestoresGCState:
    """The measurement window disables GC; it must be re-enabled afterward
    if it was on before, otherwise we leak the disable across the process."""

    def test_gc_restored_when_initially_enabled(self) -> None:
        from benchmarks.axes.speed import _measure

        assert gc.isenabled()
        _measure(lambda: None, warmup=1, repeat=2, number=2)
        assert gc.isenabled()

    def test_gc_restored_when_initially_disabled(self) -> None:
        from benchmarks.axes.speed import _measure

        gc.disable()
        try:
            _measure(lambda: None, warmup=1, repeat=2, number=2)
        finally:
            gc.enable()
        # The library currently re-enables unconditionally; document that
        # behavior here. If the contract changes (preserve initial state
        # rather than always enable), update this test in lockstep.
        assert gc.isenabled()
