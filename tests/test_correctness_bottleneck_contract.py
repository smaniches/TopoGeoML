"""Numerical contract for persistence-diagram comparison against ripser.

Finite persistence diagrams are multisets modulo diagonal matching. Sorting
barcode rows by nearly equal birth coordinates is not a valid matching.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gudhi")

from benchmarks.axes.correctness import measure_correctness


class _FixedCloud:
    name = "contract_cloud"
    version = "1.0.0"

    def generate(self, seed: int, n_points: int) -> torch.Tensor:
        return torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
        )

    def expected_h1(self, n_points: int) -> int:
        return -1


def _compare_diagrams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reference_h0: np.ndarray,
    reference_h1: np.ndarray,
    actual_h0: np.ndarray,
    actual_h1: np.ndarray,
):
    import ripser

    monkeypatch.setattr(
        ripser,
        "ripser",
        lambda *_args, **_kwargs: {"dgms": [reference_h0, reference_h1]},
    )

    class _Backend:
        name = "fixed_backend"
        version = "1.0.0"

        @staticmethod
        def compute_diagram(X: torch.Tensor, max_dim: int) -> list[torch.Tensor]:
            return [
                torch.from_numpy(actual_h0.copy()),
                torch.from_numpy(actual_h1.copy()),
            ]

    return measure_correctness(_Backend, _FixedCloud(), seeds=[0])


def test_lexicographic_bar_alignment_is_not_diagram_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array(
        [[0.5, 0.8], [0.50000001, 1.2]], dtype=np.float64
    )
    actual = np.array(
        [[0.50000001, 0.8], [0.5, 1.2]], dtype=np.float64
    )
    h0 = np.array([[0.0, 1.0]], dtype=np.float64)
    report = _compare_diagrams(
        monkeypatch,
        reference_h0=h0,
        reference_h1=reference,
        actual_h0=h0,
        actual_h1=actual,
    )
    seed = report.per_seed[0]
    assert seed.max_abs_diff_h1 > 0.3  # legacy sorting is misleading
    assert seed.bottleneck_h1 < 1e-6  # valid optimal matching
    assert seed.diagram_match_pass and report.overall_pass


def test_extra_near_diagonal_bar_is_not_a_topological_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref_h0 = np.array([[0.0, 1.0]], dtype=np.float64)
    actual_h0 = np.array(
        [[0.0, 1.0], [0.0, 1e-10]], dtype=np.float64
    )
    empty = np.empty((0, 2), dtype=np.float64)
    report = _compare_diagrams(
        monkeypatch,
        reference_h0=ref_h0,
        reference_h1=empty,
        actual_h0=actual_h0,
        actual_h1=empty,
    )
    seed = report.per_seed[0]
    assert seed.max_abs_diff_h0 == float("inf")
    assert seed.bottleneck_h0 < 1e-6
    assert seed.n_finite_bars_h0_backend == 2
    assert seed.n_finite_bars_h0_ripser == 1
    assert report.overall_pass


def test_material_bar_mismatch_remains_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h0 = np.array([[0.0, 1.0]], dtype=np.float64)
    reference_h1 = np.array([[0.5, 0.8]], dtype=np.float64)
    actual_h1 = np.array([[0.5, 1.5]], dtype=np.float64)
    report = _compare_diagrams(
        monkeypatch,
        reference_h0=h0,
        reference_h1=reference_h1,
        actual_h0=h0,
        actual_h1=actual_h1,
    )
    seed = report.per_seed[0]
    assert seed.bottleneck_h1 > 1e-6
    assert not seed.diagram_match_pass
    assert not report.overall_pass
