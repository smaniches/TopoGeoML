"""
Tests for RipsFiltration.

Validates that ripser-backed Rips persistence correctly recovers the known
topology of canonical synthetic point clouds:
  - circle: H_0=1, H_1=1
  - line:   H_0=1, H_1=0
  - 2 disjoint circles: H_0=2 (one infinite + one long), H_1=2
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from topogeoml.core.filtrations import RipsFiltration


def _count_significant_bars(
    bars: NDArray[np.float64],
    persistence_threshold: float,
) -> int:
    """Count bars (incl. infinite) with persistence (death - birth) >= threshold."""
    if bars.size == 0:
        return 0
    deaths = bars[:, 1].copy()
    births = bars[:, 0]
    # Infinite deaths are always significant.
    persistent = (deaths - births) >= persistence_threshold
    return int(persistent.sum())


def test_rips_recovers_circle_topology(noisy_circle: NDArray[np.float64]) -> None:
    """Noisy circle should have one significant H_1 bar."""
    rips = RipsFiltration(max_homology_dim=1)
    diag = rips.compute(noisy_circle)
    assert diag.max_dim == 1
    # H_1: one prominent loop. Threshold chosen well above noise scale (0.05).
    n_loops = _count_significant_bars(diag.bars[1], persistence_threshold=0.5)
    assert n_loops == 1, f"expected 1 significant loop, got {n_loops}"


def test_rips_recovers_line_topology(noisy_line: NDArray[np.float64]) -> None:
    """Noisy line should have zero significant H_1 bars."""
    rips = RipsFiltration(max_homology_dim=1)
    diag = rips.compute(noisy_line)
    n_loops = _count_significant_bars(diag.bars[1], persistence_threshold=0.5)
    assert n_loops == 0, f"expected 0 loops on a line, got {n_loops}"


def test_rips_recovers_two_circles(noisy_two_circles: NDArray[np.float64]) -> None:
    """Two disjoint circles should have 2 significant H_1 bars."""
    rips = RipsFiltration(max_homology_dim=1)
    diag = rips.compute(noisy_two_circles)
    n_loops = _count_significant_bars(diag.bars[1], persistence_threshold=0.5)
    assert n_loops == 2, f"expected 2 loops on two circles, got {n_loops}"


def test_provenance_recorded(noisy_circle: NDArray[np.float64]) -> None:
    rips = RipsFiltration(max_homology_dim=1, metric="euclidean", coeff=2)
    diag = rips.compute(noisy_circle)
    p = diag.provenance
    assert p.filtration == "rips"
    assert p.metric == "euclidean"
    assert p.max_homology_dim == 1
    assert p.n_points == 50
    assert p.ambient_dim == 2
    assert p.extra.get("coeff") == 2
    assert p.extra.get("backend") == "ripser"


def test_rips_float64_output(noisy_circle: NDArray[np.float64]) -> None:
    """All output bars must be float64 (elite-code-standards §1.3)."""
    rips = RipsFiltration(max_homology_dim=1)
    diag = rips.compute(noisy_circle)
    for dim, arr in diag.bars.items():
        assert arr.dtype == np.float64, f"dim {dim}: dtype is {arr.dtype}"


def test_rips_rejects_invalid_max_homology_dim() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        RipsFiltration(max_homology_dim=-1)


def test_rips_rejects_invalid_max_edge_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        RipsFiltration(max_edge_length=0.0)


def test_rips_rejects_non_2d_input() -> None:
    rips = RipsFiltration(max_homology_dim=0)
    with pytest.raises(ValueError, match="2D"):
        rips.compute(np.array([1.0, 2.0, 3.0]))


def test_rips_with_max_edge_length(noisy_circle: NDArray[np.float64]) -> None:
    """A small edge cutoff should produce more (smaller) connected components."""
    rips_unbounded = RipsFiltration(max_homology_dim=0)
    rips_bounded = RipsFiltration(max_homology_dim=0, max_edge_length=0.05)

    diag_u = rips_unbounded.compute(noisy_circle)
    diag_b = rips_bounded.compute(noisy_circle)

    # Bounded should have more components persisting (cutoff before they merge).
    assert diag_b.n_bars(0) >= diag_u.n_bars(0)
