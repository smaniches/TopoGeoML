"""Tests for diagram vectorizers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topogeoml.core.diagrams import DiagramProvenance, PersistenceDiagram
from topogeoml.core.filtrations import RipsFiltration
from topogeoml.core.vectorizers import (
    BettiCurveVectorizer,
    PersistenceImageVectorizer,
)


def _provenance() -> DiagramProvenance:
    return DiagramProvenance(
        filtration="rips",
        metric="euclidean",
        max_homology_dim=1,
        max_edge_length=None,
        n_points=50,
        ambient_dim=2,
    )


def _make_empty_diagram() -> PersistenceDiagram:
    bars = {0: np.empty((0, 2), dtype=np.float64), 1: np.empty((0, 2), dtype=np.float64)}
    return PersistenceDiagram(bars=bars, provenance=_provenance())


# ---------- PersistenceImageVectorizer ----------


def test_persistence_image_output_shape_and_dtype(noisy_circle: NDArray[np.float64]) -> None:
    rips = RipsFiltration(max_homology_dim=1)
    diag = rips.compute(noisy_circle)
    vec = PersistenceImageVectorizer(homology_dims=(0, 1), resolution=20, sigma=0.1, fallback_max=2.0)
    feats = vec.transform_one(diag)
    assert feats.dtype == np.float64
    assert feats.shape == (vec.output_dim,)
    assert feats.shape == (2 * 20 * 20,)


def test_persistence_image_empty_diagram_returns_zeros() -> None:
    vec = PersistenceImageVectorizer(homology_dims=(0, 1), resolution=10, fallback_max=1.0)
    feats = vec.transform_one(_make_empty_diagram())
    assert feats.shape == (2 * 10 * 10,)
    np.testing.assert_array_equal(feats, np.zeros_like(feats))


def test_persistence_image_discriminates_circle_vs_line(
    noisy_circle: NDArray[np.float64],
    noisy_line: NDArray[np.float64],
) -> None:
    """
    Circle vs line should produce visibly different feature vectors,
    especially in the H_1 channel.
    """
    rips = RipsFiltration(max_homology_dim=1)
    vec = PersistenceImageVectorizer(homology_dims=(0, 1), resolution=20, sigma=0.1, fallback_max=2.5)
    f_circle = vec.transform_one(rips.compute(noisy_circle))
    f_line = vec.transform_one(rips.compute(noisy_line))
    # The H_1 half (second 400 features) should be much heavier for the circle.
    h1_circle = f_circle[400:].sum()
    h1_line = f_line[400:].sum()
    assert h1_circle > h1_line, f"H_1 mass circle={h1_circle:.4f} should exceed line={h1_line:.4f}"


# ---------- BettiCurveVectorizer ----------


def test_betti_curve_output_shape_and_dtype(noisy_circle: NDArray[np.float64]) -> None:
    rips = RipsFiltration(max_homology_dim=1)
    diag = rips.compute(noisy_circle)
    vec = BettiCurveVectorizer(homology_dims=(0, 1), resolution=50, fallback_max=2.0)
    feats = vec.transform_one(diag)
    assert feats.dtype == np.float64
    assert feats.shape == (2 * 50,)


def test_betti_curve_h1_peak_on_circle(noisy_circle: NDArray[np.float64]) -> None:
    """β_1(t) should reach ≥1 somewhere along the filtration for a circle."""
    rips = RipsFiltration(max_homology_dim=1)
    diag = rips.compute(noisy_circle)
    vec = BettiCurveVectorizer(homology_dims=(1,), resolution=200, fallback_max=2.5)
    h1_curve = vec.transform_one(diag)
    assert h1_curve.max() >= 1.0


def test_betti_curve_h1_line_much_smaller_than_circle(
    noisy_line: NDArray[np.float64],
    noisy_circle: NDArray[np.float64],
) -> None:
    """
    β_1 mass should be dramatically smaller for a line than a circle.

    Noise at scale σ=0.05 can produce a handful of short-lived spurious loops
    at the noise scale; that's expected. The semantic invariant is that the
    line carries far less topological signal in H_1 than the circle.
    """
    rips = RipsFiltration(max_homology_dim=1)
    vec = BettiCurveVectorizer(homology_dims=(1,), resolution=200, fallback_max=2.5)
    h1_line = vec.transform_one(rips.compute(noisy_line))
    h1_circle = vec.transform_one(rips.compute(noisy_circle))
    # Circle's integral should dominate by at least 10x.
    line_integral = float(h1_line.sum())
    circle_integral = float(h1_circle.sum())
    assert circle_integral > 10.0 * line_integral, (
        f"expected circle ≫ line in H_1; got line={line_integral:.2f}, "
        f"circle={circle_integral:.2f}"
    )


def test_betti_curve_h0_starts_at_n_points(noisy_circle: NDArray[np.float64]) -> None:
    """At t=0 (or just above), β_0 should equal n_points (every point its own component)."""
    rips = RipsFiltration(max_homology_dim=0)
    diag = rips.compute(noisy_circle)
    vec = BettiCurveVectorizer(homology_dims=(0,), resolution=100, fallback_max=2.5)
    h0 = vec.transform_one(diag)
    # First sample at t=0: all 50 points are alive.
    assert h0[0] == 50.0
