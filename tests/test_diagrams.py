"""Tests for PersistenceDiagram dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from topogeoml.core.diagrams import DiagramProvenance, PersistenceDiagram


def _provenance() -> DiagramProvenance:
    return DiagramProvenance(
        filtration="rips",
        metric="euclidean",
        max_homology_dim=1,
        max_edge_length=None,
        n_points=10,
        ambient_dim=2,
    )


def test_diagram_construction_valid() -> None:
    bars = {
        0: np.array([[0.0, 1.0], [0.0, np.inf]], dtype=np.float64),
        1: np.array([[0.2, 0.8]], dtype=np.float64),
    }
    diag = PersistenceDiagram(bars=bars, provenance=_provenance())
    assert diag.max_dim == 1
    assert diag.n_bars(0) == 2
    assert diag.n_bars(1) == 1
    assert diag.n_bars(2) == 0


def test_diagram_rejects_non_float64() -> None:
    bars = {0: np.array([[0.0, 1.0]], dtype=np.float32)}
    with pytest.raises(TypeError, match="float64"):
        PersistenceDiagram(bars=bars, provenance=_provenance())


def test_diagram_rejects_wrong_shape() -> None:
    bars = {0: np.array([0.0, 1.0], dtype=np.float64)}  # 1D
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        PersistenceDiagram(bars=bars, provenance=_provenance())


def test_diagram_rejects_negative_dim() -> None:
    bars = {-1: np.array([[0.0, 1.0]], dtype=np.float64)}
    with pytest.raises(ValueError, match="non-negative"):
        PersistenceDiagram(bars=bars, provenance=_provenance())


def test_lifetimes_finite_only() -> None:
    bars = {0: np.array([[0.0, 1.0], [0.0, np.inf], [0.5, 2.0]], dtype=np.float64)}
    diag = PersistenceDiagram(bars=bars, provenance=_provenance())
    lifetimes = diag.lifetimes(0, finite_only=True)
    assert lifetimes.shape == (2,)
    np.testing.assert_allclose(np.sort(lifetimes), [1.0, 1.5])


def test_lifetimes_with_infinite() -> None:
    bars = {0: np.array([[0.0, 1.0], [0.0, np.inf]], dtype=np.float64)}
    diag = PersistenceDiagram(bars=bars, provenance=_provenance())
    lifetimes = diag.lifetimes(0, finite_only=False)
    assert lifetimes.shape == (2,)
    assert np.isinf(lifetimes).sum() == 1


def test_total_persistence() -> None:
    bars = {1: np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)}
    diag = PersistenceDiagram(bars=bars, provenance=_provenance())
    assert diag.total_persistence(1, p=1.0) == pytest.approx(1.0 + 1.5)
    assert diag.total_persistence(1, p=2.0) == pytest.approx(1.0 + 2.25)
    assert diag.total_persistence(0, p=1.0) == 0.0  # empty dimension


def test_empty_diagram() -> None:
    diag = PersistenceDiagram(bars={}, provenance=_provenance())
    assert diag.max_dim == -1
    assert diag.n_bars(0) == 0
    assert diag.lifetimes(0).size == 0


def test_diagram_immutable() -> None:
    """Frozen dataclass: replacement requires new construction."""
    bars = {0: np.array([[0.0, 1.0]], dtype=np.float64)}
    diag = PersistenceDiagram(bars=bars, provenance=_provenance())
    with pytest.raises(Exception):  # FrozenInstanceError under dataclasses
        diag.bars = {}  # type: ignore[misc]


def test_repr_contains_summary() -> None:
    bars = {0: np.array([[0.0, 1.0]], dtype=np.float64)}
    diag = PersistenceDiagram(bars=bars, provenance=_provenance())
    s = repr(diag)
    assert "H0=1" in s
    assert "rips" in s
    assert "euclidean" in s
