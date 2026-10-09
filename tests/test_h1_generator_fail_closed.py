"""Issue #114: critical-gradient pairing rejects incomplete or inconsistent evidence.

These are defensive numerical contracts, not invented results. GUDHI is an
optional dependency for the selected tied-filtration generator convention.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gudhi")

from topogeoml.nn.diff_ph import _critical_edges_h1, _gudhi_h1_generator_edges


def _square_distances() -> np.ndarray:
    square = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    return np.linalg.norm(square[:, None, :] - square[None, :, :], axis=2)


def test_gudhi_returns_no_generator_matches_for_no_requested_finite_bars() -> None:
    matched = _gudhi_h1_generator_edges(
        _square_distances(), np.empty((0, 2), dtype=np.float64)
    )
    assert matched == []


def test_gudhi_rejects_more_requested_finite_bars_than_generators() -> None:
    # Exactly one significant H1 bar exists for the square. Pretending
    # ripser returned two is a mismatched persistence pairing contract.
    bars = np.array(
        [[1.0, np.sqrt(2.0)], [1.0, np.sqrt(2.0)]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="fewer H1 generator pairs"):
        _gudhi_h1_generator_edges(_square_distances(), bars)


def test_gudhi_rejects_bar_values_inconsistent_with_reference() -> None:
    bars = np.array([[1.0, 1.8]], dtype=np.float64)
    with pytest.raises(ValueError, match="beyond float32 tolerance"):
        _gudhi_h1_generator_edges(_square_distances(), bars)


def test_gudhi_rejects_invented_bar_when_complex_has_no_h1() -> None:
    triangle = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.2, 0.8]],
        dtype=np.float64,
    )
    distances = np.linalg.norm(
        triangle[:, None, :] - triangle[None, :, :], axis=2
    )
    invented = np.array([[0.5, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="fewer H1 generator pairs"):
        _gudhi_h1_generator_edges(distances, invented)


def test_missing_generator_mapping_is_not_silently_fabricated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from topogeoml.nn import diff_ph

    # A direct critical-edge contract test: the external library reports
    # an ambiguous bar but does not provide an applicable generator.
    monkeypatch.setattr(
        diff_ph,
        "_gudhi_h1_generator_edges",
        lambda *_args: [None],
    )
    distances = _square_distances()
    bars = np.array([[1.0, np.sqrt(2.0)]], dtype=np.float64)
    cocycle = [np.array([[0, 1, 1]], dtype=np.int64)]
    with pytest.raises(ValueError, match="No matching H1 generator"):
        _critical_edges_h1(
            distances, cocycle, bars,
            gradients_requested=True, tie_policy="gudhi",
        )


def test_generic_route_rejects_wrong_edge_filtration_value() -> None:
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    false_bar = np.array([[0.3, 0.7]], dtype=np.float64)
    cocycle = [np.array([[0, 1, 1]], dtype=np.int64)]
    with pytest.raises(ValueError, match="does not reproduce"):
        _critical_edges_h1(
            distances, cocycle, false_bar,
            gradients_requested=True,
        )


def test_truncated_essential_h1_death_has_no_finite_death_edge() -> None:
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    essential = np.array([[1.0, np.inf]], dtype=np.float64)
    cocycle = [np.array([[0, 1, 1]], dtype=np.int64)]
    births, deaths = _critical_edges_h1(
        distances, cocycle, essential,
        gradients_requested=True,
    )
    assert births == [(0, 1)]
    assert deaths == [None]
