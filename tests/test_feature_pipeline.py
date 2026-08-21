"""
End-to-end tests for TopologyFeaturePipeline.

Covers:
  - sklearn API compliance (fit / transform / fit_transform / Pipeline composition)
  - Output shape and dtype contracts
  - Provenance capture
  - Topological discrimination on canonical shapes
  - Integration with a downstream sklearn classifier
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topogeoml.pipelines.feature_pipeline import (
    FitProvenance,
    TopologyFeaturePipeline,
)


def _make_shape_dataset(
    rng: np.random.Generator,
    n_per_class: int = 10,
    n_points: int = 40,
    noise: float = 0.05,
) -> tuple[list[NDArray[np.float64]], NDArray[np.int64]]:
    """
    Build a tiny synthetic dataset: noisy circles (label 1) vs noisy lines (label 0).
    Topology should perfectly separate these classes.
    """
    X: list[NDArray[np.float64]] = []
    y: list[int] = []
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False, dtype=np.float64)
    t = np.linspace(-1.0, 1.0, n_points, dtype=np.float64)

    for _ in range(n_per_class):
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        circle += noise * rng.standard_normal(circle.shape)
        X.append(np.ascontiguousarray(circle, dtype=np.float64))
        y.append(1)

    for _ in range(n_per_class):
        line = np.stack([t, np.zeros_like(t)], axis=1)
        line += noise * rng.standard_normal(line.shape)
        X.append(np.ascontiguousarray(line, dtype=np.float64))
        y.append(0)

    return X, np.array(y, dtype=np.int64)


# ---------- Basic API ----------


def test_pipeline_fit_returns_self(noisy_circle: NDArray[np.float64]) -> None:
    pipe = TopologyFeaturePipeline()
    result = pipe.fit([noisy_circle])
    assert result is pipe


def test_pipeline_output_shape_and_dtype(
    noisy_circle: NDArray[np.float64],
    noisy_line: NDArray[np.float64],
) -> None:
    pipe = TopologyFeaturePipeline(max_homology_dim=1, resolution=10)
    feats = pipe.fit_transform([noisy_circle, noisy_line])
    assert feats.dtype == np.float64
    # H_0 + H_1 channels, each 10x10 = 100 features → 200 total.
    assert feats.shape == (2, 200)


def test_pipeline_betti_curve_output_shape(noisy_circle: NDArray[np.float64]) -> None:
    pipe = TopologyFeaturePipeline(
        max_homology_dim=1,
        vectorizer="betti_curve",
        resolution=50,
    )
    feats = pipe.fit_transform([noisy_circle])
    # 2 dims × 50 samples = 100.
    assert feats.shape == (1, 100)


def test_pipeline_provenance_captured(noisy_circle: NDArray[np.float64]) -> None:
    pipe = TopologyFeaturePipeline(max_homology_dim=1, resolution=15, vectorizer="persistence_image")
    pipe.fit([noisy_circle, noisy_circle])
    prov = pipe.fit_provenance_
    assert isinstance(prov, FitProvenance)
    assert prov.n_samples_seen == 2
    assert prov.ambient_dim == 2
    assert prov.max_homology_dim == 1
    assert prov.vectorizer == "persistence_image"
    assert prov.output_dim == 2 * 15 * 15
    assert prov.pipeline_version == "0.0.7"
    assert "fallback_max" in prov.extras


def test_pipeline_rejects_empty_batch() -> None:
    pipe = TopologyFeaturePipeline()
    with pytest.raises(ValueError, match="empty batch"):
        pipe.fit([])


def test_pipeline_rejects_bad_vectorizer() -> None:
    pipe = TopologyFeaturePipeline(vectorizer="garbage")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vectorizer must be"):
        pipe.fit([np.zeros((10, 2), dtype=np.float64)])


def test_pipeline_accepts_3d_array_input(
    noisy_circle: NDArray[np.float64],
    noisy_line: NDArray[np.float64],
) -> None:
    """3D ndarray of shape (n_samples, n_points, dim) should work."""
    batch = np.stack([noisy_circle, noisy_line], axis=0)  # (2, 50, 2)
    pipe = TopologyFeaturePipeline(max_homology_dim=1, resolution=10)
    feats = pipe.fit_transform(batch)
    assert feats.shape == (2, 200)


def test_pipeline_transform_before_fit_raises(noisy_circle: NDArray[np.float64]) -> None:
    pipe = TopologyFeaturePipeline()
    with pytest.raises(Exception):  # sklearn NotFittedError
        pipe.transform([noisy_circle])


# ---------- Discrimination ----------


def test_pipeline_separates_circles_from_lines(rng: np.random.Generator) -> None:
    """
    End-to-end smoke: topology features should let a simple classifier
    perfectly separate circles from lines on a small synthetic dataset.
    """
    X, y = _make_shape_dataset(rng, n_per_class=8, n_points=40, noise=0.05)

    full = Pipeline(
        [
            ("topology", TopologyFeaturePipeline(max_homology_dim=1, resolution=10, vectorizer="betti_curve")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=500)),
        ]
    )
    full.fit(X, y)
    pred = full.predict(X)
    # Training accuracy should be 100% on this trivial separable task.
    accuracy = float(np.mean(pred == y))
    assert accuracy == 1.0, f"expected perfect train accuracy, got {accuracy:.3f}"


# ---------- sklearn Pipeline composition ----------


def test_pipeline_composes_in_sklearn_pipeline(noisy_circle: NDArray[np.float64]) -> None:
    """TopologyFeaturePipeline should compose with StandardScaler."""
    composite = Pipeline(
        [
            ("topology", TopologyFeaturePipeline(max_homology_dim=1, resolution=8)),
            ("scale", StandardScaler()),
        ]
    )
    features = composite.fit_transform([noisy_circle, noisy_circle])
    assert features.shape == (2, 128)
    # After StandardScaler with single-class data, columns with zero variance
    # are passed through; verify no NaN/inf (elite-code-standards §1).
    assert np.all(np.isfinite(features))


# ---------- Reproducibility ----------


def test_pipeline_deterministic_given_fixed_inputs(
    noisy_circle: NDArray[np.float64],
) -> None:
    """Same input → identical output, byte for byte (no internal RNG)."""
    pipe1 = TopologyFeaturePipeline(max_homology_dim=1, resolution=10)
    pipe2 = TopologyFeaturePipeline(max_homology_dim=1, resolution=10)
    f1 = pipe1.fit_transform([noisy_circle])
    f2 = pipe2.fit_transform([noisy_circle])
    np.testing.assert_array_equal(f1, f2)
