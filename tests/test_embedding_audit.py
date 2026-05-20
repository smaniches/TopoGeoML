"""Tests for embedding topology audit (item 9)."""

from __future__ import annotations

import numpy as np
import pytest

from topogeoml.audits import EmbeddingTopologyAudit, audit_embedding


def _sample_circle(n: int, radius: float, center: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    pts += 0.02 * rng.standard_normal(pts.shape)
    return np.ascontiguousarray(pts, dtype=np.float64)


def test_audit_returns_structured_report() -> None:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((100, 8))
    audit = audit_embedding(emb)
    assert isinstance(audit, EmbeddingTopologyAudit)
    assert audit.n_points_audited == 100
    assert audit.ambient_dim == 8


def test_audit_detects_single_circle_topology() -> None:
    """A circular embedding should yield β_1 ≈ 1."""
    rng = np.random.default_rng(42)
    emb = _sample_circle(n=80, radius=1.0, center=(0.0, 0.0), rng=rng)
    audit = audit_embedding(emb, persistence_threshold=0.5)
    # β_0=1 (single component), β_1=1 (one loop).
    assert audit.beta_1_estimate == 1, (
        f"expected 1 loop, got {audit.beta_1_estimate}; "
        f"longest_h1={audit.longest_h1_lifetime:.4f}"
    )
    assert audit.longest_h1_lifetime > 0.5


def test_audit_detects_two_circle_topology() -> None:
    """Two well-separated circles: β_1 ≈ 2."""
    rng = np.random.default_rng(42)
    c1 = _sample_circle(n=40, radius=1.0, center=(0.0, 0.0), rng=rng)
    c2 = _sample_circle(n=40, radius=1.0, center=(5.0, 0.0), rng=rng)
    emb = np.concatenate([c1, c2], axis=0)
    audit = audit_embedding(emb, persistence_threshold=0.5)
    assert audit.beta_1_estimate == 2
    assert audit.beta_0_estimate == 2


def test_audit_subsamples_large_inputs() -> None:
    """Audit should respect max_points cap."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5000, 4))
    audit = audit_embedding(emb, max_points=200)
    assert audit.n_points_audited == 200
    assert audit.provenance["subsampled"] is True
    assert audit.provenance["n_points_original"] == 5000


def test_audit_provenance_fields() -> None:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 4))
    audit = audit_embedding(emb, max_homology_dim=1, seed=7)
    assert audit.provenance["max_homology_dim"] == 1
    assert audit.provenance["seed"] == 7
    assert audit.provenance["filtration_backend"] == "ripser"


def test_audit_summary_is_a_string() -> None:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 4))
    audit = audit_embedding(emb)
    s = audit.summary()
    assert isinstance(s, str)
    assert "β_0" in s and "β_1" in s


def test_audit_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        audit_embedding(np.zeros(10))


def test_audit_rejects_single_point() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        audit_embedding(np.zeros((1, 4)))


def test_audit_default_threshold_uses_nn_scale() -> None:
    """Default persistence_threshold is set to 2 * median NN distance."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 4))
    audit = audit_embedding(emb)
    # Threshold equals 2 * median_nn_distance.
    assert audit.persistence_threshold == pytest.approx(2.0 * audit.median_nn_distance)
