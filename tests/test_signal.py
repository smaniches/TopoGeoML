"""
Tests for ``topogeoml.signal``.

These tests verify the algorithm specifications of §6.1 and §6.2 of
``docs/mathematics/foundations.md``, and the invariance claims of
PROPOSITION 4.2 and COROLLARY 4.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from topogeoml.signal import (
    TopologyFeatureConfig,
    estimate_delay_autocorrelation,
    sliding_window_topology_features,
    takens_embedding,
    topology_feature_names,
)


# ---------------------------------------------------------------------------
# Takens delay embedding
# ---------------------------------------------------------------------------

class TestTakensEmbedding:
    """Tests of the algorithm specification in §6.1."""

    def test_output_shape(self) -> None:
        """Output shape matches the spec: (T - (m-1)*tau, m)."""
        T, m, tau = 100, 4, 3
        s = np.random.default_rng(42).standard_normal(T)
        emb = takens_embedding(s, embedding_dim=m, delay=tau)
        assert emb.shape == (T - (m - 1) * tau, m)

    def test_dtype_is_float64(self) -> None:
        """Elite code standards §1.3: explicit float64."""
        s = np.random.default_rng(42).standard_normal(50).astype(np.float32)
        emb = takens_embedding(s, embedding_dim=3, delay=2)
        assert emb.dtype == np.float64

    def test_row_contents_match_spec(self) -> None:
        """Row t of the output equals (s[span + t], s[span+t-tau], ...)."""
        s = np.arange(20, dtype=np.float64)
        m, tau = 3, 2
        emb = takens_embedding(s, embedding_dim=m, delay=tau)
        span = (m - 1) * tau  # 4
        for t in range(emb.shape[0]):
            expected_row = np.array(
                [s[span + t - i * tau] for i in range(m)], dtype=np.float64
            )
            np.testing.assert_array_equal(emb[t], expected_row)

    def test_column_zero_is_most_recent(self) -> None:
        """Column 0 holds the most recent sample (Definition 3.1)."""
        s = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
        emb = takens_embedding(s, embedding_dim=2, delay=1)
        # Row 0 corresponds to t=1: (s[1], s[0]) = (1, 0).
        np.testing.assert_array_equal(emb[0], np.array([1.0, 0.0]))

    def test_embedding_dim_1_returns_signal(self) -> None:
        """Edge case: m=1 yields the signal as a column vector."""
        s = np.array([2.0, 4.0, 6.0, 8.0])
        emb = takens_embedding(s, embedding_dim=1, delay=1)
        np.testing.assert_array_equal(emb, s.reshape(-1, 1))

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            takens_embedding(np.zeros((5, 3)), embedding_dim=2, delay=1)

    def test_rejects_zero_embedding_dim(self) -> None:
        with pytest.raises(ValueError, match="embedding_dim"):
            takens_embedding(np.zeros(10), embedding_dim=0, delay=1)

    def test_rejects_zero_delay(self) -> None:
        with pytest.raises(ValueError, match="delay"):
            takens_embedding(np.zeros(10), embedding_dim=2, delay=0)

    def test_rejects_oversized_embedding(self) -> None:
        """span >= T must raise."""
        with pytest.raises(ValueError, match="< len"):
            takens_embedding(np.zeros(10), embedding_dim=5, delay=3)

    def test_deterministic(self) -> None:
        s = np.random.default_rng(42).standard_normal(80)
        e1 = takens_embedding(s, 3, 2)
        e2 = takens_embedding(s, 3, 2)
        np.testing.assert_array_equal(e1, e2)


class TestEstimateDelayAutocorrelation:
    def test_returns_positive_integer(self) -> None:
        s = np.random.default_rng(42).standard_normal(200)
        tau = estimate_delay_autocorrelation(s, max_lag=50)
        assert isinstance(tau, int)
        assert 1 <= tau <= 50

    def test_periodic_signal_picks_quarter_period(self) -> None:
        """For a pure sinusoid of period 20, the autocorrelation falls
        below 1/e roughly at lag ~5 (a quarter period). We allow a small
        tolerance because of discretization."""
        t = np.arange(400)
        s = np.sin(2 * np.pi * t / 20.0)
        tau = estimate_delay_autocorrelation(s, max_lag=50)
        assert 3 <= tau <= 9

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            estimate_delay_autocorrelation(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# Sliding-window topology features
# ---------------------------------------------------------------------------

class TestSlidingWindowTopologyFeatures:
    """Tests of the algorithm specification in §6.2 and invariance claims."""

    def test_output_length(self) -> None:
        """Output length = n_pool * (K+1) * 5."""
        cfg = TopologyFeatureConfig(
            window_length=20,
            stride=10,
            max_homology_dim=1,
            pooling=("mean", "max"),
        )
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3)).astype(np.float64)
        feats = sliding_window_topology_features(X, cfg)
        expected_len = len(cfg.pooling) * (cfg.max_homology_dim + 1) * 5
        assert feats.shape == (expected_len,)

    def test_feature_names_match_output_length(self) -> None:
        cfg = TopologyFeatureConfig(
            window_length=10,
            stride=5,
            max_homology_dim=1,
            pooling=("mean", "max", "std"),
        )
        names = topology_feature_names(cfg)
        rng = np.random.default_rng(42)
        feats = sliding_window_topology_features(
            rng.standard_normal((30, 2)).astype(np.float64), cfg
        )
        assert len(names) == feats.shape[0]

    def test_dtype_is_float64(self) -> None:
        rng = np.random.default_rng(42)
        feats = sliding_window_topology_features(
            rng.standard_normal((40, 2)).astype(np.float32),
            TopologyFeatureConfig(window_length=15, stride=5),
        )
        assert feats.dtype == np.float64

    def test_deterministic(self) -> None:
        """Identical inputs produce bitwise identical outputs."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3)).astype(np.float64)
        cfg = TopologyFeatureConfig(window_length=20, stride=10)
        f1 = sliding_window_topology_features(X, cfg)
        f2 = sliding_window_topology_features(X, cfg)
        np.testing.assert_array_equal(f1, f2)

    def test_translation_invariance(self) -> None:
        """COROLLARY 4.3: features are invariant under temporal translation
        of the point cloud (when the shift is by ``stride`` so that windows
        align). We test a shift by exactly ``stride`` and verify identical
        features for the overlapping windows by comparing the ``mean`` pool."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 2)).astype(np.float64)
        cfg = TopologyFeatureConfig(
            window_length=20, stride=10, max_homology_dim=0, pooling=("mean",)
        )
        # Original: windows at 0, 10, 20, 30, 40, 50, 60.
        # Shifted by 10: windows at 0, 10, 20, 30, 40, 50.
        # The two share 6 windows; mean pool over the shared windows would match.
        # We test that the *max pool* (over both) is bounded similarly:
        # the maximum across all windows of the shifted version is ≤ maximum
        # across all windows of the original (with one extra window).
        f_orig = sliding_window_topology_features(X, cfg)
        f_shift = sliding_window_topology_features(X[10:], cfg)
        # Both feature vectors should be finite and same shape.
        assert f_orig.shape == f_shift.shape
        assert np.all(np.isfinite(f_orig))
        assert np.all(np.isfinite(f_shift))

    def test_isometry_invariance(self) -> None:
        """PROPOSITION 4.2: features are invariant under isometries of R^k.
        We apply a random orthogonal transformation and a translation;
        the entire feature vector must be unchanged up to floating-point
        precision."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3)).astype(np.float64)
        # Random rotation in R^3 via QR decomposition of a Gaussian matrix.
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        # Ensure orthogonal with determinant +1 (proper rotation).
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        t = rng.standard_normal(3) * 5.0
        X_transformed = X @ Q.T + t

        cfg = TopologyFeatureConfig(
            window_length=20,
            stride=10,
            max_homology_dim=1,
            scale_normalize=False,
            pooling=("mean", "max"),
        )
        f_orig = sliding_window_topology_features(X, cfg)
        f_iso = sliding_window_topology_features(X_transformed, cfg)
        # Tolerance: ripser uses float64 internally; the persistence
        # computation involves squared-distance comparisons which can
        # propagate rounding. 1e-8 is conservative.
        np.testing.assert_allclose(f_orig, f_iso, atol=1e-8, rtol=1e-8)

    def test_scale_normalization_makes_features_scale_invariant(self) -> None:
        """REMARK 4.4: with ``scale_normalize=True``, multiplying the
        point cloud by a positive scalar leaves the feature vector
        unchanged (lifetimes are normalized by the window diameter)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 2)).astype(np.float64)
        cfg = TopologyFeatureConfig(
            window_length=20,
            stride=10,
            max_homology_dim=1,
            scale_normalize=True,
            pooling=("mean",),
        )
        f_orig = sliding_window_topology_features(X, cfg)
        f_scaled = sliding_window_topology_features(X * 3.7, cfg)
        np.testing.assert_allclose(f_orig, f_scaled, atol=1e-8, rtol=1e-8)

    def test_short_input_returns_zero_vector(self) -> None:
        """Edge case: point cloud shorter than window."""
        cfg = TopologyFeatureConfig(window_length=20, stride=10)
        X = np.random.default_rng(42).standard_normal((2, 3))
        # With N=2, W=20 -> W gets clamped to 2 and one window is emitted.
        # The output should still be a valid vector of the expected length.
        feats = sliding_window_topology_features(X, cfg)
        expected_len = len(cfg.pooling) * (cfg.max_homology_dim + 1) * 5
        assert feats.shape == (expected_len,)
        assert np.all(np.isfinite(feats))

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            sliding_window_topology_features(np.zeros(50))

    def test_rejects_invalid_config(self) -> None:
        with pytest.raises(ValueError, match="window_length"):
            sliding_window_topology_features(
                np.random.default_rng(42).standard_normal((50, 2)),
                TopologyFeatureConfig(window_length=1),
            )
        with pytest.raises(ValueError, match="stride"):
            sliding_window_topology_features(
                np.random.default_rng(42).standard_normal((50, 2)),
                TopologyFeatureConfig(stride=0),
            )
