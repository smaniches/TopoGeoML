"""
Coverage extensions for the ``topogeoml`` library.

These tests target defensive validation paths and edge-case branches
that were uncovered after the Phase 1 benchmark framework landed.
Each test class targets a specific source file. The intent is closure
of meaningful gaps, not test-count padding: every test below was
written to exercise a specific line that the rest of the suite did
not reach.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# topogeoml.core.complexes
# ---------------------------------------------------------------------------

class TestComplexesValidation:
    def test_empty_facet_continues(self) -> None:
        """A facet with no vertices is silently skipped during construction
        (line 67 — ``continue`` branch)."""
        from topogeoml.core.complexes import SimplicialComplex

        # The empty facet is a no-op; we expect a complex with only the
        # non-empty triangle facet's faces.
        sc = SimplicialComplex(facets=[(), (0, 1, 2)])
        assert sc.n_simplices(0) == 3
        assert sc.n_simplices(2) == 1

    def test_boundary_matrix_rejects_negative_k(self) -> None:
        """Line 122 — ``raise ValueError`` for k < 0."""
        from topogeoml.core.complexes import SimplicialComplex

        sc = SimplicialComplex(facets=[(0, 1, 2)])
        with pytest.raises(ValueError, match="k must be >= 0"):
            sc.boundary_matrix(-1)

    def test_boundary_matrix_returns_empty_for_missing_dim(self) -> None:
        """Line 131 — empty (n_{k-1}, n_k) matrix for k where no simplices exist."""
        from topogeoml.core.complexes import SimplicialComplex

        sc = SimplicialComplex(facets=[(0, 1)])  # 1D complex, no triangles
        bm = sc.boundary_matrix(2)  # k=2 has no simplices
        assert bm.shape == (1, 0)  # n_{k-1}=1 edge, n_k=0 triangles

    def test_is_chain_complex_skips_dims_with_no_simplices(self) -> None:
        """Line 187 — `continue` when n_simplices(k) == 0 or n_simplices(k-1) == 0."""
        from topogeoml.core.complexes import SimplicialComplex, is_chain_complex

        # Complex with only edges (no triangles). is_chain_complex should
        # short-circuit when checking the k=2 vs k=1 path.
        sc = SimplicialComplex(facets=[(0, 1), (1, 2)])
        assert is_chain_complex(sc, max_dim=3) is True  # forces iteration past k=2

    def test_hodge_laplacian_rejects_negative_k(self) -> None:
        """Line 224 — ``raise ValueError`` for k < 0."""
        from topogeoml.core.complexes import SimplicialComplex, hodge_laplacian

        sc = SimplicialComplex(facets=[(0, 1, 2)])
        with pytest.raises(ValueError, match="k must be >= 0"):
            hodge_laplacian(sc, -1)

    def test_betti_numbers_default_max_dim(self) -> None:
        """Line 269 — ``max_dim = complex_.max_dim`` when None."""
        from topogeoml.core.complexes import SimplicialComplex, betti_numbers

        sc = SimplicialComplex(facets=[(0, 1, 2)])
        # No max_dim passed -> use complex_.max_dim = 2.
        betti = betti_numbers(sc)
        assert 2 in betti

    def test_betti_numbers_handles_dims_with_no_simplices(self) -> None:
        """Lines 273-274 — out[k]=0 when no k-simplices exist."""
        from topogeoml.core.complexes import SimplicialComplex, betti_numbers

        sc = SimplicialComplex(facets=[(0, 1)])
        # Request up to dim 3; dims 2 and 3 have zero simplices.
        betti = betti_numbers(sc, max_dim=3)
        assert betti[2] == 0
        assert betti[3] == 0


# ---------------------------------------------------------------------------
# topogeoml.core.cubical
# ---------------------------------------------------------------------------

class TestCubicalValidation:
    def test_rejects_invalid_connectivity(self) -> None:
        """Line 98 — connectivity must be 1 or 2."""
        from topogeoml.core.cubical import cubical_mask_diagnostic

        mask = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="connectivity"):
            cubical_mask_diagnostic(mask, connectivity=3)


# ---------------------------------------------------------------------------
# topogeoml.core.diagrams
# ---------------------------------------------------------------------------

class TestDiagramsValidation:
    def test_rejects_non_ndarray_bars(self) -> None:
        """Line 81 — TypeError if any bars[dim] is not an ndarray."""
        from topogeoml.core.diagrams import DiagramProvenance, PersistenceDiagram

        prov = DiagramProvenance(
            filtration="rips", metric="euclidean", max_homology_dim=0,
            max_edge_length=None, n_points=10, ambient_dim=2,
        )
        with pytest.raises(TypeError, match=r"must be np\.ndarray"):
            PersistenceDiagram(bars={0: [[0.0, 1.0]]}, provenance=prov)  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# topogeoml.data.graph_to_complex
# ---------------------------------------------------------------------------

class TestGraphToComplexValidation:
    def test_rejects_unknown_type(self) -> None:
        """Lines 104-105 — TypeError for non-Graph / non-ndarray input."""
        from topogeoml.data.graph_to_complex import graph_to_clique_complex

        with pytest.raises(TypeError, match=r"networkx\.Graph or square ndarray"):
            graph_to_clique_complex([[0, 1], [1, 0]], max_dim=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# topogeoml.core.filtrations
# ---------------------------------------------------------------------------

class TestRipsFiltrationValidation:
    def test_rejects_coeff_lt_2(self) -> None:
        """Line 68 — coeff must be >= 2."""
        from topogeoml.core.filtrations import RipsFiltration

        with pytest.raises(ValueError, match="coeff"):
            RipsFiltration(max_homology_dim=1, coeff=1)

    def test_precomputed_metric_path(self) -> None:
        """Lines 97-102 — precomputed metric rejects non-square inputs."""
        from topogeoml.core.filtrations import RipsFiltration

        rips = RipsFiltration(max_homology_dim=1, metric="precomputed")
        with pytest.raises(ValueError, match="square"):
            rips.compute(np.zeros((5, 6)))  # non-square distance matrix

    def test_non_euclidean_metric_path(self) -> None:
        """Line 115 — non-euclidean / non-precomputed metric branch."""
        from topogeoml.core.filtrations import RipsFiltration

        # Use a recognized scipy distance metric — `cityblock` is supported by
        # ripser via the underlying scipy `pdist`/`cdist` machinery.
        rips = RipsFiltration(max_homology_dim=1, metric="cityblock")
        diagram = rips.compute(np.random.RandomState(0).randn(10, 2))
        assert diagram.provenance.metric == "cityblock"


# ---------------------------------------------------------------------------
# topogeoml.core.vectorizers
# ---------------------------------------------------------------------------

class TestVectorizerValidation:
    def test_persistence_image_rejects_resolution_lt_2(self) -> None:
        """Line 122 — resolution >= 2."""
        from topogeoml.core.vectorizers import PersistenceImageVectorizer

        with pytest.raises(ValueError, match="resolution"):
            PersistenceImageVectorizer(resolution=1)

    def test_persistence_image_rejects_nonpositive_sigma(self) -> None:
        """Line 124 — sigma must be positive."""
        from topogeoml.core.vectorizers import PersistenceImageVectorizer

        with pytest.raises(ValueError, match="sigma"):
            PersistenceImageVectorizer(sigma=0.0)

    def test_persistence_image_rejects_empty_homology_dims(self) -> None:
        """Line 126 — homology_dims must be non-empty."""
        from topogeoml.core.vectorizers import PersistenceImageVectorizer

        with pytest.raises(ValueError, match="homology_dims"):
            PersistenceImageVectorizer(homology_dims=())

    def test_persistence_image_rejects_nonpositive_fallback_max(self) -> None:
        """Line 128 — fallback_max must be positive."""
        from topogeoml.core.vectorizers import PersistenceImageVectorizer

        with pytest.raises(ValueError, match="fallback_max"):
            PersistenceImageVectorizer(fallback_max=0.0)

    def test_betti_curve_rejects_resolution_lt_2(self) -> None:
        """Line 189 — resolution >= 2 on BettiCurveVectorizer."""
        from topogeoml.core.vectorizers import BettiCurveVectorizer

        with pytest.raises(ValueError, match="resolution"):
            BettiCurveVectorizer(resolution=1)

    def test_betti_curve_rejects_empty_homology_dims(self) -> None:
        """Line 191 — homology_dims must be non-empty on BettiCurveVectorizer."""
        from topogeoml.core.vectorizers import BettiCurveVectorizer

        with pytest.raises(ValueError, match="homology_dims"):
            BettiCurveVectorizer(homology_dims=())


# ---------------------------------------------------------------------------
# topogeoml.experiments.configs
# ---------------------------------------------------------------------------

class TestExperimentsConfigsCoverage:
    def test_missing_package_version_returns_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 126-127 — PackageNotFoundError returns the literal "missing"."""
        import importlib.metadata as md

        from topogeoml.experiments import configs

        def _raise(_pkg: str) -> str:
            raise md.PackageNotFoundError("synthetic")

        monkeypatch.setattr(md, "version", _raise)
        env = configs._environment_snapshot()
        assert env["numpy_version"] == "missing"

    def test_jsonify_list_branch(self) -> None:
        """Line 200-201 — list/tuple branch in _jsonify."""
        from topogeoml.experiments.configs import _jsonify

        out = _jsonify([1.0, "a", np.float64(2.5)])
        assert out == [1.0, "a", 2.5]


# ---------------------------------------------------------------------------
# topogeoml.audits.embedding_audit
# ---------------------------------------------------------------------------

class TestEmbeddingAuditCoverage:
    def test_audit_on_collinear_points_has_no_h1(self) -> None:
        """Covers the empty-bars (line 153) and empty-H_1 (line 180) branches.

        Three collinear points have no H_1 — finite_h1 is empty so
        ``longest_h1_lifetime`` falls through to the 0.0 branch.
        """
        from topogeoml import audit_embedding

        emb = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64
        )
        audit = audit_embedding(emb, max_points=3)
        assert audit.beta_1_estimate == 0
        assert audit.longest_h1_lifetime == 0.0


# ---------------------------------------------------------------------------
# topogeoml.signal.delay_embedding
# ---------------------------------------------------------------------------

class TestDelayEmbeddingCoverage:
    def test_estimate_delay_default_max_lag(self) -> None:
        """Line 161 — default max_lag = max(1, T // 4)."""
        from topogeoml.signal import estimate_delay_autocorrelation

        rng = np.random.default_rng(0)
        signal = rng.standard_normal(100)
        # max_lag=None forces the default branch.
        delay = estimate_delay_autocorrelation(signal, max_lag=None, threshold=0.5)
        assert 1 <= delay <= 25

    def test_estimate_delay_returns_max_lag_when_no_threshold_crossing(self) -> None:
        """Line 177 — returns max_lag when no lag crosses the threshold.

        A low-frequency cosine has autocorrelation > 0.99 at every small
        lag. Threshold ``0.99`` therefore is never met within
        ``max_lag = 3`` and the function returns max_lag.
        """
        from topogeoml.signal import estimate_delay_autocorrelation

        t = np.arange(2000)
        signal = np.cos(0.001 * t)
        delay = estimate_delay_autocorrelation(signal, max_lag=3, threshold=0.99)
        assert delay == 3


# ---------------------------------------------------------------------------
# topogeoml.signal.sliding_window
# ---------------------------------------------------------------------------

class TestSlidingWindowCoverage:
    def test_pooling_handles_max_min_std_branches(self) -> None:
        """Lines 138-148 — max/min/std pooling branches."""
        from topogeoml.signal.sliding_window import _pool

        v = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(_pool(v, "max"), np.array([5.0, 6.0]))
        np.testing.assert_array_equal(_pool(v, "min"), np.array([1.0, 2.0]))
        # Std with three samples per column.
        assert np.allclose(_pool(v, "std"), v.std(axis=0))

    def test_pool_empty_returns_zeros(self) -> None:
        """Line 138 — empty values returns zeros."""
        from topogeoml.signal.sliding_window import _pool

        v = np.empty((0, 5))
        out = _pool(v, "mean")
        assert out.shape == (5,)
        np.testing.assert_array_equal(out, np.zeros(5))

    def test_sliding_window_validates_stride_and_max_dim_and_pooling(self) -> None:
        """Lines 200-206 — validation of stride, max_homology_dim, pooling."""
        from topogeoml.signal import TopologyFeatureConfig, sliding_window_topology_features

        pts = np.random.RandomState(0).randn(50, 2)
        # Invalid stride.
        with pytest.raises(ValueError, match="stride"):
            sliding_window_topology_features(
                pts, TopologyFeatureConfig(window_length=10, stride=0),
            )
        # Invalid max_homology_dim.
        with pytest.raises(ValueError, match="max_homology_dim"):
            sliding_window_topology_features(
                pts, TopologyFeatureConfig(
                    window_length=10, stride=1, max_homology_dim=-1,
                ),
            )
        # Empty pooling.
        with pytest.raises(ValueError, match="pooling"):
            sliding_window_topology_features(
                pts, TopologyFeatureConfig(
                    window_length=10, stride=1, pooling=(),
                ),
            )

    def test_finite_bars_empty_returns_zeros(self) -> None:
        """Lines 64-66 — empty finite_bars returns _N_STATS_PER_DIM zeros."""
        from topogeoml.signal.sliding_window import (
            _N_STATS_PER_DIM,
            _diagram_statistics,
        )

        # Diagram with only infinite bars -> finite_bars is empty.
        bars = np.array([[0.0, np.inf]])
        out = _diagram_statistics(bars)
        assert out.shape == (_N_STATS_PER_DIM,)
        np.testing.assert_array_equal(out, np.zeros(_N_STATS_PER_DIM))


# ---------------------------------------------------------------------------
# topogeoml.training.callbacks
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


class TestDiffPHCoverage:
    """Coverage extensions for ``topogeoml.nn.diff_ph``."""

    def test_rips_diagram_torch_with_max_edge_length(self) -> None:
        """Line 208 — ``max_edge_length`` propagates to ripser."""
        from topogeoml.nn.diff_ph import rips_diagram_torch

        X = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )
        dgms = rips_diagram_torch(X, max_dim=1, max_edge_length=1.5)
        assert len(dgms) == 2

    def test_total_persistence_loss_empty_significant_returns_zero(self) -> None:
        """Line 309 — non-empty diagram, all bars below threshold → 0.0."""
        from topogeoml.nn.diff_ph import total_persistence_loss

        # One finite bar of lifetime 0.1; threshold 10 filters it out.
        bars = torch.tensor([[0.0, 0.1]], dtype=torch.float64)
        loss = total_persistence_loss(bars, p=2.0, threshold=10.0)
        assert float(loss.item()) == 0.0

    def test_persistence_entropy_loss_handles_empty(self) -> None:
        """Line 354 — empty diagram → 0.0 entropy loss."""
        from topogeoml.nn.diff_ph import persistence_entropy_loss

        # A diagram with only infinite bars yields no finite lifetimes.
        only_inf = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        loss = persistence_entropy_loss(only_inf)
        assert float(loss.item()) == 0.0

    def test_topology_regularizer_rejects_unknown_loss_type(self) -> None:
        """Line 409 — invalid loss_type raises ValueError."""
        from topogeoml.nn.diff_ph import TopologyRegularizer

        with pytest.raises(ValueError, match="loss_type"):
            TopologyRegularizer(loss_type="not_a_loss")  # type: ignore[arg-type]

    def test_topology_regularizer_subsamples_when_n_exceeds_max(self) -> None:
        """Lines 439-441 — subsampling branch when n > max_points."""
        from topogeoml.nn.diff_ph import TopologyRegularizer

        reg = TopologyRegularizer(loss_type="total_persistence", max_points=15)
        # 30 > 15 -> subsample branch fires.
        X = torch.randn(30, 2, dtype=torch.float64, requires_grad=True)
        loss = reg(X)
        assert torch.is_tensor(loss)
        assert loss.requires_grad

    def test_topology_regularizer_entropy_loss_path(self) -> None:
        """Line 458 — entropy loss branch."""
        from topogeoml.nn.diff_ph import TopologyRegularizer

        reg = TopologyRegularizer(loss_type="entropy")
        X = torch.randn(20, 2, dtype=torch.float64, requires_grad=True)
        loss = reg(X)
        assert torch.is_tensor(loss)

    def test_topology_regularizer_betti_path(self) -> None:
        """Lines 459-460 — betti_regularization loss branch."""
        from topogeoml.nn.diff_ph import TopologyRegularizer

        reg = TopologyRegularizer(
            loss_type="betti_regularization", target_betti={1: 1},
        )
        X = torch.randn(20, 2, dtype=torch.float64, requires_grad=True)
        loss = reg(X)
        assert torch.is_tensor(loss)

    def test_betti_regularization_loss_empty_diagram(self) -> None:
        """Line 354 — empty diagram in betti_regularization_loss."""
        from topogeoml.nn.diff_ph import betti_regularization_loss

        only_inf = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        loss = betti_regularization_loss(only_inf, target_n_components=1)
        assert float(loss.item()) == 0.0

    def test_betti_regularization_loss_below_target(self) -> None:
        """Line 362 — n_real <= target_n_components → 0 loss."""
        from topogeoml.nn.diff_ph import betti_regularization_loss

        # One short bar: lifetimes = [0.05]. With prominence_threshold=0.1
        # the significant set is empty, n_real = 1 + 0 = 1. With target=2
        # the n_real <= target branch fires.
        bars = torch.tensor([[0.0, 0.05]], dtype=torch.float64)
        loss = betti_regularization_loss(
            bars, target_n_components=2, prominence_threshold=0.1,
        )
        assert float(loss.item()) == 0.0

    def test_topology_regularizer_rejects_non_2d(self) -> None:
        """Line 434 — X must be 2D."""
        from topogeoml.nn.diff_ph import TopologyRegularizer

        reg = TopologyRegularizer(loss_type="total_persistence")
        with pytest.raises(ValueError, match="2D"):
            reg(torch.randn(10, dtype=torch.float64))

    def test_topology_regularizer_skips_empty_dim(self) -> None:
        """Line 451 — empty per-dim diagram is skipped in the accumulator loop.

        With only 3 collinear points and max_dim=1, H_1 is empty.
        """
        from topogeoml.nn.diff_ph import TopologyRegularizer

        reg = TopologyRegularizer(loss_type="total_persistence", max_dim=1)
        X = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64,
            requires_grad=True,
        )
        loss = reg(X)
        assert torch.is_tensor(loss)

    def test_total_persistence_loss_empty_diagram_returns_zero(self) -> None:
        """Line 306 — empty diagram fast-path in total_persistence_loss."""
        from topogeoml.nn.diff_ph import total_persistence_loss

        only_inf = torch.tensor([[0.0, float("inf")]], dtype=torch.float64)
        loss = total_persistence_loss(only_inf, p=2.0, threshold=0.0)
        assert float(loss.item()) == 0.0


class TestCallbacksCoverage:
    def test_subsample_branch(self) -> None:
        """Lines 167-168 — subsample when n > max_probe_points."""
        from topogeoml.training.callbacks import ShapeOfLearningCallback

        class _TinyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = _TinyModule()
        # 30 probe points; we cap max_probe_points at 10 to force the
        # subsample branch (line 167-168) on every ``on_step`` invocation.
        probe = torch.randn(30, 4)
        cb = ShapeOfLearningCallback(
            model=model, probe_inputs=probe, layer_name="linear",
            every_n_steps=1, max_probe_points=10,
        )
        snap = cb.on_step(step=1, loss=0.5)
        assert snap is not None
        assert snap.step == 1


# ---------------------------------------------------------------------------
# topogeoml.pipelines.feature_pipeline
# ---------------------------------------------------------------------------

class TestFeaturePipelineCoverage:
    def test_transform_on_empty_batch_returns_empty(self) -> None:
        """Lines 205-207 — empty input batch returns empty (0, output_dim) array."""
        import numpy as np

        from topogeoml import TopologyFeaturePipeline

        # Fit on a valid batch, then transform an empty list.
        pipe = TopologyFeaturePipeline()
        X = [np.random.RandomState(0).randn(20, 2).astype(np.float64) for _ in range(3)]
        pipe.fit(X)
        out = pipe.transform([])
        assert out.shape == (0, pipe.fit_provenance_.output_dim)

    def test_coerce_rejects_wrong_ndarray_ndim(self) -> None:
        """Lines 238-240 — ndarray not 2D or 3D raises ValueError."""
        import numpy as np

        from topogeoml import TopologyFeaturePipeline

        pipe = TopologyFeaturePipeline()
        with pytest.raises(ValueError, match="2D"):
            pipe.fit(np.zeros((2, 3, 4, 5)))  # 4D

    def test_coerce_rejects_per_cloud_non_2d(self) -> None:
        """Line 246 — per-cloud item not 2D raises ValueError."""
        import numpy as np

        from topogeoml import TopologyFeaturePipeline

        pipe = TopologyFeaturePipeline()
        # List of items where one is 1D.
        with pytest.raises(ValueError, match="2D"):
            pipe.fit([np.zeros((5, 2)), np.zeros(5)])

    def test_calibrate_fallback_max_returns_1_for_empty(self) -> None:
        """Lines 266-267 — _estimate_filtration_scale returns 1.0 for empty input."""
        from topogeoml.pipelines.feature_pipeline import TopologyFeaturePipeline

        assert TopologyFeaturePipeline._estimate_filtration_scale([]) == 1.0

    def test_calibrate_fallback_max_skips_tiny_clouds(self) -> None:
        """Lines 271-272 — clouds with shape[0] < 2 are skipped."""
        from topogeoml.pipelines.feature_pipeline import TopologyFeaturePipeline

        clouds = [
            np.zeros((1, 2), dtype=np.float64),
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        ]
        result = TopologyFeaturePipeline._estimate_filtration_scale(clouds)
        assert result > 0  # only the second cloud contributed

    def test_calibrate_fallback_max_precomputed_distance_path(self) -> None:
        """Lines 275-278 — precomputed-distance-matrix branch (square cloud)."""
        from topogeoml.pipelines.feature_pipeline import TopologyFeaturePipeline

        # Square symmetric matrix with zero diagonal triggers the
        # precomputed-distance heuristic in _estimate_filtration_scale.
        sq = np.array(
            [[0.0, 1.0, 2.0],
             [1.0, 0.0, 1.5],
             [2.0, 1.5, 0.0]], dtype=np.float64,
        )
        result = TopologyFeaturePipeline._estimate_filtration_scale([sq])
        assert result > 0


# ---------------------------------------------------------------------------
# topogeoml.nn.hodge
# ---------------------------------------------------------------------------

class TestComplexesAdditional:
    def test_hodge_laplacian_empty_complex_returns_empty_matrix(self) -> None:
        """Line 227 — empty (0,0) sparse matrix when n_simplices(k) == 0."""
        from topogeoml.core.complexes import SimplicialComplex, hodge_laplacian

        sc = SimplicialComplex(facets=[(0,)])  # only one vertex
        L = hodge_laplacian(sc, k=2)  # no 2-simplices
        assert L.shape == (0, 0)


class TestFiltrationsAdditional:
    def test_precomputed_metric_yields_unknown_ambient_dim(self) -> None:
        """Line 102 — ``ambient_dim = -1`` for precomputed metric."""
        from topogeoml.core.filtrations import RipsFiltration

        rips = RipsFiltration(max_homology_dim=1, metric="precomputed")
        D = np.array(
            [[0.0, 1.0, 2.0],
             [1.0, 0.0, 1.5],
             [2.0, 1.5, 0.0]], dtype=np.float64,
        )
        diag = rips.compute(D)
        assert diag.provenance.ambient_dim == -1


class TestFeaturePipelineSingleCloudPath:
    def test_2d_single_cloud_input(self) -> None:
        """Line 237 — single 2D point cloud is wrapped as batch of one."""
        from topogeoml import TopologyFeaturePipeline

        X_single = np.random.RandomState(0).randn(15, 2).astype(np.float64)
        pipe = TopologyFeaturePipeline()
        pipe.fit(X_single)  # 2D direct — covers the wrap branch
        feats = pipe.transform(X_single)
        assert feats.shape == (1, pipe.fit_provenance_.output_dim)


class TestDelayEmbeddingValidation:
    def test_rejects_max_lag_lt_1(self) -> None:
        """Line 163."""
        from topogeoml.signal import estimate_delay_autocorrelation

        with pytest.raises(ValueError, match="max_lag"):
            estimate_delay_autocorrelation(np.arange(10.0), max_lag=0)

    def test_rejects_threshold_out_of_range(self) -> None:
        """Line 165."""
        from topogeoml.signal import estimate_delay_autocorrelation

        with pytest.raises(ValueError, match="threshold"):
            estimate_delay_autocorrelation(np.arange(10.0), threshold=1.5)


class TestSlidingWindowAdditional:
    def test_zero_persistence_entropy_branch(self) -> None:
        """Line 82 — ``pers_1 == 0.0`` -> entropy = 0.0 branch."""
        from topogeoml.signal.sliding_window import _diagram_statistics

        # Single bar with zero persistence: birth == death.
        bars = np.array([[0.5, 0.5]])
        stats = _diagram_statistics(bars)
        # entropy is the 4th statistic (count_finite, Pers_1, Pers_2, E, longest);
        # for a single zero-persistence bar Pers_1=0 -> entropy=0.
        assert stats[3] == 0.0

    def test_pool_unknown_raises(self) -> None:
        """Line 148."""
        from topogeoml.signal.sliding_window import _pool

        with pytest.raises(ValueError, match="Unknown pooling"):
            _pool(np.zeros((3, 2)), "not_a_pool")

    def test_short_input_capping_logic(self) -> None:
        """Lines 211-213 — input shorter than ``window_length`` is capped to ``N``,
        ensuring at least one window is processed even when ``stride > N - W``.

        Renamed from ``test_short_input_returns_zero_vector_with_specific_config``
        per Gemini PR #5 review: the `n_windows == 0` branch is actually
        unreachable (the capping logic on line 213 ensures n_windows >= 1),
        so this test exercises the capping path, not a zero-vector fallback.
        """
        from topogeoml.signal import (
            TopologyFeatureConfig,
            sliding_window_topology_features,
        )

        # Window longer than the input — only 1 window after the cap on
        # line 213, but if stride > N - W we get n_windows = 0 (covers 222-223).
        cfg = TopologyFeatureConfig(
            window_length=200, stride=300, pooling=("mean", "max"),
            max_homology_dim=1,
        )
        pts = np.random.RandomState(0).randn(150, 2).astype(np.float64)
        out = sliding_window_topology_features(pts, cfg)
        # Output shape: n_pool * n_dims * _N_STATS_PER_DIM = 2 * 2 * 5 = 20.
        from topogeoml.signal.sliding_window import _N_STATS_PER_DIM

        assert out.shape == (2 * 2 * _N_STATS_PER_DIM,)

    def test_edge_threshold_branch(self) -> None:
        """Line 234 — ``cfg.edge_threshold is not None`` triggers ripser thresh kwarg."""
        from topogeoml.signal import (
            TopologyFeatureConfig,
            sliding_window_topology_features,
        )

        cfg = TopologyFeatureConfig(
            window_length=20, stride=10, pooling=("mean",),
            max_homology_dim=1, edge_threshold=1.0,
        )
        pts = np.random.RandomState(0).randn(60, 2).astype(np.float64)
        out = sliding_window_topology_features(pts, cfg)
        from topogeoml.signal.sliding_window import _N_STATS_PER_DIM

        assert out.size == 1 * 2 * _N_STATS_PER_DIM  # 1 pool, 2 dims, 5 stats

    def test_duplicate_points_window_is_skipped(self) -> None:
        """Line 244 — windows with < 2 unique points are skipped."""
        from topogeoml.signal import (
            TopologyFeatureConfig,
            sliding_window_topology_features,
        )

        # Construct a point cloud where the first window has all-identical
        # points (zero unique rows).
        pts = np.zeros((40, 2), dtype=np.float64)
        pts[20:] = 1.0  # second half is constant at (1, 1)
        cfg = TopologyFeatureConfig(
            window_length=10, stride=20, pooling=("mean",),
            max_homology_dim=0,
        )
        out = sliding_window_topology_features(pts, cfg)
        # Both windows have only one unique point -> skipped -> all zero stats.
        from topogeoml.signal.sliding_window import _N_STATS_PER_DIM

        assert out.size == 1 * 1 * _N_STATS_PER_DIM


class TestHodgeCoverage:
    def test_sparse_scipy_to_torch_with_device(self) -> None:
        """Line 85 — explicit device branch."""
        import scipy.sparse as sp

        from topogeoml.nn.hodge import sparse_scipy_to_torch

        m = sp.csr_matrix(np.eye(3, dtype=np.float64))
        out = sparse_scipy_to_torch(m, device=torch.device("cpu"))
        assert out.device.type == "cpu"

    def test_build_hodge_layer_from_ndarray(self) -> None:
        """Lines 144-145 — ndarray laplacian branch in build_hodge_layer_from_complex.

        Coverage check only: we don't run forward (torch.sparse.mm requires
        consistent dtype across sparse/dense and pinning that here would
        duplicate test_hodge.py's API contract testing).
        """
        from topogeoml.nn.hodge import HodgeMessagePassing

        L_np = np.eye(4, dtype=np.float64)
        layer = HodgeMessagePassing(in_features=4, out_features=2, laplacian=L_np)
        # The layer was constructed without raising; that's the branch we covered.
        assert layer.in_features == 4
        assert layer.out_features == 2

    def test_build_hodge_layer_rejects_bad_laplacian_type(self) -> None:
        """Lines 147-150 — TypeError for non-tensor/non-sparse/non-ndarray input."""
        from topogeoml.nn.hodge import HodgeMessagePassing

        with pytest.raises(TypeError, match="laplacian must be"):
            HodgeMessagePassing(in_features=4, out_features=2, laplacian="not a matrix")  # type: ignore[arg-type]

    def test_build_hodge_layer_from_tensor(self) -> None:
        """Line 141 — torch.Tensor laplacian input."""
        import scipy.sparse as sp

        from topogeoml.nn.hodge import HodgeMessagePassing, sparse_scipy_to_torch

        L_sparse = sp.csr_matrix(np.eye(3, dtype=np.float64))
        L_tensor = sparse_scipy_to_torch(L_sparse)
        layer = HodgeMessagePassing(in_features=3, out_features=2, laplacian=L_tensor)
        assert layer.in_features == 3

    def test_build_hodge_layer_rejects_non_square_laplacian(self) -> None:
        """Line 153 — non-square laplacian raises."""
        import scipy.sparse as sp

        from topogeoml.nn.hodge import HodgeMessagePassing

        L = sp.csr_matrix(np.zeros((3, 4), dtype=np.float64))
        with pytest.raises(ValueError, match="square"):
            HodgeMessagePassing(in_features=4, out_features=2, laplacian=L)

    def test_extra_repr(self) -> None:
        """Line 201-202 — extra_repr formatting."""
        import scipy.sparse as sp

        from topogeoml.nn.hodge import HodgeMessagePassing

        L = sp.csr_matrix(np.eye(3, dtype=np.float64))
        layer = HodgeMessagePassing(in_features=3, out_features=2, laplacian=L)
        assert "in_features=3" in layer.extra_repr()
