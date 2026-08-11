"""
H011b execution-contract tests.

The H011b preregistration (docs/hypotheses/HYPOTHESIS-011b-l1-collab.md)
fixes a 30-seed COLLAB comparison of `l1-hodge-residual` against three
matched controls, and requires the final artifact to report a triangle
census under the exact dataset loader. Before this file existed the
benchmark suite had no test for the L_1 model, the degree-feature loader,
or any census tooling. These tests close that gap without running any part
of the confirmatory experiment:

  - L_1 correctness on known graphs (filled triangle: the up-Laplacian
    B_2 B_2^T must be present, giving L_1 = 3I; triangle-free path:
    down-term only);
  - L_1 memoization is numerically invisible (cached forward == fresh
    forward, eviction on garbage collection);
  - exact four-arm capacity match at COLLAB's input_dim=1 / 3 classes;
  - triangle census values on synthetic samples, and census edge
    reconstruction identical to the model's own reconstruction;
  - degree-feature loader consistency (degree column equals the L_0
    diagonal) on MUTAG, which CI already downloads.

Offline tests use synthetic GraphSamples; network-touching tests follow
the existing ``_has_pyg`` gating convention.
"""

from __future__ import annotations

import gc

import pytest

torch = pytest.importorskip("torch")

from benchmarks.hodge.datasets import GraphSample, _graph_to_laplacian
from benchmarks.hodge.models import (
    _L1_OPERATOR_CACHE,
    L1HodgeResidualClassifier,
    _L1HodgeResidualGraphClassifier,
)


def _has_pyg() -> bool:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


def _collab_cached() -> bool:
    """True when the COLLAB TUDataset is already in the local cache.

    The COLLAB download is ~70 MB; the loader smoke test below runs only
    where the experiment would run (cache prewarmed), never forcing the
    download in unit CI.
    """
    import os
    from pathlib import Path

    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return (Path(base) / "topogeoml" / "benchmarks" / "hodge" / "COLLAB").is_dir()


def _triangle_l0() -> torch.Tensor:
    ei = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    return _graph_to_laplacian(3, ei)


def _path_l0() -> torch.Tensor:
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    return _graph_to_laplacian(3, ei)


@pytest.fixture(autouse=True)
def _clear_l1_cache():
    _L1_OPERATOR_CACHE.clear()
    yield
    _L1_OPERATOR_CACHE.clear()


class TestL1Operator:
    def test_filled_triangle_l1_includes_up_term(self) -> None:
        """K_3 lifts to a filled 2-simplex: L_1 = B_1^T B_1 + B_2 B_2^T = 3I.

        The down-term alone (cycle C_3 without the face) has nonzero
        off-diagonals, so equality with 3I proves the B_2 B_2^T up-term is
        present and correct — the mechanism H011b exists to test.
        """
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        L1, edge_pairs = model._compute_l1(3, _triangle_l0())
        assert edge_pairs == [(0, 1), (0, 2), (1, 2)]
        expected = 3.0 * torch.eye(3, dtype=torch.float64)
        assert torch.equal(L1.to_dense(), expected)

    def test_triangle_free_path_l1_is_down_term_only(self) -> None:
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        L1, edge_pairs = model._compute_l1(3, _path_l0())
        assert edge_pairs == [(0, 1), (1, 2)]
        expected = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64)
        assert torch.equal(L1.to_dense(), expected)

    def test_edgeless_graph_yields_empty_operator(self) -> None:
        ei = torch.zeros((2, 0), dtype=torch.int64)
        L0 = _graph_to_laplacian(2, ei)
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        L1, edge_pairs = model._compute_l1(2, L0)
        assert edge_pairs == []
        assert L1.shape == (0, 0)


class TestL1OperatorCache:
    def test_second_call_returns_cached_objects(self) -> None:
        L0 = _triangle_l0()
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        first = model._propagation_operator(3, L0)
        second = model._propagation_operator(3, L0)
        assert first[0] is second[0]
        assert first[1] is second[1]

    def test_cache_shared_across_model_instances(self) -> None:
        """A fresh model per seed must still reuse the per-graph operator."""
        L0 = _triangle_l0()
        model_a = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        model_b = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        first = model_a._propagation_operator(3, L0)
        second = model_b._propagation_operator(3, L0)
        assert first[0] is second[0]

    def test_cached_equals_fresh_recomputation(self) -> None:
        L0 = _triangle_l0()
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        cached_norm, cached_edges = model._propagation_operator(3, L0)
        _L1_OPERATOR_CACHE.clear()
        fresh_norm, fresh_edges = model._propagation_operator(3, L0)
        assert cached_edges == fresh_edges
        assert cached_norm is not None and fresh_norm is not None
        assert torch.equal(cached_norm.to_dense(), fresh_norm.to_dense())

    def test_cached_operator_is_normalized_composition(self) -> None:
        """The cached operator equals normalize(raw L_1) exactly."""
        from benchmarks.hodge.models import _symmetric_normalize_sparse

        L0 = _triangle_l0()
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        raw_L1, _ = model._compute_l1(3, L0)
        norm, _ = model._propagation_operator(3, L0)
        assert norm is not None
        expected = _symmetric_normalize_sparse(raw_L1)
        assert torch.equal(norm.to_dense(), expected.to_dense())

    def test_forward_identical_with_and_without_cache(self) -> None:
        """End-to-end guarantee: memoization cannot change any logit."""
        x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        L0 = _triangle_l0()

        torch.manual_seed(7)
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        cold = model.forward_one(x, L0)  # populates the cache
        warm = model.forward_one(x, L0)  # served from the cache
        _L1_OPERATOR_CACHE.clear()
        uncached = model.forward_one(x, L0)  # recomputed from scratch
        assert torch.equal(cold, warm)
        assert torch.equal(cold, uncached)

    def test_cache_evicts_when_laplacian_is_collected(self) -> None:
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        L0 = _triangle_l0()
        model._propagation_operator(3, L0)
        assert len(_L1_OPERATOR_CACHE) == 1
        del L0
        gc.collect()
        assert len(_L1_OPERATOR_CACHE) == 0

    def test_version_records_memoization_change(self) -> None:
        assert L1HodgeResidualClassifier.version == "1.0.1"


class TestCollabCapacityMatch:
    def test_four_arms_have_identical_parameter_count(self) -> None:
        """At COLLAB's input_dim=1 / num_classes=3, all four preregistered
        arms have exactly 1219 trainable parameters — an exact match, with
        no tolerance needed (contrast NCI1's 2.78% sheaf margin)."""
        from benchmarks.hodge.models import REGISTERED

        arms = [
            "l1-hodge-residual",
            "hodge-mp-residual",
            "gin-residual",
            "mlp-baseline",
        ]
        counts = {}
        for name in arms:
            module = REGISTERED[name].build(1, 3, seed=0)
            counts[name] = sum(p.numel() for p in module.parameters())
        assert counts == {name: 1219 for name in arms}

    def test_l1_arm_forwards_on_triangle(self) -> None:
        """Fills the registry-forward gap: the H011b arms absent from
        ``test_every_registered_model_forwards_on_a_triangle``."""
        from benchmarks.hodge.models import REGISTERED

        x = torch.ones((3, 1), dtype=torch.float64)
        L0 = _triangle_l0()
        for name in ("l1-hodge-residual", "gin-residual", "sheaf-residual"):
            model = REGISTERED[name].build(1, 3, seed=0)
            logits = model.forward_one(x, L0)
            assert logits.shape == (3,)
            assert torch.isfinite(logits).all()


class TestTriangleCensus:
    def test_census_on_synthetic_samples(self) -> None:
        from benchmarks.hodge.triangle_census import census_from_samples

        samples = [
            GraphSample(
                x=torch.ones((3, 1), dtype=torch.float64),
                laplacian=_triangle_l0(),
                y=0,
            ),
            GraphSample(
                x=torch.ones((3, 1), dtype=torch.float64),
                laplacian=_path_l0(),
                y=1,
            ),
        ]
        census = census_from_samples(samples)
        assert census["per_graph_triangles"] == [1, 0]
        summary = census["summary"]
        assert summary["n_graphs"] == 2
        assert summary["n_graphs_with_triangles"] == 1
        assert summary["fraction_with_triangles"] == 0.5
        assert summary["total_triangles"] == 1
        assert summary["triangles_max"] == 1

    def test_census_edges_match_model_reconstruction(self) -> None:
        """The census must observe exactly the edge set the L_1 model sees."""
        from benchmarks.hodge.triangle_census import edges_from_l0

        L0 = _triangle_l0()
        model = _L1HodgeResidualGraphClassifier(input_dim=1, num_classes=3)
        _, model_edges = model._compute_l1(3, L0)
        assert edges_from_l0(L0) == model_edges

    def test_census_cli_writes_json(self, tmp_path, monkeypatch) -> None:
        import json

        import benchmarks.hodge.triangle_census as tc

        class _StubDataset:
            name = "stub"
            version = "0.0.0"

            @staticmethod
            def load():
                samples = [
                    GraphSample(
                        x=torch.ones((3, 1), dtype=torch.float64),
                        laplacian=_triangle_l0(),
                        y=0,
                    ),
                ]
                return samples, 1, 2

        monkeypatch.setitem(tc.DATASETS, "stub", _StubDataset())
        out = tmp_path / "census.json"
        rc = tc.main(["--dataset", "stub", "--output", str(out)])
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["schema_version"] == tc.CENSUS_SCHEMA_VERSION
        assert payload["dataset"] == "stub"
        assert payload["dataset_version"] == "0.0.0"
        assert payload["per_graph_triangles"] == [1]
        assert payload["summary"]["n_graphs_with_triangles"] == 1


@pytest.mark.skipif(not _has_pyg(), reason="torch-geometric not installed")
class TestDegreeFeatureLoader:
    def test_degree_features_equal_l0_diagonal_on_mutag(self) -> None:
        """The degree-feature path (COLLAB's loader) must produce a degree
        column identical to the L_0 diagonal of the same stored sample.
        MUTAG is used because CI already downloads it."""
        from benchmarks.hodge.datasets import _load_tudataset_degree_features

        samples, input_dim, num_classes = _load_tudataset_degree_features("MUTAG")
        assert input_dim == 1
        assert num_classes == 2
        assert len(samples) == 188
        for sample in samples[:10]:
            diag = sample.laplacian.to_dense().diagonal()
            assert torch.equal(sample.x[:, 0], diag)


@pytest.mark.skipif(
    not _has_pyg() or not _collab_cached(),
    reason="COLLAB cache not prewarmed (loader smoke test runs only where the experiment runs)",
)
class TestCOLLABLoaderSmoke:
    def test_collab_shape_contract(self) -> None:
        from benchmarks.hodge.datasets import get_dataset

        samples, input_dim, num_classes = get_dataset("collab").load()
        assert len(samples) == 5000
        assert input_dim == 1
        assert num_classes == 3
        assert samples[0].x.shape[1] == 1
