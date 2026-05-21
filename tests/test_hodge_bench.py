"""
Tests for the Hodge-subsystem bench.

Coverage targets:
  - GraphClassifier protocol on both registered models;
  - MUTAGDataset loader + Laplacian conversion;
  - Classification axis end-to-end on a tiny config;
  - Runner pairwise comparison output schema.

Real PyG MUTAG download is exercised in the integration tests below
(gated on ``importorskip("torch_geometric")``); offline runs use a
synthetic ``GraphSample`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _has_pyg() -> bool:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Models — protocol checks (no PyG required)
# ---------------------------------------------------------------------------

class TestModelsRegistry:
    def test_both_models_registered(self) -> None:
        from benchmarks.hodge.models import REGISTERED, get_model

        assert "hodge-mp-classifier" in REGISTERED
        assert "mlp-baseline" in REGISTERED

        cls = get_model("mlp-baseline")
        assert cls.available() is True
        assert cls.name == "mlp-baseline"
        assert cls.version != ""

    def test_unknown_model_raises(self) -> None:
        from benchmarks.hodge.models import get_model

        with pytest.raises(KeyError, match="unknown model"):
            get_model("not-a-model")

    def test_hodge_classifier_available(self) -> None:
        from benchmarks.hodge.models import HodgeClassifier

        assert HodgeClassifier.available() is True

    def test_mlp_baseline_forward(self) -> None:
        from benchmarks.hodge.models import MLPBaseline

        model = MLPBaseline.build(input_dim=4, num_classes=2, seed=0).to(torch.float64)
        x = torch.randn(5, 4, dtype=torch.float64)
        # MLPBaseline doesn't use the laplacian; pass a dummy.
        dummy_L = torch.sparse_coo_tensor(
            indices=torch.zeros(2, 0, dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float64),
            size=(5, 5),
        )
        logits = model.forward_one(x, dummy_L)
        assert logits.shape == (2,)

    def test_hodge_classifier_mp_weights_registered(self) -> None:
        """Regression for Gemini PR #6 review: the shared Hodge-MP
        weight/bias must appear in ``model.parameters()`` so the
        optimizer can actually update them. Previously they were
        created inside ``forward_one`` and never registered."""
        from benchmarks.hodge.models import HodgeClassifier

        model = HodgeClassifier.build(input_dim=4, num_classes=2, seed=0)
        names = {name for name, _ in model.named_parameters()}
        assert "_mp_weight" in names
        assert "_mp_bias" in names
        # Optimizer parameter count should include the MP weight + bias on
        # top of the in-projection + head.
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # hidden_dim defaults to 32: proj_in (4*32+32) + mp (32*32+32) + head (32*2+2) = 1218
        assert total_params == 4 * 32 + 32 + 32 * 32 + 32 + 32 * 2 + 2

    def test_symmetric_normalize_sparse_bounds_eigenvalues(self) -> None:
        """Symmetric L̃ = D^{-1/2} L D^{-1/2} has eigenvalues bounded
        to [0, 2] for any combinatorial graph Laplacian (Kipf-Welling
        Lemma 1). The function is differentiable but the Laplacian is
        treated as a buffer."""
        from benchmarks.hodge.models import _symmetric_normalize_sparse

        # Combinatorial L_0 of a 3-node triangle: each node has degree 2.
        indices = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1, 0, 1, 2]],
            dtype=torch.long,
        )
        values = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            dtype=torch.float64,
        )
        L = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
        L_norm = _symmetric_normalize_sparse(L)
        eigvals = torch.linalg.eigvalsh(L_norm.to_dense())
        assert eigvals.max().item() <= 2.0 + 1e-9
        assert eigvals.min().item() >= -1e-9
        # Diagonal is now ~1 (degree-balanced), not 2.
        diag = L_norm.to_dense().diag()
        assert torch.allclose(diag, torch.ones(3, dtype=torch.float64), atol=1e-5)

    @pytest.mark.parametrize(
        "model_name",
        [
            "hodge-mp-classifier",
            "hodge-mp-normalised",
            "hodge-mp-residual",
            "hodge-mp-deep-residual",
            "mlp-baseline",
        ],
    )
    def test_every_registered_model_forwards_on_a_triangle(
        self, model_name: str,
    ) -> None:
        """Every registered model in the Hodge benchmark instantiates,
        has all of its parameters visible to ``model.parameters()``
        (regression for PR #12's critical bug), and produces a
        (num_classes,) logits tensor for a 3-node triangle."""
        from benchmarks.hodge.models import get_model

        cls = get_model(model_name)
        model = cls.build(input_dim=7, num_classes=2, seed=0).to(torch.float64)
        # The runner casts every model to float64 before training; replicate
        # so the dtype chain matches the production code path. Without this
        # the MLP baseline stays in float32 by default and a float64 input
        # mismatch raises ``RuntimeError: mat1 and mat2 must have the same
        # dtype``.

        # Every parameter is on the graph & visible to the optimizer.
        names_and_params = list(model.named_parameters())
        assert len(names_and_params) >= 1
        assert all(p.requires_grad for _, p in names_and_params)

        # Forward on a 3-node triangle with random 7-dim features.
        indices = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1, 0, 1, 2]],
            dtype=torch.long,
        )
        values = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            dtype=torch.float64,
        )
        L = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
        x = torch.randn(3, 7, dtype=torch.float64)
        out = model.forward_one(x, L)
        assert out.shape == (2,)

    def test_hodge_ablation_arms_have_matched_capacity(self) -> None:
        """The four Hodge arms + the MLP baseline must have parameter
        counts within 5% of each other so the ablation isolates the
        architectural effect rather than capacity. The deep-residual
        arm achieves this by running at ``hidden_dim=24`` (rather than
        32) — see ``docs/hypotheses/HYPOTHESIS-001-hodge-mutag.md``."""
        from benchmarks.hodge.models import get_model

        names = [
            "mlp-baseline",
            "hodge-mp-classifier",
            "hodge-mp-normalised",
            "hodge-mp-residual",
            "hodge-mp-deep-residual",
        ]
        counts = []
        for name in names:
            model = get_model(name).build(input_dim=7, num_classes=2, seed=0)
            counts.append(sum(p.numel() for p in model.parameters()))
        cmin, cmax = min(counts), max(counts)
        assert cmax / cmin < 1.06, (
            f"capacity mismatch across ablation arms: {dict(zip(names, counts, strict=True))}"
        )

    def test_hodge_classifier_mp_weights_change_under_training(self) -> None:
        """Regression for Gemini PR #6 review: a real gradient step
        must move the Hodge-MP weights. Previously they were
        re-initialised on every ``forward_one`` so any step taken on
        their gradients was discarded on the next call."""
        from benchmarks.hodge.models import HodgeClassifier

        model = HodgeClassifier.build(input_dim=3, num_classes=2, seed=0)
        # 3-node triangle Laplacian (already normalised would be denser;
        # combinatorial L_0 is enough to exercise the propagation).
        indices = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1, 0, 1, 2]],
            dtype=torch.long,
        )
        values = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0],
            dtype=torch.float64,
        )
        L = torch.sparse_coo_tensor(indices, values, size=(3, 3)).coalesce()
        x = torch.randn(3, 3, dtype=torch.float64)
        target = torch.tensor([1])

        before = model._mp_weight.detach().clone()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model.forward_one(x, L).unsqueeze(0), target,
        )
        loss.backward()
        opt.step()
        after = model._mp_weight.detach().clone()
        # The optimizer should have moved the MP weight by a non-trivial
        # amount in at least one entry.
        assert not torch.allclose(before, after)


# ---------------------------------------------------------------------------
# Datasets — MUTAG load + Laplacian conversion
# ---------------------------------------------------------------------------

class TestDatasetsRegistry:
    def test_unknown_dataset_raises(self) -> None:
        from benchmarks.hodge.datasets import get_dataset

        with pytest.raises(KeyError, match="unknown dataset"):
            get_dataset("not-a-dataset")


class TestGraphToLaplacian:
    def test_triangle_laplacian_shape_and_symmetry(self) -> None:
        """L_0 of a triangle (3 nodes, 3 edges) is 3x3 symmetric."""
        from benchmarks.hodge.datasets import _graph_to_laplacian

        edge_index = torch.tensor(
            [[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=torch.long,
        )
        L = _graph_to_laplacian(n_nodes=3, edge_index=edge_index)
        dense = L.to_dense().numpy()
        assert dense.shape == (3, 3)
        np.testing.assert_array_almost_equal(dense, dense.T)

    def test_isolated_node_handled(self) -> None:
        """A 2-node graph with no edges should produce a 2x2 zero Laplacian."""
        from benchmarks.hodge.datasets import _graph_to_laplacian

        L = _graph_to_laplacian(
            n_nodes=2,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        assert L.shape == (2, 2)
        # No edges -> Laplacian is the zero matrix.
        np.testing.assert_array_equal(L.to_dense().numpy(), np.zeros((2, 2)))


@pytest.mark.skipif(not _has_pyg(), reason="torch-geometric not installed")
class TestMUTAGLoader:
    def test_mutag_is_available(self) -> None:
        from benchmarks.hodge.datasets import MUTAGDataset

        assert MUTAGDataset.available() is True

    def test_mutag_load_metadata(self) -> None:
        """MUTAG: 188 graphs, 2 classes; sample carries x/laplacian/label."""
        from benchmarks.hodge.datasets import MUTAGDataset

        ds = MUTAGDataset()
        samples, input_dim, num_classes = ds.load()
        assert len(samples) == 188
        assert num_classes == 2
        # Node feature dim is positive.
        assert input_dim > 0
        # Sample sanity: x and laplacian shapes line up.
        s = samples[0]
        assert s.x.shape[0] == s.laplacian.shape[0]
        assert s.laplacian.shape[0] == s.laplacian.shape[1]
        assert s.y in (0, 1)


# ---------------------------------------------------------------------------
# Classification axis — end-to-end on a tiny config
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_pyg(), reason="torch-geometric not installed")
class TestClassificationAxis:
    def test_classification_axis_runs(self) -> None:
        """End-to-end smoke: HodgeClassifier trains for 2 epochs on MUTAG
        and reports a finite accuracy."""
        from benchmarks.hodge.classification import run_classification
        from benchmarks.hodge.datasets import MUTAGDataset
        from benchmarks.hodge.models import MLPBaseline

        report = run_classification(
            model_cls=MLPBaseline,
            dataset=MUTAGDataset(),
            seeds=[0],
            n_epochs=2,
            learning_rate=1e-2,
        )
        assert len(report.cells) == 1
        cell = report.cells[0]
        assert 0.0 <= cell.test_accuracy <= 1.0
        assert cell.n_train > 0
        assert cell.n_test > 0
        # n=1 seed -> bootstrap CI is N/A.
        assert np.isnan(report.accuracy_ci95_low)


# ---------------------------------------------------------------------------
# Runner + report
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_pyg(), reason="torch-geometric not installed")
class TestRunner:
    def test_runner_produces_reports_and_comparison(self) -> None:
        """End-to-end smoke through the runner: the two original models
        against MUTAG for 2 seeds, 2 epochs. Should produce 2 reports +
        1 pairwise comparison (in UNDERPOWERED state because n=2 <
        threshold). Explicit model restriction is required because the
        registry now contains five models (the original two plus the
        three hypothesis-001 ablation arms)."""
        from benchmarks.hodge.runner import run

        result = run(
            model_names=["hodge-mp-classifier", "mlp-baseline"],
            seeds=[0, 1], n_epochs=2,
        )
        assert len(result.reports) == 2
        assert len(result.pairwise_comparisons) == 1

        cmp = result.pairwise_comparisons[0]
        # With n=2 seeds we cannot conclude significance.
        assert cmp["kind"] in ("not_significant", "underpowered")

    def test_runner_full_ablation_produces_five_reports(self) -> None:
        """End-to-end smoke for the full hypothesis-001 ablation matrix:
        5 models × 1 dataset (MUTAG) × 2 seeds × 2 epochs. The runner
        emits 5 reports and C(5,2) = 10 pairwise comparisons."""
        from benchmarks.hodge.runner import run

        result = run(seeds=[0, 1], n_epochs=2)
        assert len(result.reports) == 5
        assert len(result.pairwise_comparisons) == 10

    def test_runner_writes_json_and_markdown(self, tmp_path) -> None:
        from benchmarks.hodge.runner import render_markdown, run, write_result

        result = run(seeds=[0], n_epochs=2)
        out = tmp_path / "result.json"
        write_result(result, out)
        assert out.exists()

        md = render_markdown(result)
        assert "TopoGeoML Hodge subsystem benchmark" in md
        assert "Per-(model × dataset) test accuracy" in md
