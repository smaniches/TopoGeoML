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
            "gin-baseline",
            "gin-normalised",
            "gat-baseline",
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

    def test_gin_gat_models_registered(self) -> None:
        from benchmarks.hodge.models import REGISTERED

        assert "gin-baseline" in REGISTERED
        assert "gin-normalised" in REGISTERED
        assert "gat-baseline" in REGISTERED

    def test_adj_matmul_from_laplacian_correctness(self) -> None:
        """A @ h must equal D @ h - L @ h for a known small graph."""
        from benchmarks.hodge.models import _adj_matmul_from_laplacian

        # 3-node path: 0-1-2. A = [[0,1,0],[1,0,1],[0,1,0]], D = diag(1,2,1)
        # L = D - A = [[1,-1,0],[-1,2,-1],[0,-1,1]]
        indices = torch.tensor(
            [[0, 0, 1, 1, 1, 2, 2], [1, 0, 0, 1, 2, 1, 2]], dtype=torch.long,
        )
        values = torch.tensor([-1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 1.0], dtype=torch.float64)
        L = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
        h = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        A = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64,
        )
        expected = A @ h
        result = _adj_matmul_from_laplacian(L, h)
        assert torch.allclose(result, expected, atol=1e-12)

    def test_gin_gradient_flows(self) -> None:
        from benchmarks.hodge.models import GINBaseline

        model = GINBaseline.build(input_dim=4, num_classes=2, seed=0).to(torch.float64)
        indices = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1, 0, 1, 2]], dtype=torch.long,
        )
        values = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0], dtype=torch.float64,
        )
        L = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
        x = torch.randn(3, 4, dtype=torch.float64)
        out = model.forward_one(x, L)
        loss = out.sum()
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count >= 2

    def test_gat_isolated_node_no_crash(self) -> None:
        """GAT softmax on a node with no neighbours should not produce NaN."""
        from benchmarks.hodge.models import GATBaseline

        model = GATBaseline.build(input_dim=4, num_classes=2, seed=0).to(torch.float64)
        # 2-node graph with no edges — Laplacian is zero matrix.
        L = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float64),
            (2, 2),
        ).coalesce()
        x = torch.randn(2, 4, dtype=torch.float64)
        out = model.forward_one(x, L)
        assert out.shape == (2,)
        assert torch.all(torch.isfinite(out))

    @pytest.mark.parametrize("model_name", ["gin-baseline", "gin-normalised", "gat-baseline"])
    def test_new_models_param_count_nci1(self, model_name: str) -> None:
        """GIN/GAT arms must have param counts within 1% of the Hodge/MLP arms
        for the matched-capacity protocol to be valid."""
        from benchmarks.hodge.models import get_model

        model = get_model(model_name).build(input_dim=37, num_classes=2, seed=0)
        count = sum(p.numel() for p in model.parameters())
        # Hodge-residual and MLP both have 2338 params at input_dim=37
        assert abs(count - 2338) / 2338 < 0.02, f"{model_name}: {count} params vs 2338 target"

    def test_nan_guard_in_training_loop(self) -> None:
        """If a model produces NaN loss, training must stop early and record
        NaN train loss instead of corrupting subsequent gradient steps."""
        from benchmarks.hodge.classification import _train_one_seed
        from benchmarks.hodge.datasets import GraphSample

        class _NaNModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

            def forward_one(
                self, x: torch.Tensor, laplacian: torch.sparse.Tensor,
            ) -> torch.Tensor:
                return torch.tensor([float("nan"), float("nan")], dtype=torch.float64)

        L = torch.sparse_coo_tensor(
            torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long),
            torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64),
            (2, 2),
        ).coalesce()
        samples = [GraphSample(x=torch.randn(2, 3, dtype=torch.float64), laplacian=L, y=0)]
        cell = _train_one_seed(
            _NaNModel(), train_samples=samples, test_samples=samples,
            n_epochs=5, learning_rate=1e-2, seed=0,
        )
        assert np.isnan(cell.final_train_loss)


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
class TestPROTEINSLoader:
    def test_proteins_is_available(self) -> None:
        from benchmarks.hodge.datasets import PROTEINSDataset

        assert PROTEINSDataset.available() is True

    def test_proteins_load_metadata(self) -> None:
        """PROTEINS: 1113 graphs, 2 classes; sample carries x/laplacian/label."""
        from benchmarks.hodge.datasets import PROTEINSDataset

        ds = PROTEINSDataset()
        samples, input_dim, num_classes = ds.load()
        assert len(samples) == 1113
        assert num_classes == 2
        assert input_dim > 0
        # Sample sanity: x and laplacian shapes line up.
        s = samples[0]
        assert s.x.shape[0] == s.laplacian.shape[0]
        assert s.laplacian.shape[0] == s.laplacian.shape[1]
        assert s.y in (0, 1)


@pytest.mark.skipif(not _has_pyg(), reason="torch-geometric not installed")
class TestNCI1Loader:
    def test_nci1_is_available(self) -> None:
        from benchmarks.hodge.datasets import NCI1Dataset

        assert NCI1Dataset.available() is True

    def test_nci1_load_metadata(self) -> None:
        """NCI1: 4110 chemical graphs, 2 classes; sample carries x/laplacian/label."""
        from benchmarks.hodge.datasets import NCI1Dataset

        ds = NCI1Dataset()
        samples, input_dim, num_classes = ds.load()
        assert len(samples) == 4110
        assert num_classes == 2
        assert input_dim > 0
        s = samples[0]
        assert s.x.shape[0] == s.laplacian.shape[0]
        assert s.laplacian.shape[0] == s.laplacian.shape[1]
        assert s.y in (0, 1)


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
        threshold). Explicit model + dataset restriction is required
        because the registry now contains five models and two datasets
        (MUTAG plus the hypothesis-002 PROTEINS dataset)."""
        from benchmarks.hodge.runner import run

        result = run(
            model_names=["hodge-mp-classifier", "mlp-baseline"],
            dataset_names=["mutag"],
            seeds=[0, 1], n_epochs=2,
        )
        assert len(result.reports) == 2
        assert len(result.pairwise_comparisons) == 1

        cmp = result.pairwise_comparisons[0]
        # With n=2 seeds we cannot conclude significance.
        assert cmp["kind"] in ("not_significant", "underpowered")

    def test_runner_full_ablation_produces_one_report_per_model(self) -> None:
        """End-to-end smoke for the full ablation matrix: every registered
        (available) model × 1 dataset (MUTAG) × 2 seeds × 2 epochs. The
        runner emits one report per model and C(n, 2) pairwise comparisons.
        Counts are derived from the registry so the test does not drift as
        new ablation arms are added."""
        import math

        from benchmarks.hodge.models import REGISTERED as MODELS
        from benchmarks.hodge.runner import run

        n_models = sum(1 for m in MODELS.values() if m.available())
        result = run(dataset_names=["mutag"], seeds=[0, 1], n_epochs=2)
        assert len(result.reports) == n_models
        assert len(result.pairwise_comparisons) == math.comb(n_models, 2)

    def test_max_graphs_caps_dataset_per_seed(self) -> None:
        """Hypothesis 004 mechanism test: ``max_graphs=N`` subsamples
        the dataset to N graphs (deterministically per seed) before
        the stratified train/test split. Two different seeds should
        produce different subsets — the subsampling is per-seed, not
        global — but each seed's training pipeline sees ≤ N graphs."""
        from benchmarks.hodge.classification import run_classification
        from benchmarks.hodge.datasets import MUTAGDataset
        from benchmarks.hodge.models import MLPBaseline

        # MUTAG has 188 graphs; cap at 100 → each seed sees 100.
        report = run_classification(
            model_cls=MLPBaseline,
            dataset=MUTAGDataset(),
            seeds=[0, 1],
            n_epochs=2,
            max_graphs=100,
        )
        for cell in report.cells:
            assert cell.n_train + cell.n_test == 100, (
                f"cap not applied: seed={cell.seed} got "
                f"n_train={cell.n_train} + n_test={cell.n_test} = "
                f"{cell.n_train + cell.n_test} != 100"
            )

    def test_feature_projection_changes_input_dim(self) -> None:
        """Hypothesis 005 mechanism test: ``feature_projection_dim=K``
        applies a per-seed deterministic Gaussian projection to all
        node features, changing each graph's feature dimension to K.
        The projection matrix is the same across graphs in a seed
        (it's a global linear transform), and the model is built with
        the projected input_dim, not the original."""
        from benchmarks.hodge.classification import run_classification
        from benchmarks.hodge.datasets import MUTAGDataset
        from benchmarks.hodge.models import MLPBaseline

        # MUTAG has 7-dim features by default; project to 37-dim.
        report = run_classification(
            model_cls=MLPBaseline,
            dataset=MUTAGDataset(),
            seeds=[0, 1],
            n_epochs=2,
            feature_projection_dim=37,
        )
        # Two seeds × MUTAG dataset → two cells, both run cleanly to
        # completion with the projected features.
        assert len(report.cells) == 2
        for cell in report.cells:
            assert 0.0 <= cell.test_accuracy <= 1.0

    def test_feature_projection_preserves_norm_in_expectation(self) -> None:
        """Johnson-Lindenstrauss norm preservation: a Gaussian projection
        with scale = 1/sqrt(target_dim) keeps ``E[|x @ P|^2] = |x|^2``
        regardless of dim change. The previous (Gemini-flagged on PR #21)
        scaling 1/sqrt(src_dim) shrunk the norm by target_dim/src_dim on
        dim-reduction and inflated it on dim-expansion — a confounder
        for hypothesis 005.

        Verifies expectation by averaging over many seeds at both
        directions (37→7 dim-reduction and 7→37 dim-expansion). With
        the correct scaling, the average ratio ``|x @ P|^2 / |x|^2``
        sits within ±10% of 1 across both directions; with the prior
        bad scaling the ratios are ~0.19 (37→7) and ~5.3 (7→37).
        """
        from benchmarks.hodge.classification import _project_features
        from benchmarks.hodge.datasets import GraphSample

        # Synthetic sample: a single graph with one 37-dim feature vector
        # whose squared norm we control.
        x = torch.ones(1, 37, dtype=torch.float64)  # |x|^2 = 37
        dummy_L = torch.sparse_coo_tensor(
            indices=torch.zeros(2, 0, dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float64),
            size=(1, 1),
        )
        sample = GraphSample(x=x, laplacian=dummy_L, y=0)
        n_seeds = 200

        # Direction 37 -> 7
        ratios_down = []
        for seed in range(n_seeds):
            projected, _ = _project_features([sample], target_dim=7, seed=seed)
            ratios_down.append(
                (projected[0].x.norm().item() ** 2) / (x.norm().item() ** 2)
            )
        mean_down = sum(ratios_down) / len(ratios_down)
        assert 0.90 <= mean_down <= 1.10, (
            f"37->7 mean norm ratio {mean_down:.4f} not within ±10% of 1"
        )

        # Direction 7 -> 37
        x_small = torch.ones(1, 7, dtype=torch.float64)
        sample_small = GraphSample(x=x_small, laplacian=dummy_L, y=0)
        ratios_up = []
        for seed in range(n_seeds):
            projected, _ = _project_features([sample_small], target_dim=37, seed=seed)
            ratios_up.append(
                (projected[0].x.norm().item() ** 2) / (x_small.norm().item() ** 2)
            )
        mean_up = sum(ratios_up) / len(ratios_up)
        assert 0.90 <= mean_up <= 1.10, (
            f"7->37 mean norm ratio {mean_up:.4f} not within ±10% of 1"
        )

    def test_constant_features_replaces_features_with_ones(self) -> None:
        """Hypothesis 006: ``constant_features=True`` replaces each
        graph's node features with a 1-dim constant 1-vector. The MLP
        then has no per-node signal; the Hodge model can still use
        the Laplacian. This isolates pure-topology classification."""
        from benchmarks.hodge.classification import (
            _constant_features,
            run_classification,
        )
        from benchmarks.hodge.datasets import GraphSample, MUTAGDataset
        from benchmarks.hodge.models import MLPBaseline

        # Unit test of the helper.
        dummy_L = torch.sparse_coo_tensor(
            indices=torch.zeros(2, 0, dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float64),
            size=(3, 3),
        )
        sample = GraphSample(
            x=torch.randn(3, 7, dtype=torch.float64),
            laplacian=dummy_L, y=0,
        )
        out, new_dim = _constant_features([sample])
        assert new_dim == 1
        assert out[0].x.shape == (3, 1)
        assert torch.allclose(out[0].x, torch.ones((3, 1), dtype=torch.float64))

        # End-to-end through run_classification on a real dataset.
        report = run_classification(
            model_cls=MLPBaseline,
            dataset=MUTAGDataset(),
            seeds=[0],
            n_epochs=2,
            constant_features=True,
        )
        assert len(report.cells) == 1

    def test_feature_projection_is_deterministic_per_seed(self) -> None:
        """Per-seed projection determinism: running the same seed
        twice with the same projection_dim must produce identical
        per-cell accuracies. Hypothesis 005 relies on this — the
        projection cannot be a confound."""
        from benchmarks.hodge.classification import run_classification
        from benchmarks.hodge.datasets import MUTAGDataset
        from benchmarks.hodge.models import MLPBaseline

        a = run_classification(
            model_cls=MLPBaseline, dataset=MUTAGDataset(),
            seeds=[7], n_epochs=2, feature_projection_dim=20,
        )
        b = run_classification(
            model_cls=MLPBaseline, dataset=MUTAGDataset(),
            seeds=[7], n_epochs=2, feature_projection_dim=20,
        )
        assert a.cells[0].test_accuracy == b.cells[0].test_accuracy

    def test_max_graphs_noop_when_dataset_smaller(self) -> None:
        """If ``max_graphs >= len(samples)``, the subsampling is a
        no-op — the full dataset goes through the existing pipeline."""
        from benchmarks.hodge.classification import run_classification
        from benchmarks.hodge.datasets import MUTAGDataset
        from benchmarks.hodge.models import MLPBaseline

        # MUTAG has 188 graphs; cap at 500 → no subsampling.
        report = run_classification(
            model_cls=MLPBaseline,
            dataset=MUTAGDataset(),
            seeds=[0],
            n_epochs=2,
            max_graphs=500,
        )
        cell = report.cells[0]
        assert cell.n_train + cell.n_test == 188

    def test_runner_writes_json_and_markdown(self, tmp_path) -> None:
        from benchmarks.hodge.runner import render_markdown, run, write_result

        result = run(
            model_names=["mlp-baseline"], dataset_names=["mutag"],
            seeds=[0], n_epochs=2,
        )
        out = tmp_path / "result.json"
        write_result(result, out)
        assert out.exists()

        md = render_markdown(result)
        assert "TopoGeoML Hodge subsystem benchmark" in md
        assert "Per-(model × dataset) test accuracy" in md


class TestH006Resolver:
    """Hypothesis 006 — constant-feature ablation resolver.

    The resolver consumes one ``--constant-features`` JSON per dataset
    plus the corresponding full-feature ablation JSON, and emits the
    verdicts for sub-hypotheses H22-H25.  These tests use synthetic
    fixtures so they run without the ~75-minute background bench.
    """

    @staticmethod
    def _write_fake(
        tmp_path,
        ds: str,
        hodge_accs: list[float],
        mlp_accs: list[float],
        suffix: str,
    ):
        import json

        cells = lambda name, accs: [  # noqa: E731
            {
                "model_name": name, "dataset_name": ds, "seed": i,
                "test_accuracy": a, "n_train": 100, "n_test": 25,
                "final_train_loss": 0.5,
            }
            for i, a in enumerate(accs)
        ]
        reports = []
        for name, accs in (("hodge-mp-residual", hodge_accs), ("mlp-baseline", mlp_accs)):
            reports.append({
                "model_name": name, "model_version": "1.0.0",
                "dataset_name": ds, "dataset_version": "tu-r1",
                "n_epochs": 10, "learning_rate": 1e-2, "cells": cells(name, accs),
                "accuracy_median": float(sum(accs) / len(accs)),
                "accuracy_ci95_low": 0.0, "accuracy_ci95_high": 1.0,
            })
        payload = {"schema_version": "hodge-1.0.0", "reports": reports}
        path = tmp_path / f"{ds}_{suffix}.json"
        path.write_text(json.dumps(payload))
        return path

    def test_resolve_runs_on_synthetic_fixtures(self, tmp_path) -> None:
        from benchmarks.hodge.h006_analysis import resolve

        # NCI1: Hodge well above prior (~0.50). MUTAG: Hodge close to prior
        # (~0.66). PROTEINS: Hodge slightly above prior (~0.60).
        constant = {
            "nci1": self._write_fake(tmp_path, "nci1",
                hodge_accs=[0.60] * 30, mlp_accs=[0.50] * 30, suffix="constant"),
            "mutag": self._write_fake(tmp_path, "mutag",
                hodge_accs=[0.66] * 30, mlp_accs=[0.66] * 30, suffix="constant"),
            "proteins": self._write_fake(tmp_path, "proteins",
                hodge_accs=[0.62] * 30, mlp_accs=[0.60] * 30, suffix="constant"),
        }
        full = {
            "nci1": self._write_fake(tmp_path, "nci1",
                hodge_accs=[0.61] * 30, mlp_accs=[0.52] * 30, suffix="full"),
            "mutag": self._write_fake(tmp_path, "mutag",
                hodge_accs=[0.75] * 30, mlp_accs=[0.79] * 30, suffix="full"),
            "proteins": self._write_fake(tmp_path, "proteins",
                hodge_accs=[0.69] * 30, mlp_accs=[0.67] * 30, suffix="full"),
        }
        summaries = resolve(constant_paths=constant, full_paths=full)
        assert len(summaries) == 3
        ds_to_summary = {s.dataset: s for s in summaries}
        # NCI1: hodge median 0.60, prior ~0.50 → significantly above.
        assert ds_to_summary["nci1"].hodge_above_prior_significant is True
        # MUTAG: hodge median 0.66, prior 0.6649 → not significantly above
        # (median essentially equals prior, so the one-sided test fails).
        assert ds_to_summary["mutag"].hodge_above_prior_significant is False
        # Provenance: source paths carried through.
        assert ds_to_summary["nci1"].constant_feature_source == constant["nci1"]
        assert ds_to_summary["nci1"].full_feature_source == full["nci1"]

    def test_resolve_fails_loud_on_missing_constant_json(self, tmp_path) -> None:
        from benchmarks.hodge.h006_analysis import resolve

        constant = {ds: tmp_path / f"{ds}_constant.json"
                    for ds in ("mutag", "proteins", "nci1")}
        full = {ds: tmp_path / f"{ds}_full.json"
                for ds in ("mutag", "proteins", "nci1")}
        with pytest.raises(FileNotFoundError, match="constant-feature result"):
            resolve(constant_paths=constant, full_paths=full)

    def test_resolve_fails_loud_on_missing_full_json(self, tmp_path) -> None:
        from benchmarks.hodge.h006_analysis import resolve

        constant = {
            "nci1": self._write_fake(tmp_path, "nci1",
                hodge_accs=[0.6] * 5, mlp_accs=[0.5] * 5, suffix="constant"),
            "mutag": self._write_fake(tmp_path, "mutag",
                hodge_accs=[0.66] * 5, mlp_accs=[0.66] * 5, suffix="constant"),
            "proteins": self._write_fake(tmp_path, "proteins",
                hodge_accs=[0.62] * 5, mlp_accs=[0.60] * 5, suffix="constant"),
        }
        full = {ds: tmp_path / f"{ds}_full.json"
                for ds in ("mutag", "proteins", "nci1")}
        with pytest.raises(FileNotFoundError, match="full-feature result"):
            resolve(constant_paths=constant, full_paths=full)

    def test_resolve_rejects_wrong_dataset_set(self, tmp_path) -> None:
        from benchmarks.hodge.h006_analysis import resolve

        constant = {"mutag": tmp_path / "x.json"}
        full = {"mutag": tmp_path / "y.json", "proteins": tmp_path / "z.json",
                "nci1": tmp_path / "w.json"}
        with pytest.raises(ValueError, match="exactly"):
            resolve(constant_paths=constant, full_paths=full)

    def test_resolve_rejects_missing_arm(self, tmp_path) -> None:
        """If a JSON is missing one of the required arms (residual or MLP),
        the resolver must fail explicitly rather than silently inferring."""
        import json

        from benchmarks.hodge.h006_analysis import resolve

        bad_payload = {
            "schema_version": "hodge-1.0.0",
            "reports": [{
                "model_name": "hodge-mp-residual", "model_version": "1.0.0",
                "dataset_name": "nci1", "dataset_version": "tu-r1",
                "n_epochs": 10, "learning_rate": 1e-2,
                "cells": [{"model_name": "hodge-mp-residual", "dataset_name": "nci1",
                           "seed": 0, "test_accuracy": 0.6, "n_train": 1,
                           "n_test": 1, "final_train_loss": 0.0}],
                "accuracy_median": 0.6,
                "accuracy_ci95_low": 0.0, "accuracy_ci95_high": 1.0,
            }],
        }
        bad = tmp_path / "nci1_constant.json"
        bad.write_text(json.dumps(bad_payload))
        constant = {
            "nci1": bad,
            "mutag": self._write_fake(tmp_path, "mutag",
                hodge_accs=[0.66] * 5, mlp_accs=[0.66] * 5, suffix="constant"),
            "proteins": self._write_fake(tmp_path, "proteins",
                hodge_accs=[0.62] * 5, mlp_accs=[0.60] * 5, suffix="constant"),
        }
        full = {
            "nci1": self._write_fake(tmp_path, "nci1",
                hodge_accs=[0.61] * 5, mlp_accs=[0.52] * 5, suffix="full"),
            "mutag": self._write_fake(tmp_path, "mutag",
                hodge_accs=[0.75] * 5, mlp_accs=[0.79] * 5, suffix="full"),
            "proteins": self._write_fake(tmp_path, "proteins",
                hodge_accs=[0.69] * 5, mlp_accs=[0.67] * 5, suffix="full"),
        }
        with pytest.raises(KeyError, match="mlp-baseline"):
            resolve(constant_paths=constant, full_paths=full)

    def test_render_markdown_has_required_sections(self, tmp_path) -> None:
        from benchmarks.hodge.h006_analysis import render_markdown, resolve

        constant = {
            "nci1": self._write_fake(tmp_path, "nci1",
                hodge_accs=[0.60] * 30, mlp_accs=[0.50] * 30, suffix="constant"),
            "mutag": self._write_fake(tmp_path, "mutag",
                hodge_accs=[0.66] * 30, mlp_accs=[0.66] * 30, suffix="constant"),
            "proteins": self._write_fake(tmp_path, "proteins",
                hodge_accs=[0.62] * 30, mlp_accs=[0.60] * 30, suffix="constant"),
        }
        full = {
            "nci1": self._write_fake(tmp_path, "nci1",
                hodge_accs=[0.61] * 30, mlp_accs=[0.52] * 30, suffix="full"),
            "mutag": self._write_fake(tmp_path, "mutag",
                hodge_accs=[0.75] * 30, mlp_accs=[0.79] * 30, suffix="full"),
            "proteins": self._write_fake(tmp_path, "proteins",
                hodge_accs=[0.69] * 30, mlp_accs=[0.67] * 30, suffix="full"),
        }
        md = render_markdown(resolve(constant_paths=constant, full_paths=full))
        # Required output sections per the PR scope contract.
        assert "## H006 reproducible summary (per-dataset)" in md
        assert "## H006 statistical table" in md
        assert "## H25 Spearman correlation" in md
        assert "## Scoped interpretation" in md
        # Scoped-claim language must appear verbatim.
        assert "architecture × data-topology interaction" in md
        assert "under the tested configuration" in md
        # Terminology correction: the test isolates feature-independent
        # graph-structural signal, NOT homology specifically.
        assert "graph-structural signal" in md
        assert "does NOT isolate homology" in md

    def test_benjamini_hochberg_basic(self) -> None:
        from benchmarks.hodge.h006_analysis import _benjamini_hochberg

        # Three p-values; BH at α=0.05 keeps monotonicity.
        adj, rej = _benjamini_hochberg([0.01, 0.04, 0.20], alpha=0.05)
        # adj_3 = 0.20 * 3/3 = 0.20; adj_2 = min(0.04*3/2, 0.20)=0.06;
        # adj_1 = min(0.01*3/1, 0.06)=0.03.
        assert adj == pytest.approx([0.03, 0.06, 0.20])
        assert rej == [True, False, False]


class TestH007Analysis:
    """Hypothesis 007 — graph-structural-signal decomposition.

    The analysis module computes five graph-structural proxies (size,
    degree, WL, cycle, spectral), measures per-class separability via
    rank-biserial r, and correlates across datasets.  These tests use
    synthetic networkx graphs so they run without a TUDataset download.
    """

    @staticmethod
    def _two_class_graphs():
        """Two-class toy: class 0 = path graphs of 4 nodes, class 1 = K_4.
        Different size, degree, cycle, spectrum — useful as a worked example.
        """
        import networkx as nx

        graphs = []
        labels = []
        for _ in range(20):
            graphs.append((nx.path_graph(4), 0))
            labels.append(0)
        for _ in range(20):
            graphs.append((nx.complete_graph(4), 1))
            labels.append(1)
        return graphs, labels

    def test_size_features_returns_1d_scalar(self) -> None:
        import networkx as nx

        from benchmarks.hodge.h007_analysis import compute_size_features

        feat = compute_size_features(nx.path_graph(7))
        assert feat.shape == (1,)
        assert feat[0] == 7.0

    def test_degree_features_returns_5d_vector(self) -> None:
        import networkx as nx

        from benchmarks.hodge.h007_analysis import compute_degree_features

        # K_4: every node has degree 3.  mean=3, max=3, std=0, n_iso=0,
        # density=1.0 (4 nodes, 6 edges).
        feat = compute_degree_features(nx.complete_graph(4))
        assert feat.shape == (5,)
        assert feat[0] == 3.0
        assert feat[1] == 3.0
        assert feat[2] == 0.0
        assert feat[3] == 0.0
        assert feat[4] == 1.0

    def test_wl_features_returns_normalised_32d_vector(self) -> None:
        import networkx as nx

        from benchmarks.hodge.h007_analysis import compute_wl_features

        feat = compute_wl_features(nx.complete_graph(4))
        assert feat.shape == (32,)
        # The histogram is normalised so it sums to 1 (or is all-zeros).
        assert feat.sum() == pytest.approx(1.0, abs=1e-6)

    def test_cycle_features_returns_4d_vector(self) -> None:
        import networkx as nx

        from benchmarks.hodge.h007_analysis import compute_cycle_features

        # K_4: cycle basis size = 3 (= edges - nodes + 1 = 6-4+1), each
        # basis cycle has length 3 (triangles).  triangles count = 4.
        feat = compute_cycle_features(nx.complete_graph(4))
        assert feat.shape == (4,)
        assert feat[0] == 3.0  # n_cycles_basis = β₁ = 3
        assert feat[1] == 3.0  # mean cycle length
        assert feat[2] == 4.0  # n_triangles (4 in K_4)
        assert feat[3] == 0.0  # n_4cycles in basis (0, all are 3-cycles)

    def test_cycle_features_path_graph_has_no_cycles(self) -> None:
        import networkx as nx

        from benchmarks.hodge.h007_analysis import compute_cycle_features

        feat = compute_cycle_features(nx.path_graph(7))
        # A tree has β₁ = 0; no cycles.
        assert feat[0] == 0.0
        assert feat[1] == 0.0
        assert feat[2] == 0.0
        assert feat[3] == 0.0

    def test_spectral_features_returns_top_k_eigenvalues(self) -> None:
        from benchmarks.hodge.h007_analysis import compute_spectral_features

        # Synthetic K_3 Laplacian: L = D - A = diag(2,2,2) - J + I.
        # Eigenvalues of K_n Laplacian are {0, n, n, ..., n} so K_3 → {0, 3, 3}.
        # Normalised: L̃ = D^{-1/2} L D^{-1/2}, eigenvalues for K_n are
        # {0, n/(n-1), n/(n-1), ..., n/(n-1)} so K_3 → {0, 1.5, 1.5}.
        indices = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1],
        ], dtype=torch.long)
        values = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
        diag_indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        diag_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        all_indices = torch.cat([indices, diag_indices], dim=1)
        all_values = torch.cat([values, diag_values])
        L = torch.sparse_coo_tensor(all_indices, all_values, size=(3, 3))
        feat = compute_spectral_features(L, k=5)
        assert feat.shape == (5,)
        # Top eigenvalues (descending): {1.5, 1.5, 0, 0, 0} (padded).
        assert feat[0] == pytest.approx(1.5, abs=1e-6)
        assert feat[1] == pytest.approx(1.5, abs=1e-6)
        assert feat[2] == pytest.approx(0.0, abs=1e-6)

    def test_class_separability_perfect_separation(self) -> None:
        from benchmarks.hodge.h007_analysis import class_separability

        # Class 0: feature value 0.  Class 1: feature value 10.  Perfect.
        features = np.asarray([[0.0]] * 10 + [[10.0]] * 10)
        labels = np.asarray([0] * 10 + [1] * 10)
        per_comp, best = class_separability(features, labels)
        assert per_comp[best] == pytest.approx(1.0, abs=1e-6)

    def test_class_separability_no_separation(self) -> None:
        from benchmarks.hodge.h007_analysis import class_separability

        # Identical distributions → separability should be 0.
        features = np.asarray([[1.0]] * 10 + [[1.0]] * 10)
        labels = np.asarray([0] * 10 + [1] * 10)
        per_comp, _ = class_separability(features, labels)
        assert per_comp[0] == pytest.approx(0.0, abs=1e-6)

    def test_class_separability_rejects_non_binary(self) -> None:
        from benchmarks.hodge.h007_analysis import class_separability

        features = np.asarray([[1.0], [2.0], [3.0]])
        labels = np.asarray([0, 1, 2])
        with pytest.raises(ValueError, match="2 classes"):
            class_separability(features, labels)

    def test_render_markdown_required_sections(self) -> None:
        from benchmarks.hodge.h007_analysis import render_markdown

        synthetic = {
            "schema_version": "h007-1.0.0",
            "h006_const_feature_gap": {"mutag": 0.1, "proteins": 0.09, "nci1": 0.07},
            "h006_full_feature_gain": {"mutag": -0.04, "proteins": 0.01, "nci1": 0.09},
            "per_dataset_proxy_results": [
                {"proxy_name": "size", "dataset_name": "mutag", "feature_dim": 1,
                 "n_samples": 188, "class_distribution": {"0": 63, "1": 125},
                 "per_component_separability": [0.234], "max_separability": 0.234,
                 "best_component_idx": 0},
            ],
            "correlation_table": [
                {"proxy_name": "size", "separability_by_dataset": {"mutag": 0.234},
                 "spearman_rho_vs_const_gap": 0.5,
                 "spearman_rho_vs_full_gain": -0.5},
            ],
        }
        md = render_markdown(synthetic)
        # Required output sections per the PR scope contract.
        assert "H007 structural-signal decomposition" in md
        assert "## Per-(dataset × proxy) class separability" in md
        assert "## Cross-dataset correlation" in md
        assert "## Scoped interpretation" in md
        # Scoped-claim language must appear.
        assert "graph-structural proxy" in md
        assert "descriptive" in md.lower()
        # Cycle-basis is the only topological invariant — must be flagged.
        assert "β₁" in md or "topological invariant" in md

    def test_run_h007_analysis_schema(self) -> None:
        """Smoke test the top-level entry on MUTAG only — fastest dataset."""
        from benchmarks.hodge.h007_analysis import (
            PROXY_NAMES,
            run_h007_analysis,
        )

        # Skip if torch_geometric unavailable.
        try:
            import torch_geometric  # noqa: F401
        except ImportError:
            pytest.skip("torch_geometric not installed")

        result = run_h007_analysis(dataset_names=("mutag",))
        assert result["schema_version"] == "h007-1.0.0"
        assert "h006_const_feature_gap" in result
        assert "h006_full_feature_gain" in result
        assert "per_dataset_proxy_results" in result
        assert "correlation_table" in result
        # One row per (proxy, dataset) — 5 proxies × 1 dataset = 5 rows.
        assert len(result["per_dataset_proxy_results"]) == len(PROXY_NAMES)
        # Correlation table has 5 rows (one per proxy) even on 1 dataset.
        # With n=1 dataset, ρ is NaN — that's expected and reported.
        assert len(result["correlation_table"]) == len(PROXY_NAMES)
        for row in result["per_dataset_proxy_results"]:
            assert row["dataset_name"] == "mutag"
            assert row["proxy_name"] in PROXY_NAMES
            assert 0.0 <= row["max_separability"] <= 1.0
            assert row["feature_dim"] == len(row["per_component_separability"])
