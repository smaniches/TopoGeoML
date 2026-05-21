"""
Tests for the gudhi-python and Hofer-2017-reference backends, plus the
runner's handling of non-differentiable backends.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _has_gudhi() -> bool:
    try:
        import gudhi  # noqa: F401
    except ImportError:
        return False
    return True


def _circle(n: int, seed: int, noise: float = 0.0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    if noise > 0:
        pts += noise * rng.standard_normal(pts.shape)
    return torch.from_numpy(np.ascontiguousarray(pts, dtype=np.float64))


# ---------------------------------------------------------------------------
# gudhi-python (non-differentiable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestGudhiBackend:
    def test_registers_and_is_marked_non_differentiable(self) -> None:
        from benchmarks.backends import get_backend

        cls = get_backend("gudhi-python")
        assert cls.differentiable is False
        assert cls.available() is True
        assert cls.version != ""

    def test_rejects_float32(self) -> None:
        from benchmarks.backends import get_backend

        cls = get_backend("gudhi-python")
        X = torch.zeros((10, 2), dtype=torch.float32)
        with pytest.raises(TypeError, match="must be float64"):
            cls.compute_diagram(X, max_dim=1)

    def test_h1_loop_on_circle(self) -> None:
        from benchmarks.backends import get_backend

        cls = get_backend("gudhi-python")
        X = _circle(n=20, seed=0)
        dgms = cls.compute_diagram(X, max_dim=1)
        # Exactly one H_1 finite bar for a clean circle.
        h1 = dgms[1]
        finite = h1[torch.isfinite(h1).all(dim=1)]
        assert finite.shape[0] == 1
        # The loop's persistence should be on the order of (chord, diameter)
        # ~ (2*sin(pi/20), 2). Just sanity-check both finite.
        b, d = float(finite[0, 0]), float(finite[0, 1])
        assert 0.0 < b < 1.0
        assert b < d <= 2.5

    def test_loss_longest_h1_raises_not_implemented(self) -> None:
        from benchmarks.backends import get_backend

        cls = get_backend("gudhi-python")
        X = _circle(n=10, seed=0)
        with pytest.raises(NotImplementedError, match="non-differentiable"):
            cls.loss_longest_h1(X)

    def test_agrees_with_ripser_on_h1(self) -> None:
        """gudhi and ripser are independent backbones; agreement on H_1 is
        a strong correctness signal."""
        from ripser import ripser

        from benchmarks.backends import get_backend

        X = _circle(n=20, seed=0)
        # ripser reference
        ref = ripser(X.numpy(), maxdim=1)["dgms"]
        ref_h1 = ref[1][np.isfinite(ref[1]).all(axis=1)] if ref[1].size else np.empty((0, 2))

        # gudhi
        cls = get_backend("gudhi-python")
        dgms = cls.compute_diagram(X, max_dim=1)
        gudhi_h1 = dgms[1].numpy()
        gudhi_h1 = gudhi_h1[np.isfinite(gudhi_h1).all(axis=1)]

        assert ref_h1.shape == gudhi_h1.shape == (1, 2)
        assert np.allclose(ref_h1[np.lexsort((ref_h1[:, 1], ref_h1[:, 0]))], gudhi_h1[np.lexsort((gudhi_h1[:, 1], gudhi_h1[:, 0]))], atol=1e-6)


# ---------------------------------------------------------------------------
# Hofer-2017-reference (differentiable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestHofer2017Backend:
    def test_registers_and_is_differentiable(self) -> None:
        from benchmarks.backends import get_backend

        cls = get_backend("hofer-2017-reference")
        assert cls.differentiable is True
        assert cls.available() is True

    def test_rejects_float32(self) -> None:
        from benchmarks.backends import get_backend

        cls = get_backend("hofer-2017-reference")
        X = torch.zeros((10, 2), dtype=torch.float32)
        with pytest.raises(TypeError, match="must be float64"):
            cls.compute_diagram(X, max_dim=1)

    def test_h1_diagram_matches_ripser(self) -> None:
        from ripser import ripser

        from benchmarks.backends import get_backend

        X = _circle(n=15, seed=0)
        ref_h1 = ripser(X.numpy(), maxdim=1)["dgms"][1]
        ref_h1 = ref_h1[np.isfinite(ref_h1).all(axis=1)]

        cls = get_backend("hofer-2017-reference")
        dgms = cls.compute_diagram(X, max_dim=1)
        hofer_h1 = dgms[1].detach().numpy()
        hofer_h1 = hofer_h1[np.isfinite(hofer_h1).all(axis=1)]

        # Both libraries should agree to f64 round-off.
        assert ref_h1.shape == hofer_h1.shape == (1, 2)
        assert np.allclose(ref_h1[np.lexsort((ref_h1[:, 1], ref_h1[:, 0]))], hofer_h1[np.lexsort((hofer_h1[:, 1], hofer_h1[:, 0]))], atol=1e-6)

    def test_loss_is_differentiable(self) -> None:
        from benchmarks.backends import get_backend

        cls = get_backend("hofer-2017-reference")
        X = _circle(n=12, seed=0).requires_grad_(True)
        loss = cls.loss_longest_h1(X)
        assert loss.requires_grad
        loss.backward()
        assert X.grad is not None
        assert torch.all(torch.isfinite(X.grad))
        # The gradient should be nontrivial: the loss DOES depend on X.
        assert float(torch.linalg.norm(X.grad).item()) > 1e-9

    def test_loss_zero_on_empty_h1(self) -> None:
        """A 3-point colinear cloud has no H_1 — loss must be a 0-dim tensor
        that depends on X (so autograd never breaks)."""
        from benchmarks.backends import get_backend

        cls = get_backend("hofer-2017-reference")
        X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64,
                         requires_grad=True)
        loss = cls.loss_longest_h1(X)
        assert loss.shape == ()
        assert float(loss.item()) == 0.0

    def test_hofer_handles_max_dim_zero(self) -> None:
        """``max_dim = 0`` requests only H_0 — covers the
        no-H_k-finite-bars branch in ``_compute_rips_persistence``."""
        from benchmarks.backends import get_backend

        cls = get_backend("hofer-2017-reference")
        X = _circle(n=5, seed=0)
        dgms = cls.compute_diagram(X, max_dim=0)
        # Only one tensor for H_0; no H_k contribution to pack.
        assert len(dgms) == 1
        assert dgms[0].dtype == torch.float64

    def test_hofer_include_essential_false_branch(self) -> None:
        """Exercise the ``include_essential=False`` branch of the internal
        function via the underlying helper."""
        from benchmarks.backends.hofer_2017_reference import _compute_rips_persistence

        X = _circle(n=8, seed=0)
        packed_with = _compute_rips_persistence(X, max_dim=1, include_essential=True)
        packed_without = _compute_rips_persistence(X, max_dim=1, include_essential=False)
        # Excluding the essential class produces fewer rows.
        assert packed_without.shape[0] < packed_with.shape[0]

    def test_hofer_loss_handles_single_point(self) -> None:
        """A single point has no H_0 finite bars and no H_1 — exercises the
        ``finite.numel() == 0`` branch of ``loss_longest_h1``."""
        from benchmarks.backends import get_backend

        cls = get_backend("hofer-2017-reference")
        # Two coincident points: only H_0 essential, no H_1.
        X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        loss = cls.loss_longest_h1(X)
        assert loss.shape == ()
        assert float(loss.item()) == 0.0


@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestGudhiEmptyDimBars:
    def test_gudhi_returns_empty_tensor_for_dim_with_no_bars(self) -> None:
        """When a homology dimension has zero bars, the gudhi backend emits
        an empty (0, 2) tensor for that slot — covers the empty-bucket branch."""
        from benchmarks.backends import get_backend

        cls = get_backend("gudhi-python")
        # Two-point cloud has H_0 (1 essential bar) and no H_1.
        X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        dgms = cls.compute_diagram(X, max_dim=1)
        assert dgms[1].shape == (0, 2)


# ---------------------------------------------------------------------------
# Cross-backend agreement: TopoGeoML diff-PH vs Hofer 2017 reference
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestCrossBackendAgreement:
    def test_h1_diagram_agreement_between_topogeoml_and_hofer(self) -> None:
        from benchmarks.backends import get_backend

        X = _circle(n=15, seed=0)
        topo = get_backend("topogeoml-diff-ph").compute_diagram(X, max_dim=1)
        hofer = get_backend("hofer-2017-reference").compute_diagram(X, max_dim=1)

        topo_h1 = topo[1].detach().numpy()
        hofer_h1 = hofer[1].detach().numpy()
        topo_h1 = topo_h1[np.isfinite(topo_h1).all(axis=1)]
        hofer_h1 = hofer_h1[np.isfinite(hofer_h1).all(axis=1)]

        # Two independent algorithmic paths to the same diagram.
        assert topo_h1.shape == hofer_h1.shape == (1, 2)
        assert np.allclose(topo_h1[np.lexsort((topo_h1[:, 1], topo_h1[:, 0]))], hofer_h1[np.lexsort((hofer_h1[:, 1], hofer_h1[:, 0]))], atol=1e-6)

    def test_loss_value_agreement(self) -> None:
        from benchmarks.backends import get_backend

        X = _circle(n=12, seed=0)
        loss_a = float(get_backend("topogeoml-diff-ph").loss_longest_h1(X).item())
        loss_b = float(get_backend("hofer-2017-reference").loss_longest_h1(X).item())
        assert loss_a == pytest.approx(loss_b, abs=1e-9)


# ---------------------------------------------------------------------------
# Runner integration: non-differentiable backend handling
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gudhi(), reason="gudhi not installed")
class TestRunnerSkipsNonDifferentiable:
    def test_gudhi_skipped_on_stability_axis(self) -> None:
        from benchmarks.runner import run

        result = run(
            backend_names=["gudhi-python"],
            dataset_names=["mnist_mock_digit_0"],
            axis_names=["stability"],
        )
        assert len(result.cells) == 1
        cell = result.cells[0]
        assert not cell.success
        assert cell.error_kind == "SkippedNonDifferentiable"

    def test_gudhi_runs_correctness_axis(self) -> None:
        from benchmarks.runner import run

        result = run(
            backend_names=["gudhi-python"],
            dataset_names=["mnist_mock_digit_1"],
            axis_names=["correctness"],
        )
        assert len(result.cells) == 1
        cell = result.cells[0]
        assert cell.success
        assert cell.payload is not None
        assert "overall_pass" in cell.payload

    def test_differentiable_backend_runs_all_axes(self) -> None:
        from benchmarks.runner import run

        result = run(
            backend_names=["hofer-2017-reference"],
            dataset_names=["mnist_mock_digit_1"],
            axis_names=["correctness", "optimization"],
        )
        assert len(result.cells) == 2
        for cell in result.cells:
            assert cell.success, f"axis {cell.axis_name} failed: {cell.error_message}"
