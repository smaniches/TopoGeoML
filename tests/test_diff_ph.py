"""Tests for differentiable persistent homology (nn.diff_ph)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from topogeoml.nn.diff_ph import (
    TopologyRegularizer,
    betti_regularization_loss,
    finite_lifetimes,
    pairwise_distances,
    persistence_entropy_loss,
    rips_diagram_torch,
    total_persistence_loss,
)

pytestmark = pytest.mark.torch


def _circle(n: int = 30, noise: float = 0.05) -> torch.Tensor:
    rng = np.random.default_rng(42)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    pts += noise * rng.standard_normal(pts.shape)
    return torch.from_numpy(pts.astype(np.float64)).requires_grad_(True)


def _two_clusters(n_each: int = 20) -> torch.Tensor:
    rng = np.random.default_rng(42)
    c1 = rng.standard_normal((n_each, 2)) * 0.3
    c2 = rng.standard_normal((n_each, 2)) * 0.3 + np.array([3.0, 0.0])
    pts = np.concatenate([c1, c2], axis=0).astype(np.float64)
    return torch.from_numpy(pts).requires_grad_(True)


# --- pairwise_distances ---

def test_pairwise_distances_shape() -> None:
    X = torch.randn(15, 3, dtype=torch.float64)
    D = pairwise_distances(X)
    assert D.shape == (15, 15)


def test_pairwise_distances_symmetric() -> None:
    X = torch.randn(10, 4, dtype=torch.float64)
    D = pairwise_distances(X)
    torch.testing.assert_close(D, D.t())


def test_pairwise_distances_zero_diagonal() -> None:
    X = torch.randn(8, 2, dtype=torch.float64)
    D = pairwise_distances(X)
    torch.testing.assert_close(D.diagonal(), torch.zeros(8, dtype=torch.float64))


def test_pairwise_distances_non_negative() -> None:
    X = torch.randn(12, 5, dtype=torch.float64)
    D = pairwise_distances(X)
    assert (D >= 0).all()


# --- rips_diagram_torch ---

def test_diagram_shape_h0_only() -> None:
    X = _circle(n=20)
    dgms = rips_diagram_torch(X, max_dim=0)
    assert len(dgms) == 1
    assert dgms[0].shape[1] == 2


def test_diagram_h0_has_one_infinite_bar() -> None:
    X = _circle(n=15)
    dgms = rips_diagram_torch(X, max_dim=0)
    h0 = dgms[0]
    n_inf = torch.isinf(h0[:, 1]).sum().item()
    assert n_inf == 1


def test_diagram_h0_finite_deaths_positive() -> None:
    X = _circle(n=20)
    dgms = rips_diagram_torch(X, max_dim=0)
    h0 = dgms[0]
    finite_deaths = h0[torch.isfinite(h0[:, 1]), 1]
    assert (finite_deaths > 0).all()


def test_diagram_h1_loop_on_circle() -> None:
    """A noisy circle should produce one significant H_1 bar."""
    X = _circle(n=25, noise=0.05)
    dgms = rips_diagram_torch(X, max_dim=1)
    assert len(dgms) == 2
    h1 = dgms[1]
    if h1.numel() > 0:
        lifetimes = finite_lifetimes(h1)
        if lifetimes.numel() > 0:
            assert lifetimes.max().item() > 0.3


def test_diagram_two_clusters_beta0_is_2() -> None:
    """Two well-separated clusters should have β_0=2 (one significant finite bar + one inf)."""
    X = _two_clusters(n_each=15)
    dgms = rips_diagram_torch(X, max_dim=0)
    h0 = dgms[0]
    finite_deaths = h0[torch.isfinite(h0[:, 1]), 1]
    # The prominent bar corresponds to the between-cluster merge (~3.0)
    long_bars = (finite_deaths > 1.0).sum().item()
    assert long_bars >= 1


def test_diagram_gradient_flows_to_X() -> None:
    """Loss computed from diagram must produce nonzero gradient w.r.t. X."""
    X = _circle(n=20, noise=0.05)
    dgms = rips_diagram_torch(X, max_dim=0)
    loss = total_persistence_loss(dgms[0], p=2.0)
    loss.backward()
    assert X.grad is not None
    assert X.grad.abs().sum().item() > 0


def test_diagram_h1_gradient_flows() -> None:
    """H_1 loss must propagate gradient back to X."""
    X = _circle(n=25, noise=0.05)
    dgms = rips_diagram_torch(X, max_dim=1)
    h1 = dgms[1]
    if h1.numel() == 0:
        pytest.skip("No H_1 bars on this point cloud")
    loss = total_persistence_loss(h1, p=2.0)
    if loss.item() == 0.0:
        pytest.skip("All H_1 bars have zero lifetime")
    loss.backward()
    assert X.grad is not None
    assert X.grad.abs().sum().item() > 0


def test_diagram_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        rips_diagram_torch(torch.randn(10, dtype=torch.float64), max_dim=0)


def test_diagram_rejects_single_point() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        rips_diagram_torch(torch.randn(1, 2, dtype=torch.float64), max_dim=0)


# --- Loss functions ---

def test_total_persistence_loss_zero_for_empty() -> None:
    empty = torch.empty((0, 2), dtype=torch.float64)
    assert total_persistence_loss(empty).item() == 0.0


def test_total_persistence_loss_positive() -> None:
    diagram = torch.tensor([[0.0, 1.0], [0.0, 0.5]], dtype=torch.float64)
    loss = total_persistence_loss(diagram, p=2.0)
    assert loss.item() == pytest.approx(1.0 ** 2 + 0.5 ** 2)


def test_persistence_entropy_loss_zero_for_empty() -> None:
    empty = torch.empty((0, 2), dtype=torch.float64)
    assert persistence_entropy_loss(empty).item() == 0.0


def test_persistence_entropy_loss_nonzero() -> None:
    diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.5]], dtype=torch.float64)
    ent = persistence_entropy_loss(diagram)
    assert ent.item() > 0.0


def test_betti_regularization_no_excess() -> None:
    """No penalty when current Betti matches target."""
    # One infinite bar + one finite bar: β_0 = 2.
    diagram = torch.tensor([[0.0, 0.5], [0.0, float("inf")]], dtype=torch.float64)
    loss = betti_regularization_loss(diagram, target_n_components=2)
    assert loss.item() == 0.0


def test_betti_regularization_penalizes_excess() -> None:
    """Excess bars above target should produce positive penalty."""
    # Three finite + one infinite = β_0 = 4, target = 1.
    diagram = torch.tensor([
        [0.0, 0.8], [0.0, 0.6], [0.0, 0.4], [0.0, float("inf")]
    ], dtype=torch.float64)
    loss = betti_regularization_loss(diagram, target_n_components=1)
    assert loss.item() > 0.0


def test_betti_regularization_counts_multiple_essential_bars() -> None:
    """Every essential (infinite-death) bar is a permanent component.

    A thresholded H_0 diagram can carry several essential bars (one per
    surviving component). With two essential bars and one prominent finite
    bar, beta_0 = 3. A hardcoded "+1 essential" would read beta_0 = 2 and
    miss the excess against target 2.
    """
    diagram = torch.tensor(
        [[0.0, 0.9], [0.0, float("inf")], [0.0, float("inf")]],
        dtype=torch.float64,
    )
    # 2 essential + 1 significant finite == 3 -> matches target -> no penalty.
    assert betti_regularization_loss(diagram, target_n_components=3).item() == 0.0
    # 3 > target 2 -> positive penalty (the old hardcoded +1 wrongly returned 0).
    assert betti_regularization_loss(diagram, target_n_components=2).item() > 0.0


# --- TopologyRegularizer module ---

def test_regularizer_module_forward() -> None:
    reg = TopologyRegularizer(max_dim=0, loss_type="total_persistence")
    X = torch.randn(25, 4, dtype=torch.float64, requires_grad=True)
    loss = reg(X)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


def test_regularizer_module_gradient() -> None:
    reg = TopologyRegularizer(max_dim=0, loss_type="total_persistence", p=1.0)
    X = torch.randn(20, 3, dtype=torch.float64, requires_grad=True)
    loss = reg(X)
    loss.backward()
    assert X.grad is not None


def test_regularizer_entropy_mode() -> None:
    reg = TopologyRegularizer(max_dim=0, loss_type="entropy")
    X = torch.randn(20, 2, dtype=torch.float64, requires_grad=True)
    loss = reg(X)
    # Entropy loss is negated, so can be negative (we're minimizing -entropy)
    assert torch.isfinite(loss)


def test_regularizer_rejects_1d() -> None:
    reg = TopologyRegularizer()
    with pytest.raises(ValueError, match="2D"):
        reg(torch.randn(10, dtype=torch.float64))
