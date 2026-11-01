"""Zero topology penalties must keep the training autograd path intact.

These checks distinguish a legitimate zero gradient from a detached constant.
Essential bars have infinite deaths; no `inf * 0` workaround is permitted.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch")

from topogeoml.nn.diff_ph import (
    TopologyRegularizer,
    betti_regularization_loss,
    finite_lifetimes,
    persistence_entropy_loss,
    rips_diagram_torch,
    total_persistence_loss,
)

pytestmark = pytest.mark.torch


def _zero_losses() -> tuple[Callable[[torch.Tensor], torch.Tensor], ...]:
    return (
        total_persistence_loss,
        persistence_entropy_loss,
        lambda d: betti_regularization_loss(d, target_n_components=1),
    )


@pytest.mark.parametrize("loss_index", [0, 1, 2])
def test_empty_diagram_zero_backpropagates(loss_index: int) -> None:
    points = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    diagram = points[:0, :]
    loss = _zero_losses()[loss_index](diagram)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert points.grad is not None
    torch.testing.assert_close(points.grad, torch.zeros_like(points))


@pytest.mark.parametrize("loss_index", [0, 1, 2])
def test_all_essential_bars_zero_without_nan_gradients(loss_index: int) -> None:
    diagram = torch.tensor(
        [[0.0, float("inf")], [0.0, float("inf")]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert finite_lifetimes(diagram).requires_grad
    loss = _zero_losses()[loss_index](diagram)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert diagram.grad is not None
    torch.testing.assert_close(diagram.grad, torch.zeros_like(diagram))


def test_total_persistence_threshold_zero_retains_backward() -> None:
    diagram = torch.tensor(
        [[0.0, 0.5], [0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = total_persistence_loss(diagram, threshold=2.0)
    assert loss.item() == 0.0
    loss.backward()
    assert diagram.grad is not None
    torch.testing.assert_close(diagram.grad, torch.zeros_like(diagram))


def test_betti_target_met_returns_differentiable_zero() -> None:
    diagram = torch.tensor(
        [[0.0, 0.5], [0.0, float("inf")]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = betti_regularization_loss(diagram, target_n_components=5)
    assert loss.item() == 0.0
    loss.backward()
    assert diagram.grad is not None
    torch.testing.assert_close(diagram.grad, torch.zeros_like(diagram))


def test_regularizer_no_excess_betti_supports_training_backward() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    regularizer = TopologyRegularizer(
        max_dim=0, loss_type="betti_regularization", target_betti={0: 10}
    )
    loss = regularizer(points)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert points.grad is not None
    torch.testing.assert_close(points.grad, torch.zeros_like(points))


def test_all_essential_h0_from_truncated_filtration_keeps_backward() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    h0 = rips_diagram_torch(points, max_dim=0, max_edge_length=0.1)[0]
    assert torch.isinf(h0[:, 1]).all()
    loss = betti_regularization_loss(h0, target_n_components=1)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert points.grad is not None
    torch.testing.assert_close(points.grad, torch.zeros_like(points))


def test_empty_h1_diagram_retains_zero_gradient_connection() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    h1 = rips_diagram_torch(points, max_dim=1)[1]
    assert h1.shape == (0, 2)
    assert h1.requires_grad
    h1.sum().backward()
    assert points.grad is not None
    torch.testing.assert_close(points.grad, torch.zeros_like(points))
