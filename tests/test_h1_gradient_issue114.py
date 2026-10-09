"""Issue #114: H1 critical edge gradients must respect locality.

A disconnected edge with the same length as an unrelated loop's death
filtration is not a critical simplex for that loop. Returning its distance
as the reconstructed death value gives a correct *value* but a false gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ripser = pytest.importorskip("ripser")

from topogeoml.nn.diff_ph import rips_diagram_torch


def _separated_loop(external_length: float) -> torch.Tensor:
    """A square loop and a separate edge more than 8 units away."""
    return torch.tensor(
        [
            [-10.0, 0.0],
            [-10.0 + external_length, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )


def _longest_h1_lifetime_np(coords: np.ndarray) -> float:
    diagram = ripser.ripser(coords, maxdim=1)["dgms"][1]
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    assert finite.shape[0] == 1
    return float(np.max(finite[:, 1] - finite[:, 0]))


@pytest.mark.torch
@pytest.mark.parametrize("external_length", [float(np.sqrt(2.0)), 1.5])
def test_unrelated_external_edge_has_zero_h1_lifetime_gradient(
    external_length: float,
) -> None:
    points = _separated_loop(external_length)
    diagrams = rips_diagram_torch(points, max_dim=1)
    h1 = diagrams[1]
    finite = h1[torch.isfinite(h1).all(dim=1)]
    assert len(finite) == 1
    lifetime = (finite[:, 1] - finite[:, 0]).max()
    grad = torch.autograd.grad(lifetime, points)[0]

    # Moving a remote edge does not affect the square's persistence interval.
    eps = 1e-3
    raw = points.detach().numpy()
    for vertex in (0, 1):
        perturbed_plus = raw.copy()
        perturbed_minus = raw.copy()
        perturbed_plus[vertex, 0] += eps
        perturbed_minus[vertex, 0] -= eps
        finite_difference = (
            _longest_h1_lifetime_np(perturbed_plus)
            - _longest_h1_lifetime_np(perturbed_minus)
        ) / (2 * eps)
        assert finite_difference == pytest.approx(0.0, abs=1e-9)
        assert grad[vertex, 0].item() == pytest.approx(0.0, abs=1e-9)
    assert torch.allclose(grad[:2], torch.zeros_like(grad[:2]), atol=1e-9)
