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
    """A square loop and a separate edge more than 8 units away.

    For sqrt(2), use the integer vector (1, 1): the Euclidean distance
    is exactly equal in float64 to the square diagonal. Subtracting
    (-10 + sqrt(2)) - (-10) introduces a rounding difference and can
    accidentally mask the bug being tested.
    """
    external_endpoint = (
        [-9.0, 1.0] if external_length == float(np.sqrt(2.0))
        else [-10.0 + external_length, 0.0]
    )
    return torch.tensor(
        [
            [-10.0, 0.0],
            external_endpoint,
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
    pytest.importorskip("gudhi")
    points = _separated_loop(external_length)
    diagrams = rips_diagram_torch(
        points, max_dim=1, h1_tie_policy="gudhi"
    )
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


@pytest.mark.torch
def test_default_rejects_nondifferentiable_h1_edge_ties() -> None:
    points = _separated_loop(float(np.sqrt(2.0)))
    with pytest.raises(ValueError, match="Non-identifiable H1 gradient"):
        rips_diagram_torch(points, max_dim=1)

    # Forward-only persistence values remain available without selecting
    # any differentiable branch through a tied critical simplex.
    h1 = rips_diagram_torch(points.detach(), max_dim=1)[1]
    finite = h1[torch.isfinite(h1).all(dim=1)]
    assert len(finite) == 1
    assert (finite[0, 1] - finite[0, 0]).item() == pytest.approx(
        np.sqrt(2.0) - 1.0, abs=1e-8
    )


@pytest.mark.torch
def test_invalid_h1_tie_policy_is_explicit() -> None:
    with pytest.raises(ValueError, match="h1_tie_policy"):
        rips_diagram_torch(_separated_loop(1.5), h1_tie_policy="silent")  # type: ignore[arg-type]


@pytest.mark.torch
def test_gudhi_policy_missing_optional_dependency_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "gudhi", None)
    with pytest.raises(ImportError, match=r"topogeoml\[tda\]"):
        rips_diagram_torch(
            _separated_loop(float(np.sqrt(2.0))),
            max_dim=1,
            h1_tie_policy="gudhi",
        )


@pytest.mark.torch
def test_generic_h1_gradient_agrees_with_finite_differences() -> None:
    # A non-symmetric quadrilateral has a unique H1 bar, born at one side
    # and killed by one diagonal. Finite differences use ripser directly.
    base = np.array(
        [[0.0, 0.0], [1.15, 0.07], [1.09, 1.06], [-0.11, 0.96]],
        dtype=np.float64,
    )
    points = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    h1 = rips_diagram_torch(points, max_dim=1)[1]
    finite = h1[torch.isfinite(h1).all(dim=1)]
    assert len(finite) == 1
    lifetime = finite[0, 1] - finite[0, 0]
    gradient = torch.autograd.grad(lifetime, points)[0].detach().numpy()
    epsilon = 1e-3
    for i in range(len(base)):
        for j in range(base.shape[1]):
            plus, minus = base.copy(), base.copy()
            plus[i, j] += epsilon
            minus[i, j] -= epsilon
            fd = (
                _longest_h1_lifetime_np(plus) - _longest_h1_lifetime_np(minus)
            ) / (2 * epsilon)
            assert gradient[i, j] == pytest.approx(fd, abs=2e-4)
