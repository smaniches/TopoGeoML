"""Mathematical invariant tests for the corrected scalar sheaf benchmark arm."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _graph_laplacian(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> torch.Tensor:
    dense = torch.zeros((n_nodes, n_nodes), dtype=torch.float64)
    for i, j in edges:
        dense[i, i] += 1.0
        dense[j, j] += 1.0
        dense[i, j] -= 1.0
        dense[j, i] -= 1.0
    return dense.to_sparse().coalesce()


def _sheaf_model(
    input_dim: int = 2,
    num_classes: int = 2,
    hidden_dim: int = 3,
):
    from benchmarks.hodge.models import _SheafResidualGraphClassifier

    torch.manual_seed(7)
    return _SheafResidualGraphClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )


def _set_identity_restrictions(model) -> None:
    with torch.no_grad():
        model._sheaf_learner.weight.zero_()
        model._sheaf_learner.bias.fill_(1.0)


def test_one_undirected_edge_has_one_exact_coboundary_row() -> None:
    model = _sheaf_model(hidden_dim=2)
    _set_identity_restrictions(model)
    proj = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    laplacian = _graph_laplacian(2, [(0, 1)])

    delta = model._build_sheaf_coboundary(proj, laplacian)
    expected_delta = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    torch.testing.assert_close(delta, expected_delta, rtol=0.0, atol=0.0)

    learned = model._build_sheaf_laplacian(proj, laplacian)
    torch.testing.assert_close(
        learned,
        laplacian.to_dense(),
        rtol=0.0,
        atol=0.0,
    )


def test_identity_restrictions_recover_graph_laplacian_with_isolated_vertex() -> None:
    model = _sheaf_model(hidden_dim=2)
    _set_identity_restrictions(model)
    proj = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
        dtype=torch.float64,
    )
    laplacian = _graph_laplacian(3, [(0, 1)])

    learned = model._build_sheaf_laplacian(proj, laplacian)
    torch.testing.assert_close(
        learned,
        laplacian.to_dense(),
        rtol=0.0,
        atol=0.0,
    )

    normalized = model._normalize_sheaf_laplacian(learned)
    expected_normalized = torch.tensor(
        [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        normalized,
        expected_normalized,
        rtol=0.0,
        atol=0.0,
    )


def test_random_learned_operator_is_symmetric_positive_semidefinite() -> None:
    model = _sheaf_model(hidden_dim=3)
    proj = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.1],
            [-0.3, 0.2, 0.7],
            [0.8, -0.4, 0.2],
        ],
        dtype=torch.float64,
    )
    laplacian = _graph_laplacian(
        4,
        [(0, 1), (1, 2), (0, 2), (2, 3)],
    )

    learned = model._build_sheaf_laplacian(proj, laplacian)
    torch.testing.assert_close(
        learned,
        learned.transpose(0, 1),
        rtol=0.0,
        atol=1e-12,
    )
    eigenvalues = torch.linalg.eigvalsh(learned)
    assert eigenvalues.detach().min().item() >= -1e-10


def test_operator_is_consistent_under_node_permutation() -> None:
    model = _sheaf_model(hidden_dim=3)
    proj = torch.tensor(
        [
            [0.2, 0.1, -0.3],
            [0.7, -0.4, 0.5],
            [-0.2, 0.9, 0.6],
        ],
        dtype=torch.float64,
    )
    laplacian = _graph_laplacian(3, [(0, 1), (1, 2)])
    learned = model._build_sheaf_laplacian(proj, laplacian)

    permutation = torch.tensor([2, 0, 1])
    permuted_proj = proj[permutation]
    dense_laplacian = laplacian.to_dense()
    permuted_laplacian = dense_laplacian[permutation][:, permutation].to_sparse()
    permuted_learned = model._build_sheaf_laplacian(
        permuted_proj,
        permuted_laplacian,
    )

    expected = learned[permutation][:, permutation]
    torch.testing.assert_close(
        permuted_learned,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_real_loss_backpropagates_into_sheaf_and_message_parameters() -> None:
    model = _sheaf_model(hidden_dim=4)
    with torch.no_grad():
        model._mp_bias.fill_(0.5)
    x = torch.tensor(
        [[0.2, -0.1], [0.4, 0.8], [-0.7, 0.3]],
        dtype=torch.float64,
    )
    laplacian = _graph_laplacian(3, [(0, 1), (1, 2), (0, 2)])

    logits = model.forward_one(x, laplacian)
    loss = logits.square().sum()
    loss.backward()

    learner_grad = model._sheaf_learner.weight.grad
    message_grad = model._mp_weight.grad
    assert learner_grad is not None
    assert message_grad is not None
    assert torch.isfinite(learner_grad).all()
    assert torch.isfinite(message_grad).all()
    assert float(learner_grad.abs().sum()) > 0.0
    assert float(message_grad.abs().sum()) > 0.0


def test_nci1_parameter_count_is_exact_and_within_capacity_tolerance() -> None:
    from benchmarks.hodge.models import HodgeResidualClassifier, SheafResidualBaseline

    sheaf = SheafResidualBaseline.build(input_dim=37, num_classes=2, seed=0)
    control = HodgeResidualClassifier.build(input_dim=37, num_classes=2, seed=0)

    sheaf_count = sum(parameter.numel() for parameter in sheaf.parameters())
    control_count = sum(parameter.numel() for parameter in control.parameters())
    learner_count = sum(
        parameter.numel() for parameter in sheaf._sheaf_learner.parameters()
    )

    assert SheafResidualBaseline.version == "2.0.0"
    assert learner_count == 65
    assert sheaf_count == 2403
    assert control_count == 2338
    assert (sheaf_count - control_count) / control_count < 0.05
