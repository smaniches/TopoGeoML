"""
Graph-classifier wrappers for the Hodge bench.

Each model implements the :class:`GraphClassifier` protocol. The runner
constructs a fresh instance per (seed, dataset) cell, trains it for a
fixed budget, then measures test accuracy.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class GraphClassifier(Protocol):
    """A graph classifier the Hodge bench can train and evaluate."""

    name: ClassVar[str]
    version: ClassVar[str]

    @staticmethod
    def available() -> bool: ...

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module: ...


class _HodgeGraphClassifier(nn.Module):
    """HodgeMP layer + sum-pool + linear head.

    For each graph we precompute the L_0 Hodge Laplacian (clique
    complex of the graph), apply one Hodge message-passing step to
    the node features, sum-pool across nodes, and run a linear
    classifier.

    The Hodge propagation step replicates ``HodgeMessagePassing``
    inline — ``activation(L @ x @ W + b)`` — but with **shared**
    learnable ``W`` and ``b`` across graphs. Previously the layer was
    constructed inside ``forward_one`` per graph, which (a) reset its
    Xavier-initialised weights on every call and (b) left those weights
    out of ``model.parameters()`` so the optimizer never saw them.
    Caught by Gemini's PR #6 review; this rewrite fixes it.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head = nn.Linear(hidden_dim, num_classes)
        self._activation = nn.ReLU()
        self._proj_in = nn.Linear(input_dim, hidden_dim)
        # Shared Hodge-MP parameters. Xavier init for ``W`` matches the
        # default in ``topogeoml.nn.hodge.HodgeMessagePassing``; ``b``
        # starts at zero per the original layer's convention.
        self._mp_weight = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=torch.float64)
        )
        self._mp_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        nn.init.xavier_uniform_(self._mp_weight)
        # Project the head to float64 so the dtype chain stays consistent
        # with the inline Hodge propagation.
        self.head = self.head.to(torch.float64)
        self._proj_in = self._proj_in.to(torch.float64)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        """Forward for a single graph: x is (n_nodes, input_dim), output is
        (num_classes,) logits.

        Inline Hodge propagation step (mirroring
        ``HodgeMessagePassing.forward``):
            h ← activation( L_norm @ proj_in(x) @ W_shared + b_shared )
        with ``W_shared`` and ``b_shared`` defined in ``__init__`` so
        the optimizer actually updates them.
        """
        h = self._proj_in(x)
        propagated = torch.sparse.mm(laplacian, h)
        h = self._activation(propagated @ self._mp_weight + self._mp_bias)
        graph_emb = h.sum(dim=0)
        return self.head(graph_emb)


class HodgeClassifier:
    """HodgeMP-based graph classifier."""

    name: ClassVar[str] = "hodge-mp-classifier"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import topogeoml.nn.hodge  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _HodgeGraphClassifier(input_dim, num_classes)


class _MLPGraphClassifier(nn.Module):
    """No-topology baseline: feature-MLP applied per-node, then sum-pool.

    This is the natural control for ``HodgeClassifier`` — both apply a
    per-node transformation and then sum-pool to a graph embedding. The
    Hodge model differs only in that its per-node transformation
    incorporates the Laplacian (= the graph structure).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        # Baseline ignores the laplacian — that's the point.
        del laplacian
        h = self.net(x)
        return self.head(h.sum(dim=0))


class MLPBaseline:
    """No-topology baseline (feature MLP + sum-pool)."""

    name: ClassVar[str] = "mlp-baseline"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def build(input_dim: int, num_classes: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return _MLPGraphClassifier(input_dim, num_classes)


REGISTERED: dict[str, type[GraphClassifier]] = {
    HodgeClassifier.name: HodgeClassifier,
    MLPBaseline.name: MLPBaseline,
}


def get_model(name: str) -> type[GraphClassifier]:
    if name not in REGISTERED:
        known = ", ".join(sorted(REGISTERED))
        raise KeyError(f"unknown model {name!r}; registered: {known}")
    return REGISTERED[name]
