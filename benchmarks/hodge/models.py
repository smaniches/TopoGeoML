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
    complex of the graph), apply one ``HodgeMessagePassing`` layer to
    the node features, sum-pool across nodes, and run a linear
    classifier. One Laplacian per graph is cached in the model's
    instance dict so we don't pay the construction cost more than once
    per forward.
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
        # The HodgeMP layer is constructed lazily per-graph because each
        # graph has its own Laplacian.
        self._proj_in = nn.Linear(input_dim, hidden_dim)

    def forward_one(
        self, x: torch.Tensor, laplacian: torch.sparse.Tensor
    ) -> torch.Tensor:
        """Forward for a single graph: x is (n_nodes, input_dim), output is
        (num_classes,) logits."""
        from topogeoml.nn.hodge import HodgeMessagePassing

        h = self._proj_in(x)
        layer = HodgeMessagePassing(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            laplacian=laplacian,
        ).to(torch.float64)
        h = self._activation(layer(h))
        # Sum-pool across nodes -> graph embedding.
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
