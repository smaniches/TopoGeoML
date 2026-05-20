"""
ShapeSnapshot: one topology measurement taken during training.

Each snapshot records the topological state of a model's activations
at a specific training step, enabling ShapeOfLearning trajectory analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShapeSnapshot:
    """
    Topological state of model activations at one training step.

    Fields
    ------
    step : int
        Global training step index.
    train_loss : float
        Training loss value at this step.
    betti_0 : int
        Estimated β_0 (connected components) of activations.
    betti_1 : int
        Estimated β_1 (loops) of activations. -1 if max_dim < 1.
    total_persistence_h0 : float
        L¹ total persistence for H_0.
    total_persistence_h1 : float
        L¹ total persistence for H_1. 0.0 if max_dim < 1.
    persistence_entropy_h0 : float
        Persistence entropy for H_0.
    persistence_entropy_h1 : float
        Persistence entropy for H_1. 0.0 if max_dim < 1.
    longest_h1_lifetime : float
        Lifetime of the most persistent H_1 bar. 0.0 if none.
    mean_nn_distance : float
        Mean nearest-neighbour distance in activation space (density proxy).
    n_points_used : int
        Number of activation samples used for this snapshot.
    layer_name : str
        Name of the model layer whose activations were probed.
    divergence_score : float
        Topology divergence relative to baseline window. 0.0 before baseline set.
    extras : dict
        Any additional diagnostics (backend-specific or user-defined).
    """

    step: int
    train_loss: float
    betti_0: int
    betti_1: int
    total_persistence_h0: float
    total_persistence_h1: float
    persistence_entropy_h0: float
    persistence_entropy_h1: float
    longest_h1_lifetime: float
    mean_nn_distance: float
    n_points_used: int
    layer_name: str
    divergence_score: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (all values are Python scalars)."""
        return {
            "step": int(self.step),
            "train_loss": float(self.train_loss),
            "betti_0": int(self.betti_0),
            "betti_1": int(self.betti_1),
            "total_persistence_h0": float(self.total_persistence_h0),
            "total_persistence_h1": float(self.total_persistence_h1),
            "persistence_entropy_h0": float(self.persistence_entropy_h0),
            "persistence_entropy_h1": float(self.persistence_entropy_h1),
            "longest_h1_lifetime": float(self.longest_h1_lifetime),
            "mean_nn_distance": float(self.mean_nn_distance),
            "n_points_used": int(self.n_points_used),
            "layer_name": self.layer_name,
            "divergence_score": float(self.divergence_score),
            "extras": dict(self.extras),
        }


@dataclass
class DivergenceAlert:
    """
    Raised when topology diverges while training loss doesn't.

    This is the core signal of ShapeOfLearning: the model is learning
    something that changes representational structure but isn't yet
    visible in the training objective.
    """

    step: int
    divergence_score: float
    topology_delta: float
    loss_delta: float
    message: str
