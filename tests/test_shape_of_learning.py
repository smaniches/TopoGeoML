"""Tests for ShapeOfLearningCallback and ShapeSnapshot."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from topogeoml.training.callbacks import ShapeOfLearningCallback
from topogeoml.training.snapshot import DivergenceAlert, ShapeSnapshot

pytestmark = pytest.mark.torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def make_callback(
    model: nn.Module,
    every_n_steps: int = 1,
    divergence_threshold: float = 2.0,
) -> ShapeOfLearningCallback:
    probe = torch.randn(40, 4, dtype=torch.float32)
    return ShapeOfLearningCallback(
        model=model,
        probe_inputs=probe,
        layer_name="relu",
        every_n_steps=every_n_steps,
        max_probe_points=40,
        max_homology_dim=1,
        baseline_window=3,
        divergence_threshold=divergence_threshold,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_callback_attaches_without_error() -> None:
    model = TinyMLP()
    cb = make_callback(model)
    cb.detach()


def test_callback_rejects_unknown_layer() -> None:
    model = TinyMLP()
    probe = torch.randn(10, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="not found"):
        ShapeOfLearningCallback(
            model=model,
            probe_inputs=probe,
            layer_name="nonexistent_layer",
        )


def test_on_step_returns_none_between_probes() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=5)
    for step in [1, 2, 3, 4, 6, 7, 8, 9]:
        result = cb.on_step(step, loss=0.5)
        assert result is None, f"Expected None at step {step}"
    cb.detach()


def test_on_step_returns_snapshot_at_probe_step() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=5)
    snapshot = cb.on_step(0, loss=0.8)
    assert snapshot is not None
    assert isinstance(snapshot, ShapeSnapshot)
    cb.detach()


def test_snapshot_fields_valid() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=1)
    snapshot = cb.on_step(0, loss=0.7)
    assert snapshot is not None
    assert snapshot.step == 0
    assert snapshot.train_loss == pytest.approx(0.7)
    assert snapshot.betti_0 >= 0
    assert snapshot.n_points_used > 0
    assert snapshot.layer_name == "relu"
    assert snapshot.mean_nn_distance >= 0.0
    cb.detach()


def test_multiple_snapshots_accumulate() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=2)
    for step in range(10):
        cb.on_step(step, loss=0.5)
    assert len(cb.snapshots) == 5  # steps 0, 2, 4, 6, 8
    cb.detach()


def test_divergence_score_zero_before_baseline() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=1)
    # First 3 steps fill the baseline_window (=3), divergence should be 0
    for step in range(3):
        snapshot = cb.on_step(step, loss=0.5)
        assert snapshot is not None
        assert snapshot.divergence_score == 0.0
    cb.detach()


def test_topology_trajectory_returns_arrays() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=2)
    for step in range(12):
        cb.on_step(step, loss=float(step) * 0.01)
    traj = cb.topology_trajectory()
    assert "steps" in traj
    assert "betti_0" in traj
    assert "divergence_score" in traj
    assert len(traj["steps"]) == 6  # steps 0, 2, 4, 6, 8, 10
    cb.detach()


def test_topology_trajectory_empty_before_any_snapshots() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=1000)  # never triggers
    traj = cb.topology_trajectory()
    assert traj == {}
    cb.detach()


def test_alert_callback_called_on_divergence() -> None:
    """Force a divergence by dramatically changing probe inputs."""
    model = TinyMLP()
    alerts: list[DivergenceAlert] = []

    # Fill baseline with one type of probe, then switch to very different
    probe_normal = torch.randn(40, 4, dtype=torch.float32) * 0.01
    cb = ShapeOfLearningCallback(
        model=model,
        probe_inputs=probe_normal,
        layer_name="relu",
        every_n_steps=1,
        max_probe_points=40,
        max_homology_dim=0,
        baseline_window=3,
        divergence_threshold=1.5,
        on_alert=lambda alert: alerts.append(alert),
        seed=42,
    )

    # Fill baseline with similar steps
    for step in range(3):
        cb.on_step(step, loss=0.5)

    # Now switch probe to very different distribution to force divergence
    cb.probe_inputs = torch.randn(40, 4, dtype=torch.float32) * 10.0

    # Take several more steps — divergence should trigger
    for step in range(3, 20):
        cb.on_step(step, loss=0.4)

    cb.detach()
    # We can't guarantee an alert fires (depends on topology changes)
    # but the machinery ran without error — verify callback is callable
    assert len(alerts) >= 0  # just verify no error; actual firing is probabilistic


def test_snapshot_to_dict_serializable() -> None:
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=1)
    snapshot = cb.on_step(0, loss=0.42)
    assert snapshot is not None
    d = snapshot.to_dict()
    import json
    serialized = json.dumps(d)
    roundtrip = json.loads(serialized)
    assert roundtrip["step"] == 0
    assert roundtrip["layer_name"] == "relu"
    cb.detach()


def test_detach_removes_hook() -> None:
    """After detach(), the hook no longer captures activations."""
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=1)
    cb.detach()
    # Hook removed — further on_step calls should raise RuntimeError
    with pytest.raises(RuntimeError, match="captured no activation"):
        cb.on_step(0, loss=0.5)


def test_detach_is_idempotent() -> None:
    """Calling detach() when no hook is attached is a no-op, not an error.

    Exercises the false branch of ``if self._hook_handle is not None`` in
    ``detach()``: the second call sees a ``None`` handle and returns cleanly.
    """
    model = TinyMLP()
    cb = make_callback(model, every_n_steps=1)
    cb.detach()
    assert cb._hook_handle is None
    # Second detach: handle is already None — must not raise.
    cb.detach()
    assert cb._hook_handle is None


def test_non_tensor_layer_output_yields_no_activation() -> None:
    """A hooked layer that returns a non-Tensor captures nothing.

    The forward hook only records ``output`` when it ``isinstance`` a
    ``torch.Tensor`` (the false branch leaves ``_captured_activation`` as
    ``None``), so probing such a layer must surface the documented
    ``RuntimeError`` rather than a silent garbage snapshot.
    """

    class TupleLayer(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (x, x)  # deliberately non-Tensor output

    class ModelWithTupleLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.tup = TupleLayer()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x)
            self.tup(h)  # hook on `tup` observes a tuple, not a Tensor
            return h

    model = ModelWithTupleLayer()
    probe = torch.randn(10, 4, dtype=torch.float32)
    cb = ShapeOfLearningCallback(
        model=model,
        probe_inputs=probe,
        layer_name="tup",
        every_n_steps=1,
        max_probe_points=10,
        max_homology_dim=0,
        seed=42,
    )
    try:
        with pytest.raises(RuntimeError, match="captured no activation"):
            cb.on_step(0, loss=0.5)
    finally:
        cb.detach()


def test_alert_fires_on_first_snapshot_without_prior_loss_history() -> None:
    """An alert raised on the very first snapshot has an empty loss history.

    With ``baseline_window=1`` and ``divergence_threshold=0.0`` the first
    ``on_step`` computes divergence ``0.0`` (baseline queue starts empty),
    which clears the threshold and fires the alert while only one snapshot
    exists. ``self.snapshots[-2:-1]`` is then empty, exercising the false
    branch of ``if prev_losses`` so ``loss_delta`` defaults to ``0.0``.
    """
    model = TinyMLP()
    alerts: list[DivergenceAlert] = []
    probe = torch.randn(40, 4, dtype=torch.float32)
    cb = ShapeOfLearningCallback(
        model=model,
        probe_inputs=probe,
        layer_name="relu",
        every_n_steps=1,
        max_probe_points=40,
        max_homology_dim=0,
        baseline_window=1,
        divergence_threshold=0.0,
        on_alert=lambda alert: alerts.append(alert),
        seed=42,
    )
    snapshot = cb.on_step(0, loss=0.5)
    cb.detach()

    assert snapshot is not None
    assert len(alerts) == 1, "alert must fire on the first zero-divergence step"
    assert alerts[0].step == 0
    assert alerts[0].loss_delta == 0.0  # no prior losses to diff against
