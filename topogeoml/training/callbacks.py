"""
ShapeOfLearning: topology monitoring callback for PyTorch training loops.

Probes the topology of model activations at regular intervals during
training. Detects representational divergence — structural change in the
learned representation that precedes (or contradicts) the training loss.

Empirical claim (and what supports it)
--------------------------------------
On a controlled overfitting regime (200-sample ``sklearn.load_digits``,
MLP with 64 hidden units, Adam, LR=1e-2, 600 steps, 30 independent
seeds), the topology divergence score fires *no later than* a textbook
val-loss-ratio watchdog, and in 14 of 30 seeds fires 10-30 steps
earlier (rank-biserial r = +1.000; paired Wilcoxon p_raw = 5.77e-04;
BCa 95% CI on the median advantage = [0, 10] steps — the CI lower
bound is set by the topology baseline-window floor at step 30, not by
a lack of effect).

The full empirical report (per-seed table + statistical analysis) is
in ``notebooks/results/topology_predicts_divergence_30seeds.md``;
reproduce with ``python notebooks/topology_predicts_divergence.py
--n-seeds 30``.

The floor effect (every topology firing landing at step 30 — the
earliest step its baseline window allows) means the magnitude is
censored from below. The directional verdict (topology never loses)
is robust; the magnitude estimate is a lower bound.

Usage
-----
    callback = ShapeOfLearningCallback(
        model=model,
        probe_inputs=X_val[:200],   # fixed probe set, not the training batch
        layer_name="relu2",
        every_n_steps=50,
    )

    for step, (X_batch, y_batch) in enumerate(train_loader):
        loss = model(X_batch)
        loss.backward()
        optimizer.step()
        snapshot = callback.on_step(step, loss=loss.item())
        if snapshot and snapshot.divergence_score > 2.0:
            print(f"Step {step}: topology divergence detected!")

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "topogeoml.training.callbacks requires PyTorch."
    ) from exc

from topogeoml.core.filtrations import RipsFiltration
from topogeoml.training.snapshot import DivergenceAlert, ShapeSnapshot


class ShapeOfLearningCallback:
    """
    Monitor topology of model activations throughout training.

    Attaches a forward hook to a named layer and, every `every_n_steps`,
    passes `probe_inputs` through the model, extracts activations, and
    computes their topological summary. Maintains a rolling baseline and
    computes a divergence score comparing recent topology to baseline.

    Parameters
    ----------
    model : nn.Module
        The model being trained.
    probe_inputs : torch.Tensor
        Fixed held-out inputs for topology probing. NOT training data —
        must be independent to avoid leakage. Shape: (n_probe, ...).
    layer_name : str
        Name of the layer to probe (key in model.named_modules()).
    every_n_steps : int
        How often to take a snapshot. Dense probing is expensive (runs PH).
    max_probe_points : int
        Maximum activation samples per snapshot (subsamples if more).
    max_homology_dim : int
        Highest PH dimension to compute (1 = H_0 + H_1).
    baseline_window : int
        Number of snapshots to use as the baseline window.
    divergence_threshold : float
        Divergence score above which DivergenceAlert is raised.
    on_alert : callable, optional
        Called with a DivergenceAlert when divergence is detected.
    seed : int
        RNG seed for reproducible subsampling.
    """

    def __init__(
        self,
        model: nn.Module,
        probe_inputs: torch.Tensor,
        layer_name: str,
        every_n_steps: int = 50,
        max_probe_points: int = 300,
        max_homology_dim: int = 1,
        baseline_window: int = 5,
        divergence_threshold: float = 2.0,
        on_alert: Callable[[DivergenceAlert], None] | None = None,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.probe_inputs = probe_inputs
        self.layer_name = layer_name
        self.every_n_steps = every_n_steps
        self.max_probe_points = max_probe_points
        self.max_homology_dim = max_homology_dim
        self.baseline_window = baseline_window
        self.divergence_threshold = divergence_threshold
        self.on_alert = on_alert
        self._rng = np.random.default_rng(seed)  # §6: seeded RNG

        self.snapshots: list[ShapeSnapshot] = []
        self._baseline_queue: deque[dict[str, float]] = deque(
            maxlen=baseline_window
        )
        self._hook_handle: Any = None
        self._captured_activation: torch.Tensor | None = None

        self._attach_hook()

    def _attach_hook(self) -> None:
        """Register forward hook on the named layer."""
        target: nn.Module | None = None
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                target = module
                break
        if target is None:
            available = [n for n, _ in self.model.named_modules() if n]
            raise ValueError(
                f"Layer '{self.layer_name}' not found in model. "
                f"Available layers: {available[:10]}..."
            )

        def _hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            # Detach: we only read activations, never differentiate through them.
            if isinstance(output, torch.Tensor):
                self._captured_activation = output.detach()

        self._hook_handle = target.register_forward_hook(_hook)

    def detach(self) -> None:
        """Remove the forward hook. Call when training ends."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _extract_activations(self) -> NDArray[np.float64]:
        """
        Run probe_inputs through model, return activations as float64 numpy.
        Subsamples to max_probe_points (§3: no Python sample loops for compute).
        """
        self._captured_activation = None
        with torch.no_grad():
            self.model(self.probe_inputs)

        if self._captured_activation is None:
            raise RuntimeError(
                f"Hook on layer '{self.layer_name}' captured no activation. "
                "Layer may not have been called during forward pass."
            )

        acts = self._captured_activation.cpu()
        # Flatten spatial dimensions: (batch, ...) -> (batch, features)
        acts_flat = acts.reshape(acts.shape[0], -1)
        n = acts_flat.shape[0]

        if n > self.max_probe_points:
            idx = self._rng.choice(n, size=self.max_probe_points, replace=False)
            acts_flat = acts_flat[idx]

        return acts_flat.numpy().astype(np.float64, copy=False)  # §1.3: explicit dtype

    def _compute_snapshot(
        self,
        step: int,
        train_loss: float,
        acts: NDArray[np.float64],
    ) -> ShapeSnapshot:
        """Compute full topological summary of an activation matrix."""
        n, _dim = acts.shape

        # Nearest-neighbour distances (vectorized, §3.1: no Python sample loops)
        from sklearn.neighbors import NearestNeighbors
        nn_model = NearestNeighbors(n_neighbors=2).fit(acts)
        dists, _ = nn_model.kneighbors(acts)
        nn_d = dists[:, 1]  # §1.2: no sqrt on negative (sklearn handles this)
        mean_nn = float(np.mean(nn_d))

        # Rips persistence
        rips = RipsFiltration(
            max_homology_dim=self.max_homology_dim,
            max_edge_length=float(
                np.percentile(nn_d, 95) * 4.0  # data-adaptive cutoff, §3
            ),
        )
        diagram = rips.compute(acts)

        def _count_significant(dim_k: int, threshold: float) -> int:
            bars = diagram.bars.get(dim_k)
            if bars is None or bars.size == 0:  # pragma: no cover
                # Defensive: ripser always returns at least one finite H_0
                # bar for a non-degenerate point cloud. Path fires only if
                # callers feed in a manually-constructed empty diagram.
                return 0
            finite_mask = np.isfinite(bars[:, 1])
            count = int((~finite_mask).sum())  # infinite bars always count
            if finite_mask.any():
                lifetimes = bars[finite_mask, 1] - bars[finite_mask, 0]
                count += int((lifetimes >= threshold).sum())
            return count

        sig_threshold = 2.0 * float(np.median(nn_d))

        b0 = _count_significant(0, sig_threshold)
        b1 = _count_significant(1, sig_threshold) if self.max_homology_dim >= 1 else -1

        tp_h0 = diagram.total_persistence(0, p=1.0) if 0 in diagram.bars else 0.0
        tp_h1 = diagram.total_persistence(1, p=1.0) if 1 in diagram.bars else 0.0

        h1_bars = diagram.bars.get(1)
        if h1_bars is not None and h1_bars.size > 0:
            finite_h1 = h1_bars[np.isfinite(h1_bars[:, 1])]
            longest_h1 = float(
                (finite_h1[:, 1] - finite_h1[:, 0]).max()
            ) if finite_h1.size > 0 else 0.0
        else:
            longest_h1 = 0.0

        # Persistence entropy for H_0
        h0_bars = diagram.bars.get(0, np.empty((0, 2), dtype=np.float64))
        h0_finite = h0_bars[np.isfinite(h0_bars[:, 1])]
        if h0_finite.size > 0:
            lifetimes_h0 = h0_finite[:, 1] - h0_finite[:, 0]
            total_h0 = lifetimes_h0.sum() + 1e-300  # §1.4
            p_h0 = np.maximum(lifetimes_h0 / total_h0, 1e-300)
            ent_h0 = float(-np.sum(p_h0 * np.log(p_h0)))
        else:  # pragma: no cover
            # Reachable only when the activation point cloud has no finite
            # H_0 lifetimes — happens for degenerate clouds (all identical
            # points). Defensive fallback.
            ent_h0 = 0.0

        h1_finite_bars = (
            h1_bars[np.isfinite(h1_bars[:, 1])]
            if h1_bars is not None and h1_bars.size > 0
            else np.empty((0, 2), dtype=np.float64)
        )
        if h1_finite_bars.size > 0:
            lifetimes_h1 = h1_finite_bars[:, 1] - h1_finite_bars[:, 0]
            total_h1 = lifetimes_h1.sum() + 1e-300  # §1.4
            p_h1 = np.maximum(lifetimes_h1 / total_h1, 1e-300)
            ent_h1 = float(-np.sum(p_h1 * np.log(p_h1)))
        else:
            ent_h1 = 0.0

        return ShapeSnapshot(
            step=step,
            train_loss=float(train_loss),
            betti_0=b0,
            betti_1=b1,
            total_persistence_h0=float(tp_h0),
            total_persistence_h1=float(tp_h1),
            persistence_entropy_h0=ent_h0,
            persistence_entropy_h1=ent_h1,
            longest_h1_lifetime=longest_h1,
            mean_nn_distance=mean_nn,
            n_points_used=n,
            layer_name=self.layer_name,
        )

    def _compute_divergence(self, snapshot: ShapeSnapshot) -> float:
        """
        Compute topology divergence vs baseline.

        Uses a multi-dimensional z-score across the key topology statistics.
        Returns 0.0 until the baseline window is filled.
        """
        if len(self._baseline_queue) < self.baseline_window:
            return 0.0

        # Topology feature vector for comparison
        def _features(s: dict[str, float]) -> NDArray[np.float64]:
            return np.array([
                s["betti_0"],
                s["total_persistence_h0"],
                s["persistence_entropy_h0"],
                s["total_persistence_h1"],
                s["longest_h1_lifetime"],
                s["mean_nn_distance"],
            ], dtype=np.float64)

        baseline_arr = np.stack([_features(b) for b in self._baseline_queue])
        baseline_mean = baseline_arr.mean(axis=0)
        baseline_std = baseline_arr.std(axis=0) + 1e-300  # §1.4

        current = _features({
            "betti_0": snapshot.betti_0,
            "total_persistence_h0": snapshot.total_persistence_h0,
            "persistence_entropy_h0": snapshot.persistence_entropy_h0,
            "total_persistence_h1": snapshot.total_persistence_h1,
            "longest_h1_lifetime": snapshot.longest_h1_lifetime,
            "mean_nn_distance": snapshot.mean_nn_distance,
        })

        z_scores = np.abs((current - baseline_mean) / baseline_std)
        return float(z_scores.max())

    def on_step(self, step: int, loss: float) -> ShapeSnapshot | None:
        """
        Call at every training step. Returns a ShapeSnapshot every
        `every_n_steps` steps; otherwise returns None.

        Parameters
        ----------
        step : int
            Current training step (0-indexed).
        loss : float
            Current training loss value.

        Returns
        -------
        ShapeSnapshot or None
        """
        if step % self.every_n_steps != 0:
            return None

        self.model.eval()
        try:
            acts = self._extract_activations()
        finally:
            self.model.train()

        snapshot = self._compute_snapshot(step, loss, acts)
        divergence = self._compute_divergence(snapshot)
        snapshot.divergence_score = divergence

        # Update baseline queue with current stats
        self._baseline_queue.append(snapshot.to_dict())

        self.snapshots.append(snapshot)

        # Raise alert if threshold exceeded
        if divergence >= self.divergence_threshold and self.on_alert is not None:
            # Compute loss delta vs baseline
            prev_losses = [s.train_loss for s in self.snapshots[-self.baseline_window - 1:-1]]
            loss_delta = 0.0
            if prev_losses:
                loss_delta = float(abs(loss - np.mean(prev_losses)))

            alert = DivergenceAlert(
                step=step,
                divergence_score=divergence,
                topology_delta=divergence,
                loss_delta=loss_delta,
                message=(
                    f"Step {step}: topology divergence score={divergence:.2f} "
                    f"(threshold={self.divergence_threshold}). "
                    f"β_0={snapshot.betti_0}, "
                    f"total_pers_H0={snapshot.total_persistence_h0:.4f}. "
                    f"Train loss={loss:.6f}."
                ),
            )
            self.on_alert(alert)

        return snapshot

    def topology_trajectory(self) -> dict[str, NDArray[np.float64]]:
        """
        Return time series arrays of all tracked topology statistics.

        Returns dict with keys: steps, train_loss, betti_0, betti_1,
        total_persistence_h0, total_persistence_h1, persistence_entropy_h0,
        longest_h1_lifetime, mean_nn_distance, divergence_score.
        """
        if not self.snapshots:
            return {}
        return {
            "steps": np.array([s.step for s in self.snapshots], dtype=np.float64),
            "train_loss": np.array([s.train_loss for s in self.snapshots], dtype=np.float64),
            "betti_0": np.array([s.betti_0 for s in self.snapshots], dtype=np.float64),
            "betti_1": np.array([s.betti_1 for s in self.snapshots], dtype=np.float64),
            "total_persistence_h0": np.array([s.total_persistence_h0 for s in self.snapshots], dtype=np.float64),
            "total_persistence_h1": np.array([s.total_persistence_h1 for s in self.snapshots], dtype=np.float64),
            "persistence_entropy_h0": np.array([s.persistence_entropy_h0 for s in self.snapshots], dtype=np.float64),
            "longest_h1_lifetime": np.array([s.longest_h1_lifetime for s in self.snapshots], dtype=np.float64),
            "mean_nn_distance": np.array([s.mean_nn_distance for s in self.snapshots], dtype=np.float64),
            "divergence_score": np.array([s.divergence_score for s in self.snapshots], dtype=np.float64),
        }
