"""
Classification axis — train each registered model on each registered
dataset for a fixed budget across N seeds, measure test accuracy,
report bootstrap CIs and paired Wilcoxon test results.

Methodology
-----------
For each (dataset, model, seed) we:
  1. Split the dataset 80/20 stratified by label.
  2. Build a fresh model instance.
  3. Train for ``n_epochs`` epochs of Adam(lr=1e-2), batch size 1
     (graphs vary in size; per-graph forward avoids padding overhead).
  4. Measure test accuracy.

Results across seeds form the comparison sample. Paired Wilcoxon
(across the matched seed list) drives the significance decision.
Bootstrap 95% CI is reported alongside the point estimate.

Why per-graph training (no padding/batching)
--------------------------------------------
The HodgeMP layer takes a Laplacian as a fixed buffer. Batching
graphs with different topology would either require block-diagonal
Laplacians (PyG's standard trick) or padding to the largest graph.
For the Phase-1 sample of MUTAG (188 graphs, avg 18 nodes) per-graph
training is fast enough and avoids the engineering cost of either
batching strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from benchmarks.hodge.datasets import GraphSample


@dataclass(frozen=True)
class ClassificationCell:
    model_name: str
    dataset_name: str
    seed: int
    test_accuracy: float
    n_train: int
    n_test: int
    final_train_loss: float


@dataclass(frozen=True)
class ClassificationReport:
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    n_epochs: int
    learning_rate: float
    cells: list[ClassificationCell]
    accuracy_median: float
    accuracy_ci95_low: float
    accuracy_ci95_high: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "cells"},
            "cells": [asdict(c) for c in self.cells],
        }


def _train_one_seed(
    model: nn.Module,
    train_samples: list[GraphSample],
    test_samples: list[GraphSample],
    *,
    n_epochs: int,
    learning_rate: float,
    seed: int,
) -> ClassificationCell:
    torch.manual_seed(seed)

    # Cast model to float64 to match the graph features / Laplacian dtype.
    model = model.to(torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    final_train_loss = float("nan")
    # Single RNG state for the whole training run so consecutive epochs
    # draw *different* permutations. Previously the RNG was reseeded
    # inside the epoch loop, producing identical sample order every
    # epoch — equivalent to running ``len(train_samples) * n_epochs``
    # SGD steps with a fixed-order curriculum, which is not SGD.
    # Caught by Gemini's PR #6 review.
    epoch_rng = np.random.default_rng(seed)
    for _ in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        perm = epoch_rng.permutation(len(train_samples))
        for idx in perm:
            sample = train_samples[int(idx)]
            opt.zero_grad()
            logits = model.forward_one(sample.x, sample.laplacian).unsqueeze(0)
            loss = loss_fn(logits, torch.tensor([sample.y]))
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        final_train_loss = epoch_loss / max(len(train_samples), 1)

    # Test accuracy.
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample in test_samples:
            logits = model.forward_one(sample.x, sample.laplacian)
            pred = int(torch.argmax(logits).item())
            if pred == sample.y:
                correct += 1
    acc = correct / max(len(test_samples), 1)

    return ClassificationCell(
        model_name="",  # filled by caller
        dataset_name="",  # filled by caller
        seed=seed,
        test_accuracy=acc,
        n_train=len(train_samples),
        n_test=len(test_samples),
        final_train_loss=final_train_loss,
    )


def _stratified_split(
    samples: list[GraphSample], test_fraction: float, seed: int
) -> tuple[list[GraphSample], list[GraphSample]]:
    """Stratified train/test split by label."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_label.setdefault(s.y, []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for indices in by_label.values():
        shuffled = list(rng.permutation(indices))
        n_test = max(1, round(test_fraction * len(shuffled)))
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]


def _project_features(
    samples: list[GraphSample],
    target_dim: int,
    seed: int,
) -> tuple[list[GraphSample], int]:
    """Apply a per-seed deterministic linear projection to all node features.

    Used by hypothesis 005 to isolate the feature-dim mechanism behind the
    residual-scale effect: projecting NCI1's 37-dim features down to 7-dim
    (matching MUTAG) — or expanding MUTAG's 7-dim up to 37-dim (matching
    NCI1) — leaves the graph topology, sample size, and label distribution
    intact while varying only the input feature dimensionality.

    The projection matrix is the same Gaussian draw across all graphs for
    a given seed, so the operation is a *global* linear transform of the
    feature space — not per-graph noise.

    Returns the (new samples, new input_dim) tuple.
    """
    if not samples:
        return samples, target_dim
    src_dim = samples[0].x.shape[1]
    rng = np.random.default_rng(seed)
    # Gaussian projection scaled by 1/sqrt(src_dim) so the post-projection
    # feature norms are O(1) regardless of dim change (Johnson-Lindenstrauss
    # scaling for dim-reduction; same scale works for dim-expansion).
    P = torch.from_numpy(
        rng.normal(loc=0.0, scale=1.0 / max(1.0, src_dim**0.5), size=(src_dim, target_dim))
    ).to(torch.float64)
    projected = [
        GraphSample(x=s.x @ P, laplacian=s.laplacian, y=s.y)
        for s in samples
    ]
    return projected, target_dim


def run_classification(
    *,
    model_cls: Any,
    dataset: Any,
    seeds: list[int],
    n_epochs: int = 20,
    learning_rate: float = 1e-2,
    test_fraction: float = 0.2,
    max_graphs: int | None = None,
    feature_projection_dim: int | None = None,
) -> ClassificationReport:
    """Run the classification axis for one (model, dataset) pair.

    Parameters
    ----------
    max_graphs
        Optional cap. If set and ``len(samples) > max_graphs``,
        subsample the dataset (deterministically per seed) BEFORE the
        stratified train/test split. Used by hypothesis 004 to isolate
        the sample-size mechanism behind the residual-scale effect:
        running NCI1 at ``max_graphs=188`` produces a MUTAG-sized
        subset with NCI1's native feature distribution intact, so the
        only variable that changes vs full NCI1 is sample count.
    feature_projection_dim
        Optional target dimensionality for a per-seed deterministic
        Gaussian projection applied to all node features. Used by
        hypothesis 005 to isolate the feature-dim mechanism: setting
        this to 7 on NCI1 produces NCI1-7d (matching MUTAG's feature
        dim while preserving NCI1's sample size and graph statistics);
        setting it to 37 on MUTAG produces MUTAG-37d (matching NCI1's
        feature space). The projection matrix is the same across all
        graphs in a given seed but varies across seeds so the
        projection itself is not a confound.
    """
    from benchmarks.stats import bootstrap_ci

    samples, input_dim, num_classes = dataset.load()

    cells: list[ClassificationCell] = []
    for seed in seeds:
        torch.manual_seed(seed)
        # Per-seed deterministic subsampling. Done BEFORE the train/test
        # split so the stratification still applies to the subsampled
        # set (and the test fraction stays at 20% of the subset, not of
        # the full dataset).
        if max_graphs is not None and len(samples) > max_graphs:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(samples), size=max_graphs, replace=False)
            seed_samples = [samples[int(i)] for i in indices]
        else:
            seed_samples = samples
        # Per-seed deterministic feature projection (hypothesis 005).
        # Applied AFTER subsampling so the projection doesn't waste
        # work on subsampled-away graphs.
        if feature_projection_dim is not None and feature_projection_dim > 0:
            seed_samples, effective_input_dim = _project_features(
                seed_samples, target_dim=feature_projection_dim, seed=seed,
            )
        else:
            effective_input_dim = input_dim
        train_samples, test_samples = _stratified_split(
            seed_samples, test_fraction=test_fraction, seed=seed,
        )
        model = model_cls.build(effective_input_dim, num_classes, seed=seed)
        cell = _train_one_seed(
            model, train_samples, test_samples,
            n_epochs=n_epochs, learning_rate=learning_rate, seed=seed,
        )
        # Backfill names — dataclasses are frozen so build a new instance.
        cell = ClassificationCell(
            model_name=model_cls.name,
            dataset_name=dataset.name,
            seed=cell.seed,
            test_accuracy=cell.test_accuracy,
            n_train=cell.n_train,
            n_test=cell.n_test,
            final_train_loss=cell.final_train_loss,
        )
        cells.append(cell)

    accs = np.asarray([c.test_accuracy for c in cells], dtype=np.float64)
    if accs.size >= 2:
        ci = bootstrap_ci(
            accs, statistic="median", confidence_level=0.95,
            n_resamples=10_000, seed=0,
        )
        median, lo, hi = ci.point_estimate, ci.ci_low, ci.ci_high
    else:  # pragma: no cover
        # Single-seed runs report a point estimate only.
        median = float(np.median(accs)) if accs.size else float("nan")
        lo = hi = float("nan")

    return ClassificationReport(
        model_name=model_cls.name,
        model_version=model_cls.version,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        cells=cells,
        accuracy_median=median,
        accuracy_ci95_low=lo,
        accuracy_ci95_high=hi,
    )
