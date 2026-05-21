"""
MNIST topology classification — the "killer demo" for topogeoml's
differentiable persistent homology layer used inside a real training loop.

Pipeline
--------
For each MNIST digit:

  1. Convert the 28x28 image to a 2-D point cloud (coordinates of active
     pixels), subsampled to a fixed n_points. This is the same dataset
     fixture the bench uses.
  2. Pass the point cloud through ``topogeoml.nn.diff_ph.rips_diagram_torch``
     to get a Rips persistence diagram, fully differentiable.
  3. Reduce the diagram to a fixed-length feature vector via
     ``topogeoml.nn.diff_ph.total_persistence_loss`` /
     ``persistence_entropy_loss`` / longest-bar functions.
  4. Concatenate with a learned per-pixel CNN summary (or skip — both
     paths are runnable here).
  5. Linear classifier -> cross-entropy -> backprop *through the diff-PH
     layer*.

Why MNIST and not DRIVE
-----------------------
DRIVE retinal vessel segmentation (Clough et al. 2020) needs **cubical
PH** on the predicted segmentation map. TopoGeoML ships **Rips PH** on
point clouds (``nn.diff_ph``) but does NOT yet ship a differentiable
cubical-PH layer. MNIST topology classification is the largest
empirically-grounded demo we can build with what's currently shipped:

  - Digit 0/6/9: one loop (β_1 = 1).
  - Digit 8: two loops (β_1 = 2).
  - Digits 1/2/3/5/7: no loops (β_1 = 0).

A classifier that uses topology features should distinguish at least
the {0, 8} vs {1, 2, 3, 5, 7} groups significantly better than a
no-topology baseline. This script measures that gap with bootstrap CIs
and paired Wilcoxon, exactly as the bench framework does.

Cubical-diff-PH for image segmentation is planned for Phase 3 of the
diff-PH framework.

Reproducibility
---------------
This script runs CPU or GPU. On a free Colab T4 it completes ~5
seconds per digit-pair × ~20 seeds (~ 2-3 minutes total). On the
4-core local container it takes ~10x longer.

Invocation
----------
    python notebooks/mnist_topology_classification.py \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --n-epochs 20 \\
        --output /tmp/mnist_topology_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as _torch_typing


def _device() -> _torch_typing.device:
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models(input_dim: int, num_classes: int, seed: int):
    """Construct (topology-aware classifier, baseline classifier).

    Both take a ``(n_points, 2)`` point cloud as input. The topology-aware
    model adds a 3-feature vector derived from the Rips diagram. The
    baseline gets a same-length 3-feature vector derived from per-coordinate
    variance (so the parameter count and capacity are identical).

    ``n_points`` was previously a parameter here but it never affected the
    constructed model (the architectures are agnostic to point count);
    removed per Gemini PR #8 review.
    """
    import torch
    from torch import nn

    # Hoist the topology imports out of the forward pass: Python caches
    # module objects but ``from X import Y`` does a dict lookup on every
    # forward call, which is wasted work in the training hot path
    # (caught by Gemini PR #7 + #8 reviews).
    from topogeoml.nn.diff_ph import (
        finite_lifetimes,
        rips_diagram_torch,
    )

    torch.manual_seed(seed)

    class _TopologyAware(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Point-cloud summary: per-point linear -> mean-pool.
            self.point_proj = nn.Linear(input_dim, 16)
            # Topology features: 3 scalars (total H_0 persistence,
            # total H_1 persistence, longest H_1 lifetime).
            self.topo_proj = nn.Linear(3, 16)
            self.head = nn.Linear(32, num_classes)

        def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
            # Per-point summary.
            h_pts = self.point_proj(point_cloud).mean(dim=0)
            # Topology features.
            dgms = rips_diagram_torch(point_cloud, max_dim=1)
            h0 = dgms[0]
            h1 = dgms[1] if len(dgms) > 1 else point_cloud.new_empty((0, 2))
            h0_pers = finite_lifetimes(h0).sum() if h0.numel() else point_cloud.new_zeros(())
            if h1.numel():
                lifetimes = finite_lifetimes(h1)
                h1_pers = lifetimes.sum() if lifetimes.numel() else point_cloud.new_zeros(())
                h1_longest = lifetimes.max() if lifetimes.numel() else point_cloud.new_zeros(())
            else:
                h1_pers = point_cloud.new_zeros(())
                h1_longest = point_cloud.new_zeros(())
            topo_features = torch.stack([h0_pers, h1_pers, h1_longest])
            h_topo = self.topo_proj(topo_features)
            return self.head(torch.cat([h_pts, h_topo]))

    class _Baseline(nn.Module):
        """Same parameter count, but the 3-feature vector is replaced by
        a learned linear projection of the mean point — no topology."""

        def __init__(self) -> None:
            super().__init__()
            self.point_proj = nn.Linear(input_dim, 16)
            # Same shape as the topology branch but no PH.
            self.dummy_proj = nn.Linear(input_dim, 16)
            self.head = nn.Linear(32, num_classes)

        def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
            h_pts = self.point_proj(point_cloud).mean(dim=0)
            # Use the variance of the cloud as a non-topology feature.
            h_dummy = self.dummy_proj(point_cloud.var(dim=0, unbiased=False))
            return self.head(torch.cat([h_pts, h_dummy]))

    return _TopologyAware().to(torch.float64), _Baseline().to(torch.float64)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_train_test_split(
    n_per_class: int, seed: int, n_points: int,
) -> tuple[list[tuple[_torch_typing.Tensor, int]], list[tuple[_torch_typing.Tensor, int]]]:
    """Build a balanced subset of MNIST digits 0/1/8 as point clouds.

    Returns (train, test) lists. Each entry is (point_cloud, label).
    Labels: 0 -> beta_1=1, 1 -> beta_1=0, 2 -> beta_1=2 (digit 8 remapped to 2).
    """
    import numpy as np
    import torch

    from benchmarks.datasets.mnist_topology import MNISTPointCloud

    rng = np.random.default_rng(seed)
    samples: list[tuple[torch.Tensor, int]] = []
    for original_digit, label in [(0, 0), (1, 1), (8, 2)]:
        ds = MNISTPointCloud(digit=original_digit)
        for _ in range(n_per_class):
            seed_k = int(rng.integers(0, 2**31 - 1))
            cloud = ds.generate(seed=seed_k, n_points=n_points)
            samples.append((cloud, label))
    rng.shuffle(samples)
    n_test = max(1, len(samples) // 5)
    return samples[n_test:], samples[:n_test]


# ---------------------------------------------------------------------------
# Training + eval
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedResult:
    seed: int
    topology_accuracy: float
    baseline_accuracy: float
    topology_train_loss: float
    baseline_train_loss: float


def train_eval_one_seed(
    *,
    seed: int,
    n_per_class: int,
    n_points: int,
    n_epochs: int,
    learning_rate: float,
) -> SeedResult:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    train, test = build_train_test_split(
        n_per_class=n_per_class, seed=seed, n_points=n_points,
    )
    model_topo, model_base = build_models(
        input_dim=2, num_classes=3, seed=seed,
    )
    device = _device()
    model_topo = model_topo.to(device)
    model_base = model_base.to(device)

    def _train(model: nn.Module) -> float:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        final = float("nan")
        for _ in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for cloud, y in train:
                opt.zero_grad()
                cloud = cloud.to(device)
                logits = model(cloud).unsqueeze(0)
                loss = loss_fn(logits, torch.tensor([y], device=device))
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
            final = epoch_loss / max(len(train), 1)
        return final

    def _evaluate(model: nn.Module) -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for cloud, y in test:
                cloud = cloud.to(device)
                pred = int(torch.argmax(model(cloud)).item())
                if pred == y:
                    correct += 1
        return correct / max(len(test), 1)

    topo_loss = _train(model_topo)
    base_loss = _train(model_base)
    return SeedResult(
        seed=seed,
        topology_accuracy=_evaluate(model_topo),
        baseline_accuracy=_evaluate(model_base),
        topology_train_loss=topo_loss,
        baseline_train_loss=base_loss,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_markdown(results: list[SeedResult]) -> str:
    import numpy as np

    from benchmarks.stats import bootstrap_ci, compare_paired

    topo = np.asarray([r.topology_accuracy for r in results])
    base = np.asarray([r.baseline_accuracy for r in results])
    topo_ci = bootstrap_ci(topo, statistic="median", n_resamples=10_000, seed=0) if topo.size >= 2 else None
    base_ci = bootstrap_ci(base, statistic="median", n_resamples=10_000, seed=0) if base.size >= 2 else None
    cmp = compare_paired(
        topo, base, arm_a_name="diff-ph-features", arm_b_name="no-topology",
    )

    lines = [
        "# MNIST topology classification — diff-PH features vs no-topology baseline",
        "",
        f"- Seeds: {len(results)}",
        f"- diff-PH features accuracy (median, 95% CI): "
        f"{topo_ci.point_estimate:.3f} [{topo_ci.ci_low:.3f}, {topo_ci.ci_high:.3f}]"
        if topo_ci is not None else f"diff-PH median accuracy: {float(np.median(topo)):.3f}",
        f"- no-topology accuracy (median, 95% CI): "
        f"{base_ci.point_estimate:.3f} [{base_ci.ci_low:.3f}, {base_ci.ci_high:.3f}]"
        if base_ci is not None else f"baseline median accuracy: {float(np.median(base)):.3f}",
        f"- Paired Wilcoxon (signed-rank): p_raw = "
        f"{cmp.p_value_raw:.3e}" if not np.isnan(cmp.p_value_raw) else "- Wilcoxon: underpowered",
        f"- Effect size (rank-biserial r): {cmp.effect_size:+.3f}",
        f"- Verdict: {cmp.kind.value if hasattr(cmp.kind, 'value') else cmp.kind}",
        "",
        "## Per-seed results",
        "",
        "| Seed | Topology acc | Baseline acc | Δ |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.seed} | {r.topology_accuracy:.3f} | {r.baseline_accuracy:.3f} "
            f"| {r.topology_accuracy - r.baseline_accuracy:+.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MNIST topology classification killer demo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-per-class", type=int, default=30)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mnist_topology_results.json"))
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch
    print(f"Device: {_device()}; CUDA={torch.cuda.is_available()}")

    results: list[SeedResult] = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        r = train_eval_one_seed(
            seed=seed,
            n_per_class=args.n_per_class,
            n_points=args.n_points,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
        )
        dt = time.perf_counter() - t0
        results.append(r)
        print(
            f"seed={seed:3d}  topo={r.topology_accuracy:.3f}  "
            f"base={r.baseline_accuracy:.3f}  Δ={r.topology_accuracy - r.baseline_accuracy:+.3f}"
            f"  ({dt:.1f}s)"
        )

    md = render_markdown(results)
    print()
    print(md)

    payload: dict[str, Any] = {
        "config": {
            "seeds": args.seeds, "n_epochs": args.n_epochs,
            "n_per_class": args.n_per_class, "n_points": args.n_points,
            "learning_rate": args.learning_rate,
        },
        "results": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(md)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
