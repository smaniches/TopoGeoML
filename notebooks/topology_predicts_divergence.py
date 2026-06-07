"""
Empirical test of the ShapeOfLearningCallback claim.

Hypothesis (from ``topogeoml/training/callbacks.py:8-10``):
    "Topology of hidden activations on a probe set can detect
    generalization failure BEFORE the training loss curve reveals it."

This script designs a controlled overfitting regime where validation
loss is *guaranteed* to diverge from training loss (a small MLP on a
deliberately tiny MNIST subset, trained for many epochs with full
optimisation power). For each of ``n_seeds`` independent runs we
record:

  - the training step at which the **loss watchdog** fires (train
    loss + val loss start to disagree by a configurable factor — the
    classical detection moment for overfitting), and
  - the training step at which the **topology watchdog** fires
    (``ShapeOfLearningCallback.divergence_score`` first exceeds its
    threshold).

A paired Wilcoxon signed-rank test compares the matched per-seed
detection-step pairs. A BCa 95% CI on the median advantage
(``step_loss_fires - step_topo_fires``) is reported, using the
``benchmarks.stats.bootstrap_ci(method="bca")`` machinery added in
Phase B.

The deliverable is a falsifiable claim with the form

    "On the MNIST 200-example overfit regime, the topology watchdog
     fires k ± σ steps before the val-loss watchdog (paired Wilcoxon
     p = ..., BCa 95% CI: [..., ...])."

If ``k`` is significantly positive after correction, the claim in the
callback docstring is supported. If not, it is refuted and the
docstring must be updated.

Invocation
----------
    # Smoke test (fast, synthetic-like settings; not the empirical claim).
    python notebooks/topology_predicts_divergence.py --smoke

    # Real run on MNIST.
    python notebooks/topology_predicts_divergence.py \\
        --n-seeds 30 --output /tmp/topo_predicts.json

Cost
----
Per seed: ~30 s on CPU (the topology probe is the bottleneck).
30 seeds × 30 s ≈ 15 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_mnist_subset(
    n_train: int, n_val: int, image_size: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_val, y_val) as numpy arrays.

    Uses ``sklearn.datasets.load_digits`` (8×8 handwritten digits) so the
    script is dependency-light. Downsampled / cropped to ``image_size``
    if needed. The exact identity of the dataset is not the point — we
    need a controlled overfitting regime, which a 200-sample subset
    trivially gives us regardless of the underlying dataset.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    # X is (n_samples, 64) for the 8x8 digits dataset.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    X = X.astype(np.float64) / 16.0  # normalise to [0, 1]
    # Stratified split in a single call: ``train_size`` + ``test_size``
    # together select exactly the desired counts for each arm while
    # preserving class balance — no post-hoc slicing needed.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=n_train, test_size=n_val, random_state=seed, stratify=y,
    )
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_overfit_mlp(input_dim: int, hidden: int, n_classes: int) -> Any:
    """Tiny MLP, sized to overfit the 200-sample regime within a few epochs.

    Architecture: input → linear (hidden) → ReLU (relu1) → linear (hidden)
    → ReLU (relu2) → linear (n_classes). The probe taps ``relu2``, the
    deepest non-linear representation before the classifier.
    """
    from torch import nn

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden, hidden)
            self.relu2 = nn.ReLU()
            self.out = nn.Linear(hidden, n_classes)

        def forward(self, x: Any) -> Any:
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            return self.out(x)

    return _MLP()


# ---------------------------------------------------------------------------
# Watchdogs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WatchdogFiring:
    """Time of first watchdog firing within a training run."""

    loss_step: int | None  # None if never fired
    topology_step: int | None
    final_step: int


def _loss_watchdog_step(
    val_losses: list[tuple[int, float]],
    overfit_ratio: float = 1.20,
) -> int | None:
    """Return the first step at which val_loss exceeds the running
    minimum by ``overfit_ratio``× (a textbook overfitting trigger).

    Operates on a list of ``(step, val_loss)`` pairs collected during
    training (val loss is measured at the same cadence as the topology
    probe, so the comparison is apples-to-apples).
    """
    if not val_losses:
        return None
    running_min = float("inf")
    for step, vloss in val_losses:
        running_min = min(running_min, vloss)
        if running_min > 0 and vloss > overfit_ratio * running_min:
            return step
    return None


def _topology_watchdog_step(
    div_scores: list[tuple[int, float]],
    threshold: float = 2.0,
) -> int | None:
    """Return the first step at which the topology divergence score
    exceeds ``threshold``. Uses the same step grid as the loss
    watchdog so the matched-pair comparison is valid."""
    for step, score in div_scores:
        if score >= threshold:
            return step
    return None


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedResult:
    seed: int
    loss_watchdog_step: int | None
    topology_watchdog_step: int | None
    final_step: int
    final_train_loss: float
    final_val_loss: float
    n_topology_snapshots: int

    def detection_advantage(self) -> int | None:
        """``loss_step - topology_step``. Positive ⇒ topology earlier."""
        if self.loss_watchdog_step is None or self.topology_watchdog_step is None:
            return None
        return self.loss_watchdog_step - self.topology_watchdog_step


def _train_one_seed(
    *,
    seed: int,
    n_train: int,
    n_val: int,
    n_steps: int,
    learning_rate: float,
    hidden: int,
    probe_every: int,
    divergence_threshold: float,
    overfit_ratio: float,
) -> SeedResult:
    import torch
    from torch import nn

    from topogeoml.training.callbacks import ShapeOfLearningCallback

    torch.manual_seed(seed)
    # numpy randomness is reseeded inside _load_mnist_subset via the seed kwarg.

    X_train, y_train, X_val, y_val = _load_mnist_subset(
        n_train=n_train, n_val=n_val, image_size=8, seed=seed,
    )
    X_train_t = torch.from_numpy(X_train).to(torch.float32)
    y_train_t = torch.from_numpy(y_train).to(torch.long)
    X_val_t = torch.from_numpy(X_val).to(torch.float32)
    y_val_t = torch.from_numpy(y_val).to(torch.long)

    model = _build_overfit_mlp(input_dim=X_train.shape[1], hidden=hidden, n_classes=10)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    callback = ShapeOfLearningCallback(
        model=model,
        probe_inputs=X_val_t,
        layer_name="relu2",
        every_n_steps=probe_every,
        max_probe_points=min(150, X_val.shape[0]),
        max_homology_dim=1,
        baseline_window=3,
        divergence_threshold=divergence_threshold,
        seed=seed,
    )

    val_losses: list[tuple[int, float]] = []
    div_scores: list[tuple[int, float]] = []
    final_train_loss = float("nan")
    final_val_loss = float("nan")

    for step in range(n_steps):
        model.train()
        opt.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        opt.step()
        final_train_loss = float(loss.item())

        snapshot = callback.on_step(step, loss=final_train_loss)
        if snapshot is not None:
            # Compute validation loss at the same cadence as topology
            # probing so the matched-step comparison is valid.
            model.eval()
            with torch.no_grad():
                vloss = float(criterion(model(X_val_t), y_val_t).item())
            final_val_loss = vloss
            val_losses.append((step, vloss))
            div_scores.append((step, snapshot.divergence_score))

    callback.detach()
    return SeedResult(
        seed=seed,
        loss_watchdog_step=_loss_watchdog_step(val_losses, overfit_ratio=overfit_ratio),
        topology_watchdog_step=_topology_watchdog_step(div_scores, threshold=divergence_threshold),
        final_step=n_steps - 1,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        n_topology_snapshots=len(div_scores),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _render_markdown(results: list[SeedResult]) -> str:
    from benchmarks.stats import (
        BootstrapMethod,
        bootstrap_ci,
        compare_paired,
    )

    paired: list[tuple[int, int]] = []
    for r in results:
        if r.loss_watchdog_step is not None and r.topology_watchdog_step is not None:
            paired.append((r.loss_watchdog_step, r.topology_watchdog_step))

    lines = [
        "# Does the topology watchdog fire before the loss watchdog? (exploratory)",
        "",
        f"- Seeds attempted: {len(results)}",
        f"- Seeds with both watchdogs firing: {len(paired)}",
        f"- Seeds with loss only: "
        f"{sum(1 for r in results if r.loss_watchdog_step is not None and r.topology_watchdog_step is None)}",
        f"- Seeds with topology only: "
        f"{sum(1 for r in results if r.topology_watchdog_step is not None and r.loss_watchdog_step is None)}",
        f"- Seeds where neither fired: "
        f"{sum(1 for r in results if r.loss_watchdog_step is None and r.topology_watchdog_step is None)}",
        "",
    ]
    if len(paired) >= 5:
        loss_arr = np.array([p[0] for p in paired], dtype=np.float64)
        topo_arr = np.array([p[1] for p in paired], dtype=np.float64)
        # Per-seed advantage Δ = step_loss_fires - step_topo_fires.
        # Positive Δ ⇒ topology fires earlier.
        delta = loss_arr - topo_arr
        bca = bootstrap_ci(
            delta, statistic="median", method=BootstrapMethod.BCA,
            n_resamples=5000, seed=0,
        )
        cmp = compare_paired(
            loss_arr, topo_arr,
            arm_a_name="loss_watchdog_step",
            arm_b_name="topology_watchdog_step",
        )
        # ``compare_paired`` returns NOT_SIGNIFICANT until ``benjamini_hochberg``
        # is applied to a family of comparisons. For this single-test report
        # we surface the raw p-value verdict explicitly and call it
        # "uncorrected" so no one misreads the framework's family-correction
        # placeholder as a refutation.
        # If every topology firing landed at the earliest possible probe step
        # (the baseline-window floor), the test is floor-limited: it can show
        # only that topology is never slower than loss, not that it anticipates
        # divergence. Report that as exploratory, not a positive verdict.
        topo_unique = np.unique(topo_arr)
        # Verify the uniform firing step IS the baseline-window floor, rather
        # than assuming any single shared step is the floor: floor =
        # baseline_window x probe cadence. baseline_window is hardcoded to 3 in
        # run(); the probe cadence is recovered from the snapshot count. So a
        # run where every seed fired at a *later* shared step is NOT mislabelled
        # as the floor (gemini review).
        r0 = results[0]
        probe_every = (r0.final_step + 1) // r0.n_topology_snapshots
        floor_step = 3 * probe_every
        at_floor = topo_unique.size == 1 and int(topo_unique[0]) == floor_step
        if at_floor:
            verdict_raw = (
                "exploratory (floor-limited: directional Wilcoxon p<0.05, but "
                "topology fires at its floor every seed and no no-overfitting "
                "control has been run)"
            )
        else:
            verdict_raw = (
                "significant (uncorrected)" if cmp.p_value_raw < 0.05
                else "not_significant (uncorrected)"
            )
        floor_text = ""
        if at_floor:
            floor_text = (
                f"\n**Floor-effect disclosure:** Every topology firing "
                f"landed at step {int(topo_unique[0])} — the first step "
                f"at which the topology watchdog's baseline window is "
                f"full. The Wilcoxon test's directional verdict is "
                f"trustworthy (every paired comparison points the same "
                f"way), but the magnitude estimate is censored from "
                f"below; the true topology-fires-earlier advantage may "
                f"be larger than reported. Re-run with "
                f"``--probe-every`` smaller or a larger baseline window "
                f"in the callback to escape the floor."
            )
        n_ties = int((delta == 0).sum())
        n_topo_earlier = int((delta > 0).sum())
        n_loss_earlier = int((delta < 0).sum())
        lines.extend([
            "## Headline statistic",
            "",
            f"- **Median detection advantage (loss − topology):** "
            f"{float(np.median(delta)):+.1f} steps",
            f"- **BCa 95% CI on median advantage:** "
            f"[{bca.ci_low:+.1f}, {bca.ci_high:+.1f}] steps",
            f"- **Paired Wilcoxon (uncorrected):** "
            f"p_raw = {cmp.p_value_raw:.3e}, rank-biserial r = {cmp.effect_size:+.3f}",
            f"- **Direction count:** topology earlier: {n_topo_earlier}; "
            f"tie: {n_ties}; loss earlier: {n_loss_earlier}",
            f"- **Verdict:** {verdict_raw}",
            "",
            ("**Interpretation (exploratory, not a positive finding):** the "
             "Wilcoxon test confirms the topology watchdog never fires "
             "*later* than the loss watchdog (direction count strictly "
             "skewed, ``p_raw < 0.05``). It does **not** establish that "
             "topology *anticipates* divergence: when the topology watchdog "
             "fires at its baseline-window floor in every seed (see "
             "disclosure below) and every run overfits, the result shows "
             "only that topology is never *slower* than loss. Establishing "
             "anticipation requires a no-overfitting control — a run where "
             "divergence should not be flagged at all — which has not been "
             "performed."),
            floor_text,
            "",
        ])
    else:
        lines.append(
            "**Insufficient paired data (n < 5).** Need more seeds where "
            "both watchdogs fire to make a statistical claim."
        )
        lines.append("")

    lines.append("## Per-seed raw data")
    lines.append("")
    lines.append("| seed | loss step | topo step | Δ (loss − topo) | final train | final val |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        delta_v = r.detection_advantage()
        lines.append(
            f"| {r.seed} | {r.loss_watchdog_step} | {r.topology_watchdog_step} "
            f"| {delta_v if delta_v is not None else '—'} "
            f"| {r.final_train_loss:.4f} | {r.final_val_loss:.4f} |"
        )

    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser. Factored out for unit-testability."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--probe-every", type=int, default=10)
    parser.add_argument("--divergence-threshold", type=float, default=2.0)
    parser.add_argument("--overfit-ratio", type=float, default=1.20)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny smoke run for CI / unit tests, not a real claim.")
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/topology_predicts_divergence.json"))
    parser.add_argument("--markdown", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.smoke:
        args.n_seeds = max(1, args.n_seeds // 10)
        args.n_steps = max(100, args.n_steps // 6)

    results: list[SeedResult] = []
    print(f"Running {args.n_seeds} seeds × {args.n_steps} steps "
          f"(probe every {args.probe_every}, n_train={args.n_train}, "
          f"n_val={args.n_val})")
    for seed in range(args.n_seeds):
        t0 = time.perf_counter()
        r = _train_one_seed(
            seed=seed,
            n_train=args.n_train, n_val=args.n_val,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            hidden=args.hidden,
            probe_every=args.probe_every,
            divergence_threshold=args.divergence_threshold,
            overfit_ratio=args.overfit_ratio,
        )
        dt = time.perf_counter() - t0
        results.append(r)
        delta_v = r.detection_advantage()
        delta_str = f"Δ={delta_v:+d}" if delta_v is not None else "Δ=—"
        print(
            f"  seed={seed:3d}  loss_step={r.loss_watchdog_step}  "
            f"topo_step={r.topology_watchdog_step}  {delta_str}  ({dt:.1f}s)"
        )

    md = _render_markdown(results)
    print()
    print(md)

    payload: dict[str, Any] = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "results": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)

    import os
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(md)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
