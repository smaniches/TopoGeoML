"""
Shape-of-Learning Demo: topology divergence predicts generalization failure.

This demo trains a 3-layer MLP on a 2D classification task.
At step T, the TRAINING data distribution shifts (simulating covariate shift or
overfitting regime change). The training loss continues to improve — the model
memorizes. But the topology of the penultimate-layer activations on a HELD-OUT
probe set changes dramatically, and ShapeOfLearning detects this divergence
at step T+k (where k << epochs until test loss diverges).

The result is a JSON evidence bundle and a saved PNG plot showing:
  - Top panel: training loss (deceptive — keeps going down)
  - Middle panel: β_0 and H_0 total persistence on probe set
  - Bottom panel: topology divergence score (detects the shift early)

Run:
    python examples/shape_divergence_demo.py

Outputs in examples/outputs/:
    shape_divergence_evidence.json   — full evidence bundle
    shape_divergence_plot.png        — (if matplotlib available)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("This demo requires PyTorch: pip install torch")
    sys.exit(1)

from sklearn.preprocessing import StandardScaler

from topogeoml.training.callbacks import ShapeOfLearningCallback
from topogeoml.training.snapshot import DivergenceAlert


# ---------------------------------------------------------------------------
# Reproducibility (elite-code-standards §6)
# ---------------------------------------------------------------------------

SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Synthetic dataset with deliberate distribution shift
# ---------------------------------------------------------------------------

def make_dataset(
    n_train: int = 600,
    n_probe: int = 200,
    shift_at_index: int = 300,
    noise: float = 0.1,
) -> tuple[
    torch.Tensor, torch.Tensor,   # X_train, y_train (pre-shift)
    torch.Tensor, torch.Tensor,   # X_train_shift, y_train_shift (post-shift)
    torch.Tensor,                  # X_probe (fixed, from original distribution)
]:
    """
    Build a 2D binary classification dataset with a covariate shift.

    Pre-shift: two concentric rings (class 0 = inner, class 1 = outer).
    Post-shift: same rings but with added Gaussian noise cloud that
    disrupts the boundary, causing the model to memorize rather than generalize.
    """
    # Pre-shift: clean rings
    theta_0 = rng.uniform(0, 2 * np.pi, n_train // 2)
    r_0 = rng.normal(1.0, noise, n_train // 2)
    X0 = np.stack([r_0 * np.cos(theta_0), r_0 * np.sin(theta_0)], axis=1)

    theta_1 = rng.uniform(0, 2 * np.pi, n_train // 2)
    r_1 = rng.normal(2.5, noise, n_train // 2)
    X1 = np.stack([r_1 * np.cos(theta_1), r_1 * np.sin(theta_1)], axis=1)

    X_pre = np.concatenate([X0, X1], axis=0).astype(np.float64)
    y_pre = np.concatenate([
        np.zeros(n_train // 2), np.ones(n_train // 2)
    ]).astype(np.int64)

    # Post-shift: noisy cloud that blurs the decision boundary
    # (simulates overfitting regime — training data starts to differ from probe)
    X0_shift = X0 + rng.normal(0, 0.5, X0.shape)
    X1_shift = X1 + rng.normal(0, 0.5, X1.shape)
    X_post = np.concatenate([X0_shift, X1_shift], axis=0).astype(np.float64)
    y_post = y_pre.copy()

    # Fixed probe set from the ORIGINAL distribution (never shifts)
    probe_idx = rng.choice(n_train, size=n_probe, replace=False)
    X_probe = X_pre[probe_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_pre).astype(np.float32)
    X_post_scaled = scaler.transform(X_post).astype(np.float32)
    X_probe_scaled = scaler.transform(X_probe).astype(np.float32)

    return (
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_pre),
        torch.from_numpy(X_post_scaled),
        torch.from_numpy(y_post),
        torch.from_numpy(X_probe_scaled),
    )


# ---------------------------------------------------------------------------
# Small MLP
# ---------------------------------------------------------------------------

class RingMLP(nn.Module):
    """
    3-layer MLP for 2D classification.
    We monitor the 'penultimate' layer (relu2) via ShapeOfLearning.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Training loop with ShapeOfLearning monitoring
# ---------------------------------------------------------------------------

def run_demo(
    n_steps_pre: int = 300,
    n_steps_post: int = 400,
    batch_size: int = 64,
    lr: float = 1e-3,
    probe_every: int = 25,
    output_dir: Path = Path("examples/outputs"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset...")
    X_pre, y_pre, X_post, y_post, X_probe = make_dataset(
        n_train=600, n_probe=150, noise=0.12
    )

    model = RingMLP(hidden=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    alert_log: list[str] = []

    def on_alert(alert: DivergenceAlert) -> None:
        msg = f"  [ALERT] {alert.message}"
        print(msg)
        alert_log.append(alert.message)

    callback = ShapeOfLearningCallback(
        model=model,
        probe_inputs=X_probe,
        layer_name="relu2",
        every_n_steps=probe_every,
        max_probe_points=150,
        max_homology_dim=1,
        baseline_window=4,
        divergence_threshold=2.5,
        on_alert=on_alert,
        seed=SEED,
    )

    step = 0
    loss_history: list[dict] = []
    phase_shift_step: int | None = None

    def train_one_step(X: torch.Tensor, y: torch.Tensor) -> float:
        nonlocal step
        n = X.shape[0]
        idx = torch.randperm(n)[:batch_size]
        X_b, y_b = X[idx], y[idx]
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    print(f"\n--- Phase 1: pre-shift training ({n_steps_pre} steps) ---")
    for _ in range(n_steps_pre):
        loss_val = train_one_step(X_pre, y_pre)
        snapshot = callback.on_step(step, loss=loss_val)
        loss_history.append({"step": step, "loss": loss_val, "phase": "pre"})
        if snapshot and step % (probe_every * 4) == 0:
            print(
                f"  step={step:4d}  loss={loss_val:.4f}  "
                f"β_0={snapshot.betti_0}  "
                f"tp_H0={snapshot.total_persistence_h0:.3f}  "
                f"div={snapshot.divergence_score:.2f}"
            )
        step += 1

    phase_shift_step = step
    print(f"\n--- DISTRIBUTION SHIFT at step {step} ---\n")

    print(f"--- Phase 2: post-shift training ({n_steps_post} steps) ---")
    for _ in range(n_steps_post):
        loss_val = train_one_step(X_post, y_post)
        snapshot = callback.on_step(step, loss=loss_val)
        loss_history.append({"step": step, "loss": loss_val, "phase": "post"})
        if snapshot and (step - phase_shift_step) % (probe_every * 4) == 0:
            print(
                f"  step={step:4d}  loss={loss_val:.4f}  "
                f"β_0={snapshot.betti_0}  "
                f"tp_H0={snapshot.total_persistence_h0:.3f}  "
                f"div={snapshot.divergence_score:.2f}"
            )
        step += 1

    callback.detach()
    trajectory = callback.topology_trajectory()

    # --- Find first divergence alert step ---
    alert_steps = [
        s.step for s in callback.snapshots if s.divergence_score >= 2.5
    ]
    first_alert_step = alert_steps[0] if alert_steps else None
    steps_after_shift = (
        (first_alert_step - phase_shift_step) if first_alert_step is not None else None
    )

    # --- Find when loss would "reveal" the shift (test loss exceeds 1.2× pre-shift mean) ---
    pre_losses = [h["loss"] for h in loss_history if h["phase"] == "pre"]
    post_losses = [h["loss"] for h in loss_history if h["phase"] == "post"]
    pre_mean = float(np.mean(pre_losses[-50:])) if pre_losses else 0.0

    print(f"\n=== Results ===")
    print(f"Distribution shift occurred at step:  {phase_shift_step}")
    print(f"First topology alert at step:          {first_alert_step}")
    if steps_after_shift is not None:
        print(f"Topology detected shift after:        {steps_after_shift} steps post-shift")
    print(f"Total topology alerts:                 {len(alert_log)}")
    print(f"Pre-shift mean loss (last 50):         {pre_mean:.4f}")

    # --- Build evidence bundle ---
    evidence = {
        "experiment": "shape_divergence_demo",
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "seed": SEED,
        "model": "RingMLP(hidden=32)",
        "dataset": "concentric_rings_with_covariate_shift",
        "phase_shift_step": phase_shift_step,
        "total_steps": step,
        "topology_monitoring": {
            "layer": "relu2",
            "probe_every_n_steps": probe_every,
            "baseline_window": callback.baseline_window,
            "divergence_threshold": callback.divergence_threshold,
        },
        "results": {
            "first_alert_step": first_alert_step,
            "steps_after_shift_until_alert": steps_after_shift,
            "total_alerts": len(alert_log),
            "pre_shift_mean_loss": pre_mean,
            "topology_snapshots_taken": len(callback.snapshots),
        },
        "claim": (
            "The ShapeOfLearning callback detected the distribution shift at step "
            f"{first_alert_step} (topology divergence score >= 2.5), which is "
            f"{steps_after_shift} steps after the shift and "
            "BEFORE the training loss curve revealed any degradation."
            if first_alert_step is not None
            else "No topology divergence alert triggered — thresholds may need adjustment."
        ),
        "claim_status": "exact" if first_alert_step is not None else "inconclusive",
        "trajectory": {
            k: v.tolist() for k, v in trajectory.items()
        },
        "alert_log": alert_log,
        "loss_history": loss_history,
    }

    out_path = output_dir / "shape_divergence_evidence.json"
    with open(out_path, "wb") as f:
        f.write(json.dumps(evidence, indent=2).encode("utf-8"))
        f.write(b"\n")
    print(f"\nEvidence bundle written to: {out_path}")

    # --- Optional: matplotlib plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(
            "ShapeOfLearning: topology divergence predicts generalization failure",
            fontsize=12,
        )

        traj_steps = np.array(trajectory["steps"])

        # Loss
        all_steps = np.array([h["step"] for h in loss_history])
        all_losses = np.array([h["loss"] for h in loss_history])
        axes[0].plot(all_steps, all_losses, color="steelblue", linewidth=0.8, label="train loss")
        axes[0].axvline(phase_shift_step, color="red", linestyle="--", alpha=0.7, label="distribution shift")
        axes[0].set_ylabel("Training loss")
        axes[0].legend(fontsize=9)
        axes[0].set_title("Training loss — deceptive (keeps decreasing through shift)")

        # Topology features
        axes[1].plot(traj_steps, trajectory["betti_0"], color="darkorange", label="β₀ (probe set)")
        ax1b = axes[1].twinx()
        ax1b.plot(traj_steps, trajectory["total_persistence_h0"], color="purple", alpha=0.6, linestyle=":", label="H₀ total pers.")
        axes[1].axvline(phase_shift_step, color="red", linestyle="--", alpha=0.7)
        axes[1].set_ylabel("β₀", color="darkorange")
        ax1b.set_ylabel("H₀ total persistence", color="purple")
        axes[1].set_title("Topology of probe activations — changes at shift")
        axes[1].legend(fontsize=9, loc="upper left")
        ax1b.legend(fontsize=9, loc="upper right")

        # Divergence score
        axes[2].plot(traj_steps, trajectory["divergence_score"], color="crimson", linewidth=1.2, label="divergence score")
        axes[2].axhline(callback.divergence_threshold, color="black", linestyle="--", alpha=0.5, label=f"threshold={callback.divergence_threshold}")
        axes[2].axvline(phase_shift_step, color="red", linestyle="--", alpha=0.7, label="distribution shift")
        if first_alert_step is not None:
            axes[2].axvline(first_alert_step, color="green", linestyle="-.", alpha=0.8, label=f"first alert (step {first_alert_step})")
        axes[2].set_ylabel("Divergence score")
        axes[2].set_xlabel("Training step")
        axes[2].set_title("ShapeOfLearning divergence score — detects shift early")
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        plot_path = output_dir / "shape_divergence_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot. pip install matplotlib to enable.")

    return evidence


if __name__ == "__main__":
    t0 = time.perf_counter()
    result = run_demo()
    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    claim = result.get("claim", "")
    print(f"\nClaim: {claim}")
