"""
Example: shape classification via topological features.

Two-class synthetic dataset (noisy circles vs noisy lines) classified with
topological features only. Demonstrates the end-to-end TopologyFeaturePipeline
flow and verifies that topology alone separates the classes.

Run:
    python examples/circles_vs_lines.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topogeoml import TopologyFeaturePipeline


def make_dataset(
    n_per_class: int = 25,
    n_points: int = 40,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[list[NDArray[np.float64]], NDArray[np.int64]]:
    """Build a balanced circles-vs-lines dataset."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False, dtype=np.float64)
    t = np.linspace(-1.0, 1.0, n_points, dtype=np.float64)

    X: list[NDArray[np.float64]] = []
    y: list[int] = []
    for _ in range(n_per_class):
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        circle += noise * rng.standard_normal(circle.shape)
        X.append(np.ascontiguousarray(circle, dtype=np.float64))
        y.append(1)

        line = np.stack([t, np.zeros_like(t)], axis=1)
        line += noise * rng.standard_normal(line.shape)
        X.append(np.ascontiguousarray(line, dtype=np.float64))
        y.append(0)

    return X, np.array(y, dtype=np.int64)


def main() -> None:
    print("Building synthetic dataset...")
    X, y = make_dataset(n_per_class=25, n_points=40, noise=0.05)
    print(f"  {len(X)} samples, {(y == 1).sum()} circles, {(y == 0).sum()} lines")

    pipeline = Pipeline(
        [
            (
                "topology",
                TopologyFeaturePipeline(
                    max_homology_dim=1,
                    resolution=15,
                    vectorizer="persistence_image",
                ),
            ),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=500)),
        ]
    )

    print("\nCross-validating (5-fold)...")
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=1)
    print(f"  CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in scores]}")

    print("\nFitting on full dataset...")
    pipeline.fit(X, y)
    train_acc = pipeline.score(X, y)
    print(f"  Train accuracy: {train_acc:.4f}")

    print("\nFit provenance:")
    prov = pipeline.named_steps["topology"].fit_provenance_
    print(f"  n_samples_seen   = {prov.n_samples_seen}")
    print(f"  ambient_dim      = {prov.ambient_dim}")
    print(f"  max_homology_dim = {prov.max_homology_dim}")
    print(f"  vectorizer       = {prov.vectorizer}")
    print(f"  output_dim       = {prov.output_dim}")
    print(f"  fallback_max     = {prov.extras['fallback_max']:.4f}")
    print(f"  pipeline_version = {prov.pipeline_version}")


if __name__ == "__main__":
    main()
