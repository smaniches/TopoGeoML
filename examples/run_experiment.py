"""
Run a TopoGeoML v0.0.1 experiment from a YAML config and write JSON results.

Usage:
    python examples/run_experiment.py examples/configs/synthetic_shapes.yaml

Designed as the canonical entry point for benchmarks: every run produces a
single JSON artifact with config echo, results, timing, environment snapshot,
and UTC timestamp. Reproducibility contract complete.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topogeoml import TopologyFeaturePipeline
from topogeoml.experiments import (
    ExperimentConfig,
    load_experiment_config,
    write_results,
)


def build_synthetic_shapes(
    n_per_class: int,
    n_points: int,
    noise: float,
    seed: int,
) -> tuple[list[NDArray[np.float64]], NDArray[np.int64]]:
    """Build the circles-vs-lines dataset described in DatasetConfig."""
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


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Execute one experiment from a parsed config; return results dict."""
    timings: dict[str, float] = {}

    if config.dataset.name != "synthetic_shapes":
        raise NotImplementedError(
            f"v0.0.1 supports dataset='synthetic_shapes' only; got '{config.dataset.name}'"
        )
    if config.pipeline.kind != "topology_feature":
        raise NotImplementedError(
            f"v0.0.1 supports pipeline.kind='topology_feature' only; got '{config.pipeline.kind}'"
        )

    t0 = time.perf_counter()
    X, y = build_synthetic_shapes(
        n_per_class=config.dataset.n_per_class,
        n_points=config.dataset.n_points,
        noise=config.dataset.noise,
        seed=config.dataset.seed,
    )
    timings["dataset_build_s"] = time.perf_counter() - t0

    pipeline = Pipeline(
        [
            (
                "topology",
                TopologyFeaturePipeline(
                    max_homology_dim=config.pipeline.max_homology_dim,
                    max_edge_length=config.pipeline.max_edge_length,
                    vectorizer=config.pipeline.vectorizer,  # type: ignore[arg-type]
                    resolution=config.pipeline.resolution,
                    sigma=config.pipeline.sigma,
                    metric=config.pipeline.metric,
                ),
            ),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=config.validation.seed, max_iter=500)),
        ]
    )

    cv = StratifiedKFold(
        n_splits=config.validation.cv_folds,
        shuffle=True,
        random_state=config.validation.seed,
    )

    t1 = time.perf_counter()
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=config.validation.metric, n_jobs=1)
    timings["cross_val_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    pipeline.fit(X, y)
    train_score = pipeline.score(X, y)
    timings["full_fit_s"] = time.perf_counter() - t2

    prov = pipeline.named_steps["topology"].fit_provenance_

    return {
        "metric": config.validation.metric,
        "cv_scores": scores.tolist(),
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "train_score": float(train_score),
        "n_samples": len(X),
        "feature_dim": prov.output_dim,
        "fit_provenance": {
            "n_samples_seen": prov.n_samples_seen,
            "ambient_dim": prov.ambient_dim,
            "max_homology_dim": prov.max_homology_dim,
            "vectorizer": prov.vectorizer,
            "output_dim": prov.output_dim,
            "metric": prov.metric,
            "pipeline_version": prov.pipeline_version,
            "fallback_max": prov.extras.get("fallback_max"),
        },
        "_timings_s": timings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a TopoGeoML YAML experiment.")
    parser.add_argument("config", help="Path to experiment YAML file")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    print(f"Running experiment: {config.name}")
    print(f"  dataset: {config.dataset.name}")
    print(f"  pipeline: {config.pipeline.kind} / {config.pipeline.vectorizer}")
    print(f"  validation: {config.validation.cv_folds}-fold CV on {config.validation.metric}")

    results = run_experiment(config)
    timing = results.pop("_timings_s")
    assert isinstance(timing, dict)

    out_path = write_results(config, results, timing=timing)
    print(f"\nResults:")
    print(f"  CV {config.validation.metric}: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print(f"  Train {config.validation.metric}: {results['train_score']:.4f}")
    print(f"  Feature dim: {results['feature_dim']}")
    print(f"  Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
