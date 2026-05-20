"""
Experiment configuration loader and result serializer.

Item 10 of the v0.0.1 scope.

An experiment is fully specified by a YAML file describing:
  - dataset: how to generate / load inputs
  - pipeline: which TopoGeomML pipeline + hyperparameters
  - validation: how to score (CV folds, seed, metric)
  - output: where to write JSON results

Results are written as a single JSON object with mandatory provenance:
  - config (full echoed config)
  - results (metric values)
  - timing (per-step seconds)
  - environment (package versions, RNG seeds)
  - timestamp (UTC ISO 8601)

This is enough to reproduce or audit any run. v0.1 will add MLflow / W&B
adapters, but the JSON output is the canonical truth.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class DatasetConfig:
    """Synthetic-dataset spec for v0.0.1. Real-data loaders land in v0.1."""

    name: str  # e.g. "synthetic_shapes"
    n_per_class: int = 25
    n_points: int = 40
    noise: float = 0.05
    seed: int = 42


@dataclass
class PipelineConfig:
    """TopologyFeaturePipeline hyperparameters."""

    kind: str  # "topology_feature"
    max_homology_dim: int = 1
    max_edge_length: float | None = None
    vectorizer: str = "persistence_image"  # or "betti_curve"
    resolution: int = 15
    sigma: float = 0.1
    metric: str = "euclidean"


@dataclass
class ValidationConfig:
    """CV / scoring spec."""

    cv_folds: int = 5
    metric: str = "accuracy"
    seed: int = 42


@dataclass
class OutputConfig:
    """Where to write the JSON results."""

    path: str  # absolute or relative
    overwrite: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment spec."""

    name: str
    dataset: DatasetConfig
    pipeline: PipelineConfig
    validation: ValidationConfig
    output: OutputConfig
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            dataset=DatasetConfig(**data["dataset"]),
            pipeline=PipelineConfig(**data["pipeline"]),
            validation=ValidationConfig(**data["validation"]),
            output=OutputConfig(**data["output"]),
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment file into ExperimentConfig."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping; got {type(data).__name__}")
    return ExperimentConfig.from_dict(data)


def _environment_snapshot() -> dict[str, Any]:
    """Capture environment / version info for reproducibility."""
    import importlib.metadata as md

    def _ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "missing"

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "topogeoml_version": _ver("topogeoml"),
        "numpy_version": _ver("numpy"),
        "scipy_version": _ver("scipy"),
        "scikit_learn_version": _ver("scikit-learn"),
        "ripser_version": _ver("ripser"),
        "persim_version": _ver("persim"),
        "networkx_version": _ver("networkx"),
    }


def write_results(
    config: ExperimentConfig,
    results: dict[str, Any],
    timing: dict[str, float] | None = None,
) -> Path:
    """
    Serialize experiment results to JSON at config.output.path.

    Parameters
    ----------
    config : ExperimentConfig
    results : dict[str, Any]
        Metric values. Must be JSON-serializable.
    timing : dict[str, float], optional
        Per-step timing in seconds.

    Returns
    -------
    Path
        Absolute path to the written file.

    Raises
    ------
    FileExistsError
        If output.overwrite=False and the path already exists.
    """
    out_path = Path(config.output.path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"results file already exists: {out_path} (set output.overwrite: true to replace)"
        )

    payload: dict[str, Any] = {
        "name": config.name,
        "description": config.description,
        "tags": list(config.tags),
        "config": config.to_dict(),
        "results": _jsonify(results),
        "timing": dict(timing) if timing else {},
        "environment": _environment_snapshot(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    # Binary write (elite-code-standards §2.1) — atomic-ish via temp + rename
    # would be better; v0.0.1 keeps the simple form. Atomic write lands in v0.1.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        f.write(b"\n")
    os.replace(tmp_path, out_path)

    return out_path


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy / non-JSON values into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
