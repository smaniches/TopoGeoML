"""Tests for experiment configs and result writing (item 10)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topogeoml.experiments import (
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    PipelineConfig,
    ValidationConfig,
    load_experiment_config,
    write_results,
)


def _sample_config(out_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test_exp",
        description="test",
        tags=["a", "b"],
        dataset=DatasetConfig(name="synthetic_shapes", n_per_class=4, n_points=20, noise=0.05, seed=1),
        pipeline=PipelineConfig(
            kind="topology_feature",
            max_homology_dim=1,
            vectorizer="betti_curve",
            resolution=20,
        ),
        validation=ValidationConfig(cv_folds=3, metric="accuracy", seed=42),
        output=OutputConfig(path=str(out_path), overwrite=True),
    )


def test_load_yaml_round_trip(tmp_path: Path) -> None:
    """YAML written to disk loads back into an equivalent ExperimentConfig."""
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        """
name: t1
description: round-trip test
tags: [synthetic]
dataset:
  name: synthetic_shapes
  n_per_class: 5
  n_points: 30
  noise: 0.03
  seed: 1
pipeline:
  kind: topology_feature
  max_homology_dim: 1
  vectorizer: persistence_image
  resolution: 10
  sigma: 0.1
  metric: euclidean
validation:
  cv_folds: 3
  metric: accuracy
  seed: 42
output:
  path: out.json
  overwrite: true
""".lstrip()
    )
    cfg = load_experiment_config(cfg_path)
    assert cfg.name == "t1"
    assert cfg.dataset.n_per_class == 5
    assert cfg.pipeline.vectorizer == "persistence_image"
    assert cfg.validation.cv_folds == 3
    assert cfg.output.overwrite is True


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "does_not_exist.yaml")


def test_load_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- not a mapping\n- still not a mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config(p)


def test_write_results_creates_file(tmp_path: Path) -> None:
    out_path = tmp_path / "results.json"
    cfg = _sample_config(out_path)
    results = {"cv_mean": 0.95, "cv_std": 0.02}
    written = write_results(cfg, results, timing={"fit_s": 1.5})
    assert written.exists()
    assert written == out_path.resolve()


def test_write_results_contains_required_fields(tmp_path: Path) -> None:
    out_path = tmp_path / "results.json"
    cfg = _sample_config(out_path)
    write_results(cfg, {"cv_mean": 0.9}, timing={"fit_s": 0.1})
    payload = json.loads(out_path.read_text())
    for key in ["name", "config", "results", "timing", "environment", "timestamp_utc"]:
        assert key in payload, f"missing key: {key}"
    assert payload["config"]["pipeline"]["kind"] == "topology_feature"
    assert payload["results"]["cv_mean"] == 0.9


def test_write_results_environment_captures_versions(tmp_path: Path) -> None:
    out_path = tmp_path / "results.json"
    cfg = _sample_config(out_path)
    write_results(cfg, {})
    payload = json.loads(out_path.read_text())
    env = payload["environment"]
    assert "topogeoml_version" in env
    assert "numpy_version" in env
    assert "python_version" in env


def test_write_results_jsonifies_numpy_values(tmp_path: Path) -> None:
    """numpy floats / arrays must be converted to JSON-safe values."""
    out_path = tmp_path / "results.json"
    cfg = _sample_config(out_path)
    results = {
        "scores": np.array([0.1, 0.2, 0.3]),
        "mean": np.float64(0.2),
        "count": np.int64(3),
        "flag": np.bool_(True),
    }
    write_results(cfg, results)
    payload = json.loads(out_path.read_text())
    assert payload["results"]["scores"] == [0.1, 0.2, 0.3]
    assert payload["results"]["mean"] == 0.2
    assert payload["results"]["count"] == 3
    assert payload["results"]["flag"] is True


def test_write_results_refuses_overwrite(tmp_path: Path) -> None:
    out_path = tmp_path / "results.json"
    out_path.write_text("{}\n")
    cfg = _sample_config(out_path)
    cfg.output.overwrite = False
    with pytest.raises(FileExistsError):
        write_results(cfg, {})


def test_write_results_creates_parent_dirs(tmp_path: Path) -> None:
    """Missing intermediate directories are created."""
    out_path = tmp_path / "nested" / "deep" / "results.json"
    cfg = _sample_config(out_path)
    write_results(cfg, {"x": 1})
    assert out_path.exists()
