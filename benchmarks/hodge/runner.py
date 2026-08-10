"""
Hodge-benchmark runner for seeded graph-classification comparisons.

The runner executes the requested model-by-dataset cells, performs paired
Wilcoxon comparisons across shared seeds, and applies Benjamini-Hochberg
correction at alpha=0.05 within each dataset's requested model family.

The generic alpha=0.05 report is descriptive. Scientific hypothesis verdicts
must use the preregistered threshold in the corresponding hypothesis document,
which can be stricter than 0.05.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.hodge.classification import ClassificationReport, run_classification
from benchmarks.hodge.datasets import REGISTERED as DATASETS
from benchmarks.hodge.models import REGISTERED as MODELS
from benchmarks.stats import benjamini_hochberg, compare_paired

SCHEMA_VERSION = "hodge-1.0.0"


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod in ("torch", "numpy", "scipy", "torch_geometric"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[mod] = "not installed"
    return versions


@dataclass
class HodgeRunResult:
    schema_version: str
    timestamp_utc: str
    platform_string: str
    python_version: str
    git_commit_sha: str = ""
    dependency_versions: dict[str, str] = field(default_factory=dict)
    reports: list[ClassificationReport] = field(default_factory=list)
    pairwise_comparisons: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp_utc": self.timestamp_utc,
            "platform_string": self.platform_string,
            "python_version": self.python_version,
            "git_commit_sha": self.git_commit_sha,
            "dependency_versions": self.dependency_versions,
            "reports": [r.as_dict() for r in self.reports],
            "pairwise_comparisons": self.pairwise_comparisons,
        }


def run(
    *,
    model_names: list[str] | None = None,
    dataset_names: list[str] | None = None,
    seeds: list[int] | None = None,
    n_epochs: int = 20,
    learning_rate: float = 1e-2,
    max_graphs: int | None = None,
    feature_projection_dim: int | None = None,
    constant_features: bool = False,
) -> HodgeRunResult:
    """Run the classification axis on every requested model/dataset/seed cell.

    When at least two models are present for a dataset, compute paired
    Wilcoxon comparisons over the shared seed list and apply Benjamini-Hochberg
    correction at alpha=0.05 within that requested model family.

    ``model_names=None`` means every model currently registered in ``MODELS``.
    ``dataset_names=None`` means every dataset currently registered in
    ``DATASETS``. Historical reproduction must therefore pass the original
    model and dataset families explicitly; otherwise registry growth can change
    both the executed experiment and the BH-adjusted p-values.
    """
    if model_names is None:
        model_names = list(MODELS)
    if dataset_names is None:
        dataset_names = list(DATASETS)
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    result = HodgeRunResult(
        schema_version=SCHEMA_VERSION,
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        platform_string=platform.platform(),
        python_version=platform.python_version(),
        git_commit_sha=_get_git_sha(),
        dependency_versions=_get_dependency_versions(),
    )

    # Per-(model, dataset) classification reports.
    per_dataset_reports: dict[str, list[ClassificationReport]] = {}
    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]
        if not dataset.available():  # pragma: no cover
            continue
        for model_name in model_names:
            model_cls = MODELS[model_name]
            if not model_cls.available():  # pragma: no cover
                continue
            report = run_classification(
                model_cls=model_cls,
                dataset=dataset,
                seeds=seeds,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                max_graphs=max_graphs,
                feature_projection_dim=feature_projection_dim,
                constant_features=constant_features,
            )
            result.reports.append(report)
            per_dataset_reports.setdefault(dataset_name, []).append(report)

    # Pairwise comparisons (only meaningful when at least two models are present).
    for dataset_name, reports in per_dataset_reports.items():
        if len(reports) < 2:  # pragma: no cover
            continue
        raw_comparisons = []
        for i in range(len(reports)):
            for j in range(i + 1, len(reports)):
                r_a = reports[i]
                r_b = reports[j]
                seed_map_a = {c.seed: c.test_accuracy for c in r_a.cells}
                seed_map_b = {c.seed: c.test_accuracy for c in r_b.cells}
                common = sorted(set(seed_map_a) & set(seed_map_b))
                arm_a = np.asarray([seed_map_a[s] for s in common])
                arm_b = np.asarray([seed_map_b[s] for s in common])
                cmp = compare_paired(
                    arm_a, arm_b,
                    arm_a_name=r_a.model_name, arm_b_name=r_b.model_name,
                )
                raw_comparisons.append(cmp)
        family = benjamini_hochberg(raw_comparisons, alpha=0.05)
        for cmp in family.comparisons:
            result.pairwise_comparisons.append({
                "dataset": dataset_name,
                **cmp.as_dict(),
            })

    return result


def write_result(result: HodgeRunResult, path: Path) -> None:
    """Atomically write a result JSON file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    tmp.replace(path)


def render_markdown(result: HodgeRunResult) -> str:
    """Render benchmark results without converting generic p-values into hypotheses."""
    lines: list[str] = []
    lines.append("# TopoGeoML Hodge subsystem benchmark")
    lines.append("")
    lines.append(f"- Schema: `{result.schema_version}`")
    lines.append(f"- Timestamp (UTC): {result.timestamp_utc}")
    lines.append(f"- Platform: {result.platform_string}")
    lines.append(f"- Python: {result.python_version}")
    if result.git_commit_sha:
        lines.append(f"- Git commit: `{result.git_commit_sha}`")
    if result.dependency_versions:
        deps = ", ".join(f"{k}={v}" for k, v in sorted(result.dependency_versions.items()))
        lines.append(f"- Dependencies: {deps}")
    lines.append("")
    lines.append("## Per-(model × dataset) test accuracy")
    lines.append("")
    lines.append("| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |")
    lines.append("|---|---|---|---|")
    for r in result.reports:
        ci_str = (
            f"{r.accuracy_median:.3f} [{r.accuracy_ci95_low:.3f}, {r.accuracy_ci95_high:.3f}]"
            if not np.isnan(r.accuracy_ci95_low)
            else f"{r.accuracy_median:.3f} (CI n/a)"
        )
        lines.append(f"| `{r.model_name}` | `{r.dataset_name}` | {ci_str} | {len(r.cells)} |")
    lines.append("")

    if result.pairwise_comparisons:
        lines.append("## Paired Wilcoxon plus Benjamini-Hochberg correction")
        lines.append("")
        lines.append(
            "The final column reports generic BH significance at alpha=0.05 for this "
            "requested model family. It is not a preregistered hypothesis verdict."
        )
        lines.append("")
        lines.append("| Dataset | A | B | median delta | p_raw | p_BH | Effect (r) | BH at 0.05 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in result.pairwise_comparisons:
            bh_status = {
                "significant": "significant",
                "not_significant": "not significant",
                "underpowered": "insufficient paired samples",
            }.get(c["kind"], "unknown")
            p_raw_str = f"{c['p_value_raw']:.3e}" if not np.isnan(c["p_value_raw"]) else "n/a"
            p_bh_str = (
                f"{c['p_value_bh_adjusted']:.3e}"
                if c["p_value_bh_adjusted"] is not None else "n/a"
            )
            lines.append(
                f"| {c['dataset']} | `{c['arm_a_name']}` | `{c['arm_b_name']}` "
                f"| {c['median_diff']:+.4f} | {p_raw_str} | {p_bh_str} "
                f"| {c['effect_size']:.3f} | {bh_status} |"
            )
        lines.append("")
        lines.append(
            "_Use the corresponding preregistration to determine the scientific "
            "decision threshold. Some TopoGeoML sub-hypotheses require p_BH < 0.01._"
        )

    return "\n".join(lines)


__all__ = [
    "SCHEMA_VERSION",
    "HodgeRunResult",
    "render_markdown",
    "run",
    "write_result",
]
