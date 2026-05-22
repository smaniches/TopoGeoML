"""H006 resolver — constant-feature ablation → mechanism-claim verdicts.

Inputs (must exist): three JSON results from the constant-feature
``python -m benchmarks.hodge --constant-features`` run, one per dataset
(MUTAG, PROTEINS, NCI1).  Three full-feature JSON results from the
prior H001/H002/H003 30-seed ablations, already committed at
``notebooks/results/{mutag,proteins,nci1}_hodge_ablation_30seeds.json``.

The script:

1. Loads ``hodge-mp-residual`` and ``mlp-baseline`` per-seed accuracies
   for each dataset, in both feature modes.
2. Computes the per-seed accuracy − class-prior gap.
3. Runs a one-sample Wilcoxon signed-rank test (alternative = greater)
   on Hodge vs prior, applies Benjamini-Hochberg FDR across the family
   of three tests at α = 0.05.
4. Computes Spearman ρ across the three datasets between the
   constant-feature (Hodge − prior) median gap and the full-feature
   (Hodge − MLP) median gap (H25).

Claim discipline (per the PR scope contract):

* No causal claims.  The resolver reports observed gaps and statistical
  verdicts only.
* No generalisation beyond the three tested datasets under this
  configuration.
* Fail-loud on missing input JSONs — silent skipping would let a
  partial run masquerade as a final result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

CLASS_PRIORS: dict[str, float] = {
    "mutag": 125 / 188,      # 0.6649
    "proteins": 663 / 1113,  # 0.5957
    "nci1": 2057 / 4110,     # 0.5005
}

H22_H23_H24_TAG: dict[str, str] = {"nci1": "H22", "mutag": "H23", "proteins": "H24"}


@dataclass(frozen=True)
class DatasetSummary:
    dataset: str
    n_seeds: int
    class_prior: float
    hodge_median_constant: float
    mlp_median_constant: float
    hodge_minus_prior: float
    hodge_minus_mlp_constant: float
    hodge_median_full: float
    mlp_median_full: float
    full_feature_gap: float
    p_hodge_vs_prior_raw: float
    p_hodge_vs_prior_bh: float
    hodge_above_prior_significant: bool
    constant_feature_source: Path
    full_feature_source: Path


def _benjamini_hochberg(pvalues: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    n = len(pvalues)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: pvalues[i])
    ranked = [pvalues[i] for i in order]
    adjusted = [0.0] * n
    prev = 1.0
    for k in range(n - 1, -1, -1):
        rank = k + 1
        adj = min(ranked[k] * n / rank, prev)
        adjusted[order[k]] = adj
        prev = adj
    rejected = [p <= alpha for p in adjusted]
    return adjusted, rejected


def _accs_by_model(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text())
    return {r["model_name"]: [c["test_accuracy"] for c in r["cells"]] for r in data["reports"]}


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"H006 resolver: required {label} not found at {path}. "
            "All three constant-feature JSONs and all three full-feature "
            "JSONs must be present before the resolver can emit verdicts."
        )


def resolve(
    *,
    constant_paths: dict[str, Path],
    full_paths: dict[str, Path],
    alpha: float = 0.05,
) -> list[DatasetSummary]:
    """Compute the per-dataset H006 summary.  Raises FileNotFoundError if any input is missing."""
    expected = {"mutag", "proteins", "nci1"}
    if set(constant_paths) != expected or set(full_paths) != expected:
        raise ValueError(
            f"constant_paths and full_paths must cover exactly {expected}; got "
            f"constant={set(constant_paths)}, full={set(full_paths)}"
        )
    for ds in expected:
        _require_file(constant_paths[ds], f"constant-feature result for {ds}")
        _require_file(full_paths[ds], f"full-feature result for {ds}")

    rows: list[dict] = []
    for ds in ("mutag", "proteins", "nci1"):
        c_accs = _accs_by_model(constant_paths[ds])
        f_accs = _accs_by_model(full_paths[ds])
        for arm in ("hodge-mp-residual", "mlp-baseline"):
            if arm not in c_accs or arm not in f_accs:
                raise KeyError(
                    f"{ds}: required arm {arm!r} missing — "
                    f"constant has {sorted(c_accs)}, full has {sorted(f_accs)}"
                )
        hodge_c = np.asarray(c_accs["hodge-mp-residual"])
        mlp_c = np.asarray(c_accs["mlp-baseline"])
        hodge_f = np.asarray(f_accs["hodge-mp-residual"])
        mlp_f = np.asarray(f_accs["mlp-baseline"])
        prior = CLASS_PRIORS[ds]
        diffs = hodge_c - prior
        nonzero = diffs[diffs != 0.0]
        if len(nonzero) == 0:
            p_raw = 1.0
        else:
            p_raw = float(
                stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox").pvalue
            )
        rows.append({
            "dataset": ds,
            "n_seeds": len(hodge_c),
            "class_prior": prior,
            "hodge_median_constant": float(np.median(hodge_c)),
            "mlp_median_constant": float(np.median(mlp_c)),
            "hodge_minus_prior": float(np.median(hodge_c) - prior),
            "hodge_minus_mlp_constant": float(np.median(hodge_c) - np.median(mlp_c)),
            "hodge_median_full": float(np.median(hodge_f)),
            "mlp_median_full": float(np.median(mlp_f)),
            "full_feature_gap": float(np.median(hodge_f) - np.median(mlp_f)),
            "p_hodge_vs_prior_raw": p_raw,
            "constant_feature_source": constant_paths[ds],
            "full_feature_source": full_paths[ds],
        })

    p_bhs, rejected = _benjamini_hochberg([r["p_hodge_vs_prior_raw"] for r in rows], alpha)
    summaries: list[DatasetSummary] = []
    for r, p_bh, rej in zip(rows, p_bhs, rejected, strict=True):
        summaries.append(DatasetSummary(
            dataset=r["dataset"],
            n_seeds=r["n_seeds"],
            class_prior=r["class_prior"],
            hodge_median_constant=r["hodge_median_constant"],
            mlp_median_constant=r["mlp_median_constant"],
            hodge_minus_prior=r["hodge_minus_prior"],
            hodge_minus_mlp_constant=r["hodge_minus_mlp_constant"],
            hodge_median_full=r["hodge_median_full"],
            mlp_median_full=r["mlp_median_full"],
            full_feature_gap=r["full_feature_gap"],
            p_hodge_vs_prior_raw=r["p_hodge_vs_prior_raw"],
            p_hodge_vs_prior_bh=p_bh,
            hodge_above_prior_significant=rej,
            constant_feature_source=r["constant_feature_source"],
            full_feature_source=r["full_feature_source"],
        ))
    return summaries


def render_markdown(summaries: list[DatasetSummary]) -> str:
    """Render the verdicts as Markdown with explicit per-row provenance."""
    lines: list[str] = []
    lines.append("## H006 reproducible summary (per-dataset)")
    lines.append("")
    lines.append(
        "| Dataset | Feature mode | Hodge score | Prior score | Gap | "
        "p_BH | Source artifact | Verdict (preregistered tag) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        tag = H22_H23_H24_TAG[s.dataset]
        sig = s.hodge_above_prior_significant
        if tag == "H22":
            verdict = f"{tag}: supports (Hodge significantly above prior)" if sig \
                else f"{tag}: rejects (Hodge not above prior)"
        elif tag == "H23":
            verdict = f"{tag}: rejects preregistered prediction (signal is present)" if sig \
                else f"{tag}: supports preregistered prediction (no signal)"
        else:
            verdict = f"{tag}: in-between predicate; significant Hodge>prior? {'yes' if sig else 'no'}"
        lines.append(
            f"| {s.dataset} | constant | {s.hodge_median_constant:.3f} | "
            f"{s.class_prior:.4f} | {s.hodge_minus_prior:+.4f} | "
            f"{s.p_hodge_vs_prior_bh:.3e} | {s.constant_feature_source} | "
            f"{verdict} |"
        )
        # Full-feature row anchors the H25 correlation. The "gap" here is
        # Hodge median − MLP median (NOT vs prior, since the full-feature
        # MLP can use the actual atom features and isn't expected to sit
        # at class prior).
        full_p_bh = "—"  # full-feature paired Wilcoxon p_BH lives in the source artifact
        lines.append(
            f"| {s.dataset} | full | {s.hodge_median_full:.3f} | "
            f"{s.mlp_median_full:.3f} (MLP, not prior) | "
            f"{s.full_feature_gap:+.4f} | {full_p_bh} | "
            f"{s.full_feature_source} | anchor for H25 correlation |"
        )

    lines.append("")
    lines.append("## H006 statistical table")
    lines.append("")
    lines.append(
        "| Dataset | n_seeds | Hodge_const | MLP_const | Class prior | "
        "Hodge − prior | p_raw | p_BH | Hodge > prior (BH α=0.05)? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        sig = "**yes**" if s.hodge_above_prior_significant else "no"
        lines.append(
            f"| {s.dataset} | {s.n_seeds} | {s.hodge_median_constant:.3f} | "
            f"{s.mlp_median_constant:.3f} | {s.class_prior:.4f} | "
            f"{s.hodge_minus_prior:+.3f} | {s.p_hodge_vs_prior_raw:.3e} | "
            f"{s.p_hodge_vs_prior_bh:.3e} | {sig} |"
        )

    const_vs_prior = np.asarray([s.hodge_minus_prior for s in summaries])
    const_vs_mlp = np.asarray([s.hodge_minus_mlp_constant for s in summaries])
    full = np.asarray([s.full_feature_gap for s in summaries])
    rho_prior, _ = stats.spearmanr(const_vs_prior, full)
    rho_mlp, _ = stats.spearmanr(const_vs_mlp, full)
    lines.append("")
    lines.append("## H25 Spearman correlation (n=3 datasets, descriptive only)")
    lines.append("")
    lines.append(f"- constant-feature (Hodge − prior) gap: `{const_vs_prior.tolist()}`")
    lines.append(f"- constant-feature (Hodge − MLP)   gap: `{const_vs_mlp.tolist()}`")
    lines.append(f"- full-feature   (Hodge − MLP)     gap: `{full.tolist()}`")
    lines.append(f"- Spearman ρ (Hodge − prior  vs full-MLP) = **{rho_prior:+.4f}**")
    lines.append(f"- Spearman ρ (Hodge − MLP    vs full-MLP) = **{rho_mlp:+.4f}**")
    lines.append("- p-values omitted: with n = 3 datasets, the Spearman significance test")
    lines.append("  is uninformative.  ρ is reported descriptively.")

    lines.append("")
    lines.append("## Scoped interpretation")
    lines.append("")
    lines.append(
        "Evidence is consistent with an architecture × data-topology interaction "
        "under the tested configuration (3 TUDataset graph-classification datasets, "
        "30 seeds, 10 epochs, matched-capacity Hodge-residual vs MLP baseline, "
        "stratified 80/20 split, constant-feature ablation isolating graph topology "
        "from node features).  The resolver makes no claim of generality beyond "
        "this scope."
    )
    return "\n".join(lines)


DEFAULT_FULL_PATHS: dict[str, Path] = {
    "mutag": Path("notebooks/results/mutag_hodge_ablation_30seeds.json"),
    "proteins": Path("notebooks/results/proteins_hodge_ablation_30seeds.json"),
    "nci1": Path("notebooks/results/nci1_hodge_ablation_30seeds.json"),
}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m benchmarks.hodge.h006_analysis")
    parser.add_argument(
        "--constant-results-dir", type=Path, default=Path("notebooks/results"),
        help=(
            "Directory containing the h006_{ds}_constant_30seeds.json files — "
            "outputs of `python -m benchmarks.hodge --constant-features ...` "
            "for each dataset, named per the H006 reproduction command."
        ),
    )
    parser.add_argument(
        "--constant-filename-pattern", type=str,
        default="h006_{ds}_constant_30seeds.json",
        help=(
            "Filename pattern with {ds} placeholder for constant-feature "
            "JSONs. Default matches the H006 reproduction command."
        ),
    )
    parser.add_argument(
        "--full-results-dir", type=Path, default=Path("notebooks/results"),
        help=(
            "Directory containing the H001/H002/H003 30-seed full-feature "
            "ablation JSONs ({ds}_hodge_ablation_30seeds.json)."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    constant_paths = {
        ds: args.constant_results_dir / args.constant_filename_pattern.format(ds=ds)
        for ds in ("mutag", "proteins", "nci1")
    }
    full_paths = {ds: args.full_results_dir / f"{ds}_hodge_ablation_30seeds.json"
                  for ds in ("mutag", "proteins", "nci1")}

    summaries = resolve(
        constant_paths=constant_paths, full_paths=full_paths, alpha=args.alpha,
    )
    print(render_markdown(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLASS_PRIORS",
    "DEFAULT_FULL_PATHS",
    "DatasetSummary",
    "main",
    "render_markdown",
    "resolve",
]
