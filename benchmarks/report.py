"""
Report rendering — turn a ``RunResult`` JSON into Markdown for humans.

Two rendering rules, strictly enforced:

  1. Never assert a directional claim ("backend A is faster than B")
     without a statistical-significance label. If the BH-adjusted p-value
     is not below alpha, the report says "no significant difference".
     If the comparison is underpowered (n < threshold) the report says
     "preliminary — n=X seeds, increase to ≥ 20 to claim significance".

  2. Always surface caveats: hardware, CPU-only-vs-GPU, sample size.
     A reader who quotes the table out of context should still be able
     to tell whether the result is robust.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.stats import benjamini_hochberg, compare_paired


def _provenance_block(prov: dict[str, Any]) -> list[str]:
    lines = ["## Provenance", ""]
    lines.append(f"- Schema: `{prov.get('schema_version', '?')}`")
    lines.append(f"- Timestamp (UTC): {prov.get('timestamp_utc', '?')}")
    git_dirty = " (dirty)" if prov.get("git_dirty") else ""
    sha = prov.get("git_sha") or "(not a git checkout)"
    lines.append(f"- Git SHA: `{sha}`{git_dirty}")
    lines.append(f"- Python: {prov.get('python_version', '?')}")
    lines.append(f"- PyTorch: {prov.get('torch_version', '?')}")
    lines.append(f"- TopoGeoML: {prov.get('topogeoml_version', '?')}")
    lines.append(f"- torch-topological: {prov.get('torch_topological_version', '?')}")
    lines.append(f"- numpy: {prov.get('numpy_version', '?')}, scipy: {prov.get('scipy_version', '?')}")
    lines.append(f"- Platform: {prov.get('platform_string', '?')}")
    lines.append(f"- CPU count: {prov.get('cpu_count', '?')} · "
                 f"System memory: {prov.get('process_memory_total_mb', '?')} MiB")
    lines.append(
        "- Determinism: "
        + ("enabled" if prov.get("deterministic_algorithms_set") else "disabled")
    )
    lines.append("")
    return lines


def _correctness_section(cells: list[dict[str, Any]]) -> list[str]:
    relevant = [c for c in cells if c["axis_name"] == "correctness" and c["success"]]
    if not relevant:
        return []
    out = ["## Correctness (vs ripser reference)", ""]
    out.append("Pass = every per-seed finite-bar diagram matches ripser to ``atol`` and "
               "the backend preserves ``float64`` dtype through the call.")
    out.append("")
    out.append("| Backend | Dataset | n_points | atol | Max H_0 diff | Max H_1 diff | Pass |")
    out.append("|---|---|---|---|---|---|---|")
    for cell in relevant:
        p = cell["payload"]
        per_seed = p.get("per_seed", [])
        if not per_seed:
            continue
        max_h0 = max((s["max_abs_diff_h0"] for s in per_seed), default=float("nan"))
        max_h1 = max((s["max_abs_diff_h1"] for s in per_seed), default=float("nan"))
        verdict = "PASS" if p.get("overall_pass") else "FAIL"
        out.append(
            f"| `{cell['backend_name']}` | `{cell['dataset_name']}` "
            f"| {p.get('n_points', '?')} | {p.get('atol', '?'):.0e} "
            f"| {max_h0:.2e} | {max_h1:.2e} | **{verdict}** |"
        )
    out.append("")
    return out


def _stability_section(cells: list[dict[str, Any]]) -> list[str]:
    relevant = [c for c in cells if c["axis_name"] == "stability" and c["success"]]
    if not relevant:
        return []
    out = ["## Stability (Cohen-Steiner + gradient Lipschitz + gradcheck)", ""]
    out.append(
        "Three sub-measurements:"
        "  (1) Cohen-Steiner theorem violations — should be 0 for any correct backend."
        "  (2) Gradient Lipschitz approximation (median + 95% bootstrap CI) — lower is better."
        "  (3) Autograd ``gradcheck`` pass rate against finite-difference reference."
    )
    out.append("")
    out.append("| Backend | Dataset | n_points | CS violations | Lipschitz (median, 95% CI) | gradcheck pass |")
    out.append("|---|---|---|---|---|---|")
    for cell in relevant:
        p = cell["payload"]
        cs_v = p.get("n_theorem_violations", "?")
        lip_med = p.get("lipschitz_median", float("nan"))
        lip_lo = p.get("lipschitz_ci95_low", float("nan"))
        lip_hi = p.get("lipschitz_ci95_high", float("nan"))
        gc_rate = p.get("gradcheck_pass_rate", float("nan"))
        if not np.isnan(lip_lo) and not np.isnan(lip_hi):
            lip_str = f"{lip_med:.4f} [{lip_lo:.4f}, {lip_hi:.4f}]"
        else:
            lip_str = f"{lip_med:.4f} (CI not available, n<2)"
        out.append(
            f"| `{cell['backend_name']}` | `{cell['dataset_name']}` "
            f"| {p.get('n_points', '?')} | {cs_v} | {lip_str} "
            f"| {gc_rate*100:.0f}% |"
        )
    out.append("")
    return out


def _speed_section(cells: list[dict[str, Any]]) -> list[str]:
    """Per-(n,operation) paired comparison across backends."""
    relevant = [c for c in cells if c["axis_name"] == "speed" and c["success"]]
    if not relevant:
        return []
    out = ["## Speed (forward and forward+backward)", ""]
    out.append(
        "Each cell reports the min-of-medians across 5 outer measurement passes "
        "(20 inner calls per pass), with GC disabled during the measurement window. "
        "When two backends are present, a paired Wilcoxon signed-rank test (with "
        "BH multi-hypothesis correction across all (n, operation) comparisons) "
        "determines significance."
    )
    out.append("")

    # Group by backend → list of (n, seed, operation, median_ms)
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for cell in relevant:
        rows = cell["payload"].get("rows", [])
        by_backend[cell["backend_name"]] = rows

    backend_names = sorted(by_backend)
    if len(backend_names) < 1:
        return []

    # Single-backend reporting — just the raw table.
    if len(backend_names) == 1:
        b = backend_names[0]
        out.append("Single backend in this run; no comparison reported.")
        out.append("")
        out.append("| backend | n | seed | operation | median (ms) | 95% CI (ms) |")
        out.append("|---|---|---|---|---|---|")
        for r in by_backend[b]:
            out.append(
                f"| `{b}` | {r['n_points']} | {r['seed']} | {r['operation']} "
                f"| {r['point_estimate_ms']:.3f} "
                f"| [{r['ci95_low_ms']:.3f}, {r['ci95_high_ms']:.3f}] |"
            )
        out.append("")
        return out

    # Multi-backend: do paired comparisons per (n, operation) cell.
    out.append("| n | operation | "
               + " | ".join(f"{b} (median ms)" for b in backend_names)
               + " | comparison | p_raw | p_BH | effect (r) |")
    out.append("|---|---|" + "|".join(["---"] * len(backend_names)) + "|---|---|---|---|")

    raw_comparisons = []
    for n in sorted({r["n_points"] for r in by_backend[backend_names[0]]}):
        for op in sorted({r["operation"] for r in by_backend[backend_names[0]]}):
            seeds_a = {
                r["seed"]: r for r in by_backend[backend_names[0]]
                if r["n_points"] == n and r["operation"] == op
            }
            seeds_b = {
                r["seed"]: r for r in by_backend[backend_names[1]]
                if r["n_points"] == n and r["operation"] == op
            }
            common = sorted(set(seeds_a) & set(seeds_b))
            if not common:
                continue
            arm_a = np.array([seeds_a[s]["point_estimate_ms"] for s in common])
            arm_b = np.array([seeds_b[s]["point_estimate_ms"] for s in common])
            cmp = compare_paired(
                arm_a, arm_b,
                arm_a_name=backend_names[0],
                arm_b_name=backend_names[1],
            )
            raw_comparisons.append((n, op, cmp))

    if raw_comparisons:
        family = benjamini_hochberg([c for _, _, c in raw_comparisons], alpha=0.05)
        for (n, op, _), cmp in zip(raw_comparisons, family.comparisons, strict=True):
            med_a = cmp.median_a
            med_b = cmp.median_b
            label = {
                "significant": f"**{cmp.arm_a_name} ≠ {cmp.arm_b_name}**",
                "not_significant": "no diff",
                "underpowered": "underpowered",
            }.get(cmp.kind.value if hasattr(cmp.kind, "value") else cmp.kind, "?")
            p_raw_str = f"{cmp.p_value_raw:.3e}" if not np.isnan(cmp.p_value_raw) else "—"
            p_bh_str = (
                f"{cmp.p_value_bh_adjusted:.3e}"
                if cmp.p_value_bh_adjusted is not None else "—"
            )
            out.append(
                f"| {n} | {op} | {med_a:.3f} | {med_b:.3f} "
                f"| {label} | {p_raw_str} | {p_bh_str} | {cmp.effect_size:.3f} |"
            )
    out.append("")
    out.append(
        f"_Family of {len(raw_comparisons)} paired comparisons; "
        f"{family.n_significant_after_bh if raw_comparisons else 0} significant after "
        "Benjamini-Hochberg FDR correction at α=0.05._"
    )
    out.append("")
    return out


def _optimization_section(cells: list[dict[str, Any]]) -> list[str]:
    relevant = [c for c in cells if c["axis_name"] == "optimization" and c["success"]]
    if not relevant:
        return []
    out = ["## Optimization (descent on longest-H_1 inflation)", ""]
    out.append(
        "Diagnostic axis. Subgradient choices may legitimately differ across "
        "backends (Hofer 2017; Carrière 2021), so this is not a ranking."
    )
    out.append("")
    out.append("| Backend | Dataset | Objective | Final loss (median, 95% CI) | n_steps | lr |")
    out.append("|---|---|---|---|---|---|")
    for cell in relevant:
        p = cell["payload"]
        med = p.get("final_loss_median", float("nan"))
        lo = p.get("final_loss_ci95_low", float("nan"))
        hi = p.get("final_loss_ci95_high", float("nan"))
        if not np.isnan(lo) and not np.isnan(hi):
            ci_str = f"{med:.4f} [{lo:.4f}, {hi:.4f}]"
        else:
            ci_str = f"{med:.4f} (CI n/a)"
        out.append(
            f"| `{cell['backend_name']}` | `{cell['dataset_name']}` "
            f"| `{p.get('objective', '?')}` | {ci_str} "
            f"| {p.get('n_steps', '?')} | {p.get('learning_rate', '?')} |"
        )
    out.append("")
    return out


def _failures_section(cells: list[dict[str, Any]]) -> list[str]:
    failures = [c for c in cells if not c["success"]]
    if not failures:
        return []
    out = ["## Failures", ""]
    for c in failures:
        out.append(
            f"- `{c['backend_name']}` / `{c['dataset_name']}` / `{c['axis_name']}`: "
            f"`{c['error_kind']}` — {c['error_message']}"
        )
    out.append("")
    return out


def render_markdown(run_result_json: dict[str, Any]) -> str:
    """Render a complete markdown report from a deserialized RunResult."""
    prov = run_result_json.get("provenance", {})
    cells = run_result_json.get("cells", [])

    lines: list[str] = []
    lines.append("# TopoGeoML benchmark — diff-PH comparison")
    lines.append("")
    lines.append(
        "Statistical claims in this report obey the framework's reporting "
        "rule: directional differences are reported only when a "
        "Benjamini-Hochberg-adjusted p-value is below α=0.05. Otherwise "
        "the entry reads ``no diff`` or ``underpowered``."
    )
    lines.append("")
    lines.extend(_provenance_block(prov))
    lines.extend(_correctness_section(cells))
    lines.extend(_stability_section(cells))
    lines.extend(_speed_section(cells))
    lines.extend(_optimization_section(cells))
    lines.extend(_failures_section(cells))
    return "\n".join(lines)


def render_from_file(path: Path) -> str:
    """Convenience: read a JSON run-result file and render markdown."""
    data = json.loads(path.read_text())
    return render_markdown(data)


__all__ = ["render_from_file", "render_markdown"]
