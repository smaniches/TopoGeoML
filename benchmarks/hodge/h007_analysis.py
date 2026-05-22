"""H007 structural-signal decomposition.

For each of MUTAG, PROTEINS, NCI1 and each of five graph-structural proxies
(size, degree, WL subtree, cycle, normalised Laplacian spectrum), compute the
per-class separability metric and ask whether any proxy's dataset-by-dataset
separability explains the H006 constant-feature gap or the full-feature
Hodge-vs-MLP gain.

Claim discipline (per the PR scope contract):

* Every proxy is described as a *graph-structural proxy*, not a "topology
  mechanism".  Only the cycle-basis-size proxy specifically isolates a
  topological invariant (β₁); the rest may co-vary with topology but do
  not isolate it.
* No causal claim is asserted.  The Spearman correlations are reported
  descriptively because n=3 datasets carries no inferential power.
* No new model architecture, no leaderboard update.

The script is fully deterministic — no seeds; the same input produces the
same output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# H006 anchor numbers — read these from the artifacts to verify, but for the
# Spearman correlation we only need the dataset-by-dataset ordering of the
# gaps.  Values are committed in the resolution commit `073b253`.
H006_CONST_FEATURE_GAP: dict[str, float] = {
    "mutag": 0.0983,
    "proteins": 0.0882,
    "nci1": 0.0707,
}

H006_FULL_FEATURE_GAIN: dict[str, float] = {
    "mutag": -0.0395,
    "proteins": 0.0112,
    "nci1": 0.0864,
}


@dataclass(frozen=True)
class ProxyResult:
    """Per-(dataset, proxy) separability summary."""

    proxy_name: str
    dataset_name: str
    feature_dim: int
    n_samples: int
    class_distribution: dict[int, int]
    per_component_separability: list[float]
    max_separability: float
    best_component_idx: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "proxy_name": self.proxy_name,
            "dataset_name": self.dataset_name,
            "feature_dim": self.feature_dim,
            "n_samples": self.n_samples,
            "class_distribution": {str(k): v for k, v in self.class_distribution.items()},
            "per_component_separability": self.per_component_separability,
            "max_separability": self.max_separability,
            "best_component_idx": self.best_component_idx,
        }


@dataclass(frozen=True)
class CorrelationRow:
    """Per-proxy correlation across the three datasets."""

    proxy_name: str
    separability_by_dataset: dict[str, float]
    spearman_rho_vs_const_gap: float
    spearman_rho_vs_full_gain: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "proxy_name": self.proxy_name,
            "separability_by_dataset": self.separability_by_dataset,
            "spearman_rho_vs_const_gap": self.spearman_rho_vs_const_gap,
            "spearman_rho_vs_full_gain": self.spearman_rho_vs_full_gain,
        }


# ---------------------------------------------------------------------------
# Graph → networkx adapter.  All proxies operate on networkx graphs so the
# downstream code is dataset-independent.
# ---------------------------------------------------------------------------


def _samples_to_nx(samples: list, dataset_name: str) -> list:
    """Convert a list of GraphSample (from benchmarks.hodge.datasets) to
    (networkx.Graph, int label) pairs.

    The Laplacian is symmetric so we read it once, set the diagonal to zero,
    and treat any non-zero entry as an edge (off-diagonal entries of L = D − A
    are exactly −A_ij for i ≠ j).
    """
    import networkx as nx

    out: list[tuple[Any, int]] = []
    for sample in samples:
        L_dense = sample.laplacian.to_dense().numpy()
        # Off-diagonal entries are non-zero exactly on edges.
        np.fill_diagonal(L_dense, 0.0)
        # L is symmetric; we want |L_ij| > 0 → edge.
        adj = (np.abs(L_dense) > 1e-12).astype(np.int64)
        g = nx.from_numpy_array(adj)
        out.append((g, int(sample.y)))
    del dataset_name
    return out


# ---------------------------------------------------------------------------
# Per-graph proxies.  Each returns a fixed-length 1-D numpy array of float64.
# ---------------------------------------------------------------------------


def compute_size_features(graph: Any) -> np.ndarray:
    """Graph-size proxy: 1-D scalar = number of nodes."""
    return np.asarray([float(graph.number_of_nodes())], dtype=np.float64)


def compute_degree_features(graph: Any) -> np.ndarray:
    """Degree-distribution proxy: 5-D vector
    [mean_degree, max_degree, std_degree, n_isolated, edge_density]."""
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros(5, dtype=np.float64)
    degs = np.asarray([d for _, d in graph.degree()], dtype=np.float64)
    n_isolated = float(np.sum(degs == 0))
    n_edges = graph.number_of_edges()
    density = (2.0 * n_edges) / (n * (n - 1)) if n > 1 else 0.0
    return np.asarray([
        float(degs.mean()),
        float(degs.max()),
        float(degs.std()),
        n_isolated,
        density,
    ], dtype=np.float64)


def compute_wl_features(graph: Any, n_iter: int = 2, n_buckets: int = 32) -> np.ndarray:
    """Weisfeiler-Lehman subtree histogram, bucketed to ``n_buckets`` dims.

    Iteration 0 starts every node at the same label.  At iteration k, each
    node's label becomes ``hash((self_label, sorted(neighbour_labels)))``.
    Final per-graph feature: 32-bucket histogram of all labels at all
    iterations 1..n_iter, normalised by total label count.
    """
    if graph.number_of_nodes() == 0:
        return np.zeros(n_buckets, dtype=np.float64)
    labels = {v: 0 for v in graph.nodes()}
    histogram = np.zeros(n_buckets, dtype=np.float64)
    for _ in range(n_iter):
        new_labels: dict[Any, int] = {}
        for v in graph.nodes():
            neigh = tuple(sorted(labels[u] for u in graph.neighbors(v)))
            new_labels[v] = hash((labels[v], neigh)) & 0xFFFF
        labels = new_labels
        for lbl in labels.values():
            histogram[lbl % n_buckets] += 1.0
    total = histogram.sum()
    if total > 0:
        histogram = histogram / total
    return histogram


def compute_cycle_features(graph: Any) -> np.ndarray:
    """Cycle-structure proxy: 4-D vector
    [n_cycles_basis, mean_cycle_length, n_triangles, n_4cycles].

    `n_cycles_basis` equals the rank of the first homology group H₁ (β₁) of
    the graph viewed as a 1-complex — this IS a topological invariant.
    The other three components are local cycle counts (3- and 4-cycles).
    """
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return np.zeros(4, dtype=np.float64)
    # Cycle basis: NX uses Paton's algorithm; runs on simple undirected graphs.
    try:
        basis = nx.cycle_basis(graph)
    except nx.NetworkXNotImplemented:  # pragma: no cover — only for directed
        basis = []
    n_basis = len(basis)
    mean_len = float(np.mean([len(c) for c in basis])) if n_basis > 0 else 0.0
    # Count triangles via NX's specialised counter; sum over nodes / 3.
    triangles = sum(nx.triangles(graph).values()) // 3
    # Count 4-cycles by enumerating cycles of length 4 in the basis.  This
    # undercounts because the basis is not the set of all simple cycles, but
    # the basis-rank component captures the global structure; per-graph
    # 4-cycle count from basis is a deterministic proxy.
    n4 = sum(1 for c in basis if len(c) == 4)
    return np.asarray([float(n_basis), mean_len, float(triangles), float(n4)],
                      dtype=np.float64)


def compute_spectral_features(laplacian, k: int = 5) -> np.ndarray:
    """Top-k eigenvalues of the symmetrically-normalised Laplacian.

    For a graph with combinatorial Laplacian L = D − A, the normalised
    Laplacian is L̃ = D^{-1/2} L D^{-1/2}.  Eigenvalues lie in [0, 2] and
    are well-defined invariants of the graph's connectivity structure.
    """
    import torch

    L_dense = laplacian.to_dense().numpy().astype(np.float64)
    n = L_dense.shape[0]
    if n == 0:
        return np.zeros(k, dtype=np.float64)
    deg = np.diag(L_dense).copy()
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 1e-12
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    D = np.diag(deg_inv_sqrt)
    Lt = D @ L_dense @ D
    # Symmetrise to kill round-off.
    Lt = 0.5 * (Lt + Lt.T)
    eigs = np.linalg.eigvalsh(Lt)
    # Top-k descending.
    eigs_sorted = np.sort(eigs)[::-1]
    out = np.zeros(k, dtype=np.float64)
    out[: min(k, len(eigs_sorted))] = eigs_sorted[:k]
    del torch  # only needed for type compatibility with caller
    return out


# ---------------------------------------------------------------------------
# Per-class separability.
# ---------------------------------------------------------------------------


def class_separability(features: np.ndarray, labels: np.ndarray) -> tuple[list[float], int]:
    """For each column of ``features``, compute |rank-biserial r| =
    |2U/(n₁n₂) − 1| from the Mann-Whitney U test between the two classes.

    Returns ``(per_component_separability, best_component_idx)``.  The metric
    is in [0, 1] where 0 = chance, 1 = perfect class separation by that
    1-D feature.
    """
    if features.ndim == 1:
        features = features[:, None]
    classes = sorted(set(labels.tolist()))
    if len(classes) != 2:
        raise ValueError(
            f"H007 separability requires exactly 2 classes; got {classes}"
        )
    mask0 = labels == classes[0]
    mask1 = labels == classes[1]
    n0 = int(mask0.sum())
    n1 = int(mask1.sum())
    if n0 == 0 or n1 == 0:
        return [0.0] * features.shape[1], 0
    per_comp: list[float] = []
    for j in range(features.shape[1]):
        x0 = features[mask0, j]
        x1 = features[mask1, j]
        # Mann-Whitney U handles constants (including identical constants)
        # correctly: identical constants give U = n0*n1/2 → r = 0; disjoint
        # constants give U = 0 or n0*n1 → |r| = 1.  No early-return needed.
        u = stats.mannwhitneyu(x0, x1, alternative="two-sided").statistic
        r = 2.0 * u / (n0 * n1) - 1.0
        per_comp.append(float(abs(r)))
    best = int(np.argmax(per_comp))
    return per_comp, best


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------


PROXY_NAMES = ("size", "degree", "wl", "cycle", "spectral")


def _compute_proxy_features(graphs_with_labels: list, samples: list, proxy_name: str
                            ) -> np.ndarray:
    """Compute the feature matrix (n_samples, feature_dim) for one proxy."""
    if proxy_name == "size":
        rows = [compute_size_features(g) for g, _ in graphs_with_labels]
    elif proxy_name == "degree":
        rows = [compute_degree_features(g) for g, _ in graphs_with_labels]
    elif proxy_name == "wl":
        rows = [compute_wl_features(g) for g, _ in graphs_with_labels]
    elif proxy_name == "cycle":
        rows = [compute_cycle_features(g) for g, _ in graphs_with_labels]
    elif proxy_name == "spectral":
        rows = [compute_spectral_features(s.laplacian) for s in samples]
    else:
        raise ValueError(f"unknown proxy {proxy_name!r}")
    return np.vstack(rows).astype(np.float64)


def run_h007_analysis(
    *, dataset_names: tuple[str, ...] = ("mutag", "proteins", "nci1"),
) -> dict[str, Any]:
    """Compute per-dataset, per-proxy separability and the cross-dataset
    correlation table.

    Returns a JSON-serialisable dict with two top-level keys:
    ``per_dataset_proxy_results`` and ``correlation_table``.
    """
    from benchmarks.hodge.datasets import REGISTERED as DATASETS

    per_dataset: list[ProxyResult] = []
    for ds_name in dataset_names:
        ds = DATASETS[ds_name]
        if not ds.available():  # pragma: no cover
            continue
        samples, _, _ = ds.load()
        graphs = _samples_to_nx(samples, ds_name)
        labels_arr = np.asarray([y for _, y in graphs], dtype=np.int64)
        class_dist: dict[int, int] = {}
        for y in labels_arr.tolist():
            class_dist[y] = class_dist.get(y, 0) + 1
        for proxy_name in PROXY_NAMES:
            features = _compute_proxy_features(graphs, samples, proxy_name)
            per_comp, best = class_separability(features, labels_arr)
            per_dataset.append(ProxyResult(
                proxy_name=proxy_name,
                dataset_name=ds_name,
                feature_dim=features.shape[1],
                n_samples=features.shape[0],
                class_distribution=class_dist,
                per_component_separability=per_comp,
                max_separability=max(per_comp) if per_comp else 0.0,
                best_component_idx=best,
            ))

    # Correlation table.  For each proxy, gather max-separability per dataset
    # and Spearman-correlate against (a) H006 const-feature gap and (b)
    # H006 full-feature gain.  Descriptive only (n=3).
    correlations: list[CorrelationRow] = []
    for proxy_name in PROXY_NAMES:
        sep_by_ds = {
            r.dataset_name: r.max_separability
            for r in per_dataset if r.proxy_name == proxy_name
        }
        ordered_datasets = [d for d in dataset_names if d in sep_by_ds]
        sep_arr = np.asarray([sep_by_ds[d] for d in ordered_datasets])
        const_arr = np.asarray([H006_CONST_FEATURE_GAP[d] for d in ordered_datasets])
        full_arr = np.asarray([H006_FULL_FEATURE_GAIN[d] for d in ordered_datasets])
        rho_const, _ = stats.spearmanr(sep_arr, const_arr) if len(sep_arr) >= 2 else (float("nan"), float("nan"))
        rho_full, _ = stats.spearmanr(sep_arr, full_arr) if len(sep_arr) >= 2 else (float("nan"), float("nan"))
        correlations.append(CorrelationRow(
            proxy_name=proxy_name,
            separability_by_dataset={k: float(v) for k, v in sep_by_ds.items()},
            spearman_rho_vs_const_gap=float(rho_const),
            spearman_rho_vs_full_gain=float(rho_full),
        ))

    return {
        "schema_version": "h007-1.0.0",
        "h006_const_feature_gap": H006_CONST_FEATURE_GAP,
        "h006_full_feature_gain": H006_FULL_FEATURE_GAIN,
        "per_dataset_proxy_results": [r.as_dict() for r in per_dataset],
        "correlation_table": [c.as_dict() for c in correlations],
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render the H007 result as Markdown."""
    lines: list[str] = []
    lines.append("# H007 structural-signal decomposition")
    lines.append("")
    lines.append("## Per-(dataset × proxy) class separability (max |rank-biserial r|)")
    lines.append("")
    lines.append("| Dataset | Proxy | Feature dim | Max separability | Best component idx |")
    lines.append("|---|---|---|---|---|")
    for r in result["per_dataset_proxy_results"]:
        lines.append(
            f"| {r['dataset_name']} | {r['proxy_name']} | {r['feature_dim']} | "
            f"{r['max_separability']:.4f} | {r['best_component_idx']} |"
        )
    lines.append("")
    lines.append("## Cross-dataset correlation (n=3, descriptive only)")
    lines.append("")
    lines.append("| Proxy | mutag | proteins | nci1 | ρ vs H006 const-gap | ρ vs H006 full-gain |")
    lines.append("|---|---|---|---|---|---|")
    for c in result["correlation_table"]:
        sep = c["separability_by_dataset"]
        lines.append(
            f"| {c['proxy_name']} | {sep.get('mutag', float('nan')):.4f} | "
            f"{sep.get('proteins', float('nan')):.4f} | "
            f"{sep.get('nci1', float('nan')):.4f} | "
            f"{c['spearman_rho_vs_const_gap']:+.4f} | "
            f"{c['spearman_rho_vs_full_gain']:+.4f} |"
        )
    lines.append("")
    lines.append("## Scoped interpretation")
    lines.append("")
    lines.append(
        "Each entry above is a *graph-structural proxy*.  The cycle-basis-"
        "size component of the `cycle` proxy is the only entry that "
        "specifically isolates a topological invariant (β₁, the rank of "
        "the first homology group).  The other proxies (size, degree, "
        "WL subtree, spectral) may co-vary with topology but do not "
        "isolate it.  With n=3 datasets, the Spearman ρ values are "
        "reported descriptively only — they carry no inferential power."
    )
    return "\n".join(lines)


def write_result(result: dict[str, Any], path: Path) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m benchmarks.hodge.h007_analysis")
    parser.add_argument(
        "--output", type=Path,
        default=Path("notebooks/results/h007_structural_decomposition.json"),
    )
    parser.add_argument(
        "--markdown", type=Path,
        default=Path("notebooks/results/h007_structural_decomposition.md"),
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args(argv)

    dataset_names = tuple(args.datasets) if args.datasets else ("mutag", "proteins", "nci1")
    result = run_h007_analysis(dataset_names=dataset_names)
    write_result(result, args.output)
    md = render_markdown(result)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "H006_CONST_FEATURE_GAP",
    "H006_FULL_FEATURE_GAIN",
    "PROXY_NAMES",
    "CorrelationRow",
    "ProxyResult",
    "class_separability",
    "compute_cycle_features",
    "compute_degree_features",
    "compute_size_features",
    "compute_spectral_features",
    "compute_wl_features",
    "main",
    "render_markdown",
    "run_h007_analysis",
    "write_result",
]
