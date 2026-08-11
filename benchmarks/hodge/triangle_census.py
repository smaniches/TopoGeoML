"""
Triangle census for Hodge-bench datasets.

H011b's preregistration requires the final confirmatory artifact to
report "the triangle census actually observed under the exact dataset
loader and preprocessing used for the experiment"
(``docs/hypotheses/HYPOTHESIS-011b-l1-collab.md``). No benchmark code
produced that census before this module existed.

The census reconstructs each graph from the ``GraphSample.laplacian``
stored by the dataset adapter — the same reconstruction the
``l1-hodge-residual`` model performs in ``_compute_l1`` — so the counted
edge set is exactly the edge set the experiment's models observe. Triangle
counts use ``networkx.triangles`` (each triangle counted once).

Usage::

    python -m benchmarks.hodge.triangle_census --dataset collab \
        --output notebooks/results/h011b_collab_l1_30seeds.census.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.hodge.datasets import REGISTERED as DATASETS
from benchmarks.hodge.datasets import GraphSample

CENSUS_SCHEMA_VERSION = "hodge-triangle-census-1.0.0"


def edges_from_l0(laplacian: torch.Tensor) -> list[tuple[int, int]]:
    """Undirected edge list reconstructed from a stored L_0 Laplacian.

    Mirrors the reconstruction in
    ``benchmarks.hodge.models._L1HodgeResidualGraphClassifier._compute_l1``:
    every off-diagonal coordinate of the sparse L_0 is an edge endpoint
    pair; duplicates (the symmetric entry) collapse to one undirected edge.
    """
    L = laplacian.coalesce()
    indices = L.indices()
    off_diag = indices[0] != indices[1]
    src = indices[0][off_diag].tolist()
    dst = indices[1][off_diag].tolist()
    seen: set[tuple[int, int]] = set()
    for s, d in zip(src, dst, strict=True):
        edge = (min(s, d), max(s, d))
        seen.add(edge)
    return sorted(seen)


def triangle_count(n_nodes: int, laplacian: torch.Tensor) -> int:
    """Number of triangles in the graph reconstructed from a stored L_0."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges_from_l0(laplacian))
    return int(sum(nx.triangles(g).values())) // 3


def census_from_samples(samples: list[GraphSample]) -> dict[str, Any]:
    """Per-graph triangle census plus the summary H011b's audit requires."""
    counts = [triangle_count(s.x.shape[0], s.laplacian) for s in samples]
    arr = np.asarray(counts, dtype=np.int64)
    n_with = int((arr > 0).sum())
    summary: dict[str, Any] = {
        "n_graphs": int(arr.size),
        "n_graphs_with_triangles": n_with,
        "fraction_with_triangles": float(n_with / arr.size) if arr.size else 0.0,
        "total_triangles": int(arr.sum()),
        "triangles_min": int(arr.min()) if arr.size else 0,
        "triangles_median": float(np.median(arr)) if arr.size else 0.0,
        "triangles_mean": float(arr.mean()) if arr.size else 0.0,
        "triangles_p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "triangles_max": int(arr.max()) if arr.size else 0,
    }
    return {"summary": summary, "per_graph_triangles": counts}


def run_census(dataset_name: str) -> dict[str, Any]:
    """Load a registered dataset through its adapter and produce the census."""
    from benchmarks.hodge.runner import _get_git_sha

    dataset = DATASETS[dataset_name]
    samples, input_dim, num_classes = dataset.load()
    census = census_from_samples(samples)
    return {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "git_commit_sha": _get_git_sha(),
        **census,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.hodge.triangle_census",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = run_census(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(args.output)

    s = result["summary"]
    print(
        f"{result['dataset']}: {s['n_graphs_with_triangles']}/{s['n_graphs']} graphs "
        f"({100.0 * s['fraction_with_triangles']:.1f}%) contain triangles; "
        f"median {s['triangles_median']:.0f}, max {s['triangles_max']}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
