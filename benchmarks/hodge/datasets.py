"""
TUDataset adapters for the Hodge bench.

Each adapter loads a graph-classification dataset from PyG's TUDataset
collection, converts each graph to a (node-features, Hodge-Laplacian,
label) triple, and caches the dataset under
``$XDG_CACHE_HOME/topogeoml/benchmarks/hodge``.

References
----------
Morris, C., Kriege, N. M., Bause, F., et al. (2020). TUDataset.
  *ICML 2020 GRL+ workshop*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class GraphSample:
    """One graph: node features, Hodge L_0 (sparse tensor), label."""

    x: torch.Tensor
    laplacian: torch.Tensor  # sparse_coo
    y: int


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    root = Path(base) / "topogeoml" / "benchmarks" / "hodge"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _graph_to_laplacian(
    n_nodes: int, edge_index: torch.Tensor
) -> torch.Tensor:
    """Compute the L_0 Hodge Laplacian of a graph as a sparse torch tensor.

    Uses ``topogeoml.data.graph_to_clique_complex`` (max_dim=1) +
    ``topogeoml.core.hodge_laplacian``. The result is the standard
    combinatorial graph Laplacian D - A under the discrete Hodge theory
    framing.
    """
    import networkx as nx

    from topogeoml.core.complexes import hodge_laplacian
    from topogeoml.data.graph_to_complex import graph_to_clique_complex
    from topogeoml.nn.hodge import sparse_scipy_to_torch

    # Build a NetworkX graph including all nodes (even isolated ones).
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    if edge_index.numel() > 0:
        edges = edge_index.t().tolist()
        g.add_edges_from((int(u), int(v)) for u, v in edges)

    sc = graph_to_clique_complex(g, max_dim=1, include_isolated_vertices=True)
    L = hodge_laplacian(sc, k=0)
    # We need an (n_nodes, n_nodes) Laplacian. SimplicialComplex assigns indices
    # in lexicographic order; for vertices 0..n-1 this is just the natural order.
    return sparse_scipy_to_torch(L, dtype=torch.float64)


@dataclass(frozen=True)
class MUTAGDataset:
    """MUTAG: 188 molecular graphs, 2 classes (mutagenicity).

    Citation: Debnath et al. 1991; Morris et al. 2020 TUDataset.
    """

    name: str = "mutag"
    version: str = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import torch_geometric  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    def load(self) -> tuple[list[GraphSample], int, int]:
        """Load and convert MUTAG. Returns (samples, input_dim, num_classes)."""
        return _load_tudataset("MUTAG")


@dataclass(frozen=True)
class PROTEINSDataset:
    """PROTEINS: 1113 protein graphs, 2 classes (enzyme vs non-enzyme).

    Nodes are secondary-structure elements (helix/sheet/turn, 3-dim
    one-hot or 32-dim continuous depending on the PyG release). Average
    graph size: 39 nodes, 73 edges. Roughly 6x larger than MUTAG by
    graph count, putting it above the discrimination ceiling that
    Errica et al. 2020 flagged for MUTAG.

    Citation: Borgwardt et al. 2005, *Bioinformatics* 21; Dobson &
    Doig 2003, *J. Mol. Biol.* 330; Morris et al. 2020 TUDataset.
    """

    name: str = "proteins"
    version: str = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import torch_geometric  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        return True

    def load(self) -> tuple[list[GraphSample], int, int]:
        """Load and convert PROTEINS. Returns (samples, input_dim, num_classes)."""
        return _load_tudataset("PROTEINS")


def _load_tudataset(name: str) -> tuple[list[GraphSample], int, int]:
    """Generic TUDataset → list[GraphSample] adapter.

    Shared between every dataset class so the loader logic is
    centralised. Handles the case where ``num_node_features == 0`` by
    falling back to an identity-matrix feature (one-hot per node), and
    the case where ``num_node_features > 0`` by casting to float64 to
    match the downstream Hodge propagation dtype.
    """
    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root=str(_cache_root()), name=name)
    samples: list[GraphSample] = []
    for g in ds:
        x = (
            g.x.to(torch.float64)
            if g.x is not None
            else torch.eye(g.num_nodes, dtype=torch.float64)
        )
        L = _graph_to_laplacian(g.num_nodes, g.edge_index)
        samples.append(GraphSample(x=x, laplacian=L, y=int(g.y.item())))
    input_dim = (
        int(ds.num_node_features)
        if ds.num_node_features > 0
        else int(samples[0].x.shape[1])
    )
    return samples, input_dim, int(ds.num_classes)


REGISTERED: dict[str, Any] = {
    "mutag": MUTAGDataset(),
    "proteins": PROTEINSDataset(),
}


def get_dataset(name: str) -> Any:
    if name not in REGISTERED:
        known = ", ".join(sorted(REGISTERED))
        raise KeyError(f"unknown dataset {name!r}; registered: {known}")
    return REGISTERED[name]
