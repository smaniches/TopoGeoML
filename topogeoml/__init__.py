"""
TopoGeoML — topology-aware geometric machine learning.

A Python-first research and engineering stack connecting persistent homology,
higher-order topological domains, and geometric deep learning into reproducible
ML pipelines.

v0.0.5 public surface:
    Core:
        PersistenceDiagram, DiagramProvenance, RipsFiltration,
        PersistenceImageVectorizer, BettiCurveVectorizer,
        SimplicialComplex, hodge_laplacian, is_chain_complex, betti_numbers,
        CubicalDiagnostic, cubical_mask_diagnostic.
    Data:
        graph_to_clique_complex.
    Pipelines:
        TopologyFeaturePipeline.
    Audits:
        audit_embedding, EmbeddingTopologyAudit.
    Experiments:
        load_experiment_config, write_results, ExperimentConfig and dataclasses.

torch-dependent symbols (`nn.hodge.HodgeMessagePassing`, `nn.diff_ph.*`,
`training.callbacks.ShapeOfLearningCallback`) are reachable via explicit
`from topogeoml.<subpkg> import ...` and are NOT auto-imported here, so the
core package imports cleanly in torch-less environments.

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
License: MIT
"""

from topogeoml._version import __version__
from topogeoml.audits.embedding_audit import (
    EmbeddingTopologyAudit,
    audit_embedding,
)
from topogeoml.core.complexes import (
    SimplicialComplex,
    betti_numbers,
    hodge_laplacian,
    is_chain_complex,
)
from topogeoml.core.cubical import CubicalDiagnostic, cubical_mask_diagnostic
from topogeoml.core.diagrams import DiagramProvenance, PersistenceDiagram
from topogeoml.core.filtrations import RipsFiltration
from topogeoml.core.vectorizers import (
    BettiCurveVectorizer,
    PersistenceImageVectorizer,
)
from topogeoml.data.graph_to_complex import graph_to_clique_complex
from topogeoml.experiments.configs import (
    ExperimentConfig,
    load_experiment_config,
    write_results,
)
from topogeoml.pipelines.feature_pipeline import TopologyFeaturePipeline
from topogeoml.signal import (
    TopologyFeatureConfig,
    estimate_delay_autocorrelation,
    sliding_window_topology_features,
    takens_embedding,
)
from topogeoml.training.snapshot import DivergenceAlert, ShapeSnapshot

__all__ = [
    "BettiCurveVectorizer",
    "CubicalDiagnostic",
    "DiagramProvenance",
    "DivergenceAlert",
    "EmbeddingTopologyAudit",
    "ExperimentConfig",
    "PersistenceDiagram",
    "PersistenceImageVectorizer",
    "RipsFiltration",
    "ShapeSnapshot",
    "SimplicialComplex",
    "TopologyFeatureConfig",
    "TopologyFeaturePipeline",
    "__version__",
    "audit_embedding",
    "betti_numbers",
    "cubical_mask_diagnostic",
    "estimate_delay_autocorrelation",
    "graph_to_clique_complex",
    "hodge_laplacian",
    "is_chain_complex",
    "load_experiment_config",
    "sliding_window_topology_features",
    "takens_embedding",
    "write_results",
]
