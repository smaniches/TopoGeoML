"""
Topology-aware feature extraction for discrete signals.

This subpackage implements the framework specified in
``docs/mathematics/foundations.md``. The mathematical content is:

- DEFINITION 1.2.1, 1.2.2 — discrete signals and sliding-window point clouds.
- DEFINITION 2.1, 2.2, 2.3 — Vietoris–Rips filtration and persistent homology.
- THEOREM 2.4 — stability of persistence diagrams (cited, not proven).
- DEFINITION 3.1, THEOREM 3.2 — Takens delay embedding.
- DEFINITION 4.1, PROPOSITION 4.2 — natural symmetry group and its
  invariance under Vietoris–Rips persistence.
- DEFINITION 5.1, 5.2, 5.5 — total persistence, persistence entropy,
  topology feature vector.

Public objects:

``takens_embedding(signal, embedding_dim, delay)``
    Construct the Takens delay embedding (§6.1).

``estimate_delay_autocorrelation(signal, max_lag, threshold)``
    Heuristic delay selection by autocorrelation threshold.

``sliding_window_topology_features(point_cloud, config)``
    Extract pooled topology features (§6.2).

``TopologyFeatureConfig``
    Configuration dataclass for the feature extractor.

``topology_feature_names(config)``
    Human-readable feature names matching the output of the extractor.
"""

from topogeoml.signal.delay_embedding import (
    estimate_delay_autocorrelation,
    takens_embedding,
)
from topogeoml.signal.sliding_window import (
    TopologyFeatureConfig,
    sliding_window_topology_features,
    topology_feature_names,
)

__all__ = [
    "TopologyFeatureConfig",
    "estimate_delay_autocorrelation",
    "sliding_window_topology_features",
    "takens_embedding",
    "topology_feature_names",
]
