"""
Hodge-subsystem bench — validates ``topogeoml.nn.hodge.HodgeMessagePassing``
as a graph-classification building block on a real-data TUDataset
benchmark (MUTAG; more datasets to follow).

Scope (Phase 1 of the Geo subsystem)
------------------------------------
The first iteration of this bench answers one question:

    *Does a HodgeMP-based classifier outperform a feature-MLP baseline
    on the MUTAG molecular-graph classification task, with the
    difference statistically significant under paired Wilcoxon
    signed-rank with Benjamini-Hochberg correction?*

The framework is intentionally simpler than ``benchmarks/`` proper:
one model registry, one dataset, one axis. The shared statistical
machinery in ``benchmarks/stats.py`` is reused so the reporting
discipline (no claim without a significance label) is identical.

References
----------
Lim, L.-H. (2020). "Hodge Laplacians on Graphs." *SIAM Review* 62(3).
Morris, C., Kriege, N. M., Bause, F., et al. (2020). "TUDataset: A
  collection of benchmark datasets for learning with graphs." *ICML 2020
  Workshop on Graph Representation Learning and Beyond*.
"""
