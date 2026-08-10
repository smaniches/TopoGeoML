---
title: Overview
nav_order: 1
description: "TopoGeoML software, evidence, and scientific scope."
---

# TopoGeoML
{: .no_toc }

**TopoGeoML is a topology-aware machine learning library with an attached preregistered research record.**

The installed package provides persistent-homology features for ordinary ML pipelines, differentiable Vietoris-Rips and cubical topology primitives for PyTorch, simplicial and Hodge operators, topology features for signals, embedding diagnostics, and reproducible experiment metadata.

The graph-classification investigation is one application of those tools. Its negative and inconclusive results narrow where the tested graph operators should be used, but they do not define the value of the library.

## Software surface

| Use case | Module | Scope |
|---|---|---|
| Point-cloud topology as ML features | `topogeoml.pipelines.TopologyFeaturePipeline` | scikit-learn compatible persistence images and Betti curves |
| Differentiable Vietoris-Rips topology | `topogeoml.nn.diff_ph` | PyTorch critical-value routing and topology regularization |
| Differentiable cubical topology | `topogeoml.nn.cubical_diff_ph` | `CubicalTopologyLoss` and persistence diagrams for image-like tensors |
| Simplicial and Hodge computation | `topogeoml.core`, `topogeoml.nn.hodge` | complexes, boundary operators, Hodge Laplacians, minimal fixed-complex message passing |
| Signal topology | `topogeoml.signal` | Takens embedding and sliding-window persistent-homology features |
| Embedding diagnostics | `topogeoml.audits` | prototype topology audit with a heuristic significance threshold |
| Reproducible research | `benchmarks/`, `docs/`, `notebooks/` | seeded experiments, statistical analysis, provenance, and claim-to-evidence mapping |

Required CI enforces 100% line and 100% branch coverage on the importable `topogeoml` package under full dependencies, together with mypy strict and ruff. The `benchmarks/` tree is research infrastructure outside the package-coverage invariant.

## Graph-classification finding

Across the tested matched-capacity graph-classification configurations, no unique `L_0` Hodge advantage is supported. H008c shows that normalized adjacency in the same external-residual architecture reaches 0.629 on NCI1 versus 0.609 for Hodge. H010 finds a significant adjacency advantage on MUTAG, no significant operator difference on PROTEINS, and the same favorable adjacency direction on NCI1.

H008c also shows that the tested external-residual adjacency formulation recovers performance after the normalized internal-self formulation does not. The causal statement is deliberately scoped: the two formulations place and parameterize the self path differently, so the result identifies a successful tested architecture rather than proving that residual connections alone are the sole mechanism in arbitrary models.

A narrow positive result remains. On NCI1, `hodge-mp-residual` outperforms the matched-capacity MLP baseline by a median 8.6 percentage points (`p_BH = 4.83 x 10^-3`). The comparison survives investigation-wide Benjamini-Hochberg correction but not Bonferroni, and the later operator ablations show that the improvement is not unique to Hodge propagation.

The genuinely higher-order question remains open. H011's NCI1 `L_1` arm does not significantly outperform MLP and is evaluated on a dataset where 96% of graphs contain no triangles. H011b on triangle-rich COLLAB has only a directional smoke result so far.

These experiments are mechanism studies at deliberately constrained capacity. They are not benchmark-performance claims and they do not establish that TopoGeoML should replace well-tuned graph neural networks.

## Statistical language

A non-significant pairwise test is reported as no detected difference at the tested power. It is not proof of equivalence. An explicit equivalence procedure would be required to make an equality claim.

Smoke runs and exploratory diagnostics are also kept separate from confirmatory results. In particular, the H011b COLLAB `L_1` experiment has only a directional smoke result, and the topology-divergence callback study is exploratory because it is floor-limited and lacks a non-overfitting negative control.

## How the investigation was run

- **Preregistration.** Each hypothesis document was committed before its corresponding experiment ran.
- **Seeded analysis.** The confirmatory graph experiments use repeated seeded runs and paired comparisons where appropriate.
- **Multiplicity control.** Benjamini-Hochberg correction is applied within declared comparison families, with a separate investigation-wide analysis across the deduplicated comparison set.
- **Negative results retained.** Refutations, null results, and unresolved experiments remain in the public record.

## Start here

- [Current project status](../STATUS.md)
- [Claims to evidence]({% link CLAIMS_TO_EVIDENCE.md %})
- [Statistical summary]({% link STATISTICAL_SUMMARY.md %})
- [Hypotheses H001-H011b]({% link hypotheses/index.md %})
- [Historical research report]({% link RESEARCH_REPORT.md %}), Version 0.0.2 through H008c
- [Limitations and scope]({% link limitations.md %})
- [Mathematical foundations]({% link mathematics/foundations.md %})

Code, installation instructions, examples, and the research harness are in the [GitHub repository](https://github.com/smaniches/TopoGeoML).

## Citation

```bibtex
@software{maniches_topogeoml_2026,
  author  = {Maniches, Santiago},
  title   = {TopoGeoML: A Preregistered Investigation into Topology-Aware Graph Classification},
  year    = {2026},
  version = {0.0.6},
  doi     = {10.5281/zenodo.20365816},
  url     = {https://doi.org/10.5281/zenodo.20365816}
}
```
