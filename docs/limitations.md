---
title: Limitations & scope
nav_order: 7
description: "Scientific, statistical, and engineering limits of the current release."
---

# Limitations & scope
{: .no_toc }

This page is the concise website summary. The canonical engineering and scientific limitations document is [`LIMITATIONS.md`](https://github.com/smaniches/TopoGeoML/blob/main/LIMITATIONS.md) at the repository root.

{: .warning }
> **Every graph-classification result is configuration-bound.** The main experiments are short-budget matched-capacity mechanism studies, primarily one layer, hidden_dim = 32, 10 to 20 epochs, Adam at `1e-2`, and no batch normalisation. They are not competitive benchmark submissions and do not establish state-of-the-art model quality.

## Empirical scope

- The H003 NCI1 Hodge-residual versus MLP comparison has a positive difference of +8.6 pp at the tested configuration and survives the retrospective 59-comparison BH sensitivity analysis, but not the corresponding Bonferroni sensitivity threshold.
- Later H008c and H010 controls do not support a unique `L_0` Hodge advantage. The matched normalized-adjacency arm is higher on MUTAG and NCI1 under the declared H010 family threshold, while PROTEINS detects no significant operator difference.
- H008c shows that the tested external-residual adjacency formulation recovers NCI1 performance after the normalized internal-self formulation does not. Because those formulations place and parameterize the self path differently, the result should not be generalized into a universal claim that residual connections alone are the sole mechanism.
- H009 does not show a learned scalar sheaf improvement over fixed Hodge. H41 remains inconclusive under its own preregistered 0.01 falsification threshold.
- H011 on NCI1 does not show a positive `L_1` advantage over the node-level controls. NCI1 is also structurally unsuitable for the intended triangle-rich mechanism because 96% of its graphs contain no triangles.
- H011b on COLLAB remains unresolved. The one-seed smoke result is directional only; the preregistered 30-seed confirmatory run has not completed.

## Statistical scope

- A non-significant comparison is not an equivalence result. The project uses “no significant difference detected” unless an explicit equivalence procedure is performed.
- The hypothesis sequence is adaptive: later hypotheses were generated from earlier results, although each new experiment was preregistered before its own result was observed.
- The investigation-wide 59-comparison BH calculation is therefore reported as a retrospective multiplicity sensitivity analysis over the realized comparison set, not as a prospectively guaranteed 5% program-level FDR procedure.
- Exact Wilcoxon power or minimum-detectable-effect guarantees are not stated without an explicit generative model or simulation. Future confirmatory studies should preregister a power or sensitivity analysis tied to the planned endpoint and threshold.

## Toolkit scope

- `RipsFiltration` exposes Vietoris-Rips persistence. Alpha, Cech, witness, and general lower-star filtrations are not part of that public API.
- Differentiable cubical persistence and `CubicalTopologyLoss` are implemented and gradient-tested, but no powered end-to-end segmentation study currently demonstrates downstream benefit.
- The public feature pipeline provides persistence images and Betti curves. It is not a broad catalog of every persistence representation or diagram metric.
- `HodgeMessagePassing` is a minimal fixed-complex building block, not a variable-topology batched simplicial neural-network framework.
- The embedding audit is a prototype diagnostic with a heuristic persistence threshold, not an exact topological certification system.
- Persistence and clique-complex construction can become expensive on large or dense inputs. The current package does not claim a general high-throughput GPU persistence backend.

## What is not claimed

- Topology improves graph classification or machine learning in general.
- Hodge propagation is generally better than a well-tuned GNN.
- H008c proves a universal residual-only causal mechanism.
- A non-significant result establishes equality.
- The current `L_1` architecture provides unique higher-order signal on triangle-rich data.
- `CubicalTopologyLoss` improves every segmentation task.
- Results generalize beyond their stated datasets, architectures, training budgets, and statistical designs.

For detailed numerical, API, numerical-stability, and platform limitations, use the canonical [`LIMITATIONS.md`](https://github.com/smaniches/TopoGeoML/blob/main/LIMITATIONS.md).
