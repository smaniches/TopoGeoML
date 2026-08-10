---
title: H009 · Sheaf Laplacian
parent: Hypotheses (H001-H011b)
nav_order: 11
---

# Hypothesis 009: Does a learned sheaf Laplacian outperform fixed operators on NCI1?

**Status: INVALIDATED BY IMPLEMENTATION AUDIT 2026-08-09. The H009 numerical artifact is retained for provenance but is not valid evidence for a sheaf-Laplacian claim. H39-H41 were resolved 2026-08-10 by the corrective replication [H009-R](HYPOTHESIS-009R-sheaf-corrective-replication.md): H39 supported, H40 not supported, H41 falsification condition not met.**

The original experiment was preregistered and executed as intended, but a later implementation audit found that the `sheaf-residual` model did not construct the scalar cellular-sheaf Laplacian described by the hypothesis. This is an implementation invalidation, not a statistical reinterpretation.

## 1. Why the original H009 result is invalid

A scalar cellular sheaf on an undirected graph assigns one pair of endpoint restriction scalars to each undirected edge `e = {i, j}`. With an orientation chosen only for construction, the coboundary row for that edge contains the two endpoint restrictions and the sheaf Laplacian is

`L_F = delta.T @ delta`.

Consequently:

- `L_F` is symmetric positive semidefinite;
- for an edge `{i, j}`, the off-diagonal entries are the same in both directions;
- the diagonal contains one squared restriction contribution per incident undirected edge;
- in the identity-restriction case, the scalar construction reduces to the ordinary graph Laplacian `L_0`.

The implementation used in the archived H009 run violates that construction. `GraphSample.laplacian` is a symmetric graph Laplacian and therefore stores both off-diagonal orientations `(i, j)` and `(j, i)`. `_SheafResidualGraphClassifier.forward_one` consumed every off-diagonal entry independently, predicted a separate restriction pair for each orientation, assigned `L_F[i, j]` and `L_F[j, i]` from those independent predictions, and accumulated both orientations into the diagonal.

Therefore the matrix used in H009 was not guaranteed to be symmetric or positive semidefinite and was not, in general, representable as the claimed `delta.T @ delta`. Even if every predicted restriction were exactly one, the diagonal contributions would be counted twice, so the construction would not reduce to the ordinary graph Laplacian as claimed.

### Capacity-accounting error

The original document also misstated the parameter count. At `input_dim=37` and `hidden_dim=32`:

- `proj_in`: 1,216 parameters;
- `sheaf_learner = Linear(64, 2)`: 130 parameters, not 65;
- message-passing weight and bias: 1,056 parameters;
- classifier head: 66 parameters;
- total `sheaf-residual`: **2,468 parameters**.

The Hodge, normalized-adjacency, and MLP controls have 2,338 parameters. The sheaf arm is therefore about **5.56% larger**, slightly outside the experiment's stated 5% matched-capacity tolerance. This discrepancy is secondary to the invalid operator construction but must also be corrected before a rerun.

## 2. Consequence for the scientific record

The archived result `notebooks/results/h009_nci1_sheaf_30seeds.{json,md}` remains in the repository because deleting an invalidated result would damage the audit trail. Its numbers describe the behavior of the historical implementation only. They must not be cited as evidence about a cellular sheaf Laplacian, Neural Sheaf Diffusion, or learned sheaf operators in general.

Accordingly, at the time of invalidation:

- H39, H40, and H41 became **unresolved** (they have since been resolved by the corrective replication [H009-R](HYPOTHESIS-009R-sheaf-corrective-replication.md), whose artifact — not this one — carries their verdicts);
- H009 is excluded from the validated evidence index;
- H009-specific comparisons are excluded from the investigation-wide multiplicity sensitivity analysis; the H009-R comparison family is the valid replacement.

This invalidation does not alter the H001-H008c or H010-H011 result artifacts, which use different model implementations.

## 3. Original preregistered question

**Falsification target.** Whether a data-dependent scalar sheaf Laplacian, where edge-level restriction maps are learned from node features, outperforms both the fixed Hodge Laplacian and the fixed normalized adjacency on NCI1 under the matched-capacity protocol with an external residual.

**Theoretical construction intended by the preregistration.** For each undirected edge `e = {i, j}`, one learned pair of scalar restrictions `f_{i<-e}` and `f_{j<-e}` defines the edge's coboundary row. The resulting `L_F = delta.T @ delta` is symmetric positive semidefinite. The identity-restriction case recovers the ordinary graph Laplacian.

### Original preregistered sub-hypotheses

The decision rules below are preserved because they were specified before the original run. They will be reused only if the corrected implementation and capacity-matching protocol remain materially faithful to the intended experiment.

| ID | Sub-hypothesis | Prediction | Falsified if |
|---|---|---|---|
| **H39** | `sheaf-residual` strictly beats `mlp-baseline` on NCI1 | `p_BH < 0.05` | `p_BH >= 0.05` |
| **H40** | `sheaf-residual` strictly beats `hodge-mp-residual` on NCI1 | uncertain | `p_BH >= 0.05` or sheaf < Hodge |
| **H41** | `sheaf-residual` at least matches `gin-residual` on NCI1 | `p_BH >= 0.05` or sheaf > gin-residual | sheaf strictly underperforms gin-residual at `p_BH < 0.01` |

## 4. Requirements for a valid H009 rerun

These requirements were satisfied by the `sheaf-residual` 2.0.0 repair (merge `b89d196`) and the preregistered [H009-R](HYPOTHESIS-009R-sheaf-corrective-replication.md) run, except that the optional investigation-wide sensitivity analysis (item 8) has not been regenerated. The list is preserved as written:

1. **One restriction pair per undirected edge.** The model must canonicalize each graph edge once, for example with `i < j`, before predicting endpoint restrictions.
2. **Construction test.** The learned operator must be assembled from that one edge set with symmetric off-diagonals and one diagonal contribution per endpoint incidence.
3. **PSD test.** For randomized small graphs and randomized model parameters, the constructed unnormalized `L_F` must be symmetric to numerical tolerance and have no materially negative eigenvalues beyond floating-point tolerance.
4. **Identity-reduction test.** With all endpoint restrictions fixed to one, `L_F` must equal the ordinary combinatorial `L_0` exactly to numerical tolerance.
5. **Gradient test.** Gradients must reach both the restriction learner and the downstream message-passing parameters through a real loss.
6. **Capacity accounting.** The corrected arm's exact trainable-parameter count must be reported from `model.parameters()` and brought within the preregistered tolerance or the capacity change must be explicitly preregistered as a corrected protocol.
7. **Fresh result artifact.** The 30-seed NCI1 experiment must be rerun from the corrected commit. Historical H009 numbers cannot be recycled.
8. **Fresh multiplicity analysis.** Only the corrected H009 comparisons may enter the investigation-wide sensitivity analysis.

## 5. Historical artifact

The 2026-05-25 H009 run produced a median accuracy of 0.604 for the historical `sheaf-residual` implementation. That number is retained solely as provenance for the invalidated implementation and has no current inferential status.

## References

- Bodnar, C., Di Giovanni, F., Chamberlain, B., Lio, P., & Bronstein, M. (2022). Neural Sheaf Diffusion: A topological perspective on heterophily and oversmoothing in GNNs. *NeurIPS 2022*.
- Hansen, J. & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3, 315-358.
