# Reproducing the Empirical Results

This guide separates two goals that should not be conflated:

1. **Current-code rerun:** execute the same experimental design and comparison family using the current repository code.
2. **Historical numerical replication:** recover the historical code revision and as much of the recorded software environment as the archived evidence actually contains, then rerun that design.

The distinction matters because the Hodge benchmark registry has grown over time. When `python -m benchmarks.hodge` is invoked without `--models`, the runner uses every model currently registered. That can change the pairwise comparison family and therefore the Benjamini-Hochberg adjusted p-values. Every confirmatory command below names the model family explicitly.

Do not write reproduction output over the archived evidence files. The commands below use a separate `reproductions/` directory.

## 1. Environment

Python 3.11 and 3.12 are supported by the current package. For the graph-classification experiments and the topology-divergence experiment:

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
pip install -e ".[torch,dev]"
mkdir -p reproductions
```

The `torch` extra provides PyTorch and PyTorch Geometric. The `dev` extra provides the repository test and analysis tooling. The separate `bench` extra installs the cross-backend differentiable-PH benchmark stack, including `torch-topological`; it is not required for the H001-H011 graph experiments below.

For the authoritative full-dependency package test environment, use `pip install -e ".[all]"` as documented in [`REVIEWER.md`](REVIEWER.md).

## 2. Reproduction levels

### Current-code rerun

Run the exact dataset, model family, seeds, and experimental modifiers specified below on the current checkout. This checks whether the present implementation still executes the intended design.

Do not assume current-code numbers must be numerically identical to an old artifact if implementation or dependencies have changed since that experiment.

### Historical numerical replication

Provenance coverage differs by artifact. Newer result files record a `git_commit_sha` and selected dependency versions. Several earlier H001-H008b artifacts do not contain both fields, so this repository does not pretend that every historical environment can be reconstructed from the JSON alone.

If the artifact contains embedded provenance:

1. Read its recorded git commit, Python version, and dependency versions.
2. Create a clean checkout at that commit.
3. Recreate the recorded environment as closely as possible.
4. Run the exact historical model family and design.
5. Write the new result to a separate location and compare the complete per-seed output, not only the median.

If an older artifact lacks a recorded commit, use repository history to locate the commit that introduced the artifact:

```bash
git log --diff-filter=A --format='%H %cs %s' -- notebooks/results/ARTIFACT.json
```

Then inspect the artifact and its contemporaneous hypothesis at that revision:

```bash
git show COMMIT:notebooks/results/ARTIFACT.json
git show COMMIT:docs/hypotheses/HYPOTHESIS-FILE.md
```

This history-based recovery establishes a historical code anchor. It does not reconstruct dependency versions that were never recorded. When environment provenance is incomplete, describe the exercise as a **historical-code rerun**, not an exact historical numerical replication.

The historical experiment environments were not captured by complete transitive lockfiles. The repository therefore does not claim that a future machine can reproduce every floating-point result bit-for-bit from metadata alone.

No universal numerical tolerance is declared here. A tolerance used for confirmatory reproduction should be justified before comparing results and should reflect the endpoint and environment being reproduced. A p-value should not be called reproduced merely because it remains on the same side of a threshold after undocumented numerical drift.

## 3. Exploratory topology-divergence experiment

This experiment is exploratory, not a confirmatory claim that topology anticipates overfitting. The published run is floor-limited and has no non-overfitting negative control.

```bash
python notebooks/topology_predicts_divergence.py \
  --n-seeds 30 \
  --output reproductions/topology_predicts_divergence_30seeds.json
```

Archived evidence: `notebooks/results/topology_predicts_divergence_30seeds.{json,md}`.

Published directional result: 14 topology-earlier, 16 ties, 0 loss-earlier; paired Wilcoxon p_raw = 5.77 x 10^-4. Because the topology trigger fires at its earliest possible probe step in every seed, this result does not establish anticipatory prediction.

## 4. H001: MUTAG architectural ablation

The original comparison family contains five arms. Naming all five is required to reproduce the original BH family.

```bash
python -m benchmarks.hodge \
  --datasets mutag \
  --models \
    hodge-mp-classifier \
    hodge-mp-normalised \
    hodge-mp-residual \
    hodge-mp-deep-residual \
    mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 20 \
  --output reproductions/h001_mutag_30seeds.json \
  --markdown reproductions/h001_mutag_30seeds.md
```

Archived evidence: `notebooks/results/mutag_hodge_ablation_30seeds.{json,md}`.

Key archived result: normalized Hodge and MLP both have median 0.789 with p_BH = 0.714 for that pair. This is **no significant difference detected**, not an equivalence result. The combinatorial Hodge arm significantly underperforms MLP.

## 5. H002: PROTEINS replication

Use the same five-arm family as H001.

```bash
python -m benchmarks.hodge \
  --datasets proteins \
  --models \
    hodge-mp-classifier \
    hodge-mp-normalised \
    hodge-mp-residual \
    hodge-mp-deep-residual \
    mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h002_proteins_30seeds.json \
  --markdown reproductions/h002_proteins_30seeds.md
```

Archived evidence: `notebooks/results/proteins_hodge_ablation_30seeds.{json,md}`.

Key archived result: normalized Hodge median 0.688 versus MLP 0.675, p_BH = 0.548. No significant positive difference is detected.

## 6. H003: NCI1 scale escalation

Use the same five-arm family. Omitting `--models` on the current code would run additional model arms and change the BH comparison family.

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models \
    hodge-mp-classifier \
    hodge-mp-normalised \
    hodge-mp-residual \
    hodge-mp-deep-residual \
    mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h003_nci1_30seeds.json \
  --markdown reproductions/h003_nci1_30seeds.md
```

Archived evidence: `notebooks/results/nci1_hodge_ablation_30seeds.{json,md}`.

Key archived result: Hodge-residual median 0.609 versus MLP 0.523, median difference +0.086, p_BH = 4.83 x 10^-3 within the H003 family. The comparison is regime-bound and later operator controls show that the positive difference is not unique to the Hodge operator.

## 7. H004: NCI1 sample-size ablation

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"

for N in 188 1113 2000 4110; do
  python -m benchmarks.hodge \
    --datasets nci1 \
    --models hodge-mp-residual mlp-baseline \
    --seeds $SEEDS \
    --n-epochs 10 \
    --max-graphs "$N" \
    --output "reproductions/h004_nci1_n${N}_30seeds.json" \
    --markdown "reproductions/h004_nci1_n${N}_30seeds.md"
done
```

Archived evidence: `notebooks/results/h004_nci1_n{188,1113,2000,4110}_30seeds.{json,md}`.

The archived NCI1-at-188 comparison does not reproduce the negative MUTAG Hodge-residual difference. That supports the narrow conclusion that sample count alone is insufficient to explain the cross-dataset sign change.

## 8. H005: feature-dimensionality ablation

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"

python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual mlp-baseline \
  --seeds $SEEDS \
  --n-epochs 10 \
  --feature-projection-dim 7 \
  --output reproductions/h005_nci1_7d_30seeds.json \
  --markdown reproductions/h005_nci1_7d_30seeds.md

python -m benchmarks.hodge \
  --datasets mutag \
  --models hodge-mp-residual mlp-baseline \
  --seeds $SEEDS \
  --n-epochs 10 \
  --feature-projection-dim 37 \
  --output reproductions/h005_mutag_37d_30seeds.json \
  --markdown reproductions/h005_mutag_37d_30seeds.md
```

Archived evidence: `notebooks/results/h005_{nci1_7d,mutag_37d}_30seeds.{json,md}`.

The archived NCI1 projection has a positive Hodge-residual versus MLP difference; the MUTAG dimensional expansion does not produce a significant difference. Feature dimensionality alone therefore does not transfer the dataset behavior.

## 9. H006: constant-feature ablation

The generic benchmark runner produces Hodge-versus-MLP pairwise comparisons. H006's published evidence instead tests Hodge against each dataset's class-prior control and applies BH correction across the three datasets. Reproducing H006 therefore requires both the three benchmark runs and the dedicated resolver.

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"

for DS in mutag proteins nci1; do
  python -m benchmarks.hodge \
    --datasets "$DS" \
    --models hodge-mp-residual mlp-baseline \
    --seeds $SEEDS \
    --n-epochs 10 \
    --constant-features \
    --output "reproductions/h006_${DS}_constant_30seeds.json" \
    --markdown "reproductions/h006_${DS}_constant_30seeds.md"
done

python -m benchmarks.hodge.h006_analysis \
  --constant-results-dir reproductions \
  --full-results-dir notebooks/results \
  > reproductions/h006_analysis.md
```

The resolver performs the one-sample Wilcoxon tests against class prior and BH correction across MUTAG, PROTEINS, and NCI1. Its `--full-results-dir notebooks/results` input supplies the archived H001-H003 full-feature controls used only for the H25 descriptive cross-dataset correlation. The H006 class-prior p-values come from the newly generated constant-feature files in `reproductions/`.

Archived evidence: `notebooks/results/h006_{mutag,proteins,nci1}_constant_30seeds.{json,md}`.

The archived Hodge arm is above the class-prior control on all three datasets. This demonstrates exploitable graph structure for the tested graph-aware architecture under constant node features; it does not establish a unique Hodge mechanism.

## 10. H007: structural proxy decomposition

H007 is a deterministic analysis rather than a seeded training comparison.

```bash
python -m benchmarks.hodge.h007_analysis \
  --output reproductions/h007_structural_decomposition.json \
  --markdown reproductions/h007_structural_decomposition.md
```

Archived evidence: `notebooks/results/h007_structural_decomposition.{json,md}`.

The five tested structural proxies all rank MUTAG > PROTEINS > NCI1. With only three datasets, the rank reversal relative to the full-feature Hodge-versus-MLP differences is descriptive and should not be treated as a population-level correlation result.

## 11. H008: matched-capacity GIN/GAT comparison

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual gin-baseline gat-baseline mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h008_nci1_gin_gat_30seeds.json \
  --markdown reproductions/h008_nci1_gin_gat_30seeds.md
```

Archived evidence: `notebooks/results/h008_nci1_gin_gat_30seeds.{json,md}`.

The tiny matched-capacity GIN and GAT arms are at the class prior in this protocol. This is not a claim about well-tuned GIN/GAT performance or theoretical expressiveness. H008 alone does not isolate the architectural cause.

## 12. H008b: normalized GIN with internal self contribution

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual gin-normalised gin-baseline mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h008b_nci1_gin_normalised_30seeds.json \
  --markdown reproductions/h008b_nci1_gin_normalised_30seeds.md
```

Archived evidence: `notebooks/results/h008b_nci1_gin_normalised_30seeds.{json,md}`.

Normalization alone does not recover the tested internal-self GIN formulation.

## 13. H008c: matched external-residual adjacency control

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual gin-residual gin-normalised mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h008c_nci1_gin_residual_30seeds.json \
  --markdown reproductions/h008c_nci1_gin_residual_30seeds.md
```

Archived evidence: `notebooks/results/h008c_nci1_gin_residual_30seeds.{json,md}`.

Key archived values: `gin-residual` 0.629, Hodge-residual 0.609, `gin-normalised` 0.500, MLP 0.523. `gin-residual` versus Hodge has median difference +0.0195 and p_BH = 0.0101.

The clean operator comparison is `gin-residual` versus Hodge-residual because those two use the same external-residual computation. The `gin-normalised` versus `gin-residual` comparison also changes the self-path placement and parameterization, so it should not be interpreted as a universal residual-only causal proof.

## 14. H009: invalidated historical sheaf experiment

The historical H009 run is **not a current scientific reproduction target**. A later implementation audit showed that the `sheaf-residual` arm used in that run did not construct the scalar cellular-sheaf Laplacian stated by the hypothesis, and its parameter count also fell outside the stated matched-capacity tolerance.

Archived evidence remains at `notebooks/results/h009_nci1_sheaf_30seeds.{json,md}` for provenance only. Do not rerun the currently registered historical `sheaf-residual` implementation and treat the result as evidence for H39-H41.

The valid sequence is:

1. repair the sheaf operator so one undirected edge defines one restriction pair and the unnormalized operator is symmetric PSD by construction;
2. verify identity reduction to the ordinary graph Laplacian, gradient flow, isolated-vertex behavior, endpoint consistency, and exact trainable-parameter count;
3. merge the repaired implementation;
4. preregister a corrective H009 replication before observing its result;
5. run a fresh 30-seed NCI1 experiment and write a new artifact.

Until that sequence is complete, H39-H41 remain unresolved and the historical H009 pairwise statistics have no current inferential status.

## 15. H010: cross-dataset Hodge versus normalized adjacency

MUTAG:

```bash
python -m benchmarks.hodge \
  --datasets mutag \
  --models hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 20 \
  --output reproductions/h010_mutag_operator_30seeds.json \
  --markdown reproductions/h010_mutag_operator_30seeds.md
```

PROTEINS:

```bash
python -m benchmarks.hodge \
  --datasets proteins \
  --models hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h010_proteins_operator_30seeds.json \
  --markdown reproductions/h010_proteins_operator_30seeds.md
```

NCI1 is reused from H008c rather than redefined as a new statistical family.

Archived evidence: `notebooks/results/h010_{mutag,proteins}_operator_30seeds.{json,md}` plus the H008c NCI1 artifact.

H42 is confirmed on MUTAG. H43 detects no significant Hodge-versus-gin difference on PROTEINS. H44 remains inconclusive because the preregistered prediction required association with a dataset-level property and no such association was established. H45 and H46 are refuted.

## 16. H011: NCI1 L1 edge propagation

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --models l1-hodge-residual hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h011_nci1_l1_30seeds.json \
  --markdown reproductions/h011_nci1_l1_30seeds.md
```

Archived evidence: `notebooks/results/h011_nci1_l1_30seeds.{json,md}`.

Archived medians: L1 0.590, L0 Hodge 0.609, gin-residual 0.629, MLP 0.523. H47 and H48 are refuted under their stated rules; H49 is refuted because gin-residual is higher at p_BH = 0.00676. The result does not resolve the intended triangle-rich higher-order mechanism because 96% of NCI1 graphs contain no triangles.

## 17. H011b: COLLAB L1 follow-up

H011b is **not a completed result**. Its preregistered confirmatory design is 30 seeds. A one-seed, one-epoch smoke run exists, and a separate 18-seed compute attempt exceeded the GitHub Actions time limit.

The preregistered command is:

```bash
python -m benchmarks.hodge \
  --datasets collab \
  --models l1-hodge-residual hodge-mp-residual gin-residual mlp-baseline \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10 \
  --output reproductions/h011b_collab_l1_30seeds.json \
  --markdown reproductions/h011b_collab_l1_30seeds.md
```

A completed confirmatory artifact should also record the triangle census under the exact loaded COLLAB graphs. Until the 30-seed design is completed, the existing smoke result must not be promoted into a higher-order classification claim.

## 18. Comparing a reproduction with archived evidence

Compare complete structured outputs rather than visually comparing headline medians alone. At minimum inspect:

- git commit SHA when recorded or recovered from history;
- Python and dependency versions when recorded;
- dataset and model names;
- exact seed set;
- per-seed test accuracies;
- medians and confidence intervals;
- raw paired-test p-values;
- the exact BH comparison family and adjusted p-values.

Do not combine one arm run on one machine with another arm run in a different environment and then treat the result as the original paired comparison. Run all arms of a comparison family in one invocation when reproducing a paired experiment.

If a current-code rerun differs materially from an archived result, first determine whether the code revision, dependency versions, dataset cache/version, or comparison family changed. File an issue with the new result JSON and environment provenance rather than selecting a post hoc tolerance that makes the comparison pass.

## 19. Source of truth

For each empirical result, use this precedence:

1. the preregistered hypothesis file for the decision rule;
2. the committed JSON artifact for the raw historical result;
3. the Markdown artifact for a human-readable rendering of that result;
4. [`LEADERBOARD.md`](LEADERBOARD.md) and [`STATUS.md`](STATUS.md) for the audited current interpretation;
5. the historical Version 0.0.2 research report only for the H001-H008c historical snapshot it explicitly covers.

An invalidation notice in the current hypothesis/evidence documents supersedes the scientific interpretation of an older artifact without deleting that artifact from the audit trail.

The benchmark CLI itself reports generic BH significance at alpha=0.05. Some preregistered sub-hypotheses use stricter thresholds, so the hypothesis file controls the scientific verdict when those differ.
