---
title: Statistical summary
nav_order: 4
---

# Statistical Summary

This document records the inferential contract for the current TopoGeoML graph-classification investigation. It separates preregistered decisions inside each experiment from retrospective summaries across the evolving research program.

## 1. Current status

TopoGeoML contains 14 preregistered hypothesis documents from H001 through H011b. The program is sequential and adaptive: later hypotheses were designed in response to earlier results, but each new hypothesis was committed before its own experiment was run.

H009 is currently invalidated by implementation audit. The archived H009 result is retained for provenance but is not valid evidence about a cellular-sheaf Laplacian. H39, H40, and H41 are unresolved until the sheaf operator is repaired, mathematically validated, and rerun under a corrected protocol.

Because the previous investigation-wide 59-comparison and 76-entry sensitivity tables included H009 comparisons, those tables are withdrawn as current evidence. They should not be recomputed or quoted as a current project-level multiplicity result until the corrected H009 experiment has produced a valid result artifact.

This withdrawal does not change the valid within-experiment results for H001-H008c or H010-H011. Those experiments use different implementations and retain their own preregistered decision rules and result artifacts.

## 2. Within-experiment inference

Each preregistered experiment defines its comparison family and decision rule before its result is observed. Pairwise Wilcoxon p-values are adjusted with Benjamini-Hochberg within the declared comparison family unless the hypothesis document states otherwise.

The decision threshold is not always 0.05. Some sub-hypotheses explicitly use 0.01. Therefore:

- a BH-adjusted p-value below 0.05 is not automatically a preregistered confirmation when the declared rule requires 0.01;
- a result between 0.01 and 0.05 can be conventionally significant while remaining inconclusive under a stricter preregistered rule;
- the hypothesis document, not a generic alpha value, determines the confirmation or falsification label;
- a non-significant comparison is a failure to reject the null under the stated test and threshold, not evidence of equivalence.

The current evidence index in [`../LEADERBOARD.md`](../LEADERBOARD.md) uses this vocabulary explicitly and marks H009 as invalidated rather than assigning scientific verdicts to the historical sheaf artifact.

## 3. Investigation-wide multiplicity

The research program was not specified as one fixed comparison family before any data were observed. H001-H011b were generated sequentially, with later questions motivated by earlier outcomes. A single retrospective BH correction across the realized program can be useful as a sensitivity analysis, but it is not equivalent to a prospectively specified program-level false-discovery-rate procedure.

The repository previously reported a retrospective sensitivity analysis over 59 de-duplicated comparisons and 76 computed entries. That analysis included comparisons from the now-invalidated H009 experiment. It is therefore retained only in git history as part of the audit trail and is not a current project-level statistical claim.

After a corrected H009 rerun, any new investigation-wide analysis must be regenerated from the validated comparison set. The procedure must document:

1. exactly which comparisons are included;
2. which repeated baseline comparisons are de-duplicated;
3. which raw p-values enter the correction;
4. how invalidated or exploratory experiments are handled;
5. that the resulting global correction is retrospective unless the full family was prospectively fixed.

No current conclusion in TopoGeoML depends on preserving the withdrawn 59/76 table.

## 4. Non-significance and equivalence

TopoGeoML does not interpret `p >= alpha` as equality. Current documents use language such as "no significant difference detected" unless an explicit equivalence design was performed.

A future equivalence claim should preregister an equivalence margin and a procedure designed to test that margin. The absence of a detected difference is not sufficient.

The repository also does not publish exact minimum-detectable-effect values for the paired Wilcoxon test without a documented power model. Power depends on the distribution of paired differences, zero and tie structure, the effect definition, and the planned decision threshold. A prospective power statement should come from an explicit analytical model or simulation tied to the intended experiment.

## 5. Adaptive hypothesis generation

The research sequence is adaptive:

- H001-H007 developed the initial graph-classification and mechanism questions.
- H008-H010 followed from the earlier operator and architecture results.
- H011 followed from the conclusion that node-level `L_0` Hodge propagation did not show a unique advantage over the matched normalized-adjacency control.
- H011b was created after the NCI1 triangle census showed that H011 was a poor test of a triangle-rich `L_1` mechanism.

Preregistering each experiment before its own execution protects that experiment from changing its prediction after seeing its result. It does not turn the full adaptive program into one prospectively specified group-sequential design.

For any result intended to support a broad external claim, the strongest next step is an independent confirmatory experiment with a fixed hypothesis, fixed comparison family, pre-specified endpoint, justified sensitivity analysis, fixed multiplicity procedure, and data not used to generate the hypothesis.

## 6. Configuration scope

The reported graph experiments are bounded to the configurations recorded in their hypothesis documents and result artifacts. Common settings include one message-passing layer, `hidden_dim=32`, 10 to 20 epochs, Adam at `1e-2`, no batch normalisation, and 30 seeded repetitions for confirmatory graph experiments.

H011b preregisters a 30-seed COLLAB design. A separate 18-seed compute attempt exceeded the GitHub Actions time limit and is not a completed substitute for that design. The one-seed smoke result is directional infrastructure evidence only and licenses no statistical claim.

Results at different depths, training budgets, hidden dimensions, optimizers, preprocessing choices, or datasets can differ. No claim of generality beyond the tested configuration is made.

## 7. H009 repair requirement

A corrected H009 result may enter the validated record only after all of the following are satisfied:

1. one restriction pair is constructed per undirected edge;
2. the unnormalized learned operator is symmetric and positive semidefinite by construction;
3. identity restrictions recover the ordinary graph Laplacian to numerical tolerance;
4. gradients reach the restriction learner and downstream message-passing parameters;
5. exact trainable-parameter counts are reported and the capacity protocol is satisfied or explicitly corrected before the rerun;
6. a fresh 30-seed NCI1 result artifact is produced from the corrected commit;
7. the evidence index and any investigation-wide multiplicity sensitivity analysis are regenerated from the validated result set.

The historical H009 artifact remains public because preserving invalidated evidence is part of the audit trail.

## References

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.
