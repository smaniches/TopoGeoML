---
title: Statistical summary
nav_order: 4
---

# Statistical Summary

This document records the inferential contract for the current TopoGeoML graph-classification investigation. It separates preregistered decisions inside each experiment from retrospective summaries across the evolving research program.

## 1. Current status

TopoGeoML contains 14 preregistered hypothesis documents from H001 through H011b. The program is sequential and adaptive: later hypotheses were designed in response to earlier results, but each new hypothesis was committed before its own experiment was run.

H009 is invalidated by implementation audit. The archived H009 result is retained for provenance but is not valid evidence about a cellular-sheaf Laplacian. The sheaf operator was subsequently repaired and invariant-tested (`sheaf-residual` 2.0.0, merge `b89d196`), the corrective protocol was preregistered as H009-R before execution, and the confirmatory 30-seed NCI1 run completed on 2026-08-10 (`notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}`). Under the preregistered rules, H39 is supported (sheaf above the matched MLP, p_BH = 4.73 x 10^-3), H40 is not supported (sheaf versus Hodge p_BH = 0.428), and H41's falsification condition is not met (sheaf below gin-residual at p_BH = 0.0342, between the conventional 0.05 level and the preregistered 0.01 threshold, therefore inconclusive and not an equivalence result).

Because the previous investigation-wide 59-comparison and 76-entry sensitivity tables included invalidated H009 comparisons, those tables remain withdrawn as current evidence. The valid H009-R artifact now completes the comparison set from which a new retrospective analysis could be regenerated, but no such regeneration has been performed; until one is produced under the documentation requirements of section 3, no project-level multiplicity table is claimed.

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

The corrected H009 rerun (H009-R) is now complete, so the validated comparison set exists. Any new investigation-wide analysis must be regenerated from that set, and the H009-R six-comparison family, not the invalidated H009 family, is the sheaf entry. The procedure must document:

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

## 7. H009 repair requirement — resolved by H009-R

A corrected H009 result may enter the validated record only after all of the following are satisfied. Items 1 through 6 are met, and item 7's evidence-index update is done; the investigation-wide sensitivity analysis named in item 7 has not been regenerated and remains withdrawn (section 3):

1. one restriction pair is constructed per undirected edge — enforced by the `sheaf-residual` 2.0.0 implementation and its regression tests (merge `b89d196`);
2. the unnormalized learned operator is symmetric and positive semidefinite by construction — invariant-tested at the same merge;
3. identity restrictions recover the ordinary graph Laplacian to numerical tolerance — invariant-tested;
4. gradients reach the restriction learner and downstream message-passing parameters — invariant-tested;
5. exact trainable-parameter counts are reported and the capacity protocol is satisfied — 2,403 versus 2,338 parameters (+2.78%, inside the 5% tolerance), fixed in the H009-R preregistration before the run;
6. a fresh 30-seed NCI1 result artifact is produced from the corrected commit — `notebooks/results/h009r_nci1_sheaf_v2_30seeds.{json,md}` from commit `79329df`;
7. the evidence index is updated from the validated result set — done in `LEADERBOARD.md` Claim 12b; the optional investigation-wide multiplicity sensitivity analysis has not been regenerated and remains withdrawn (section 3).

The historical H009 artifact remains public because preserving invalidated evidence is part of the audit trail. The H009-R verdicts are recorded in section 1 and in `docs/hypotheses/HYPOTHESIS-009R-sheaf-corrective-replication.md` section 9.

## References

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.
