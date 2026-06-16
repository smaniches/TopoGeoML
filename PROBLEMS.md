# Known Problems, Things We Got Wrong, Things Deferred

A persistent, honest record of defects caught and corrected, self-corrections to
early framing, process gaps and how they were closed, and work deferred without a
timeline. This file is maintained alongside `CHANGELOG.md`: the changelog records
what shipped; this file records what was wrong, what is still open, and what we
chose not to do. Modeled on the companion `homology-cliff/PROBLEMS.md`.

## Things we got wrong (caught and corrected before tagging)

1. **Critical bug in `benchmarks/hodge/models.py` — `HodgeMessagePassing` weights
   were never trained (fixed in PR #12).** The original layer was instantiated
   *inside* `forward_one()`, once per graph. Because the module was constructed on
   every forward call rather than once in `__init__`, its parameters were never
   registered with the optimizer and were re-randomised on every forward pass. The
   consequence is severe and was initially invisible: the pre-PR-#12 MUTAG numbers
   did not measure the Hodge architecture at all — they measured a 2-layer MLP read
   out through a *random, untrained* topology filter. The fix moves layer
   construction into `__init__` so parameters are owned by the module and seen by
   the optimizer. Two regression tests were added to prevent recurrence (they assert
   the layer's parameters appear in `.parameters()` and that gradients flow to them).
   Recorded in `CHANGELOG.md` (0.0.2 → Fixed).

2. **"Topology beats MLP" was over-claimed on MUTAG, then refuted on PROTEINS
   (PR #15, PR #16).** A single-dataset MUTAG result is not evidence of a general
   advantage. The preregistered two-dataset replication (hypothesis 002, PROTEINS)
   showed the strong "topology beats MLP" claim does **not** replicate (p_BH = 0.548,
   equality), and the MUTAG combinatorial-Laplacian *harm* shrank by roughly 10×.
   The claim was demoted to its defensible form: symmetric-normalised one-layer Hodge
   MP *matches* an MLP, and only the residual+normalised variant *strictly beats* MLP,
   and only on the larger NCI1 dataset (hypothesis 003, p_BH = 4.83 × 10⁻³). The
   residual variant's verdict actually **inverts** between datasets — it underperforms
   MLP on MUTAG (p_BH = 0.019) and beats it on NCI1 — which is itself reported rather
   than hidden. Recorded in `CHANGELOG.md` (0.0.2 → Added — empirical results).

3. **Topology-divergence claim demoted from a result to exploratory.** The
   `ShapeOfLearningCallback.divergence_score` comparison was reconciled to
   "exploratory" because the supporting regime is floor-limited and lacks a proper
   control; the investigation-wide multiple-comparison correction (FDR over the full
   set of distinct comparisons) was applied as the primary analysis rather than
   reporting only the favorable subset. Recorded in the recent reconciliation history
   on `main` (PRs #45, #46) and in `LEADERBOARD.md`.

## Process / CI gaps, and how they were closed

1. **mypy was clean locally but not gated in CI (deferral now closed).** At v0.0.2,
   `CHANGELOG.md` recorded `mypy topogeoml: 0 errors (CI enforcement deferred to a
   separate PR pending constrained-env reproduction)`. That deferral has since been
   closed: `.github/workflows/ci.yml` runs `mypy topogeoml` as a **hard, gating**
   step (no `continue-on-error`), and `pyproject.toml` sets `[tool.mypy] strict =
   true`. Type-checking now blocks merge on every push and pull request. The local
   sandbox in which this entry was written resolves a newer mypy / numpy-stubs
   combination than the versions resolved on the CI runner, and surfaces a small
   number of `[type-arg]` notes on generic `numpy` aliases that the CI-resolved
   stubs do not; the authoritative signal is the CI run, which is green on `main`.

2. **Coverage is enforced by a dedicated full-dependency gate, not the default test
   job.** The `test` job installs only `.[dev]`, so the torch / torch-geometric
   modules are import-skipped and its coverage is intentionally partial; that job
   reports coverage but does not fail on it. A separate `coverage-gate` job installs
   `.[all]` (CPU torch wheels) and runs the full suite with
   `--cov-branch --cov-fail-under=100`. A repository-level `fail_under` in
   `pyproject.toml` is *intentionally omitted*: it would make the partial-coverage
   `test` job fail, so the gate lives only where 100% is genuinely achievable.

## Things deferred (no implementation timeline)

These are recorded so the absence of a feature is never mistaken for an oversight.
From `CHANGELOG.md` (0.0.2 → Deferred indefinitely):

- PH metric cascade (Euclidean → Spectral → Fermat)
- TopoNetX integration for non-simplicial complexes
- GPU-batched Vietoris-Rips
- MLflow / W&B experiment tracking
- Multi-rank simplicial neural network (full SCN)
- Real DRIVE retinal-vessel numbers — the pipeline shipped (PR #9), but the
  GPU run is user-side and has not been executed in this repository.

## Honest epistemic statement

TopoGeoML is a pre-1.0 research toolkit. Its headline empirical claim (residual +
symmetric-normalised Hodge message passing strictly beats a matched-capacity MLP on
NCI1, p_BH = 4.83 × 10⁻³) is preregistered, FDR-corrected, and reproducible from
`REPRODUCING.md`. The framework also owns its null and negative results (the MUTAG
advantage did not replicate on PROTEINS; the residual variant's verdict inverts
across datasets), one corrected critical training bug, and a list of features it has
chosen not to build. Type-checking and a 100% branch-coverage gate are enforced in
CI. It is not a finished product, and it does not claim to be. It aims to be honest
about exactly what has and has not been demonstrated.
