# Reviewer Guide

This is the shortest practical verification path for the current repository. Runtime depends on network access, dependency cache state, CPU/GPU availability, and platform. No fixed completion time is claimed.

## 1. Install

From a clean checkout:

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML

# Development checks without the full optional stack.
pip install -e ".[dev]"

# Authoritative full-dependency package gate and research tooling.
pip install -e ".[all]"
```

## 2. Verify the package coverage invariant

The required full-dependency CI gate is equivalent to:

```bash
pytest -m "not gpu" \
  --cov=topogeoml \
  --cov-branch \
  --cov-fail-under=100
```

Expected contract: 100% line and 100% branch coverage on the importable `topogeoml` package. The exact verified snapshot for the current main baseline is recorded in [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) Claim 1.

The `benchmarks/` tree is research infrastructure and is not included in the 100% package-coverage claim.

For the lighter dev-only test matrix:

```bash
pytest
```

Torch-dependent paths can skip without the corresponding optional dependencies, so the dev-only run is not the authoritative package-coverage proof.

## 3. Type check

```bash
mypy topogeoml
```

Expected: zero mypy errors under the repository configuration.

## 4. Lint

```bash
ruff check topogeoml tests benchmarks scripts notebooks
```

Expected: all configured ruff checks pass.

## 5. Exercise the graph benchmark path

A short smoke run is useful for checking that the real benchmark pipeline executes, but it is not a reproduction of a 30-seed scientific claim:

```bash
python -m benchmarks.hodge \
  --datasets mutag \
  --seeds 0 1 2 \
  --n-epochs 5
```

For exact confirmatory designs, use [`REPRODUCING.md`](REPRODUCING.md) and the corresponding preregistration file rather than extrapolating from a smoke run.

The current graph result should be reviewed in two layers:

1. H003 contains a narrow positive NCI1 Hodge-residual versus MLP difference (+8.6 pp, p_BH = 4.83 x 10^-3) under the matched-capacity protocol.
2. H008c and H010 show that this difference is not unique to the Hodge `L_0` operator. Normalized adjacency in the matched external-residual architecture is at least as strong in the tested controls. H008c identifies the successful external-residual self-path formulation, but the causal claim is scoped because the internal-self and external-residual formulations are not computationally identical.

See [`STATUS.md`](STATUS.md) and [`LEADERBOARD.md`](LEADERBOARD.md) for the audited current interpretation.

## 6. Inspect the evidence chain

- [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md): public claims mapped to code, tests, CI, artifacts, and limitations
- [`LEADERBOARD.md`](LEADERBOARD.md): complete empirical evidence index through H011, plus the pending H011b frontier
- [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md): within-experiment rules and retrospective investigation-wide multiplicity sensitivity analysis
- [`REPRODUCING.md`](REPRODUCING.md): reproduction commands
- [`STATUS.md`](STATUS.md): current software and research state

## 7. Inspect limitations

- [`LIMITATIONS.md`](LIMITATIONS.md): canonical current engineering and scientific limitations
- [`docs/limitations.md`](docs/limitations.md): concise documentation-site summary
- [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md): historical Version 0.0.2 report through H008c, explicitly labeled as historical

Important current boundaries include:

- most graph studies use one layer, hidden_dim=32, and short training budgets;
- non-significance is not treated as equivalence;
- the former 59-comparison global BH analysis was a retrospective sensitivity analysis over an adaptively generated research program, not a prospectively guaranteed program-level FDR procedure; it is currently withdrawn because it included invalidated H009 comparisons and has not been regenerated from the validated comparison set;
- H011b COLLAB remains incomplete at the preregistered 30-seed design;
- `CubicalTopologyLoss` is implemented and gradient-tested, but downstream segmentation benefit is not yet established by a powered study.

## 8. Audit preregistration

Git history is the preregistration timestamp. For example:

```bash
git log --format="%H %ai %s" \
  -- docs/hypotheses/HYPOTHESIS-008-gin-gat-comparison.md
```

For a specific hypothesis, compare the earliest preregistration commit with the timestamp/provenance recorded in the corresponding result artifact. The current hypothesis documents preserve the decision rules and separately document any post-result corrections to interpretation.

## 9. Review current source, not historical summaries

The canonical current interpretation is distributed intentionally:

- software capability: `README.md`, package source, tests;
- scientific result: hypothesis file plus committed result artifact;
- cross-experiment interpretation: `STATUS.md`, `LEADERBOARD.md`, `docs/STATISTICAL_SUMMARY.md`;
- historical narrative: `docs/RESEARCH_REPORT.md` only.

A reviewer should not infer current H009-H011 status from the historical Version 0.0.2 report.
