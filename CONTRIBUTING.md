# Contributing to TopoGeoML

TopoGeoML is an active research project. Contributions that extend the empirical record, fix bugs, improve documentation, or reproduce existing results are welcome.

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## What's accepted

- **Bug fixes** with regression tests
- **Documentation improvements** (typos, clarifications, additional examples)
- **Reproduction reports** confirming or contradicting published results on different hardware
- **New hypotheses** following the preregistration discipline below
- **Additional datasets** integrated into the benchmark harness
- **Architecture variants** tested against the existing matched-capacity protocol

---

## Adding a new hypothesis

The preregistration pattern is the project's core research methodology. To add a new hypothesis:

1. **Write the hypothesis document** in `docs/hypotheses/HYPOTHESIS-NNN-description.md` with:
   - Falsifiable sub-predictions with explicit statistical thresholds
   - An outcome decision tree specifying what each result pattern means
   - A reproduction command
   - A wall-clock budget estimate

2. **Commit the document BEFORE running the experiment.** This is non-negotiable. The preregistration timestamp (git commit) must precede the results.

3. **Run the experiment** with >= 20 seeds (the minimum for paired Wilcoxon to have power at moderate effect sizes; 30 seeds is the project standard).

4. **Save results** to `notebooks/results/` as both JSON (machine-readable) and Markdown (human-readable).

5. **Update LEADERBOARD.md** with a new claim row following the existing format.

6. **Report negative results.** Selective reporting is the failure mode this project exists to prevent. If your hypothesis is refuted, that is a valid and publishable result.

See `LEADERBOARD.md` section "How to add a new claim" for the complete checklist.

---

## Code standards

The following floor is enforced:

- **`float64` dtype** on every numerical array (no silent float32 downcasting)
- **No Python sample loops** for numerical computation (construction loops permitted)
- **Reproducible RNG:** `random_state=42` / `np.random.default_rng(42)` or seed passed explicitly
- **Provenance dict** on every fit + every benchmark cell
- **100% line and 100% branch coverage** on the `topogeoml/` package with full dependencies, enforced by the full-deps `coverage-gate` CI job (`--cov-branch --cov-fail-under=100`); the `benchmarks/` research harness is outside the gated package scope.
- **ruff clean** across all source directories
- **Every empirical claim** must point to either a literature citation or an in-repo experiment

---

## Testing

```bash
pytest                                           # full suite
pytest -m "not slow"                             # skip slow tests
pytest --cov=topogeoml --cov=benchmarks          # with coverage
ruff check topogeoml tests benchmarks scripts notebooks  # lint
```

Test markers: `slow`, `torch`, `gudhi`, `gpu`. Tests requiring optional dependencies skip cleanly when those packages are not installed.

---

## Statistical discipline

All empirical claims must use:

- **BCa bootstrap 95% CIs** (minimum 10,000 replicates)
- **Paired Wilcoxon signed-rank** (matched by seed) for arm-vs-arm comparisons
- **Benjamini-Hochberg FDR** correction across comparison families
- **Rank-biserial r** as the effect-size measure

Do not report uncorrected p-values as the primary result. Do not cherry-pick seeds, epochs, or hyperparameters after seeing results. The `benchmarks/stats.py` module implements all required procedures.

---

## Reproducing existing results

See [`REPRODUCING.md`](REPRODUCING.md) for step-by-step instructions to reproduce every empirical claim in the leaderboard.

---

## Citation

If you use this work in your research, please cite via the `CITATION.cff` file (GitHub will render a "Cite this repository" button). If you extend a specific hypothesis, cite the hypothesis document by its path:

```
docs/hypotheses/HYPOTHESIS-NNN-description.md (commit hash: ...)
```

---

## Pull request checklist

- [ ] CI green (ruff + pytest + coverage)
- [ ] 100% line and 100% branch coverage maintained on the `topogeoml/` package (with full dependencies; the `coverage-gate` CI job enforces this); no regression in `benchmarks/` harness coverage
- [ ] If empirical: preregistered hypothesis doc committed before results
- [ ] If empirical: LEADERBOARD.md updated with the new claim
- [ ] If empirical: per-seed JSON + Markdown report in `notebooks/results/`
- [ ] No selective reporting; negative results documented with the same rigour

---

## Releasing

A reviewed version-bump PR merged into protected `main` is the approval to
publish. The package's version is declared in `topogeoml/_version.py`; the
other public version surfaces and the release date must agree in CI. Normal
code, documentation, and dependency merges never publish a new release
unless the version has been advanced beyond the latest tag.

Prepare one focused release PR that:

1. Updates `topogeoml/_version.py`, `CITATION.cff`, `.zenodo.json`,
   `README.md`, `docs/index.md`, `LIMITATIONS.md`, and all other
   version-bound metadata.
2. Adds a matching `## [X.Y.Z]` entry to `CHANGELOG.md`.
3. Passes required CI checks and is reviewed and merged deliberately.

After successful main-branch CI, `.github/workflows/auto-tag-release.yml`
checks out **the exact commit whose CI passed**, confirms it still equals
`main`, verifies the release version and changelog, rejects a non-increasing
version, and creates an annotated `vX.Y.Z` tag at that commit. Existing tags
are never moved. The workflow explicitly dispatches `release.yml` against
that tag with `tag=vX.Y.Z`: GitHub suppresses ordinary push-triggered
workflow cascades from `GITHUB_TOKEN`-created tags, so dispatch is mandatory.

`.github/workflows/release.yml` also accepts a manually pushed version tag.
Its `workflow_dispatch` entrypoint requires both `--ref vX.Y.Z` and a
matching `--field tag=vX.Y.Z`; the unprivileged validation job rejects
branch dispatch, mismatched versions, and missing changelog entries before
any build or publication job starts. Its existing release supply-chain
contract is unchanged: build once, attest provenance, derive and verify a
CycloneDX SBOM from the exact wheel, publish missing PyPI artifacts through
Trusted Publishing, and sign/attach the same distributions to GitHub Release.

**Recovery:** If a publication job fails after the build succeeds, use
**Re-run failed jobs** on its original run or `gh run rerun RUN_ID --failed`.
This preserves the original build artifacts. Do not start a new run or move
the release tag to work around partial publication; immutable PyPI files
cannot safely be overwritten. The PyPI job verifies SHA-256 hashes for
pre-existing files and uploads only absent distributions. If tagging
succeeded but dispatch never created a release run, first confirm no existing
run or published artifact for that tag, then dispatch that existing exact tag:

```bash
gh workflow run release.yml --ref vX.Y.Z --field tag=vX.Y.Z
```

GitHub Releases and PyPI publication are automated by this mechanism; a
separate Zenodo deposit must still be checked against the versioned tag and
DOI before claiming that external archival synchronization is complete.
