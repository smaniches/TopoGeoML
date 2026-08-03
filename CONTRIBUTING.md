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
- **100% line and 100% branch coverage** on the `topogeoml/` package with full dependencies, enforced by the full-deps `coverage-gate` CI job (`--cov-branch --cov-fail-under=100`); the `benchmarks/` research harness is high but below 100% (cross-backend tests need the `bench` extra) and is intentionally outside the gated scope.
- **ruff clean** across all source directories
- **Every empirical claim** must point to either a literature citation or an in-repo experiment

---

## Testing

```bash
pytest                                           # full suite (504 tests)
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

A release has two explicit maintainer approvals: merging the hand-authored
release PR, then creating the release tag from that exact reviewed commit.
No workflow creates tags automatically.

Prepare one focused release PR that:

1. bumps `topogeoml/_version.py` and the version references in
   `CITATION.cff`, `.zenodo.json`, `README.md`, `docs/index.md`, and
   `LIMITATIONS.md`;
2. adds the `## [X.Y.Z]` section to `CHANGELOG.md`;
3. passes the complete required CI suite; and
4. is reviewed and merged deliberately.

After merge, copy the exact squash-merge commit SHA from the release PR and
create an annotated tag on that commit. Do not tag a moving branch name or an
unreviewed later `main` commit.

```bash
set -euo pipefail

git fetch origin main --tags

# Replace the value below with the release PR's exact squash-merge commit SHA.
RELEASE_SHA="PASTE_FULL_40_CHARACTER_COMMIT_SHA_HERE"

git cat-file -e "${RELEASE_SHA}^{commit}"
git merge-base --is-ancestor "$RELEASE_SHA" origin/main

VERSION=$(git show "$RELEASE_SHA:topogeoml/_version.py" \
  | sed -n 's/^__version__ = "\(.*\)"/\1/p')
TAG="v$VERSION"

test -n "$VERSION"
printf '%s\n' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'
git show "$RELEASE_SHA:CHANGELOG.md" | grep -qF "## [$VERSION]"

if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing tag $TAG" >&2
  exit 1
fi

git tag -a "$TAG" "$RELEASE_SHA" -m "$TAG"
test "$(git rev-parse "${TAG}^{commit}")" = "$RELEASE_SHA"
git show --no-patch --decorate "$TAG"
git push origin "refs/tags/$TAG"
```

The tag push triggers `.github/workflows/release.yml`, which checks out that
exact tag, builds the distributions, generates provenance and an SBOM,
publishes through PyPI Trusted Publishing, signs the artifacts with Sigstore,
and creates the GitHub Release.

The publication workflow is deliberately tag-triggered only. Do not start a
new workflow run to recover a partial release: a new run would rebuild the
wheel and sdist and could produce files that differ from artifacts already
accepted by PyPI.

Inspect the original tag-triggered run and the external state before recovery.
If `build` succeeded and a downstream job failed, use **Re-run failed jobs** on
the original run, or run `gh run rerun RUN_ID --failed`. This preserves the
original tag and commit and lets the retried downstream job download the
`release-artifacts` associated with that workflow run without rerunning a
successful build. If PyPI may have accepted only part of the distribution set,
stop and reconcile the exact filenames and hashes before retrying any job.
Never move or overwrite a published release tag.
