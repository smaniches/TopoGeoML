# Contributing to TopoGeoML

TopoGeoML is an active research project. Contributions that extend the empirical record, fix bugs, improve documentation, or reproduce existing results are welcome.

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
- **100% coverage** on `topogeoml/` and `benchmarks/` (enforced by CI)
- **ruff clean** across all source directories
- **Every empirical claim** must point to either a literature citation or an in-repo experiment

---

## Testing

```bash
pytest                                           # full suite (497 tests)
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
- [ ] 100% coverage maintained on `topogeoml/` and `benchmarks/`
- [ ] If empirical: preregistered hypothesis doc committed before results
- [ ] If empirical: LEADERBOARD.md updated with the new claim
- [ ] If empirical: per-seed JSON + Markdown report in `notebooks/results/`
- [ ] No selective reporting; negative results documented with the same rigour
