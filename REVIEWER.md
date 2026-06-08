# Reviewer Guide

A 10-minute path to verify this repository's claims.

## 1. Install (2 minutes)

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML

# Core (type checking, lint, non-torch tests):
pip install -e ".[dev]"

# Full (all tests, coverage, reproduction of empirical results):
pip install -e ".[all]"
```

## 2. Run tests (3 minutes)

```bash
# With full dependencies (torch installed):
pytest --cov=topogeoml --cov=benchmarks

# Without torch (dev-only install):
pytest
```

Expected: 497 tests pass. With full dependencies (`pip install -e ".[all]"`), 100% coverage on `topogeoml/` and `benchmarks/`. Without torch, torch-dependent tests skip cleanly and coverage is partial (nn/ code paths not exercised).

## 3. Type check (30 seconds)

```bash
mypy topogeoml
```

Expected: 0 errors.

## 4. Lint (15 seconds)

```bash
ruff check topogeoml tests benchmarks scripts notebooks
```

Expected: all checks passed.

## 5. Reproduce one result (~2 minutes)

The primary finding is **negative** — encoding topological structure via the Hodge
Laplacian confers no unique advantage once an external residual connection is present
(see [`docs/index.md`](docs/index.md) and [`STATUS.md`](STATUS.md)). The narrow positive
lead is on **NCI1** (+8.6 pp, p_BH = 4.83 × 10⁻³; survives investigation-wide BH but not
Bonferroni; regime-bound — see the caveat in the README).

The fastest way to exercise the exact code path behind those findings is a short smoke of
the Hodge benchmark CLI (the full NCI1 headline is the same command at scale; see
[`REPRODUCING.md`](REPRODUCING.md) §H003). Requires full dependencies (`.[all]`):

```bash
# ~2 min: 3 seeds x 5 epochs on MUTAG (188 graphs). Smoke of the real
# benchmarks.hodge code path; not the full headline result.
python -m benchmarks.hodge --datasets mutag --seeds 0 1 2 --n-epochs 5
```

Expected: a Markdown comparison table is printed and a JSON artifact is written to
`benchmarks/hodge/leaderboard/current.json`. The full headline (NCI1, 30 seeds,
10 epochs, ~2 h CPU) is documented in [`REPRODUCING.md`](REPRODUCING.md) §H003.

> The topology-divergence watchdog
> (`python notebooks/topology_predicts_divergence.py --n-seeds 30`, ~15 min CPU)
> is reproducible too, but it is **exploratory (floor-limited, no control yet)**, not a
> headline claim — the topology watchdog fires at its earliest possible step every seed,
> so the data show only that it is never *slower* than the loss watchdog, not that it
> anticipates divergence (see README §1).

## 6. Inspect the evidence chain

Each empirical claim maps to a JSON artifact and reproduction command:

- [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) — every README claim with evidence path, command, and tolerance
- [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md) — investigation-wide FDR analysis (59 distinct comparisons; 76 total computed)
- [`LEADERBOARD.md`](LEADERBOARD.md) — per-claim status table
- [`REPRODUCING.md`](REPRODUCING.md) — full reproduction guide

## 7. Inspect limitations

- [`LIMITATIONS.md`](LIMITATIONS.md) — what the toolkit does not do
- [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) §4.5 — limitations of the empirical findings
- All results are bounded to: 1-layer, hidden_dim=32, 10-20 epochs, Adam(lr=1e-2), no batch normalisation, 3-4 TUDataset benchmarks

## 8. Preregistration audit

Every hypothesis document in `docs/hypotheses/` was committed before its experiment ran. The git history serves as the timestamp. To verify:

```bash
git log --format="%H %ai %s" -- docs/hypotheses/HYPOTHESIS-008-gin-gat-comparison.md | tail -1
```

Compare the commit timestamp to the experiment result timestamp in the corresponding JSON artifact.
