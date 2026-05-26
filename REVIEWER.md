# Reviewer Guide

A 10-minute path to verify this repository's claims.

## 1. Install (2 minutes)

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
pip install -e ".[dev]"
```

## 2. Run tests (3 minutes)

```bash
pytest --cov=topogeoml --cov=benchmarks --cov-fail-under=100
```

Expected: 497 tests pass, 100% coverage on `topogeoml/` and `benchmarks/`. Torch-dependent tests skip cleanly if torch is not installed.

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

## 5. Reproduce one result (5 minutes)

The fastest reproducible claim is the topology-divergence callback (Claim 1 in LEADERBOARD.md):

```bash
python notebooks/topology_predicts_divergence.py --n-seeds 5
```

Expected: topology watchdog fires no later than the loss watchdog on all seeds. Full 30-seed reproduction takes ~2 minutes.

## 6. Inspect the evidence chain

Each empirical claim maps to a JSON artifact and reproduction command:

- [`docs/CLAIMS_TO_EVIDENCE.md`](docs/CLAIMS_TO_EVIDENCE.md) — every README claim with evidence path, command, and tolerance
- [`docs/STATISTICAL_SUMMARY.md`](docs/STATISTICAL_SUMMARY.md) — investigation-wide FDR analysis (76 comparisons)
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
