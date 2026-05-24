# Reproducing the Empirical Results

This document provides step-by-step instructions to reproduce every empirical claim in the TopoGeoML leaderboard. Each claim is independently reproducible from a fresh install.

---

## System Requirements

- Python 3.11 or 3.12
- pip (any recent version)
- ~4 GB disk space for PyTorch Geometric dataset cache
- CPU sufficient for all experiments (GPU optional, not required)

---

## Installation

```bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
pip install -e ".[bench,dev]"
```

The `bench` extra installs PyTorch, PyTorch Geometric, torchvision, GUDHI, and torch-topological. The `dev` extra installs pytest, ruff, mypy, and hypothesis.

---

## Reproduction Commands

### Topology Divergence Callback (Claim 1)

Validates that `ShapeOfLearningCallback.divergence_score` fires no later than a val-loss watchdog.

```bash
python notebooks/topology_predicts_divergence.py --n-seeds 30
```

- **Wall time:** ~2 minutes
- **Output:** `notebooks/results/topology_predicts_divergence_30seeds.{json,md}`
- **Expected:** Direction count 14 topology-earlier / 16 tie / 0 loss-earlier; p_raw = 5.77 x 10^-4

---

### Hypothesis 001: MUTAG Ablation (Claim 2)

Five-arm matched-capacity ablation on MUTAG mutagenicity benchmark.

```bash
python -m benchmarks.hodge \
  --datasets mutag \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 20
```

- **Wall time:** ~15 minutes (CPU)
- **Output:** `notebooks/results/mutag_hodge_ablation_30seeds.{json,md}`
- **Expected:** `hodge-mp-normalised` median 0.789, p_BH = 0.714 vs MLP

---

### Hypothesis 002: PROTEINS Replication (Claim 3)

Cross-dataset replication on PROTEINS protein-graph benchmark.

```bash
python -m benchmarks.hodge \
  --datasets proteins \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10
```

- **Wall time:** ~30 minutes (CPU)
- **Output:** `notebooks/results/proteins_hodge_ablation_30seeds.{json,md}`
- **Expected:** All arms match MLP (all p_BH > 0.3)

---

### Hypothesis 003: NCI1 Scale Escalation (Claim 4 -- THE HEADLINE)

The framework's strict positive-difference real-data result.

```bash
python -m benchmarks.hodge \
  --datasets nci1 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --n-epochs 10
```

- **Wall time:** ~2 hours (CPU)
- **Output:** `notebooks/results/nci1_hodge_ablation_30seeds.{json,md}`
- **Expected:** `hodge-mp-residual` median 0.609, p_BH = 4.83 x 10^-3 vs MLP (+8.6 pp)

---

### Hypothesis 004: Sample-Size Mechanism Test

Subsampling NCI1 to {188, 1113, 2000, 4110} graphs.

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
for N in 188 1113 2000 4110; do
  python -m benchmarks.hodge \
    --datasets nci1 \
    --models hodge-mp-residual mlp-baseline \
    --seeds $SEEDS \
    --n-epochs 10 \
    --max-graphs $N \
    --output notebooks/results/h004_nci1_n${N}_30seeds.json \
    --markdown notebooks/results/h004_nci1_n${N}_30seeds.md
done
```

- **Wall time:** ~2 hours total (CPU)
- **Output:** `notebooks/results/h004_nci1_n{188,1113,2000,4110}_30seeds.{json,md}`
- **Expected:** Hodge advantage monotone in n; never crosses zero

---

### Hypothesis 005: Feature-Density Mechanism Test

Feature-dimension manipulation on NCI1 and MUTAG.

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"

# Direction A: NCI1 features 37 -> 7
python -m benchmarks.hodge \
  --datasets nci1 \
  --models hodge-mp-residual mlp-baseline \
  --seeds $SEEDS \
  --n-epochs 10 \
  --feature-projection-dim 7 \
  --output notebooks/results/h005_nci1_7d_30seeds.json \
  --markdown notebooks/results/h005_nci1_7d_30seeds.md

# Direction B: MUTAG features 7 -> 37
python -m benchmarks.hodge \
  --datasets mutag \
  --models hodge-mp-residual mlp-baseline \
  --seeds $SEEDS \
  --n-epochs 10 \
  --feature-projection-dim 37 \
  --output notebooks/results/h005_mutag_37d_30seeds.json \
  --markdown notebooks/results/h005_mutag_37d_30seeds.md
```

- **Wall time:** ~65 minutes total (CPU)
- **Output:** `notebooks/results/h005_{nci1_7d,mutag_37d}_30seeds.{json,md}`
- **Expected:** NCI1-7d: Hodge wins (0.581 vs 0.500); MUTAG-37d: no difference

---

### Hypothesis 006: Constant-Feature Ablation

Graph-topology-only classification (all node features replaced with constant vector).

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
for DS in mutag proteins nci1; do
  python -m benchmarks.hodge \
    --datasets $DS \
    --models hodge-mp-residual mlp-baseline \
    --seeds $SEEDS \
    --n-epochs 10 \
    --constant-features \
    --output notebooks/results/h006_${DS}_constant_30seeds.json \
    --markdown notebooks/results/h006_${DS}_constant_30seeds.md
done
```

- **Wall time:** ~75 minutes total (CPU)
- **Output:** `notebooks/results/h006_{mutag,proteins,nci1}_constant_30seeds.{json,md}`
- **Expected:** All datasets: Hodge significantly above class prior (all p_BH < 5e-4)

---

### Hypothesis 007: Structural Proxy Decomposition

Deterministic graph-structural analysis (no seeded sampling).

```bash
python -m benchmarks.hodge.h007_analysis \
  --output notebooks/results/h007_structural_decomposition.json \
  --markdown notebooks/results/h007_structural_decomposition.md
```

- **Wall time:** ~5 minutes (CPU)
- **Output:** `notebooks/results/h007_structural_decomposition.{json,md}`
- **Expected:** All proxies rank MUTAG > PROTEINS > NCI1; rho = -1.0 vs full-feature gain

---

## Verification

After running any reproduction command, compare your output against the archived reports:

```bash
# Example: compare NCI1 headline numbers
python -c "
import json
with open('notebooks/results/nci1_hodge_ablation_30seeds.json') as f:
    data = json.load(f)
# Check that hodge-mp-residual median is within [0.58, 0.63]
# and MLP median is within [0.50, 0.57]
print('Hodge-residual median:', data['arms']['hodge-mp-residual']['median'])
print('MLP median:', data['arms']['mlp-baseline']['median'])
"
```

---

## Expected Numerical Variation

Due to hardware-specific floating-point differences (CPU architecture, BLAS implementation, PyTorch version), exact per-seed accuracies may differ slightly from the archived reports. Expected variation:

- Per-seed accuracy: +/- 0.01 (1 percentage point)
- Median across 30 seeds: +/- 0.005 (0.5 percentage points)
- BCa CI bounds: +/- 0.01
- Wilcoxon p-values: may shift by a small factor but should remain on the same side of significance thresholds

If your results diverge from the archived reports by more than these bounds, please open an issue with your hardware/software versions and the full JSON output.

---

## Full Suite

To reproduce all experiments sequentially:

```bash
# Total wall time: ~6-7 hours on CPU
python notebooks/topology_predicts_divergence.py --n-seeds 30
python -m benchmarks.hodge --datasets mutag --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 20
python -m benchmarks.hodge --datasets proteins --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10
python -m benchmarks.hodge --datasets nci1 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --n-epochs 10
# H004-H007 commands as listed above
```
