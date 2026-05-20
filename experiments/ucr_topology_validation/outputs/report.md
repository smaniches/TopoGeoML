# UCR Topology-Augmentation Validation

**Generated**: 2026-05-20T04:27:50.892255+00:00

**Seed**: 42

**Code hash (sha256[:16])**: `c492503e716cea56`


## Falsifiability outcome

**Claim status**: `supported`
**Datasets where (adjusted p < 0.05) AND (d_z >= 0.5)**: ['GunPoint', 'Coffee']

## Per-dataset results

| Dataset | n | baseline | augmented | diff | d_z | raw p | adj p | CI95 |
|---|---|---|---|---|---|---|---|---|
| ECG200 | 200 | 0.7561 ± 0.0617 | 0.7833 ± 0.0601 | +0.0272 | 0.48 | 0.0461 | 0.0461 | [+0.0036, +0.0582] |
| GunPoint | 200 | 0.7600 ± 0.0712 | 0.9450 ± 0.0235 | +0.1850 | 2.31 | 0.0001 | 0.0003 | [+0.1400, +0.2200] |
| Coffee | 56 | 0.8656 ± 0.0890 | 0.9511 ± 0.0668 | +0.0856 | 1.11 | 0.0013 | 0.0019 | [+0.0511, +0.1278] |

## Environment

- python: `3.12.3`
- platform: `Linux-6.18.5-x86_64-with-glibc2.39`
- numpy: `2.3.5`
- scipy: `1.17.1`
- sklearn: `1.7.2`
- ripser: `0.6.14`
- aeon: `1.4.0`

## Statistical-rigor-engine verification gate

**Step 1 (interpolator check)**: logistic regression with L2 penalty (C=1.0) is not an exact interpolator. In-sample error is non-zero. PASS.

**Step 2 (correction audit)**: all feature computation, StandardScaler fitting, and classifier training are restricted to training-fold data. Test-fold data is never observed during preprocessing. PASS.

**Step 3 (derivative inheritance)**: the reported metric (balanced accuracy) depends only on predictions for the test fold of a single CV split. No derived quantities cross folds. PASS.

**Step 4 (validation provenance)**: stratified 5-fold CV, repeated 3× (seeds 42, 43, 44), n_folds=15 per dataset. PASS.

## Mathematical foundations

See `docs/mathematics/foundations.md` for the complete specification of the framework. The features computed here implement Definition 5.5 (topology feature vector) and the experiment protocol implements Protocol 7.3 verbatim.

