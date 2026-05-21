# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-21T20:29:19Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-classifier` | `nci1` | 0.506 [0.501, 0.511] | 30 |
| `hodge-mp-normalised` | `nci1` | 0.516 [0.511, 0.523] | 30 |
| `hodge-mp-residual` | `nci1` | 0.609 [0.581, 0.625] | 30 |
| `hodge-mp-deep-residual` | `nci1` | 0.603 [0.594, 0.623] | 30 |
| `mlp-baseline` | `nci1` | 0.523 [0.513, 0.566] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| nci1 | `hodge-mp-classifier` | `hodge-mp-normalised` | -0.0103 | 1.070e-03 | 1.783e-03 | -0.630 | **hodge-mp-classifier ≠ hodge-mp-normalised** |
| nci1 | `hodge-mp-classifier` | `hodge-mp-residual` | -0.1034 | 3.725e-09 | 3.725e-08 | -0.933 | **hodge-mp-classifier ≠ hodge-mp-residual** |
| nci1 | `hodge-mp-classifier` | `hodge-mp-deep-residual` | -0.0967 | 3.505e-06 | 1.752e-05 | -0.867 | **hodge-mp-classifier ≠ hodge-mp-deep-residual** |
| nci1 | `hodge-mp-classifier` | `mlp-baseline` | -0.0170 | 1.043e-04 | 2.607e-04 | -0.778 | **hodge-mp-classifier ≠ mlp-baseline** |
| nci1 | `hodge-mp-normalised` | `hodge-mp-residual` | -0.0931 | 3.453e-05 | 1.151e-04 | -0.733 | **hodge-mp-normalised ≠ hodge-mp-residual** |
| nci1 | `hodge-mp-normalised` | `hodge-mp-deep-residual` | -0.0864 | 9.029e-04 | 1.783e-03 | -0.586 | **hodge-mp-normalised ≠ hodge-mp-deep-residual** |
| nci1 | `hodge-mp-normalised` | `mlp-baseline` | -0.0067 | 2.275e-01 | 2.528e-01 | -0.214 | no diff |
| nci1 | `hodge-mp-residual` | `hodge-mp-deep-residual` | +0.0067 | 6.143e-01 | 6.143e-01 | 0.200 | no diff |
| nci1 | `hodge-mp-residual` | `mlp-baseline` | +0.0864 | 3.378e-03 | 4.826e-03 | 0.533 | **hodge-mp-residual ≠ mlp-baseline** |
| nci1 | `hodge-mp-deep-residual` | `mlp-baseline` | +0.0797 | 9.426e-03 | 1.178e-02 | 0.357 | **hodge-mp-deep-residual ≠ mlp-baseline** |

_No claim made without a statistically significant result after BH correction at α=0.05._