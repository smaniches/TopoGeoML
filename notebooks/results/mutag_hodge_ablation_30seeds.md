# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-21T17:57:16Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-classifier` | `mutag` | 0.697 [0.658, 0.750] | 30 |
| `hodge-mp-normalised` | `mutag` | 0.789 [0.763, 0.816] | 30 |
| `hodge-mp-residual` | `mutag` | 0.750 [0.724, 0.789] | 30 |
| `hodge-mp-deep-residual` | `mutag` | 0.776 [0.737, 0.789] | 30 |
| `mlp-baseline` | `mutag` | 0.789 [0.763, 0.816] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| mutag | `hodge-mp-classifier` | `hodge-mp-normalised` | -0.0921 | 1.251e-04 | 6.254e-04 | -0.643 | **hodge-mp-classifier ≠ hodge-mp-normalised** |
| mutag | `hodge-mp-classifier` | `hodge-mp-residual` | -0.0526 | 1.119e-01 | 1.598e-01 | -0.280 | no diff |
| mutag | `hodge-mp-classifier` | `hodge-mp-deep-residual` | -0.0789 | 2.156e-02 | 5.390e-02 | -0.310 | no diff |
| mutag | `hodge-mp-classifier` | `mlp-baseline` | -0.0921 | 5.657e-05 | 5.657e-04 | -0.760 | **hodge-mp-classifier ≠ mlp-baseline** |
| mutag | `hodge-mp-normalised` | `hodge-mp-residual` | +0.0395 | 7.253e-02 | 1.209e-01 | 0.333 | no diff |
| mutag | `hodge-mp-normalised` | `hodge-mp-deep-residual` | +0.0132 | 1.849e-01 | 2.054e-01 | 0.259 | no diff |
| mutag | `hodge-mp-normalised` | `mlp-baseline` | +0.0000 | 7.143e-01 | 7.143e-01 | 0.130 | no diff |
| mutag | `hodge-mp-residual` | `hodge-mp-deep-residual` | -0.0263 | 1.675e-01 | 2.054e-01 | -0.217 | no diff |
| mutag | `hodge-mp-residual` | `mlp-baseline` | -0.0395 | 5.742e-03 | 1.914e-02 | -0.310 | **hodge-mp-residual ≠ mlp-baseline** |
| mutag | `hodge-mp-deep-residual` | `mlp-baseline` | -0.0132 | 5.077e-02 | 1.015e-01 | -0.481 | no diff |

_No claim made without a statistically significant result after BH correction at α=0.05._