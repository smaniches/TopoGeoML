# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-21T18:50:48Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-classifier` | `proteins` | 0.646 [0.605, 0.700] | 30 |
| `hodge-mp-normalised` | `proteins` | 0.688 [0.670, 0.704] | 30 |
| `hodge-mp-residual` | `proteins` | 0.686 [0.670, 0.717] | 30 |
| `hodge-mp-deep-residual` | `proteins` | 0.695 [0.659, 0.709] | 30 |
| `mlp-baseline` | `proteins` | 0.675 [0.596, 0.706] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| proteins | `hodge-mp-classifier` | `hodge-mp-normalised` | -0.0426 | 7.194e-02 | 3.390e-01 | -0.143 | no diff |
| proteins | `hodge-mp-classifier` | `hodge-mp-residual` | -0.0404 | 1.332e-01 | 3.390e-01 | -0.200 | no diff |
| proteins | `hodge-mp-classifier` | `hodge-mp-deep-residual` | -0.0493 | 1.108e-01 | 3.390e-01 | -0.071 | no diff |
| proteins | `hodge-mp-classifier` | `mlp-baseline` | -0.0291 | 4.730e-01 | 6.460e-01 | -0.071 | no diff |
| proteins | `hodge-mp-normalised` | `hodge-mp-residual` | +0.0022 | 9.569e-01 | 9.569e-01 | -0.034 | no diff |
| proteins | `hodge-mp-normalised` | `hodge-mp-deep-residual` | -0.0067 | 5.168e-01 | 6.460e-01 | 0.133 | no diff |
| proteins | `hodge-mp-normalised` | `mlp-baseline` | +0.0135 | 3.285e-01 | 5.475e-01 | 0.000 | no diff |
| proteins | `hodge-mp-residual` | `hodge-mp-deep-residual` | -0.0090 | 7.292e-01 | 8.102e-01 | 0.034 | no diff |
| proteins | `hodge-mp-residual` | `mlp-baseline` | +0.0112 | 1.356e-01 | 3.390e-01 | 0.241 | no diff |
| proteins | `hodge-mp-deep-residual` | `mlp-baseline` | +0.0202 | 2.133e-01 | 4.265e-01 | 0.133 | no diff |

_No claim made without a statistically significant result after BH correction at α=0.05._