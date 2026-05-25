# TopoGeoML Hodge subsystem benchmark

- Schema: `hodge-1.0.0`
- Timestamp (UTC): 2026-05-25T00:30:45Z
- Platform: Linux-6.18.5-x86_64-with-glibc2.39
- Python: 3.11.15
- Git commit: `290c70f7254860a113a1add37535a8a4a9d3dac9`
- Dependencies: numpy=2.4.6, scipy=1.17.1, torch=2.12.0+cu130, torch_geometric=2.7.0

## Per-(model × dataset) test accuracy

| Model | Dataset | Accuracy (median, 95% bootstrap CI) | n_seeds |
|---|---|---|---|
| `hodge-mp-residual` | `proteins` | 0.686 [0.670, 0.717] | 30 |
| `gin-residual` | `proteins` | 0.675 [0.657, 0.709] | 30 |
| `mlp-baseline` | `proteins` | 0.675 [0.596, 0.706] | 30 |

## Pairwise Wilcoxon signed-rank (paired) + Benjamini-Hochberg FDR

| Dataset | A | B | median Δ | p_raw | p_BH | Effect (r) | Verdict |
|---|---|---|---|---|---|---|---|
| proteins | `hodge-mp-residual` | `gin-residual` | +0.0112 | 1.944e-01 | 2.916e-01 | 0.379 | no diff |
| proteins | `hodge-mp-residual` | `mlp-baseline` | +0.0112 | 1.356e-01 | 2.916e-01 | 0.241 | no diff |
| proteins | `gin-residual` | `mlp-baseline` | +0.0000 | 7.846e-01 | 7.846e-01 | -0.071 | no diff |

_No claim made without a statistically significant result after BH correction at α=0.05._