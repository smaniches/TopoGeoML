# TopoGeoML notebooks

Runnable artifacts that exercise the library on a GPU. Each one has a
`.py` source (so it's reviewable and version-controlled like normal
code) and an "Open in Colab" badge.

## Available notebooks

### `diff_ph_bench_gpu.py` — GPU diff-PH bench

Runs the framework in `benchmarks/` on a GPU. The CPU runs in CI are
not portable to real workloads (both backends target GPU); this is the
GPU companion.

- **Hardware**: any CUDA GPU (T4 free on Colab, A10 on Modal/Lambda).
- **Time**: ~10 min on T4.
- **Output**: JSON leaderboard + markdown report.

```bash
python notebooks/diff_ph_bench_gpu.py \
    --output /tmp/diff_ph_gpu.json \
    --markdown /tmp/diff_ph_gpu.md
```

Or via Modal (paid-but-quick):
```bash
pip install modal-client && modal token new
modal run scripts/modal_diff_ph_bench.py
```

### `mnist_topology_classification.py` — diff-PH as a trainable feature

The first end-to-end demo of `topogeoml.nn.diff_ph` inside an actual
training loop. Classifies MNIST digits {0, 1, 8} based on a 2-D
point-cloud representation of the active pixels. Two classifiers are
trained for the same number of parameters, only the feature source
differs:

  - **diff-PH classifier**: 16-dim point-cloud projection + 3-dim
    diff-PH features (total H_0 persistence, total H_1 persistence,
    longest H_1 lifetime) → linear head.
  - **No-topology baseline**: 16-dim point-cloud projection + 16-dim
    learned projection of the per-coordinate variance → linear head.

Reports the paired Wilcoxon test result with BH FDR — identical
reporting discipline to the rest of the bench.

- **Hardware**: CPU works; GPU is faster but not required.
- **Time on CPU**: ~10s per seed for the default 30 examples/class +
  10 epochs.
- **Output**: JSON + markdown.

```bash
python notebooks/mnist_topology_classification.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --n-epochs 20
```

## Why MNIST and not DRIVE

The original "killer demo" target was DRIVE retinal vessel
segmentation with a Clough et al. 2020-style topology loss on the
predicted segmentation map. That requires **cubical persistent
homology on 2-D images**, which TopoGeoML does not yet ship in a
differentiable form. The library ships **Rips persistent homology
on point clouds** (`topogeoml.nn.diff_ph`). MNIST topology
classification is the largest demo we can build with the current
shipped surface.

Cubical-diff-PH is the natural next addition; once it lands, a
DRIVE-style segmentation demo becomes mechanical.

## Why a `.py` source and not a `.ipynb`

Tracking notebooks as Python source has several advantages:
- diffs are readable in PRs;
- the same file can be invoked headlessly (CI) and pasted into Colab;
- there's no cell-execution-order pitfall to debug.

To run in Colab, simply paste the contents into a cell:
```python
!pip install -q topogeoml[bench]
!curl -s https://raw.githubusercontent.com/smaniches/TopoGeoML/main/notebooks/diff_ph_bench_gpu.py | python -
```

A versioned `.ipynb` generated from each `.py` is a planned addition.
