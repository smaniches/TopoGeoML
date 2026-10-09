# TopoGeoML

## 1. What this is and what problem it solves

TopoGeoML is a Python library for using topology in machine learning. It turns point clouds into fixed-length features for scikit-learn, constructs simplicial and Hodge operators, and provides experimental topology losses for PyTorch. The repository also contains a separate graph-classification research record. These are different deliverables: a working library does not imply a positive research result.

## 2. Why existing approaches fall short

A persistence engine returns birth–death intervals, not a fitted feature transformer, a training loss, or an auditable experiment. Connecting these pieces by hand risks mixing training and evaluation data or assigning gradients to the wrong critical edges. TopoGeoML supplies those connections; it does not replace the underlying numerical libraries.

## 3. What is genuinely new and what is inherited

The repository-specific work is the assembly of train-fitted feature calibration, typed diagram provenance, optional PyTorch critical-value routing, simplicial operators, and evidence-oriented experiment runners. Vietoris–Rips persistent homology comes from `ripser`; the optional cubical and tied-generator computations use `GUDHI`; classical numerical and model interfaces come from NumPy, SciPy, scikit-learn, NetworkX, and PyTorch. Persistence images, Betti curves, and Hodge Laplacians are established methods. The code and test record do not establish a new topology theorem or superiority over competing toolkits.

## 4. Quickstart

Use Python **3.11 or 3.12**, Git, an internet connection for installation, and a **POSIX shell** (Linux/macOS). Run these commands exactly, starting outside the repository:

~~~bash
git clone https://github.com/smaniches/TopoGeoML.git
cd TopoGeoML
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/circles_vs_lines.py
~~~

A successful run prints `Building synthetic dataset`, cross-validation and training scores, and a `Fit provenance` block. Those scores are observations from your run, not guaranteed benchmark targets. This installs the current checkout, which can differ from the most recently published package version. Windows commands and the smaller direct API example are in [Usage](docs/USAGE.md).

## 5. Where it breaks

Vietoris–Rips complexes can become expensive for large point clouds; topology summaries discard information; and feature calibration must be learned only on training data. The differentiable one-dimensional homology path rejects ambiguous critical-edge gradients by default rather than inventing a derivative. Explicit GUDHI tie selection does not prove a unique derivative. The graph study includes invalidated, negative, and unfinished experiments. See [Limitations](docs/LIMITATIONS.md) before relying on scientific or scaling claims.

## 6. Technical documentation

[Architecture](docs/ARCHITECTURE.md) · [Design decisions](docs/DESIGN_DECISIONS.md) · [Limitations](docs/LIMITATIONS.md) · [Usage and troubleshooting](docs/USAGE.md)

Research evidence and reproduction: [Status](STATUS.md) · [Claims to evidence](docs/CLAIMS_TO_EVIDENCE.md) · [Reproducing results](REPRODUCING.md).
