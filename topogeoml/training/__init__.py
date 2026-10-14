"""
Training-time topology monitoring.

    callbacks.py  — ShapeOfLearningCallback: topology monitor for training loops (torch-gated)
    snapshot.py   — ShapeSnapshot / DivergenceAlert dataclasses (pure-NumPy)

Differentiable topology losses live in ``topogeoml.nn.diff_ph`` (torch-gated).

`ShapeOfLearningCallback` requires torch and is therefore reachable only via
explicit `from topogeoml.training.callbacks import ShapeOfLearningCallback`.
"""

from topogeoml.training.snapshot import DivergenceAlert, ShapeSnapshot

__all__ = ["DivergenceAlert", "ShapeSnapshot"]
