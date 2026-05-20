"""
Training-time topology monitoring and differentiable topology losses.

v0.1 ships:
    callbacks.py  — ShapeOfLearningCallback: topology monitor for training loops
    snapshot.py   — ShapeSnapshot / DivergenceAlert dataclasses
    diff_ph is in topogeoml.nn.diff_ph (torch-gated)
"""

from topogeoml.training.callbacks import ShapeOfLearningCallback
from topogeoml.training.snapshot import DivergenceAlert, ShapeSnapshot

__all__ = ["DivergenceAlert", "ShapeOfLearningCallback", "ShapeSnapshot"]
