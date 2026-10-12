"""
Differentiable topology-aware PyTorch modules.

    hodge.py            — HodgeMessagePassing layer
    diff_ph.py          — Differentiable Rips persistence + topology losses
    cubical_diff_ph.py  — Differentiable cubical persistence (GUDHI-backed)

All submodules require torch and are imported explicitly, e.g.
``from topogeoml.nn.diff_ph import rips_diagram_torch``.
"""

__all__: list[str] = []
