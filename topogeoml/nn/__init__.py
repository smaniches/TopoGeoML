"""
Differentiable topology-aware PyTorch modules.

v0.0.1 ships:
    hodge.py — minimal Hodge message passing layer (item 8). Requires torch.

v0.1 will add: drift-tensor correction, topology losses, pooling, regularizers.

Importing this subpackage does NOT trigger torch import; only `hodge` does.
"""

__all__: list[str] = []
