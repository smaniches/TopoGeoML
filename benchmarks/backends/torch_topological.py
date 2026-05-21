"""
Backend wrapper: torch-topological (Bastian Rieck et al.).

PyPI package ``torch-topological``. Uses gudhi/ripser internally; provides
``VietorisRipsComplex`` returning ``PersistenceInformation`` objects.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from benchmarks.backends import register_backend


@register_backend
class TorchTopological:
    name: ClassVar[str] = "torch-topological"
    version: ClassVar[str] = ""
    differentiable: ClassVar[bool] = True

    @staticmethod
    def available() -> bool:
        try:
            import torch_topological
        except ImportError:  # pragma: no cover  -- defensive: bench extras install torch-topological.
            return False
        TorchTopological.version = str(torch_topological.__version__)
        return True

    @staticmethod
    def compute_diagram(X: torch.Tensor, max_dim: int) -> list[torch.Tensor]:
        from torch_topological.nn import VietorisRipsComplex

        if X.dtype != torch.float64:
            raise TypeError(
                f"{TorchTopological.name}: input X must be float64, got {X.dtype}"
            )
        vr = VietorisRipsComplex(dim=max_dim)
        result = vr.forward(X)
        # Convert the heterogeneous result into the same list-of-bars-tensor
        # layout that PHBackend.compute_diagram is required to return.
        by_dim: dict[int, torch.Tensor] = {}
        for info in result:
            by_dim[int(info.dimension)] = info.diagram
        return [
            by_dim.get(k, X.new_empty((0, 2), dtype=torch.float64))
            for k in range(max_dim + 1)
        ]

    @staticmethod
    def loss_longest_h1(X: torch.Tensor) -> torch.Tensor:
        from torch_topological.nn import VietorisRipsComplex

        vr = VietorisRipsComplex(dim=1)
        diagrams = vr.forward(X)
        h1_info = next((r for r in diagrams if int(r.dimension) == 1), None)
        if h1_info is None or h1_info.diagram.numel() == 0:
            return torch.zeros((), dtype=X.dtype, device=X.device) + 0.0 * X.sum()
        dgm = h1_info.diagram
        finite_mask = torch.isfinite(dgm).all(dim=1)
        finite = dgm[finite_mask]
        if finite.numel() == 0:  # pragma: no cover
            # Defensive: torch-topological with ``keep_infinite_features=False``
            # (our default) does not emit infinite-death bars in dim=1 on
            # finite point clouds. This guard would fire if the upstream
            # behavior changes to include essential H_1 classes.
            return torch.zeros((), dtype=X.dtype, device=X.device) + 0.0 * X.sum()
        max_lifetime: torch.Tensor = (finite[:, 1] - finite[:, 0]).max()
        return -max_lifetime
