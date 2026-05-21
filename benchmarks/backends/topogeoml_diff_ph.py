"""
Backend wrapper: TopoGeoML's `nn/diff_ph.py` (Elder Lemma + cocycle indexing).

The wrapped library is part of this repository; ``available()`` is True iff
PyTorch is installed in the current environment.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from benchmarks.backends import register_backend


@register_backend
class TopoGeoMLDiffPH:
    """TopoGeoML's differentiable Vietoris-Rips PH layer."""

    name: ClassVar[str] = "topogeoml-diff-ph"
    version: ClassVar[str] = ""  # populated in `available()`; see below
    differentiable: ClassVar[bool] = True

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401

            import topogeoml  # noqa: F401

            # ripser ships with topogeoml's core deps; diff_ph imports it.
            from topogeoml.nn import diff_ph as _diff_ph  # noqa: F401
        except ImportError:  # pragma: no cover  -- defensive: this PR's CI installs topogeoml.
            return False

        import topogeoml as _topogeoml

        # Populate the class-level version lazily — only after a successful
        # availability check, so the registry can read it.
        TopoGeoMLDiffPH.version = str(_topogeoml.__version__)
        return True

    @staticmethod
    def compute_diagram(X: torch.Tensor, max_dim: int) -> list[torch.Tensor]:
        from topogeoml.nn.diff_ph import rips_diagram_torch

        if X.dtype != torch.float64:
            raise TypeError(
                f"{TopoGeoMLDiffPH.name}: input X must be float64, got {X.dtype}"
            )
        return rips_diagram_torch(X, max_dim=max_dim)

    @staticmethod
    def loss_longest_h1(X: torch.Tensor) -> torch.Tensor:
        from topogeoml.nn.diff_ph import rips_diagram_torch

        diagrams = rips_diagram_torch(X, max_dim=1)
        h1 = diagrams[1] if len(diagrams) > 1 else X.new_empty((0, 2))
        finite_mask = torch.isfinite(h1).all(dim=1) if h1.numel() else torch.zeros(0, dtype=torch.bool)
        finite = h1[finite_mask] if h1.numel() else h1
        if finite.numel() == 0:
            # 0 * sum(X) — depends on X for autograd safety, contributes no gradient.
            return torch.zeros((), dtype=X.dtype, device=X.device) + 0.0 * X.sum()
        return -(finite[:, 1] - finite[:, 0]).max()
