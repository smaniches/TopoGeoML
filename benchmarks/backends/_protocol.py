"""
PHBackend protocol — the contract every differentiable-PH library wrapper
must satisfy.

Adding a new method to the benchmark is a single file in `benchmarks/backends/`
that defines a class implementing this protocol and decorates it with
`@register_backend`. No other changes are required; the runner discovers it
via the registry.

Each backend must:

  1. Report ``available()`` honestly — return False if the underlying
     library is not importable in the current environment. The runner
     will skip unavailable backends without erroring.

  2. Compute persistence diagrams that agree with ripser (the reference)
     on synthetic inputs to within ``atol = 1e-6`` for finite bars.
     This is asserted by ``tests/test_benchmarks.py``.

  3. Preserve ``torch.float64`` from input to diagram output. The
     correctness axis asserts this on every backend.

  4. Produce diagrams that carry gradients back to the input point
     cloud through ``torch.autograd``. The optimization axis checks
     that ``loss.backward()`` populates ``X.grad`` with finite values.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import torch


@runtime_checkable
class PHBackend(Protocol):
    """A differentiable persistent-homology implementation."""

    #: Short identifier used in reports and leaderboard JSON.
    #: Lowercase, hyphenated, stable across versions of the wrapper.
    name: ClassVar[str]

    #: Version string of the wrapped library at registration time.
    #: Populated by ``__init_subclass__``/``register_backend``.
    version: ClassVar[str]

    @staticmethod
    def available() -> bool:
        """Return True iff the wrapped library is importable in this env."""
        ...

    @staticmethod
    def compute_diagram(X: torch.Tensor, max_dim: int) -> list[torch.Tensor]:
        """
        Compute the Vietoris-Rips persistence diagram of ``X`` up to
        homology dimension ``max_dim`` (inclusive).

        Parameters
        ----------
        X : torch.Tensor
            ``(n, d)`` float64 point cloud, ``requires_grad`` either True
            or False; the returned tensors must inherit autograd from ``X``
            when ``X.requires_grad`` is True.
        max_dim : int
            Highest homology dimension to compute.

        Returns
        -------
        list[torch.Tensor]
            ``[H_0_bars, H_1_bars, ..., H_max_dim_bars]``. Each bars tensor
            is ``(n_bars, 2)`` float64 with columns ``(birth, death)``.
            Infinite-death bars are allowed; the correctness axis filters
            them with an explicit ``isfinite`` mask before reduction.
        """
        ...

    @staticmethod
    def loss_longest_h1(X: torch.Tensor) -> torch.Tensor:
        """
        Differentiable scalar: ``-max(d_i - b_i)`` over finite H_1 bars.

        Returns a 0-dim tensor that gradient-descends X toward larger H_1
        loops. If no finite H_1 bar exists, returns a 0-valued tensor that
        still depends on X (so autograd doesn't break on the empty case).
        """
        ...
