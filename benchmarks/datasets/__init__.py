"""Dataset registry. Adding a dataset = adding a file here + registering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


@runtime_checkable
class Dataset(Protocol):
    """A reproducible source of point clouds for the bench.

    The ``name`` and ``version`` attributes are read at registration time
    and are immutable thereafter; we expose them as properties so frozen
    dataclasses implementing this protocol satisfy mypy strict mode.
    """

    @property
    def name(self) -> str:
        """Stable identifier; used in leaderboard JSON keys."""

    @property
    def version(self) -> str:
        """Semver-style version. Bumping invalidates prior leaderboard entries."""

    def generate(self, seed: int, n_points: int) -> torch.Tensor:
        """
        Produce a deterministic ``(n_points, d)`` float64 point cloud
        seeded by ``seed``. Must be a no-op w.r.t. the global RNG state
        (uses a local ``numpy.random.default_rng``).
        """
        ...

    def expected_h1(self, n_points: int) -> int:
        """Expected H_1 Betti number; used by the correctness axis as a sanity check."""
        ...


_REGISTRY: dict[str, Dataset] = {}


def register_dataset(dataset: Dataset) -> Dataset:
    if dataset.name in _REGISTRY and _REGISTRY[dataset.name] is not dataset:
        raise ValueError(f"dataset {dataset.name!r} already registered")
    _REGISTRY[dataset.name] = dataset
    return dataset


def get_dataset(name: str) -> Dataset:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"no dataset named {name!r}; registered: {known}")
    return _REGISTRY[name]


def all_dataset_names() -> Iterator[str]:
    return iter(_REGISTRY)


from benchmarks.datasets import mnist_topology as _mnist  # noqa: E402, F401

__all__ = ["Dataset", "all_dataset_names", "get_dataset", "register_dataset"]
