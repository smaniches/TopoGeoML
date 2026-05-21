"""Backend registry. Importing this module auto-registers all built-in backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.backends._protocol import PHBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

_REGISTRY: dict[str, type[PHBackend]] = {}


def register_backend(cls: type[PHBackend]) -> type[PHBackend]:
    """Decorator: add a backend class to the global registry.

    The class must expose ``name`` (str) and implement the `PHBackend`
    protocol. Registration is idempotent; re-registering the same ``name``
    raises ``ValueError`` to surface bugs at import time rather than at
    runtime.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"backend class {cls.__name__} must define a non-empty 'name' class attribute"
        )
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(
            f"backend name {name!r} is already registered to "
            f"{_REGISTRY[name].__name__}; cannot re-register {cls.__name__}"
        )
    _REGISTRY[name] = cls
    return cls


def available_backends() -> list[type[PHBackend]]:
    """Return registered backends whose underlying library is importable."""
    return [cls for cls in _REGISTRY.values() if cls.available()]


def get_backend(name: str) -> type[PHBackend]:
    """Look up a backend by name. Raises KeyError if not registered."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"no backend named {name!r}; registered: {known}")
    return _REGISTRY[name]


def all_backend_names() -> Iterator[str]:
    """Iterate over every registered backend name (regardless of availability)."""
    return iter(_REGISTRY)


# Importing the wrapper modules auto-registers them via the decorator below.
from benchmarks.backends import gudhi_python as _gudhi_python  # noqa: E402, F401
from benchmarks.backends import hofer_2017_reference as _hofer_2017  # noqa: E402, F401
from benchmarks.backends import topogeoml_diff_ph as _topogeoml  # noqa: E402, F401
from benchmarks.backends import torch_topological as _torch_top  # noqa: E402, F401

__all__ = [
    "PHBackend",
    "all_backend_names",
    "available_backends",
    "get_backend",
    "register_backend",
]
