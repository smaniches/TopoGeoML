"""Shared test fixtures: synthetic point clouds with known topology."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic random generator (elite-code-standards §6: seeded RNG)."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def noisy_circle(rng: np.random.Generator) -> NDArray[np.float64]:
    """50 points on the unit circle with σ=0.05 Gaussian noise. Expected: β₀=1, β₁=1."""
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False, dtype=np.float64)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    pts += 0.05 * rng.standard_normal(pts.shape)
    return np.ascontiguousarray(pts, dtype=np.float64)


@pytest.fixture(scope="session")
def noisy_line(rng: np.random.Generator) -> NDArray[np.float64]:
    """50 points on a line segment with σ=0.05 noise. Expected: β₀=1, β₁=0."""
    t = np.linspace(-1.0, 1.0, 50, dtype=np.float64)
    pts = np.stack([t, np.zeros_like(t)], axis=1)
    pts += 0.05 * rng.standard_normal(pts.shape)
    return np.ascontiguousarray(pts, dtype=np.float64)


@pytest.fixture(scope="session")
def noisy_two_circles(rng: np.random.Generator) -> NDArray[np.float64]:
    """Two disjoint noisy circles in R^2. Expected: β₀=2, β₁=2."""
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False, dtype=np.float64)
    c1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    c2 = np.stack([np.cos(theta) + 4.0, np.sin(theta)], axis=1)
    pts = np.concatenate([c1, c2], axis=0)
    pts += 0.05 * rng.standard_normal(pts.shape)
    return np.ascontiguousarray(pts, dtype=np.float64)
