"""
Synthetic point-cloud fixtures.

Synthetic data is the foundation of the bench: deterministic, controllable
ground truth, fast to generate, no licensing concerns. Real-data datasets
(MUTAG, MNIST topology, DRIVE) land in Phase 3 of the framework roadmap.

Every fixture documents its expected Betti numbers so the correctness axis
can compare backend output against ground truth, not just against ripser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from benchmarks.datasets import register_dataset


@dataclass(frozen=True)
class NoisyCircle:
    """Points on the unit circle with isotropic Gaussian noise.

    Expected topology: β_0 = 1, β_1 = 1 below the noise scale.
    """

    name: str = "noisy_circle"
    version: str = "1.0.0"
    noise: float = 0.05  # standard deviation of the additive Gaussian

    def generate(self, seed: int, n_points: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
        pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        pts += (self.noise * rng.standard_normal(pts.shape)).astype(np.float64)
        return torch.from_numpy(np.ascontiguousarray(pts)).to(torch.float64)

    def expected_h1(self, n_points: int) -> int:
        return 1


@dataclass(frozen=True)
class TwoCircles:
    """Two disjoint unit circles, centered at (-2, 0) and (+2, 0).

    Expected topology: β_0 = 2, β_1 = 2.
    """

    name: str = "two_circles"
    version: str = "1.0.0"
    noise: float = 0.05
    separation: float = 4.0

    def generate(self, seed: int, n_points: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        n_left = n_points // 2
        n_right = n_points - n_left
        theta_left = np.linspace(0.0, 2.0 * np.pi, n_left, endpoint=False, dtype=np.float64)
        theta_right = np.linspace(0.0, 2.0 * np.pi, n_right, endpoint=False, dtype=np.float64)
        left = np.stack([np.cos(theta_left) - self.separation / 2, np.sin(theta_left)], axis=1)
        right = np.stack([np.cos(theta_right) + self.separation / 2, np.sin(theta_right)], axis=1)
        pts = np.concatenate([left, right], axis=0)
        pts += (self.noise * rng.standard_normal(pts.shape)).astype(np.float64)
        return torch.from_numpy(np.ascontiguousarray(pts)).to(torch.float64)

    def expected_h1(self, n_points: int) -> int:
        return 2


@dataclass(frozen=True)
class GaussianBlob:
    """Isotropic 2D Gaussian point cloud — no topological signal.

    Expected topology: β_0 = 1, β_1 = 0. Spurious short bars are expected
    at the noise scale; the optimization axis uses this as a control.
    """

    name: str = "gaussian_blob"
    version: str = "1.0.0"
    scale: float = 1.0

    def generate(self, seed: int, n_points: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        pts = (self.scale * rng.standard_normal((n_points, 2))).astype(np.float64)
        return torch.from_numpy(np.ascontiguousarray(pts)).to(torch.float64)

    def expected_h1(self, n_points: int) -> int:
        return 0


register_dataset(NoisyCircle())
register_dataset(TwoCircles())
register_dataset(GaussianBlob())
