"""
Stability axis — the framework's centerpiece.

Differentiable persistent homology only becomes a useful loss/regularizer
when its output is *stable* under input perturbation. Three sub-measurements:

(1) Cohen-Steiner stability margin.

    Cohen-Steiner / Edelsbrunner / Harer (2007), with the Vietoris-Rips
    specialization from Chazal / de Silva / Oudot (2014): for two finite
    point clouds X, X' in a common ambient L_2 space,

    .. math::
       d_B\\bigl(\\mathrm{Dgm}_R(X), \\mathrm{Dgm}_R(X')\\bigr) \\;\\le\\; 2 \\cdot d_H^{L_2}(X, X').

    The factor of 2 arises because the Rips filtration value of a
    1-simplex :math:`\\{x_i, x_j\\}` is :math:`\\|x_i - x_j\\|_2`, and an
    L_2 Hausdorff perturbation of size :math:`\\delta` can shift both
    endpoints by up to :math:`\\delta` in opposite directions, changing
    the filtration value by up to :math:`2\\delta`.

    We measure the stability margin

    .. math::
       \\mathrm{margin} \\;=\\; 2 \\cdot d_H^{L_2}(X, X') \\;-\\; d_B(\\mathrm{Dgm}(X), \\mathrm{Dgm}(X')).

    Non-negative margins satisfy the theorem; persistently negative
    margins across seeds flag either a backend bug or a regression in
    the bottleneck-distance implementation.

    References for the factor-of-2 specialization:
    Chazal, F., de Silva, V., & Oudot, S. (2014). "Persistence stability
      for geometric complexes." *Geometriae Dedicata*, 173(1), 193-214.

(2) Gradient Lipschitz approximation.

    For a fixed loss :math:`L`, the empirical Lipschitz constant of
    :math:`\\nabla_X L` over input-space perturbations is
    :math:`\\sup_\\varepsilon \\frac{\\| \\nabla L(X+\\varepsilon) - \\nabla L(X) \\|_2}{\\| \\varepsilon \\|_2}`.
    Smaller is better for training: it means gradient descent is well-conditioned.

(3) Autograd gradient check.

    ``torch.autograd.gradcheck`` against numerical finite differences with
    a tight tolerance, on a small input where the comparison is exact.
    A backend that fails this is producing wrong gradients regardless of
    how fast it is.

References
----------
Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). "Stability of
  persistence diagrams." *Discrete & Computational Geometry*, 37(1), 103-120.
Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An
  Introduction*. American Mathematical Society. Chapter VIII.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from benchmarks.backends import PHBackend
from benchmarks.datasets import Dataset
from benchmarks.stats import bootstrap_ci


@dataclass(frozen=True)
class CohenSteinerResult:
    """Per-perturbation-level stability margin for Vietoris-Rips persistence.

    A backend satisfies the theorem iff every ``margin`` is :math:`\\ge 0`
    (within numerical slack). ``perturbation_inf_norm`` is the noise
    generation parameter; ``hausdorff_l2`` is the realized L_2 Hausdorff
    distance between point clouds; the bound is :math:`2 \\cdot d_H^{L_2}`.
    """

    perturbation_inf_norm: float
    hausdorff_l2: float
    bottleneck_distance: float
    cohen_steiner_bound: float  # = 2 * hausdorff_l2
    margin: float  # = cohen_steiner_bound - bottleneck_distance
    satisfies_theorem: bool


@dataclass(frozen=True)
class StabilityReport:
    backend_name: str
    backend_version: str
    dataset_name: str
    dataset_version: str
    n_points: int

    # (1) Cohen-Steiner — one result per (seed, perturbation_level)
    cohen_steiner_pairs: list[CohenSteinerResult]
    n_theorem_violations: int

    # (2) Gradient Lipschitz approximation — one estimate per seed
    gradient_lipschitz_estimates: list[float]
    lipschitz_median: float
    lipschitz_ci95_low: float
    lipschitz_ci95_high: float

    # (3) Autograd gradient check — pass/fail flag per seed
    gradcheck_passes: list[bool]
    gradcheck_pass_rate: float

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cohen_steiner_pairs"] = [asdict(r) for r in self.cohen_steiner_pairs]
        return d


# Default perturbation magnitudes, log-spaced. Chosen to span six orders of
# magnitude from machine-epsilon (1e-7 ≈ float32 eps) to "large perturbation
# that should still respect Cohen-Steiner because the theorem is uniform".
_DEFAULT_PERTURBATIONS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def _bottleneck_distance_finite(d1: np.ndarray, d2: np.ndarray) -> float:
    """Bottleneck distance between two persistence diagrams (finite bars only).

    Delegates to ``gudhi.bottleneck.bottleneck_distance`` — the authoritative
    implementation backed by the Kerber-Morozov-Nigmetov algorithm
    (Edelsbrunner & Harer 2010, Chapter VIII; Kerber-Morozov-Nigmetov,
    SoCG 2017). The bottleneck distance is the *min-max* matching across
    diagrams (with diagonal augmentation), distinct from the Wasserstein
    distances which are min-sum.

    Why not roll our own
    --------------------
    An earlier version of this function used ``scipy.optimize.linear_sum_assignment``,
    which minimizes the *sum* of L_inf costs (Hungarian algorithm = Wasserstein-1).
    That is mathematically different from the bottleneck distance
    (= W_infinity), and would systematically over-estimate :math:`d_B` on
    diagrams with multiple roughly-equally-large discrepancies. The
    Cohen-Steiner stability check is only meaningful with the true
    bottleneck distance; a min-sum surrogate produces both false negatives
    on the theorem (sum overestimates max) and false positives elsewhere.
    Gemini's review on PR #3 flagged this — fix lands here.

    Both ``d1`` and ``d2`` are ``(n, 2)`` arrays of (birth, death) pairs.
    Infinite-death bars are dropped — Cohen-Steiner bounds the finite part.
    """
    import gudhi.bottleneck

    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    d1 = d1[np.isfinite(d1).all(axis=1)] if d1.size else d1.reshape(0, 2)
    d2 = d2[np.isfinite(d2).all(axis=1)] if d2.size else d2.reshape(0, 2)

    if d1.shape[0] == 0 and d2.shape[0] == 0:
        return 0.0

    # gudhi accepts either diagram empty and handles the diagonal projection
    # internally with the correct semantics.
    return float(gudhi.bottleneck.bottleneck_distance(d1, d2))


def measure_stability(
    backend: type[PHBackend],
    dataset: Dataset,
    *,
    n_points: int = 50,
    seeds: list[int] | None = None,
    perturbation_inf_norms: tuple[float, ...] = _DEFAULT_PERTURBATIONS,
    gradcheck_n_points: int = 5,
    gradcheck_eps: float = 1e-6,
    gradcheck_atol: float = 1e-4,
    bootstrap_seed: int = 0,
) -> StabilityReport:
    """Run the three stability sub-measurements.

    Parameters
    ----------
    backend, dataset
        Registered backend class and dataset to evaluate.
    n_points
        Point cloud size for the Cohen-Steiner and Lipschitz measurements.
    seeds
        Seeds used for both data generation and perturbation. Default
        ``[0, 1, 2, 3, 4]`` — small for Phase 1 (Cohen-Steiner is per-seed
        × per-perturbation-level so the total cell count is already 5×6=30).
    perturbation_inf_norms
        Magnitudes of additive uniform noise tested against the theorem.
    gradcheck_n_points
        Point count for the autograd gradient check. Small because finite
        differences are O(n_input * n_output) per evaluation.
    gradcheck_eps, gradcheck_atol
        Forwarded to :func:`torch.autograd.gradcheck`.
    bootstrap_seed
        Seeds the Lipschitz-CI bootstrap.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    cs_results: list[CohenSteinerResult] = []
    lipschitz_estimates: list[float] = []
    gradcheck_passes: list[bool] = []

    for seed in seeds:
        # --- (1) Cohen-Steiner across perturbation levels --------------------
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X = dataset.generate(seed=seed, n_points=n_points)
        dgms_base = backend.compute_diagram(X, max_dim=1)
        h1_base = dgms_base[1].detach().numpy() if len(dgms_base) > 1 else np.empty((0, 2))

        for eps in perturbation_inf_norms:
            # Uniform[-eps, eps] perturbation — bounds per-coord noise by eps.
            noise = rng.uniform(-eps, eps, size=tuple(X.shape)).astype(np.float64)
            X_pert = X + torch.from_numpy(noise)
            dgms_pert = backend.compute_diagram(X_pert, max_dim=1)
            h1_pert = dgms_pert[1].detach().numpy() if len(dgms_pert) > 1 else np.empty((0, 2))

            # Upper bound on L_2 Hausdorff distance via the natural
            # point-wise correspondence x_i ↔ x_i + noise_i. The true
            # Hausdorff is bounded by max_i ||noise_i||_2; using the
            # upper bound makes our margin conservative (loose toward
            # passing), which is the right direction — we want to flag
            # a backend bug, not a tighter-than-necessary bound.
            per_point_l2 = np.linalg.norm(noise, axis=1)
            hausdorff_l2 = float(per_point_l2.max())
            cs_bound = 2.0 * hausdorff_l2  # Rips-stability factor of 2.

            db = _bottleneck_distance_finite(h1_base, h1_pert)
            margin = float(cs_bound - db)
            # Slack absorbs accumulated f64 round-off in
            # linear_sum_assignment and in ripser's filtration value
            # computation; 32 ulps relative to the bound magnitude is
            # conservative and well below any realistic backend error.
            slack = max(1e-12, 32.0 * np.finfo(np.float64).eps * max(cs_bound, db, 1.0))
            cs_results.append(CohenSteinerResult(
                perturbation_inf_norm=float(eps),
                hausdorff_l2=hausdorff_l2,
                bottleneck_distance=db,
                cohen_steiner_bound=cs_bound,
                margin=margin,
                satisfies_theorem=(margin >= -slack),
            ))

        # --- (2) Gradient Lipschitz approximation ----------------------------
        # Pick a perturbation at mid-scale; record ||grad(L; X+δ) - grad(L; X)||_2 / ||δ||_2.
        delta_scale = 1e-3
        X_req = X.detach().clone().requires_grad_(True)
        loss_base = backend.loss_longest_h1(X_req)
        loss_base.backward()  # type: ignore[no-untyped-call]
        grad_base = X_req.grad.detach().clone() if X_req.grad is not None else torch.zeros_like(X_req)

        X_pert_req = (X + torch.from_numpy(rng.uniform(-delta_scale, delta_scale, size=tuple(X.shape))).to(torch.float64)).detach().requires_grad_(True)
        loss_pert = backend.loss_longest_h1(X_pert_req)
        loss_pert.backward()  # type: ignore[no-untyped-call]
        grad_pert = X_pert_req.grad.detach().clone() if X_pert_req.grad is not None else torch.zeros_like(X_pert_req)

        delta = (X_pert_req - X_req).detach()
        delta_norm = float(torch.linalg.norm(delta).item())
        if delta_norm < 1e-12:  # pragma: no cover
            # Degenerate — skip rather than divide by ~0. The default
            # perturbation magnitude (1e-3) makes this practically
            # unreachable; covered as a safety net for callers that pass
            # delta_scale = 0 or hit a numerical wipeout.
            continue
        grad_diff_norm = float(torch.linalg.norm(grad_pert - grad_base).item())
        lipschitz_estimates.append(grad_diff_norm / delta_norm)

        # --- (3) Autograd gradcheck on a small input -------------------------
        # gradcheck is expensive: O(n_input * n_output) finite-difference passes.
        # Limit to gradcheck_n_points and wrap the loss to take a single
        # tensor argument (which is what gradcheck needs).
        X_gc = dataset.generate(seed=seed, n_points=gradcheck_n_points).requires_grad_(True)

        def _loss_for_gradcheck(z: torch.Tensor) -> torch.Tensor:
            return backend.loss_longest_h1(z)

        try:
            passed = torch.autograd.gradcheck(
                _loss_for_gradcheck,
                (X_gc,),
                eps=gradcheck_eps,
                atol=gradcheck_atol,
                raise_exception=False,
            )
        except RuntimeError:  # pragma: no cover
            # Some backends raise rather than return False on subgradient
            # discontinuities. We record this as a fail but do not crash.
            # The two backends shipped in Phase 1 respect raise_exception=False
            # cleanly; this branch is a safety net for backends added in
            # Phase 2+ whose gradcheck contract is looser.
            passed = False
        gradcheck_passes.append(bool(passed))

    # Bootstrap CI for the Lipschitz estimate's median, if we have ≥2 seeds.
    if len(lipschitz_estimates) >= 2:
        lc = bootstrap_ci(
            np.asarray(lipschitz_estimates),
            statistic="median",
            confidence_level=0.95,
            n_resamples=10_000,
            seed=bootstrap_seed,
        )
        lip_median, lip_lo, lip_hi = lc.point_estimate, lc.ci_low, lc.ci_high
    elif len(lipschitz_estimates) == 1:
        lip_median = lipschitz_estimates[0]
        lip_lo = lip_hi = float("nan")
    else:  # pragma: no cover
        # Reachable only when every seed hit the delta_norm < 1e-12 branch
        # above and skipped its Lipschitz contribution. Defensive.
        lip_median = lip_lo = lip_hi = float("nan")

    return StabilityReport(
        backend_name=backend.name,
        backend_version=getattr(backend, "version", "") or "",
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        n_points=n_points,
        cohen_steiner_pairs=cs_results,
        n_theorem_violations=sum(1 for r in cs_results if not r.satisfies_theorem),
        gradient_lipschitz_estimates=lipschitz_estimates,
        lipschitz_median=lip_median,
        lipschitz_ci95_low=lip_lo,
        lipschitz_ci95_high=lip_hi,
        gradcheck_passes=gradcheck_passes,
        gradcheck_pass_rate=(
            float(sum(gradcheck_passes)) / len(gradcheck_passes)
            if gradcheck_passes else 0.0
        ),
    )


__all__ = ["CohenSteinerResult", "StabilityReport", "measure_stability"]
