"""
Tests for ``notebooks/topology_predicts_divergence.py``.

The empirical claim itself is tested by running the script (out of band);
these unit tests cover the deterministic helpers + CLI + the smoke
training path so a regression in any of those will fail CI.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Imports / module health
# ---------------------------------------------------------------------------

class TestModuleImports:
    def test_module_importable(self) -> None:
        import notebooks.topology_predicts_divergence as mod
        assert callable(mod.main)
        assert callable(mod._build_argparser)
        assert callable(mod._loss_watchdog_step)
        assert callable(mod._topology_watchdog_step)


class TestCli:
    def test_argparser_defaults(self) -> None:
        from notebooks.topology_predicts_divergence import _build_argparser

        ns = _build_argparser().parse_args([])
        assert ns.n_seeds == 30
        assert ns.n_steps == 600
        assert ns.probe_every == 10
        assert ns.divergence_threshold == pytest.approx(2.0)
        assert ns.overfit_ratio == pytest.approx(1.20)
        assert ns.smoke is False

    def test_smoke_flag_scales_down(self) -> None:
        from notebooks.topology_predicts_divergence import _build_argparser

        ns = _build_argparser().parse_args(["--smoke"])
        assert ns.smoke is True


# ---------------------------------------------------------------------------
# Watchdog logic (pure functions, no training).
# ---------------------------------------------------------------------------

class TestLossWatchdog:
    def test_returns_first_step_above_ratio(self) -> None:
        from notebooks.topology_predicts_divergence import _loss_watchdog_step

        # Val loss declines, then climbs above ratio × running_min.
        series = [(0, 1.0), (10, 0.5), (20, 0.4), (30, 0.5), (40, 0.6)]
        # Running min after step 20 is 0.4. Step 30: 0.5 > 1.20 * 0.4 = 0.48 → fires.
        assert _loss_watchdog_step(series, overfit_ratio=1.20) == 30

    def test_fires_only_when_clearly_above_ratio(self) -> None:
        from notebooks.topology_predicts_divergence import _loss_watchdog_step

        # A series where the ratio is borderline at step 30 but exceeded at 40.
        series = [(0, 1.0), (10, 0.5), (20, 0.4), (30, 0.45), (40, 0.60)]
        # Step 30: 0.45 < 1.20 * 0.4 = 0.48; step 40: 0.60 > 0.48 → fires there.
        assert _loss_watchdog_step(series, overfit_ratio=1.20) == 40

    def test_never_fires_when_monotone_decreasing(self) -> None:
        from notebooks.topology_predicts_divergence import _loss_watchdog_step

        series = [(0, 1.0), (10, 0.9), (20, 0.7), (30, 0.5)]
        assert _loss_watchdog_step(series, overfit_ratio=1.20) is None

    def test_empty_series_returns_none(self) -> None:
        from notebooks.topology_predicts_divergence import _loss_watchdog_step

        assert _loss_watchdog_step([], overfit_ratio=1.20) is None


class TestTopologyWatchdog:
    def test_returns_first_step_above_threshold(self) -> None:
        from notebooks.topology_predicts_divergence import _topology_watchdog_step

        series = [(0, 0.0), (10, 0.5), (20, 1.5), (30, 2.5), (40, 3.0)]
        assert _topology_watchdog_step(series, threshold=2.0) == 30

    def test_never_fires_when_all_below_threshold(self) -> None:
        from notebooks.topology_predicts_divergence import _topology_watchdog_step

        series = [(0, 0.5), (10, 1.0), (20, 1.5)]
        assert _topology_watchdog_step(series, threshold=2.0) is None

    def test_empty_series_returns_none(self) -> None:
        from notebooks.topology_predicts_divergence import _topology_watchdog_step

        assert _topology_watchdog_step([], threshold=2.0) is None


# ---------------------------------------------------------------------------
# Dataclass / helpers
# ---------------------------------------------------------------------------

class TestSeedResult:
    def test_detection_advantage_both_fired(self) -> None:
        from notebooks.topology_predicts_divergence import SeedResult

        r = SeedResult(
            seed=0, loss_watchdog_step=100, topology_watchdog_step=70,
            final_step=599, final_train_loss=0.01, final_val_loss=0.30,
            n_topology_snapshots=60,
        )
        # Topology fired 30 steps earlier than loss.
        assert r.detection_advantage() == 30

    def test_detection_advantage_negative_means_loss_earlier(self) -> None:
        from notebooks.topology_predicts_divergence import SeedResult

        r = SeedResult(
            seed=0, loss_watchdog_step=50, topology_watchdog_step=80,
            final_step=599, final_train_loss=0.01, final_val_loss=0.30,
            n_topology_snapshots=60,
        )
        assert r.detection_advantage() == -30

    @pytest.mark.parametrize("loss_step,topo_step", [(None, 100), (100, None), (None, None)])
    def test_detection_advantage_none_when_either_missing(
        self, loss_step: int | None, topo_step: int | None,
    ) -> None:
        from notebooks.topology_predicts_divergence import SeedResult

        r = SeedResult(
            seed=0, loss_watchdog_step=loss_step, topology_watchdog_step=topo_step,
            final_step=599, final_train_loss=0.01, final_val_loss=0.30,
            n_topology_snapshots=60,
        )
        assert r.detection_advantage() is None


# ---------------------------------------------------------------------------
# Reporting (light functional test).
# ---------------------------------------------------------------------------

class TestRendering:
    def test_render_markdown_insufficient_data(self) -> None:
        from notebooks.topology_predicts_divergence import (
            SeedResult,
            _render_markdown,
        )

        # 2 paired seeds — below the n>=5 threshold for a statistical claim.
        results = [
            SeedResult(seed=0, loss_watchdog_step=100, topology_watchdog_step=80,
                       final_step=199, final_train_loss=0.01, final_val_loss=0.30,
                       n_topology_snapshots=20),
            SeedResult(seed=1, loss_watchdog_step=120, topology_watchdog_step=70,
                       final_step=199, final_train_loss=0.01, final_val_loss=0.31,
                       n_topology_snapshots=20),
        ]
        md = _render_markdown(results)
        assert "Insufficient paired data" in md
        assert "## Per-seed raw data" in md

    def test_render_markdown_full_statistical_block(self) -> None:
        from notebooks.topology_predicts_divergence import (
            SeedResult,
            _render_markdown,
        )

        # 6 paired seeds with a clear positive advantage and no floor-effect tie.
        results = [
            SeedResult(
                seed=i, loss_watchdog_step=100 + 5 * i,
                topology_watchdog_step=60 + 5 * i,
                final_step=199, final_train_loss=0.01, final_val_loss=0.30,
                n_topology_snapshots=20,
            )
            for i in range(6)
        ]
        md = _render_markdown(results)
        # Headline statistics present.
        assert "BCa 95% CI" in md
        assert "Paired Wilcoxon" in md
        assert "Median detection advantage" in md
        # Topology firings span multiple steps → no floor-effect disclosure.
        assert "Floor-effect disclosure" not in md

    def test_render_markdown_surfaces_floor_disclosure(self) -> None:
        """When every topology firing lands on the same step (the
        baseline-window floor), the report must surface a floor-effect
        disclosure so readers don't misread the censored magnitude."""
        from notebooks.topology_predicts_divergence import (
            SeedResult,
            _render_markdown,
        )

        results = [
            SeedResult(
                seed=i, loss_watchdog_step=30 + 10 * (i % 4),
                topology_watchdog_step=30,  # same for every seed
                final_step=199, final_train_loss=0.01, final_val_loss=0.30,
                n_topology_snapshots=20,
            )
            for i in range(6)
        ]
        md = _render_markdown(results)
        assert "Floor-effect disclosure" in md

    def test_render_markdown_significant_uncorrected_verdict(self) -> None:
        """A consistently positive advantage produces a ``significant
        (uncorrected)`` verdict — distinct from the ``not_significant``
        placeholder ``compare_paired`` returns before BH."""
        from notebooks.topology_predicts_divergence import (
            SeedResult,
            _render_markdown,
        )

        results = [
            SeedResult(
                seed=i, loss_watchdog_step=80 + i,
                topology_watchdog_step=40,
                final_step=199, final_train_loss=0.01, final_val_loss=0.30,
                n_topology_snapshots=20,
            )
            for i in range(8)
        ]
        md = _render_markdown(results)
        assert "significant (uncorrected)" in md
        assert "topology earlier:" in md

    def test_render_markdown_counts_match(self) -> None:
        from notebooks.topology_predicts_divergence import (
            SeedResult,
            _render_markdown,
        )

        # Mixed firings: 2 both, 1 loss only, 1 topo only, 1 neither.
        results = [
            SeedResult(seed=0, loss_watchdog_step=100, topology_watchdog_step=70,
                       final_step=199, final_train_loss=0, final_val_loss=0,
                       n_topology_snapshots=20),
            SeedResult(seed=1, loss_watchdog_step=110, topology_watchdog_step=80,
                       final_step=199, final_train_loss=0, final_val_loss=0,
                       n_topology_snapshots=20),
            SeedResult(seed=2, loss_watchdog_step=100, topology_watchdog_step=None,
                       final_step=199, final_train_loss=0, final_val_loss=0,
                       n_topology_snapshots=20),
            SeedResult(seed=3, loss_watchdog_step=None, topology_watchdog_step=80,
                       final_step=199, final_train_loss=0, final_val_loss=0,
                       n_topology_snapshots=20),
            SeedResult(seed=4, loss_watchdog_step=None, topology_watchdog_step=None,
                       final_step=199, final_train_loss=0, final_val_loss=0,
                       n_topology_snapshots=20),
        ]
        md = _render_markdown(results)
        assert "Seeds attempted: 5" in md
        assert "Seeds with both watchdogs firing: 2" in md
        assert "Seeds with loss only: 1" in md
        assert "Seeds with topology only: 1" in md
        assert "Seeds where neither fired: 1" in md


# ---------------------------------------------------------------------------
# End-to-end smoke (one tiny seed) — exercises the training loop.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("sklearn", reason="sklearn not installed") is None,  # type: ignore[truthy-bool]
    reason="sklearn not installed",
)
class TestTrainingLoopSmoke:
    def test_one_seed_returns_well_formed_result(self) -> None:
        from notebooks.topology_predicts_divergence import _train_one_seed

        r = _train_one_seed(
            seed=0,
            n_train=80, n_val=80,
            n_steps=40,
            learning_rate=1e-2,
            hidden=16,
            probe_every=10,
            divergence_threshold=2.0,
            overfit_ratio=1.20,
        )
        # Either fire or don't — both are allowed in such a short run.
        assert r.loss_watchdog_step is None or 0 <= r.loss_watchdog_step < 40
        assert r.topology_watchdog_step is None or 0 <= r.topology_watchdog_step < 40
        assert r.n_topology_snapshots == 4  # steps 0, 10, 20, 30
        assert r.final_step == 39
