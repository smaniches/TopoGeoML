"""CLI entry: ``python -m benchmarks``.

Single command. Default behavior: run every available backend × dataset ×
axis combination, write JSON to ``leaderboard/current.json``, write
markdown to ``$GITHUB_STEP_SUMMARY`` (if set) and stdout.

For local exploration, the user can restrict the run with ``--backends``,
``--datasets``, ``--axes``.

The ``--quick`` flag thins per-axis seed/repeat counts so the full bench
fits under CI's 30-minute wall-clock budget. The full-rigor numbers come
from running without ``--quick`` (locally or on a GPU runner with extra
budget); the configuration is recorded in the JSON output so it cannot be
silently confused with full-rigor data.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from benchmarks.report import render_markdown
from benchmarks.runner import run, write_result


def _quick_axis_kwargs() -> dict[str, dict[str, Any]]:
    """CI smoke-tier overrides: cuts wall clock by ~3x while still
    exercising every axis on every (available) backend × dataset cell.

    Trade-offs:
      - ``seeds`` is shortened to 3 from 5 across every axis — bootstrap
        CIs widen but directional comparisons remain interpretable
        (the report surfaces the seed count alongside every number).
      - ``speed.n_points_list`` drops the n=300 size which is dominated
        by the per-call cost of large-n Vietoris-Rips; the small/medium
        scales still discriminate backends.
      - ``speed`` outer × inner is halved (3 × 10 = 30 measurements vs
        5 × 20 = 100). min-of-medians remains the estimator, which is
        robust to the smaller window.
      - ``optimization.n_steps`` drops from 200 to 60. Empirically
        diff-PH optimizations converge within 30-50 steps on the
        synthetic fixtures used here; 60 is comfortable.
    """
    return {
        "correctness": {"seeds": [0, 1, 2]},
        "stability": {"seeds": [0, 1, 2]},
        "speed": {
            "seeds": [0, 1, 2],
            "n_points_list": [30, 100],
            "repeat": 3,
            "number": 10,
        },
        "optimization": {"seeds": [0, 1, 2], "n_steps": 60},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=(
            "Run the TopoGeoML differentiable-PH benchmark with full "
            "statistical reporting and provenance."
        ),
    )
    parser.add_argument("--backends", nargs="+", default=None,
                        help="restrict to these backend names (default: all available)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="restrict to these dataset names (default: all registered)")
    parser.add_argument("--axes", nargs="+", default=None,
                        choices=["correctness", "stability", "speed", "optimization"],
                        help="restrict to these axes (default: all)")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/leaderboard/current.json"),
                        help="JSON output path (default: benchmarks/leaderboard/current.json)")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="optional path to write the markdown report")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Thin per-axis seed/repeat counts to fit under CI's 30-min "
            "budget. Statistical power is roughly 3x reduced. Use for "
            "framework validation; full-rigor numbers require running "
            "without --quick (locally or on a longer-budget runner)."
        ),
    )
    args = parser.parse_args(argv)

    axis_kwargs = _quick_axis_kwargs() if args.quick else None
    result = run(
        backend_names=args.backends,
        dataset_names=args.datasets,
        axis_names=args.axes,
        axis_kwargs=axis_kwargs,
    )
    write_result(result, args.output)

    md = render_markdown(result.as_dict())
    print(md)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)

    # When run in GitHub Actions, also write the markdown to the step summary.
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(md)

    # Return non-zero on *unexpected* failures only. Two cell categories
    # are expected behaviour and must not fail the workflow:
    #
    #   - ``SkippedNonDifferentiable``: non-differentiable backends like
    #     ``gudhi-python`` cannot satisfy autograd-required axes, so the
    #     runner emits a clean skip rather than silently omitting the row.
    #   - ``UnavailableBackend``: a registered backend whose optional
    #     dependency is not installed in the current environment. The
    #     runner records the absence so reports can flag missing
    #     comparison points, but it is not a *bug* — it is environment
    #     configuration.
    #
    # The previous "any non-success → exit 1" check counted both as
    # failures and made the CPU ``benchmark.yml`` workflow fail on every
    # PR that included a non-differentiable backend in the registry.
    EXPECTED_NON_FAILURES = frozenset({
        "SkippedNonDifferentiable",
        "UnavailableBackend",
    })
    real_failures = [
        c for c in result.cells
        if not c.success and c.error_kind not in EXPECTED_NON_FAILURES
    ]
    if real_failures:
        # Surface the failing cells on stderr so CI logs make the cause
        # obvious without requiring the JSON artifact to be downloaded.
        print(
            f"\n[benchmarks] {len(real_failures)} unexpected cell failure(s):",
            file=sys.stderr,
        )
        for c in real_failures:
            print(
                f"  - {c.backend_name} / {c.dataset_name} / {c.axis_name}: "
                f"{c.error_kind}: {(c.error_message or '')[:200]}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
