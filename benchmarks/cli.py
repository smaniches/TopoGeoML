"""CLI entry: ``python -m benchmarks``.

Single command. Default behavior: run every available backend × dataset ×
axis combination, write JSON to ``leaderboard/current.json``, write
markdown to ``$GITHUB_STEP_SUMMARY`` (if set) and stdout.

For local exploration, the user can restrict the run with ``--backends``,
``--datasets``, ``--axes``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from benchmarks.report import render_markdown
from benchmarks.runner import run, write_result


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
    args = parser.parse_args(argv)

    result = run(
        backend_names=args.backends,
        dataset_names=args.datasets,
        axis_names=args.axes,
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

    # Return non-zero if any cell failed — CI should be loud about that.
    failed_cells = [c for c in result.cells if not c.success]
    return 1 if failed_cells else 0


if __name__ == "__main__":
    sys.exit(main())
