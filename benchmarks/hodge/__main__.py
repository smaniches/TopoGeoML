"""``python -m benchmarks.hodge`` — Hodge-bench CLI."""

import argparse
import os
import sys
from pathlib import Path

from benchmarks.hodge.runner import render_markdown, run, write_result  # pragma: no cover


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="python -m benchmarks.hodge")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmarks/hodge/leaderboard/current.json"),
    )
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(
        model_names=args.models, dataset_names=args.datasets,
        seeds=args.seeds, n_epochs=args.n_epochs, learning_rate=args.lr,
    )
    write_result(result, args.output)
    md = render_markdown(result)
    print(md)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
