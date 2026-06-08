"""``python -m benchmarks.hodge`` — Hodge-bench CLI."""

import argparse
import contextlib
import os
import sys
from pathlib import Path

from benchmarks.hodge.runner import render_markdown, run, write_result  # pragma: no cover


def _reconfigure_stdout_utf8() -> None:  # pragma: no cover
    """Best-effort switch of stdout/stderr to UTF-8.

    The markdown report uses non-ASCII glyphs; on a cp1252 Windows console a
    bare ``print`` of it raises ``UnicodeEncodeError``. ``reconfigure`` exists
    on standard ``TextIOWrapper`` streams (Python 3.7+); a redirected or
    wrapped stream may lack it, so the call is guarded.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError, TypeError):
                reconfigure(encoding="utf-8")


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
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help=(
            "Optional cap on dataset size. If set, subsample (per seed, "
            "deterministically) before the stratified train/test split. "
            "Used by hypothesis 004 to isolate sample-size as the "
            "mechanism behind the residual-vs-MLP effect on NCI1."
        ),
    )
    parser.add_argument(
        "--feature-projection-dim",
        type=int,
        default=None,
        help=(
            "Optional target dimensionality for a per-seed deterministic "
            "Gaussian projection of all node features. Used by hypothesis "
            "005 to isolate feature dim as a candidate mechanism: setting "
            "this to 7 on NCI1 produces NCI1-7d (MUTAG dim, NCI1 size); "
            "setting it to 37 on MUTAG produces MUTAG-37d (NCI1 dim, "
            "MUTAG size)."
        ),
    )
    parser.add_argument(
        "--constant-features",
        action="store_true",
        help=(
            "Replace each graph's node features with a constant 1-vector "
            "of shape (n_nodes, 1). The MLP baseline then sees no "
            "node-level information and falls to the class prior; the "
            "Hodge model can still use the Laplacian. Used by hypothesis "
            "006 to measure the pure-topology classification signal in "
            "a dataset."
        ),
    )
    args = parser.parse_args(argv)

    # The markdown report contains non-ASCII glyphs (Delta, times, ...). On
    # Windows the console defaults to cp1252, so a bare ``print(md)`` raises
    # UnicodeEncodeError. Reconfigure stdout to UTF-8 where supported.
    _reconfigure_stdout_utf8()

    result = run(
        model_names=args.models, dataset_names=args.datasets,
        seeds=args.seeds, n_epochs=args.n_epochs, learning_rate=args.lr,
        max_graphs=args.max_graphs,
        feature_projection_dim=args.feature_projection_dim,
        constant_features=args.constant_features,
    )
    # Persist all file artifacts before any console print so a console
    # encoding failure can never lose results.
    write_result(result, args.output)
    md = render_markdown(result)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md, encoding="utf-8")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(md, encoding="utf-8")
    # File artifacts are already persisted; if UTF-8 reconfiguration was not
    # supported (e.g. a redirected cp1252 stream), fall back to a lossy encode
    # so the run never crashes at the final console print.
    try:
        print(md)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(md.encode(encoding, errors="replace").decode(encoding))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
