"""Allows ``python -m benchmarks`` to dispatch to the CLI.

Module-level dispatch: only executed via ``python -m benchmarks``. Tests
exercise ``benchmarks.cli.main`` directly, which is the equivalent
entry-point; running this from inside pytest would require fork/exec.
"""

from benchmarks.cli import main  # pragma: no cover

raise SystemExit(main())  # pragma: no cover
