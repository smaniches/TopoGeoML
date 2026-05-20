"""Allows ``python -m benchmarks`` to dispatch to the CLI."""

from benchmarks.cli import main

raise SystemExit(main())
