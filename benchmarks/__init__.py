"""
TopoGeoML benchmark framework.

Goal — be the most rigorous comparison suite available for differentiable
persistent homology in Python. Whoever ships the benchmark sets the
methodology; whoever sets the methodology defines what "best" means.

Design — pluggable backends (one wrapper per method), pluggable datasets
(one fixture per data source), pluggable axes (one measurement per dimension
of "best"). The orchestrator (`runner.py`) is a matrix product over the
three. Provenance is recorded for every result. The leaderboard is versioned
JSON; PRs that move numbers produce a structured diff.

See `benchmarks/README.md` for the methodology and how to add a backend.
"""
