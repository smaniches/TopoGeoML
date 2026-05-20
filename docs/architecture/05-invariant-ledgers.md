# Invariant Ledgers

**STATUS:** PROPOSED
**Phase:** v0.1
**Spec source:** TopoGeomML Specification Part II §7
**Depends on:** [03-claim-graphs.md](03-claim-graphs.md)

---

## What it is

An invariant ledger is an append-only log of topological invariants — Betti numbers, persistence entropy, descriptor moments, cohort-stratified versions of all of these — observed across model versions, training runs, and deployment windows.

A new entry is appended every time a new evidence bundle is produced. Regression detection runs as a query: "which invariants in entry N differ from entry N-1 beyond the configured threshold?"

## Why it matters

Most ML monitoring tracks input-output distributions and prediction quality. None tracks the **structure of representations** across time. A model whose accuracy is stable while its latent β₁ drifts from 12 to 4 between releases has undergone something significant that point metrics miss. The invariant ledger makes that drift observable.

The ledger is also the **time axis** of the claim graph. A claim at version V can be compared to its analog at V-1, V-2, V-N. Trends become first-class.

## Core constructs

- `Invariant` — a named scalar or vector quantity with a defined extraction procedure
- `Entry` — one observation: version, timestamp, invariant values, evidence reference
- `Ledger` — the append-only log itself
- `RegressionRule` — declarative test that flags an entry against a baseline
- `Baseline` — a fixed entry or rolling window used for comparison

## Open questions

- [ ] **Storage**: JSONL is simple and append-friendly; SQLite is queryable; a real time-series DB is overkill. Lean toward JSONL with optional SQLite index.
- [ ] **Baseline selection**: previous release? Rolling mean of last N? Per-cohort baselines?
- [ ] **Threshold language**: absolute (β₁ changed by >2)? relative (β₁ changed by >20%)? Statistical (>2σ over rolling window)?
- [ ] **Invariant inventory**: what's the **minimum useful set** for v0.1? Candidate: β₀, β₁, persistence entropy H₀ / H₁, total persistence H₁, cohort-stratified versions of all four.

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | None directly; the JSON output schema is per-run, no cross-run aggregation |
| v0.1 | `topogeoml.ledger.Ledger` with JSONL storage, append/query/compare API |
| v0.2 | Declarative regression rules; CLI `topogeoml ledger compare v3.7 v3.8` |
| v1.0 | Ledger as a first-class deployment-gate input |
