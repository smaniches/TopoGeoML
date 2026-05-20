# Benchmark Foundry

**STATUS:** PROPOSED
**Phase:** v0.1
**Spec source:** TopoGeomML Specification Part II §14

---

## What it is

A reproducible cross-task benchmark harness. Run the same set of topology-aware pipelines against a curated set of tasks (synthetic and real), produce a standard report comparing TopoGeomML configurations against published baselines.

The Foundry is not a single benchmark — it is the **infrastructure** that turns a config + a dataset + a baseline-set into a reproducible comparison artifact.

## Why it matters

A package without a benchmark suite is a toy. A package whose benchmark suite is run from a notebook by one researcher is a toy that pretends. The Foundry forces:

- Datasets ingested via standardized loaders
- Pipelines specified via YAML
- Results emitted in a standard schema
- Comparisons against published baselines tracked as named contenders
- Output as a Benchmark Foundry Report (a markdown + JSON pair shippable to a paper appendix or PR)

Without the Foundry, every claim about TopoGeomML's value is a personal communication.

## Target datasets (initial list, v0.1)

- Synthetic shapes (already in v0.0.1)
- MUTAG (small molecule classification)
- PROTEINS (protein structure classification)
- IMDB-BINARY (social network classification)
- COX2 (chemical compound activity)
- ZINC subset (molecular property regression)

## Core constructs

- `BenchmarkTask` — dataset loader + train/test split + metric + baseline registry
- `Contender` — named pipeline (TopoGeomML config or external baseline) registered for a task
- `BenchmarkReport` — the standard output: a `BenchmarkFoundry.Report` JSON + a markdown rendering
- `FoundryRunner` — runs all (task × contender) pairs and emits the report

## Open questions

- [ ] **Baseline ingestion**: do we run external baselines (XGBoost on raw graph stats, GIN, etc.) inside the Foundry, or do we ingest published numbers?
- [ ] **Statistical comparison protocol**: McNemar test? Paired Wilcoxon? Bayesian posterior comparison?
- [ ] **Compute budget**: should the Foundry track wall-clock and report Pareto frontiers, or just metric-quality?
- [ ] **CI integration**: should the Foundry run on every PR (slow)? On every release (right)?

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | Seed: `examples/run_experiment.py` is a one-task, one-contender Foundry |
| v0.1 | Multi-task Foundry with MUTAG, PROTEINS as initial real-data tasks |
| v0.2 | Statistical comparison protocol, Pareto reporting |
| v1.0 | Public Foundry leaderboard with versioned numbers, CI gating |
