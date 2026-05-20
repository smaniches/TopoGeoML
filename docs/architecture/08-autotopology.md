# AutoTopology Search and Complexity Penalty

**STATUS:** PROPOSED
**Phase:** v0.1
**Spec source:** TopoGeomML Specification Part II §12, §13

---

## What it is

AutoTopology is a search procedure over filtration parameters, vectorizer settings, and descriptor choices to find the configuration that gives the most informative topological features for a downstream task — subject to a **complexity penalty** that prevents overfitting to the dataset.

The complexity penalty is the differentiator. A grid search over filtration parameters that picks the configuration maximizing downstream accuracy will overfit. AutoTopology penalizes high-resolution / high-sigma configurations to control degrees of freedom.

## Why it matters

In v0.0.1, the user picks `max_edge_length`, `resolution`, `sigma`, `vectorizer` by hand. That's fine for synthetic benchmarks. For real datasets it is the dominant source of failure: tuned-by-eye configurations look great on the validation set and collapse on holdout. AutoTopology with a real complexity penalty is the discipline that makes topological features usable in production.

## Core constructs

- `SearchSpace` — declarative description of the parameter space (Hyperopt-style)
- `Penalty` — function that maps configuration → complexity scalar
- `Scorer` — task-specific evaluator (CV accuracy, AUC, etc.)
- `AutoTopology` — orchestrator: minimizes `−Scorer(config) + λ · Penalty(config)`
- `SearchReport` — structured result with provenance for every evaluated configuration

## Open questions

- [ ] **Penalty form**: BIC-style (penalty = k · log(n) where k = effective DoF)? AIC-style? Custom?
- [ ] **What counts as DoF** for a persistence-image configuration of (resolution=20, sigma=0.1)? Empirically calibrated?
- [ ] **Inner CV vs outer CV**: AutoTopology's search must be wrapped in an outer CV to get honest performance estimates. Implement nested-CV by default?
- [ ] **Search backend**: Optuna? Hyperopt? In-tree random + Bayesian?

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | None |
| v0.1 | `topogeoml.autotopology.SearchSpace` + grid search + BIC-style penalty + nested CV |
| v0.2 | Bayesian search via Optuna adapter; multi-metric penalty |
| v1.0 | Integrated with Benchmark Foundry as a baseline procedure |
