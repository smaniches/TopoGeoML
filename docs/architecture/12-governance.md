# Governance — Model Cards and Deployment Gates

**STATUS:** PROPOSED
**Phase:** v0.1 (model cards from evidence); v0.2 (deployment gates)
**Spec source:** TopoGeomML Specification Part II §22, §23
**Depends on:** [03-claim-graphs.md](03-claim-graphs.md), [04-evidence-bundles.md](04-evidence-bundles.md), [05-invariant-ledgers.md](05-invariant-ledgers.md)

---

## What it is

The two product surfaces of layer 5:

1. **Model card generation** — produce a model card (Mitchell et al. 2018 structured form) from an evidence bundle, automatically.
2. **Deployment gates** — CI/CD hooks that block model promotion when the claim graph of the candidate release contains regressions against the baseline.

## Why it matters

These are the commercial product. Everything else in TopoGeomML is research infrastructure that enables these two surfaces. Healthcare, finance, defense, and increasingly public-sector ML all mandate model cards; today they are filled in by hand. A system that generates them from signed evidence is a defensible product.

Deployment gates are the natural extension: once a model card is structured, refusing to merge a release whose card violates a policy is mechanical. This is the layer where TopoGeomML becomes part of the CI pipeline, not an exploratory tool.

## Model card from evidence

Input: an evidence bundle (see [04-evidence-bundles.md](04-evidence-bundles.md)) plus a model card template.

Output: a populated model card. Every claim in the card is footnoted with the evidence-bundle hash that supports it. Sections like "Intended use", "Training data" are populated from the config echo. Sections like "Performance", "Cohort breakdown", "Failure modes" are populated from the audit and shape-of-learning artifacts.

Sections that **cannot** be auto-populated (ethical considerations, intended population, social context) are flagged for human authorship and tracked separately in the card status.

## Deployment gates

Input: candidate release's claim graph + baseline release's claim graph + a policy file.

Behaviour: run policy queries against the diff. Examples:

```yaml
gates:
  - name: no_critical_invariant_regression
    require: |
      for every claim c with severity=critical in baseline:
        candidate has a claim c' that supersedes(c) or refines(c)
        and value(c') is within tolerance(c)

  - name: cohort_fairness_preserved
    require: |
      for every cohort C:
        latent_beta_1(candidate, C) >= 0.5 * latent_beta_1(baseline, C)
```

A failed gate blocks merge / promotion. Gate decisions are themselves logged into the invariant ledger.

## Core constructs

- `ModelCard` — structured model card object with content-addressed footnotes
- `CardTemplate` — section list with population sources
- `Gate` — declarative policy rule
- `GateOutcome` — pass/fail with the specific claims that triggered it
- `CIRunner` — adapter for GitHub Actions / GitLab CI

## Open questions

- [ ] **Card schema**: align with the Hugging Face Model Card Schema? Or with Google's PAIR? Or define our own that supersets both?
- [ ] **Gate languages**: declarative YAML for policies vs Python for flexibility. Leaning YAML for v0.1.
- [ ] **Card UI**: do we ship a renderer (markdown / HTML / PDF)? Or just JSON and let consumers render?
- [ ] **Approval workflows**: gate failures can be overridden by human review. Where is the override recorded?

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | None directly; JSON output is the precursor |
| v0.1 | `topogeoml.governance.ModelCard` + template + CLI to generate from a bundle |
| v0.2 | Deployment-gate CLI; GitHub Action |
| v1.0 | Approval workflows; multi-org gate federation |
