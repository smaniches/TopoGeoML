# Claim Graphs

**STATUS:** PROPOSED
**Phase:** v0.1
**Spec source:** TopoGeomML Specification Part II §5
**Depends on:** [00-positioning.md](00-positioning.md), [01-five-layer-architecture.md](01-five-layer-architecture.md), [04-evidence-bundles.md](04-evidence-bundles.md)

---

## What it is

A claim graph is a directed graph whose nodes are **claims** about a model or training run, and whose edges describe how claims relate. Every claim points to the evidence that supports it (an artifact in an evidence bundle); evidence is verified by content hash.

A claim is a structured proposition like:

```
Claim(
  subject="model_id=v3.7-rc2",
  predicate="latent_topology",
  property="beta_1",
  value=2,
  cohort="all_users",
  evidence_id="sha256:abc...",
  validation="held_out_test"
)
```

An edge has a typed semantics: `supports`, `contradicts`, `refines`, `supersedes`.

## Why it matters

Today, "the model is good" reduces to a number (accuracy = 0.93). A claim graph makes "good" into a compound, audit-able structure: a set of claims about cohorts, invariants, robustness checks, and fairness probes, each tied to specific evidence. A regulator, auditor, or downstream team can walk the graph and ask "which claim breaks first under intervention X?"

The deployment gate (see [12-governance.md](12-governance.md)) is a query over a claim graph: *"refuse to promote model M unless every claim with severity ≥ critical is supported by current evidence."*

## Core constructs

- `Claim` — the proposition; carries subject, predicate, property, value, cohort, validation, severity
- `Edge` — typed relationship between two claims
- `ClaimGraph` — a versioned directed graph with append-only writes
- `Query` — a structured question over the graph (e.g., "all critical claims about model M")
- `Verifier` — checks that claimed evidence hashes match the bundle

## Open questions

- [ ] What's the **canonical claim ontology**? Borrow from W3C PROV? Define our own?
- [ ] How does a claim **expire**? (Evidence ages; the latent topology of a year-old model may no longer reflect production data.)
- [ ] **Multi-party claims**: who attests to a claim — the training pipeline, the auditor, both?
- [ ] **Privacy**: a claim about a cohort may itself leak information about cohort membership; how is that handled?

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | Seed: `DiagramProvenance` + `FitProvenance` are micro-claims with no graph yet |
| v0.1 | `topogeoml.claims.Claim` dataclass, `topogeoml.claims.ClaimGraph` with JSON serialization, basic query API |
| v0.2 | Cryptographic attestation (sigstore or in-tree); claim-graph diff between releases |
| v1.0 | Federated claim graphs (multiple parties attest); regulator-readable export |

## Cross-references

- [04-evidence-bundles.md](04-evidence-bundles.md) — the artifact format that backs claims
- [05-invariant-ledgers.md](05-invariant-ledgers.md) — the time-series record of invariants that becomes a sequence of claims
- [12-governance.md](12-governance.md) — how deployment gates query claim graphs
