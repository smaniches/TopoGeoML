# EMLIR — Epistemic ML Intermediate Representation

**STATUS:** PROPOSED — uncertain whether needed
**Phase:** earliest v0.2; possibly never if Pydantic + JSON suffices
**Spec source:** TopoGeomML Specification Part II §4
**Depends on:** [00-positioning.md](00-positioning.md), [01-five-layer-architecture.md](01-five-layer-architecture.md)

---

## What it is

EMLIR is a proposed intermediate representation for describing ML training and evaluation as a directed graph of operations whose outputs carry epistemic type information. It is to evidence-graphs what MLIR is to compiler IRs: a stable, lower-level representation that multiple frontends and consumers can target.

A node in EMLIR represents an operation (a fit, a transform, a topology measurement). An edge represents a data dependency. Each value has an **epistemic type** beyond its computational type: not just `float64`, but `float64 + provenance + validation_method + correction_audit_status`.

## Why it matters

If layer 4 and layer 5 components both consume the output of layer 3 operators, they need a shared vocabulary that survives serialization. Today, that vocabulary is implicit (the `FitProvenance` dataclass). EMLIR would make it explicit and tool-able: third-party verifiers, IDE plugins, evidence visualizers, regulatory auditors could all read EMLIR without depending on Python.

## Why it might not be needed

Strong prior: Pydantic models + JSON Schema + a small set of Python dataclasses cover 90% of what EMLIR would do, at 10% of the implementation cost. The question is whether the missing 10% (cross-language tooling, formal verification hooks, IR-level transformations) ever becomes load-bearing.

## Core constructs (if built)

- `Node` — an operation: kind, parameters, input bindings, output bindings
- `Edge` — a typed data dependency
- `EpistemicType` — the type system extension carrying provenance / validation / correction-audit
- `Module` — a unit of compilation: a set of nodes + edges + entry points
- `Verifier` — a pass that walks a Module and checks invariants

## Open questions

- [ ] Is EMLIR's domain layer 4 only (training/eval IR), or does it extend to layer 5 (governance IR)?
- [ ] What's the textual syntax? (S-expression, JSON, YAML, custom). MLIR's pseudo-textual / binary duality is appealing.
- [ ] Is there a credible cross-language consumer that justifies the lift?
- [ ] When would a non-Python consumer of an evidence bundle need EMLIR instead of JSON?

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | None — concept only |
| v0.1 | None — Pydantic + JSON for everything |
| v0.2 | **Decision point**: revisit based on v0.1 pain points |
| v0.3+ | If decided to build: bootstrap with a minimal node/edge/type schema, single Python frontend, single verifier pass |

## Default position

**Do not build EMLIR until v0.1 has shipped and explicitly demonstrated that Pydantic + JSON cannot carry the necessary semantics.** The cost of over-engineering an IR is paid in every PR for years.
