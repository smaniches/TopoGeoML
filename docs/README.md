# TopoGeoML Documentation

This directory holds long-form documentation that lives in-repo and ships with each release.

## Structure

```
docs/
├── README.md                    ← you are here
├── RESEARCH_REPORT.md           ← primary academic artifact
├── hypotheses/                  ← preregistered hypothesis series
│   ├── HYPOTHESIS-001-hodge-mutag.md
│   ├── HYPOTHESIS-002-hodge-proteins.md
│   ├── HYPOTHESIS-003-hodge-nci1.md
│   ├── HYPOTHESIS-004-sample-size-mechanism.md
│   ├── HYPOTHESIS-005-feature-density-mechanism.md
│   ├── HYPOTHESIS-006-graph-topology-mechanism.md
│   └── HYPOTHESIS-007-graph-structural-signal-decomposition.md
└── mathematics/
    └── foundations.md
```

## Reading order

1. [Top-level README.md](../README.md) for the project overview and empirical results summary.
2. [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for the full structured technical report.
3. [hypotheses/](hypotheses/) for per-hypothesis preregistrations and resolved outcomes (HYPOTHESIS-001 through 007).
4. [LEADERBOARD.md](../LEADERBOARD.md) for the navigable empirical claim table.
5. [REPRODUCING.md](../REPRODUCING.md) for step-by-step reproduction instructions.

## Hypothesis documents

Each hypothesis document follows a fixed structure:

- Falsifiable sub-predictions with explicit statistical thresholds
- Pre-specified outcome decision tree
- Experimental design (seeds, epochs, arms, statistical procedure)
- Resolved outcome (appended after execution; original predictions preserved)

Documents are committed BEFORE the experiment runs. The git history serves as the preregistration timestamp.

---

Santiago Maniches (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)) — TOPOLOGICA LLC
