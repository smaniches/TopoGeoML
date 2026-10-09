# TopoGeoML documentation

The [repository README](../README.md) provides installation and the first run. This directory holds implementation, operational, and scientific documentation. The installed wheel contains the `topogeoml` package; these Markdown research documents are maintained in the source repository and on the documentation site, not as importable Python modules.

## Maintainer reading order

1. [Architecture](ARCHITECTURE.md): implemented modules, data flow, and boundaries.
2. [Design decisions](DESIGN_DECISIONS.md): observable choices, costs, and alternatives, with undocumented historical rationale called out.
3. [Limitations](LIMITATIONS.md): failure cases, numerical behavior, scaling, and limits on empirical conclusions.
4. [Usage](USAGE.md): full API and command-line workflows, installation, configuration, and troubleshooting.
5. [Reviewer guide](../REVIEWER.md): lint, typing, tests, and evidence verification.

## Research record

- [Current status](../STATUS.md) and [empirical leaderboard](../LEADERBOARD.md) separate supported, negative, inconclusive, and invalidated findings.
- [Claims to Evidence](CLAIMS_TO_EVIDENCE.md) maps statements to code, tests, and artifacts; [Statistical Summary](STATISTICAL_SUMMARY.md) defines the inference boundaries.
- [Hypothesis index](hypotheses/index.md) links preregistered designs through H011b, including the corrected H009-R study. Historical H009 evidence is invalidated rather than silently replaced.
- [Reproducing](../REPRODUCING.md) explains current-code reruns versus historical replication and names the intended comparison families.
- [Mathematical foundations](mathematics/foundations.md) provides definitions used in the signal-topology modules.
- [Research scope](research_scope.md) preserves the concise documentation-site summary at its previous published URL.
- [Research report](RESEARCH_REPORT.md) is a historical snapshot through H008c, **not** the most recent claim register.

Benchmark and publication numbers belong to committed evidence files, not to this index. See each experiment's recorded seeds, software context, and decision rule before reusing a result.
