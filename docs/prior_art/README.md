# RE-call prior art evidence system

This directory is the canonical, reviewable evidence corpus behind the public prior art matrix.

The corpus separates five things:

1. Sources, which are external papers, repositories, documentation, benchmarks, protocols, or standards.
2. Systems, which are the products or research systems being compared.
3. Capabilities, which are stable, defined matrix dimensions.
4. Claims, which are atomic statements about one system and one capability.
5. Reviews, which record the human decision that allows a claim into the generated report.

The accepted capability values are `verified`, `partial`, `not_evidenced`, `contradicted`, and
`unknown`. `not_evidenced` means that reviewed sources did not establish a capability. `unknown`
means that the investigation is incomplete. Neither value proves that a system lacks a capability.

## Workflow

Run validation and rendering from the repository root:

```text
python -m tools.prior_art validate
python -m tools.prior_art render
python -m tools.prior_art render --check
python -m tools.prior_art check-experiments benchmarks/my_probe.py
python -m tools.prior_art check-links
```

The link check is intentionally manual because external availability is unstable. It does not
rewrite source records.

## Evidence policy

Secondary sources may discover candidates, but accepted `verified` claims require primary evidence.
Each accepted claim has a source, an evidence locator, a human review record, and a short evidence
note. The renderer never creates a claim from silence and never emits an automatic claim of novelty.

## Generated reports

* [Capability matrix](generated_matrix.md)
* [Gap report](generated_gap_report.md)
* [Machine readable summary](generated_summary.json)
* [Public prior art position](../PRIOR_ART.md)
