# RE-call documentation

This directory separates the product documentation from the evaluation record. Start with the
operating guides when you are deciding whether to use RE-call. Use the evidence and research
sections when you want to audit a claim or reproduce a benchmark.

## Product path

| Document | Use it for |
|---|---|
| [../README.md](../README.md) | Product overview, five-minute proof, setup, and integrations. |
| [PRODUCTION.md](PRODUCTION.md) | Production posture, supported boundaries, and known limits. |
| [CALIBRATION.md](CALIBRATION.md) | Calibration workflow and generation-aware serving. |
| [MIGRATIONS.md](MIGRATIONS.md) | Migration roles, serving DSNs, and schema operations. |
| [DEPENDENCIES.md](DEPENDENCIES.md) | Dependency floors, ceilings, extras, and packaging rationale. |
| [AUTH.md](AUTH.md) | Authentication, scopes, tenant isolation, and deployment limits. |
| [SECURITY_MODEL.md](SECURITY_MODEL.md) | Detailed threat model behind the root security policy. |
| [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md) | MCP setup and tool behavior. |

## Architecture

| Document | Use it for |
|---|---|
| [WRITEUP.md](WRITEUP.md) | Architecture, trust semantics, and evaluation summary. |
| [CASE_STUDY.md](CASE_STUDY.md) | Origin story, redacted production context, and public/private boundary. |
| [REASONING_CONTRACT.md](REASONING_CONTRACT.md) | Session 1 reasoning vocabulary, invariants, baseline fixture, and non goals. |

## Evidence and limits

| Document | Use it for |
|---|---|
| [EVIDENCE.md](EVIDENCE.md) | The shortest evidence path: claims, measurements, limits, and withdrawn claims. |
| [../results/FINDINGS.md](../results/FINDINGS.md) | What the measurements establish, where they stop, and which claims were corrected. |
| [../results/RESULTS.md](../results/RESULTS.md) | Published tables and result summaries. |
| [../results/ARTIFACTS.md](../results/ARTIFACTS.md) | Checksum and artifact map for readers auditing a claim. |
| [../CHANGELOG.md](../CHANGELOG.md) | Release notes and upgrade warnings moved out of the README. |

## Research archive

| Document | Scope |
|---|---|
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Rules for benchmark runs, artifact retention, and post-run review. |
| [archive/CHANGELOG_FULL.md](archive/CHANGELOG_FULL.md) | Full historical changelog. |
| [RAG_TRAINING_STUDY.md](RAG_TRAINING_STUDY.md) | When fine-tuning embeddings helps, and when it does not. |
| [ENTAILMENT_SUPERSESSION_STUDY.md](ENTAILMENT_SUPERSESSION_STUDY.md) | Near-miss abstention, entailment, and write-time supersession. |
| [REFERENCE_TIME_DESIGN.md](REFERENCE_TIME_DESIGN.md) | Reference-time handling for temporal retrieval. |
| [their-harness-parity.md](their-harness-parity.md) | Running RE-call inside Mem0's benchmark harness. |

## Draft papers and generated artifacts

The MTRAG draft files are research artifacts, not product documentation:

| File | Status |
|---|---|
| [MTRAG_ARXIV_DRAFT.md](MTRAG_ARXIV_DRAFT.md) | Draft manuscript. |
| [MTRAG_ARXIV_PAPER.md](MTRAG_ARXIV_PAPER.md) | Paper text. |
| [MTRAG_ARXIV_PAPER.tex](MTRAG_ARXIV_PAPER.tex) | LaTeX source. |

Generated binaries and local render outputs should stay out of narrative documentation unless a
document explicitly depends on them.
