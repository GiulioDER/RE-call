# RE-call documentation

This directory separates the product documentation from the evaluation record. Start with the
operating guides when you are deciding whether to use RE-call. Use the evidence sections when you
want to audit a claim or reproduce a benchmark.

## Product path

| Document | Use it for |
|---|---|
| [../README.md](../README.md) | Product overview, setup, and integrations. |
| [API.md](API.md) | Supported Python, CLI, and MCP surface. |
| [REPOSITORY_MAP.md](REPOSITORY_MAP.md) | Product, evidence, benchmark support, and archive boundaries. |
| [PRODUCTION.md](PRODUCTION.md) | Production posture, supported boundaries, and known limits. |
| [CALIBRATION.md](CALIBRATION.md) | Calibration workflow and generation-aware serving. |
| [MIGRATIONS.md](MIGRATIONS.md) | Migration roles, serving DSNs, and schema operations. |
| [OPERATING_MODES.md](OPERATING_MODES.md) | Local, production, quality, hosted, and evaluation operating modes. |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Full environment variable reference behind the root template. |
| [DEPENDENCIES.md](DEPENDENCIES.md) | Dependency floors, ceilings, extras, and packaging rationale. |
| [AUTH.md](AUTH.md) | Authentication, scopes, tenant isolation, and deployment limits. |
| [SECURITY_MODEL.md](SECURITY_MODEL.md) | Detailed threat model behind the root security policy. |
| [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md) | MCP setup and tool behavior. |

## Architecture

| Document | Use it for |
|---|---|
| [WRITEUP.md](WRITEUP.md) | Architecture, trust semantics, and evaluation summary. |
| [CASE_STUDY.md](CASE_STUDY.md) | Origin story, redacted production context, and public/private boundary. |
| [REASONING_CONTRACT.md](REASONING_CONTRACT.md) | Reasoning vocabulary, invariants, Session 1 baseline fixture, and Session 6 evaluation controls. |

## Evidence and limits

| Document | Use it for |
|---|---|
| [EVIDENCE.md](EVIDENCE.md) | The shortest evidence path: claims, measurements, limits, and withdrawn claims. |
| [../results/FINDINGS.md](../results/FINDINGS.md) | What the measurements establish, where they stop, and which claims were corrected. |
| [../results/README.md](../results/README.md) | Map of committed result summaries and compact artifacts. |
| [../results/RESULTS.md](../results/RESULTS.md) | Published tables and result summaries. |
| [../results/ARTIFACTS.md](../results/ARTIFACTS.md) | Checksum and artifact map for readers auditing a claim. |
| [../CHANGELOG.md](../CHANGELOG.md) | Release notes and upgrade warnings moved out of the README. |

## Research archive

| Document | Scope |
|---|---|
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Rules for benchmark runs, artifact retention, and post-run review. |
| [VISIBILITY_BENCHMARKS.md](VISIBILITY_BENCHMARKS.md) | Public leaderboard submission track for Kaggle AgentEval, EnterpriseRAG-Bench, LiveRAG, and CRAG. |
| [ENTERPRISE_RAG_VAST.md](ENTERPRISE_RAG_VAST.md) | Vast.ai GPU runbook for the EnterpriseRAG-Bench SPLADE arm. |
| [archive/CHANGELOG_FULL.md](archive/CHANGELOG_FULL.md) | Full historical changelog. |
| [RAG_TRAINING_STUDY.md](RAG_TRAINING_STUDY.md) | When fine-tuning embeddings helps, and when it does not. |
| [ENTAILMENT_SUPERSESSION_STUDY.md](ENTAILMENT_SUPERSESSION_STUDY.md) | Near-miss abstention, entailment, and write-time supersession. |
| [REFERENCE_TIME_DESIGN.md](REFERENCE_TIME_DESIGN.md) | Reference-time handling for temporal retrieval. |
| [their-harness-parity.md](their-harness-parity.md) | Running RE-call inside Mem0's benchmark harness. |

Generated binaries and local render outputs should stay out of narrative documentation unless a
document explicitly depends on them.
