# RE-call documentation

This directory separates the product documentation from the evaluation record. Start with the
operating guides when you are deciding whether to use RE-call. Use the evidence sections when you
want to audit a claim or reproduce a benchmark.

The one canonical path after `recall quickstart` is `recall setup` (the guided wizard in the
README's "full install"). [WIZARD.md](WIZARD.md) documents the scriptable `recall wizard
--headless` form of the same install, and [FIRST_CALIBRATION.md](FIRST_CALIBRATION.md) is the
manual step-by-step for anyone who wants to see each command the wizard runs.

## Product path

| Document | Use it for |
|---|---|
| [../README.md](../README.md) | Product overview, setup, and integrations. |
| [API.md](API.md) | Supported Python, CLI, and MCP surface. |
| [REPOSITORY_MAP.md](REPOSITORY_MAP.md) | Product, evidence, benchmark support, and archive boundaries. |
| [WIZARD.md](WIZARD.md) | The install wizard's headless and GUI front ends, config format, and refusals. |
| [FIRST_CALIBRATION.md](FIRST_CALIBRATION.md) | Walkthrough from an indexed folder to a trusted, certified corpus, with the traps named where you hit them. |
| [PRODUCTION.md](PRODUCTION.md) | Production posture, supported boundaries, and known limits. |
| [CALIBRATION.md](CALIBRATION.md) | Calibration workflow and generation-aware serving. |
| [GENERATIONS.md](GENERATIONS.md) | Immutable generations: build, validate, promote, roll back. |
| [MIGRATIONS.md](MIGRATIONS.md) | Migration roles, serving DSNs, and schema operations. |
| [OPERATING_MODES.md](OPERATING_MODES.md) | Local, production, quality, hosted, and evaluation operating modes. |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Full environment variable reference behind the root template. |
| [DEPENDENCIES.md](DEPENDENCIES.md) | Dependency floors, ceilings, extras, and packaging rationale. |
| [MODEL_LICENSES.md](MODEL_LICENSES.md) | Licenses of the embedding and reranking models the extras pull in. |
| [AUTH.md](AUTH.md) | Authentication, scopes, tenant isolation, and deployment limits. |
| [SECURITY_MODEL.md](SECURITY_MODEL.md) | Detailed threat model behind the root security policy. |
| [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md) | MCP setup and tool behavior. |
| [CODEX_RECALL_INTEGRATION.md](CODEX_RECALL_INTEGRATION.md) | Automatic Codex installation, hooks, plugin layout, and shared memo contract. |
| [USING_WITH_AGENT_SDK.md](USING_WITH_AGENT_SDK.md) | In-process tools for a Claude Agent SDK application: no MCP server, the same tool surface, and the boundaries the server was providing for you. |
| [ENTERPRISE_RETRIEVAL.md](ENTERPRISE_RETRIEVAL.md) | The enterprise control plane: routes, profiles, and multi-store serving. |
| [REASONING_OPERATIONS.md](REASONING_OPERATIONS.md) | Opt-in reasoning commands, MCP tools, failure behavior, review policy, and metrics. |
| [DECISION_LEDGER.md](DECISION_LEDGER.md) | Opt-in append-only records of every search decision: trigger, evidence, verdicts, governing calibration, and refusals. |

## Architecture

| Document | Use it for |
|---|---|
| [WRITEUP.md](WRITEUP.md) | Architecture, trust semantics, and evaluation summary. |
| [ENGINEERING.md](ENGINEERING.md) | Engineering decisions and the measurements behind them. |
| [Validity Frontmatter 1.0](https://github.com/GiulioDER/validity-frontmatter) | The open vocabulary RE-call implements (`valid_from`, `valid_until`, `supersedes`), its resolution rules and its verdict algorithm. MIT licensed and maintained in its own repository, so it stays implementable without RE-call. RE-call is the Python implementation; a zero-dependency TypeScript one ships alongside the spec. |
| [CASE_STUDY.md](CASE_STUDY.md) | Origin story, redacted production context, and public/private boundary. |
| [PRIOR_ART.md](PRIOR_ART.md) | How RE-call relates to existing memory and retrieval systems. |
| [REASONING_CONTRACT.md](REASONING_CONTRACT.md) | Reasoning vocabulary, invariants, Session 1 baseline fixture, and Session 6 evaluation controls. |
| [REASONING_API.md](REASONING_API.md) | Typed reasoning request and response surface, provider ports, serialization, and validation rules. |
| [REASONING_GRAPH.md](REASONING_GRAPH.md) | Generation-bound reasoning graph projection schema and invariants. |
| [REASONING_RELEASE_NOTES.md](REASONING_RELEASE_NOTES.md) | Experimental reasoning release notes, migration notes, and limitations. |
| [INFERENCE_PROPOSALS.md](INFERENCE_PROPOSALS.md) | Inference proposals: how review candidates are produced and constrained. |
| [TRUTH_EXTRACTION_DESIGN.md](TRUTH_EXTRACTION_DESIGN.md) | Model backed extraction of truth claims from prose, the validation ladder, and the reviewed write path back into corpus frontmatter. |

## Evidence and limits

| Document | Use it for |
|---|---|
| [EVIDENCE.md](EVIDENCE.md) | The shortest evidence path: claims, measurements, limits, and withdrawn claims. |
| [../results/FINDINGS.md](../results/FINDINGS.md) | What the measurements establish, where they stop, and which claims were corrected. |
| [../results/README.md](../results/README.md) | Map of committed result summaries and compact artifacts. |
| [../results/RESULTS.md](../results/RESULTS.md) | Published tables and result summaries. |
| [../results/ARTIFACTS.md](../results/ARTIFACTS.md) | Checksum and artifact map for readers auditing a claim. |
| [../CHANGELOG.md](../CHANGELOG.md) | Release notes and upgrade warnings moved out of the README. |

## Benchmarks and studies

| Document | Scope |
|---|---|
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Rules for benchmark runs, artifact retention, and post-run review. |
| [ATM_BENCH.md](ATM_BENCH.md) | ATM-Bench full-split results from the benchmark's own evaluator, what the numbers may be compared against, and where the remaining loss is. |
| [MTRAG_BENCHMARK.md](MTRAG_BENCHMARK.md) | MTRAG setup, the retrieval ladder, the abstention result, and the scope boundaries on it. |
| [VISIBILITY_BENCHMARKS.md](VISIBILITY_BENCHMARKS.md) | Public leaderboard submission track for Kaggle AgentEval, EnterpriseRAG-Bench, LiveRAG, and CRAG. |
| [ENTERPRISE_RAG_VAST.md](ENTERPRISE_RAG_VAST.md) | Vast.ai GPU runbook for the EnterpriseRAG-Bench SPLADE arm. |
| [ENTERPRISE_RAG_SUBMISSION.md](ENTERPRISE_RAG_SUBMISSION.md) | EnterpriseRAG-Bench answer artifacts, score summaries, and reproduction steps for leaderboard review. |
| [ENTERPRISE_RAG_REASONING_TRIAGE.md](ENTERPRISE_RAG_REASONING_TRIAGE.md) | EnterpriseRAG reasoning-lane triage record (kept in place; result artifacts point at it). |
| [RAG_TRAINING_STUDY.md](RAG_TRAINING_STUDY.md) | When fine-tuning embeddings helps, and when it does not. |
| [ENTAILMENT_SUPERSESSION_STUDY.md](ENTAILMENT_SUPERSESSION_STUDY.md) | Near-miss abstention, entailment, and write-time supersession. |
| [AGENT_MEMORY_FIELD_REVIEW.md](AGENT_MEMORY_FIELD_REVIEW.md) | Reading notes on the agent-memory field (kept in place; a preregistration and a test point at it). |
| [REASONING_SESSION8_AUDIT.md](REASONING_SESSION8_AUDIT.md) | Session 8 reasoning release decision (kept in place; result artifacts point at it). |
| [their-harness-parity.md](their-harness-parity.md) | Running RE-call inside Mem0's benchmark harness. |

## Design notes

Live design records for shipped subsystems. They stay in `docs/` rather than the archive because
code, CI, or the claim gate reference them by path; retired one-shot designs move to
[archive/](archive/).

| Document | Scope |
|---|---|
| [REFERENCE_TIME_DESIGN.md](REFERENCE_TIME_DESIGN.md) | Reference-time handling for temporal retrieval. |
| [DERIVED_BLOCK_DESIGN.md](DERIVED_BLOCK_DESIGN.md) | Machine-owned regenerable block for `contradicts` and `same_entity`, isolated from extraction. |
| [UNCALIBRATED_FIRST_RUN_DESIGN.md](UNCALIBRATED_FIRST_RUN_DESIGN.md) | What an uncalibrated first run may serve, and every gate between it and production. |
| [EVAL_CALIBRATION_FLEET_DESIGN.md](EVAL_CALIBRATION_FLEET_DESIGN.md) | The calibration evaluation fleet design. |

## Archive

[archive/](archive/) holds the full historical changelog
([archive/CHANGELOG_FULL.md](archive/CHANGELOG_FULL.md)), retired designs, competitor teardowns,
and outreach drafts. Archived documents are frozen: the citation checkers deliberately skip them.

Generated binaries and local render outputs should stay out of narrative documentation unless a
document explicitly depends on them.
