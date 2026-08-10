# RE-call documentation

This directory is the project wiki for design, operations, research protocol, and longer studies.
The README stays short; detailed evidence and limits live here or in `results/`.

## Start here

| Document | Use it for |
|---|---|
| [WRITEUP.md](WRITEUP.md) | Architecture, trust semantics, and evaluation summary. |
| [AUTH.md](AUTH.md) | Authentication, scopes, tenant isolation, and deployment limits. |
| [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md) | MCP setup and tool behavior. |
| [CASE_STUDY.md](CASE_STUDY.md) | Origin story, redacted production context, and public/private boundary. |
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Rules for benchmark runs, artifact retention, and post-run review. |
| [REASONING_CONTRACT.md](REASONING_CONTRACT.md) | Session 1 reasoning vocabulary, invariants, baseline fixture, and non goals. |

## Evidence and limits

| Document | Use it for |
|---|---|
| [../results/FINDINGS.md](../results/FINDINGS.md) | What the measurements establish, where they stop, and which claims were corrected. |
| [../results/RESULTS.md](../results/RESULTS.md) | Published tables and result summaries. |
| [../results/ARTIFACTS.md](../results/ARTIFACTS.md) | Which committed artifact produced each result. |
| [../CHANGELOG.md](../CHANGELOG.md) | Release notes and upgrade warnings moved out of the README. |

## Research notes

| Document | Scope |
|---|---|
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
