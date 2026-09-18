# AML clean reranker ablation decision

## Decision

The next AML coding experiment compares exactly two independent RE-call Hosted configurations:

1. `B0_raw`: raw evidence, Voyage Context 4 dense retrieval, exact PostgreSQL lexical retrieval,
   reranker OFF, and graph OFF.
2. `B1_raw_rerank`: the identical corpus and candidate construction, with Voyage
   `rerank-2.5` ON.

DeepSeek V4 Flash is the local coding agent used to reduce diagnostic cost. It is not an internal
Search reasoning stage. The official AML run follows the mandatory GPT 4o mini policy.

## Why the prior matrix cannot decide this

Measured on VPS2 on 2026-09-18, the completed C0 through C4 retrieval ladder selected C0. The
later arms were cumulative rather than independent. C3 tested reranking only after SPLADE and
procedure compilation, both of which had already failed their gates. The cumulative selector also
made later promotion impossible after the first failure.

The immutable selection artifact is:

`/home/sentiment/agent-memory-bench-coding-4826a7b0/results/aml-coding-memory-matrix-v1/714d4a81-a0abe03e-4826a7b0-retrieval-repair/selection.json`

Its SHA-256 is
`19e75d615fa991752d9234bf6ccffeec17ac3c4039df4fd3907f8f9a34ce74fc`.

Remeasure the recorded values with:

```bash
ssh vps2 'cd /home/sentiment/agent-memory-bench-coding-4826a7b0 && python -m json.tool results/aml-coding-memory-matrix-v1/714d4a81-a0abe03e-4826a7b0-retrieval-repair/selection.json'
```

## Exclusions

* Graph is gated by real relation activation. The raw AML transcript corpus previously produced
  zero semantic relations. Zero authored or eligible relations makes graph ineligible.
* Entailment is excluded because the existing judge falsely rejected correct evidence on real
  RE-call chunk shapes.
* SPLADE, procedure compilation, facets, and packing are excluded after failing the completed
  retrieval ladder.
* DeepSeek Pro is excluded because Flash versus Pro would measure the coding agent, not the
  retrieval feature being selected.

## Frozen execution shape

The present screen contains 12 tasks, two variants, and three seeds, for 72 cells. A passing
reranker receives an independent confirmation over 34 tasks, five isolated corpus conditions,
two variants, and three seeds, for 1,020 cells. Task execution uses exactly three concurrent
workers.

Each condition receives one raw dense embedding pass. The reranker variant reuses the baseline
tenant. The full experiment therefore performs exactly five Voyage dense embedding passes, with
no concurrent embedding or indexing.

The canonical numeric gates, task rosters, predictions, stop rules, and artifact paths are frozen
in AMB preregistration 089. Runtime implementation does not begin until the RE-call decision and
AMB preregistration commits are signed and their hashes are recorded in durable memory.

## Official boundary

Voyage embedding and reranking eligibility for AML Open-source Methods remains unconfirmed. The
organizer's written answer is required before spending an AML run. Smoke is the first scarce run;
Full is spent only after Smoke verifies the contract, provider identities, expected call counts,
and zero fallbacks.
