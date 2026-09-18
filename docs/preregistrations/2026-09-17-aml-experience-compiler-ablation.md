# Preregistration: AML engineering experience compiler ablation

Date: 2026-09-17

Status: locked before implementation measurements

## Question

Does compiling one coding session into typed, source-grounded engineering experience records
improve evidence retrieval and executable coding task success over raw chronological messages?
Does retaining raw messages beside compiled records recover exact details that compilation loses?

This experiment isolates memory representation. It does not test query facets, reranking, compact
packing, graph expansion, confidence abstention, or agent activation.

## Arms

1. `E0_raw`: store and search raw chronological message segments only.
2. `E1_compiled`: store and search compiled engineering experience records only.
3. `E2_compiled_raw`: store and search the same compiled records plus the same raw message
   segments used by `E0_raw`.

All arms use the exact user scope, sessions, messages, task queries, Voyage 4 embedder, dense and
PostgreSQL lexical retrieval, reciprocal rank fusion, candidate width 100, returned top K, model,
sandbox, checker, seed, timeout, and retry policy. Query facets, reranking, and evidence packing
remain disabled in every arm.

## Compiler contract

Each compiled record has one of these kinds: symptom, root cause, failed attempt, successful
repair, architectural decision, procedure, validation, constraint, or repository fact.

Each record may contain task shape, problem, action, outcome, validation, technical entities,
event time, and supported supersession references. It must identify the exact source session and
carry at least one raw evidence span. Every span names a message ordinal and exact character start
and end offsets, and its quoted text must equal that source slice byte for byte after JSON decoding.

Unsupported spans reject the record. Unsupported entities are removed. Outcome and validation
remain populated only when their exact text occurs in a source message. Event times remain
populated only when they equal a supplied message timestamp. Supersession references remain only
when they name a previously stored record supplied to the compiler.

If compilation fails or yields no supported record, `E1_compiled` and `E2_compiled_raw` use the
same deterministic technical extraction fallback. Add must still return only after the selected
arm is durably stored and immediately searchable.

## Deterministic apparatus checks

Before any provider or task-success measurement, tests must prove:

1. `E0_raw` persists no compiled record.
2. `E1_compiled` persists no raw record.
3. `E2_compiled_raw` persists both record types.
4. Raw chunks reconstruct every source message in order and carry exact character offsets.
5. Compiled evidence spans resolve to the exact source message slices.
6. Fabricated message ordinals, offsets, quotes, entities, outcomes, validation, event times, and
   supersession references cannot enter stored compiled records.
7. Compiler failure leaves every arm searchable according to its declared representation.
8. Exact user isolation, idempotency, immediate searchability, restart persistence, and purge
   behavior remain unchanged.

Each new behavior test requires a red proof against a plausible mutation of its production symbol.
Collection, import, fixture, network, and timeout failures are not red proofs.

## Retrieval replay

Run the three arms over the complete public agent-memory-bench corpus. Search uses the original
task prompt. Ground-truth fact terms are used only after Search for scoring and never enter Add,
Search, compiler prompts, stored metadata, or retrieval queries.

Report per arm:

1. Any answer-bearing evidence at ranks 1, 5, 10, and 100.
2. Complete answer-bearing evidence coverage at ranks 5, 10, and 100.
3. Mean reciprocal rank of the first answer-bearing record.
4. Source-session recall and duplicate-session concentration.
5. Returned characters and items.
6. Add and Search median and p95 latency.
7. Compiler rejection, fallback, and unsupported-field removal counts.

`E1_compiled` passes the retrieval gate when complete top 10 coverage is no worse than `E0_raw`
by more than one absolute percentage point and mean reciprocal rank is higher. `E2_compiled_raw`
passes when it improves complete top 10 coverage or recovers at least half of the cases lost by
`E1_compiled`, without lower mean reciprocal rank than `E0_raw`.

## Executable task gate

The screening and final task populations, seeds, admission rules, and checker integrity controls
remain those frozen in `2026-09-12-aml-hosted-industry-v1.md`. The screen compares all three arms.
Only arms that pass retrieval replay advance.

The compiler representation passes only when its candidate-only wins exceed baseline-only wins on
the final paired population and it causes no more than two new failures in superseded and adjacent
memory controls. Task success is the primary endpoint. Retrieval metrics are mechanism evidence,
not a substitute.

## Predictions

I predict `E1_compiled` improves first-hit rank and cross-vocabulary retrieval for feature tasks,
failed attempts, and architectural decisions, but loses some exact strings from raw tool output.
I predict `E2_compiled_raw` retains most of that ranking improvement while recovering exact paths,
symbols, error text, and validation details. I therefore predict `E2_compiled_raw` is the best
task-success arm, even if `E1_compiled` is the most token-efficient arm.

## Stop rules

Stop and mark the run invalid if any arm receives different source messages or queries, a source
span does not resolve exactly, a benchmark label enters the product, memory crosses a user scope,
Add acknowledges before Search can observe the records, a paired model or checker drifts, or raw
benchmark content enters application logs.

Do not spend an AML Full run on this ablation. Promote only the locally selected representation
into the complete hosted candidate, then use Smoke for contract compatibility before Full.

## Artifacts

Store immutable raw results under `docs/results/aml-experience-compiler-v1/`. Record the checkout
commit, dependency lock digest, model and prompt identities, corpus and task digests, arm settings,
timings, fallback counts, and invalid cells. Append conclusions below this line without changing
the registration above it.

<!-- RESULTS APPEND BELOW; EVERYTHING ABOVE IS FROZEN -->

Apparatus readiness is recorded in
`docs/results/aml-experience-compiler-v1/APPARATUS_READINESS.md`. No provider-backed measurement
has started.

## Measured result, 2026-09-18 UTC

The complete frozen retrieval replay finished on 196 sessions, 2,281 messages, and 34 tasks.
The mechanical selector retained `E0_raw`. `E1_compiled` failed with 0% complete top 10 coverage
and 0.0321 mean reciprocal rank, compared with 11.76% and 0.2470 for raw. `E2_compiled_raw`
recovered all four top 10 cases lost by E1 and matched raw complete top 10 coverage at 11.76%,
but its 0.2100 mean reciprocal rank was below raw, so it also failed the frozen gate.

Neither compiler arm advances to the executable task screen. The dominant mechanism failure was
evidence grounding: E1 rejected 437 of 495 proposed records and used deterministic fallback for
168 of 196 sessions; E2 rejected 417 of 467 and used fallback for 171 of 196 sessions. The full
identity record, diagnostics, hashes, caveats, and immutable artifacts are recorded in
`docs/results/aml-experience-compiler-v1/MEASURED_RESULTS.md` and its
`952bc048-vps2-retry2/` artifact directory.
