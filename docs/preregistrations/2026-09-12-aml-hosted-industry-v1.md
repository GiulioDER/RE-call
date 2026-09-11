# Preregistration: RE-call Hosted 1.0 AML coding candidate

Date: 2026-09-12

Status: locked before implementation measurements

## Objective

Build and evaluate a fixed hosted RE-call memory configuration for the Agent Memory Leaderboard
Coding Industry Board. The private AML suite is unavailable during development, so this record
does not predict a private leaderboard score from the local proxy. It asks whether coding-session
compilation, task-shaped query planning, reranking, and compact evidence packing improve executable
task success over raw-session hybrid retrieval.

The product name is `RE-call Hosted 1.0`. The candidate is a real hosted product profile and must
not contain benchmark answers, task identifiers, dataset-specific rules, or a generated final
answer in Search output.

## Fixed implementation stack

All generative work performed by the memory system during Add or Search uses `gpt-4o-mini` with
temperature zero and structured JSON output. Dense embeddings use `voyage-4`. Reranking uses
`voyage:rerank-2.5`. If AML rules prohibit either Voyage model, that configuration is ineligible
for the formal AML run; changing the models requires a new preregistration.

The datastore is PostgreSQL with pgvector. Retrieval uses one exact `user_id` scope, dense search,
PostgreSQL lexical search, reciprocal rank fusion, and at most 100 candidates per active leg.
Confidence abstention is disabled. Tenant, temporal, supersession, provenance, malformed-record,
and source-isolation checks remain active.

The Search response contains stored evidence only. Query planning may generate retrieval facets,
but it may not generate or rewrite a final answer.

## Arms

1. `A0_raw`: raw message memories, dense plus lexical retrieval, no query planning, no reranking,
   and full returned chunks.
2. `A1_compiler`: A0 plus the coding-session compiler.
3. `A2_facets`: A1 plus at most four `gpt-4o-mini` evidence-seeking query facets.
4. `A3_rerank`: A2 plus `voyage:rerank-2.5` over the fused candidate pool.
5. `A4_pack_5000`: A3 plus duplicate removal, diversity selection, raw rescue, and a 5,000
   character evidence budget.
6. `A4_pack_7000`: the same treatment with a 7,000 character evidence budget.
7. `A4_pack_9000`: the same treatment with a 9,000 character evidence budget.

Every arm uses the same corpus bytes, task prompts, downstream coding model, execution sandbox,
checker, seed, retry policy, and timeout. Only the memory behavior named above may differ.

## Deterministic retrieval replay

Run every arm over the complete public `agent-memory-bench` corpus. Use the exact task prompt as
the Search query and the checked-in `fact_terms` only for scoring after retrieval. The memory
system must never receive `fact_terms`.

Report, per arm:

1. Any answer-bearing evidence hit at ranks 1, 5, 10, and within the returned character budget.
2. Complete answer-bearing evidence coverage within the returned character budget.
3. Mean reciprocal rank of the first answer-bearing record.
4. Returned item count and character count.
5. Add latency and Search latency at median and p95.
6. Compiler, query-planner, embedder, and reranker failure and fallback counts.

Select the smallest A4 context budget whose complete evidence coverage is within one absolute
percentage point of the best A4 budget. A tie selects the smaller budget.

## Executable screening population

The fixed twelve-task screen is:

1. `xs-evolve-lease`
2. `xs-join-batch`
3. `xs-widen-manifest`
4. `fa-dedup-key`
5. `ts-mig-name`
6. `ts-semver-pin`
7. `ts-retry-cap`
8. `ts-config-layer`
9. `ts-atomic-write`
10. `ts-idempotent-run`
11. `ts-glob-hidden`
12. `ts-quote-shell`

Run seeds 0, 1, and 2. Compare `A0_raw` against the replay-selected candidate. Report paired task
success, candidate-only wins, baseline-only wins, net wins, timeouts, invalid cells, and memory
stage failures. A timeout is an outcome and is not retried.

## Final executable population

The final confirmation contains all 34 executable tasks under `tasks/`, excluding
`smoke-config-port`, at seeds 0, 1, and 2. Compare only `A0_raw` with the candidate that passed the
twelve-task screen. The final population therefore contains 102 paired task-seed cells.

The candidate passes the performance gate only when all of these hold:

1. Candidate-only wins minus baseline-only wins is at least 8 cells.
2. No more than 2 new candidate failures occur in superseded and adjacent-memory controls.
3. Every admitted cell proves that the intended memory arm was available.
4. No model, prompt digest, corpus digest, task digest, or checker differs between paired cells.

The local result is a release gate, not an estimate of AML task resolution.

## API and reliability gates

The release candidate must pass all contract, authentication, exact-user isolation, immediate
searchability, idempotency replay, conflicting replay, process restart, and data-purge tests.

At declared Add concurrency 16 and Search concurrency 16, run a 30 minute soak. The candidate
passes only with zero failed requests, Add p95 below 30 seconds, and Search p95 below 5 seconds.
Health checks are excluded from these latency distributions and reported separately.

## Predictions

I predict that A1 improves complete evidence coverage on the three cross-session tasks and the
failed-attempt task, but has little effect on one-fact tasks. I predict that A2 improves retrieval
for prompts whose goal vocabulary differs from the historical symptom or operation vocabulary.
I predict that A3 improves rank quality without changing candidate recall. I predict that A4 at
7,000 characters preserves evidence coverage while reducing irrelevant context relative to A3.

I predict at least 8 net executable wins for the selected candidate over A0 across the 102-cell
final population. This prediction is intentionally stronger than a simple positive direction
because the AML Full quota may be used only once every three months.

## Red proof requirement

Before trusting each new behavior test, demonstrate that the exact test fails in its intended
assertion against either the pre-fix consumer boundary or a deliberate plausible mutation of the
production symbol. An import failure, collection failure, fixture failure, network failure, or
timeout is not a valid red proof. Record the test node ID, mutation, targeted production symbol,
and assertion failure in a committed proof receipt before any benchmark measurement.

## Cost and stop rules

The total provider budget for implementation and registered measurements is 500 US dollars. Stop
before a new provider call when the recorded spend reaches 450 dollars, leaving 50 dollars for
recovery and smoke validation. Do not substitute an unregistered model after the stop.

Stop and mark the run invalid if evaluation memory crosses a `user_id`, if Add returns before its
records are searchable, if Search returns generated answers, if a task label enters the memory
system, if a paired configuration drifts, or if raw benchmark content appears in application logs.

Do not run AML Full unless the API, reliability, and final executable gates all pass. An AML Smoke
run tests compatibility only and cannot promote a weak candidate.

## Artifacts

Store immutable raw measurement artifacts under `docs/results/aml-hosted-v1/`. Record the checkout
commit, dependency lock digest, model identities, prompt digests, corpus and task digests, timings,
fallback counts, costs, and every invalid cell. Append conclusions below this line without editing
any number or prediction above it.

<!-- RESULTS APPEND BELOW; EVERYTHING ABOVE IS FROZEN -->
