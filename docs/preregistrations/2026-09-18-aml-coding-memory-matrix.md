# Preregistration: AML coding memory configuration matrix

Date: 2026-09-18

Status: locked before implementation measurement

## Objective

Select one RE-call Hosted coding memory configuration for AML Smoke and Full evaluation. The
local agent-memory-bench result is a release decision aid, not an estimate of the private CAMBench
score.

The experiment tests the progressive value of learned sparse retrieval, grounded procedure
memory, hosted reranking, and task-conditioned evidence packing. It does not test graph expansion,
entailment filtering, confidence abstention, or alternative reasoning models.

## Fixed system identity

Every arm uses the registered `voyage-context-4-v1` embedding profile. Add embeds all ordered
chunks from one source session as one contextualized document group. Search uses the corresponding
contextualized query encoder.

Every arm also uses PostgreSQL exact lexical retrieval, dense retrieval, reciprocal rank fusion,
candidate width 100 per active leg, the same exact `user_id` boundary, and the same source corpus.
When enabled, learned sparse retrieval uses `prithivida/Splade_PP_en_v1` at revision
`762be6a7206e2f299182705972a65e5c46e62be2`, with at most 1,000 nonzero terms. When enabled,
reranking uses `voyage:rerank-2.5` over the complete fused candidate pool.

All generative Add and Search work uses `openai/gpt-4o-mini` through the existing frozen compiler
and query planner. Search returns stored evidence only. The local downstream coding agent is
`deepseek/deepseek-v4-flash`, with the same prompt, sandbox, checker, timeout, retries, and seeds in
every arm.

## Five arms

1. `C0_raw_lexical`: raw chronological messages, Context 4 dense retrieval, and PostgreSQL exact
   lexical retrieval.
2. `C1_splade`: C0 plus SPLADE learned sparse retrieval. Exact lexical retrieval remains active.
3. `C2_procedure`: C1 plus source-grounded compiled engineering records while retaining the same
   raw evidence.
4. `C3_rerank`: C2 plus Voyage reranking over the fused candidate pool.
5. `C4_task_pack`: C3 plus query-only task classification, up to four evidence-seeking facets,
   duplicate removal, session diversity, raw rescue, supersession handling, and a 7,000 character
   evidence budget. Feature queries prioritize architectural decisions, procedures, repository
   conventions, interfaces, and validation examples. Bug-fix queries prioritize symptoms, root
   causes, failed attempts, successful repairs, exact diagnostics, and validation traces. Unknown
   queries use the neutral priority order.

The task classifier sees only the original query and optional answer choices. Benchmark task
labels, fact terms, checker data, and gold answers never enter Add, Search, stored metadata, or
planner prompts.

## Deterministic replay

Run all five arms over the complete checked-in agent-memory-bench corpus and task roster in the
present condition. Search receives the original task prompt only. Checked-in `fact_terms` are
applied after retrieval for scoring.

Report any answer-bearing evidence at ranks 1, 5, 10, and 100; complete evidence coverage at ranks
5, 10, and 100; first-hit reciprocal rank; source-session recall; duplicate-session
concentration; returned items and characters; Add and Search median and p95 latency; compiler,
planner, sparse, and reranker fallback or failure counts; and results by task routing class.

An arm is invalid if its corpus, task set, original queries, provider identities, prompts,
candidate width, or post-retrieval labels drift. Learned sparse coverage must equal dense chunk
coverage before Search starts.

## Executable screen

Run every arm on the following twelve tasks at seeds 0, 1, and 2:

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

For each arm report task success, candidate-only wins and baseline-only wins against C0, net wins,
timeouts, invalid cells, memory calls, memory evidence delivery, provider fallbacks, latency, and
estimated provider cost. A timeout is an outcome and is not retried.

The promoted candidate is the nonbaseline arm with the most screen successes, provided it has
positive net wins against C0 and passes the retrieval gate. Ties are broken by higher complete
coverage at rank 10, then higher mean reciprocal rank, then fewer returned characters, then lower
Search p95. If no arm has positive net wins, C0 remains the candidate.

## Final executable confirmation

Compare C0 with the promoted candidate over all 34 executable tasks except `smoke-config-port`, at
seeds 0, 1, and 2. The final population contains 102 paired task and seed cells.

The candidate passes only when candidate-only wins minus baseline-only wins is at least 8, no more
than 2 new candidate failures occur under superseded or adjacent-memory controls, every admitted
cell proves that the intended memory arm was available, and every model, prompt, corpus, task,
checker, seed, retry, and timeout identity remains paired.

## Predictions

I predict C1 raises complete rank 10 coverage by at least 0.02 over C0 while keeping Search p95
below three times C0. I predict C2 raises mean reciprocal rank by at least 0.02 over C1 and is most
helpful on cross-session and failed-attempt tasks. I predict C3 raises mean reciprocal rank by at
least 0.03 over C2 without reducing rank 100 complete coverage by more than 0.01. I predict C4
keeps complete returned-budget coverage within 0.01 of C3 while reducing mean returned characters
by at least 0.30.

I predict C4 wins the screen and reaches at least 8 net executable wins over C0 in the 102-cell
confirmation. These predictions are directional release criteria for the local proxy, not a
private AML score forecast.

## Stop rules

Stop and mark the affected arm invalid if evaluation memory crosses a user scope, Add acknowledges
before all enabled indexes are searchable, a contextual document group changes chunk order,
learned sparse coverage is partial, Search returns generated answers, a benchmark label enters the
product, provider identity drifts, or raw benchmark content enters application logs.

Do not interrupt an existing provider-backed replay or start a second embedding process on VPS2.
Do not spend an AML Full run on this matrix. Only the locally selected, contract-verified release
candidate may proceed to AML Smoke.

## Artifacts

Store immutable replay artifacts under `docs/results/aml-coding-memory-matrix-v1/`. Record both
repository commits, dependency lock digests, model and prompt identities, task and corpus digests,
SPLADE revision and device, configuration settings, fallback counts, costs, and invalid cells.
Append results below this line without changing any number or prediction above it.

<!-- RESULTS APPEND BELOW; EVERYTHING ABOVE IS FROZEN -->

## Premeasurement correction, 2026-09-18

No matrix measurement had started when this correction was recorded. The 102-cell final grid is
explicitly the present condition, so the earlier requirement about new failures under superseded
or adjacent-memory controls is not measurable inside that population. It is removed from the
present-condition pass gate rather than inferred from absent cells. The final pass gate therefore
requires at least 8 net executable wins, complete treatment admission, and paired identities.
Control-condition safety remains a separately preregistered follow-up and cannot be used to rescue
or reject this present-condition result.

## Premeasurement corpus reuse amendment, 2026-09-18

No matrix measurement had started when this amendment was recorded. To avoid paying for identical
Voyage embeddings more than once, corpus identity and reuse are now part of the frozen artifact
lineage. C0 creates the raw dense corpus. C1 reuses that exact tenant and those exact raw chunks,
then builds only the SPLADE sidecar without calling Voyage. C2 independently creates the raw plus
procedure dense corpus and its SPLADE sidecar. C3 and C4 reuse that exact tenant, those exact
chunks, and the complete sidecar without calling Voyage or Add.

The five-arm retrieval pass and twelve-task executable screen therefore each require exactly two
dense embedding passes. Reused arms report `corpus_reused=true`,
`dense_embedding_pass=false`, and null Add latency. C0 and C2 report
`corpus_reused=false`, `dense_embedding_pass=true`, and measured Add latency. Search metrics remain
paired because source bytes, chunk bytes, chunk order, user boundary, and queries do not change.
The final executable comparison creates one dense corpus per distinct representation in that
two-arm comparison; C1 may reuse C0 because both use the raw representation.

## In-flight apparatus amendment, 2026-09-18

C0 completed before this amendment and its immutable artifact has SHA-256
`83ab52717a551724d646c2912d0c61fd2bfbc7ec6be3d81107543ee073968d19`. C1 produced no replay
artifact and executed no Search before its client abandoned the still-running SPLADE preparation
request at the general 180 second transport limit. The server continued the same sidecar build.
At 13 minutes it had encoded 576 of 2,284 dense chunks, projecting to about 52 minutes on the
frozen CPU device.

Only the `/v1/sparse/backfill` transport window is therefore amended to 7,200 seconds. Add remains
180 seconds, Search remains 60 seconds, task timeouts and retry rules remain unchanged, and no
corpus, query, model, provider, score, gate, prediction, or selection number changes. C0 is not
rerun or rewritten. C1 resumes only after the original sidecar build reaches a terminal state.
Every reused arm records `sparse_backfill_timeout_seconds=7200`; non-reuse arms record null.

## In-flight C2 compiler repair amendment, 2026-09-18

C2 produced no replay artifact and executed no Search before Add rejected a compiler record whose
only substantive field was removed by evidence grounding. The partial procedure tenant held 8 of
196 durable idempotency receipts, 99 dense chunks, and 99 matching SPLADE sidecars. C0 and C1
remain immutable.

The product repair revalidates each grounded record, rejects any record that lost all substantive
fields, and lets the existing deterministic fallback keep Add searchable. C2 resumes in the same
tenant without Delete and replays every deterministic request ID. The first 8 successful requests
must resolve from their receipts without provider or embedding work; the remaining requests finish
the same second dense embedding pass. The artifact records `ingest_resumed=true` only for C2.

C0 and C1 retain served commit `714d4a8190ce8318c9458d9fb185cea9422bdb91`. C2, C3, and C4 must
share the signed repair commit, while every other served identity field remains equal. This
apparatus repair changes no corpus bytes, query, model, provider, score, gate, prediction, or
selection number.

## In-flight C2 worktree execution correction, 2026-09-18

The first repaired C2 resume also produced no replay artifact and executed no Search. It advanced
the same procedure tenant to 60 durable receipts, 764 dense chunks, and matching sparse coverage,
then repeated the prior validation failure. The service reported the repair commit from its
environment, but the shared virtual environment console script imported the checkout where that
environment was installed, so the repaired source was not executing.

The service unit now runs the pinned working directory with `python -m recall_aml` instead of the
shared `recall-hosted` console script. C2 resumes again without Delete from the same request IDs.
The 60 receipts must remain cache hits, and the next compiler diagnostic must include the new
`rejected_substance` field. No benchmark identity other than the signed repair commit changes.
