# Preregistration: RE-call retrieval performance baseline v1

Written 2026-09-06. This protocol is committed before any measurement for this baseline. The
prediction and configuration sections are frozen by that commit. Results belong in a separate
timestamped artifact under `benchmarks/results/`; this file is not edited to improve a result.
Any change to the protocol requires a new amendment file that names the original commit and the
reason for the change.

## Prior work searched

Search performed 2026-09-06 through the RE-call documentation memory corpus with the queries:

* `RE-call benchmark preregistration performance baseline cold warm retrieval concurrency memory`
* `performance benchmark retrieval embedding reranking trust stages fast quality profiles k candidate pool`
* `benchmark tests skipped database hangs preregistration commit before measurement`

The search was calibrated and trusted, but marked stale, so current source and runtime output take
precedence. It found three relevant boundaries:

* `benchmarks/latency.py` measures sequential adapter latency and warmup, but not the profile,
  candidate pool, concurrency, error, memory, or per stage matrix fixed here.
* `recall/profiles.py` defines the shipped fast and quality profiles. Both use candidate pool 20 and
  return k 5. Fast has no reranker. Quality uses the pinned local reranker and separate admission
  limits.
* `recall/retriever.py` and `recall_mcp/service.py` already expose stage timings for query
  embedding, dense and sparse retrieval, fusion, reranking, trust evaluation, evidence assembly,
  and admission. `docs/ENTERPRISE_RETRIEVAL.md` documents this additive cost surface.

No prior result was found that is directly comparable across every dimension in this protocol.
Existing quality or latency numbers must not be substituted for this baseline.

## Question and prediction

This baseline asks: what latency, failure, concurrency, and memory cost does the current retrieval
path pay across cold and warm execution, the fast and quality profiles, result depth, candidate
pool size, and offered concurrency?

The pre-registered predictions are:

1. Warm retrieval will have lower p50 and p95 than cold retrieval for both profiles because model,
   database, and operating system caches are established before the warm sample.
2. Quality will have higher p50 and p95 than fast, with the difference concentrated in the
   reranking stage rather than embedding, database retrieval, or trust evaluation.
3. Increasing k while holding candidate pool fixed will have a smaller effect than increasing the
   candidate pool, except when the requested k approaches the realised pool size.
4. Increasing offered concurrency will increase p95 before it materially changes p50, and the
   first nonzero error rate will be caused by the configured admission limit or an infrastructure
   failure, not by silently dropping a completed response.
5. Peak resident memory will be higher for quality than fast because the quality profile loads a
   reranker. Peak memory will be reported as a process measurement, not inferred from Python heap
   allocations.

These are directional predictions only. No pass or fail threshold is chosen after seeing a number.

## Fixed workload

The benchmark uses one immutable corpus snapshot and one immutable query set for every arm.

* Corpus: the benchmark fixture selected by the run manifest, with its source and content digest
  recorded in every result artifact. The benchmark must refuse to mix generations or tenants.
* Queries: the fixed query file selected by the run manifest, in deterministic order, with its
  content digest recorded. Queries are not sampled differently per arm.
* Embedding: the same registered embedding profile for every arm. Query embedding is included in
  the stage timings. Indexing and document embedding are outside this retrieval baseline.
* Retrieval: the current hybrid retrieval path with its normal trust policy. No answer generation,
  network reranker, prompt construction, or result postprocessing is included in the timed request.
* Hardware and software: record operating system, CPU model and count, RAM, Python version, package
  revision, PostgreSQL and pgvector versions, embedding profile identity, reranker identity and
  artifact digest, database generation, and relevant environment overrides.
* Isolation: run on an otherwise idle host. Do not run two benchmark workers, embedding processes,
  or index refreshes at once. Do not mutate the production or shared memory corpus.

The default matrix is deliberately bounded:

| Dimension | Fixed levels |
|---|---|
| Profile | `fast`, `quality` |
| Retrieval state | cold, warm |
| k | 1, 5, 10 |
| Candidate pool per leg | 20, 50, 100 |
| Offered concurrency | 1, 2, 4, 8 |

Every valid combination is run. A combination with candidate pool smaller than k is invalid and
must be recorded as refused rather than silently clamped. The shipped profile identity and the
explicit candidate pool are both recorded, because production profiles currently fix their own
candidate pool while direct baseline sweeps may vary it.

## Sampling and cold versus warm definition

Each configuration is run in a fresh worker process. The worker performs a fixed warmup of 10
queries, discards those samples, then records 50 sequential requests at offered concurrency 1 and
50 completed requests for each concurrent level above 1. The query order wraps deterministically
when the query set is shorter than the requested sample count.

* Cold means the first request after worker startup and retrieval object construction. Its timing
  includes one time-to-first-request observation and its stage surface. It is reported separately,
  not combined with warm requests.
* Warm means the post-warmup samples in the same worker. Model loading, connection establishment,
  and cache priming are not charged to the warm request samples.
* For concurrent runs, each request has a client-side start and end timestamp. A request is
  completed only when a valid retrieval result or an explicit error is returned.
* The first request is not treated as a representative p50 sample. It remains a named cold sample,
  while warm percentiles are computed only from the post-warmup population.

## Measurements

For every profile, state, k, candidate pool, and concurrency combination, publish:

* p50 and p95 wall latency in milliseconds for all requests, successful requests, and failed
  requests when the failed population is nonempty;
* error rate as failed requests divided by attempted requests, with counts for successes, explicit
  retrieval refusals, timeouts, transport errors, and unexpected exceptions;
* peak resident set size in bytes for the worker and the delta from the worker baseline, including
  a separate cold and warm observation;
* p50 and p95 for the separate stages `query_embedding`, `dense_retrieval`,
  `sparse_retrieval`, `learned_sparse_retrieval`, `fusion`, `reranking`, `trust_evaluation`,
  `evidence_assembly`, and `admission_wait` when the stage is present;
* the stage sample count and error count. A disabled stage remains present with zero duration when
  the runtime contract reports it, and an absent stage is a harness error rather than a zero;
* throughput, offered concurrency, completed concurrency, and the number of requests shed before
  query embedding;
* the full configuration identity, including profile, k, candidate pool, reranker status,
  embedding identity, generation, corpus digest, query digest, and worker revision.

Stage percentiles are calculated independently from the raw per-request stage observations. Total
latency is not reconstructed by summing rounded stage percentiles. Percentiles use the repository's
nearest-rank convention from `recall.observability.percentile`, with unrounded samples retained in
the raw artifact.

## Error and validity rules

An error is any attempted request that does not return a valid retrieval result. Admission shedding,
budget exhaustion, database errors, embedder errors, reranker errors, timeouts, and transport errors
are distinct error classes. A request shed before embedding contributes to error rate and capacity
counts, but has no retrieval stage samples.

A run is invalid and must not be interpreted as a baseline if:

* the corpus or query digest differs between arms;
* the worker uses a different embedding or reranker artifact than the manifest declares;
* the host is not recorded or another embedding, index, or benchmark process overlaps it;
* the database generation or tenant changes during the run;
* fewer than the prescribed samples are attempted without an explicit refusal artifact;
* the database dependent path is skipped. A missing database is a refused run, not a green zero;
* a result is silently clamped, retried, or dropped without a recorded event and original error.

Retries are disabled for benchmark requests. If a retry is needed to diagnose infrastructure, it is
outside the measured run and must be disclosed in the result note.

## Artifact and analysis plan

The runner writes one JSON artifact per run containing the protocol revision, git commit, manifest,
host metadata, raw request rows, raw stage rows, memory observations, error events, and derived
percentiles. A compact Markdown summary may be generated from that JSON, but the JSON is the source
of truth.

The primary tables are:

1. warm p50 and p95 wall latency by profile, k, candidate pool, and concurrency;
2. warm p50 and p95 per stage by the same dimensions;
3. cold first request latency and memory by profile, k, and candidate pool;
4. error rate and error class counts by offered concurrency;
5. peak worker RSS and RSS delta by profile and retrieval state.

Cold versus warm and fast versus quality are paired by query and configuration wherever the worker
can preserve the same sequence. No post hoc removal of slow queries, failed requests, or high memory
observations is allowed. If a configuration is invalid, the result is marked invalid and the reason
is retained.

## Reproduction command

The measurement command will be added after this preregistration is committed. It must print the
preregistration commit, refuse an unstamped or dirty protocol, and write the result artifact without
modifying this file.
