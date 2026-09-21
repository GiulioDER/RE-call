# AML CAMBench Coding exact parity candidate preregistration

Date frozen: 2026-09-20

Status: preregistered before hosted parity replay, AML Smoke, or AML Full measurement

## Decision and correction boundary

The candidate for the next CAMBench Coding run is `C6_code4_exact_bm25`. This record does not
edit or reinterpret the earlier `C5_code4_bm25` preregistration. It replaces C5 for future
official use because an independent prelaunch audit found three material differences between C5
and the successful Voyage Code4 screen.

1. The screen ranked exact cosine scores over every window, while C5 used approximate HNSW.
2. The screen resolved dense, BM25, and RRF ties by stable corpus window order, while C5 used
   request derived chunk identifiers or no secondary dense order.
3. The screen windowed normalized transcript content, while C5 inserted timestamp, role, and
   content labels before windowing.

C5 remains a valid historical deployment receipt. It is not licensed for AML Smoke or Full.

The intended division remains Commercial Products. External Voyage embeddings are therefore not
presented as eligible for the Open source Methods prize.

## Frozen Add path

1. Normalize PostgreSQL NUL characters with the existing hosted contract.
2. Preserve AML message order and join message content with one space. Do not insert role,
   timestamp, or content labels.
3. Split the content stream into raw whitespace separated windows of 160 words with stride 120.
4. Embed every window with registered profile `voyage-code-4-v1`, model `voyage-code-4`, width
   1024, using document input semantics.
5. Persist raw windows only.
6. Derive opaque chunk identity from source session identifier, segment, and content. Transport
   request identifiers do not affect chunk identity.

For the parity replay against the pinned Agent Memory Benchmark corpus, the caller must supply the
same normalized field stream used by `readable_text`. The local hosted adapter currently adds tool
labels for a different ingestion contract, so that adapter output is not accepted as parity input
without a separately measured renderer comparison. This statement concerns the local adapter. It
does not claim that the private AML runner adds those labels.

## Frozen Search path

1. Embed the original query once with `voyage-code-4-v1` using query input semantics.
2. Compute exact tenant scoped pgvector cosine scores with ordered index scans disabled.
3. Return dense top 100 ordered by cosine distance, source session identifier under PostgreSQL C
   collation, numeric window segment, then opaque chunk identifier.
4. Rank every window for the same user with `canonical-bm25-k1-1.5-b0.75-v1`.
5. Keep the top 100 positive BM25 windows, resolving equal scores by the same stable window key.
6. Fuse dense and BM25 ranks with unweighted reciprocal rank fusion, constant 60.
7. Resolve fused score ties by the same stable window key and return at most 100 results.

Missing or malformed source session and segment metadata is a refusal for stable ranking. It does
not silently fall back to request order or chunk identifier order.

The candidate does not use a compiler, facet generator, reranker, SPLADE encoder, code neighbour
expansion, graph sidecar, evidence packer, or multimodal embedding path.

## Official execution settings

The live Coding contract requires Max Add concurrency from 16 through 64 and Search concurrency
from 16 through 256. Set both Evaluation page values to 16. Search Top K remains platform fixed at
100. The service may retain its measured internal semaphore of 3, which queues excess platform
requests inside the single application process.

Do not start a job while the live track catalog reports `worker_available: false` for the selected
phase.

## Evidence carried forward

The paired Code4 Task Solve replay admitted 100 pairs and recorded 67 successful Code4 attempts
against 59 successful Code3 attempts, a net gain of 8. Its source retrieval check recovered all 34
queries at recall at 10 and increased MRR from 0.5063 to 0.8464. Those measurements license
Voyage Code4 plus canonical BM25 as the mechanism. They do not license C5 implementation details
that differed from the measured screen.

## New parity and launch gates

No AML Smoke or Full run may start until every gate below passes for one clean committed release.

1. A direct fixture proves content only session rendering, 160 by 120 windowing, and transport
   request independent chunk identity.
2. Dense, BM25, and final RRF tie tests prove source session plus numeric segment ordering.
3. A PostgreSQL test with 101 identical embeddings returns stable ordinals 0 through 99 for an
   exact top 100 request.
4. The pinned 1,220 window corpus replay compares the reference screen and the hosted C6 path for
   all 34 queries. It records top 100 overlap, first relevant rank, recall at 10, recall at 100,
   and MRR. Any difference is reported before a launch decision.
5. Sixteen simultaneous Add clients and sixteen simultaneous Search clients complete with zero
   errors or timeouts against the public endpoint. Exact dense and total Search latency are
   recorded separately.
6. The built wheel and clean commit are bound by a C6 release manifest.
7. `/version` reports C6, exact dense true, ordering profile
   `source-session-c-collation-segment-v1`, renderer profile `message-content-only-v1`, Voyage
   Code4, canonical BM25, window size 160, and stride 120.
8. The candidate generation is empty immediately before AML Smoke.
9. Health, isolation, idempotency, deletion, restart persistence, and external HTTPS checks pass.
10. The live Coding worker pool is available and AML Smoke passes for the bound version.

If any gate fails, stop. Diagnose the failure, create a new preregistration and release identity,
then repeat the gates. Do not patch a bound deployment in place and continue.

## Full run rule

Use one Full run for CAMBench Coding only after all gates and AML Smoke pass. Record the job
identity, bound version identity, start time, completion state, and platform result without editing
this preregistration. The second Full allowance remains unspent unless a later candidate earns it
under a separate preregistration.
