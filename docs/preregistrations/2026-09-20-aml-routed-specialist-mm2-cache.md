# Routed specialist MM2 and shared embedding cache qualification

Status: frozen when committed. No live provider qualification described here may begin before this
record and its implementation are committed.

## Question

Can `C7_routed_specialists` incorporate the retained `MM2_dual` implementation and reuse exact
hosted embeddings without changing its coding ranking, cross-model isolation, or tenant isolation?

This is a private preflight for Agent Memory Benchmark. It does not authorize the official Agent
Memory Leaderboard Smoke or evaluation.

## Frozen design

The branch incorporates PR 684 through signed commit
`f083b395faa3ef7330037deb838ea1e8f7bc8a9c`. The scientific MemEye verdict remains `NO_GAIN`.
The explicit operational decision retains MM2 because it had the highest valid answer quality while
preserving the shared Recall at 10 and Recall at 100.

C7 uses the retained component as follows:

1. Coding and ambiguous text route only to Code4 plus canonical BM25.
2. Explicit conversational memory routes only to Context4 plus canonical BM25.
3. Image input or an explicit visual query routes to the MM2 path, which preserves exact media and
   fuses text and Voyage Multimodal 3.5 rankings by reciprocal rank.
4. No raw similarity score crosses an embedding space.

The optional shared cache is a bounded SQLite file configured with
`RECALL_AML_EMBED_CACHE_PATH`. Code4 cache keys bind the full registered profile, dimension,
purpose, and exact text. Context4 keys additionally bind the complete ordered document group and
ordinal. MM2 keys bind the exact canonical structured input and separate document from query
vectors. The same cache file is used by the isolated C6 and C7 processes, while their database
tables and user namespaces remain separate.

The cache wrapper is inside the existing VPS2 embedding lock. Therefore a second process looks up
the cache only after the first process has finished filling a miss. A cache hit may avoid a provider
call but cannot avoid Add, persistence, Search, tenant filtering, or rank fusion.

## Qualification

The prior C6 versus C7 nine gate qualification is repeated at the new immutable commit, with these
additional requirements:

1. `/version` reports `voyage-multimodal-3.5-v2`, embedding locking, and embedding cache enabled.
2. Code4 and Context4 still contain the same 1,220 logical chunk identities and payloads, with
   distinct vectors.
3. All 34 frozen coding prompts route to Code4 and retain C6 versus C7 ordered top 100 parity.
4. The conversational and visual canaries select Context4 and MM2 respectively and return the
   planted evidence at the preregistered rank.
5. Unit level cross-instance proofs demonstrate a second exact Code4 passage, Code4 query,
   Context4 ordered group, MM2 document, and MM2 query is served without invoking the fake provider.
6. Separate user namespaces remain unable to retrieve one another's evidence, including when their
   source content is identical and their derived vector came from one cache entry.

## Predictions

1. Every earlier nine gate qualification remains green.
2. All 34 coding rankings retain exact C6 versus C7 ordered parity through rank 100.
3. The new visual canary remains rank 1 after the v2 pixel fitting change.
4. Cache enabled and cache disabled paths produce the same stored chunk ids and the same retrieval
   order for deterministic fake embeddings.
5. The cache reduces live provider work on repeated inputs. This is an operational prediction, not
   a task quality claim. If live provider call counts cannot be observed without changing the
   provider boundary, the unit proof and shared cache identity are reported separately and this
   prediction remains unmeasured rather than inferred from latency.

## Decision rule

Any ranking, routing, shared identity, tenant isolation, or visual canary failure rejects the
integration. An unobservable live provider call count does not reject quality equivalence, but it
prevents a measured cost saving claim. The five condition AMB run may begin only after every safety
and ranking requirement passes.

<!-- results and append only corrections go below this line; everything above is frozen -->

## Qualification result

The private qualification ran on immutable commit
`28ed55c77928f3526f0c2c52ffc5d32dce0b7303` and completed at
`2026-09-20T21:06:41.127874+00:00`. All nine gates passed and
`official_aml_launched` was false.

Measured results:

1. C6 and C7 each stored 1,220 windows from 196 sessions.
2. The primary and Context4 indexes shared all 1,220 chunk identities and payloads. Their vectors
   differed for all 1,220 chunks.
3. All 34 coding prompts selected Code4 and had exact C6 versus C7 ordered parity through rank 100.
   Source recall at 10 was 34 of 34 and mean reciprocal rank was 0.861111.
4. The Code4, Context4, and MM2 canaries each returned the planted item at rank 1. MM2 preserved
   the ordered structured content byte exactly.
5. No forbidden response key was exposed and all five qualification namespaces were deleted with
   HTTP 200.
6. Both endpoints reported embedding cache and embedding lock enabled. The shared cache contained
   2,488 vectors after qualification. Unit tests proved exact reuse across independent wrapper
   instances for Code4 passages and queries, Context4 ordered document groups, and MM2 documents
   and queries.

Live provider call counts were not observable without changing the provider boundary, so the
cost saving prediction remains unmeasured. The cache behavior itself is proved at unit level and
the live endpoint configuration is proved by the version gates.

## Route observability follow up

The first five condition preparation revealed that the public Search diagnostics did not expose
the selected specialist route or embedding profile even though the internal qualification checked
them. No model session ran. Commit `4334084d13e38d02d881b2207c85152858e608d9`
adds the two diagnostic headers with a mutation proved HTTP test.

The full nine gate qualification was repeated at that commit and passed at
`2026-09-20T21:29:24.356286+00:00`, again with exact top 100 coding parity, rank 1 Code4,
Context4, and MM2 canaries, complete cleanup, and `official_aml_launched: false`. The subsequent
five condition preparation recorded `code` and `voyage-code-4-v1` for all 69 distinct C7 task
Searches.
