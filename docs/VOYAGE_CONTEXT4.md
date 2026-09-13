# Voyage Context 4 operations

Voyage Context 4 is a separate registered embedding profile, not a larger `voyage-4` setting.
Queries use `contextualized_embed` with `input_type="query"`. Documents use
`input_type="document"` and nested inputs, where each inner list is one logical document:

```text
inputs=[["chunk 1", "chunk 2", "chunk 3"]]
```

The production profile is `voyage-context-4-v1`, model `voyage-context-4`, dimension 1024, and
grouping policy `voyage-context-document-v1`. Its grouping policy is explicit and participates in
the pipeline fingerprint.

## Grouping policy

The default boundary is one manifest object per group. A conversation, chat export, or workspace
may use a shared `context_group_id` only when the ingestion manifest has identified those objects
as one logical document. Unrelated files must retain distinct group IDs. Graph and structured
records use one group per logical record set or snapshot, never one group for an entire unrelated
workspace.

Incremental builds reuse a source only when its bytes, version, pipeline fingerprint, and complete
group membership fingerprint still match. A change to group membership therefore re-embeds every
member of that group. Empty objects remain valid zero chunk sources. Failed embedding leaves the
previous active generation untouched.

## Provider limits and cache semantics

The implementation requests float vectors at dimension 1024, allows up to 1,000 inputs and 16,000
chunks per request, and uses a conservative 60,000 character bound. Pre-chunked requests stay
within the provider's 32K token limit. Oversized groups are split only between original chunks,
never truncated, then flattened back with exact one-to-one alignment. Provider retries are bounded
and non-retryable response shape or dimension errors fail the build.

Passage vectors are not stored in the normal per-text embedding cache because a passage vector
depends on its neighboring chunks and group membership. Query vectors use the ordinary query path.

## Build and rollout

Use one embedding process on VPS2 under `.locks/embed.lock`. Build a candidate from a verified
manifest, validate it, and keep it ready but unpromoted while the evidence gates run:

```bash
RECALL_ENV=production RECALL_LOCAL_ALLOWLIST=/path/to/corpus \
  recall --tenant memory --embedder voyage-context:voyage-context-4 \
  generation build manifest.json --manifest-sha256 SHA256 --manifest-size BYTES \
  --embedder-provider voyage-context --project memory-corpus --no-commit-stamp
recall --tenant memory generation validate GENERATION
recall --tenant memory --embedder voyage-context:voyage-context-4 \
  calibration calibrate --generation GENERATION --queries LABELS.json --publish
```

The generation fingerprint and profile identity must match at calibration and serving time. Do not
reuse a Voyage 4 calibration. Compare representative queries against the active Voyage 4
generation, including evidence IDs, sparse fusion, reranking, trust status, latency, provider
errors, and abstention behavior. Promote only after those checks and the rollback target are
recorded:

```bash
RECALL_ENV=production recall --tenant memory generation promote GENERATION
RECALL_ENV=production recall --tenant memory generation rollback
```

Keep the previous Voyage 4 generation available through the retention window. Reranking remains a
separate second stage: evaluate the Context 4 embedding gain without a reranker, then evaluate the
composition with the existing Voyage reranker.

## When to choose it

Use Context 4 when cross-chunk context improves retrieval enough to justify hosted embedding cost,
egress, and request latency. Use Voyage 4 when lower cost, simpler per-text semantics, or a
smaller operational surface is more important. The frozen LoCoMo comparison and the four-arm
production-path evidence are recorded in `docs/results/2026-09-13-voyage-context4-production-path.md`.
