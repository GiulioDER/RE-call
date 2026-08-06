# Learned sparse retrieval (SPLADE) as a pgvector `sparsevec` leg

`design · 2026-08-06`

## Why

MTRAGEval Task A rank tracks how much **learned sparse** is in the stack: rank 1 is ELSER (learned
sparse, single retriever, 0.5776), rank 2 is LION-SP-8B (learned sparse on an LLM backbone), rank 3
is dense + SPLADE-v3 + RRF. **Pure dense finishes 14th of 38.** RE-call is dense-primary with a
LEXICAL sparse leg and scores 0.4227, below both rank 14 and the organizers' own ELSER baseline of
0.480. That is a structural gap and tuning the dense side does not close it.

ELSER needs Elasticsearch, which is a non-starter for a Postgres-native system. SPLADE-v3 is open
and pgvector has carried `sparsevec` since 0.7, so learned sparse is implementable in the store
RE-call already has, with no new search engine.

## The measurable claim

On MTRAG-human **dev**, does `dense + SPLADE` beat `dense + ts_rank`? Embedder, fusion, reranker
and chunking are held fixed; the sparse leg's backend is the only thing that varies.

Arms, frozen before any score is observed:

| arm | legs | role |
|---|---|---|
| `hybrid_lexical` | dense + ts_rank | control (today's config, on the dev split) |
| `hybrid_splade` | dense + SPLADE | **primary** |
| `splade_only` | SPLADE | ablation; lexical alone is 0.2542 |
| `hybrid_both` | dense + ts_rank + SPLADE | secondary, pre-registered so it is not a post-hoc rescue |
| `dense_only` | dense | ablation floor |

`Recall@100` is recorded for every arm as a byproduct of retrieving at depth 100. It is **not** the
gating diagnostic the source memo proposed; that sequencing was considered and declined by the user
on 2026-08-06, and this document records the decision rather than re-litigating it.

⚠️ The current harness scores `mtragun-human`, which is the **sealed** test set. This work adds a
dev-split path. No arm below is ever scored on the sealed set.

## Measured platform facts

Probed against the running pgvector **0.8.4**, not read from documentation:

| property | result |
|---|---|
| `sparsevec(30522)` type | supported |
| `<#>` inner product (SPLADE's scoring function) | supported |
| HNSW `sparsevec_ip_ops` | supported |
| non-zeros the **type** accepts | 16,000 |
| non-zeros an **HNSW-indexed** column accepts | **1,000** |

The last row is load bearing and was not predicted. Inserting a 1,001-non-zero vector into an
HNSW-indexed column raises `sparsevec cannot have more than 1000 non-zero elements for hnsw index`.
SPLADE passage expansions can exceed that, so **top-k pruning to <=1000 is a required, asserted step
in the writer**, not a detail. It fails loudly rather than silently, which is the right direction,
but a writer that relies on the database to reject its mistakes is not a writer that prunes.

## Models

| model | license | backbone | dim |
|---|---|---|---|
| `prithivida/Splade_PP_en_v1` | apache-2.0 | bert-base-uncased | 30522 |
| `naver/splade-v3` | **cc-by-nc-sa-4.0** | BertForMaskedLM | 30522 |

Both verified from the model cards. RE-call ships MIT on PyPI, so the Apache-2.0 encoder is the
**default** and `naver/splade-v3` is opt-in by explicit config, for the number that is comparable to
MTRAGEval rank 3. Weights are downloaded by the user at runtime and never vendored. The licence is
documented at the config site, so nobody adopts a non-commercial default by accident.

Same dimensionality, so one schema serves both. They are still distinct identities: vectors from one
must never be compared against the other.

## Components

### `recall/sparse.py`

`SparseEncoder` protocol and `SpladeEncoder`. Inference is written directly against
`transformers`: `log1p(relu(logits)) * attention_mask`, max-pooled over the sequence.

Not `sentence-transformers`. Its `SparseEncoder` API needs ST >= 5.0, while `rerank` and `entail`
pin `>=3.0` with a comment in `pyproject.toml` saying to bump all three together. Writing ~20 lines
against `transformers` avoids dragging the reranker's floor up as a side effect of a retrieval
experiment, and gives direct control over pruning.

Ships as a new `sparse` extra. Nothing existing changes its floor.

### `SparseProfile`

A frozen identity dataclass beside `EmbeddingProfile`, deliberately **not** an extension of it.
`EmbeddingProfile.fingerprint()` documents stability as a contract and is pinned by a test that
transcribes its encoding independently; a new field there re-partitions every dense cache in
existence. The sparse identity therefore stands alone.

🔑 **The top-k prune budget is a field of this identity.** Pruning changes the stored vector, so a
corpus encoded at top-k 512 is not the same corpus as one encoded at top-k 1000, and the fingerprint
has to say so.

### Schema — migrations `0012` (transactional) and `0013` (concurrent index)

```sql
CREATE TABLE recall_sparse_v1 (
    tenant_id   TEXT NOT NULL,
    chunk_table TEXT NOT NULL,
    profile_id  TEXT NOT NULL,
    id          TEXT NOT NULL,
    vec         sparsevec(30522) NOT NULL,
    nnz         INTEGER NOT NULL,
    PRIMARY KEY (tenant_id, chunk_table, profile_id, id)
);
CREATE INDEX CONCURRENTLY recall_sparse_v1_vec_idx
    ON recall_sparse_v1 USING hnsw (vec sparsevec_ip_ops);
```

Two files, because `load_migrations` requires each migration to declare exactly one execution mode
(`schema.py:229`).

**Global, not per-target, and the reason is mechanical.** `GLOBAL_MIGRATION_START = "0008"`
(`schema.py:61`) routes every migration numbered >=0008 to the `__global__` ledger, applied once per
database. A per-target templated sidecar at 0012 would be created for whichever table migrated first
and skipped for every other one, in silence. The MTRAG harness provisions one table per domain
(`benchmarks/mtrag/run.py:334`), so that failure mode is not hypothetical here; it is the normal
path. `chunk_table` in the key is what lets one global table serve every target.

`nnz` is stored rather than derived so "how much did pruning remove?" is answerable from the table
without re-encoding.

New file only. Migration `0008` was once edited in place, which made every pre-existing database
recreate itself via `DROP SCHEMA public CASCADE` — a class CI on an empty database structurally
cannot see. No existing migration is touched.

### `PgVectorStore.query_learned_sparse()`

Sibling of `query_sparse`. Timed under `STORE_QUERY_METRIC{leg=learned_sparse}`. Takes the same
`vec=` argument so hits carry their true dense cosine and stay comparable with dense hits
downstream, exactly as the lexical leg does.

Filtering by `(chunk_table, profile_id)` against a shared HNSW index means a filtered vector scan;
the store's existing `_hnsw_filtered_tuning` handles that case and is reused rather than
reimplemented.

### `HybridRetriever(sparse_backend=...)`

`"lexical"` (default, behaviour unchanged) | `"splade"` | `"both"`. Same RRF fusion; `"both"` adds a
third ranking to the existing two.

**Fail closed.** With `sparse_backend="splade"` and no rows in the sidecar for this
`(tenant, chunk_table, profile)`, the retriever **raises**. It does not degrade to dense-only. That
precise silent degradation has already happened here once: the conjunctive `websearch_to_tsquery`
returned rows for 0 of 150 questions, `_rrf` fused a single non-empty list, and the retriever
quietly became dense-only with nothing failing and no test noticing.

## Erasure, and the production refusal

The sidecar cannot carry `ON DELETE CASCADE`: its parent table name is a column value, not a schema
reference. The erasure paths that exist — `generations.py:784 forget()` and
`control_plane.py:449 erase_sources_from_pending()` — cannot know about a table introduced after
they were written.

SPLADE vectors are term weights over the vocabulary, so an un-erased sidecar row is partially
reconstructable content, not an opaque hash. That is a compliance hole.

⛔ **The learned sparse leg therefore refuses to initialise when `RECALL_ENV=production`.** Wiring
the sidecar into `forget`, `replace_sources` and the erasure outbox is the precondition for lifting
that refusal, and it is deliberately out of scope here: this experiment is benchmark-grade and is
labelled as such rather than shipped as if it were not.

## Encode pipeline

`scripts/encode_sparse.py` reads passages and writes pruned sparse vectors to `jsonl.zst`. It takes
**no DSN and no secrets**, so it runs on rented GPU hardware that never touches the database or the
credentials. A separate loader streams the artifact into the sidecar.

The split is deliberate: a failed or reclaimed rental costs the remaining encode, not the completed
work, and no credential ever leaves the trusted hosts.

366,479 passages across four domains (clapnq 183,408 · cloud 72,442 · fiqa 61,022 · govt 49,607).
Throughput is **measured on 1,000 passages and the extrapolation reported before** any full run is
started. No hour estimate in this document is a plan.

Query-time encoding is a single CPU forward pass (~30ms), so serving needs no GPU and stays
deployable on VPS2.

## Testing

Integration tests run against a real PostgreSQL on a **dedicated database** — never the shared
`recall` one, which other sessions are using concurrently.

Each guard is **mutated**, because a guard that has never been shown to fail is a hypothesis:

| guard | mutation that must turn it red |
|---|---|
| raises when the sidecar is empty | make it return `[]` instead |
| pruning actually prunes | remove the top-k; the 1001-nnz insert must fail |
| `sparse_backend` changes which documents return | ignore the parameter |
| `RECALL_ENV=production` refusal | drop the check |
| prune budget is in the fingerprint | drop the field; two budgets must stop colliding |

The third is the one that matters most: without it, every other test passes with the parameter wired
to nothing.

An end-to-end check on a toy corpus proves a lexical-miss / semantic-hit document is retrieved by
SPLADE and not by `ts_rank` — the capability claim, executed rather than asserted.

## What could make this null

- **VPS2 is `is-system-running: starting`** after the 2026-07-31 filesystem incident. The
  measurement depends on that host.
- The 4-domain figure is already optimistic: the official leaderboard scored 6, including Banking
  and Telco, which the MTRAG-UN authors report as much harder.
- ELSER's rank-1 result is partly an **annotation artefact** (annotators primarily reviewed
  ELSER-retrieved passages). SPLADE has no such advantage. Beating `ts_rank` is the honest target;
  0.578 is not.
- **HNSW recall on 30522-dimension sparse vectors is unmeasured.** If it is poor the fallback is an
  exact scan (183k rows max per domain). This is measured, not assumed, before any result is read.
- The lexical leg's latency did not generalise from synthetic to real corpora before (4ms vs 496ms).
  The learned sparse leg's latency is measured on the real corpus for the same reason.

## Out of scope

Multi-query diversity with nested RRF, and the reranker upgrade to BGE-reranker-v2-m3. Both are
separate levers deserving separate measurements. Erasure integration, per the refusal above.
