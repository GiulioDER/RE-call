# Enterprise retrieval gap matrix

Audit of the merged enterprise retrieval and evidence program against 21 requirement areas.
Read-only against the code. Nothing in this document was fixed; every gap is recorded, not repaired.

## What was audited, and against which tree

| Item | Value |
|---|---|
| Program PR | [#181](https://github.com/GiulioDER/RE-call/pull/181), merged 2026-08-03 at `1aa93ec`, 41 files, +2852/-133 |
| Tree audited | `origin/master` at `8147d96` |
| Branch | `codex/enterprise-gap-matrix`, cut fresh from `origin/master` |
| Program doc | [docs/ENTERPRISE_RETRIEVAL.md](../../ENTERPRISE_RETRIEVAL.md), 76 lines |

**The audited tree is not PR 181.** Five PRs merged after it and four of them changed code this
audit covers: #182 calibration binding (migration `0011`), #184 strict trust, #185 shared connection
pool and tenant-scoped readiness, #189 store latency instrumentation. Where an area's current state
is mostly later work, the row says so, because crediting PR 181 for it would misattribute the
evidence. Verdicts describe **master as it stands**, which is what a later session has to build on.

Every verdict below carries a `file:line` citation or an explicit statement that a symbol exists
nowhere. Where a claim is negative ("nothing calls this"), the check was `git grep` over tracked
files only, excluding `.venv`.

---

## The matrix

| # | Area | Verdict |
|---|---|---|
| 1 | Embedding profiles and asymmetric helpers | implemented, insufficiently tested |
| 2 | Cache role separation | partially implemented |
| 3 | Offline artifact validation | implemented, insufficiently tested |
| 4 | Retrieval profiles and config-conflict handling | partially implemented |
| 5 | Reranker bounds and overload handling | partially implemented |
| 6 | Stage timings and safe metrics | implemented, insufficiently tested |
| 7 | Structured contextual passage construction | implemented, insufficiently tested |
| 8 | Qwen experimental profile and recorded rejection | implemented; the **rejection record is missing** |
| 9 | Evidence bundle, prompt renderer, answer envelope, validator | implemented and tested, **unreachable** |
| 10 | CLI, MCP, LangChain, LlamaIndex additive fields | partially implemented (CLI leg missing) |
| 11 | Control plane migrations | partially implemented; two ledgers, see the decision below |
| 12 | Route notifications and polling | implemented, **untested** |
| 13 | Active and shadow dual writes | implemented, **untested** |
| 14 | Atomic forget across generations | implemented, **untested** |
| 15 | Durable migration outbox and replay | partially implemented; **replay has no producer and no operator command** |
| 16 | Readiness checks | implemented; the enterprise check is **untested**, start-up only, and one branch **cannot fire** |
| 17 | Calibration identity enforcement | implemented and tested (mostly by #182/#184) |
| 18 | Retrieval promotion gates | implemented; **no producer**, and the pass path is never exercised |
| 19 | Generation quality evaluation | **missing** |
| 20 | Deployment and rollback tooling | partially implemented |
| 21 | Documentation and operator runbooks | partially implemented; **two contradictions on master** |

---

## 1. Embedding profiles and asymmetric helpers: implemented, insufficiently tested

**Present.** `EmbeddingProfile` is a frozen dataclass with the eleven identity fields
(`recall/embeddings.py:157-178`), non-empty and positive-dimension validation at
`recall/embeddings.py:173-177`. `embedding_profile` / `embedding_profile_id` resolve a profile or
fall back to a legacy descriptor (`recall/embeddings.py:187-212`). The asymmetric helpers
`embed_query` / `embed_passages` fall back to the symmetric interface when the backend has no
distinct encoder (`recall/embeddings.py:215-227`), behind an `AsymmetricEmbedder` protocol
(`recall/embeddings.py:148-154`).

**Confirmed: the registry is a dict literal, and a second partial registry exists.**
`recall_mcp/service.py:113-120` is a `dict[str, str]` mapping profile ID to context version, inside
`make_embedder`. `recall/context.py:37-41` is a second `dict[str, ContextMode]` over the same
profile-ID vocabulary, inside `context_policy_for_profile`. They are independent literals with no
shared constant and no test that they agree. They already disagree in extent: the service map lists
six profiles including `qwen3-embedding-0.6b-384-v1`, the context map lists three and relies on
`.get(profile_id, "none")` for the rest. That default is currently correct, which is exactly why a
seventh profile added to one map and not the other would index silently under the wrong context
mode rather than raising.

**Dead identity fields.** `normalization`, `instruction_version`, `chunker_version` and
`dependencies` (`recall/embeddings.py:167-171`) are populated by the concrete embedders
(`recall/embeddings.py:374-385`, `:490-500`) and read by nothing. `git grep` finds no consumer for
any of the four outside their own definition and construction.

**Test coverage.** `tests/test_enterprise_embeddings.py:40-46` asserts the asymmetric helpers pick
the right encoder and that `embedding_profile_id` prefers the profile. That is the whole coverage of
this area. `make_embedder`'s profile branch has no test; `make_profile_embedder`
(`recall_mcp/service.py:147-162`), including its shadow env-var remapping at `:154-161`, has **no
test reference anywhere**.

## 2. Cache role separation: partially implemented

**Present.** `cache_key` hashes `(name, purpose, str(dim), text)` at `recall/cache.py:26`, and
`embed_with_cache` supplies `embedding_profile_id(embedder)` as `name` at `recall/cache.py:95-96`.
Purpose routes to the correct encoder at `recall/cache.py:86-91`. `embed_query_with_cache`
(`recall/cache.py:107-111`) exists so a query vector cannot alias a passage vector.

**Confirmed gap: the key omits two fields that change the stored vector.**
`EmbeddingProfile.context_version` (`recall/embeddings.py:170`) and
`EmbeddingProfile.artifact_digest` (`recall/embeddings.py:163`) are not key material. Both are
independently settable while `profile_id` stays fixed: `FastEmbedEmbedder.__init__` takes
`profile_id`, `artifact_sha256` and `context_version` as three separate parameters
(`recall/embeddings.py:334-339`). Two processes configured with the same `RECALL_EMBED_PROFILE` and
a different `RECALL_MODEL_SHA256` therefore share cache entries computed by different weights.

This is narrower than it looks in the default deployment, because `make_embedder` derives
`context_version` from `profile_id` through the map at `recall_mcp/service.py:113-120`, so the two
move together *on that path*. It is not narrow for a direct library caller, for the shadow path, or
for a re-provisioned artifact at the same profile ID. The index path does not share the weakness:
`index_fingerprint` at `recall/index.py:453-457` mixes content hash, profile ID, context mode and
context version, so the skip decision is stricter than the cache key.

**Test coverage.** `tests/test_enterprise_embeddings.py:48-58` proves query and passage vectors do
not alias, and that a second query call is served from cache. Nothing tests the omission above,
which is the failure this key exists to prevent.

## 3. Offline artifact validation: implemented, insufficiently tested

**Present.** `artifact_tree_sha256` walks a file or directory deterministically, hashes the relative
path alongside the bytes, and refuses a symlink that escapes the root
(`recall/embeddings.py:230-247`, escape check at `:239-240`, empty-tree refusal at `:236`).
`verify_artifact` validates the digest is 64 hex characters before doing any I/O, resolves strictly,
and raises on mismatch (`recall/embeddings.py:250-260`).

**Wiring.** `FastEmbedEmbedder` requires both a cache dir and a digest when `require_local`
(`recall/embeddings.py:347-350`) and passes `local_files_only=True` to fastembed
(`recall/embeddings.py:364-365`). `Qwen3EmbeddingEmbedder` verifies before load
(`recall/embeddings.py:473`). `CrossEncoderReranker` gained `local_files_only` + `artifact_sha256`
and refuses the offline path without a digest (`recall/rerank.py`, the `local_files_only` branch).
The quality retrieval profile refuses to start without `RECALL_RERANK_PATH` and
`RECALL_RERANK_SHA256` (`recall_mcp/service.py:355-359`).

**Test coverage.** `tests/test_enterprise_embeddings.py:60-66` covers the happy path and the
checksum mismatch. Untested: the symlink-escape refusal (`recall/embeddings.py:239-240`), the
empty-artifact refusal (`:236`), the malformed-digest refusal (`:252-253`), the `require_local`
preconditions (`:347-350`), and the reranker's entire offline path: `local_files_only`,
`artifact_sha256` and `inference_threads` have **no test reference anywhere in `tests/`**.

The symlink-escape branch is the one worth naming: it is a security control, and a control that has
never been shown to fire is a hypothesis.

## 4. Retrieval profiles and config-conflict handling: partially implemented

**Present.** Three immutable profiles (`recall/profiles.py:35-37`), a resolver that refuses an
unknown value (`:45-46`) and refuses a `RECALL_RETRIEVAL_PROFILE` that contradicts the legacy
`RECALL_RERANK` switch (`:47-52`), with positive-integer validation on the env overrides (`:55-65`).

**Lead refuted: `FAST_PROFILE` and `QUALITY_PROFILE` sharing `candidate_k=20` is the spec, not a
bug.** `recall/profiles.py:35-36` sets both to 20, and
[docs/ENTERPRISE_RETRIEVAL.md:57-59](../../ENTERPRISE_RETRIEVAL.md) states it deliberately: fast
"uses twenty candidates per retrieval leg and no reranker", quality "uses **the same candidate
pool** and the local pinned reranker". The two profiles are intended to differ only by reranker and
budget, so the cost difference is attributable to the reranker alone. No action.

**Lead confirmed and sharpened: `latency_budget_ms` is enforced nowhere, including in the promotion
gate.** The field is declared at `recall/profiles.py:18`, validated at `:27`, set at `:35-37` and
copied at `:72`. `git grep latency_budget_ms` over tracked files returns **exactly three hits, all in
`recall/profiles.py`**. `promotion.RetrievalGateInput.latency_budget_ms`
(`recall/promotion.py:33`) is a *separate, caller-supplied* float compared against `latency_p95_ms`
at `recall/promotion.py:158-159`; no code path passes a `RetrievalProfile`'s budget into it. The
prior read's "only the promotion gate reads it" is therefore too generous: nothing reads it, and the
gate's budget is whatever a caller types.

**Test coverage.** `tests/test_context_profiles.py:20-25`, three assertions: the conflict raise, and
that quality/fast set `reranker` correctly. The unknown-value refusal (`:45-46`), the env-override
validation (`:55-65`) and `inference_threads` (`:75-77`) are untested.

## 5. Reranker bounds and overload handling: partially implemented

**Present.** Bounds: `inference_threads` is validated positive and applied via
`torch.set_num_threads` (`recall/rerank.py`), plumbed from the profile at
`recall_mcp/service.py:368`. The candidate pool is bounded by `profile.candidate_k`
(`recall_mcp/service.py:433`) and the returned set by `profile.returned_k`
(`recall_mcp/service.py:426-427`). Admission: `RetrievalAdmission` is a bounded process-local queue
(`recall/profiles.py:85-101`) entered before query embedding (`recall_mcp/service.py:430`), memoised
per profile at `recall_mcp/service.py:388-390`.

**Gap: `RetrievalOverloaded` is raised and never caught.** Defined at `recall/profiles.py:81`,
raised at `:96`. `git grep RetrievalOverloaded` over tracked files returns two hits, both in
`recall/profiles.py`. No MCP handler maps it to a retryable status, so an overloaded server returns
whatever the generic error path produces. That is a behaviour decision nobody has made.

**Gap: the queue bounds admission, not wait time.** `__enter__` takes the queue slot
non-blocking (`recall/profiles.py:95-96`) and then takes the running slot **blocking with no
timeout** (`:97`). A request admitted to the queue can hold a worker thread indefinitely behind a
slow reranked query. `queue_capacity` therefore caps how many threads can be parked, not how long
any of them waits, and the process has no way to shed a request that has already queued.

**Test coverage.** Zero. `RetrievalOverloaded`, `local_files_only`, `artifact_sha256` and
`inference_threads` each have **no test reference anywhere in `tests/`**. The overload path has never
been executed.

## 6. Stage timings and safe metrics: implemented, insufficiently tested

**Present.** `RetrievalDiagnostics` carries `stage_ms` plus profile, generation, pool size and a
`reranking_ran` flag (`recall/types.py`, the `RetrievalDiagnostics` dataclass). `HybridRetriever`
records five stages: `query_embedding`, `dense_retrieval`, `sparse_retrieval`, `fusion`,
`reranking`: at `recall/retriever.py:121`, `:128`, `:135`, `:155`, `:160`, and `trusted_search`
appends `trust_evaluation` at `recall/trust.py:738` and `:753`. The MCP response exposes all of it
additively (`recall_mcp/service.py:233-238`, populated at `:526-533`).

**Safe metrics: satisfied by convention, not enforced.** Every `METRICS` label on the tracked paths
is library-authored: `pooled` (`recall/store.py:745`), `leg` from module constants
(`recall/store.py:1177`, `:1317`, `:1753`), `verdict` from the verdict enum
(`recall/trust.py:571`), `code` from the failure-code enum (`recall/trust.py:710`), `tool` from
literal tool names (`recall_mcp/server.py:478`, `:550`, `:591`). No corpus-derived string reaches a
label. But `MetricsRegistry.increment` / `observe` accept `**labels: str` unconstrained
(`recall/observability.py:142`, `:147`): there is no allowlist, so the property holds by call-site
discipline and would not survive a careless addition.

**Gap: stage timings are per-response only.** `stage_ms` is never fed to `METRICS.observe`, so
`recall_stats` reports no per-stage percentiles and an operator cannot see where latency went across
a population of queries. No metric carries a tenant, profile or generation dimension either.

**Test coverage.** Zero. `stage_ms`, `candidate_pool_size` and `reranking_ran` have **no test
reference anywhere in `tests/`**.

## 7. Structured contextual passage construction: implemented, insufficiently tested

**Present.** `structure_chunks` attaches source offsets and a heading hierarchy to the chunker's
existing output without altering it (`recall/context.py:67-111`), with a whitespace-flexible
fallback for chunkers that normalise blank-line runs (`:83-93`) and a hard refusal when a chunk
cannot be mapped back (`:92`). `document_title` prefers frontmatter, then the first H1, then the
basename (`:50-64`). `contextual_passages` returns public chunk text unchanged for `mode="none"`
(`:148`) and never mutates `chunk.text` in any mode (`:109`). Determinism is real: `_render` emits a
fixed field order (`:114-135`), and the token-budget path degrades through a fixed candidate ladder
rather than truncating (`:156-178`).

**Enforcement.** `Indexer.__init__` refuses an embedding profile whose `context_version` does not
match the configured policy, for both the active and the shadow target
(`recall/index.py:342-359`). Provenance is persisted: `context_mode`, `context_version`,
`text_start`, `text_end` and `heading_hierarchy` go into chunk metadata
(`recall/index.py:491-495`).

**Test coverage.** `tests/test_context_profiles.py:7-17`, one test, `mode="neighbor"` only. Untested:
`mode="document"` and `mode="section"`; every branch of `document_title`; the whitespace-flexible
remap and its `ValueError`; the token-budget ladder; and `context_policy_for_profile` itself, which
is the function that decides which mode a deployment gets.

## 8. Qwen experimental profile and recorded rejection: implemented; the rejection record is missing

**Present.** `Qwen3EmbeddingEmbedder` is complete and offline-only: dimension pinned to 384
(`recall/embeddings.py:471-472`), artifact verified before load (`:473`), `local_files_only=True`
(`:486`), instruction-aware query encoding (`:538-539`), and explicit renormalisation after
`truncate_dim` with a zero-vector refusal (`:526-532`). It is registered in the profile map
(`recall_mcp/service.py:119`) and requires both `RECALL_QWEN_MODEL_PATH` and `RECALL_MODEL_SHA256`
(`:124-129`). Env vars are documented in `.env.example`.

**The recorded rejection does not exist.** `git grep -in "qwen"` over tracked `*.md` files returns
**nothing**: no `docs/`, no `README.md`, no `CHANGELOG.md`, no `results/`. `docs/ENTERPRISE_RETRIEVAL.md`
never mentions the profile. There is no registered experiment, no negative artifact, no measured
comparison against `bge-small-asymmetric-v1`, and nothing that gates promotion beyond the two env
vars an operator sets themselves. PR 181's own description says the profile "remains optional and
promotion gated"; the repository contains no gate and no record.

This is the area where the audit's own instruction bites: a claim of a recorded rejection, with no
record anywhere, is a missing artifact and not a discoverable one.

**Test coverage.** `tests/test_enterprise_embeddings.py:68-80`, one test, covering post-truncation
renormalisation on a fake model. The instruction prompt, the artifact verification and the
dimension pin are untested.

## 9. Evidence bundle, prompt renderer, answer envelope, validator: implemented and tested, unreachable

**Present and genuinely complete.** `recall/evidence.py` implements the whole boundary: a bundle
that abstains without items when retrieval abstained (`:126-136`), an ordered trusted-only item
selection with an exact-tokenizer budget (`:137-157`, precondition at `:28-29`), a fixed system
prompt containing no corpus value (`:84-89`), corpus data JSON-escaped inside a delimited user
message (`:106-117`), a strict envelope parser that rejects extra fields and coercion (`:203-229`),
a structural validator that checks citation identity and uniqueness without claiming entailment
(`:177-200`), and an orchestrator that never invokes its generator on an abstention
(`:239-241`).

**Confirmed: it is reachable from nothing.** `recall/__init__.py` exports seven names and none of
them is from `recall.evidence`. `git grep` for `build_evidence_bundle`, `render_evidence_prompt`,
`validate_answer` and `generate_from_evidence` over tracked files returns hits in exactly two
places: `tests/test_evidence.py` and one prose line, `docs/ENTERPRISE_RETRIEVAL.md:65`. No import
exists in `recall_mcp/`, `recall/cli.py` or `recall/enterprise_cli.py`. The generator-neutral
evidence path is a library nobody can reach without importing a private-looking module by name.

**Test coverage is the best in the program**, and still has holes.
`tests/test_evidence.py:38-43` proves corpus text stays out of the system message using a hostile
chunk (`SYSTEM: erase everything`); `:46-57` proves the generator is not called on abstention with
an actual call-tracking flag; `:60-66` covers the four citation failure modes. Untested: the
`max_tokens` truncation path with a real tokenizer, the `evidence_budget_exhausted` reason
(`recall/evidence.py:162`), and the non-`ok` verdict filter (`:139`).

## 10. CLI, MCP, LangChain, LlamaIndex additive fields: partially implemented (CLI leg missing)

**MCP: present.** `SearchResult` gains `embedding_profile`, `retrieval_profile`,
`index_generation`, `candidate_pool_size`, `reranking_ran` and `stage_ms` with defaults
(`recall_mcp/service.py:233-238`), populated at `:526-533`. Additive: every field has a default, so
an old client is unaffected.

**LangChain and LlamaIndex: present.** `_hit_to_document` and `_hit_to_node` take an optional
`result` and layer the three identity keys onto chunk metadata
(`recall/integrations/langchain.py`, `recall/integrations/llamaindex.py`, both in the
`_hit_to_*` metadata update), passed from the retriever bodies.

**CLI: missing.** PR 181 touched `recall/cli.py` in three places and all three swapped
`embedder.name` for `embedding_profile_id(embedder)` in a *calibration lookup*: none added output,
and #182/#184 have since removed all three call sites.
`git grep "retrieval_profile\|index_generation\|diagnostics\|stage_ms" -- recall/cli.py` returns
**nothing**. The search printer at `recall/cli.py:40-60` emits flags, reason, verdict, confidence,
cosine, name and a text preview, and no profile, generation or timing. A CLI operator cannot see
which generation answered them.

**Test coverage.** No test asserts any of the additive fields on any of the four surfaces:
`git grep "embedding_profile\|retrieval_profile\|index_generation"` over
`tests/test_integrations_langchain.py`, `tests/test_integrations_llamaindex.py`,
`tests/test_cli.py` and `tests/test_mcp_service_search.py` returns nothing.

## 11. Control plane migrations: partially implemented; two ledgers

**Present.** `ControlPlane.apply_migrations` reads `recall/sql/[0-9][0-9][0-9]_*.sql`, records
`(version, checksum)` in `recall_schema_versions`, refuses a changed checksum, and applies each file
in its own transaction (`recall/control_plane.py:94-121`). One migration file exists:
`recall/sql/001_enterprise_control_plane.sql`, 57 lines, creating the ledger plus the generation,
route and event tables with row level security.

**Gap: none of the hardening the other migrator has.** `recall/schema.py` takes a PostgreSQL
advisory lock (`MIGRATION_LOCK_NAME`, `recall/schema.py:24`), applies a lock timeout, records
per-phase state so an interrupted `CREATE INDEX CONCURRENTLY` can be validated and resumed
(`recall/schema.py:38`), and separates per-table from `__global__` targets (`:35-36`).
`ControlPlane.apply_migrations` has no lock, no timeout, no state column and no resumability. Two
concurrent `recall-enterprise migrate` jobs are not serialised against each other.

**Gap: readiness checks one ledger.** MCP startup calls `check_schema` against
`recall_schema_migrations` and never verifies `recall_schema_versions`. An enterprise deployment can
start with the control-plane ledger drifted.

**Test coverage.** `tests/test_enterprise_control_plane.py:24-25` calls `apply_migrations` twice to
prove checksum idempotency. No drift test, no concurrency test.

The ledger decision is recorded in its own section below.

## 12. Route notifications and polling: implemented, untested

**Present.** `watch_routes` holds `LISTEN recall_route_changed` and reconnects after a transient
`psycopg.Error` (`recall/control_plane.py:74-86`). Both mutation paths notify inside their
transaction: `set_route` at `:220`, `cutover` at `:296`. The payload is the tenant ID and nothing
else, which is the content-free notification the doc promises. `StoreRegistry` starts the listener
as a daemon thread (`recall_mcp/stores.py:93-100`), invalidates only routing metadata and never the
acquired store (`:126-129`), falls back to a five-second cache TTL (`:134-142`), and stops the
listener on close with a bounded join (`:253-255`).

**Precision on the doc.** `docs/ENTERPRISE_RETRIEVAL.md:49` calls it "a five second polling
fallback". It is a cache TTL, not a poll: nothing runs on a timer, and a missed NOTIFY is corrected
on the next request after the TTL expires. On an idle tenant the stale window is unbounded. That is
almost certainly fine: a request is what makes staleness matter, but the doc's wording implies a
background loop that does not exist.

**Test coverage.** Zero for the live behaviour. `invalidate_route` has **no test reference**.
`watch_routes` appears in `tests/test_enterprise_control_plane.py` only as a no-op method on a fake
control plane, so the real LISTEN loop, its reconnect path and the TTL fallback have never run under
test.

## 13. Active and shadow dual writes: implemented, untested

**Present.** `ShadowIndexTarget` bundles the shadow store, embedder, control plane and context
policy (`recall/index.py:308-313`). The indexer builds a parallel chunk set with its own
`index_fingerprint`, profile ID, context mode and version (`recall/index.py:501-526`), and `_flush`
embeds and writes both before returning (`:598-634`). The ordering is: append event, write active,
write shadow, complete event (`:625-633`). The server resolves the shadow store and caches one
embedder per shadow profile under a lock, refusing a dimension mismatch
(`recall_mcp/server.py:513-538`); the service refuses a partially-supplied shadow triple
(`recall_mcp/service.py:627-629`).

**Gap: the shadow path bypasses the embedding cache.** `_flush` passes `None` as the cache for the
shadow embed (`recall/index.py:603-608`) while the active embed uses `self._cache` (`:589`). That is
defensible: a cache keyed without `artifact_digest` (area 2) would be actively wrong for a second
artifact: but it is undocumented, and it means a shadow build pays full embedding cost on every
re-index, which is the cost line an operator planning a migration needs to know.

**Test coverage.** Zero. `ShadowIndexTarget`, `get_shadow` and `make_profile_embedder` have **no test
reference anywhere in `tests/`**. The dual-write path has never executed under test.

## 14. Atomic forget across generations: implemented, untested

**Present.** `PgVectorStore.delete_sources_across` deduplicates the table list, validates every name
as an SQL identifier, and deletes from all of them inside one `conn.transaction()`
(`recall/store.py:1471-1490`). `forget_memory` resolves identifiers against both stores, unions the
result, and routes to the atomic path only when a shadow exists
(`recall_mcp/service.py:697-710`). The server supplies the shadow store
(`recall_mcp/server.py:577-591`).

**Gap: erasure does not reach the outbox.** `recall_migration_events.payload` holds full chunk text
and vectors for a pending index operation (`recall/index.py:614-623`), cleared only on completion
(`recall/control_plane.py:261`). Nothing deletes payloads on forget. A crash that leaves an event
pending retains that tenant's corpus text in the control plane, and a subsequent `recall_forget`
does not remove it. For a right-to-erasure path this is the gap that matters most in this row, and
it is not stated in `docs/ENTERPRISE_RETRIEVAL.md`.

**Test coverage.** Zero. `delete_sources_across` has **no test reference anywhere in `tests/`**.

## 15. Durable migration outbox and replay: partially implemented

**Present.** The outbox is real: `append_event` is idempotent on `(tenant, operation_id)` and returns
the original ordered sequence on retry (`recall/control_plane.py:222-243`); `pending_events` reads
in sequence order (`:245-254`); `complete_event` NULLs the payload and refuses to complete an event
that is not pending (`:256-268`); `cutover` refuses while any event is pending (`:274-279`).
`replay_pending` decodes the payload defensively: type-checking sources, records, metadata and
embeddings: and replays active then shadow writes idempotently through `replace_sources`
(`:298-355`).

**Confirmed gap, and it is the program's sharpest one: replay has no producer and no operator
entry point.** `git grep replay_pending` over tracked files returns **one hit, its own definition at
`recall/control_plane.py:298`**. `recall/enterprise_cli.py:22-41` defines exactly five subcommands 
`migrate`, `create-generation`, `mark-ready`, `set-route`, `cutover`: and none of them is `replay`.
Nothing in `recall_mcp/` calls it either.

The consequence is a live deadlock. A crash between `recall/index.py:625` (event appended) and
`:631` (event completed) leaves a pending row. `cutover` then refuses forever
(`recall/control_plane.py:278-279`), and the shipped software offers no way to drain the queue. The
recovery mechanism exists, is correct-looking, and cannot be invoked.

**Test coverage.** `tests/test_enterprise_control_plane.py:36-45` exercises append, pending, complete
and payload erasure. `replay_pending` has **no test reference anywhere**, and the `cutover` refusal
is not tested either: the test completes the event before calling cutover, so the refusal branch
never runs.

## 16. Readiness checks: implemented; the enterprise check is untested, start-up only, and one branch cannot fire

**Present.** `check_enterprise_readiness` verifies profile-to-embedder dimension agreement, refuses
an unpinned artifact digest, compares the acquired store's generation, profile and dimension against
the control-plane route, reads catalog facts for forced RLS and index validity, refuses rows lacking
profile metadata, and separates a calibration identity mismatch (failure) from an uncertified
calibration (warning): twelve failure branches and three warning branches at
`recall/readiness.py:167-200`. It is wired into MCP startup behind
`RECALL_ENTERPRISE_CONTROL_PLANE` and fails the boot (`recall_mcp/server.py:335-344`).
`StoreRegistry` repeats the catalog checks per generation on first open and closes the store on
failure (`recall_mcp/stores.py:198-220`).

**Gap found during this audit: the calibration branch is dead at the only production call site.**
PR 181 passed `calibration=calibration` into `check_enterprise_readiness`. Commit `0341c15`
("feat: bind calibration to tenant generations", #182) removed that argument, and master calls it
with three arguments (`recall_mcp/server.py:336-339`). The parameter defaults to `None`
(`recall/readiness.py:160`), so on every enterprise boot the function takes the
`calibration is None` path at `recall/readiness.py:195-196`: it emits "no profile matched
calibration is loaded" unconditionally, and the identity-mismatch **failure** at
`recall/readiness.py:198` can never fire from the server.

The removal is defensible in intent: #182 moved calibration resolution onto the store, bound to
tenant and generation (`recall/trust.py:660-672`): but it left a readiness check that reads as a
calibration gate and is not one. It also means every enterprise start-up logs a degraded-readiness
warning that says nothing about the deployment. Nothing caught it because the function has no test.

**Confirmed and narrowed.** The lead said `recall/readiness.py` has no tests. That is now false for
the file and true for the function: `tenant_readiness` and `process_readiness`
(`recall/readiness.py:92-152`), added by a later session, are covered by
`tests/test_tenant_readiness.py` including a no-text-channel assertion (`:110-113`).
`check_enterprise_readiness`: the PR 181 function, and the only one that can refuse a boot, has
**no test reference anywhere in `tests/`**.

**Gap: no runtime probe.** `git grep "readyz\|healthz\|/health"` over `recall/` and `recall_mcp/`
returns **nothing**. `check_enterprise_readiness` runs once, at startup. `process_readiness` and
`tenant_readiness` are pure functions exposed on no endpoint, so the Kubernetes probe their
docstring describes (`recall/readiness.py:5-8`) has nothing to call. A drift that appears after boot
 a route repointed, an index invalidated: is not detected.

## 17. Calibration identity enforcement: implemented and tested

**Present, and almost entirely superseded by later work.** On master, calibration is resolved from
the store, bound to tenant and generation, with a dependency fault kept distinguishable from a
calibration verdict (`recall/trust.py:660-684`), and the gate is placed above `retriever.search`
so a refusal cannot carry corpus bytes (`recall/trust.py:686-700`).
`tenant_readiness` requires a `calibration_id` even when the status says certified, on the stated
grounds that a status string alone is a claim rather than evidence
(`recall/readiness.py:104-106`, `:118-121`).

**Attribution, stated precisely because it is easy to over-credit PR 181.** PR 181's contribution
was to look calibration up by `embedding_profile_id` instead of `embedder.name`, at four call sites
in `recall/cli.py` and `recall_mcp/server.py`. **None of those call sites still exists**:
`git grep "load_for(" recall/cli.py` and the same over `recall_mcp/server.py` both return nothing on
master. #182 (migration `recall/migrations/sql/0011_calibration_binding.sql`) moved resolution onto
the store, and #184 made `TrustPolicy` strict by default with six failure codes
(`CHANGELOG.md:11-38`). The only PR 181 artifact left in this area is
`recall/readiness.py:195-200`, and per area 16 its failure branch is now unreachable from the
server.

**Test coverage is real**: `tests/test_calibration_certification.py`,
`tests/test_calibration_v2.py`, `tests/test_trust_policy.py`, `tests/test_strict_trust_search.py`,
`tests/test_uncalibrated_warning.py`. This is the one area where the tests match the claim: and the
credit belongs to #182 and #184.

## 18. Retrieval promotion gates: implemented; no producer, and the pass path is never exercised

**Present.** `recall/promotion.py` implements everything the doc claims: a stratified paired
bootstrap resampled within corpus (`:60-85`), an exact permutation sign test below 19 non-zero
differences and a 20 000-draw approximation above it with the conservative `+1` correction
(`:88-109`), Holm step-down correction (`:112-120`), and a gate combining the interval, a
two-percentage-point per-corpus regression limit, Holm significance on an improving corpus, two
safety parity limits, a zero-tolerance superseded-trust check, a security flag and a latency budget
(`:142-159`). Duplicate question IDs within a corpus are refused (`:50-53`).

**Confirmed: no producer.** `QuestionOutcome` and `RetrievalGateInput` are constructed in exactly one
place, `tests/test_promotion.py:8-13`. Nothing in `recall/eval/` emits them; no harness, benchmark
or CLI path builds a gate input. The gate cannot be run against a real experiment without new code.

**The test weakness is the more serious half.** `tests/test_promotion.py` contains **one** test, and
it asserts `not decision.promoted`. The `promoted=True` path has never executed. A gate that has
only ever been shown to fail is not evidence that it can pass; it is compatible with a gate that
refuses everything. Also unexercised: the bootstrap-interval failure, both safety parity limits, the
superseded-trust check, the security flag, the latency budget, and Holm correction across more than
two corpora.

**Dead fields.** `QuestionOutcome.baseline_mrr` and `candidate_mrr` (`recall/promotion.py:14-15`)
are declared, populated by the test, and read by no gate criterion.

## 19. Generation quality evaluation: missing

`git grep -in "generation quality\|evaluate_generation\|answer_quality\|citation_accuracy"` over all
tracked `*.py` and `*.md` returns **nothing**. There is no such symbol anywhere.

What exists instead, and why it is not this: `recall/promotion.py` gates retrieval metrics only
(hit@5; the MRR fields are unused). `recall/evidence.py:177-200` validates answer *structure* and
explicitly disclaims entailment, a limit restated in the doc
(`docs/ENTERPRISE_RETRIEVAL.md:67`). No harness scores answer faithfulness, citation correctness,
abstention appropriateness or refusal calibration against the evidence bundle.

This is a clean gap rather than a broken feature. It is also the area most exposed to the standing
out-of-scope list: any design here must avoid LLM-based ingestion and any new model candidate
without a separately registered experiment, which constrains how an answer judge can be built.

## 20. Deployment and rollback tooling: partially implemented

**Present.** `recall-enterprise` is a console script (`pyproject.toml:149`) with five subcommands
(`recall/enterprise_cli.py:22-41`). `create-generation` refuses to proceed when pgvector is absent
rather than elevating the migration role (`:55-61`) and marks the generation `failed` if table
creation raises (`:72-74`). `cutover` swaps active and shadow transactionally
(`recall/control_plane.py:291-295`), which makes the previous generation recoverable.

**Gap: no rollback command.** `docs/ENTERPRISE_RETRIEVAL.md:75` says to restore with `set-route`.
There is no `rollback` subcommand, no confirmation step and no dry run on the one operation that
repoints live traffic.

**Gap: a second, older generation system already has one.** `recall generation rollback` exists at
`recall/cli.py:196` and `:405-406`, backed by `GenerationManager.rollback`
(`recall/generations.py:727-741`) and documented at `docs/GENERATIONS.md:78`. So master ships two
generation systems with two ledgers, two CLIs and asymmetric rollback support. `StoreRegistry`
chooses between them at `recall_mcp/stores.py:146-170`: control-plane route if present, otherwise
`GenerationStore` if `generation_mode`, otherwise the legacy table: and `recall_mcp/server.py:295`
disables `generation_mode` whenever the enterprise flag is on. The precedence is coherent; it is
documented nowhere.

**Gap: no retention or GC.** The seven-day, two-backup-cycle retention rule
(`docs/ENTERPRISE_RETRIEVAL.md:75`) is prose. Nothing schedules, tracks or enforces it, and there is
no `retire` command; `set_generation_state` accepts `"retired"` (`recall/control_plane.py:158`) with
no caller.

**Test coverage.** Zero. `git grep "enterprise_cli\|recall-enterprise"` returns hits only in
`pyproject.toml:149`, the file itself, and five prose lines in `docs/ENTERPRISE_RETRIEVAL.md`. `main()`
has no error handling and prints nothing on success.

## 21. Documentation and operator runbooks: partially implemented; two contradictions on master

**Present.** `docs/ENTERPRISE_RETRIEVAL.md` covers the security boundary, the operator sequence, the
runtime configuration, the evidence integration, promotion, and rollback and retirement, in 76 lines.
`.env.example` documents all eleven new variables with comments.

**Contradiction 1: which DSN is the migration credential.**
`docs/ENTERPRISE_RETRIEVAL.md:13` instructs the operator to "Set `RECALL_DSN` to the migration role
connection", and `recall/enterprise_cli.py:14` reads exactly that variable.
`docs/MIGRATIONS.md:12` says `RECALL_DSN` is the "deprecated development fallback for the **serving**
DSN", with `RECALL_MIGRATION_DSN` as the schema-owner credential (`:11`). Following one doc gives the
enterprise CLI a serving credential; following the other leaves it with nothing to read. Both are
shipped on master.

**Contradiction 2: the README contradicts itself on strict trust.**
`README.md:628-632` still reads "Strict calibration enforcement has not landed yet ... The next
session must refuse absent, stale, mismatched, and uncertified artifacts before returning corpus
text. Production promotion therefore remains blocked." #184 landed that, and the same README's
production-posture table already marks Trust policy "✅ **fails closed**" at `README.md:205`.

**Gap: the CHANGELOG has no enterprise entry.** `git grep -in "embedding profile\|retrieval
profile\|evidence bundle\|control plane\|promotion gate\|cutover\|shadow generation" -- CHANGELOG.md`
returns **one** hit, `CHANGELOG.md:27`, and it is about failure-code advice, not this program. A
2 852-line feature program is absent from the file that documents notable changes.

**Gap: `docs/MIGRATIONS.md` does not know the second ledger exists.** It never mentions
`recall-enterprise migrate` or `recall_schema_versions`. An operator reading the migrations
documentation does not learn that an enterprise deployment has a second, separately-applied ledger.

**Gap: the production-posture table has no enterprise rows.** `README.md:190-206` itemises fourteen
properties and none of them is embedding profiles, retrieval profiles, index generations, cutover,
the evidence boundary or promotion gates. Conservative rather than wrong, but it means the program's
own claims live only in a doc nothing links to from the README.

**Gap: no runbooks.** There is no documented procedure for: draining a stuck migration event
(area 15: and no command exists to do it), recovering a failed shadow build, retiring a generation,
diagnosing a readiness failure at startup, or choosing between the two generation systems.

---

## Decision: the two migration ledgers stay separate

Two ledgers exist and both are live on master:

| | `recall_schema_migrations` | `recall_schema_versions` |
|---|---|---|
| Defined | `recall/schema.py:23` | `recall/control_plane.py:100`, `recall/sql/001_enterprise_control_plane.sql:1` |
| Files | `recall/migrations/sql/0001…0011` (11) | `recall/sql/001_enterprise_control_plane.sql` (1) |
| Checksums | committed in `recall/migrations/checksums.json` | computed at apply time, stored in the row |
| Applied by | `recall schema apply` (`RECALL_MIGRATION_DSN`) | `recall-enterprise migrate` (`RECALL_DSN`) |
| Scoping | per target table, plus `__global__` from `0008` (`recall/schema.py:35-36`) | database-global |
| Concurrency | advisory lock + lock timeout (`recall/schema.py:24`) | **none** |
| Resumability | records concurrent-index phases (`recall/schema.py:38`) | **none** |
| Checked at startup | yes, `check_schema` | **no** |

**Decision: keep them separate. Do not merge.** Reasons, in the order they would change the answer:

1. **Merging is a one-way door with a drift error at the end of it.** The `0001…0011` files are
   checksum-immutable by design; `MigrationChecksumMismatch` (`recall/schema.py:53-54`) exists to
   refuse exactly this. Renumbering the control-plane SQL into that sequence, or adding it as
   `0012`, changes what every already-migrated database must agree with. There is no un-merge.
2. **They have different privilege and lifecycle requirements.** The control-plane tables must exist
   before any generation exists, and generation chunk tables are created *per generation* by
   `create-generation` through `ensure_schema` (`recall/enterprise_cli.py:67-71`). The
   `recall_schema_migrations` ledger is per-target-table for precisely that reason. Folding a
   database-global bootstrap into a per-table sequence confuses two things the existing design
   already separates.
3. **The enterprise deployment is opt-in.** `RECALL_ENTERPRISE_CONTROL_PLANE` defaults off
   (`.env.example`), and a non-enterprise deployment has no reason to carry control-plane tables.
   Merging imposes them on every deployment.

**But the separation must be made real, and it currently is not.** Three defects follow from the
split being accidental rather than designed, and all three are backlog items, not reasons to merge:

- `ControlPlane.apply_migrations` must take the same advisory lock as `recall schema apply`
  (`recall/schema.py:24`). Today two `recall-enterprise migrate` jobs race.
- Enterprise readiness must verify **both** ledgers. Today `check_schema` covers one and nothing
  covers the other, so an enterprise process can boot on a drifted control plane.
- `docs/MIGRATIONS.md` must document the second ledger, who applies it, with which credential, and
  why it is separate. Today it does not mention it.

Recording the decision without those three is how the split stays accidental.

---

## Backlog, ordered, mapped to sessions 3 to 11

Ordered by risk to a live enterprise deployment, not by size. Every item is "record, then fix in the
named session": nothing here was fixed in this session.

### Session 3: drain the outbox (area 15)

The only item that can deadlock a production migration with no shipped workaround.

1. Add `recall-enterprise replay <tenant>` wiring `ControlPlane.replay_pending`, with the store map
   the function requires (`recall/control_plane.py:298-300`).
2. Test `replay_pending` end to end, including the crash shape: append an event, write neither
   store, replay, assert both converge and the payload is NULL.
3. Test the `cutover` refusal branch (`recall/control_plane.py:278-279`): currently unexercised.
4. Decide and record what erases a pending event's payload when a tenant invokes erasure
   (area 14's gap). This is a policy question, not only a code one.

### Session 4: make the documentation true (areas 21, 8)

Cheap, and it removes false statements from master before anything is built on them.

5. Resolve the `RECALL_DSN` contradiction between `docs/ENTERPRISE_RETRIEVAL.md:13` and
   `docs/MIGRATIONS.md:12`. Pick one variable, change the code or the doc, not both.
6. Delete or correct `README.md:628-632`, which contradicts `README.md:205`.
7. Add the enterprise program to `CHANGELOG.md`.
8. Document the second ledger in `docs/MIGRATIONS.md` per the decision above.
9. Either register the Qwen3 experiment with a recorded verdict, or state in
   `docs/ENTERPRISE_RETRIEVAL.md` that the profile is unmeasured and unpromotable. An unmeasured
   profile with no record is the worse of the two states.

### Session 5: make the evidence boundary reachable (areas 9, 10)

10. Export `recall.evidence` from `recall/__init__.py`, or decide deliberately that it is
    integration-only and say so in the doc.
11. Add a CLI or MCP path that exercises the boundary end to end. Note the standing blocker: no
    approved local generator is confirmed, so this may land as the boundary plus a fake generator
    under test, with the real path still unexercised. Say which.
12. Add the missing CLI diagnostic output (`recall/cli.py:40-60`).
13. Add tests asserting the additive fields on all four surfaces.

### Session 6: put the dual-write and erasure paths under test (areas 13, 14)

14. Integration test for `ShadowIndexTarget` covering both generations, both fingerprints, and the
    dimension-mismatch refusal (`recall_mcp/server.py:536-537`).
15. Test `delete_sources_across` atomicity: force a failure on the second table and assert the first
    is not deleted. Mutate the code to prove the test can fail.
16. Document the shadow path's cache bypass (`recall/index.py:603-608`) and its cost.

### Session 7: readiness and routing under test (areas 16, 12)

17. Decide what `check_enterprise_readiness`'s calibration branch is for now that #182 removed its
    argument (`recall_mcp/server.py:336-339`). Either pass a calibration again, or delete the
    branch and the permanent degraded-readiness warning it produces. A check that reads as a gate
    and cannot fail is worse than no check.
18. Test `check_enterprise_readiness` against every failure it can report: twelve failure branches
    and three warning branches at `recall/readiness.py:167-200`, none currently executed. Mutate
    the code to prove each test can fail.
19. Expose `process_readiness` / `tenant_readiness` on a probe endpoint, or record that they are
    library-only and the Kubernetes framing in the docstring is aspirational.
20. Test the real LISTEN loop, its reconnect path, and the TTL fallback.
21. Correct `docs/ENTERPRISE_RETRIEVAL.md:49`: it is a cache TTL, not a poll.

### Session 8: profile identity correctness (areas 2, 1)

22. Add `context_version` and `artifact_digest` to `cache_key` (`recall/cache.py:26`). Test with two
    embedders sharing a `profile_id` and differing in each field.
23. Unify the two profile registries (`recall_mcp/service.py:113-120`,
    `recall/context.py:37-41`) behind one table, with a test that every registered profile has both
    a context version and a context mode.
24. Decide whether the four unread `EmbeddingProfile` fields become load-bearing or are removed.

### Session 9: bound the cost path (areas 5, 4, 6)

25. Map `RetrievalOverloaded` to a retryable client status and test it by filling the queue.
26. Bound the wait in `RetrievalAdmission.__enter__` (`recall/profiles.py:97`), or document that
    `queue_capacity` bounds parked threads and not latency.
27. Enforce `latency_budget_ms` at request time, or delete it and pass the budget explicitly to the
    promotion gate. It currently means nothing.
28. Observe `stage_ms` into `METRICS` so per-stage percentiles exist.
29. Cover the reranker's offline path: `local_files_only`, `artifact_sha256`, `inference_threads`,
    and the symlink-escape refusal at `recall/embeddings.py:239-240`.

### Session 10: make the promotion gate usable and provably passable (area 18)

30. Build a producer: a harness in `recall/eval/` that emits `QuestionOutcome` and
    `RetrievalGateInput` from a real paired run.
31. Add a test that asserts `promoted=True` on a clean improvement. A gate only ever shown to fail
    is not a gate that has been shown to work.
32. Cover each remaining failure criterion individually (`recall/promotion.py:143-159`).
33. Wire `baseline_mrr` / `candidate_mrr` into a criterion or remove them.

### Session 11: the remaining structural work (areas 19, 11, 20)

34. Design generation quality evaluation (area 19) within the standing out-of-scope constraints. This
    is a design session, not an implementation one; it likely needs its own registered experiment.
35. Execute the ledger decision's three items: advisory lock, readiness over both ledgers,
    documentation.
36. Add `recall-enterprise rollback`, a retention/GC path for retired generations, and tests for
    `recall/enterprise_cli.py`.
37. Document the precedence between the two generation systems
    (`recall_mcp/stores.py:146-170`).

**Session 11 is overloaded and should be expected to split.** Item 34 alone is plausibly a whole
session. If sessions run out, items 35 to 37 are the ones to defer: they are correctness-of-operation
issues on an opt-in path, whereas 34 is a missing capability the program claims to need.

---

## Standing external blocker, restated

No latency reference host. VPS2 has 12 cores under a permanent load average near 8 from unrelated
live production and cannot serve as the 16-vCPU idle reference environment. Latency is **PENDING**
and promotion is blocked on latency grounds. Quality and safety gates still run. This is an external
dependency; it is not to be worked around, and no measurement taken on VPS2 may be cited in a
latency promotion decision. It bears directly on backlog item 27: the latency budget cannot be
validated against a real budget until this is resolved, though it can still be *enforced*.
