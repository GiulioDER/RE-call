# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is pre-1.0 `0.MINOR.PATCH`, so
a minor bump may still break schema or API. Dates are commit dates from `git log`, not release-tag
dates — this project does not currently tag releases.

## [0.5.0] — 2026-07-22

### Added
- Real-corpus evaluation: indexed and scored against 792 hand-written memos (6,469 chunks) with 110
  hand-labelled questions, replacing headings-as-queries as the retrieval-quality proxy.
- A rerank arm (`hybrid+rerank`) added to the ablation matrix specifically to test the cross-encoder
  reranker against the real corpus.

### Changed
- **Chunks table gains a `tenant_id` column; the primary key becomes `(tenant_id, id)`.**
  `ensure_schema()` migrates an existing table in place and assigns existing rows to the `default`
  tenant, so a single-tenant deployment upgrades without noticing.
- Abstention threshold is fitted differently (mid-gap rather than the lowest answerable sample) —
  abstains more, and more accurately: false-confidence on unanswerable queries drops from 0.205 to
  0.000 for 1.5% of answerable queries.
- `supersedes:` matching is more tolerant — `name`, `name.md`, `[name]`, `[[name]]` now all resolve
  to the same document.
- README rewritten for a technical reader, structured around what was actually measured.

### Fixed / measured (negative results published, not hidden)
- **Published the real retrieval number: hit@5 0.33 [0.21, 0.47], n=46, on 110 labelled questions** —
  the previous headings-as-queries proxy scored 0.945 and hid two-thirds of the failures.
- **The rerank arm — the predicted lever — was tested and largely falsified**: the cross-encoder
  moves hit@5 to 0.39 [0.26, 0.54] for 57× the latency, within noise of no rerank at all; the
  bottleneck is candidate recall, not ranking.

### Withdrawn
- **"FCR @calibrated 0.00"** — the threshold had been fitted and scored on the same samples; now
  cross-validated, and the fitting rule itself was replaced after it was shown to let 20.5% of
  unanswerable queries through.
- **Coverage/abstention accuracy on generated corpora** — the "unanswerable" queries were an
  answerable query plus a nonsense suffix, so nothing could separate them; rebuilt as genuinely
  off-topic questions.
- **"6× faster incremental re-index"** — understated; measured on a Linux server it is 33×.
- **Real-corpus recall@5 of 0.945** — was known-item retrieval (document headings as queries); see
  the hit@5 0.33 entry above for the honest number.

## [0.4.0] — 2026-07-21

### Added
- CCA (Comprehensive Code Audit) DEEP-tier hardening pass on top of the audit PRs to date — six
  proved defect classes fixed with regression tests quoting the input that caused each one.
- `python -m recall.eval.scale`: trust evaluation at scale on a generated corpus — Wilson intervals
  instead of point estimates, and `source`-filtered HNSW recall under index pressure (measured at
  50,600 chunks).
- Async MCP tools + optional `psycopg_pool` connection pool — the server previously served exactly
  one request at a time.
- Reconnect-and-retry with narrow `statement_timeout`/`connect_timeout` handling.
- Structured logging (text/JSON) and metrics (counters + latency percentiles) surfaced through the
  MCP `recall_stats` tool.
- Multi-tenancy: `tenant_id` scaffolding and row-level-security groundwork (landed fully in 0.5.0's
  schema migration).
- Incremental, bounded-memory indexing that prunes files deleted from disk (content-hash skip).
- `pytest-timeout` so a hanging chunker fails the run instead of hanging CI silently.

### Fixed
- Published rates re-measured **out-of-sample** rather than in-sample (#7).
- Reconnect test asserts the actual REPLAY behaviour, not a hard-coded statement count.
- Supersession map no longer goes stale across processes.
- Failed open on default credentials — closed: refuses to start against the published
  `recall:recall` credentials pointed at a non-local host (`RECALL_ALLOW_INSECURE_DSN` opt-out).

## [0.3.1] — 2026-07-18

### Added
- `recall lint --semantic`: retrieval-based check for a missing supersession edge — surfaces a memo
  whose prose describes a closure it never declared via `supersedes:`.

## [0.3.0] — 2026-07-18

### Added
- Entailment-based near-miss abstention (`recall.entailment`): a QNLI judge stacked on top of the
  calibrated cosine threshold, isolating near-miss queries (a high-similarity memory that doesn't
  actually answer the query) from the classic far-gap case a threshold already catches.
- `recall lint`: write-time completeness checks on the supersession graph, plus `--fix` to propose
  (not apply) an edge a memo's prose already states.
- `recall check`: a write-time gate for a pre-commit hook — ask for the edge while the author still
  knows it.
- Recency-steelman evaluation: "trust the newest relevant hit" tested directly against the
  declared-supersession approach, and still trusts a stale memory 83–100% of the time.

### Measured
- Entailment stage cuts near-miss false-confidence 1.00 → 0.60 and 0.80 → 0.50 — but the judge alone
  *degrades* far-gap detection; the threshold and the judge stack, neither replaces the other.

## [0.2.0] — 2026-07-17

### Added
- **The trust layer**: verdicts (`ok` / `superseded` / `expired` / `not_yet_valid` /
  `low_confidence` / …), calibrated confidence, provenance (`indexed_at`), and successor redirect
  when a stale hit was confidently retrieved.
- Runtime calibration: a persistable, per-embedder confidence threshold (`recall calibrate`).
- Validity frontmatter (`valid_from`, `valid_until`, `supersedes`) parsed from the memory itself into
  chunk metadata — authored, not inferred.
- Superseded-trust-rate evaluation comparing plain search against the trust layer.
- 31 CCA (DEEP-tier) audit fixes applied as a pre-push gate before this release.

### Measured
- Superseded-trust rate **0.00** [0.00, 0.02] (n=250) against a plain-search baseline of **1.00** —
  the foundational claim of the whole project: supersession beats similarity.

## [0.1.0] — 2026-07-06

### Added
- Initial `recall` package: `Embedder` protocol with `HashingEmbedder` (offline, deterministic) and
  `FastEmbedEmbedder` (local, no API key).
- `PgVectorStore`: dense + sparse (full-text) query against Postgres/pgvector, with freshness
  metadata.
- `Indexer`: paragraph chunking and recursive folder ingest.
- `HybridRetriever`: Reciprocal Rank Fusion of dense and sparse candidates, with gap and staleness
  honesty guards.
- CLI (`recall index` / `search` / `demo`) and a synthetic agent-memory corpus for offline testing.
- `recall_mcp`: FastMCP server exposing `recall_search`, `recall_index`, `recall_stats` as MCP tools,
  plus an example self-recall agent.
- Evaluation harness: ablation runner (`make eval`) scoring dense/hybrid/hybrid+rerank fusion,
  retrieval metrics, and a gap-threshold calibration study with an honest negative result (a fixed
  threshold does not transfer across embedders).
- Domain fine-tuning pipeline with an honest null result on a corpus the base embedder already
  saturates (later promoted to a first-class, better-targeted result: +0.00 on a rich corpus vs.
  0.31 → 0.55 held-out MRR on opaque jargon).
- CI: GitHub Actions running `ruff` + `pytest` against a real pgvector service container.
- MIT license.
