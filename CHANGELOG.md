# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is pre-1.0 `0.MINOR.PATCH`, so
a minor bump may still break schema or API. Dates are commit dates from `git log`, not release-tag
dates — this project does not currently tag releases.

## [Unreleased]

## [0.5.3] — 2026-07-24

### Added
- **A LangChain retriever** (`recall/integrations/langchain.py`, extra `langchain`) — `RecallRetriever`
  is a drop-in `langchain_core` `BaseRetriever`, so RE-call can sit behind any chain, agent or
  `create_retrieval_chain` pipeline. It differs from an ordinary vector retriever in one way, which
  is the point: **when the trust layer abstains it returns no documents**, not a best-effort
  neighbour — a plain similarity retriever always hands back its top-k, so a chain cites the closest
  vector even when that memory is stale or superseded, and the stale hit is often the
  *highest*-cosine one. Each `Document` carries the trust signal in `metadata` (`recall_verdict`,
  `recall_confidence`, `recall_cosine`, `superseded_by`). Install with
  `pip install "recall-rag[langchain]"`.
- **A LlamaIndex retriever** (`recall/integrations/llamaindex.py`, extra `llamaindex`) — the same
  adapter against `llama_index.core`, for any LlamaIndex query engine, chat engine or agent. An
  abstention becomes an empty `list[NodeWithScore]`, so a query engine synthesises from nothing
  rather than from a stale, superseded or unentailed memory; node `score` is the cosine similarity
  and the calibrated confidence rides in `metadata['recall_confidence']`. Install with
  `pip install "recall-rag[llamaindex]"`.

  Both adapters take an injectable search function, so they are unit-tested without a database, and
  both are in `dev` as well as their own extra — the `test` and `typecheck` jobs install `.[dev]`
  only, so without that the adapters would be shipped but never CI-tested or type-checked.

### Fixed
- **The README's second upgrade section said "unreleased" for changes that had already shipped.**
  It described the five breaking changes as being "on `main` … not in 0.5.0 yet" — they went out in
  0.5.1, so a reader on the published page was told a released guard was still pending. Now headed
  *Upgrading to 0.5.1*, and it states that 0.5.2 adds only the LOCOMO benchmark and changes no
  behaviour. PyPI freezes a version's description at upload and the fix landed after 0.5.2 went
  out, so the 0.5.2 project page kept the stale wording — **this release is what carries the
  correction to PyPI readers.**
- **`CITATION.cff` sat at 0.5.1 through the whole 0.5.2 release.** The version is written in three
  places and the drift test covered only two, so the one file whose entire job is to say which
  version produced a result was the one nothing checked. Bumped, and the test now asserts all
  three agree.

### Changed
- **The README documents the two framework integrations** (*Use it with LangChain or LlamaIndex*).
  They shipped in this release with no README presence at all — the only mention of either
  ecosystem was LangMem in the prior-art table.

## [0.5.2] — 2026-07-23

### Added
- **The LOCOMO benchmark** (`recall/eval/locomo.py`, `locomo_abstention.py`,
  `locomo_entailment_sweep.py`) — the standard long-term-memory benchmark Mem0 and Zep report, run
  against the retrieval layer with **no LLM judge**: retrieval is scored by exact string-match
  against LOCOMO's gold evidence turns, not by an LLM-as-judge. It is deliberately **not** a J
  score — RE-call ships no generator, so nothing here sits beside Mem0's 66.9 or Zep's 66.0.
  - **Retrieval**: evidence-turn **hit@5 0.615** [0.59, 0.64] with the free local embedder.
  - **Abstention**: the **446 adversarial questions** (22.5% of LOCOMO) that, per an independent
    audit, no published result scores. Default abstention is **0.00**; the shipped levers
    (calibration, an entailment judge) raise it to 0.37–0.77 only by refusing 26–56% of
    *legitimate* questions. A judge sweep shows a stronger QNLI cross-encoder lifts best separation
    0.197 → 0.240 but still refuses 44% of legitimate questions — the residual is the
    entity-attribution reasoning the library omits by design.
  - Full write-up in `results/FINDINGS.md` §9; 24 unit tests over the pure logic. The benchmark
    itself is a local eval (needs pgvector + a cross-encoder download), not a CI gate.

### Fixed
- **Unresolved merge-conflict markers in the README, published to PyPI.** 0.5.1 shipped with raw
  `<<<<<<< Updated upstream` / `=======` / `>>>>>>> Stashed changes` markers around the
  standard-benchmark bullet — an artifact of a `git stash` reconcile committed unresolved, so both
  the GitHub landing page and the PyPI 0.5.1 project description rendered the markers to every
  visitor. Resolved in favour of the current claim (LOCOMO runs against the library; hit@5 0.615;
  the 446-adversarial abstention boundary), dropping the stale "this repo has never run either"
  side. PyPI freezes a version's description at upload, so this needs a release.

## [0.5.1] — 2026-07-23

### Fixed
- **The README now renders on PyPI.** The project page for 0.5.0 showed a broken banner and no
  demo image, for two independent reasons — both fixed here, and both needing a release because
  PyPI freezes a version's description at upload:
  - **Relative paths don't resolve on PyPI.** `docs/banner.svg` and every `docs/…`/`results/…`
    link were repo-relative — fine on GitHub, 404 on PyPI, which has no repo to resolve them
    against. All image `src`s and doc links are now absolute (`raw.githubusercontent.com` for
    images, `github.com/.../blob/master` for docs), so they work on both.
  - **PyPI strips SVG images entirely** (its description sanitiser drops `<svg>` and SVG `<img>`
    for security). The banner now points at the existing `docs/banner.png`, and the demo has a
    new rasterised `docs/superseded-catch.png` (rendered from the SVG at 2×, reduced-motion end
    state so every row is present). The animated SVG stays in the repo for GitHub.

### Changed
- **The distribution is now published as `recall-rag`** (the import stays `recall`). `recall` on
  PyPI belongs to an unrelated Python-2-era RPC framework whose last release was in 2014; the name
  is occupied and not reclaimable, and `re-call` is rejected by PyPI's similarity guard as too
  close to it. Install with `pip install "recall-rag[fastembed]"`. ⚠️ That other `recall` package
  also provides a top-level `recall` module, so `recall` and `recall-rag` must not share an
  environment — whichever installs last wins the import path, and nothing detects it.

### Added
- **BM25 and single-leg baselines in the labelled evaluation** (`recall/eval/bm25.py`). Every
  retrieval number this project published was previously unanchored: `hit@5 = 0.705` cannot be
  read as good or bad without knowing what plain keyword matching scores on the same corpus, the
  same chunks and the same questions. `python -m recall.eval.labelled` now reports four arms —
  `bm25`, `dense`, `sparse`, `hybrid` — instead of one. On the public PEP corpus (bge-small, 44
  held-out answerable): BM25 **0.455**, sparse-only **0.023**, dense-only **0.682**, hybrid
  **0.705** — so the pipeline beats the baseline by **+0.25**, and dense carries it (hybrid's
  +0.023 over dense-alone is within the interval on this corpus).
  - The BM25 implementation is dependency-free (Okapi, `k1=1.5`, `b=0.75`, untuned) rather than
    `rank_bm25`, so the anchor for every published number cannot change under a dependency bump.
  - `PgVectorStore.iter_chunks()` streams the tenant's chunks through a server-side cursor, which
    is how the baseline indexes *exactly* the chunks the other arms search. Deliberately outside
    `_with_retry`: a mid-scan reconnect would restart the cursor and yield rows twice.
  - `HybridRetriever(use_dense=False)` completes the ablation switch that `use_sparse=False`
    started. It is an ablation, not a serving mode — with no dense leg there are no cosines, so
    `gap_warning` reports False for every query and must not be read as "no gap".
- **A prior-art section in the README.** Zep/Graphiti, Mem0, Letta and LangMem, what each does
  about a fact that stopped being true, and the one real difference here: validity is *authored*,
  not inferred. Stated as a trade — precision on the edges that exist, paid for in coverage
  (2 of 792 memos declared `supersedes:`) — rather than as a win.
- **`mypy` as a CI gate** (`[tool.mypy]`, `typecheck` job), with `disallow_untyped_defs`. It found
  two defects the test suite did not; both are in Fixed below.
- **Coverage measurement** on the test job (`pytest-cov`, over `recall` and `recall_mcp` only).
- **A release workflow** (`.github/workflows/release.yml`) publishing on a `v*` tag via PyPI
  Trusted Publishing — no API token in repository secrets. It builds once, installs the built
  *wheel* into a clean environment on 3.11 and 3.13 and imports it there, and only then publishes;
  `uv build` succeeding proves the metadata parses, not that the wheel's contents work.

### Fixed
- **`RECALL_TRANSPORT` was never validated.** An unrecognised value (`stdo`, `http`) was passed
  straight to `mcp.run(transport=...)` at the very end of startup, after a store had been opened
  and the token file read. It is now checked at import against the three transports the SDK
  accepts, and names both the bad value and the valid set.
- **`ensure_schema()` crashed with a bare `TypeError` against a foreign table.** `CREATE TABLE IF
  NOT EXISTS` is a no-op when a table of that name already exists, so pointing a store at an
  unrelated `chunks` table reached the dimension check with no `embedding` column and indexed
  `None`. It now raises a `ValueError` naming the table.
- MCP tool annotations are constructed as `ToolAnnotations` rather than passed as a bare dict, and
  `StoreRegistry` passes `table=` explicitly instead of splatting a conditional `**kwargs` — both
  were shapes a type checker could not see through.
- **A prune guard on re-index** (`recall/index.py`). Re-indexing removes rows for files gone from
  disk; that made `recall index` quietly destructive when a corpus was *missing* rather than
  deleted — an unmounted volume, an interrupted sync, a path that still resolves. It now raises
  `PruneGuardTripped` and deletes nothing when a re-index would remove **50% or more** of the
  sources under that root (`RECALL_MAX_PRUNE_FRACTION`, default `0.5`), above a floor of 5 indexed
  sources where a fraction starts to mean anything. Confirm the files really are gone, then re-run
  with `--allow-prune` (`Indexer(allow_prune=True)`).
  - **`recall index` can now fail where it previously succeeded.** That is the point, but it is a
    behaviour change for any scripted re-index.
  - "Gone from disk" is now checked against the disk. It was inferred from absence from the
    current run's glob, so re-indexing one root with a different `--glob` deleted the other glob's
    rows — and the fraction guard missed it whenever they were a minority of the corpus.
  - **"Gone" now means ENOENT, not "could not be stat'd".** The check used `Path.exists()`, which
    swallows *every* `OSError` and answers `False` — so an unreadable parent directory, a dropped
    network mount or a symlink loop was read as a deletion and the rows were removed, under the
    fraction guard and with exit 0. It now calls `os.stat` and classifies by errno: only ENOENT
    and ENOTDIR are deletions; everything else means unreachable, and unreachable is never
    pruned. (`Path.exists()` delegates to a C accelerator that swallows the error before any
    `except OSError` in Python could observe it, which is why the guard that was there did
    nothing.)
  - **A file that vanishes mid-run no longer aborts the run**, and a corpus that vanishes
    entirely no longer reports success: individual disappearances are skipped and logged, but
    when *every* candidate is gone `index` raises `FileNotFoundError` rather than reporting
    "indexed 0 files". Read failures that are not disappearances (permissions, I/O) still abort
    immediately, as before.
  - **`Indexer.index_path` now rejects `glob=` and `files=` together** with `ValueError`, instead
    of silently ignoring the glob, and re-confines a supplied `files=` list to the root rather
    than trusting the caller to have done it.
  - `recall index` reports unchanged and pruned counts, and the MCP `IndexResult` carries
    `skipped` / `deleted`. Both were computed and then discarded, so a prune happened in silence.
- **Authentication on the MCP HTTP transports** (`recall_mcp/auth.py`, `recall_mcp/stores.py`,
  [docs/AUTH.md](docs/AUTH.md)). Static bearer tokens map to a principal with a **tenant** and
  **scopes**; the tenant selects its own `PgVectorStore` and connection pool, so a principal
  cannot reach another tenant's rows. Closes the second checkbox of issue #9.
  - **Fails closed**: starting `streamable-http` or `sse` without `RECALL_AUTH_TOKENS_FILE`
    raises `AuthConfigError` and refuses to boot, rather than warning into a journal while
    serving every memory to anything that can reach the port. `stdio` is unchanged and stays
    unauthenticated by design — it is a private pipe to one client, not a listener.
  - **Tokens come from a file, never an environment variable.** There is deliberately no
    `RECALL_AUTH_TOKENS=<secret>`: env vars leak via `/proc/<pid>/environ`, `ps e`, container
    inspection and every child process. Tokens are held only as SHA-256 digests, and
    `token_sha256` lets an operator provision access without writing plaintext to disk.
  - **Three scopes** mirroring each tool's real risk — `recall:read` (search, stats),
    `recall:write` (indexing burns embedding spend), `recall:forget` (irreversible). Entries
    default to `recall:read` alone.
  - New `RECALL_TRANSPORT`, `RECALL_HOST` (defaults to loopback, not `0.0.0.0`), `RECALL_PORT`,
    `RECALL_AUTH_TOKENS_FILE`, `RECALL_AUTH_ISSUER_URL`, `RECALL_AUTH_RESOURCE_URL`.
  - Verified end-to-end against a live server on real PostgreSQL: an unauthenticated request, an
    unknown token and a malformed header each get **401**, while a valid token completes an
    `initialize` handshake — the rejection path is exercised, not only the green one.

- **Indexing budget caps**: `recall_index` / `index_memory()` (`recall_mcp/service.py`) now
  measure the candidate file set — count and total bytes, via the new `recall.index.candidate_files`
  helper — BEFORE any file is read or embedded, and refuse the whole request if it exceeds
  `RECALL_INDEX_MAX_FILES` (default 2000) or `RECALL_INDEX_MAX_BYTES` (default 20 MB). Both are
  configurable environment variables; defaults were sized against this project's own real
  workloads (the 796-memo / ~4-6 MB eval corpus, `recall code`'s ~240 KB self-index, `make demo`'s
  5-file corpus) with headroom. Closes the cost-exhaustion half of the "indexing is client-callable
  and unbounded" gap in `SECURITY.md` and issue #9's third checkbox.
- **Right-to-erasure deletion path**: `PgVectorStore.delete_sources()` is now exposed via a
  `recall forget <source>...` CLI subcommand (dry-run by default; `--yes` to actually delete) and
  a `recall_forget` MCP tool, both tenant-scoped. `forget_memory()` / `ForgetResult`
  (`recall_mcp/service.py`) report chunks removed and sources removed separately from sources not
  found, so a typo'd source is never mistaken for a successful deletion. Closes the gap tracked in
  `SECURITY.md` and issue #9.
- **HNSW recall fix for `source`-filtered dense queries**: `query_dense()` (`recall/store.py`)
  applies `WHERE source = ...` alongside the HNSW `ORDER BY embedding <=> ...`, and the index walk
  is filter-blind — it finds the globally nearest neighbours and only then discards the ones that
  fail the filter. Measured on 20,000 rows / dim 64 / a filter matching 10% of rows / 40 queries
  (`tests/test_hnsw_filtered_recall.py`'s exact corpus shape): recall@10 **0.38** with pgvector's
  own defaults (`ef_search=40`, `iterative_scan=off`), and **40/40** queries returning fewer than
  the requested `k`. Neither `hnsw.ef_search` nor `hnsw.iterative_scan` alone is enough (the first
  restores recall but a filtered scan can still exhaust it before reaching `k`; the second stops
  the truncation but not the recall loss) — `query_dense` now sets **both**,
  `hnsw.ef_search=200` + `hnsw.iterative_scan=relaxed_order`, via `SET LOCAL` inside an explicit
  transaction (the one precondition `SET LOCAL` has), scoped to ONLY the `source`-filtered branch
  — an unfiltered query already measures recall 1.000 and pays no extra cost. Takes truncation to
  **0/40** on that corpus, and to **0/30** on an independent A/B built the way a real multi-file
  index run builds one. **Recall is a different story and both measurements are published rather
  than the flattering one:** 0.38 → ~0.90 on the fixture corpus above, but **0.523 → 0.483** on the
  normally-built one, because `relaxed_order` fills to `k` with approximate matches. The claim here
  is the narrow one — filtered dense search returns `k` results when `k` exist — not a recall
  improvement. Both HNSW knobs are configurable
  via `RECALL_HNSW_EF_SEARCH_FILTERED` / `RECALL_HNSW_ITERATIVE_SCAN_FILTERED`, following the same
  `os.environ.get(..., str(DEFAULT))` convention as `RECALL_INDEX_MAX_FILES`/`_BYTES`. Measured
  cost of the fix on this corpus: filtered-query p50 latency moves from ~6ms to ~8.6ms (the extra
  `SET LOCAL` round trips + the wider search); the unfiltered arm is untouched by construction
  (~2ms p50 either way). Note: pgvector's own HNSW build carries internal randomness this project
  does not control, so the untuned recall/latency figures move some from build to build (observed
  range across several builds: 0.33-0.41 recall, always 40/40 truncated) — the regression test
  retries the corpus build when an unusually well-connected graph fails to reproduce the pathology,
  rather than loosen the assertion. Closes issue #11's third checkbox.
- **`CREATE INDEX CONCURRENTLY` in `ensure_schema()`**: every secondary index it creates
  (`tsv`, `embedding`/HNSW, `indexed_at`, `source`, `metadata->>'file'`, `tenant_id`) now builds
  `CONCURRENTLY`. `ensure_schema()` runs on every store open, not only at first bootstrap — a
  plain `CREATE INDEX` against an already-populated, live table blocks writers for as long as the
  build takes (minutes for HNSW on a real corpus). Safe here because `ensure_schema`'s connection
  is autocommit and, unlike `replace_sources`/`upsert`, is never wrapped in an explicit
  `conn.transaction()` — every statement is already its own implicit transaction, the one
  precondition `CONCURRENTLY` has (verified directly against the container). Trade-off accepted,
  not hidden: an interrupted build can leave an `INVALID` index that `IF NOT EXISTS` will not
  retry automatically (a plain `CREATE INDEX` cannot fail this way, since it is one transaction);
  documented in `recall/store.py` alongside the change. Closes issue #11's fourth checkbox.

### Fixed

- **p50/p95/p99 were reported one rank too high** (`recall/observability.py`,
  `recall/eval/scale.py`). `int(q * n)` was used as a 0-based index, but that expression *is* the
  1-based nearest rank, so every percentile returned the next sample up. On 100 samples p99
  returned the maximum, making "1% of requests are slow" indistinguishable from "one request was
  slow" — the exact discrimination a p99 exists to provide. The index is now `ceil(q*n) - 1`.
  Two copies of the formula existed and both carried the defect; they now share one
  implementation, because fixing one while publishing from the other is how the wrong number
  reached `results/` in the first place. (`_percentile` remains as an alias of the now-public
  `percentile`.)
  - Scope, stated precisely so the fix does not take credit it is not owed: this lowers a
    reported percentile by exactly one rank when `q*n` is an integer, and changes nothing
    otherwise. It does **not** explain the larger movements in the republished
    `results/*/SCALE.md` figures — those are run-to-run variance.
- **`results/*/SCALE.md` regenerated, and FINDINGS §5b now reports the spread rather than a point
  estimate.** Re-running the index-pressure arm at the *same seed* three times moved the STR
  baseline across **0.46–0.92** and p50 latency across **5.5–67.2 ms**, because the seed does not
  fix pgvector's HNSW build and `hashing-64` puts almost no signal in the vector. STR *trust* —
  the number that arm is about — was **0.00** in all three.

### Changed

- **Schema DDL now waits a bounded time for its LOCK** (`RECALL_SCHEMA_LOCK_TIMEOUT_MS`, default
  `5000`; `0` restores the old unbounded wait). `ensure_schema()` lifts `statement_timeout` so an
  HNSW build is not cancelled — but `statement_timeout` also counted lock-wait time, so lifting it
  removed the only bound on *queueing*. `CREATE INDEX CONCURRENTLY` waits for every concurrent
  transaction on the table and the tenancy ALTERs take ACCESS EXCLUSIVE, so a single
  `idle in transaction` session elsewhere could park schema setup indefinitely, with every later
  query queued behind it and no error explaining why. Work stays unbounded; waiting does not.
  **`recall index` / `recall search` can now fail after 5s of lock contention** where they
  previously waited — the DDL is idempotent and retried on the next store open.
- **`RECALL_ALLOW_INSECURE_DSN` now takes an explicit allowlist**, not any non-empty string. Only
  `1`, `true`, `yes` or `on` (case-insensitive) disable the guard; **every other value, including
  `0` and `false`, keeps it ON**. Previously `RECALL_ALLOW_INSECURE_DSN=0` *disabled* the check —
  the opposite of what anyone writing it meant. **This can fail a deployment that currently
  starts**: if you set it to a falsey-looking value and use the built-in `recall:recall`
  credentials against a non-local host, `require_secure_dsn` will now raise at startup. That is
  the intended reading; change the credentials, or set the variable to `1` deliberately.
- **The `mcp` extra now requires `mcp>=1.27.2`** (was `>=1.10`, and `>=1.7` before that). The
  1.10 floor was necessary but not sufficient: the tenant is carried in `AccessToken.claims`,
  which only exists from **1.27.2**. On 1.10–1.27.1 the package installed cleanly and then failed
  on every authenticated call, because pydantic dropped the unknown `claims` field at
  construction. Below 1.10 the server fails loudly at import instead. Upgrade with
  `pip install -U "recall[mcp]"`.
- **`pgvector>=0.4`** (was `>=0.3`). `recall/store.py` does `from pgvector import Vector`, and
  that top-level export only exists from 0.4.0 — on 0.3.x the package installs and then
  `import recall.store` raises `ImportError`. Same defect as the `mcp` floor above, found the
  same way: by installing the declared minimum instead of assuming it.
- **CI now installs the declared minimums** (new `floor` job). `pip install -e ".[dev]"` resolves
  the newest of everything, so every `>=` bound in `pyproject.toml` was untested — which is the
  only reason both wrong floors above shipped. The job resolves the lowest direct dependencies
  on the lowest supported Python (3.11) and runs the full suite. Verified in both directions:
  green at the corrected floors, red against the previous `mcp>=1.10`.
- **`pytest-timeout` is now configured** (`timeout = 120`, `timeout_method = "thread"`). It was a
  declared dependency with nothing setting a timeout, so it did nothing — while the comment
  beside it explained that a hung test "reads as 'still running', not as a regression", which was
  precisely the state CI was in.
- **Test isolation: the `recall` logger is snapshotted around every test.** `configure_logging()`
  sets `propagate = False` — correct in a process, because a propagated record could reach stdout
  and corrupt MCP JSON-RPC — but it is global and was never undone, so once one test called it
  `caplog` stopped seeing `recall.*` records for the rest of the session. The result was
  order-dependent failures that passed in isolation; the pytest version in use masked it and the
  declared floor did not. Found by the `floor` job above before it was even committed.

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
  abstains more, and more accurately: on the held-out sweep, false-confidence on unanswerable
  queries drops from 0.205 to 0.045, costing an extra 0.7% of answerable queries (false-abstain
  0.003 → 0.010). The shipped rule separately measured 0.000 gap FCR end to end, on a different
  protocol — see FINDINGS §6 for both.
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
