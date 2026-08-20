# Production posture, and what this does not do

Itemised status of every property "enterprise-grade" is usually meant to cover, the limits that
are known and measured, and the upgrade notes that can break a working deployment.

## Production posture

"Enterprise-grade" is not a single property, so here is the itemised version — verified on a real
host (PostgreSQL 17, pgvector 0.8.2, Python 3.12, connecting as an **unprivileged** role), not only
on a laptop. Unless a row says otherwise, every figure here was measured on the default
configuration: `bge-small` via fastembed, hybrid dense+sparse, no reranker, nothing leaving the
machine. See
[which configuration to run](EVIDENCE.md#so-which-configuration-should-you-actually-run).

Configuration is part of the product surface. A deployment can stay fully local for legal or data
residency reasons, choose a hosted embedder when quality is worth the dependency, enable reranking
when a human is waiting for the answer, or run the fast profile when hardware and throughput matter
more. Those choices are explicit profiles and startup checks, not silent per-request tuning.

| Property | Status | Evidence |
|---|---|---|
| **Multi-tenancy** | ✅ `tenant_id` on every row and every query, plus a row-level-security policy (`ENABLE` + `FORCE`) | Verified as a `NOSUPERUSER NOBYPASSRLS` role — a superuser bypasses RLS, so testing it as one would have passed vacuously |
| **Concurrency** | ✅ async MCPServer tools + `psycopg_pool`; blocking embedder, database, reranker and indexing work is explicitly offloaded | Tool schemas keep the injected MCP context out of client arguments, and blocking bodies use AnyIO's worker-thread limiter |
| **Timeouts / resilience** | ✅ `statement_timeout`, `connect_timeout`, narrow reconnect-and-retry | The retry refuses to re-run a `QueryCanceled`, which would escape the very timeout that fired |
| **Security posture** | ✅ fail-closed on published default credentials; index-root confinement that survives symlinks on 3.11/3.12 | `pathlib` only gained `recurse_symlinks` in 3.13 |
| **Observability** | ✅ `logging` (text/JSON), counters and latency percentiles for abstention, verdicts, reconnects; surfaced through the MCP `recall_stats` tool | The library never attaches handlers — that is the host's job |
| **Incremental indexing** | ✅ content-hash skip, bounded-memory batched writes, prunes files deleted from disk | 5,100 chunks / 1,120 files: full **7.4 s**, unchanged re-index **0.22 s** |
| **Scale characteristics** | ✅ measured at **50,600 chunks**: recall@5 1.00 filtered and unfiltered, search p50/p95/p99 | Templated text; absolute retrieval quality is optimistic |
| **Real-corpus operation** | ✅ 794 hand-written memos → 6,491 chunks, p50 **78 ms** (pre-fix, see `archive/CHANGELOG_FULL.md` `latency_ms`) | Works at this size; see the retrieval row for how well |
| **Retrieval quality, real questions** | ✅ **hit@5 0.705** [0.56, 0.82] on a public 746-doc corpus with the free local embedder · ⚠️ **0.348** on an idiosyncratic private one — see [the tables in EVIDENCE](EVIDENCE.md#retrieval-quality-it-depends-on-your-corpus-and-here-is-the-rule) | Measured on 110 hand-labelled questions per corpus, not on headings. Corpus vocabulary dominates: a cloud embedder is worth +0.28 on the hard corpus and +0.02 on the ordinary one |
| **Data erasure** | ✅ `recall forget` / `recall_forget` permanently delete a source's chunks; previews by default, `--yes` to act | The right-to-erasure path — irreversible, so it refuses to act unattended without the flag |
| **Abuse bounds** | ✅ `recall_index` refuses before embedding anything if a request exceeds `RECALL_INDEX_MAX_FILES` / `RECALL_INDEX_MAX_BYTES` | A client-callable indexer with no cap is an unbounded spend on a cloud embedder |
| **Authentication** | ✅ bearer tokens on the HTTP transports, three scopes, one tenant per principal — see [docs/AUTH.md](AUTH.md) | Starting an HTTP transport without tokens **refuses to boot** rather than warning. stdio stays unauthenticated by design: it is a private pipe, not a listener |
| **Schema migrations** | ✅ ordered SQL, committed cryptographic checksums, advisory lock, resumable concurrent indexes, separate migration/serving roles | MCP startup is SELECT-only and refuses pending, drifted, or unknown versions; supported PostgreSQL majors 16<!--@ citation-pending: CI schema matrix in .github/workflows/ci.yml -->, 17<!--@ citation-pending: CI schema matrix in .github/workflows/ci.yml --> and 18<!--@ citation-pending: CI schema matrix in .github/workflows/ci.yml --> are tested |
| **Trust policy** | ✅ **fails closed**: an absent, stale, mismatched or uncertified calibration refuses the search rather than answering from the 0.50 <!--@ citation-pending: source constant, not a measurement — `DEFAULT_GAP_THRESHOLD` in recall/guards.py --> default. Six stable failure codes; development mode must be asked for by name | The refusal is raised *before* retrieval runs, so it cannot carry corpus bytes — asserted with a store whose read methods raise if they are reached at all. See [docs/CALIBRATION.md](CALIBRATION.md) |
| **Readiness** | ✅ reported per tenant and per process, separately | One tenant's stale calibration cannot fail the process probe and evict a pod that is still serving every other tenant |
| **Index generations and cutover** | ✅ registered generations, shadow route, dual write behind a durable ordered outbox, transactional route swap with a content-free `NOTIFY`; `cutover` refuses while any event is pending or the shadow is not ready | `parity` now **refuses** two empty generations rather than printing `OK`, because two empty generations cannot disagree and that vacuous pass was indistinguishable from a real one; there is no override flag. A shadow partially filled relative to the active still fails parity on missing sources or differing chunk counts. ⚠️ What nothing catches is a pair that agrees with **each other** while both are short of the corpus on disk, so read the chunk counts `parity` prints against your own measurement; nor does parity detect two generations whose rows all lack a content hash, which compare equal while certifying nothing. `_prune_vanished` keys its candidate set on the active generation, so a source only the shadow holds survives the prune and rides the cutover |
| **Retrieval cost profiles** | ✅ `fast`, `quality`, and `code` chosen per process, never per request; a request's `k` is clamped down and never raised; contradictory configuration refuses **startup**, not the first search; over-budget requests are shed at the door, before the query is embedded | ⚠️ Each profile's concurrency and queue depth is a stated policy choice, not a measurement, and cannot be tuned until the latency blocker below clears; the values are in `recall/profiles.py` and [docs/ENTERPRISE_RETRIEVAL.md](ENTERPRISE_RETRIEVAL.md). The quality reranker digest pins one provisioned **tree**, path names included, so it identifies a directory rather than the model in general |
| **Generator-neutral evidence boundary** | ✅ fixed system prompt with no interpolation site, every corpus byte JSON-escaped inside a delimiter whose own `<` and `>` are escaped, citations must resolve to supplied chunk IDs, abstention bypasses the generator entirely | ⚠️ **No generator is chosen, shipped or configured**, so the end-to-end path is exercised against a stub only. Validation is structural: it does not claim a cited passage entails the answer |
| **Serving latency** | ❌ **PENDING, and PENDING blocks promotion** | No idle reference host exists. `decide` emits `latency_p95_ms=None` unless a certified number is passed, observed timings are recorded as `observed_diagnostic_only` and are not a gate input, and no timing taken on a loaded host is cited for any promotion decision |
| **HA / replication** | ❌ out of scope — this is a library over your Postgres | — |

> **Upgrading to the strict trust policy.** This is a breaking change. Retrieval against a corpus
> with no published, exactly-bound calibration now raises `TrustRefusal` rather than answering with
> uncertified confidence numbers. Local and research workflows opt in with
> `TrustPolicy.development()`, or `RECALL_TRUST_MODE=development` for the CLI, which still
> retrieves but marks every hit `unverified` and refuses to claim an abstention. See the
> [CHANGELOG](../CHANGELOG.md).

## What this does not do

Stated plainly, because the failure mode this library exists to prevent is confident overreach.

- **Abstention catches *far gaps*, not *near-misses*.** Where the unanswerable questions are
  genuinely off-topic it works — accuracy **1.00** on the PEPs, **0.89** on the real corpus. Where
  they are near-misses *by construction* (the haystack is the user's own history and the question
  asks about something never mentioned but topically adjacent) it fails: on LongMemEval it wrongly
  refused **48%** of questions retrieval had answered correctly. **Six** candidate signals were
  measured on the same 500 questions and all six failed — the best carries a 95% interval of
  **[0.680, 0.826]** and the ~0.90 bar sits *outside* it, so this is a measured **exclusion**, not
  a small-sample shrug. Relevance is not answerability. Independently corroborated on LOCOMO, where
  no threshold or judge configuration — including a stronger judge — crosses into usable territory.
  Nothing was retuned, because every alternative measured worse; instead `recall calibrate` reports
  your calibration set's separability with its interval, judges the bar against that interval's
  **lower bound**, and **exits non-zero rather than certify a threshold the data cannot support**.
  → [FINDINGS §9–§10](../results/FINDINGS.md)
- **Validity is authored, not inferred.** On the reference corpus, **2** of 792 memos declared
  `supersedes:` while **60** closed a decision only in prose. `recall lint --fix` was built to close
  that gap and, after review, could safely declare **zero** of them — narrating vs declaring, part
  vs whole, augmenting vs replacing are invisible to a pattern and obvious to the author. It ships
  as a reviewing aid; `recall check` moves the question to write time.
  → [#29](https://github.com/GiulioDER/RE-call/issues/29), closed; the limitation stands
- **Gap detection is bounded by the embedder.** With a weak one, no threshold separates answerable
  from unanswerable — measured, not assumed.
- **Successor and abstention accuracy are unmeasured on generated corpora.** Every synthetic
  document is the same sentence with a different opaque token, so those columns measure token
  discrimination, not the trust layer. STR, latency and scale figures are unaffected.
- **Filtered ANN search stopped truncating — which is not better recall.** An HNSW walk is
  filter-blind, so a `source`-filtered query exhausted its candidate list before finding `k`
  matches: at pgvector's defaults, **40/40** queries silently returned fewer results than asked
  for. `hnsw.ef_search=200` + `hnsw.iterative_scan=relaxed_order` fix that unambiguously (0/40 and
  0/30 in two measurements). Those two **disagree on recall** — 0.36–0.43 → 0.88–0.94 on the test
  fixture, **0.523 → 0.483** on a normally-built corpus — because `relaxed_order` fills to `k` with
  approximate matches. It trades truncation for approximation. The unfiltered path still runs at
  the defaults, and the tenant-predicate combination has not been measured on a multi-tenant table.
  Note the pathology is a **statistics race**, not graph shape: an unanalyzed table takes a
  `Seq Scan` and reports recall 1.0000 under any `ef_search`.
  → [#57](https://github.com/GiulioDER/RE-call/pull/57), [#98](https://github.com/GiulioDER/RE-call/pull/98)
- **No token revocation without a restart.** Bearer tokens, scopes and one tenant per principal
  ship ([docs/AUTH.md](AUTH.md)), but the
  token file is read at startup, so removing access takes effect on reload, not on save. Per-tenant
  rate limits and an indexing byte quota ship too, but their buckets are per process, so N workers
  admit roughly N times the rate. For revocation, rotation or per-request identity, front this with
  a real identity provider and supply the MCP SDK's `auth_server_provider`.
- **No bundled HA.** Versioned, checksum-verified migrations and an unprivileged serving role now
  ship, but replication, backups, failover and managed-Postgres operations remain yours until the
  production reference deployment lands.
- **Production promotion is gated on certification, not blocked.** Immutable lineage, atomic
  blue-green generations, and exact tenant/generation-bound calibration artifacts ship, and
  `generation promote` under `RECALL_ENV=production` now succeeds for a generation whose calibration
  resolves CERTIFIED and refuses every other status. `--unsafe-development-promotion` is refused
  outright there rather than ignored, so the development escape hatch cannot be carried into
  production by habit.

  **`generation rollback` is deliberately NOT gated**, because it is the incident path: it activates
  the previous generation whatever its calibration status, and records that status plus the
  operator's reason in the audit event. The reasoning is under "Decisions on the eleven questions"
  in `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`. Expect a rollback to be able to downgrade a
  tenant from certified to provisional, visibly.

  What has **not** landed is strict refusal at *read* time: an uncertified generation that is
  already active still returns corpus text, marked uncalibrated, rather than refusing. This
  repository does not yet claim the seven-session enterprise target is complete.

## Upgrading

Recent release detail is in [CHANGELOG.md](../CHANGELOG.md), and the full historical changelog is in
[archive/CHANGELOG_FULL.md](archive/CHANGELOG_FULL.md). Only the changes that can make something
currently working start failing are listed here.

**→ 0.6.0 — your retrieval results will change on the same corpus and the same queries.** The first
non-additive release since 0.5.1, because three defects each made retrieval return *less* than it
should have: the lexical leg ANDed every query term (so `hybrid` was in practice dense-only); the
dense leg was silently capped at 40 candidates by `hnsw.ef_search`, ignoring any larger
`candidate_k` without error; and a freshly-indexed table did not use its vector index until
autovacuum caught up. All three make results **better**, and none changes an API — but baselines,
thresholds calibrated against retrieval scores and golden-output tests will move. That is the point
of the fixes. Nothing needs reconfiguring. Two of this project's own published claims rested on the
capped dense leg and were corrected in the same pass
([FINDINGS §7, §9a](../results/FINDINGS.md)).

**→ 0.5.1 — five changes that can break a working deployment.** `RECALL_ALLOW_INSECURE_DSN` became
an explicit allowlist, so only `1|true|yes|on` disable the guard and **every other value, including
`0`, keeps it ON** — the likeliest of these to bite. The `mcp` extra now requires `mcp>=2,<3`;
1.10-1.27.1 installed cleanly but does not expose the MCPServer API this server uses. `recall index`
refuses a re-index that would prune ≥50% of a root (`PruneGuardTripped`; re-run with `--allow-prune`), so a
*missing* corpus stops being indistinguishable from a *deleted* one. The MCP HTTP transports refuse
to boot without `RECALL_AUTH_TOKENS_FILE` and meter per tenant by default; `stdio` is unchanged.
Schema DDL gives up after 5 s of lock contention (`RECALL_SCHEMA_LOCK_TIMEOUT_MS`).

**→ 0.5.0 — the chunks table gains `tenant_id` and its primary key becomes `(tenant_id, id)`.**
`ensure_schema()` migrates in place and assigns existing rows to the `default` tenant, which is also
the default `tenant=`, so a single-tenant deployment upgrades without noticing (there is a test that
builds an old-shape table and asserts the row survives). The key had to change: chunk ids derive
from the file path, so two tenants indexing the same layout produced the *same id* and one tenant's
re-index silently overwrote the other's row. Two behavioural changes ride along — the abstention
threshold is now fitted mid-gap rather than on the lowest answerable sample, so it abstains more and
more accurately (re-run `recall calibrate` and re-check any pinned threshold); and `supersedes:`
matching accepts `name`, `name.md`, `[name]` and `[[name]]`, so previously-dangling edges may start
applying and memories served as `ok` can correctly come back `superseded`.

0.5.2 (LOCOMO benchmark) and 0.5.3 (LangChain / LlamaIndex retrievers) are purely additive.
