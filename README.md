<p align="center">
  <img src="docs/banner.svg" alt="RE-call — Retrieval-Augmented Self-Recall" width="900">
</p>

<p align="center">
  <b>Trustworthy retrieval for an AI agent's own memory.</b><br>
  Every hit comes back with confidence, provenance, and validity — or the honest answer is <i>"I don't know."</i>
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PostgreSQL-16%2F17%20%C2%B7%20pgvector-336791" alt="PostgreSQL + pgvector">
  <img src="https://img.shields.io/badge/tests-352%20·%20real%20pgvector-brightgreen" alt="352 tests">
</p>

<p align="center">
  <a href="#see-it-in-one-screen">See it</a>
  &nbsp;·&nbsp;
  <a href="#what-is-actually-verified">What's verified</a>
  &nbsp;·&nbsp;
  <a href="#production-posture">Production posture</a>
  &nbsp;·&nbsp;
  <a href="#quickstart--2-minutes-no-api-key">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="#what-this-does-not-do">Limits</a>
</p>

---

**Most RAG hands back the closest vector match. That's the wrong answer more often than you'd think.**

A long-running agent piles up memory — decisions, closed experiments, incident notes — and then it
**re-litigates settled decisions**, **hallucinates over gaps** the memory can't fill, and **builds on
facts that are no longer true**. The catch: when you've reversed a decision, the *stale* memory of it
is often the **highest-cosine hit in the whole result**. Similarity search serves it, confidently.

RE-call is a retrieval engine for that memory that is *honest about what it doesn't know*. It returns
**verdict + confidence + provenance** with every hit — not just similarity — demotes memories that
were superseded or expired, and prefers an explicit **abstention** over confident noise.

## See it in one screen

<p align="center">
  <img src="docs/superseded-catch.svg" width="740" alt="recall demo: the stale rate-limit memory has the highest cosine (0.806) but is flagged superseded and demoted below the current memory; an unanswerable query returns an explicit ABSTAIN.">
</p>

<details>
<summary>same run, as text</summary>

```text
$ python -m recall.cli demo

[ok] query='how many requests per second can a client make?'
  ok           conf=1.00  cos=0.784  rate_limits_v2.md                       '# API rate limits (revised)'
  superseded   conf=1.00  cos=0.806  rate_limits_v1.md → use rate_limits_v2  '# API rate limits … limited to 100'

[ABSTAIN · gap] query='how do we handle penguins on mars?'
  reason: no hit above the calibrated confidence threshold (probable corpus gap)
```
</details>

Look at the cosines. The **stale** memory scores **higher (0.806)** than the current one — plain vector
search returns it, and the agent builds on a limit that no longer exists. RE-call flags it
`superseded`, points at its successor, and puts the *current* memory on top. When the memory genuinely
has no answer, it says so. **That ordering decision is the whole thesis.**

## What is actually verified

Every headline number below was measured, and every one carries its limit. Where a claim could not be
supported, it was withdrawn rather than softened — the withdrawals are listed too, because a claims
table without them is marketing.

| Claim | Measurement | Limit |
|---|---|---|
| **Supersession beats similarity** | Superseded-trust rate **0.00**, 95% Wilson **[0.00, 0.02]**, n=250, against a baseline of **1.00** — plain search returns the stale memory *every time* on adversarially-worded queries | Generated corpus; the successor/abstain columns on it are **not** meaningful (below) |
| **Abstention is calibrated, not guessed** | On the real corpus: threshold **0.728 ± 0.042** over 4 index rebuilds, false-abstain **0.015**, gap false-confidence **0.000** | Needs ≥ ~20 labelled samples; below that the rule loses its outlier robustness |
| **Timestamps cannot replace declared supersession** | "Trust the newest relevant hit", steelmanned, still trusts the stale memory **83–100%** of the time | — |
| **Reranking rescues a weak embedder** | Hybrid + cross-encoder lifts MRR **0.63 → 1.00** offline | Situational: a strong embedder already saturates this corpus |
| **Fine-tuning pays only for a vocabulary gap** | **+0.00** on a rich corpus; **0.31 → 0.55** held-out MRR on opaque jargon → [study](docs/RAG_TRAINING_STUDY.md) | Measure your gap first |
| **Near-misses need a judge, not a threshold** | QNLI stage cuts near-miss false-confidence **1.00 → 0.60**, **0.80 → 0.50**, same judge across embedders → [study](docs/ENTAILMENT_SUPERSESSION_STUDY.md) | Judge-alone *degrades* far-gap detection — the two stack, neither replaces the other |

Full methodology, per-embedder tables and the negative results → **[results/FINDINGS.md](results/FINDINGS.md)**.
Design rationale and the reasoning behind each guard → **[docs/WRITEUP.md](docs/WRITEUP.md)**.

### Claims that were withdrawn

A previous version of this file published each of these. They did not survive re-measurement:

- **"FCR @calibrated 0.00"** — the threshold was fitted and scored on the same samples. On separable
  data that is 0.00 by arithmetic. Now cross-validated, and the fitting rule was
  [replaced outright](results/FINDINGS.md) after it proved to let **20.5%** of unanswerable queries through.
- **Coverage and abstention accuracy on generated corpora** — the "unanswerable" queries were an
  answerable query plus a nonsense suffix, so nothing could separate them. Rebuilt as genuinely
  off-topic questions; the *document*-level degeneracy remains and is stated as unmeasured.
- **"6× faster incremental re-index"** — understated. Measured on a Linux server it is **33×**.
- **Real-corpus recall@5 of 0.945** — that used document *headings* as queries, which is known-item
  retrieval. Against 110 hand-labelled questions phrased the way a person actually asks, hit@5 is
  **0.33**. The caveat was always printed; now it has a number, and it is the weakest measured part
  of the system. → [FINDINGS §7](results/FINDINGS.md)

## Production posture

"Enterprise-grade" is not a single property, so here is the itemised version — verified on a real
host (PostgreSQL 17, pgvector 0.8.2, Python 3.12, connecting as an **unprivileged** role), not only
on a laptop.

| Property | Status | Evidence |
|---|---|---|
| **Multi-tenancy** | ✅ `tenant_id` on every row and every query, plus a row-level-security policy (`ENABLE` + `FORCE`) | Verified as a `NOSUPERUSER NOBYPASSRLS` role — a superuser bypasses RLS, so testing it as one would have passed vacuously |
| **Concurrency** | ✅ async MCP tools + `psycopg_pool`; the server previously served exactly **one** request at a time | FastMCP awaits async tools and calls sync ones *inline* — there is no thread offload |
| **Timeouts / resilience** | ✅ `statement_timeout`, `connect_timeout`, narrow reconnect-and-retry | The retry refuses to re-run a `QueryCanceled`, which would escape the very timeout that fired |
| **Security posture** | ✅ fail-closed on published default credentials; index-root confinement that survives symlinks on 3.11/3.12 | `pathlib` only gained `recurse_symlinks` in 3.13 |
| **Observability** | ✅ `logging` (text/JSON), counters and latency percentiles for abstention, verdicts, reconnects; surfaced through the MCP `recall_stats` tool | The library never attaches handlers — that is the host's job |
| **Incremental indexing** | ✅ content-hash skip, bounded-memory batched writes, prunes files deleted from disk | 5,100 chunks / 1,120 files: full **7.4 s**, unchanged re-index **0.22 s** |
| **Scale characteristics** | ✅ measured at **50,600 chunks**: recall@5 1.00 filtered and unfiltered, search p50/p95/p99 | Templated text; absolute retrieval quality is optimistic |
| **Real-corpus operation** | ✅ 794 hand-written memos → 6,491 chunks, p50 **78 ms** | Works at this size; see the retrieval row for how well |
| **Retrieval quality, real questions** | ⚠️ **hit@5 0.33** [0.21, 0.47], n=46, on 110 hand-labelled questions | Headings-as-queries scored 0.945 — the proxy hid two thirds of the failures. Reranking, candidate-pool size and chunk size were each tested and each moved it ~0.00–0.06. hit@50 plateaus at **0.50** in every configuration — a hard recall ceiling, pointing at the embedder, not the pipeline |
| **Authentication** | ❌ **not implemented** — stdio MCP carries no transport identity | [#9](https://github.com/GiulioDER/RE-call/issues/9) |
| **Schema migrations** | ❌ runtime `CREATE TABLE IF NOT EXISTS`, no versioned upgrade path | Pre-tenancy tables *are* migrated in place, with a test |
| **HA / replication** | ❌ out of scope — this is a library over your Postgres | — |

## How it works

```mermaid
flowchart LR
    Q([query]) --> E[embed]
    E --> D[dense · pgvector cosine]
    Q --> S[sparse · Postgres full-text]
    D --> F[Reciprocal Rank Fusion]
    S --> F
    F --> R[cross-encoder rerank]
    R --> G{trust layer}
    G --> O([verdict + confidence + provenance per hit, or ABSTAIN])
```

Dense semantic search and sparse keyword search each retrieve candidates; **Reciprocal Rank Fusion**
merges them, a cross-encoder reranks, and the **trust layer** judges every hit — supersession,
validity window, calibrated confidence — before it reaches the agent. Validity is plain frontmatter
in the memory itself (`supersedes: old_doc.md`, `valid_until: 2026-06-30`) — *authored, not inferred*,
because a claim honoured as written is safe and a claim guessed at is not.

## Where this comes from

RE-call is extracted from the memory system behind a production trading-research agent whose memory
outgrew its context window. That corpus is the one the numbers above were measured against:
**792 hand-written markdown memos → 6,469 chunks**, re-indexed daily.

Every guard here is a scar from a real failure — re-litigating a falsified experiment, trusting a
weak hit on an unanswerable question, building on a fact that had been reversed. Running the library
back against that corpus is also what exposed the defects listed under [Engineering](#engineering):
real files carry stray bytes, real authors write `[[wikilinks]]` where the parser expected filenames,
and real closure notes hedge.

**→ [Redacted case study](docs/CASE_STUDY.md)** — the real structure, the guards in action, and
exactly what is public versus private.

## Quickstart · 2 minutes, no API key

```bash
docker compose up -d --wait          # PostgreSQL + pgvector
pip install -e ".[fastembed]"        # local embeddings, no API key
python -m recall.cli demo            # index corpus/ and run the sample queries
```

## Use it

```bash
python -m recall.cli index ./notes                       # index a folder of markdown
python -m recall.cli search "what did we decide about caching?"
python -m recall.cli lint ./notes                        # supersession-graph health (no DB)
python -m recall.cli lint ./notes --fix                   # propose missing edges (dry run)
python -m recall.cli check ./notes/new-memo.md --strict    # write-time gate, for a pre-commit hook
```

```python
from recall.store import PgVectorStore
from recall.embeddings import FastEmbedEmbedder
from recall.trust import trusted_search

emb = FastEmbedEmbedder()
with PgVectorStore(DSN, dim=emb.dim, tenant="acme", pool_size=8) as store:
    store.ensure_schema()
    result = trusted_search(store, emb, "what is the rate limit?")
    if result.abstained:
        ...  # say you don't know — do not answer from these hits
    for hit in result.hits:
        hit.verdict      # ok | superseded | expired | not_yet_valid | low_confidence | …
        hit.confidence   # calibrated; 0.5 sits exactly on the abstention boundary
        hit.validity.superseded_by
```

Point `RECALL_DSN` at any Postgres.

> **Two operational notes.** The test suite **DROPs tables**, so it reads a separate
> `RECALL_TEST_DSN` and never `RECALL_DSN` — exporting your real DSN and running `pytest` cannot
> touch it. And the MCP server **refuses to start** if `RECALL_DSN` carries the built-in
> `recall:recall` credentials against a non-local host; set a real password, or
> `RECALL_ALLOW_INSECURE_DSN=1` to accept the risk deliberately.

> **Multi-tenancy.** Set `RECALL_TENANT` or `PgVectorStore(tenant=...)`. RLS enforces the same
> boundary in the database, so a forgotten `WHERE` returns nothing rather than another tenant's
> memories. ⚠️ **RLS is bypassed by a superuser or a `BYPASSRLS` role** — including the one in this
> repo's `docker-compose.yml`. Connect as an unprivileged role, or that second layer is decoration;
> `store.check_rls_effective()` tells you which you have, and the server warns at startup.

## Use it with Claude (MCP)

```json
{ "mcpServers": { "recall": {
    "command": "python", "args": ["-m", "recall_mcp.server"],
    "env": { "RECALL_DSN": "postgresql://...", "RECALL_TENANT": "acme" } } } }
```

Four tools: `recall_search` (verdict + confidence + provenance, or an explicit abstention),
`recall_index`, `recall_forget` (permanently delete a source's chunks — irreversible,
tenant-scoped), `recall_stats` (size, freshness, and the process metrics). Full guide →
[docs/USING_WITH_CLAUDE.md](docs/USING_WITH_CLAUDE.md).

## What this does not do

Stated plainly, because the failure mode this library exists to prevent is confident overreach.

- **No authentication.** Any client that can reach the MCP server gets that tenant's memory.
  Tenancy is enforced; identity is not established. → [#9](https://github.com/GiulioDER/RE-call/issues/9)
- **Validity is authored, not inferred.** On a real 792-memo corpus, **2** memos declared
  `supersedes:` while **60** described a closure only in prose. `recall lint --fix` was built to
  close that gap and, after review, could safely declare **zero** of them: narrating vs declaring,
  part vs whole, augmenting vs replacing are invisible to a pattern and obvious to the author. It
  ships as a **reviewing aid**, with `recall check` moving the question to write time. →
  [#29](https://github.com/GiulioDER/RE-call/issues/29)
- **Successor and abstention accuracy are unmeasured on generated corpora.** Every synthetic
  document is the same sentence with a different opaque token, so those columns measure token
  discrimination, not the trust layer. STR, latency and scale figures are unaffected.
- **Gap detection is bounded by the embedder.** With a weak one, no threshold separates answerable
  from unanswerable — measured, not assumed.
- **ANN recall is untuned.** `hnsw.ef_search` is left at its default and HNSW build
  nondeterminism measurably moves calibration. → [#11](https://github.com/GiulioDER/RE-call/issues/11)

## Engineering

**352 tests, 3 skipped.** The database-touching ones run against a real pgvector container — no mock
DB. CI runs `ruff`, the suite against PostgreSQL, and `pip-audit` over a checked-in `uv.lock`, as a
gate rather than a report.

Tests are written to fail for the right reason. A representative sample:

- the RLS tests connect as a role that **cannot bypass RLS**, because as a superuser they would pass
  while testing nothing;
- the cross-tenant test asserts the other tenant's row **exists** before checking it is invisible,
  so a silently failed write cannot make it green;
- the supersession-cache test counts real table scans, so a "fix" that quietly became *rescan every
  search* would be caught;
- the metrics test asserts the counters move on the **real retrieval path** — instrumentation that
  is never wired up reports zero forever and reads as "nothing is going wrong".

Several defects were found only by running the library against a real corpus and a real server, and
each has a regression test quoting the input that caused it: a single NUL byte in one file aborting a
792-file index; every declared supersession edge failing on reference *formatting*; five tests that
encoded the developer's own environment and failed on a correctly-configured host.

## Upgrading to 0.5.0

**The chunks table gains a `tenant_id` column and its primary key becomes `(tenant_id, id)`.**
`ensure_schema()` performs that migration in place on an existing table and assigns existing rows to
the `default` tenant, which is also the default `tenant=` — so a single-tenant deployment upgrades
without noticing. There is a test that builds an old-shape table, inserts a row, opens it with this
version, and asserts the row survives and is still retrievable.

The key had to change: chunk ids derive from the file path, so two tenants indexing the same layout
produced the *same id*, and under the old single-column key one tenant's re-index silently
overwrote the other's row.

Two behavioural changes worth knowing before you upgrade:

- **The abstention threshold is fitted differently** (mid-gap rather than on the lowest answerable
  sample). It abstains *more*, and on measured data far more accurately — false-confidence on
  unanswerable queries drops from 0.205 to 0.000 for 1.5% of answerable queries. Re-run
  `recall calibrate` and re-check any threshold you have pinned.
- **`supersedes:` matching is more tolerant.** `name`, `name.md`, `[name]` and `[[name]]` now all
  resolve to the same document, so edges that were silently dangling may start applying. That is the
  intent — on the reference corpus it took working edges from 0 to 2 — but it does mean memories
  that were served as `ok` can now correctly come back `superseded`.

## Reproduce

```bash
make eval                                        # ablations + trust + near-miss → results/
python -m recall.eval.scale --embedder hashing --filler 50000    # scale + latency
```

## License

MIT — see [LICENSE](LICENSE).
