# Security Policy

RE-call's threat surface is **data confidentiality**, not code execution. This is a library that
puts an agent's own memory in Postgres and searches it; it does not run untrusted input as code and
does not execute arbitrary third-party artifacts. What follows is a plain statement of where the real
risk sits.

## Supported versions

| Version | Supported |
|---|---|
| 0.5.x | ✅ |
| < 0.5 | ❌ |

Pre-1.0, only the current `0.x` line gets fixes. There is no LTS branch.

## The corpus is the asset

The thing this library retrieves is **an agent's own memory** — accumulated decisions, closed
experiments, incident notes, sometimes a secret pasted into prose because someone was moving fast
in a markdown file. RE-call does not redact, encrypt, or classify any of it: a chunk goes in exactly
as written and comes back exactly as written.

**There is no per-chunk access control.** Isolation is at the tenant level (`tenant_id` on every
row plus a Postgres row-level-security policy — see the README's Production posture table), not the
chunk level. Anyone who can authenticate as a tenant, or who has read access to the underlying
Postgres database, can read every memory that tenant has ever indexed. If your memory corpus
contains anything you would not want a co-tenant or a database operator to see, that content should
not be indexed in the first place — RE-call has no mechanism to selectively withhold it later.

RLS is also **bypassed by a superuser or any `BYPASSRLS` role**, including the role shipped in this
repo's `docker-compose.yml`. `store.check_rls_effective()` tells you whether your connection is
actually enforcing the boundary; the MCP server logs a warning at startup if it is not. Treat that
warning as a real finding, not noise.

## Cloud embeddings are a real egress boundary

`recall.embeddings.VoyageEmbedder` sends the **text of every chunk** to Voyage's API
(`VOYAGE_API_KEY`, the `voyageai` package). Embedding a private memory corpus with this backend
means that corpus's content leaves the host and is processed by a third-party service — that is not
a hypothetical, it is what "embed with a cloud model" means.

For a sensitive corpus, use `recall.embeddings.FastEmbedEmbedder` instead: it runs the embedding
model locally (`pip install recall-rag[fastembed]`) and never makes a network call with chunk text. This
is the default — and, as shipped, the *only* backend `recall_mcp/server.py` and
`examples/self_recall_agent.py` can select via `RECALL_EMBEDDER` (`make_embedder` in
`recall_mcp/service.py` accepts `"fastembed"` or `"hashing"` only; `VoyageEmbedder` is reached by
constructing it directly in your own code, not through that env var). Both `make eval` and
`python -m recall.eval` run the local embedder unconditionally and only add the Voyage row when
`VOYAGE_API_KEY` is present in the environment — the key-free path is the one that never leaves the
host.

**Choosing to embed with Voyage is documented, intended behaviour** when you opt into that backend,
not a vulnerability to report. What we do want reported: any place `VoyageEmbedder` or a similar
cloud path is reached *without* the caller having asked for it (an implicit fallback, a default that
silently prefers the cloud embedder over the local one, etc.).

## Credentials

- **`VOYAGE_API_KEY`** is read from the environment (`recall/embeddings.py`) or a gitignored `.env`
  loaded by `recall/_env.py`. Never commit it. It is visible to anything that can read the process
  environment of a running `recall` process — treat it with the same care as any API key.
- **The Postgres DSN (`RECALL_DSN`)** carries a password in the connection string. `recall/store.py`
  redacts it before logging (`redacted_dsn`) so a connection failure never writes a plaintext
  password to a log file or a systemd journal — but the DSN itself, wherever you configure it
  (environment, `.env`, an MCP client's config block), is a credential and should be handled as one.
- **The published `recall:recall` default credentials are for the local Docker dev database only.**
  `require_secure_dsn` makes the MCP server refuse to start if those exact credentials are pointed
  at a non-local host, and the CLI warns on the same condition; `RECALL_ALLOW_INSECURE_DSN=1` is the
  explicit, greppable opt-out for a genuinely private network. Do not set that variable to silence
  the warning without actually changing the password.
- **`.env` is never committed** (`.gitignore`) and `.env.example` documents the keys without values.

## The evaluation harness (`recall/eval/`)

`recall/eval/` is the project's own measurement harness — it indexes the project's own eval corpus
into throwaway Postgres tables, embeds it, runs retrieval, and writes `results/RESULTS.md` and
charts via `matplotlib`. It does not shell out to run arbitrary code, does not fetch or execute
third-party artifacts, and is not a tool that runs untrusted input on someone else's behalf — unlike
a code-audit tool, its job is to score *this* library against *its own* labelled queries. Running
`make eval` or `python -m recall.eval.scale` is running project code you can read in full, against a
disposable database.

The one thing worth knowing: the optional near-miss stage (`recall.entailment.QnliEntailmentJudge`)
and the local embedder (`FastEmbedEmbedder`) both load models via `sentence-transformers` /
`fastembed`, which download model weights from the Hugging Face Hub on first use if not already
cached. That is a network fetch of model artifacts, not of your corpus — but it does mean the first
run of `make eval` (or any code path that constructs those classes) is not fully offline.

## Known gaps, tracked and open

These are documented weaknesses, not undiscovered ones. They are recorded in
[issue #9](https://github.com/GiulioDER/RE-call/issues/9) and are stated here because a security
policy that lists only the limits it has already solved is misleading.

**Authentication shipped; token lifecycle is still manual.** The HTTP transports
(`streamable-http`, `sse`) now require bearer tokens and **refuse to start without them** — an
unauthenticated listener cannot be created by accident. Each token maps to a principal with a
tenant and scopes, and the tenant selects its own connection pool, so a principal cannot reach
another tenant's rows (`recall_mcp/auth.py`, `recall_mcp/stores.py`; see docs/AUTH.md).

What remains open is lifecycle, not enforcement: the token file is read at startup, so there is
**no revocation or rotation without a restart**, and a leaked token is valid until it is removed.
There is no proof-of-possession — terminate TLS in front of the server. `stdio` remains
unauthenticated by design: it is a private pipe to one client, not a listener.

**Requests are bounded individually and in aggregate; the limiter is per process.**
Each `recall_index` request is measured — candidate file count and total bytes — BEFORE anything
is read or embedded, and refused whole if it exceeds `RECALL_INDEX_MAX_FILES` (default 2000) or
`RECALL_INDEX_MAX_BYTES` (default 20 MB).

Aggregate spend is bounded by per-tenant budgets (`recall_mcp/limits.py`), debited at the same
choke point that authorises a call, so a tool cannot be metered-by-omission:

| budget | default | environment variable |
|---|---|---|
| read calls | 120 / min | `RECALL_RATE_READ_PER_MIN` |
| write calls | 20 / min | `RECALL_RATE_WRITE_PER_MIN` |
| forget calls | 10 / min | `RECALL_RATE_FORGET_PER_MIN` |
| indexed source text | 200 MB / hour | `RECALL_INDEX_BYTES_PER_HOUR` |

The byte budget is the one that bounds **cost**: request count prices a 20 MB index and a
200-byte one identically, so a caller staying under the per-request cap could previously call it
in a loop. Bytes are charged pre-flight against the set about to be embedded, so a refusal has
spent nothing. Budgets are keyed by **tenant**, not by principal — two tokens on one tenant are
one bill, and letting a tenant mint another token to double its quota would make it advisory.

Set any of these to `off` to disable it. A malformed or non-positive value falls back to the
default rather than being read as "unlimited": `0` means "no limit" to one reader and "nothing
allowed" to another, and guessing wrong in a spend control removes the cap.

Two limits worth knowing. **Buckets live in the process**, so N server workers admit roughly N
times these rates — honest for the single-process-behind-TLS deployment this targets, and the
first thing to revisit before running a fleet. And **`stdio` is not metered**: it is a private
pipe to one local client with no principal to charge, matching how authentication is scoped.
Query length is unrelated — `recall_search`'s `k` is already clamped server-side
(`MAX_SEARCH_K` in `recall_mcp/service.py`).

**Deletion is exposed; retention is mechanism, not schedule.**
`PgVectorStore.delete_sources()` (`recall/store.py:686`) is now wired into `recall forget` (CLI,
dry-run by default — pass `--yes` to actually delete) and into the `recall_forget` MCP tool
(`recall_mcp/server.py`, delegating to `forget_memory` in `recall_mcp/service.py`), both
tenant-scoped like every other write path. That closes the original gap — there is a supported
way to make the system forget an indexed memory.

Erasure has a second, automatic path: re-indexing removes rows for files that are **gone from
disk**, so deleting a memo and re-syncing erases it without a separate `forget` call. The same
mechanism handles the per-chunk case — a source's rows are replaced wholesale on re-index, so
editing a paragraph out of a file and re-indexing removes exactly that chunk. Deleting a single
chunk while leaving the file untouched is still not possible, and is not planned: the file is the
record, and an index that disagreed with it would be the more dangerous state.

**That automatic path is destructive, and is now guarded.** `recall index` does not look like a
delete command, but a corpus directory that is present-but-empty — an unmounted volume, an
interrupted sync, a path that still resolves — is indistinguishable from "the author deleted
everything", and the whole corpus was silently removed with exit code 0. A run that would drop at
least `RECALL_MAX_PRUNE_FRACTION` (default 0.5) of the sources indexed under that root is now
refused with nothing deleted, once the corpus is above a small floor where a fraction is
meaningful. `--allow-prune` proceeds deliberately.

**There is still no time-based retention policy, and that is a decision rather than a gap.**
`indexed_at` records when a file last *changed*, not when it was last seen — an unchanged file is
skipped on re-index and its timestamp does not move (measured, not assumed). A policy that purged
"memories older than N days" would therefore delete the memos that have been stable longest,
which in a memory corpus are the settled, load-bearing ones. Authored expiry (`valid_until`) is
honoured at read time by **demoting** an expired memory, not deleting it, so the trust layer can
tell you a memory is stale and show it to you anyway. If your corpus contains personal data, you
are responsible for deciding *when* to erase; this provides the mechanisms, not the schedule.

## Reporting a vulnerability

Please report privately via **[GitHub Security Advisories](https://github.com/GiulioDER/RE-call/security/advisories/new)**
on this repository rather than opening a public issue. Include what you found, how to reproduce it,
and its impact.

This is a solo-maintained project. Response is **best effort** — there is no SLA on acknowledgement
or fix timelines. You will get a reply, and a fix or a documented mitigation, as soon as I can manage
one.

## Out of scope

- **"The cloud embedder sends chunk text to Voyage."** Documented above; it is what you asked for
  when you selected `VoyageEmbedder`. Not a vulnerability.
- **Token revocation requiring a restart.** Known and documented in
  [docs/AUTH.md](docs/AUTH.md) — the token file is read at startup. A concrete exploit path
  beyond that is very much in scope; the lifecycle limitation itself is known.
- **Retrieval quality issues** (a query returning the wrong chunk, a low `hit@5`) are correctness
  bugs, not security issues — file those as regular GitHub issues, ideally with the `bug_report`
  template's confidence/verdict fields filled in.
