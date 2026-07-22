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
model locally (`pip install recall[fastembed]`) and never makes a network call with chunk text. This
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
- **Missing authentication on the MCP server.** Already tracked and stated plainly in the README's
  "What this does not do" section and [#9](https://github.com/GiulioDER/RE-call/issues/9) — stdio
  MCP carries no transport identity. If you have a concrete exploit path beyond what's already
  documented there, please do report it; the general gap is known.
- **Retrieval quality issues** (a query returning the wrong chunk, a low `hit@5`) are correctness
  bugs, not security issues — file those as regular GitHub issues, ideally with the `bug_report`
  template's confidence/verdict fields filled in.
