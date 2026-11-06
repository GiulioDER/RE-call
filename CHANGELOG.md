# Changelog

This file keeps the release surface short. The full historical changelog lives at
[docs/archive/CHANGELOG_FULL.md](docs/archive/CHANGELOG_FULL.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is pre-1.0
`0.MINOR.PATCH`, so a minor bump may still break schema or API.

## [Unreleased]

## [0.15.0] (unreleased)

RE-call 0.15.0 adds Lite, a local store in one SQLite file that needs no Docker and no database
server, and ships it in the PyPI wheel for the first time. It also adds the local memory dashboard,
audit records looked up by exact key, stated facts in the fact ledger and the `recall withdraw`
erasure counterpart, and it carries the fixes of pre-release audits of Lite, the MCP server and its
authentication, the fact ledger, provenance, the dashboard, the hooks and the first-run code. The
code audits run during this release recorded at least 1,068 bugs in the shipped packages whose fixes
are in it (95 rated P1, the highest severity, 668 P2 and 305 P3), and 190 of the pull requests
merged since 0.14.0 are fixes. The project license is now the AGPL-3.0 with a commercial license
beside it (see Changed). Several changes are breaking; read the upgrade notes before upgrading a
serving host.

### Upgrade notes

Each item is described in full under its own heading below. `docs/PRODUCTION.md` has the same
list as an ordered upgrade procedure.

* **Migrations 0026 to 0035** (fact ledger retraction, value erasure, the materialized current
  facts, withdrawal, the supersession revision table and its index) need `recall schema apply
  --with-migrations` with the migration role, then `recall schema grants --controller`. On a
  ledger that already holds events, run `recall provenance rebuild` after 0030. A database already
  at 0033 applies 0029 last. PostgreSQL's pgvector must be 0.8.0 or later.
* **Run `recall schema apply` before starting a server.** A non-enterprise multi-tenant server and
  `recall quickstart --existing-dsn` no longer migrate a table on first use.
* **Packaging:** `recall_aml`, the `hosted` extra and `recall-hosted` are gone from the wheel and
  the sdist; numpy is a core dependency; the QNLI entailment judge and its `entail` extra are
  removed, and a configuration that still sets `RECALL_ENTAILMENT` to a true value refuses.
* **Corpora re-embed or re-index once.** Every FastEmbed profile fingerprint changes once, an
  `st:` model loaded from a local directory or a pinned revision fingerprints its weights, the
  structured document version and the code chunker version move, and document extraction moves to
  v3. Upgrading `voyageai`, `fastembed`, `onnxruntime`, `tokenizers` or `sentence-transformers` on
  a serving host turns every generation that recorded a profile into `LINEAGE_MISMATCH` until it
  is rebuilt, so pin those libraries or rebuild before switching traffic (#909).
* **Rebuild semantic graphs once.** The graph builder's output changed (date handling, file
  identity, supersedes of compiled records, duplicate diagnostics), and its schema version moves
  from 2 to 3. Existing graphs keep serving under the version they were built with; run `recall
  graph rebuild --generation <id>` for each live generation to pick up the fixes. A generation left
  in `validating` by older code is rebuilt by `recall generation validate` rather than failed.
  The change is one way: code before 0.15.0 refuses a graph built at version 3, and 0.15.0
  refuses one built by a newer version. Upgrade the serving checkout and the CLI that builds and
  validates generations together, and after a rollback run `recall graph rebuild` with the older
  code for each live generation.
* **Atomic rescue artifacts built by 0.14.0 are refused**: their `manifest.json` lacks
  `embedding_fingerprint`. Rebuild them.
* **Configuration that refuses now:** a source policy file with an unknown or misspelled key, an
  enterprise control plane without `RECALL_SOURCE_POLICY_FILE`, `RECALL_OIDC_JWKS_REFRESH_SECONDS`
  above 3600, an unknown `RECALL_DEPENDENCY_INVALIDATION` value, a `RECALL_GRAPH_*` control outside
  its vocabulary (a blank value still means the default), and a `RECALL_MULTIMODAL_ENABLED` that is
  not a recognised boolean, all at startup. Source policies whose redactions are not sorted get new
  digests, so rows filtered on the old digest return nothing until a re-index. * **Calibration:** a
  labelled set that repeats a query, or labels one both ways, is refused; a v2 `recall calibration
  import` needs `--labels` and a matching `--embedder`; some v2 bundles fitted before #1106
  re-derive a different threshold and must be recalibrated. * **Behaviour that changes answers or
  scripts:** strict PostgreSQL searches refuse a term absent from the generation (`unknown_term`, on
  by default); `recall_reasoning_query` answers with the `dated` profile by default; a forget by a
  name that matches several sources refuses and lists them; an explicit glob index skips `.git`,
  `.venv`, `venv`, `node_modules`, `site-packages` and `__pycache__`; an `as_of` without an offset
  is read as UTC; a stdio server refuses a `tenant` argument naming another tenant; a dependency
  projection read that raises is `DEPENDENCY_UNAVAILABLE`; and every tool that writes a
  `supersedes:` edge is gated. * **Redis idempotency:** the request fingerprint is bound to the
  authenticated principal, so a key in flight across the upgrade needs a new key.

### Added

* **RE-call runs with no Docker and no database server: a Lite store keeps one project's memory in
  one SQLite file.** `recall setup --lite` indexes a memo folder (`--memory`, default
  `RECALL_INDEX_ROOT`, else the current directory) into `.recall/memory.db` (`--store` to change
  it), calibrates it and registers the MCP server for the project at Claude Code's local scope
  (`--no-register` leaves the client config alone). Any `RECALL_DSN=sqlite:///<path>` serves `recall
  index`, `recall search`, the stdio MCP server, `recall doctor`, which opens the file read-only and
  reports a missing file, the corpus size and width, a tenant mismatch and the calibration state,
  and `recall dashboard`, which without `--db-dsn-file` reads a `sqlite:///` DSN from
  `RECALL_SERVING_DSN` or `RECALL_DSN` (never a PostgreSQL one) and refuses `--tunnel`. Trust
  verdicts, supersession and strict trust work as on PostgreSQL: the store calibrates itself from a
  query set generated offline from its own corpus, searching with another embedder is
  `LINEAGE_MISMATCH`, and a write carries the calibration forward by re-verifying the fitted
  threshold against stored evidence, refitting in the background once the corpus has moved by more
  than a quarter or the check fails. The keyword leg reproduces PostgreSQL's `ts_rank`. A lite
  strict search also refuses, with the new verdict `unknown_term`, a question naming an entity-like
  token the store never contained (`RECALL_UNKNOWN_TERM_GATE=0` disables it), and each fit reports
  synthetic near-miss coverage. Writes to one file queue in two lanes, `write` and `refit`
  (`RECALL_LITE_WRITE_WAIT`, 300 s; `RECALL_LITE_REFIT_WAIT`, 120 s), and a busy MCP tool names the
  holder. `recall_current_facts`, `recall_apply_fact`, `recall provenance current` and `apply`,
  `recall forget`, `recall scopes` and `recall calibration list`, `show` and `export` are served,
  with operation receipts for every keyed mutation; `recall_calibration_publish` is not, because a
  certified lite calibration is in force when it is saved. A command a lite store cannot serve
  (`generation`, `graph`, `reasoning`, `idempotency`, `schema status`, `plan` and `apply`,
  `calibrate`, `demo`, `code`, `lint --semantic`, plain `setup`, `quickstart --existing-dsn
  sqlite:///...`) exits naming itself, and a non-default `--table` is refused rather than ignored.
  The plugin's DSN field is now "Database DSN" and accepts a `sqlite:///` address. See
  `docs/LITE.md`. A relative `sqlite:///` path is refused by the server, `recall doctor` and the
  dashboard; `recall setup --lite` indexes through the refresh, so the source policy and stated
  facts apply and `recall lite refresh --check` reports the store current; it refuses to register
  without the MCP SDK, and with neither `--memory` nor `RECALL_INDEX_ROOT` it refuses the home
  folder or a filesystem root. Search and `recall_current_state` name the lite generation. A forget,
  or an index that drops a source, deletes the calibration evidence quoting it, so strict search
  refuses until the next refit. SQLite `secure_delete` is on for every connection. `recall
  --embedder X lite refresh` uses X, and `recall lite refresh` follows a `sqlite:///` DSN without
  `--store`.

* **The lite store keeps itself current with its memo folder.** The MCP server refreshes a
  `recall setup --lite` store when it starts and, before a search, when the folder changed (at most
  every `RECALL_LITE_REFRESH_INTERVAL` seconds, thirty by default, waiting up to
  `RECALL_LITE_REFRESH_WAIT`, three): it indexes what changed under the store's write lane and
  refits the calibration. Like `recall_index`, it applies the source policy and refuses a folder
  over `RECALL_INDEX_MAX_FILES` or `RECALL_INDEX_MAX_BYTES`. While refreshing fails, search answers
  carry a `refresh_warning` and `recall doctor` reports it. `recall lite refresh [--check]` does the
  same by hand; `RECALL_LITE_AUTO_REFRESH=0` turns the automatic refresh off. On a lite server,
  `recall_index` now files every memo under its path relative to `RECALL_INDEX_ROOT`, as the refresh
  does, so `recall_forget` takes that name whichever indexed the memo first.

* **A local memory dashboard, `recall dashboard`.** It serves a page on 127.0.0.1 only for the memo
  folder named by `--root` (default `RECALL_INDEX_ROOT`): a review queue of claims that one memo
  replaces another (agent reports and cached arbiter judgements), where accepting writes
  `supersedes:` into the newer memo as `recall rewrite apply` does and rejecting records the claim;
  a reader for each memo (`/read`); a 3D memory graph; a person-only editor for `supersedes`,
  validity dates and status, with a ledger and undo; activity, memory health and a memo search. With
  `--db-dsn-file` (or `RECALL_DASHBOARD_DSN_FILE`), and optionally `--tunnel HOST` with
  `--tunnel-ports`, it reads the corpus database in read-only sessions: tenants, generations and
  calibration, searches recorded by the decision ledger (`RECALL_DECISION_LEDGER=1`), a Control page
  per memory, one file's page (`/corpus/source`) and a graph of any corpus (`/graph?corpus=TENANT`)
  with supersessions, authored links, Python imports and files sharing an uncommon name. A
  Configuration page shows the settings in force on this machine, in each registered RE-call MCP
  server and, with `--config-host` (and `--config-command`), on a remote host; every report comes
  from `python -m recall_mcp.config_report --json` run where the values live, so secrets are
  withheld before they leave it. `RECALL_DSN` and `RECALL_REASONING_API_KEY` are now flagged secret
  in the settings schema. The page calls no model; a second dashboard on a port already in use
  refuses to start. Every edit saves recovery bytes first and an interrupted one is reconciled; an
  accept writes only over the bytes the reviewer saw and can be undone; the editor refuses files
  that are not memos. The form secret is independent of the session cookie, so another application's
  cookie on 127.0.0.1 no longer locks you out. With no tenant chosen the corpus pages read `default`
  (pass `--reports-tenant memory` for another), they list the tenants you own, and
  `RECALL_DASHBOARD_TENANTS` pins a labelled set. The dashboard is not read-only: accepting a review
  or saving an edit writes the memo file.

* **Agents can report what memory did, through two new MCP tools.** `recall_report_stale` reports,
  with a verbatim quote from each, that one memory replaces another; the claim is queued for a
  person (`recall dashboard`, `recall rewrite`) and lands in the `.recall` sidecar or as a
  `stale_report` row in `recall_audit_events` that keeps a fingerprint of each quote, never the
  text. `recall_report_use` records, after a task, whether memory `helped`, made `no_difference` or
  `misled`, the memories used and any found wrong, as one `use_report` row; every memory named must
  exist in the served generation. Both need `recall:write` and change no memo, verdict or ranking.
  Both join the default tool list, and the re-call skill asks for one report after a task that
  searched memory. Both take an optional `idempotency_key`, and on an authenticated server a stale
  report is recorded only in the caller's own audit ledger.

* **Memos can carry an audit record, looked up by exact key.** A one-line JSON `recall_record` front
  matter key (schema `recall/schemas/recall_record.v1.json`: `bug`, `false_positive`, `lesson` and
  retraction records) is validated at index time and stored exactly as authored; an invalid one is
  dropped with a warning naming the file, and the memo is still indexed. The MCP tool
  `recall_records` and the CLI `recall records` (`--anchors FILE` with `--repo`, `--id`, `--all`,
  `--kind`, `--json`) return records by anchor (`exact`, `symbol_changed`, `file_only`) or by id
  with no embedder, superseded and expired records marked rather than hidden; more than 200 anchors
  or 500 matches is refused. `recall_related` gains the `caused_by` relation. Served on a lite
  store. The chunk metadata rule version moves, so the next generation build re-derives every source
  rather than reusing its chunks. `recall records` applies the source policy, every entry carries
  `expired`, a call with neither anchors nor ids is refused, and a memo with two `recall_record`
  lines is refused. A record holding text that cannot be stored is dropped with a warning and the
  memo is indexed without it.

* A `recall_record` of kind `bug` may carry `defect_class`, one of the fixed classes cca-audit's
  `audit-fix.md` lists ("Defect classes"), or null, and `defect_class_reason`. Both are optional,
  so every existing record stays valid; an unlisted class is refused like any other invalid
  record (the memo is indexed, or loaded, without it).

* **Facts a memo states in its `recall_graph` front matter reach the fact ledger when the memo is
  indexed.** On a lite store this happens at refresh, `recall_index` and `recall index`, by default
  for the folders the refresh follows. On PostgreSQL it is opt in: `RECALL_STATED_FACT_SOURCES`
  names the places that may state facts (a URI, an absolute path, or a path under
  `RECALL_INDEX_ROOT`), and facts are recorded at promote, rollback and a legacy index, or by the
  new `recall generation stated-facts [GEN]` for a corpus promoted earlier; set the variable in both
  the promoting and the serving process, since the rule is applied again at read time. An edited,
  superseded or removed memo, or a narrower rule, hides its facts until the change is undone, a memo
  whose content cannot be checked hides its fact, and an upload never states one.
  `recall_current_facts` gains `sources`, the memos behind each fact, and `contested`, values two
  memos state with no supersession between them, listed there and not under `facts`. A PostgreSQL
  append the controller role refuses is reported as -1 with a warning naming
  `RECALL_FACT_WRITE_DSN`. A forgotten or pruned memo that superseded another keeps that memo's
  stated facts retired, as the document layer and applied facts already do; `recall withdraw` still
  gives them back. A desktop upload's promote records stated facts through the controller DSN the
  MCP server resolved.

* **The agent that writes a memo can settle supersession as it writes.** A writer, such as an LLM
  agent calling `recall_remember`, declares the fact's key in the memo it writes instead of leaving
  keys to be extracted from prose later. `recall_current_facts` takes `subject` (and `predicate`)
  and answers with that subject's facts, matched ignoring case and spacing, plus `similar_subjects`,
  so a writer can look a subject up and repeat an existing key exactly. A `recall_remember` receipt
  for a memo stating a `fact` gains `fact_key`: the key, its state, the values in force and the
  supersession left to settle, so the writer sees which earlier memo its fact should supersede and
  can declare it. The new deterministic rule `deterministic.stated_key_successor`
  (`recall.declared_keys.successor_pairs`) proposes, for review only, that of the memos stating one
  single-valued key each supersedes the one of the latest earlier `valid_from`, whatever the value;
  it never writes an edge.

* **An opt-in assistant memory surface for a lite server.** `RECALL_ASSISTANT_MEMOS=<folder>`
  (inside `RECALL_INDEX_ROOT`) adds `recall_remember`, which writes and indexes one fact as one memo
  (with `valid_from`, `valid_until`, `supersedes` and an optional structured `fact`) and returns
  `{memo_id}`, and `recall_forget_memo`, which forgets one memo of the folder and deletes its file;
  `recall_search`, `recall_evidence`, `recall_current_facts` and `recall_apply_fact` then answer in
  a compact view that names each hit by its memo id. `RECALL_MCP_TOOLS=assistant` serves exactly
  those six tools. `RECALL_EVAL_CLOCK=<file>` makes a lite server read "now" from a file holding one
  ISO 8601 time with an offset, re-read on every call. With neither variable set nothing changes.
  `recall_remember` refuses empty text and reports the key of a memo not yet or no longer in force
  as `not_in_force`; the memo ledger is written as version 2 when it carries the request journal,
  and a newer version is refused.

* Add optional memo correction receipts, revision preconditions and persistent request identities;
  refuse a missing or already superseded correction target, legacy calls included (a replay of a
  request whose memo is already in place keeps its original reply); calls without the new parameters
  keep the `{memo_id}` reply. `recall_remember` takes `request_id`, `expected_revision` and
  `include_receipt`, and search hits carry the indexed revision a correction must name (see
  `docs/MEMO_CORRECTIONS.md`).

* **Multimodal memories live in their own opt in tenant.** `RECALL_MULTIMODAL_ENABLED=1` turns on
  the isolated `re-call-multimodal` tenant with the registered profile `voyage-multimodal-3.5-v1`
  (`RECALL_EMBEDDER=voyage-multimodal`, needs `VOYAGE_API_KEY`). `recall.multimodal.build_media_ref`
  admits an image as bounded metadata (SHA256 digest, media type, size, and an `s3://` or `file://`
  reference inside `RECALL_MULTIMODAL_OBJECT_ROOT`); media bytes never enter vector rows or
  metadata, and a reference outside the root, including one that escapes it by traversal, is
  refused. Limits: `RECALL_MULTIMODAL_MAX_MEDIA_BYTES`, `RECALL_MULTIMODAL_MAX_RESPONSE_BYTES`,
  `RECALL_MULTIMODAL_MAX_ITEMS`; the tenant and profile are fixed by `RECALL_MULTIMODAL_TENANT` and
  `RECALL_MULTIMODAL_EMBED_PROFILE`. The tenant needs its own generation and calibration and is
  never mixed into the text tenants. See `docs/MULTIMODAL.md`. The object root check refuses a
  backslash or NUL in the decoded path and a path that still changes after decoding, so `..%5C` and
  a repeatedly encoded `..` cannot escape it; an enabled multimodal tenant refuses at startup an
  object root no admission can use. The multimodal client honours `RECALL_VOYAGE_TIMEOUT_SECONDS`,
  the install hint names `voyageai>=0.3` and Pillow (and the embedder warns at construction when
  Pillow is missing), a text query works on a client that only embeds text, and fitting a very long
  thin image no longer takes one step per pixel. A leading `//` (raw or `%2F` encoded) is refused so
  a file URI cannot name a UNC host, an empty or `.` segment inside the root's own prefix is
  refused, and the projection re-checks `object_uri` under the object root. Pillow opens only the
  declared format, and an image type outside JPEG, PNG and WebP is never opened. A bad root is named
  as the fault, a `RECALL_MULTIMODAL_MAX_*` refusal names the variable, its value and its range, and
  `RECALL_MULTIMODAL_ENABLED` is read strictly everywhere, so a typo refuses instead of meaning off.
  `RECALL_VOYAGE_PARALLEL_REQUESTS` also applies to the multimodal embedder, which sends no further
  request after one fails. A `voyageai` without `multimodal_embed` is refused at construction, every
  vector must have the width the construction probe measured, and an empty image query is refused
  locally. `fit_voyage_input` and `estimate_voyage_tokens` are deprecated.

* **A search can be routed to a configured tenant by a versioned plan.**
  `RECALL_RETRIEVAL_PLANS_JSON` holds routes, each naming an allowed tenant set, a primary tenant
  and at most one rescue tenant, matched deterministically on scope, modality, source prefix or
  query terms. MCP `recall_search` and `recall_evidence` take optional `scope`, `modality` and
  `route_id`, and answer with `retrieval_plan`. Unset, search stays single tenant. Invalid
  configuration, an unknown route, a tenant outside the allowlist, a retired generation or a
  mismatched embedding profile refuses. A plan that selects any tenant other than the caller's own
  needs the new OAuth scope `recall:federate`.

* **Bounded federation across specialist tenants, off by default.** `recall.federation.federate`
  (exported from `recall`) runs a route plan's legs independently and merges them by rank
  (reciprocal rank fusion, never by comparing cosines across profiles), keeping a protected primary
  prefix and a bounded rescue tail. `RECALL_FEDERATION_MODE` is `off` (default), `shadow` (runs the
  plan without changing what is served) or `active`; bounds are `RECALL_FEDERATION_MAX_LEGS`,
  `RECALL_FEDERATION_MAX_CONCURRENCY`, `RECALL_FEDERATION_CANDIDATE_K`,
  `RECALL_FEDERATION_RESULT_K`, `RECALL_FEDERATION_PRIMARY_PREFIX`,
  `RECALL_FEDERATION_RESCUE_SLOTS`, `RECALL_FEDERATION_RRF_K` and `RECALL_FEDERATION_INVALID_LEG`
  (`omit` or `fail_closed`). `recall_evidence` always runs single tenant. See
  `docs/FEDERATED_RETRIEVAL.md`. Legs share one process-wide pool of 16 workers, a refused primary
  keeps its error type, shadow mode falls back on a slow primary, and `recall:federate` widens a
  search only to the caller's routes and, with a subject binding, to its bound tenants.

* **Experimental evidence proof mode for `recall_reasoning_query`.** `mode="evidence_proof"` checks
  that every required slot of the question binds to an exact span of trusted evidence and returns a
  `proof_obligations` receipt (decision, reason code, evidence chunk ids, missing slots), never a
  generated answer. Off unless `RECALL_REASONING_PROOF_ENABLED=1`; it calls OpenRouter only, and
  `RECALL_REASONING_PROOF_MODEL` and a pinned `RECALL_REASONING_PROOF_REVISION` are required
  (`unpinned` is refused). Other settings: `RECALL_REASONING_PROOF_API_KEY`, `_BASE_URL`,
  `_TIMEOUT`, `_MAX_TOKENS`, `_CONTEXT_TOKENS`, `_REASONING_EFFORT`, `_COST_PER_1K_TOKENS`. The
  receipt carries `calibration_coverage`.

* Add the opt-in `dated-premise` answer profile for `recall_reasoning_query`
  (`RECALL_REASONING_ANSWER_PROFILE=dated-premise`): when a question assumes something the evidence
  shows is no longer current, the response has outcome `corrected` and a cited `premise_correction`
  instead of an answer built on the old state or an abstention. Whether a superseded memory matched
  the question is decided structurally and its successor's evidence is named to the reader
  (`diagnostics.premise_notice`). The default `dated` profile, its prompt and its three-key envelope
  are unchanged.

* `recall calibration near-miss` scores a held-out near-miss question set against a calibration and
  reports a certificate (`near_miss_certificate` in status); `recall calibration served-gap`
  compares the served top-k maximum with the exact maximum calibration sampled. Both report only.

* Search and evidence now report unchecked semantic query support separately from retrieval
  eligibility, and carry calibration-bound synthetic near-miss coverage when available. Missing
  measurements remain unmeasured and report failures are unavailable. Existing retrieval decisions,
  certification and citation eligibility are unchanged. The compact assistant surface and CLI
  expose the same distinction. No verifier or additional model call is enabled.

* **`recall generation build` shows its progress while it runs.** Before embedding it prints a
  plan on stderr (an upper bound on the objects it can reuse, and a `FULL RE-EMBED` line when the
  pipeline fingerprint moved and no retained generation shares it), then an update every 25
  objects or 30 seconds, a heartbeat naming the stage and object while one embedding call holds
  the build, and a final count. The summary line on stdout is unchanged. `GenerationManager.build`
  and `build_generation` take an optional `progress` sink (`recall.build_progress`).

* **A model arbiter proposes undeclared supersession edges for review.** With
  `RECALL_SUPERSESSION_ARBITER=1` and `RECALL_ARBITER_API_KEY`, `recall rewrite plan` finds
  candidate pairs (lexical neighbours, a file name trigger, direction from `modified:` or a dated
  file name) and asks a model for a probability backed by two verbatim quotes; a pair at
  `RECALL_ARBITER_THRESHOLD` (default 0.90) becomes a `requires_review` proposal that only `recall
  rewrite apply` with a named reviewer writes. It sends an excerpt of both memos to
  `RECALL_ARBITER_BASE_URL`; `RECALL_ARBITER_MODEL`, `RECALL_ARBITER_REVISION`,
  `RECALL_ARBITER_NEIGHBOURS` and `RECALL_ARBITER_MAX_PAIRS` tune it, and answers are cached at
  `RECALL_ARBITER_CACHE`, so apply and reject after a plan make no calls. Requires the `extract`
  extra. `recall rewrite plan` now marks a proposal BLOCKED, naming the existing value, when the
  memo already declares a different `supersedes:`, where it said DECLARED.

* **`recall_evidence` can answer from 20 passages of a paged document.** With
  `RECALL_PAGED_EVIDENCE=on` and no `k` from the client, it retrieves at the profile's new
  `paged_returned_k` (20 on the legacy, fast, quality and code profiles) and keeps 20 items when at
  least three of the first five hits come from PDF or PPTX documents, else the standard five. An
  explicit `k` is never widened, `max_items` still only narrows, and hosted-quality does not opt in.
  The choice is reported in `explanation.details.evidence_depth` and counted by
  `recall_evidence_depth_total`. Off by default; `recall_search` and the CLI are unchanged.

* **Voyage retrieval profiles that send `input_type`.** `voyage-4-retrieval-v1`,
  `voyage-4-large-retrieval-v1` and `voyage-code-3-retrieval-v1` embed queries as `query` and
  passages as `document`, as Voyage recommends. Select one with `RECALL_EMBEDDER=voyage` and
  `RECALL_EMBED_PROFILE=<id>`. The existing ids and the `voyage:<model>` spelling stay untyped, and
  a corpus is served under the id it was built with.

* `RECALL_EMBEDDER=st-prompted:<hf-id>` (#908) encodes a local sentence-transformers model with
  the query and document prompts it was published with, for the models pinned in
  `recall/embedding_prompts.py`, each at a fixed Hugging Face revision. It has its own name
  (`st-prompted:<hf-id>@<version>`) and profile, never the identity of `st:<hf-id>`.
  `voyageai/voyage-4-nano` runs repository code and needs `RECALL_ACCEPT_REMOTE_MODEL_CODE=1`.
  It refuses when a local path shadows the model id, and when the installed sentence-transformers
  would route the model's text through a chat template. Behaviour change: `python -m
  recall.eval.labelled --embedder` resolves any spelling other than `hashing`, `voyage`, `st:` and
  `fastembed` through `RECALL_EMBEDDER`'s resolver and raises on an unknown one. It used to
  measure `BAAI/bge-small-en-v1.5` for all of them and report the result under the name it was
  given, so an earlier labelled result for any other spelling measured bge-small.

* **Atomic rescue can place its winner after fusion.** `RECALL_ATOMIC_RESCUE_PLACEMENT=fused`
  inserts the rescued parent at final rank six after fusion and reranking, so the retriever's top
  five are untouched; `dense` (the default) keeps the 0.14.0 placement at dense rank six. Artifacts
  are now loaded memory mapped and read only, so serving processes on one host share one copy.
  `recall.atomizer.chunk_micro_views` (24 word windows, stride 12) and the content addressed view
  store `recall.atomic_view_store` let an artifact be rebuilt embedding only chunk texts it has not
  seen.

* **Voyage embedding requests can overlap.** `RECALL_VOYAGE_PARALLEL_REQUESTS` (1 to 16, default 1)
  lets one embed call keep that many independent Voyage requests in flight. The vectors and their
  order are the ones a sequential call returns.

* **New registered embedding profile `voyage-code-4-v1`** (`voyage-code-4`, 1,024 dimensions,
  sending Voyage's `query` and `document` input types).

* Add general knowledge hooks to `recall_hooks`: an operator-curated, prebuilt `index.json` of
  procedures and warnings is injected as context before a matching `Bash` or `PowerShell` command
  (`PreToolUse`, trigger keyed) and after a failed one (`PostToolUseFailure`, ranked against the
  error by lexical BM25). At most two entries per call, each once per session, one event log line
  per trigger or failure. Injection needs the operator's own index, and the repository ships no
  entries. A registered failure hook still logs diagnostics without an index unless disabled. It
  never denies a call, never raises, loads no embedder and makes no network call. The
  `general_knowledge` block of `recall-hook.json` switches it off (`enabled: false`); a
  `recall-hook.json` that cannot be read or parsed (or is not a JSON object), a block that is not an
  object, or an `enabled` that is not a boolean also disables it, and only a missing file means the
  enabled default. A reinstall keeps the block as it was, and over a file it cannot read it writes
  the block as off. `recall setup` and the plugin register `PowerShell` in the `PreToolUse` matcher
  and a `PostToolUseFailure` hook for `Bash` and `PowerShell`. Terminal colour codes (which
  PowerShell puts in its error text) are removed before a failure is signed or ranked. A failure
  signature names the program a command runs (past `cd`, `export`, `source` and the like, `timeout`,
  and `python -m`) and the line that says what failed (a traceback's exception line, else the last
  line with an error word), not the first line. It keeps no identifying data: an ssh host that is an
  address or a dotted name is `ssh:<host>`, and addresses, mailboxes, key fingerprints, credential
  values and long tokens become placeholders. An entry scoped `project:<folder>` must be rescoped to
  `project:<owner>/<name>` in a checkout with an origin remote.

* New Claude Code hook `python -m recall_hooks pre-agent` (`PreToolUse` on `Agent`): an auditor
  subagent's prompt gains the prior bugs and lessons CCA recorded at the code in scope, read from
  the local CCA memo directory with no network or embedding. Never false positives, never a
  permission decision, fails open within about a second. Settings: `RECALL_AUDITOR_TYPES`,
  `RECALL_AUDITOR_MEMOS`, `CCA_BUGS_DIR`, `RECALL_AUDITOR_BRIEF_BUDGET`; the command `python -m
  recall_hooks warm-auditor-brief` fills its memo cache. See `docs/AUDIT_MEMORY_DESIGN.md`, "Auditor
  brief hook". The hook never follows a symbolic link or junction inside the audited repository, so
  a link committed there cannot make the machine contact another host, and it gives no block inside
  a CCA run held out from memory. `recall setup` and the Claude Code plugin do not install it: add
  the `PreToolUse` entry with matcher `Agent` to your Claude Code settings by hand. The Codex plugin
  runs it from `SubagentStart`.

* The auditor brief hook now labels and ranks a record anchored at a whole file (no symbol) as
  `file_only`, below records anchored at a real symbol, and within a level lists symbol anchors
  before symbol-less ones, then newest first. `recall_records` matching is unchanged.

* The auditor brief hook's default budget is now 3 s, up from 1 s. A CCA dispatch launches its
  auditors at once, and under that contention the hook's git calls timed out, so 0 of 9 auditors
  got the brief (measured 2026-10-27); at 3 s all 9 did. `RECALL_AUDITOR_BRIEF_BUDGET` still
  overrides it.

* **Codex installs get the hooks Claude Code installs get.** The Codex bundle
  and adapter now handle `PostToolUse` (a proven shell failure reaches the failure handler) and
  `SubagentStart` (the auditor brief, as advisory child context), and the Codex installer writes
  `project_root`, without which write-time retrieval and the SessionEnd and PreCompact index never
  ran on Codex. Both plugin bundles ship portable `session-open` and `session-close` skills, and
  startup guidance leaves calibration and freshness to the live MCP verdicts.

* `recall_search` says what an answer may cite (#906). Its result carries `evidence`: `decision`,
  `reason_code` and `citable`, the chunk ids of its own hits that `recall_evidence`'s rule admits,
  built from the same trusted result with no extra retrieval and no card registration. When
  something is citable the advice says to answer only from those hits; a result the trust gate
  could not certify adds the same DEGRADED qualification `recall_evidence` gives. `recall_evidence`
  remains the tool for card warrants, the rendered answer prompt and passages beyond the search's
  hits. `RECALL_SEARCH_EVIDENCE=off` (also `0`, `false`, `no`) sets `evidence` and the later
  `historical` field to null and drops their advice. A memo whose authored facts do not parse no
  longer fails the search or evidence call that returns it: a malformed fact is left off its
  evidence card, and a memo with no well-formed fact gets no card.

* **Current facts are materialized, so a read no longer replays the tenant's history.** Migration
  0030 adds `recall_fact_current` (one row per effective interval of each applied assertion: its own
  window less the windows of the successors that hide it), `recall_fact_current_events` and
  `recall_fact_current_state`, on PostgreSQL and in a lite file alike; migration 0031 builds the
  ledger's `(tenant_id, created_at, event_id)` index CONCURRENTLY for the replay. `current()`, the
  fact projection `recall_current_facts` serves and the conflict check of `recall_apply_fact` read
  them whenever they reflect every event of the tenant, and replay the ledger otherwise, so the
  answer is the same either way; the conflict check reads one conflict key instead of the tenant.
  Each append materializes what it wrote in its own transaction, and a failure there leaves the
  tables behind without failing the append. `recall provenance rebuild` rebuilds them from the
  ledger (`--reconcile` files only what they lack, `--check` reports, and `--check --verify`
  recomputes every row from the ledger and compares, exiting 1 on a difference, so an edited table
  is found even when the counts agree), on PostgreSQL and on a `sqlite:///` store. The tables hold
  ids and hashes, never a fact value. Grants: `recall schema grants --controller` now includes them,
  and `--strict` serving grants read them only; re-run both after migrating. Migration 0030 creates
  the tables empty: on a database whose ledger already holds events, run `recall provenance rebuild`
  once per tenant after migrating, or every read keeps replaying (`docs/MIGRATIONS.md`).

### Changed

* **A forget decides per evidence card on a lite store and the legacy table (#1233).** A fact
  whose every card cites a chunk the forget erased is retracted, and its value erased, even when
  another root still serves a memo under the same root-relative name; a value another card still
  supports stays. The erased chunk ids are recorded before deletion, so running a failed forget
  again completes it. A generation store keeps the name rule, since its chunks are read one
  generation at a time.

* **The project license changed to the AGPL-3.0, with a commercial license beside it.** Releases
  from this one on are open source under the GNU Affero General Public License v3.0, which permits
  commercial use on its conditions; closed source products and proprietary hosted services can
  take a separate commercial license instead (see `COMMERCIAL_LICENSE.md`). Outside contributions
  are accepted under a Contributor License Agreement (`CLA.md`), so contributed code can be offered
  under both. Release 0.14.0 remains under the PolyForm Noncommercial License 1.0.0 and, by the
  grant in `LICENSE_GRANT_AGPL.md`, is also available under the AGPL-3.0, as is every other version
  published under PolyForm. Earlier releases keep the MIT or Apache 2.0 terms they were published
  under.

* **BREAKING for atomic rescue: an artifact must name the embedder it was built with.**
  `manifest.json` now carries `embedding_fingerprint`, and serving refuses an artifact that lacks
  it or was built with another embedder ("atomic rescue manifest lacks embedding_fingerprint").
  Rebuild artifacts made by 0.14.0. Builds now write into a staging directory first, and refuse
  views whose parent identities disagree with their chunk ids.

* **BREAKING: upgrading an embedding library on a serving host refuses its profiled generations
  (#909).** Serving and calibration now compare a generation's recorded embedding profile and its
  fingerprint with the runtime embedder, not only the model name and dimension. The fingerprint
  includes the version of the client library the profile names (`voyageai`, `openai`, `fastembed` or
  `sentence-transformers`, plus `transformers` for `st-prompted:` models), for fastembed also the
  installed `onnxruntime` and `tokenizers` versions and its ONNX providers, and for a hosted profile
  its backend, base URL and output dimensions. Changing any of these on a serving host turns every
  generation that recorded a profile into `LINEAGE_MISMATCH`: refused in strict mode, served
  degraded and uncalibrated in development mode. Calibrating that generation is refused too. Pin the
  library on serving hosts, or build and calibrate a new generation under the new version before
  switching traffic. Generations that recorded no profile are unaffected; legacy fastembed
  generations record a fallback profile and are covered.

* **BREAKING: every tool that writes a `supersedes:` edge is gated, so none demotes a current memo
  by mistake.** `recall lint --fix`, the deterministic extraction engine behind `recall rewrite
  plan`, and the deterministic proposal rule behind `recall reasoning proposals` and the MCP
  `recall_rewrite_plan` tool no longer propose an edge from a cue the sentence does not assert
  ("this never supersedes X", "nothing replaces X", "if the canary fails, this would be replaced by
  Y", "if the canary fails; this is replaced by Y"); a cue more than 2,000 characters into an
  unbroken line is refused rather than judged on part of it. The extraction engine's revision is now
  `truth-extraction-rules-v2`, so cached extractions are recomputed. The `lint --fix` dry run shows
  each edge with an id and the sentence it came from, and `--apply` now takes the ids to write,
  directly after it (`recall lint <path> --fix --apply <id>,<id>`). A bare `--apply`, which wrote
  every proposal, and `--apply` without `--fix`, which was ignored, now exit 2 and write nothing.
  The direct-reference rule takes a document named right before the cue as the superseder ("new.md
  replaces old.md" in meeting notes), proposes it as `requires_review` with that document among its
  evidence ids, proposes nothing when another document is named elsewhere in the sentence, and skips
  a stem two files share. `lint --fix --apply`, `rewrite apply` and the dashboard's edit, undo and
  review actions refuse an edge that would close a supersession cycle in the whole corpus, before
  the write. A negation is not hidden by an aside right before the cue or its `be`, `being` or
  `been` ("this has not (as of today) been superseded by Y", "it is not, however, superseded by Y"),
  and the direct-reference rule judges a condition that falls past the 4,000 characters it reads
  cues from. `--apply` takes ids copied with the brackets the dry run prints, and refuses a repeated
  `--apply` and an id two proposals share. `lint --fix --apply` skips a memo saved while its edge
  was being checked rather than overwrite the save, and `recall.fix.apply_proposal` raises
  `recall.fix.SupersessionCycle`, a subclass of `UnreadableMemo`, for a cycle. The dashboard judges
  a memo named in another case, on a filesystem that ignores case, as the file it opens. `recall
  lint`, `recall lint --fix` and the dashboard's triage read at most 2,000 characters behind each
  cue to find its sentence when a sentence end lies within them, instead of the whole memo before
  it, with the same results as before.

* **BREAKING: schema migrations refuse pgvector older than 0.8.0.** Filtered queries set
  `hnsw.iterative_scan`, a pgvector 0.8 setting, while the docs stated a 0.7 floor. `recall schema
  apply` now refuses on an older extension and `recall doctor` fails it. Upgrade the extension to
  0.8.0 or later before migrating.

* **BREAKING: a non-enterprise multi-tenant server no longer migrates a tenant's table on
  its first request.** It checks the schema, as single-store startup does, and a pending migration
  refuses with `SchemaTooOld` naming `recall schema apply`. Under a least-privilege serving role
  the old path failed with a permission error; under a privileged one it altered the live database
  mid-request. Run `recall schema apply` with the migration role before starting the server.

* **BREAKING: the enterprise control plane refuses to start without
  `RECALL_SOURCE_POLICY_FILE` again**, as `docs/SECURITY_MODEL.md` says; a merge had dropped the
  refusal. Redis idempotency also changes: a replayed key spends one rate-limit token, and the
  request fingerprint is bound to the authenticated principal, so a key in flight across the
  upgrade needs a new key. A Redis-backed index byte debit, which was refused on every call
  ("attached to a different loop"), now works.

* **BREAKING: calibration refuses a query set that scores one query twice.** Copies that
  differed only in `relevant_ids` counted one cosine several times toward the certification
  floor; `calibrate` now refuses them, and a query labelled both answerable and unanswerable.
  Deduplicate the labelled set before calibrating. Also: a malformed bundle is a named refusal
  instead of a raw `OSError`, `UnicodeDecodeError` or `TypeError`; bundle numbers are type-checked
  so one that verified on import no longer fails every later read; a carry-forward chain exactly
  at the documented bound is allowed; `publish` locks the generation row against a concurrent
  forget; and `list`, `show`, `publish`, `export` and `import` on an unmigrated database report
  `SchemaTooOld`.

* Upgrade note for calibration import (#1095): `recall calibration import` now authenticates a
  v2 bundle, which its unkeyed checksums never did. It needs your own labelled query set
  (`--labels FILE`, the set the bundle was fitted on; its digest must equal the bundle's) and the
  embedder that wrote the generation's vectors (`--embedder`). Those queries are scored again on
  the named generation, one embedding call each, so a hosted embedder needs its credentials at
  import time. The bundle is refused if any recorded score differs from the fresh one by more
  than 0.01, if its certification does not hold on the fresh scores, or if your labels repeat a
  query. A carried bundle is judged at an error bound no looser than the default 10%, so a carry
  made with a looser `--max-error` still serves where it was made but no longer imports. A v2
  import without `--labels` or a matching `--embedder` is refused. In the library,
  `CalibrationRepository.import_bundle(path)` needs `labels=` and `embedder` (or `make_embedder=`)
  for a v2 bundle. Legacy files need neither and still import unbound. The lite fit now also
  refuses a labelled set that repeats a query.

* Upgrade note for calibration fitting (#1106): when rounding the midpoint down to the 0.001 grid
  would land on or below the unanswerable ceiling (a clean gap under about 0.002), the fitted
  threshold now sits strictly above the ceiling and at or below the answerable floor. It used to
  round down onto or below the ceiling, which confirmed the unanswerable samples sitting there.
  A v2 bundle fitted before this change on such scores re-derives a different threshold, so
  `recall calibration import` refuses it ("threshold disagrees"); recalibrate instead.

* **BREAKING: `RECALL_OIDC_JWKS_REFRESH_SECONDS` above 3600 refuses, naming the
  variable.** Every `as_of` tool argument without an offset is read as UTC, as
  `recall_reasoning_query` already did. Set a refresh of at most one hour, and pass `as_of` with an
  offset if it means local time.

* **BREAKING: an explicit glob index skips vendored folders.** `recall index` with a glob
  now prunes `.git`, `.venv`, `venv`, `node_modules`, `site-packages` and `__pycache__`, as the
  default walk does, so files inside them are no longer indexed. `recall check`, `fix`, `lint` and
  `rewrite` still walk them.

* **BREAKING: `recall quickstart --existing-dsn` no longer migrates an older install's
  `chunks` table.** Run `recall schema apply` on it first; the stack quickstart provisions itself
  still migrates. `recall demo` refuses `--table chunks`, and `recall demo` and `recall code`
  refuse while a source policy is configured.

* Upgrade note (#1149): `recall forget` and `recall_forget` refuse a name that matches more than
  one stored source, and list the exact names to pass. A tenant built as
  `docs/FIRST_CALIBRATION.md` describes stores each memo under its `recall index` path and its
  generation `file://` URI, so the short name (`notes.md`) is refused there; a script that forgets
  by it now exits 1. The forget help says so.

* **A single-tenant (stdio) server refuses a `tenant` argument that names another
  tenant**, as the authenticated path does; it used to ignore it and act on its own tenant.
  `RetrievalOverloaded` now reaches the client as a structured tool error with `reason` and
  `retry_after_seconds`, and an OIDC issuer whose `iss` ends in `/` is accepted with or without
  the slash.

* API correction for the trust failure codes: with `RECALL_DEPENDENCY_INVALIDATION=enforce`, a
  dependency projection read that raises is now `DEPENDENCY_UNAVAILABLE`, refused under every
  policy, instead of `DEPENDENCY_GRAPH_NOT_READY`. A generation with no projection is still
  `DEPENDENCY_GRAPH_NOT_READY`. Automation that retried on the second code for an outage should
  retry on the first.

* **PostgreSQL generations now refuse by name, as Lite does, a strict search naming an entity-like
  term absent from the generation's vocabulary** (`unknown_term`). This changes verdicts: it is on
  by default and `RECALL_UNKNOWN_TERM_GATE=0` disables it. Each `calibrate()` fit also measures
  synthetic near-miss coverage, stored as an audit event and reported on search results as
  `calibration_coverage`, and in `recall_calibration_status` as a new `near_miss_coverage` key and
  a coverage line appended to `message`; it never gates. Measuring it embeds up to about twice as
  many probe questions as the fit has answerable ones, which a hosted embedder bills.

* **`recall_reasoning_query` answers with dates by default.** The new `dated` answer profile
  gives the reader the question's date (`as_of`, else now), each evidence item's date
  (`valid_from`, else the chunk's first write) and a fixed set of reading instructions after the
  unchanged safety prompt. On 120 LongMemEval-S questions accuracy rose by 0.108 (95% CI 0.033 to
  0.183), and on LoCoMo by 0.185 (0.150 to 0.221). `RECALL_REASONING_ANSWER_PROFILE=plain`
  restores the previous prompt; an unknown value stops the MCP server and the CLI naming the
  variable. The profile is recorded in the response diagnostics.

* **numpy is a core dependency** (`numpy>=1.26`). A Lite store keeps its vectors in one numpy
  matrix, and pgvector 0.5.0 no longer installs numpy, so a core-only install failed on its first
  lite write.

* Every FastEmbed embedding profile now records the installed `onnxruntime` and `tokenizers`
  versions in its `dependencies` (`onnxruntime-gpu` and the other builds are named in the value), on
  the registered and the unregistered construction path alike. Before this an upgrade of either with
  fastembed unchanged, such as onnxruntime 1.30.0 to 1.31.0, left the profile fingerprint
  byte-identical, so the embedding cache, the index skip guard and the v1 calibration binding reused
  vectors computed by the old runtime or tokenizer. Every FastEmbed fingerprint changes once: the
  embedding cache re-embeds once, and existing FastEmbed index fingerprints and v1 calibration
  bindings need rebinding. Hosted and sentence-transformers profiles are unchanged.

* **Three identities move once.** An `st:<model>` embedder built from a local
  directory or a pinned revision now fingerprints the weights it loaded, so its fingerprint
  changes once. Hugging Face artifact digests use a versioned framing for new builds (the
  enterprise artifact digest is unchanged). Document extraction keeps complete, bounded text and
  its identity moves to v3, so earlier extraction results are recomputed.

* **Numbers with thousands separators match as numbers.** `1,500` is read as 1500, not
  1.500, and `1,234,567` as one number, on the indexed and the query side alike; any other comma
  stays decimal (`0,500` is 0.5). The structured document version moves to
  `table-row-groups-v2`, so the next `recall index` re-extracts every file and a new generation
  re-chunks (its pipeline fingerprint changes). Until a corpus is re-indexed, grouped figures in
  its tables get no exact-match boost.

* **Generation builds chunk Python files by symbol and anchor each chunk to it.** The `code` chunker
  is now `recall.index.CodeChunker`: a `.py` file splits on its functions, classes and methods, and
  each chunk carries `metadata["code_anchor"]` (`repo`, `file`, `symbol`, `code_sha`); a file that
  does not parse, and every other file, is chunked as before. The chunker version moves to 2, which
  changes the pipeline fingerprint, so a code tenant re-chunks on its next build. Text tenants are
  unaffected.

* **Voyage calls no longer import torch.** Embedding, contextualized embedding and reranking talk
  to Voyage through RE-call's own HTTP client (`recall._voyage_http`), which reproduces the
  SDK's requests and responses; `import voyageai` pulled in `transformers` and `torch` wherever
  they were installed. Measured on a host with torch: 46 s to 1.3 s and 559 MB to 52 MB to a
  constructed embedder. The `voyage` extra now declares `requests>=2.31`; `voyageai` stays
  installed because its version is part of every Voyage profile's identity. The multimodal
  embedder still uses the SDK.

* **A trusted search on the generation store makes fewer round trips.** Six statements per search
  instead of nine (one `set_config` for HNSW tuning, a cached generation binding and newest
  `indexed_at` per generation and corpus fingerprint); median 26.7 ms to 21.0 ms on one measured
  tenant. One `trusted_search` embeds each distinct query text once across its expansion
  re-searches; `recall_evidence` writes evidence cards on the store's own connection instead of
  opening a new one per call; a supersession `recall_related` reads no corpus when the seed has no
  edges; the atomic rescue parent costs zero or one query
  (`GenerationStore.scored_chunk_for_query`). The in memory evidence card store that
  `recall_evidence` filled for the server's life is gone.

* **Generation-scoped dense search is exact by design.** `GenerationStore.query_dense` now scans
  the active generation exactly instead of walking the shared HNSW index, unless
  `RECALL_GENERATION_DENSE_SEARCH=hnsw`. On a copy of a production table (753k rows across many
  tenants and retained generations), the filtered HNSW walk returned 0.875 of one tenant's true
  top 20 at the default tuning and 0.918 at the maximum `ef_search`, while the exact scan took 50
  to 276 ms. Production was already served the exact plan, but only through a cached generic
  plan; a planner statistic could have switched it to HNSW silently. Results now equal the exact
  scan's by construction. `RECALL_HNSW_EF_SEARCH_FILTERED` and
  `RECALL_HNSW_ITERATIVE_SCAN_FILTERED` now affect generation-scoped search only with
  `RECALL_GENERATION_DENSE_SEARCH=hnsw`. That value ignores case and surrounding spaces, and empty
  means `exact`; anything else refuses MCP server startup, naming the variable.
  `scripts/check_hnsw_recall.py` measures the `hnsw` path whatever the variable says.

* **`recall_apply_fact` no longer reads every fact current in the tenant.** On a ledger with a
  key-scoped read (lite and PostgreSQL), the controller decided on the claim's conflict key but
  first read and discarded the tenant's whole `current()`, which decoded every current fact. The key
  read is now the read a write fails closed on, so a ledger that cannot answer it still refuses with
  `ledger_unavailable` and appends nothing, and an apply no longer grows with the number of facts
  current. A ledger without a key-scoped read is still read through `current()` first.

* **A generation build refreshes planner statistics.** `GenerationManager.build` ends with a best
  effort `ANALYZE` of `recall_chunks_v1 (tenant_id, generation_id)`, so validation, calibration and
  serving plan against the new generation's real row count (estimated 1 row before, 5,026 of 5,176
  after, in the measurement); it adds about 267 ms to a build and never fails one.

* **The `re-call` skill focuses on applying evidence.** The skill shipped with the Claude Code and
  Codex plugins follows a two call discovery and source scoped evidence path, and carries a
  `references/tool-routing.md` map of every tool; `recall setup` now installs the whole skill
  directory, not only `SKILL.md`.

* **The sdist holds only what a build needs.** It is now built from an allowlist: the shipped
  packages, the plugin trees, the README, the changelog and the licence files. From 0.11.0 to
  0.14.0 it packed every tracked file, research traces included (91 MB against a 3.5 MB wheel). A
  wheel built from the new sdist is byte identical to one built from the tree. The sdist no longer
  contains `tests/` or `docs/`.

* **Prompt time memory hits carry the dates each memo states about itself.** The `UserPromptSubmit`
  hook starts each hit with a stamp such as `[written 2026-09-24; modified 2026-09-24; corrected 3x,
  last 2026-09-25]`, or `[undated]`, and the block states today's date. Dates come only from the
  file name, the frontmatter `valid_from` and `modified`, and dated correction markers; dates in the
  body and file modification times are never used.

* **The prompt-time hook no longer re-reads the whole memo store on every prompt.** Each
  memo's parsed result is cached by file name, modification time and size, so only changed memos
  are read again. (The PR measured 1.2 to 2.8 s per prompt before on about 1,350 memos.)

* **`recall lint` warns `closure-marker-unlinked` only when the marker's sentence names a memo.** A
  "replaces" or "supersedes" now needs a memo reference (`[[link]]`, `](name.md)` or a bare
  `name.md`) in the same sentence, and a marker inside a reference is skipped. A memo that only says
  "deprecated", or describes its successor without naming it, is no longer warned about.
  `CLOSURE_MARKERS` and `recall check` are unchanged.

* **Importing one `recall` module no longer imports the whole package.** `recall/__init__.py`
  resolves its re-exports on first access (PEP 562), so importing a module such as `recall.errors`
  no longer loads about fifty modules, psycopg and pgvector (about 1.2 s). `__all__`, `__version__`
  and `recall.<submodule>` access are unchanged.

* **`recall_search` advice lets a superseded or expired hit answer a question about the past.** When
  a `superseded` or `expired` hit scores at or above the calibrated threshold, the advice ends with
  a note: for a question about an earlier time, answer from that hit, cite its `chunk_id` and say it
  no longer holds. The superseded advice now says to rely on the current version for the present
  state.

* **The `recall_search` description and its citation advice no longer name `recall_evidence`.** A
  tool named there was called by small models on nearly every search, abstentions included; what
  `recall_evidence` adds is stated in its own description.

* **`recall provenance current` lists what `recall_current_facts` lists.** Its JSON gains `sources`
  and `served_state` (`unchecked`, since the CLI reads the ledger without a store), cites the card
  of every memo that restated a fact, and moves a fact that conflicts with another, with no
  supersession between their memos, from `facts` to `contested`.

* **Forgetting a memo keeps retired what it superseded, in the document graph as in the fact ledger;
  the new `recall withdraw` / `recall_withdraw` gives it back** . A delete used to remove a memo's
  `supersedes:` claim with its chunks, so the memo it replaced was served as current again (a
  privacy forget of "moved to Rome" brought Paris back, and forgetting the middle of a chain left
  two current versions), while the fact ledger already kept it retired. Every delete now writes a
  supersession record (names and dates, no text) to the audit ledger in the same transaction:
  `recall_forget`, `recall forget`, the lite memo tools, `GenerationManager.forget` on a generation
  tenant, the indexer's prune of a file gone from disk, and a source a promoted generation leaves
  out all keep the memo's claims standing as a ghost while no memo by its name is indexed. Withdraw
  voids them and splices the chain; in the ledger it appends a `withdrawn` event (migrations 0032
  and 0033) that also undoes the supersession the fact made. **Deploying needs
  `--with-migrations`.**

* The deterministic proposal rules behind `recall reasoning proposals` and `recall_rewrite_plan`
  pair memos by different keys. A `status:` or `verdict:` line names a state, not a decision
  subject, so two unrelated memos that both say "Status: done" are no longer paired, and a memo
  whose only such line is a status now takes its file name (without folder or version) as its
  subject, which the repeated-subject and contradictory-windows rules group by (so `flag_v1.md`
  "Status: enabled" and `flag_v2.md` "Status: disabled" are now compared). A version is `v` and
  digits after a separator, so `ipv4.md` and `ipv6.md` are not two versions of one memo. The
  repeated-subject rule proposes only when both memos are dated and the later one starts strictly
  later. Temporal ordering now groups by the file name without its version, in its folder, so it
  pairs a file's versions whose decision lines are worded apart, and it no longer proposes again
  an edge the repeated-subject rule proposes. The Session 3 control artifact drops its two
  `temporal_ordering` proposals (referrals 4 to 2), and every other proposal id is unchanged.

* **The deterministic proposal rules read a folder as the scope of a name.** A version series is
  one folder's files (`a/plan_v1.md` and `b/plan_v2.md` are no longer two versions of one plan),
  and a decision subject, or the file name a memo with no decision line falls back to, is shared
  only inside one folder, so two folders' status-only `decisions.md`, or two teams'
  `decision: retention`, are neither superseded nor contradicted by each other. Within a folder
  the rules pair more in two places: `plan_V2.md` is a version, as `plan_v2.md` is, and two
  memos with the same start now both get a later third as successor instead of one of them.
  Two memos that share only a file name are proposed by temporal ordering (0.64) rather than as a
  repeated decision subject (0.72). When the version and date rules emit exact opposite pairs,
  the version proposal is `requires_review` and both proposals say the two disagree; a
  disagreement across a longer chain is not detected. Version-only filenames such as
  `a/V1.md` and `b/V2.md` also retain their folder scope despite their empty versionless base.
  The rules' revision is now `session3-v3`, which changes every deterministic proposal
  id; the Session 3 control artifact keeps its six proposals and its scores under new ids.

* **A successor hides what it supersedes only while the successor itself is valid.** A memo
  `supersedes:` another from next month leaves it current until then, an expired time-bounded
  successor gives its predecessor back, and a correction for 2020 to 2022 leaves today's memo alone;
  a successor whose window cannot be read still applies at every instant. The same rule holds in the
  fact ledger, where a successor now hides the ASSERTIONS recorded before it, so a value asserted
  again after its successor (Paris, Rome, Paris) is current again, and in memo-level retirement of
  stated facts, which also follows the chain search follows. Search, `recall_current_state`, graph
  expansion, related evidence, the reasoning graph and planner, the semantic graph and the
  dashboard's corpus page now resolve `supersedes:` with one rule
  (`recall.supersession.ReferenceIndex`): a reference naming a folder by its root-relative path, any
  reference by its stem, case-insensitive and NFC, a self-claim dropped, and two roots holding one
  path kept apart. A bare name is never matched by path, so several files sharing its stem stay
  ambiguous. Semantic lint folds references the same way but matches by stem alone; `recall lint`
  and the dashboard's graph and triage pages keep their own matcher. A low-scoring successor whose
  dependency is invalidated is no longer promoted to `ok`. In dependency invalidation each root's
  copy of a path is its own document, and a `depends_on` reference resolves to the dependent's own
  root's copy first, else to the one copy outside an upload staging root; several such copies, or
  only an upload's, leave it unresolved, which invalidates the dependent. A second value of a
  set-valued predicate such as `email` is no contradiction, while a fact declaring a cardinality its
  rival does not share still conflicts; facts are NFC-normalised when built; an authored link can
  supersede an expired or future value it overlaps; and a value asserted again that names a
  successor of its earlier assertion is recorded, not dropped as a DUPLICATE. New API:
  `SupersessionGraph` with each store's `supersession_graph()`,
  `recall.supersession.store_supersession_graph` and `stored_supersession_graph`,
  `AtomicFact.cardinality` and `recall.provenance_controller.predicate_cardinality`.

* **The supersession arbiter checks direction with the model.** A pair that passes the gate gets a
  second question, which note holds the more current version, and is proposed only when the model's
  answer, backed by a verbatim quote, matches the metadata's newer note. Direction from `modified:`
  stamps put living memos backwards: on one memory store, 9 of 99 directed pairs. The check stopped
  8 of those 9 and dropped 1 of the 90 correct pairs, for one extra call per proposal.
  `RECALL_ARBITER_DIRECTION_CHECK=0` turns it off. Cached answers from earlier runs stay valid.

* **`supersedes:` holds several references.** One memo may replace two: write `supersedes: [a.md,
  b.md]`, a block sequence (`supersedes:` then `- a.md` lines), or the key repeated, as Validity
  Frontmatter 1.0 section 5 allows. Every reader sees every reference: the store's supersession
  scan, the reasoning graph, current state, `lint`, `rewrite verify`, `lint --fix`, `rewrite apply`
  and the supersession arbiter. Before this, a block sequence read as "supersedes nothing", a
  repeated key kept only its last line, and a memo that already declared one edge refused a second.
  A memo that declares one reference keeps the plain string it always had, so existing chunk
  metadata does not change.

* **A `supersedes:` reference resolves the same way in every layer.** Search resolves a reference
  that names a folder by its path, and a bare name only when no other memo shares it; `recall
  rewrite`, `recall lint`, `recall lint --fix`, the dashboard editor and its review queue now use
  the same rule (`recall.supersession.resolve_reference`). So `supersedes: a/notes.md` is no longer
  an error beside `b/notes.md`, `ambiguous-supersedes-target` is reported only when search finds the
  reference ambiguous, `recall rewrite` lists the candidates when it refuses a shared bare name, and
  `plan_rewrite` refuses a value that names more than one memo before it is stored. The cycle check
  keys a memo by its path where another memo shares its name, so a memo superseding its namesake in
  another folder is no longer refused as superseding itself, and a chain through two same-named
  memos is no longer reported as a cycle. A flat corpus keys and names every memo as before. The
  review queue keys a claim the same way, so claims about `a/notes.md` and `b/notes.md` are two
  items and rejecting one no longer hides the other; a rejection recorded under the old file name
  key still applies.

* **An agent can ask which memo a `supersedes:` reference names before it writes it.** The new MCP
  tool `recall_resolve_reference` answers by search's own rule on the served index: `resolved` with
  the exact value to write, `ambiguous` with the candidates, `dangling` or `self`. The
  `recall_index` receipt lists stored references that name more than one memo; it is advisory and
  runs after the index has committed. Under a source policy only readable memos are named or
  counted, so neither reveals that an unreadable memo exists. The re-call skill routes the tool.

* `RECALL_TRUST_MODE=development` under `RECALL_ENV=production` now logs a warning naming both
  variables when the policy is resolved (at MCP server startup, and on a `recall` search); the
  relaxed gate is still honoured. An unknown `RECALL_DEPENDENCY_INVALIDATION` value is now refused
  when the MCP server or a `recall` command starts, instead of on the first search.

* **A stdio server ignores a leftover `RECALL_AUTH_TOKENS_FILE`**, with a warning, instead
  of refusing to start under `RECALL_ENV=production`; HTTP still refuses it in production. The
  DNS rebinding allowlist is the lowercased host of `RECALL_AUTH_RESOURCE_URL`, with or without
  its default port, and the new `RECALL_ALLOWED_HOSTS` adds `host[:port]` names for an in-cluster
  service or a proxy that rewrites `Host`; a malformed entry refuses at startup.

* **The installers check their prerequisites before writing anything.** `recall wizard`
  and the desktop installer first check for the Microsoft Store Python, Docker, Docker Compose v2
  (at least 2.3.3), virtualization and RAM, with no network call; an unreadable answer does not
  block, and an install onto an existing database skips the Docker checks. A set
  `RECALL_EMBED_PROFILE` now reaches every registered server block. `recall setup --lite` with
  neither `--memory` nor `RECALL_INDEX_ROOT` refuses the home folder or a filesystem root.

* **Reasoning queries see more of the graph and stay in their source.** The temporal graph
  budget is 12 nodes instead of 8 (the 8 trusted seeds used to fill it), and it is also the
  planner's evidence cap for temporal queries. A source-scoped reasoning query no longer admits
  graph neighbours from other sources. The desktop compose stack keeps the paid answer provider
  opt in (`RECALL_REASONING_ANSWER_ENABLED`, default 0).

* **`recall setup` withholds the two heaviest local models on machines under 8 GB of RAM.** The
  bge-large embedder (~1.2 GB) and the bge reranker (~1.1 GB) stay in their menus but cannot be
  picked when total memory is under 8 GB, or when it cannot be read, and the note says which. The
  ms marco reranker and bge-base stay selectable. Picking a withheld embedder asks again rather
  than falling back to `hashing`. An 8 GB machine counts as 8 GB although its OS reports a little
  less (`RAM_REPORTING_SLACK_BYTES`, 1 GiB). This brings back `COMFORTABLE_RAM_BYTES` in
  `recall.wizard.probe`, removed below as unused, now read by `heavy_local_models_are_feasible`;
  `HardwareProbe` gains `total_ram_bytes`. The 4 GB floor, where the install is refused, is
  unchanged.

### Removed

* **BREAKING: the benchmark service package `recall_aml` is no longer distributed.** The wheel
  and the sdist no longer carry `recall_aml`, the `hosted` extra or the `recall-hosted` console
  script, which served one benchmark deployment and are not needed to use RE-call. No shipped
  module imports it, and its source stays in the repository. `pip install "recall-rag[hosted]"` now
  only warns that the extra does not exist. The `voyage` extra now declares Pillow, which
  `recall.multimodal` uses and which only `hosted` declared; Voyage already installed it.

* **The QNLI entailment judge is gone (breaking).** It was an optional, off by default stage that
  demoted hits a `cross-encoder/qnli-distilroberta-base` model judged as not answering the query.
  Measured as an evidence admission stage on a coding retrieval screen of 2026-09-19 it cut present
  complete coverage from 0.92 to 0.42 and MRR from 0.50 to 0.36, and raised Search p95 from 0.5 s to
  37 s, and no deployment ran it by default. Removed with it: `recall.entailment`, the `recall
  search --entail` flag, the setup wizard's judge question, the desktop app's "Local entailment
  judge" option, the `entail` extra, the `RECALL_ENTAILMENT`, `RECALL_ENTAILMENT_MODEL` and
  `RECALL_ENTAILMENT_REVISION` variables, the `not_entailed` verdict, the `entailment` key in
  per-stage timings, the `entailment=` parameter of `trusted_search` and of the LangChain and
  LlamaIndex retrievers, and the `recall.eval` LoCoMo entailment sweep. **A configuration that still
  sets `RECALL_ENTAILMENT` to anything but a false value now refuses**, naming the variable, at
  `recall search`, `recall demo`, `recall code`, `recall quickstart` and MCP server startup, and
  `recall setup` and `recall wizard` refuse to rewrite a `.env` that still holds it, because
  silently dropping a stricter filter would widen what that deployment is served. A value that is
  neither true nor false refuses too, as it did before. To upgrade, delete the `RECALL_ENTAILMENT`
  line from your `.env` or environment, or set it to `0`; `RECALL_ENTAILMENT_MODEL` and
  `RECALL_ENTAILMENT_REVISION` are ignored and can be deleted. A false value (`0`, the value `recall
  setup` used to write) or an unset variable is accepted. `pip install "recall-rag[entail]"` now
  only warns that the extra does not exist and installs nothing extra; install `[rerank]` if you
  need `sentence-transformers`. Committed result artifacts that measured the judge are kept
  unchanged.

* **pypdf is no longer installed by the `documents` extra.** It was never imported: PDF text has
  always come from pdfplumber, which does not depend on it. It stayed only because a capability
  check confirmed it was present, and 6.16.1 carried eight denial-of-service advisories
  (CVE-2026-102993 to CVE-2026-102999 and CVE-2026-103000). The capability check and the optional
  import contract drop their pypdf row (both already checked pdfplumber), and the generated
  Dockerfile's post-install import now asserts `pdfplumber` instead of `pypdf`, so a missing
  `documents` extra still fails the image build. Nothing extracted changes.

* **Unused definitions that nothing in the tree referenced:** `PerformanceTrace.set_if_absent`
  (`recall.observability`), `RuntimeSecretState` (`recall.ops.secrets`), the `BaseState` type alias
  (`recall.dependency_invalidation`), and the `COMFORTABLE_RAM_BYTES` constant
  (`recall.wizard.probe`). The constant was documented as withholding heavier install options
  below 8 GB of RAM, but no code ever read it; only the 4 GB `MIN_RAM_BYTES` floor is enforced.

### Fixed

* **Applying a fact on a generation store no longer refuses every evidence card.**
  `GenerationStore.generation_id` returned `"legacy"`, so `recall_apply_fact` built its controller
  for a generation that does not exist and refused each card as `GENERATION_MISMATCH`. It now names
  the generation the store reads (pinned, fixed, else active), and MCP search reads it inside the
  search's own snapshot, so `index_generation` always names the generation the hits came from.
  `recall_current_facts` reports `generation_id: null` before any generation is promoted. Nine
  inherited methods that address the legacy chunk table (`migrate_schema`, `delete_tenant_data`,
  `delete_sources_across`, `upsert_sparse` and others) now refuse by name on a generation store
  instead of failing in SQL, and `recall-enterprise readiness` builds its embedder as the server
  does.

* **`recall_apply_fact` reads one generation for the whole call**, so a promotion that
  lands mid-call no longer refuses a valid card as `GENERATION_MISMATCH` or records a fact under a
  generation other than the one its content was checked in.

* **Forgetting a memo retracts the facts it was the only evidence for.** `recall_forget`, `recall
  forget` and the lite memo tools append a `retracted` event (`SOURCE_FORGOTTEN`) for each fact
  every one of whose evidence cards names a forgotten memo, so it leaves `recall_current_facts`; a
  fact with a surviving memo stays, and a superseded fact never becomes current again.
  `ForgetResult.facts_retracted` counts them, and is -1 with a warning when the retraction failed
  (running the forget again completes it). The value stays in the append-only ledger and in the
  evidence cards, as the `recall forget` help and the `recall_forget` description now say. Migration
  0026 admits the event; a lite ledger is rebuilt in place on first use. **Deploying needs
  `--with-migrations`.**

* **A fact whose memos are no longer served is not current.** A memo pruned from disk or left out of
  a promoted generation hides its facts from `recall_current_facts` and from the conflict check of
  `recall_apply_fact`, and they return with the memo after a `generation rollback` or a restored
  file. Nothing is written to the ledger.

* **`recall generation gc` retracts for good the facts no remaining generation can bring back.**
  When gc deletes the last generation holding every memo behind a fact, it appends `retracted`
  events coded `SOURCE_REMOVED` and `recall generation gc` prints how many; a failed retraction
  never fails the gc, and the next gc completes it. Migrations 0027 and 0028. **Deploying needs
  `--with-migrations`.**

* **`recall generation gc` no longer retracts a fact that a remaining generation
  still serves**, which happened when a manifest named the memo by a different relative path. An
  indexed file replaced by a directory is now pruned, so its old rows stop being searchable.

* **Withdrawing a fact's first memo no longer gives back what a forgotten second memo retired.**
  A fact stated by two memos is one application recorded APPLIED and one recorded DUPLICATE, and
  projections listed only the APPLIED one. Forgetting the second memo and then withdrawing the
  first made the value the fact superseded current again, where the other order kept it retired.
  When the application a DUPLICATE repeats is withdrawn, the DUPLICATE now stands in for it, with
  its supersession and at its point in the ledger, so it hides exactly what the withdrawn one hid
  (`recall.provenance_controller.applications`), on PostgreSQL, lite and in memory, and in the
  current-facts tables. A retraction needs no stand-in: it takes every application of the fact.

* **A request id names one application.** A request id already used for another fact, or for the
  same fact on other cards, was answered allowed `DUPLICATE` with the earlier event and recorded
  nothing; it is now refused `REQUEST_ID_REUSED` and not recorded, while the same request sent again
  is still an idempotent `DUPLICATE`. A request id of a form the ledger mints for its own events
  (`<id>:supersession`, `retraction:<event id>`, `stated:<hash>`) is refused `REQUEST_ID_RESERVED`:
  PostgreSQL, SQLite and lite answered it `DUPLICATE`, and the in-memory ledger appended a second
  row under it. Neither code is ever written to the ledger, so no migration is needed.

* **A memo that states a fact through a DUPLICATE is one of its sources.** A new value whose memo
  links it now supersedes the fact, where it was refused `CONTRADICTION_WITHOUT_SUPERSESSION`, and
  `recall_current_facts` cites in `evidence_card_ids` the live cards of every application of a
  fact, as `sources` already did, so a first memo the store no longer serves is no longer cited.

* **An empty validity window is invalid metadata.** A memo whose `valid_from` is after its
  `valid_until` is verdicted `invalid_metadata`, as Validity Frontmatter 1.0 requires (test vector
  4), where search reported `expired` or `not_yet_valid`. `recall index`, generation builds and the
  lite refresh refuse it as they refuse a malformed date, `recall lint` reports it as
  `invalid-date`, and the lite memo writer, `recall rewrite apply` and the dashboard refuse to write
  one. A one-day window (both keys the same date) stays valid all day.

* **A fact applied under several requests is listed once.** Applying a fact that is already in
  force records a DUPLICATE event, and `recall_current_facts`, `recall provenance current` and the
  controller's conflict check listed that event beside the original, so the fact appeared once per
  request. Worse, a replacement that linked the original's evidence was then refused with
  `CONTRADICTION_WITHOUT_SUPERSESSION`, because the one fact counted as two conflicts. The current
  and candidate projections now skip DUPLICATE events on every ledger (PostgreSQL, SQLite and in
  memory); the events stay in the ledger.

* **A generation rebuild no longer re-dates the passages it rewrites (#835).** A generation-backed
  store served `indexed_at`, the build that wrote or copied a row, as every hit's
  `first_indexed_at`, and dated supersession edges the same way, so `known_as_of` replay and the
  dated answer profile treated the last rebuild as every passage's first write. Every hit path and
  supersession edge now reads the stored `first_indexed_at`. A build that rewrites a source, because
  its bytes changed or the pipeline fingerprint moved, now gives each chunk the earliest
  `first_indexed_at` any active, ready or retired generation holds for the same source and chunk
  ordinal; an ordinal none of them holds is dated at the build. Rows an earlier rebuild already
  stamped with its build time are not repaired: a later build corrects one only while a generation
  written before that rebuild is still retained (by default `gc` keeps the two most recent retired
  generations and any younger than seven days), and after that it carries the build time forward. No
  backfill ships with this change. A source whose bytes are unchanged but which a changed chunker
  splits into more chunks gives its new ordinals the earliest `first_indexed_at` held for those
  bytes.

* `recall forget` (and `recall_forget`) on a generation tenant without the enterprise
  control-plane tables no longer logs an ERROR and warns that the erased text "may be restored".
  `recall schema apply` does not create the migration outbox, so there is nothing pending to
  scrub; the scrub now reports 0 there.

* `recall doctor` on the generation route (`RECALL_ENV=production`, or
  `RECALL_INDEX_MODE=generation`) judges a tenant with no legacy rows by its active generation,
  and reports that generation's calibration status from the database, the one the server serves
  by, instead of `calibration.json`. It used to FAIL "holds NO chunks" on such a tenant while it
  served trusted answers. The legacy route is unchanged.

* `recall calibration auto` after a forget now names the calibration the forget superseded and
  the labelled refit (`recall calibration calibrate --generation ... --publish`), instead of
  reporting that no certified calibration was ever published.

* Calibration fitting (#1106) now refuses a sample that is not a finite number. A NaN top cosine,
  which a zero-norm query vector produces, used to move the fitted threshold silently.

* **A non-finite calibration value cannot certify an answer.** The trust gate treats
  a non-finite threshold as low confidence (a NaN threshold verdicted any finite score `ok`);
  `Calibration` refuses a non-finite threshold or a non-positive scale, and maps a non-finite
  cosine to 0.0. Calibration JSON with an enormous sample count degrades to unknown instead of
  raising `OverflowError`.

* **`recall calibration served-gap` refuses inputs that invert it.** A negative or NaN
  `--tolerance` is refused, and the report is withheld when a promotion lands mid-run.

* **`recall setup` no longer drops an existing table it could not see into.** It counted
  the rows of an existing relation, and dropped it when the count was zero, although rows hidden by
  forced row-level security, or committed after the count, were still there; recreating a relation
  also lost its explicit serving grants. Setup now refuses an existing relation of another
  dimension before any destructive step and names a matching embedder or a fresh target.

* **`recall setup` and `recall wizard` keep what
  they did not write, across reruns and failures.** A rerun or a failed start no longer removes
  other projects, replaces an existing serving route, or drops service environments and image
  recipes. `recall wizard` no longer rebuilds the `# recall setup` block of `.env` from
  `RECALL_DSN` and `RECALL_EMBEDDER` alone, which deleted the provider keys and the other managed
  settings. A resumed run checks that a recorded promoted generation still exists. Uninstall
  removes only this install's registrations and only a `Dockerfile` it wrote, and keeps the
  `recall setup --lite` server. Setup no longer imports project-local hook modules, ingests
  another project's memory, or ignores the selected database and table; refuses dotenv values
  holding a line separator; refuses a missing embedder or `mcp` extra before provisioning; keeps
  the pre-RE-call `~/.claude.json` backup once and never replaces it; reclaims unfinished
  generations after a failure and prints recovery commands that work; and `recall index` and
  `recall lite refresh` print their refusals instead of a traceback.

* **`recall doctor` and server registration agree with the setup they check.**
  `recall doctor` handles a malformed client config and equivalent project paths, and reports the
  relation PostgreSQL actually resolves. MCP registration pins the selected table, `chunks`
  included, setup forwards a custom table, and registering without the `mcp` extra is reported as
  an install failure naming the extra. The calibrated server block pins
  `RECALL_INDEX_MODE=generation` and sets `RECALL_INDEX_ROOT` to the memory folder.

* **An empty `RECALL_SERVING_DSN` falls through to `RECALL_DSN` in the CLI**, as it does in the MCP
  server; the wizard writes it empty into every server block. `recall wizard` refuses a global
  `--embedder`, `--serving-dsn` (`--dsn`) or `--migration-dsn` it would ignore. On PostgreSQL,
  forget and withdraw look up only the named sources instead of listing every source of the tenant.
  The desktop configuration page's SPLADE check probes only the hardware and CUDA memory it reads,
  not Docker, WSL and `systeminfo`. The `recall index --glob` help no longer offers `**/*.py` for
  code chunking, which never applied there; it names `recall generation build --chunker code`. Setup
  matches an existing Claude Code project entry only when its last folder name matches the
  project's, so a differently named junction or symbolic link to the project is no longer found.

* **MCP servers registered by `recall setup` start under a client that passes only the registered
  environment.** On Windows such a launch lost the user site-packages (`No module named
  'recall_mcp'`) and the socket stack (`WinError 10106`). Each registered entry now carries
  whichever of `APPDATA`, `LOCALAPPDATA`, `SystemRoot`, `TEMP`, `TMP`, `USERPROFILE`, `HOMEDRIVE`,
  `HOMEPATH` and `HOME` are set, plus `PYTHONSAFEPATH=1`, and never `PATH`.

* **Everything the MCP server builds reads its settings
  snapshot.** The dual-write shadow embedder, an env-less `index_memory`, the lifespan and indexing
  routes, Voyage reranking and the Agent SDK's paged evidence used the process environment, so a
  key resolved from Secrets Manager (`VOYAGE_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`)
  could be missing or different there.

* **A refused write gives back its idempotency reservation.** An erasure refused
  before its first write (no sources, a policy or tenant refusal, an ambiguous name) and a
  `recall_apply_fact` that raises (a busy lock, for example) release their reservation, so a retry
  is not told the change "may have committed". Startup refusals after the schema check close the
  limiter, store and registry. A token in its last clock-skew window is no longer validated and
  then refused.

* **On PostgreSQL a forget waits for at most one chunk of a stated-facts reconcile.** The reconcile
  holds the erasure lock per 100 statements, re-checking which memos are still served, where a
  forget waited for a whole first activation and gave up after 300 s. The lock's session sets its
  own lock and statement timeouts. Each hold also ends once it has run 30 s, after the statement in
  hand, so a large replay ledger no longer holds the lock past a forget's wait.

* **Supersession cache freshness and latest-state reads no longer scan a tenant's whole
  removal history.** Migration 0034 adds a per-tenant revision table and 0035 an index on the
  audit ledger, built CONCURRENTLY. **Deploying needs `--with-migrations`.**

* **One memo whose root is spelled two ways is one memo.** A row spelling its root in NFD
  or as a `file://` URI no longer makes supersession resolution report the memo as collided
  copies.

* **A hand-edited `file://` manifest entry in uppercase hex no longer changes the corpus
  fingerprint**; the version id is stored as its lowercase digest.

* **Erasure retires every cache keyed on the generation.** After `GenerationManager.forget` the
  supersession closure, graph readiness, calibration verdict, reasoning projection and semantic
  graph kept serving the erased source until restart; they are now keyed on the generation and its
  corpus fingerprint.

* **Deleting a generation's semantic graph retires its readiness marker**, which kept reporting
  the deleted graph as ready with its old counts. `recall_current_state` refuses an oversized
  projection before building it and no longer reads chunk text.

* **The reranker is loaded once per process again.** Since 0.14.0 every search, evidence and
  reasoning call re-verified and reloaded it, and dropped the routed profile.

* **`recall_related` and `recall_query_construction_challenge` report a trust refusal** as a
  tool error, as `recall_search` does, instead of a bare "Error executing tool".

* **An MCP write refused for a missing `idempotency_key` says so.** Under a limiter that requires
  idempotency (the Redis rate limit backend) the refusal is now `IdempotencyKeyRequired`, a tool
  error the client receives in full, where MCP reduced it to "Error executing tool".

* **`explain=true` again returns the memory audit.** `recall_search` and `recall_evidence` put
  the count only audit under `explanation.details.memory_audit`, which a server rewrite had left
  without a caller. Its `context` carries the same four keys on both tools.

* **The answer prompt names JSON, so OpenAI models can answer.** Every answer client that sends
  `response_format: json_object` was refused with HTTP 400 by OpenAI and Azure, because no
  message contained the word "json". The prompt digest label moves to `recall-answer-provider-v2`.

* **`recall_related` applies supersession in every relatedness mode** and matches the
  complete source identity before its SQL limit, so a stale or unrelated document no longer enters
  its results. Graph components load from one PostgreSQL snapshot during a concurrent erasure.

* **Reasoning queries no longer fail on concurrency, budgets or a stray `0000`.** Concurrent
  requests no longer raise `KeyError` in the shared graph validation cache; a standalone `0000`
  token (for example an entity label `v0000`) no longer fails every graph-first query on a tenant;
  the planner enforces its wall-time budget after its last step; and trusted chunks without file
  metadata match the graph's source fallback. A source scoped graph-first query filters neighbours
  by source before the candidate budget, a graph refused for a stale readiness marker re-reads it so
  a rebuild takes effect without a restart, graph expansion stays bound to the seed retrieval it
  expanded, and a graph failure before the trust gate reaches `GRAPH_PROVIDER_ERROR` with the seed
  kept. An unpadded date no longer moves `valid_from` or `effective_at` to the end of its day, and
  `valid_until` runs to the end of its day only for a bare date. The strict response decoder keeps
  each item's authority and verdict, raises `EvidenceValidationError` for a malformed `trust_state`
  or date, and refuses a citation that names no item of the trusted evidence. An authored file
  entity no longer replaces the document's own file identity.

* **A rescued parent is scored as a chunk.** Atomic rescue passed the 24 word view's cosine to the
  trust layer, whose threshold was fitted on chunk cosines, so a rescue could jump the top five or
  turn an abstention into an answer. The parent is now scored with its own cosine, and a parent the
  generation cannot score is refused. View products run under a two thread BLAS limit
  (`threadpoolctl>=3.1` joins the `atomic` extra), which cut p95 from 202 ms to 55 ms under a CPU
  quota. The window rank gate also refuses a rescue whose gate score is not a finite number, where a
  NaN margin admitted it. Dense placement leaves a selection already in the top five unchanged. A
  cold artifact load runs outside the cache lock, once per path, and `threadpoolctl` is checked at
  load rather than on every search. The artifact writer fsyncs its members and directories, loads
  the staged artifact before publishing it, and gives the published directory the umask's mode
  without ever setting the process umask; an unreadable `.npz` member is refused. The expectation
  cache is keyed on the receipt's hash, so a receipt rewritten in place is read again.

* **Atomic rescue stays out of scoped searches.** Document, structural and successor expansion
  re-searches re-ran the selection per source, could insert another source's parent, and raised
  `AtomicRescueSelectionError` for a source with fewer than five dense hits.

* **A long-lived server keeps at most two atomic rescue artifacts loaded**, where it kept
  every one it had loaded as generations changed.

* **An MCP server's first search no longer peaks at about 820 MB.** Loading the atomic rescue
  matrix validated it with whole-matrix `np.isfinite` and `np.linalg.norm`, the norm building a
  temporary as large as the matrix, in every serving process. The checks now run over 16 MiB row
  blocks, in the same order and with the same errors. Measured on the memory corpus with
  identical search results: first-search peak 820 MB to 509 MB; the resident set after the first
  search was about 13 MB higher (494 to 507 MB).

* **Voyage requests time out.** The Voyage embedder, multimodal embedder and reranker had no
  timeout, so a hung socket blocked a query indefinitely; they now use
  `RECALL_VOYAGE_TIMEOUT_SECONDS` (default 60).

* **Voyage requests stay within the provider's limits.** `voyage-context-4` parts and requests are
  bounded to 90% of the 32,000 token window, counted with Voyage's tokenizer (downloaded from
  Hugging Face on first use, else a byte bound), so a token dense PDF no longer aborts a
  generation build. A text batch refused with HTTP 400 is halved and retried instead of failing
  every attempt. `VoyageMultimodalEmbedder` packs requests under 1,000 inputs and the token cap,
  fits images to 16M pixels and 20 MB, and logs `voyage_multimodal_input_over_context`. No stored
  vector changes.

* **fastembed keeps its batch bound.** A `TypeError` raised while embedding was caught by the
  guard for `batch_size=`, and the whole list was re-embedded at 256 silently. Whether the encoder
  takes the argument is now read from its signature; an encoder that does not still embeds, with a
  warning that `RECALL_FASTEMBED_BATCH` is ignored. Indexing also no longer grows quadratically in
  the number of files, a rebuild no longer extracts an unchanged PDF before deciding to reuse it,
  and a source's chunks go in one batched insert.

* **Two processes opening a new embedding cache at once no longer crash.** Both could find the
  `used_at` column missing and add it, and the second failed at startup with `duplicate column name:
  used_at`. The cache now checks, takes the write lock and checks again.

* **`RECALL_EMBED_CACHE_MAX_MB` holds across processes.** The bytes written since the last sweep
  are kept in the cache file, so short lived instances add to one total; a cache already over its
  cap is swept on its next write.

* **The MCP `recall_index` tool now uses the shared embedding cache.** It was the one indexing entry
  point that passed no cache, so every call re-embedded unchanged text. The Agent SDK's
  `recall_index` goes through the same path and is fixed with it. `RECALL_EMBED_CACHE` moves or
  disables the cache as before.

* **The embedding cache and schema checks are sturdier.** A failed cache write rolls back
  instead of holding its transaction open, a fresh hit takes no write lock, eviction walks the
  `used_at` index, and a missing home directory disables the cache instead of failing the run. The
  Voyage width probe retries a transient error. `recall schema status` runs the same drift checks
  as serving startup and reports two migration files sharing a number. A bad `RECALL_EMBED_THREADS`
  is reported and a padded `RECALL_ENV` is read as the serving route reads it.

* **Explicit context groups keep their documents together.** Indexing preserves complete
  groups and puts vectors back in source order for the active and shadow encoders, and an erased
  member no longer lets its siblings reuse vectors computed with its text.

* **Indexing keeps what was asked for.** Unversioned machine annotations are
  excluded from authored evidence and contaminated cached chunks are rebuilt once; indented JSON
  is accepted beside JSONL; single-file indexing and reused generation chunks keep the requested
  provenance; custom statuses match whole values; and a failed LibreOffice profile allocation
  returns its pool slot.

* **Document extraction memory stays bounded.** PDF pages are closed as soon as they are read (a
  198 page filing that peaked at 1.50 GB now stays near 157 MB for 468 pages), and XLS sheets are
  parsed and released one at a time. Extracted text is unchanged.

* **A NUL byte in an extracted document no longer aborts an index run.** Block text and string
  metadata are cleaned before chunking in `recall index` and generation builds, and the source and
  count are logged.

* **Configuration and input errors are refused by name.** `recall lint`
  reports an empty `valid_from` or `valid_until` (a date on the next indented line) as
  `invalid-date` instead of passing a memo that never expires; `hashing-128` and `hashing:384`
  are refused instead of building 64 dimensions; `RECALL_LOG_LEVEL` and `RECALL_LOG_FORMAT` are
  stripped; a database without the `vector` type raises `SchemaTooOld` naming `recall schema
  apply`; unreadable or invalid UTF-8 corpus input exits 2; serving refuses a schema without an
  upsert key; and `recall doctor` names the real cause of a migration checksum mismatch.

* **`python -m recall.eval.labelled` honours `RECALL_EMBED_PROFILE`** for fastembed
  and Voyage, instead of constructing the default model, and refuses an incompatible backend.

* **A local folder cannot stand in for a pinned model.** A directory shadowing the
  reviewed Hub id of the pinned SFR embedder, the CoREB reranker or the built-in reranker is
  refused before any loader runs. Empty legacy code-profile settings and whitespace-only reranker
  flags read as unset.

* **A pooled connection is checked before the caller's work runs**, so an indeterminate
  commit is never replayed over a newer write.

* **Writers that name one table by two spellings contend for one lock.** Accepted
  PostgreSQL table aliases share a writer lock, primary corpus operations are bound to the session
  that owns it, and a lost lock fails closed.

* **The git pre-commit hook checks what will be committed**: index blobs, unusual file
  names and renamed or type-changed Markdown destinations. Supersession parsing ignores
  punctuation inside a reference.

* **MCP transport cleanup never closes servers it cannot prove are its own.**
  It stops at a configuration whose inferred ownership is ambiguous, does not treat pid 1 as an
  owner, stops at an unreadable `.mcp.json`, and SessionEnd with reason `clear` or `resume` keeps
  the live client's servers. On Windows the process listing gets 8 s instead of 2 s, which it
  needed, so cleanup runs.

* **Hook configuration and credentials fail soft.** `load_config` accepts a
  UTF-8 BOM and never raises on undecodable bytes; a credentials or token cache file that is not
  UTF-8 reads as corrupt; a `connect_timeout` below 1 s becomes 2 s instead of psycopg's default;
  memo names printed into SessionStart context are quoted and cut to 80 characters; and a
  non-numeric cached `chunks` no longer raises.

* **The write-time relay binds to the configuration it was approved with**, nested
  `MultiEdit` drafts are read at the hook boundary, Windows liveness checks no longer signal a
  console, and the Agent SDK example is a complete async script.

* **A manifest build parses the bytes whose checksum passed**, and a hosted embedder
  revision pin is refused. The Agent SDK's indexing uses its own instance environment.

* **Desktop ingestion stays inside what you picked.** Folder ingestion no
  longer follows file links out of the folder, collides on equal file names or exceeds the upload
  limits, and runs off the GUI thread; two upload names that differ only in case are refused
  instead of overwriting one on Windows; a padded upload at the byte limit is accepted; a failed
  staged-file deletion returns an incomplete-cleanup receipt; and desktop scans skip resolved
  private configuration files.

* **`fit_voyage_image(strict=True)` no longer decodes an image over Pillow's pixel limit.** Pillow
  refuses an image declaring more than twice `PIL.Image.MAX_IMAGE_PIXELS` but only warns below that,
  so a small compressed file declaring about 150M pixels was decoded in full before being scaled to
  Voyage's limit. `fit_voyage_image(strict=True)` now raises
  `DecompressionBombError` for anything over `MAX_IMAGE_PIXELS` (89,478,485 by default) before
  decoding. The default mode, used by the core tenant, still fits it.

* **The desktop update check looks for the installer the release pipeline publishes,
  `recall-install-<tag>-windows-x64.zip`.** It looked for a `.exe`, which no release carries, so
  it could not succeed. No release since v0.9.8 carries an installer, because the Windows build
  is refused unless it is signed, and until one does the check says the latest release has no
  Windows installer.

* **The restore drill checks what it restored.** Its application smoke now fails unless the
  search answer names the restored tenant and generation, where any HTTP 200 passed.

* **The OIDC client closes a redirect it refuses**, and the desktop updater refuses an installer
  URL that is not https or an asset name that is not a plain file name, before downloading.

* **ATM-Bench Recall@10 is 90.0995, not 92.8924.** The figure published in `docs/ATM_BENCH.md`,
  `docs/EVIDENCE.md`, the site and the board submission is Hit@10, the share of questions with at
  least one gold item in the top 10, because `benchmarks/atm_bench.py::metrics_for` stores that
  indicator under the key `Recall@10`. The board's Recall@10 is the mean fraction of gold items
  found; the benchmark's own evaluator gives 90.0995 on the same frozen run
  (`results/atm/atm_recall_rescore_20261004.json`). QS is unchanged. The board's maintainer found
  it.

### Security

* **On Windows, no RE-call command runs an executable from the
  current directory.** Every console script (`recall`, `recall-mcp`, the desktop entry points and
  the hook launcher) sets `NoDefaultCurrentDirectoryInExePath` before doing anything else, and the
  setup wizard, `recall uninstall` and the hardware probes set it for their own `docker`, `wsl`,
  `systeminfo` and `nvidia-smi` launches. Before this, a `git.exe` or `docker.exe` committed to a
  cloned repository ran in place of the real one when `recall index .`, the wizard or a hook ran
  there. Production `python -m` launches, including the `UserPromptSubmit` hook, now use `-P`, so
  a project-local package cannot shadow RE-call's own modules.

* **A link or path outside a memo folder is refused before it is opened.** On
  Windows, resolving a UNC target (`[[//host/share/x]]`, `[x](\\host\share\x.md)`, or a dashboard
  `/read?path=` value) contacted the share and sent the user's credentials before any containment
  check ran. The dashboard, and four other sites that resolved first, now refuse an anchored,
  drive-relative or foreign UNC name before any filesystem call and decide on the resolved target
  (`recall.confinement`). The dashboard memo reader also no longer follows a symbolic link out of
  its corpus.

* **A symbolic link in a corpus is judged by its target.** A link with a harmless name
  (`notes.md -> .env`, or a link into a folder the source policy denies) had its target indexed and
  made searchable. The scan and the source policy now judge the target, and under a policy a link
  is indexed only when the same rule governs both paths; a link that crosses rules fails the run,
  as a denied file does. `recall index` also warns, naming the file, when an indexed file has more
  than one hard link (#1123).

* **BREAKING: a source policy file with an unknown or misspelled key refuses to
  load.** A misspelled `classification` used to load as `internal`, and a misspelled
  `allow_egress` dropped the egress requirement. Any extra key, such as `$schema` or `comment`, now
  refuses at startup, as does a misspelled redaction name and a rule that omits `principals`,
  `purposes` or `redactions`. Policy digests now follow the order of the redactions, since
  overlapping patterns give different text: a policy whose redactions are not already sorted gets
  new `digest` and `content_digest` values, and rows filtered on the old digest return nothing
  until a re-index or a generation rebuild. To upgrade, remove any key the policy schema does not
  list.

* **MCP callers get the source policy clearance and egress the server grants them.**
  Clearance and egress were read from token claims no verifier sets, so every MCP caller was
  judged `internal` without egress and could not reach a `confidential` or `restricted` source, or
  a rule with `allow_egress` (it failed closed). A static token file entry may now carry
  `"clearance"` (`public`, `internal`, `confidential` or `restricted`) and `"egress_allowed"` (a
  JSON boolean), and `RECALL_OIDC_SUBJECT_CLEARANCE` (for example
  `alice@corp:confidential,svc-etl:internal+egress`) grants them by verified `sub`. Any other value
  refuses to load.

* **The published-default credential guard reads the connection libpq will
  actually make.** A libpq keyword DSN, a URI query override, the libpq environment defaults and
  `hostaddr` could carry the published default password past the guard before migrations ran.
  `recall quickstart` now applies the secure DSN policy to a database you supply, and a bare
  PostgreSQL `recall setup` after a rejected `.env` needs an explicit target.

* **Secret redaction catches more credential shapes.** Names with a credential
  word that is not their last part (`AWS_SECRET_ACCESS_KEY`), quoted header and assignment values,
  spaced separators, quoted JSON fields, letter-only IPv6 addresses and quoted mailboxes are now
  masked, in logs and error text and in the hooks' failure signatures and event log.

* **Fact and erasure checks no longer leak what a caller may not read.** A lite
  forget or withdraw checks a JSON `metadata.file` that is a number, boolean, list or object
  against the source policy, as retrieval reads it. A reused request id whose recorded event cites
  a memo the caller may not read now answers `CARD_NOT_FOUND` and records nothing, where it
  returned the event (whose id let a caller test which fact was applied), and `apply_fact_memory`
  for another tenant under a source policy raises `PermissionError` before reading.

* **Graph expansion no longer reports neighbours the source policy denies.** Under a source security
  policy, `candidate_fetched_count`, `candidate_payload_bytes`, `db_result_bytes`, a
  `security_policy` rejection and the gate reason counted policy-denied neighbours, which told a
  caller that unreadable memos existed.

* **The enterprise control plane holds the tenant outbox lock through an erasure**,
  so a replay can no longer restore deleted text, and a stale JWKS key is not reused past the
  stale window while another refresh holds the lock.

* **Setup keeps credentials and memory state out of git and away from other users.** A `.recall`
  directory RE-call creates gets a `.gitignore` of `*`, so `git add .` in a memo repository no
  longer stages the store, and `.dockerignore` excludes `.recall` at any depth. Inside a git work
  tree `recall setup` adds `.env` to `.gitignore`; the managed `.env`, the wizard state file and
  client config copies holding credentials are written owner-only (`0600`) through an atomic writer;
  a symbolic link backup target is refused. The backup `recall setup` takes of `~/.claude.json` is
  created owner-only under a name that is never reused, so a second edit in the same second keeps a
  backup of its own and an existing backup is never overwritten. `recall quickstart` points at
  `recall wizard` for a real threshold, and `recall calibrate` prints the published calibration
  guide's URL, which a wheel install can open.

* **Forget and withdraw check the source policy against every name the erased rows are authorized
  by.** `recall_forget`, `recall_withdraw`, `recall forget` and `recall withdraw` checked the names
  they were given and then erased every row those names resolved to, so a caller allowed to erase a
  memo by its absolute path could erase rows the policy denied under their root-relative `file`.
  Every store now reports each row's names, one denied name refuses the whole call before anything
  is deleted, and nothing changes when no source policy is configured.

* **`recall forget` erases the fact values a forgotten memo supplied** (migration 0029). It used
  to retract them, which hid them, while the value stayed in the append-only fact ledger, in every
  evidence card citing the memo and in replay receipts, and the ids hashing it let a low-entropy
  value be recovered by enumeration. Those are now rewritten in place under fresh random ids
  (`facts_redacted` in the result), the triggers still refusing any other change; a lite store
  also zeroes the freed bytes, and `--vacuum` compacts the file. `recall provenance erase` does the
  same for a whole tenant. `recall withdraw` erases them the same way, and a `withdrawn` event
  follows the assertion it names to its new id, as a retraction does. The materialized current
  facts (0030) are rebuilt in the same transaction, since their ids hash the erased values. A
  database already at 0033 applies 0029 last. Apply it with `--with-migrations` and re-run
  `recall schema grants --controller`. PostgreSQL's dead row versions, WAL, replicas and backups
  stay the operator's to manage.

* **`recall_forget_memo` also erases the request journal's hashes of the memo.** The assistant's
  request journal (`.recall-authors.json`) kept a forgotten memo's request identity, an unsalted
  SHA-256 over its bytes, and a correction kept one over the corrected memo's revision, so a
  guessable template turned either into an enumeration oracle for a short value. A forgotten memo's
  requests now keep only their memo id, state and time, and any retry of one is refused as
  forgotten; a correction of it keeps a hash of its own content, so a lost reply is still replayed.
  A memo published but never indexed (a crash after publication with no refresh since) is now
  forgotten instead of refused, so its retry can no longer store it. The journal's hashes of a memo
  go however it leaves the index: the generic forget (`recall forget`, `recall_forget`) erases them
  too, and so does the next index that prunes a memo deleted by hand. `recall_forget_memo` also
  refuses when rows hold the memo's bytes under its file name from another path, not only under its
  stored name. Since 2026-10-27 a forget erases the journal only of memos its own store held: the
  ones it deletes, and names it recorded deleting earlier (a `forget_memo_journal` audit event
  written before the rows go), so a re-run still completes a failed erasure while a forget in
  another store can no longer name this folder's memo.

* **The hosted memory sync cannot leak or silently drop memos.** It sends a bearer token only to
  an https endpoint or to loopback http, parsing the URL so that a user part cannot pose as
  loopback; it screens the exact bytes it uploads; it skips and names files the server would
  refuse; it budgets batches on wire bytes; and SessionStart reports a blocked sync. The screen
  also catches AWS STS, npm and PyPI tokens.

* **A late write time relay child cannot take over a newer relay.** It receives a digest of the
  token it was spawned for, never the token itself, and exits when the state holds another.

* **`mcp` is pinned at 2.2.0, up from 2.1.0** (GHSA-84m7-p3x7-pcfv, CVE-2026-59951). In 2.1.0 a
  stateful Streamable HTTP server never reclaimed session state, so a client able to reach the
  endpoint could grow its memory without bound. RE-call's HTTP server runs stateless by default
  and is exposed only when `RECALL_MCP_STATELESS=0` is set; the stdio servers are not affected.

* **`opentelemetry-api` is locked at 1.45.1, up from 1.44.0** (CVE-2026-114603). It reaches RE-call
  only through `mcp`; the dependency audit refuses 1.44.0 since the advisory was published.

* **`pyjwt` is locked at 2.15.1, up from 2.13.0** (CVE-2026-102274), and **`urllib3` at 2.8.0, up
  from 2.7.0** (CVE-2026-97687, CVE-2026-97688, CVE-2026-97689). Only `uv.lock` changes.

* **`aiohttp` is locked at 3.14.5, up from 3.14.3** (CVE-2026-111063, CVE-2026-111064,
  GHSA-2g87-pp6c-29x5); `yarl` moves to 1.25.1 with it.

* **`certifi` is locked at 2026.10.19, up from 2026.6.17** (GHSA-6q48-6c7w-9gw2).

* **`filelock` is locked at 4.2.4, up from 3.31.1, and `setuptools` at 84.2.1, up from 83.0.0**
  (CVE-2026-112940, CVE-2026-112913, CVE-2026-112907; GHSA-xmj3-9gf7-h52w). Both reach RE-call only
  transitively.

* **An environment that keeps fastembed 0.9.0 or earlier keeps a Pillow with open advisories.**
  The advisories published against Pillow 12.3.0 from 2026-10-20 are fixed in 13.0.0, and the lock
  pins 13.0.0 with fastembed 0.9.1, the first fastembed that allows it. fastembed 0.9.0 and earlier
  require `pillow<13`, and the `fastembed` extra still admits them (`fastembed>=0.3,<1`). A fresh
  `pip install "recall-rag[fastembed]"` resolves fastembed 0.9.1 and Pillow 13.0.0; in an existing
  environment run `python -m pip install --upgrade fastembed pillow`.

## [0.14.0] (2026-09-17)

### Added

* **Voyage Context 4 as a registered embedding profile.** Contextualized document groups, provider
  request bounds, group membership fingerprints, and generation identity are explicit and documented
  for staged production use.

* **Typed semantic graph relations.** Authored `depends_on` and `supersedes` metadata can be
  projected into the generation graph when targets resolve exactly. Missing, ambiguous, and
  malformed declarations become diagnostics rather than trusted relations.

* **Agent continuity surfaces.** The repository now includes an agent facing continuity guide and
  an optional isolated guestbook service with a non root runtime contract.

* **Generation-bound atomic fact rescue.** An optional active retrieval mode inserts one exact
  atomic-view winner at dense rank six before ordinary fusion and trust evaluation. Artifacts are
  immutable and bound to generation, calibration, pipeline, corpus, embedding profile, and
  dimension. The mode is off by default, bypasses scoped and security-policy requests, and fails
  closed when an active artifact is missing, malformed, stale, or cannot load its selected parent.

### Changed

* **Public reasoning graph expansion now defaults to bounded `auto` mode.** Every nonempty query
  uses the bounded one hop path unless `graph_expansion=off` is selected. The MCP path takes a
  hybrid pool of 20, expands from the top 8 provisional direct hits, protects that direct prefix,
  and caps the final context at 10 items before ordinary trust evaluation. This changes the prior
  opt in default for public reasoning surfaces; ordinary search remains unchanged.

* **Graph retrieval is temporal, category aware, and more bounded.** An `as_of` value filters
  temporal edges before ranking, budgets vary by query category, candidate ordering combines
  calibrated relevance with bounded relation features, and graph metadata and query vectors are
  reused where generation identity permits.

* **Production generation input validation is more flexible.** Production builds may use a local
  manifest only when an allowlist, SHA256 digest, and byte size are supplied. Versioned S3 manifests
  remain the preferred immutable input.

* **MCP ownership boundaries are explicit.** Reasoning, retrieval, indexing, lifecycle, graph,
  provenance, and response operations are separated into service seams. Session cleanup identifies
  and closes only transports owned by that session.

* **Hosted retrieval and provider identity are explicit.** Embedding profile identity, grouping
  policy, graph schema compatibility, and answer provider selection participate in the serving
  contract instead of being inferred from a profile name.

* **Dependency floors were refreshed.** The release includes the current supported versions of the
  signing action, uv setup, cache action, Redis client, Hypothesis, sentence-transformers,
  ONNX Runtime, and Starlette.

* **The project license changed for this and future releases.** The PolyForm Noncommercial License
  1.0.0 permits personal, educational, and noncommercial research use; commercial production,
  hosted service, and redistribution use require a separate written license. Releases previously
  published under Apache 2.0 remain under Apache 2.0.

### Fixed

* Invalid source files no longer abort a complete indexing run, calibration is invalidated after
  privacy erasure, and temporal graph fingerprints remain stable across supported datetime forms.

### Evaluation and limits

* **Generation-bound atomic fact rescue was validated on the production-shaped path.** A frozen
  96-query live replay gained 6 exact-at-5 results with no losses, while the gold-source leg gained
  7 with no losses. Atomic latency was p95 12.7005 ms and p99 36.9026 ms, with zero trust
  regressions, zero unexplained trust changes, zero errors, and exact parity for 12 of 12
  source-scoped and 12 of 12 security-policy-scoped requests. Rollback and restore-active
  rehearsals passed through fresh inherited-mode MCP processes. The active mode remains optional,
  generation-bound, and fail closed when its artifact contract is not satisfied.

## [0.13.0] (2026-09-09)

### Added

* **A deterministic provenance controller for structured facts.** Trusted retrieval can now produce
  immutable evidence cards containing source identity, source digest, validity, tenant and
  generation lineage, calibration state, rank, authored links, and structured support. The
  controller re-resolves those cards, rechecks the current source bytes and active generation, and
  accepts a fact only when a certified card supports the exact canonical claim. It records stable
  decisions for duplicates, stale or changed sources, lineage mismatches, unsupported claims,
  contradictions, fresh-search recovery, and storage failures.

  The controller is backed by an append-only, tenant-scoped fact ledger and a current-facts
  projection. Authorized events retain evidence-card snapshots, so generation cleanup cannot erase
  the provenance of an applied fact. An optional materialization outbox adds bounded, lease-based
  delivery and recovery without giving the recovery worker authority to assert facts. PostgreSQL
  and SQLite adapters are available, while the MCP service uses PostgreSQL.

* **A bounded query-construction challenge for difficult retrievals.** The original model may
  restate the memory need and propose a small number of query variants, but model output remains
  proposal data. Deterministic validation limits query size, novelty, parent evidence, candidate
  count, and refinement rounds before another trusted retrieval call. The challenge does not
  promote model text or graph metadata to evidence.

* **A production AWS reference topology and operations layer.** Terraform now describes a
  multi-AZ ECS service behind an ALB, Aurora PostgreSQL with pgvector and RDS Proxy, Valkey or
  Redis rate limiting, KMS encryption, Secrets Manager integration, versioned S3 backup receipts,
  CloudWatch monitoring, and a scheduled isolated restore drill. The optional `aws` extra supplies
  the runtime integrations without adding cloud dependencies to local installations.

* **An application-level restore smoke path.** The restore drill now starts the real application
  against the restored PostgreSQL database, selects the generation route, reaches `/readyz`, and
  completes an authenticated `recall_search`. A subprocess integration test exercises that
  contract with a real PostgreSQL service.

* **Backup, restore, and secret rotation workflows.** The CLI can inspect continuous backup and
  point-in-time recovery state, create and verify encrypted snapshots, restore a new isolated
  cluster with explicit confirmation, and verify the version identifiers reported by running ECS
  tasks. Runtime secret values are resolved once during bootstrap and are never written to logs,
  receipts, task tags, or Terraform configuration. Rotation helpers deploy replacements, verify
  every task, and require a separate confirmation before revoking the previous secret version.

* **Centralized rate limiting and durable mutation idempotency.** Redis or Valkey can enforce
  tenant budgets across the fleet, with a bounded local fallback for read-only work and fail-closed
  behavior for writes, forget, and administrative operations when the shared limiter is unavailable.
  Authenticated HTTP mutations now persist a response receipt in PostgreSQL before caching it in
  Redis. A retry replays the original response after a cache loss, or returns
  `reconciliation_required` without repeating the side effect when neither store has the result.
  The new `recall idempotency reconcile` command lets an operator persist a verified result without
  executing the mutation again.

* **Explicit health and readiness routes.** HTTP serving exposes `livez`, `readyz`, and `startupz`.
  Readiness checks the database, schema, row-level security, active generation, calibration, and
  control plane through a deterministic bounded tenant sample. The endpoint reports limiter state
  without exposing tenant inventory, and read-only Redis fallback does not make a healthy database
  unready.

* **Deny-by-default source security policy.** Operators can authorize sources by prefix, principal,
  purpose, classification, clearance, and egress permission, with optional email, phone, and secret
  redaction. Policy identity is bound into indexing metadata and scope decisions, so policy changes
  cannot silently reuse vectors created under a different source boundary.

* **Operator diagnostics and release quality gates.** `recall doctor` now diagnoses the installed
  interpreter, executable, optional capabilities, Docker, database, pgvector, schema, configured
  tenant, calibration, and agent registration without changing state. The repository also ships a
  standalone quickstart, generated architecture documentation, optional-import profile checks,
  dependency and dead-code checks, broad-exception intent checks, Terraform validation, and
  security scanning in CI.

### Changed

* **Indexing and serving now share one resolved runtime route.** `RECALL_INDEX_MODE` makes the
  legacy or generation route explicit. Production and enterprise deployments require the
  generation route, while development retains the legacy compatibility default when no mode is
  specified. This prevents reads and writes from silently selecting different storage models.

* **MCP serving uses canonical response and settings models.** Runtime configuration is validated
  once into an immutable settings snapshot, optional AWS secrets are resolved at bootstrap, and
  shared retrieval, generation, authorization, and error models are used across the MCP and agent
  surfaces. CLI command registration and API documentation are generated from the same command
  ownership boundaries, reducing drift in help and introspection.

* **Production limits are explicit and observable.** Retrieval profiles keep process-level
  concurrency and queue bounds, expose stage timings and overload reasons, and publish a measured
  performance baseline with a supplementary quality SLO policy. Active query routing remains
  shadow-only until the paired quality, security, latency, and availability gates are satisfied.

* **BREAKING: MCP mutation contracts now require request identity on HTTP.** Writes, forget
  operations, and
  administrative mutations require an idempotency key. `recall_apply_fact` exposes the same
  contract as `request_id`, and key reuse with different operations or arguments is rejected as an
  idempotency conflict.

* **BREAKING: the optional dependency boundary is safer.** The LlamaIndex extra no longer installs the host
  framework, so applications using that adapter must install `llama-index-core` explicitly. Direct
  dependencies for the MCP, desktop, benchmark, and analysis surfaces are declared where those
  modules import them, and the vulnerable framework dependency is no longer part of the project
  installation or audit set.

* **AWS edge protection and alerting are explicit.** The public ALB can use a regional WAF with
  optional source allowlist and blocklist CIDRs, per-IP `/mcp` rate limiting, AWS managed rules,
  abuse alarms, an SNS destination, and ALB and WAF CloudWatch alarms.

* **Dependency verification is stricter.** The locked `httpx2` dependency is updated to 2.12.0,
  while CI verifies exported dependency hashes and runs the documented audit policy.

### Removed

* **BREAKING: the experimental `recall_graph_first_retrieval` MCP tool was retired.** The preregistered
  graph-first probe did not rescue the frozen retrieval misses, so graph output remains available
  only through the bounded, opt-in Evidence Graph expansion attached to reasoning.

### Fixed

* **Structured fact writes no longer accept unsupported or absent evidence.** Evidence cards
  without authoritative structured support are excluded from authorization, absent-case writes are
  refused, and contradictory facts require an authored supersession link. Fresh-search recovery
  uses a canonical query and still passes through the same trust and lineage checks.

* **Lost Redis mutation responses no longer cause a second side effect.** Durable receipt lookup,
  operation and request fingerprint binding, explicit reconciliation, and reservation cleanup now
  cover cache-write failures, preflight failures, duplicate requests, and conflicting key reuse.

* **Restore validation now binds to the restored database.** The drill validates the writer endpoint,
  schema and extensions, roles and grants, forced row-level security, tenant policies, active
  generation, calibration, representative tenant-scoped search, indexes, and deterministic
  checksums. Checksum work is bounded by default and can be expanded to a full check explicitly.
  Temporary restore clusters and instances are isolated, narrowly authorized, and cleaned up only
  through explicit confirmations. The application-level smoke path also verifies the serving
  contract before the restored environment is considered healthy.

* **Impossible rate-limit costs are rejected before backend work.** Local and Redis limiters now
  return the same permanent refusal for a request whose cost can never fit the configured bucket,
  avoiding pointless retry loops and keeping the refusal semantics consistent across backends.

* **Optional capabilities fail with actionable diagnostics.** Missing packages, native loader
  failures, and missing executables are classified separately, and optional imports remain safe in
  lean installations. The executable CLI now exposes the documented doctor and operational
  commands, while MCP service extraction and generated architecture checks catch boundary drift.

### Security

* **Multi-tenant authenticated HTTP refuses ineffective row-level security.** Single-tenant and
  stdio deployments retain a startup warning, but a multi-tenant HTTP server will not start under a
  superuser or `BYPASSRLS` role.

* **Source indexing applies the same confinement and policy checks to individual files as to
  directory walks.** This closes the path that could otherwise index a credential file from the
  server working directory. Source policy remains deny-by-default, and cloud embedding remains an
  explicit operator choice.

* **Fact and materialization appends use the generated least-privilege controller boundary.** The
  serving role is read-only for the ledger, cards, and outbox, while protected database functions
  enforce tenant context for controller writes. Production task roles separate serving from restore
  infrastructure permissions.

* **Serving ECS tasks no longer receive restore permissions or RDS master secret access.** Restore
  instance creation and cleanup stay confined to the isolated restore infrastructure role.

### Evaluation and limits

* The safety-core mutation harness killed all 21 registered mutations across trust, MCP server,
  generation, and provenance-controller guards after the focused regression tests were added. This
  measures the strength of the selected tests; it is not a proof that every unregistered behavior
  is safe.

* The performance baseline and quality SLO are committed diagnostic artifacts from a local,
  production-shaped workload. They document current cost and load boundaries, but do not certify
  active routing or substitute for a certified reference host.

* The AWS stack and restore drill are reference implementation and validation paths. Terraform
  validation and live staging restore, cutover, rollback, and secret-rotation drills remain
  required before production approval.

### Upgrade notes

* Apply migrations `0018` through `0024` before enabling the provenance ledger, evidence cards,
  materialization outbox, protected controller append, or durable idempotency receipts. Generate
  and apply serving grants as part of the same deployment gate.

* Production and enterprise installations must use the generation route and separate serving,
  migration, and, when fact writes are enabled, controller database credentials. Configure
  `RECALL_FACT_WRITE_DSN` for the isolated controller role.

* **BREAKING:** HTTP clients must provide idempotency keys for mutating operations. Operators should retain the
  receipt cleanup job described in the API documentation and use reconciliation only after checking
  the original side effect in its owning system.

* Install the optional AWS dependencies with `recall-rag[aws]` for the reference deployment. Users
  of the LlamaIndex adapter must install `llama-index-core` separately because the compatibility
  extra no longer installs it implicitly.

* Configure the restore validation database contract and WAF inputs described in the AWS operations
  documentation before using the reference deployment for staging or production drills.

## [0.12.0] (2026-09-02)

### Added

* **A `UserPromptSubmit` hook that searches project memory with the user's own words, before the
  turn starts.** The existing write-time hook fires on `Write`, `Edit` and `Bash`, which is after
  the plan has been drafted, so it cannot reach the failure this one targets: re-opening a decision
  the project already settled, or re-running a measurement that already has a committed result.
  `UserPromptSubmit` is the only event that carries a query and still precedes every proposal in
  the turn.

  It reads the project's memo **files** directly, with BM25 over their titles, `description`
  frontmatter and bodies. No database, no network and no embedder, so it works where the corpus is
  a remote host the workstation cannot reach. It names at most three memos with their one-line
  summaries and paths rather than pasting them, and injects nothing when nothing matches at least
  two of the prompt's content words.

  Registered by `recall setup` and by the Claude plugin in the same change, and controlled by
  `prompt_time.enabled` in `~/.claude/recall-hook.json`; like `write_time`, the entry is installed
  whichever way the flag is set, so switching it off survives `recall hooks upgrade`.

  ⚠️ **Its benefit is unmeasured.** There is no pre-registration and no A/B behind it, and the
  write-time hook's numbers are not evidence for it: different event, different query shape,
  different mechanism. What is measured is its cost, against a 329-memo store: 754 ms to read and
  rank everything and 320 ms for a prompt too short to rank, against a 305 ms bare-interpreter
  floor, so roughly 430 ms of marginal cost once per turn. It is also **not** a substitute for the
  write-time hook on hazards: a goal-shaped query reaches a governing hazard memo far less often
  than the draft text does.

### Fixed

* **A hook no longer fails when `python -m recall_hooks` resolves out of an unrelated checkout.**
  `-m` prepends the current directory to `sys.path`, so a session whose working directory is a
  checkout of this repository runs that checkout's copy of the package, on whatever branch it
  happens to be. A branch predating one of the per-event modules raised `ImportError` out of the
  hook on every prompt or every tool call. The per-event imports are now guarded and degrade to
  the behaviour the package had before the feature existed, which is silence. Invoking with `-P`
  and an explicit `PYTHONPATH` is the deterministic fix where which copy runs actually matters.

### Added

* **Graph RAG serving is now integrated and explicitly bounded.** The deterministic Evidence Graph
  V1 remains separate from the authored supersession reasoning graph. `graph_expansion=one_hop`
  traverses the semantic graph only, starts from trusted retrieval, follows permitted directional
  relations, re-evaluates every candidate through the ordinary trust layer, and adds only trusted
  evidence. Graph metadata and model proposals never become evidence by themselves.

* **Graph expansion now performs admission checks before projecting the graph.** Readiness,
  generation binding, trusted seed availability, and the selective gate are reported as whole
  expansion refusals. Candidate rejection counters describe only candidates that were actually
  discovered.

* **The graph contract is documented precisely.** The reasoning graph and semantic graph now have
  separate schemas, relation vocabularies, consumers, and diagnostics. Automatic extraction creates
  `references` edges from links and wikilinks. Other semantic relations require explicit
  `recall_graph` declarations, while `supersedes` remains enforced upstream by trust.

* **The opt in reasoning answer path is reachable.** The Python reasoning API, CLI, and MCP tool
  can receive an answer provider while the model free audit path remains model free. The release
  adds an OpenAI compatible `/chat/completions` provider, preserves the local Ollama adapter,
  bounds paid answer calls, and routes from ordinary retrieval only when reasoning is actionable
  and an answer backend is configured.

* **Content addressed embedding reuse is enabled across indexing entry points.** The shared cache
  is bounded by `RECALL_EMBED_CACHE_MAX_MB`, defaults to 512 MB, uses packed float32 vectors, has
  LRU eviction, deduplicates repeated inputs, and falls back to re-embedding when cache storage
  fails.

* **Claude Code hooks can synchronize project memory to a hosted corpus.** Hosted credentials use
  the OS keychain where available and a protected file fallback where needed. Login, logout,
  refresh, access token caching, MCP transport, failure classification, retry policy, pending sync
  notices, and session end synchronization are included.

* **Hosted synchronization can plan before uploading.** `recall_inventory` exposes source names and
  raw content digests for a tenant. The hosted planner compares those digests with local files and
  uploads only changed or unknown files. Truncated inventories are refused and deletion is opt in.

* **A `UserPromptSubmit` memory hook is available through setup and the Claude plugin.** It uses
  local memo files and bounded BM25 ranking before a turn starts, remains fail open, and has no
  database, network, or embedder dependency.

* **Codex is now a first class RE-call integration.** `recall setup` detects Codex and installs a
  packaged Codex plugin, MCP server, shared hook configuration, and the same durable memo contract
  used by Claude Code. Lifecycle hooks cover startup, resume, clear, compact, prompt time, write
  time, pre compact refresh, and session end refresh. The adapter delegates prompt and write
  handling to the shared implementation, keeping thresholds, project discovery, and fail open
  behavior aligned across both clients. This is the merged [PR #583](https://github.com/GiulioDER/RE-call/pull/583).

* **Hosted uploads preserve relative paths and converge on re-ingest.** Path traversal, absolute
  paths, Windows device names, invalid components, and length hazards are refused. Repeated uploads
  of the same source no longer duplicate the active corpus, and changed sources can supersede old
  content.

### Changed

* **Reasoning and retrieval descriptions now expose the actual decision boundary.** Retrieval does
  not silently escalate into model execution. It emits a library authored `NEXT:` recommendation
  only for non gap trust blocks or superseded matches, and only when the configured tool surface
  can answer.

* **Open indexing is safer on constrained hosts.** The default outer embedding batch is 64 chunks,
  allocation failures identify `RECALL_INDEX_BATCH_CHUNKS` as the corrective setting, and HTTP
  transports default to stateless MCP mode with `RECALL_MCP_STATELESS` available when session state
  is required.

* **Serving verification is now read only.** The session serving scripts verify the server that a
  session will actually launch, report tenant and generation boundaries, avoid deployment state
  changes, and clean up timed out child processes.

* **Write time hook lifecycle and project binding are explicit.** The relay is integrated per
  session, setup passes the selected project root consistently, and the machine wide hook
  configuration reports when a later project installation moves that binding.

* **Dependency maintenance was refreshed.** The release line updates the Claude Agent SDK,
  Pydantic, LangChain Core, OpenAI, and python dotenv pins, caps pypdf at the supported major, and
  holds the corresponding Dependabot ranges against the tested compatibility window.

### Fixed

* **Graph expansion no longer projects a graph that the admission gate will discard.** Readiness
  retains precedence, and graph refusal counters are no longer reported as candidate rejection
  counts.

* **The reasoning answer port is no longer dead code.** Configured CLI and MCP reasoning queries can
  invoke the selected answer provider, while audit commands cannot spend a model call. Provider
  failures remain structured reasoning outcomes.

* **MCP trust refusals now reach clients as structured tool errors.** The refusal code, calibration
  state, tenant, and generation remain visible instead of being flattened into an empty result or
  an opaque server failure.

* **Indexing now reports the actual allocation remedy.** The previous outer batch default could let
  the embedder request a large padded allocation and fail part way through a run. The failure now
  names the batch setting and leaves the existing batch intact.

* **Unresolved supersession references now become ambiguous state.** Tests exercise both producer
  driven and direct consumer paths, so the guard is no longer reachable only in theory.

* **Prompt hooks no longer run an unrelated checkout by accident.** The installed module path is
  deterministic, missing optional event modules degrade to silence, and the deployed hook copy is
  checked for drift.

* **Hosted credential caches are account bound.** Switching between configured accounts no longer
  risks sending one account's access token with another project's memory. Login and logout have
  distinct user facing behavior from silent hook events.

* **The credential screen is fail closed for new hosted uploads without disclosing matched text.**
  Scanner failures are reported as a broken screen rather than as a false positive.

### Evaluation and limits

* The graph work is a serving and trust boundary improvement, not a new retrieval quality claim.
  The graph first experiment remains closed after rescuing zero of 15 frozen misses, and the answer
  provider has no quality evaluation beyond one correct live call. Graph marginal contribution and
  reasoning routing take up remain unmeasured.

* The refrozen scope probe remains an evaluation artifact, not a promoted feature. Both folder and
  facet arms rescued zero of 15 frozen misses. Facet scoping retained 23 of 31 controls, while
  baseline retained 19 of 31. Those results do not justify automatic scope selection or a change
  to the default retrieval path.

* Hosted credential screening measured zero findings on the author's corpus. Positive controls
  caught planted fixtures, but there is no labelled external corpus, so the release does not claim
  complete secret detection.

## [0.11.0] (2026-08-29)

### Added

* **A deterministic semantic graph over the corpus, and a reasoning graph above it.** Entities,
  mentions and relations are projected from chunk metadata and explicit authored declarations, with
  every relation pointing back at the chunks that evidence it (migration `0016`). The projection is
  deliberately conservative: it never changes a retrieval verdict, never promotes graph text to
  evidence, and separates `authored` relations from `candidate` ones so an inference cannot be
  mistaken for a claim somebody made. Deterministic precision was tuned against a labelled set
  rather than asserted.

  ⚠️ **The one thing the graph is measured NOT to do is rescue a retrieval miss.** The
  pre-registered graph-first probe replayed 46 frozen sessions through entity, relation and hybrid
  seeding and rescued **0 of 15** misses in every mode, against a bar of 5. That route is closed
  and recorded as closed in `docs/preregistrations/2026-08-28-graph-first-retrieval.md`; the graph
  ships for what it does describe, not as a retrieval improvement.

* **Authored authority tiers and exact dependency invalidation** (migration `0017`).
  `recall_graph.authority` declares how a memory was established (`policy`,
  `user_confirmed_decision`, `tool_observation`, `model_inference`), and `recall_graph.depends_on`
  names the sources a memory rests on. When a prerequisite expires or is superseded, dependents are
  projected as `dependency_invalidated` with the invalidation chain that reached them.

  `dependency_invalidated` relabels only a `current` record. Every other state already carries a
  more specific reason not to trust the document, and `ambiguous` and `invalid` in particular are
  fail-closed verdicts; overwriting them would replace an admission of ignorance with a confident
  explanation the reader would act on. The diagnostic and the chain are recorded either way.

* **`recall calibration auto`: re-establish a certified calibration on a rebuilt corpus without a
  person choosing when.** A corpus that grows daily invalidates a carry-forward binding routinely,
  and the previous answer was for somebody to notice. `auto` re-verifies the published threshold
  against the new generation and, when carry-forward refuses, refits on the stored certified query
  set with certification still gating the result. The query sets live in
  `recall_calibration_query_sets` rather than in a file beside the script, so a refit is
  reproducible from the database alone. Observed on the memory tenant: carry-forward refused at
  10.7% false confirm against a 10.0% bound, the refit ran, and certification passed.

* **A metadata-derivation version, so a parser change can actually reach an existing corpus.**
  Chunk reuse is keyed on tenant, URI, `sha256` and pipeline fingerprint, none of which moves when
  the code that DERIVES metadata from a document changes. Every chunk now carries
  `metadata_rule_version` and reuse requires it, so bumping that constant re-derives every source
  exactly once and later generations reuse the repaired result.

  This is a general form of a hazard `_BODY_RULE_VERSION` had already documented for one case. The
  cost of not generalising it was measured: recognising the `type:` facet reached **27 of 10,155
  chunks (0.3%)** after a `--force` rebuild, while 1,319 of 1,342 files declared one. `--force`
  builds a new *generation*; reuse is decided per *source*.

* **What a memo can now declare in frontmatter, in one place.** The parser is still deliberately
  not a YAML parser, and still keeps only what it recognises, but what it recognises has grown:

  | key | means | read by |
  |---|---|---|
  | `valid_from`, `valid_until`, `supersedes` | when this is true, and what replaced it | the trust layer |
  | `type` | the retrieval FACET: what kind of document this is | `recall.scope` |
  | `recall_graph.authority` | how the claim was established | dependency invalidation |
  | `recall_graph.depends_on` | the sources it rests on | dependency invalidation |

  ⚠️ Only the first row makes a truth claim. A facet says what a document IS and never whether it
  should be believed; nothing in the trust layer reads it. That separation is what keeps "three
  keys carry validity" true as the parser grows, and two design docs that had drifted from it were
  corrected.

* **The skills and the write-time hook now reach users down every install path.** Three of the
  four were wrong, each silently:

  | path | before | now |
  |---|---|---|
  | PyPI | no skills at all — `plugin/` is not a package | the wheel force-includes them |
  | `recall setup` | installed exactly one, named by a constant | installs every skill it discovers |
  | Claude plugin | no `PreToolUse` hook, unlike `recall setup` | all four hook events |
  | MCP registry | 3 environment variables declared | the two whose absence fails silently, plus `RECALL_MCP_TOOLS` |

  The installer's `SKILL_NAME` constant is the interesting one: adding a second skill installed
  neither it nor any future one, and nothing reported that. Discovery replaces enumeration, and
  the destination name comes from the skill's own directory rather than a constant, which is what
  stopped every skill from overwriting the first.

  `RECALL_EMBEDDER` is the omission that mattered most in the registry manifest. Several models
  emit 1024 dimensions, so pointing the server at the wrong one does not raise — it returns a
  confidently ranked list that means nothing.

* **A second published skill, `keep-memory-current`.** `check-memory-before-acting` covers the
  read side; this covers the write side, which is the half that decays first. It states the two
  moments that decide whether a memory store is worth anything — what you load when you start and
  what you write back before you stop — and the rules that make a memo findable later: one fact
  per memo with the cost that bought it, an index line is a pointer rather than the memo, delete
  what turns out to be wrong because a stale memory suppresses the retry that would correct it,
  re-index or the memo is invisible, and **verify with a search, never with a row count**.

  It ships the three failure modes rather than only the advice: a rename can fork a path-keyed
  store silently, a flat index stops working past roughly fifty entries, and any stated memo count
  will drift under concurrent sessions, so assert coverage instead of the number.

* **Section heading contextualization: a chunk is embedded with the headings it sits under.**
  A passage that says "it refuses when the lock is held" is unretrievable on its own, because the
  subject is three headings up. `recall.context` builds a deterministic contextual representation
  used **for embedding only** — the stored text and everything a caller reads are unchanged — with
  bounded title, section, neighbour and source segments so context cannot crowd out the passage. A
  1024-dimension profile ships alongside it, and the context mode and version are stamped on every
  chunk so a corpus can say which policy built it.

* **Preregistered evidence routing and state features**, with the LOCOMO routing population frozen
  so the arms are comparable across runs. Routing stays shadow or opt-in until promotion gates
  pass; nothing about the served path changes by default.

* **Folder and facet scoping: a second retrieval dimension over the vectors already stored.**
  `recall search --folder python --facet reference`, the same two arguments on the `recall_search`
  and `recall_evidence` MCP tools, and `Scope(folder=..., facet=...)` in the library. `recall
  scopes` lists what a corpus can be filtered by, with the count of documents declaring no value
  reported separately, because a scope that does not exist and a corpus with no answer both return
  nothing and only the listing tells them apart.

  🔁 **Corrected before release: BOTH dimensions need a rebuild, and the first claim here was
  wrong.** This entry originally said the folder dimension needed no re-index, on the grounds that
  the indexer already recorded a root-relative path. That is true of the `Indexer` path and was
  false of the generation build, which stored a bare basename, so the folder dimension was empty
  wherever it mattered until the fix listed under Fixed below. The facet is read from frontmatter
  at index time and lands on the next build for the same reason. Recognising that key also un-discards something every
  memo already declared: measured on the agent memory corpus, **1,312 of 1,335 files (98.3%) carry
  one**, across eight authored values.

  ⚠️ **A scope is a HARD filter and fails in the direction that hides things.** A chunk outside it
  is absent rather than ranked low, so an over-narrow scope is indistinguishable from a corpus with
  no answer. That is why the accompanying folder-affinity PRIOR reorders and prunes nothing, and
  why it ships **off** (`ScopePrior(weight=0.0)`): routing loses answers unrecoverably, and pruning
  the candidate pool would move the score distribution underneath a certified threshold fitted on
  an unfiltered one. The prior never writes `ScoredChunk.score` for the same reason. Whether it
  helps at all is unmeasured and pre-registered in
  `docs/preregistrations/2026-08-28-folder-scope-and-prior.md`.

* **A `PreToolUse` hook that searches project memory with the text the agent is about to write,
  and injects what matches.** Installed by default by `recall setup`, which asks before enabling
  it. It exists because an agent that has to decide to search mostly does not: an explicit
  instruction to search before writing measured an adoption rate of **0.067**, where this reaches
  1.00 by construction. The query is the DRAFT TEXT rather than a goal description, because draft
  text surfaced the governing memo for 11 of 11 sessions that needed it against 1 of 14 for
  goal-shaped queries.

  ⚠️ **Its benefit is plausible and unproven, and the pre-registration says so.** Paired A/B on
  executed checker endpoints: 17 of 34 control tasks failed against 12 of 34 with the hook, which
  is **6 rescues, 1 regression, McNemar exact p = 0.125** on 34 of a registered 48 pairs. That is
  not significant, and the record committed in advance that nothing ships on a non-significant
  positive. It is a default by owner decision. Full record and its four amendments:
  `docs/preregistrations/2026-08-27-write-time-hook.md`.

  What IS measured and is not in doubt: token cost is **-1% aggregate**, because it fires a median
  of 3 times a session rather than the ten the design assumed; and it rescued **nothing** on the
  hardest task family, which failed 6 of 6 in both arms with its governing memo in the corpus and
  in front of the agent on every write.

  The real cost is latency, and it is per TOOL CALL: **0.14s** when the payload is too short to
  query, **~1.0s** against a reachable corpus, **2.9s** against an unreachable one. That last
  number is why the handler stands down for five minutes after a failed connection, which brings
  a dead corpus from 2.89s to 0.19s per call. Turn it off with `write_time.enabled: false` in
  `~/.claude/recall-hook.json`; the hook is silent and costs 0.19s when disabled or unconfigured.

  Three properties it cannot lose, each tested and mutation-tested: it never denies a tool call, it
  never raises, and it never loads an embedder.

* **`RECALL_MCP_TOOLS`: serve a subset of the server's tools, to stop paying for the ones nobody
  calls.** Every tool definition is re-sent on every turn whether or not it is invoked. Measured
  2026-08-27 on `claude-haiku-4.5`: about 153 input tokens per tool per turn, so the full surface
  costs 5,727 input tokens per turn where the new `search` preset costs 3,731. Across 112 measured
  agent sessions the agents called exactly one tool, `recall_search`, and never invoked the other
  seventeen, which at that rate is roughly 30,000 wasted input tokens per fifteen-turn session.
  Presets are `all`, `search` and `read`; explicit tool names compose with them. Unset serves every
  tool, so an existing deployment is unaffected, and an unrecognised name refuses to start rather
  than silently serving less than the operator configured. This narrows what is *offered* and is
  not an authorisation boundary: scopes remain the only thing that gates execution.

* **`recall_agent`: RE-call as in-process memory for Claude Agent SDK applications.**
  `RecallAgentMemory` wraps the same `recall_mcp.service` functions the MCP server calls as
  in-process SDK tools (`recall_search`, `recall_evidence`; writes behind an explicit
  `write_tools=True`), with the same tool names, the same model-facing descriptions (drift-tested
  against the server docstrings), and byte-identical result rendering via the promoted
  `recall_mcp.service.serving_json`. Trust semantics survive the wrapping: a `TrustRefusal` is
  rendered as its wire form with `advice`, never conflated with an empty result. Ships as the
  optional `agent` extra; the package imports without the SDK installed.
  See `docs/USING_WITH_AGENT_SDK.md`.

### Fixed

* **The HNSW index served two queries that were supposed to be exact.** Both were planner cost
  decisions that track table statistics rather than code, which is why they surfaced as CI flaking
  four tests on an identical commit and identical resolved wheels. Postgres rewrites
  `min(embedding <=> v)` into an ordered scan behind an InitPlan when that costs less, and pgvector
  answers it approximately, so `top_cosine` could report a best cosine that was merely a good one.
  An approximate answer to an exact question is the failure that does not announce itself: the
  number looks reasonable and calibration is fitted on it.

* **A tool call made three database round trips where one would do.** The statement timeout now
  rides on the connection's own options and the active-generation lookup folds into the search as a
  scalar subquery. ⛔ The generation binding is NOT relaxed by folding it in: `recall_chunks_v1`
  holds every generation ever built, retired ones included, and a subquery matching no row yields
  NULL, which matches nothing — exactly what the explicit early return did.

* **The Claude Code plugin could not be pointed at the table its corpus actually lives in.**
  `recall quickstart` indexes into `quickstart_chunks` so a sample corpus can never be retrieved
  beside real memory, while `recall setup` uses `chunks`, and the plugin hard-coded neither: it
  asked for a DSN and a tenant and then read whichever table the default named. Wrong table returns
  no hits and no error. `table` is now a plugin setting, asked rather than assumed.

* **The folder dimension was empty on the only build path that runs in production.**
  `metadata->>'file'` was the bare basename on the generation build path while the `Indexer` path
  stored a root-relative one, so every chunk looked as though it sat at the corpus root:
  `folder=recall` matched nothing and returned zero hits, which a caller cannot distinguish from a
  corpus with no answer. A manifest carries no root, so one is derived from the longest directory
  prefix its objects share. Found by running the pre-registered probe, whose oracle folder arm
  retained 0 of 31 controls; zero is not corpus drift.

  ⚠️ A hit's `source` is now `recall/memo.md` rather than `memo.md` on generation-backed tenants,
  and a `source=` filter must be given the relative path. That is the value `recall_search` has
  always documented as root-relative, so the field now matches its contract.

* **54 verified defects across graph, reasoning and calibration**, from a deep audit that ran nine
  auditors, adversarial verification and two differential reviews over the projection and
  calibration paths.

* **Security: four audit findings and three hardening changes.** A loopback-only database default,
  a narrowed CI token scope, a refusal when row-level security cannot be enforced, a non-root
  container image, a JWKS response ceiling, a bounded decompression limit, and live bandit rules in
  CI. The prompt-injection surface is now named in the docs so that searching for it lands
  somewhere.

* ⚠️ **Importing a calibration bundle now re-derives its threshold, so a bundle fitted under the
  RETIRED `best_threshold` rule is refused rather than imported.** The refusal names both numbers
  and the remedy is to recalibrate; it is the safe direction, but the first person to meet it will
  read it as corruption, so it is recorded here. The rule changed when the q95 ceiling stopped
  landing on the sample maximum at the certification minimum, which is the same fix that makes a
  20-sample calibration outlier-robust. Only `import_bundle` is affected: stored artifacts are
  read through `verify_checksum` and are untouched.

* **A forged calibration bundle could install any serving threshold.** `import_bundle` re-derived
  the certification verdict from the bundle's own scores, but `Calibration.certified` never reads
  the threshold, so trivially separable scores certified honestly while the threshold beside them
  was whatever the author wrote, and `publish` promotes any certified draft. A fitted artifact's
  threshold and scale are now re-derived from its scores too. A CARRIED artifact is still not
  authenticated by this check, because its threshold is inherited and its error bound is computed
  over scores the bundle supplies: that gap is stated in the function's docstring along with what
  would close it, rather than left to be assumed absent.

* **`--manifest-sha256` and `--manifest-size` were accepted and ignored on a local manifest.** They
  were read only to build the reference that fetches an `s3://` manifest; a `file://` build parsed
  the manifest and verified nothing. `bin/build_generation_voyage.sh` passes both on every run, so
  an operator watching those digests scroll past believed a check was happening. Silently
  discarding a checksum somebody supplied is worse than never accepting one, because it
  manufactures exactly the confidence it fails to earn. They are verified now, in every
  environment, whenever they are given.

* **A local corpus could not be built in production at all**, which left `RECALL_ENV=development`
  as the only route — and that variable also selects which table is read, so the workaround for
  this gate is half of what split indexing from serving. The other half was the hosted-embedder
  gate fixed alongside it.

  The `s3://` path provides three things: allowlisted objects, content-verified objects, and bytes
  pinned forever by object versioning. `LocalObjectReader` already delivered the first two — it
  refuses without `RECALL_LOCAL_ALLOWLIST`, resolves symlinks and `..` before the containment
  check, and hashes every object it reads. It cannot deliver the third, and its own docstring says
  so: a local file can be rewritten after its manifest is written, so this is detection, not
  prevention. Production now requires the two properties that are achievable, and refuses to
  pretend about the one that is not. A whitespace-only allowlist is refused with the unset case.

* **A hosted embedder could never back a production generation, so a hosted corpus could not accept
  an upload at all.** `require_production_identity` demanded an immutable revision or artifact
  digest, and a hosted provider has neither by definition: it serves weights it may replace behind
  a stable model name, which is exactly why `RegisteredProfile` refuses to let one pin a digest.
  The two rules together made the gate unpassable rather than strict, and the only way through was
  `RECALL_ENV=development` — which also redirects reads to the legacy `chunks` table, so the
  workaround for the gate split indexing and serving across two tables with nothing reporting it.

  `EmbedderIdentity` now carries `hosted`, and admissibility is a separate question from pinning.
  `verified` is unchanged and still false for a hosted endpoint, so every caller asking "are these
  bytes pinned?" keeps its answer and the lineage record still reads `verified: false`. Nothing is
  claimed that cannot be checked.

  Two details worth knowing rather than discovering. `hosted` is deliberately **absent from
  `to_dict`**, so no pipeline fingerprint moves and no existing corpus is stranded from its
  calibration history by the upgrade. And the exemption had to be made in **two** places: admitting
  hosted at the gate alone leaves `create` refusing one line lower on `allow_unverified`, which
  both callers pass as `not pipeline.verified` — permanently true for hosted. That version passes
  every unit test of the gate and changes nothing a user can observe; a mutation run is what
  caught it, along with a third case where the flag was wired but never set.

* **The Claude Code plugin could not reach the corpus `recall quickstart` builds, and said
  nothing.** The plugin collected a DSN, a tenant and a trust mode; the MCP server had no table
  knob at all, so it opened the default `chunks` while the quickstart had indexed into
  `quickstart_chunks`. Measured 2026-08-25 by driving the stdio server with exactly the plugin's
  three variables against a live quickstart database: `recall_search` returned
  `0 relevant memory hit(s)`, with no exception and nothing naming the table. A well-formed empty
  answer is indistinguishable from an empty corpus, so the documented first-run path read as "this
  product finds nothing". `RECALL_TABLE` now exists on the legacy single-tenant store, the plugin
  asks for it, and `recall quickstart` prints all four values in the shape the plugin asks for
  them. Under `RECALL_ENV=production` or authenticated tenant routing the variable is REFUSED at
  startup rather than ignored, because those stores read `recall_chunks_v1` and could not honour
  it. A value that is not a SQL identifier is refused at import.

* **`recall quickstart`'s Docker preflight cost 14 seconds at the median and 56 at the worst.**
  It ran `docker info`, which gathers the whole system inventory, so its cost scaled with what was
  on the machine. Against the 60 second timeout it carried, that bound was only 1.06x the worst case: a
  marginally busier daemon would have reported a perfectly healthy Docker as "installed but did not
  respond". Now `docker version`, measured at 0.48s median (29.1x) with the same verdict on a live
  and a dead daemon, under a 20 second timeout. Record:
  `docs/preregistrations/2026-08-25-docker-probe-latency.md`.

* **The README told you to apply the schema by hand at `--dim 384` before running `recall setup`.**
  `setup` migrates the database itself, at whichever width the embedder you pick needs, which is
  why choosing the embedder comes first. Verified against an empty database: every pending
  migration applied unprompted, `schema_status` compatible with nothing pending. The manual
  command is kept for the case it is actually for, a serving role that cannot create tables.


* **`recall doctor`.** Six independent failures in this product present as one of two symptoms
  ("no tools" or "no hits") and only one of the six names its own cause. This checks the
  interpreter, the package, the console scripts on `PATH` (which is what a plugin install with no
  `pip install` behind it fails on), the embedder backend, Docker, the database, pgvector, the
  schema, **whether the table and tenant you are configured for actually hold any chunks**, the
  calibration and the Claude Code registration. It names the populated corpus when the configured
  one is empty, prints the repair command for every problem, writes nothing, never constructs an
  embedder (that downloads a model), and exits non-zero only when something is blocked. `--json`
  for machines.

### Changed

* **`recall --help` now says which command to start with**, rather than only listing twenty. Four
  of them can each be read as "the install". `recall setup` and `recall wizard` each explain how
  they differ from the other, and ten install-path subcommands gained the `description` that
  `recall <cmd> --help` shows; eleven operate-an-existing-install commands still lack one and are
  held under a ratchet in `tests/test_cli_discoverability.py`.
* The README uses one invocation style throughout (`recall <cmd>`) instead of switching to
  `python -m recall.cli` halfway down, which is pinned by a test over its fenced code blocks.


## [0.10.0] (2026-08-23)

### Security

* **Right-to-erasure now reaches the learned-sparse sidecar.** `delete_sources`,
  `delete_sources_across`, `replace_sources` and `generations.forget` scrub
  `recall_sparse_v1` in the same transaction as the chunk delete (`DELETE ... RETURNING id`
  feeds the scrub, so the ids come from the delete itself). Before, a forgotten chunk's
  SPLADE term weights — partially reconstructable content over a 30,522-term vocabulary —
  survived every erasure path except `drop_table`. The `RECALL_ENV=production` refusal on
  the splade backend stays until an orphan sweep exists for corpora encoded before this
  fix, and that gate now also matches `Production` and ` production ` (the bare compare
  meant a capital letter silently disabled it).
* **Forgetting a source now also unlinks its staged upload file.** `recall_forget` erased
  the DB rows and left the original text under `RECALL_INDEX_ROOT/uploads/`, where the next
  index run would re-ingest it. Cleanup is best-effort after the committed delete, reported
  in the result (`staged_files_removed`, -1 on failure with a warning in the message), and
  hard-confined to the uploads tree: a source indexed from the user's own directory is
  never deleted.
* **BREAKING: `recall_calibration_publish` now requires the `recall:admin` scope.** Publication
  changes the serve/abstain decision for every query a tenant runs — the blast radius the admin
  scope was defined for, and until now no tool enforced it: any write token could publish. An
  HTTP deployment whose write token publishes calibrations must add `"recall:admin"` to that
  principal's `scopes` (static token file) or grant it in the IdP role (OIDC). Stdio and the
  desktop-local runtime are unaffected. `recall_calibration_run` stays on write: it produces a
  draft and changes nothing served. The requirement is advertised to clients in the tool's
  `_meta` (`recall/requiredScope`), and admin calls draw on their own `admin` call budget
  (`RECALL_RATE_ADMIN_PER_MIN`, default 10/min).
* **`recall_ingest` now debits the same per-tenant `index_bytes` quota as `recall_index`.**
  Before, only the 50 MiB per-request cap and the write call budget bounded uploads, so a loop
  under the per-request cap could ingest roughly 300× the intended hourly embedding spend
  unmetered. The debit lands after staging and before any embedding, so a refusal costs nothing;
  a refused or failed ingest also removes its staged files instead of leaving them inside the
  index root for a later index run to pick up.
* **`recall_tenants` no longer hands the full tenant inventory to every read token.** In a
  multi-tenant deployment tenant ids are often customer names. An authenticated principal now
  sees its own tenant; the full provisioned list requires `recall:admin`.
* **`recall_job_status` is tenant-scoped and the job ledger is bounded.** A foreign tenant
  probing a job id gets the same `unknown` shape as a nonexistent one, and completed jobs are
  evicted by count and age instead of accumulating for the life of the process.
* **Failed bearer-token authentication is throttled before any hashing or JWKS work.**
  A process-global failure budget (`RECALL_RATE_AUTH_FAILURES_PER_MIN`, default 60/min, `off`
  supported) closes the gate against a brute-force or forgery storm. Valid tokens never touch
  it, and an identity-provider outage deliberately does not debit it. Token-file entries
  provisioned by `token_sha256` digest are now named in the boot log, since their length can
  never be verified against the 32-character floor.
* **`host.docker.internal` no longer counts as a local host for the default-credentials guard.**
  From inside a container it reaches the container HOST, which can be a shared machine. The
  compose quickstart keeps working: the guard warns instead of refusing for exactly this host.
* **Duplicate file names in one `recall_ingest` upload are refused** instead of last-writer-wins,
  and an oversized entry is refused from its encoded length before being decoded into memory.

### Added

* **`recall.errors.RecallError` is the common base of every deliberate exception.** Sixty-five
  exception classes existed with no shared root, so a consumer could not write
  `except RecallError` and had to enumerate families or catch built-ins. Every family keeps its
  historical `RuntimeError`/`ValueError` base first, so existing handlers keep working, and a
  structural test walks both packages so a new family cannot silently opt out.

* **The ATM-Bench row was accepted onto the public leaderboard on 2026-08-23**, under `RAG`, and
  it leads both columns. Five documents and the run artifact said "open, not accepted"; they now
  say what happened, and the artifact keeps its `state_as_of_2026-08-22: open` key beside a new
  `state_as_of_2026-08-23: merged` rather than overwriting it.

  ⚠️ **Acceptance retires two of the three limits and not the third.** The judge transport was
  ruled admissible and the placement is real, but the QS column is still not answer-model-matched:
  the board lists this row as answering with DeepSeek V4 Pro beside baselines answering with
  `Qwen3-VL-8B-Instruct`. Recall@10 remains the like-for-like column, and every document that
  quotes QS still says so.

* **The ATM-Bench harness that produced the published run is now in this repository, byte for
  byte.** `benchmarks/atm_full_run.py` and `benchmarks/atm_bench.py` are copied from the run's own
  commit without a character changed, hashed in `results/atm/atm_harness_20260823.json`, and pinned
  by `tests/test_atm_runner_published.py`, which fails on a single changed byte. The leaderboard
  submission had pointed at a commit that was on no public branch, so the code behind a published
  number could not be read, let alone checked.

  Both files are exempt from `ruff` and `mypy` with the reasons stated beside the exemptions: a
  style fix would make the published harness a different program from the one that ran.

  ⚠️ **This publishes the harness, not the run.** `recall/` has moved since, the answer model is a
  moving alias, and the dataset stays outside this tree, so a re-execution reproduces the method
  rather than the last decimal. `docs/ATM_BENCH.md` section 6 states all three.

### Changed

* **The wheel no longer ships the one-off research drivers.** `recall/eval` keeps its
  load-bearing slice (the wizard's calibration engine, the documented `labelled` CLI, query
  generation, and the sample corpus and labels the README points the wizard at) and drops the
  session scripts, benchmark drivers and result fixtures that never worked from a wheel;
  `recall/wizard/llm.py` (a preregistered experiment arm imported by nothing but its test)
  stays repo-only until that experiment resolves. Working from a clone is unaffected.
* **Benchmark-harness tests carry a `benchharness` marker** (1,011 of 6,591), so a
  product-only run is `pytest -m 'not benchharness'`. CI behavior is unchanged.

* **`recall quickstart` goes from a fresh `pip install` to three answered queries in one command,
  database included.** It starts a throwaway pgvector container on a free port, applies the schema,
  indexes a corpus that ships inside the wheel, and runs three queries chosen to show the retrieval
  contract rather than to flatter it: one answerable, one whose nearest match is a claim that was
  later retracted, and one it refuses. `--remove` stops the stack and destroys its volume,
  `--existing-dsn` skips Docker for anyone already running PostgreSQL.

  It deliberately does NOT calibrate, register an MCP server, or build the `recall-wizard` image
  the full installer builds. Those are the slow steps, and each is printed as a named next command
  instead. Every result carries `DEGRADED:INDEX_NOT_READY`, which is explained in the output rather
  than suppressed: the corpus genuinely has no calibration, and hiding that in the demo would
  misrepresent the one property the project is about.

  ⚠️ **`recall demo` is not a substitute and cannot be.** It indexes the relative path `corpus`,
  which exists only in a git clone, so from a PyPI install it indexes nothing. `quickstart`
  resolves its corpus from the installed package.

### Fixed

* **`RetrievalDiagnostics` reports `max_dense_score`.** The published ATM harness records
  the best dense cosine per question, and that field had only ever existed on the run's own private
  branch: publishing the harness without it would have published a program that raises
  `AttributeError` on its first retrieval record. Additive, defaulted to `None`, and populated at
  both of the retriever's construction sites.

* **The unanswerable query in the demo corpus did not abstain.** `recall demo`'s "llamas on mars"
  scores a top cosine of **0.505** against the 0.50 development threshold on `recall/eval/corpus`,
  so it answers, out of `secrets_handling.md`. It was the obvious query to reuse for `quickstart`
  and reusing it would have captioned an answer as a refusal. Replaced by measurement with one that
  clears the threshold by 0.054, and `tests/test_quickstart.py` now asserts the margin rather than
  the boolean, so drift toward the edge is visible before it flips. A query's absence from the
  corpus does not imply abstention; only the cosine does.

## [0.9.8] (2026-08-22)

### Added

* **ATM-Bench full-split results are published, with their limits attached.**
  [`docs/ATM_BENCH.md`](docs/ATM_BENCH.md) records the 2026-08-21 run over the benchmark's 1,013
  personal-memory questions, scored by ATM-Bench's own evaluator: QS 68.4264 and Recall@10 92.8924,
  backed by the committed artifact `results/atm/atm_bench_full_20260821.json`. The retrieval figures
  were recomputed for publication from the run's own retrieval records and reproduce the submitted
  values exactly.

  `benchmarks/atm_answer_diagnosis.py` decomposes the answer-side loss with **zero provider
  calls**, by aggregating the official evaluator's own per-question judgements, and refuses to
  write its artifact unless the replay reproduces the published score. Its output,
  `results/atm/atm_answer_diagnosis_20260822.json`, backs section 5 of the document; the whole
  document is under the claim gate, so CI checks those digits against it.

  ⛔ **This is not announced as a leaderboard placement.** The submission is an open pull request
  rather than an accepted row, the QS column is not answer-model-matched to the published baselines,
  the judge ran over a disclosed non-official transport, and the run commit is not yet on a public
  branch. All four are stated in the document and recorded in the artifact.

* **`recall calibration drift` says whether the corpus under a live calibration has moved far
  enough to need refitting.** Until now the drift question could only be asked *after* a rebuild:
  `resolve` compares fingerprints and answers `STALE` on any mismatch, which is a yes/no about
  identity rather than a statement about magnitude, and `carry-forward` needs a new generation to
  already exist. Nothing could be asked the question an operator actually has between rebuilds.

  Two tiers, and which one produced the verdict is always in the output. The **screen** is a
  manifest comparison over `(uri, sha256)`: no embedding, no retrieval, not even a model load. The
  **probe** replays the calibration's own stored labelled query set and measures what the frozen
  threshold now costs, per class, by the same two conditions `carry-forward` enforces. ⛔ **The
  screen firing is never reported as a verdict**: where the probe cannot run the strongest verdict
  is `recalibrate_recommended`, and the report names the check that was not made, so a directory can
  never reach `recalibrate_required` however total its delta.

* **`RECALL_AUTO_CALIBRATE` and `recall calibration auto` re-establish a calibration without asking
  for labels.** After a generation build: `off` does nothing, `warn` (the default) reports, `auto`
  additionally carries the threshold forward, or refits it on the same stored labelled evidence
  when it has to move. **Neither path loosens certification**, so the automation is in what gets
  run and never in what gets accepted, and neither invents questions: a tenant with no published
  calibration reports `skipped`, because deciding what the labelled questions should be is not a
  decision to make unattended.

* **`recall calibration carry-forward`** and the `corpus_delta` / `threshold_error_rates`
  primitives, rescued from an unmerged branch and rebased onto master.

### Changed

* ⚠️ **There is deliberately no corpus delta at which recalibration is demanded outright, and this
  reverses the design this feature started with.** Measured over 57 snapshots of three real corpus
  histories (`docs/preregistrations/2026-08-21-calibration-drift-trigger.md`): the frozen threshold
  first crossed the 0.10 error bound at a delta of **0.945** and never below it, so a delta-only
  rule at 0.25 fires on **56 of 57** snapshots and is right about **5**, a precision of **0.09**.
  The labels also proved far more durable than that rule assumed: at delta 0.981 only **27.5%** of
  the answerable queries' original evidence still existed and the false-abstain rate was **0.025**.
  What moved was the false-*confirm* rate, tracking corpus **growth** rather than change as such.

  This does **not** license raising `--max-corpus-delta` on carry-forward, and it was left at 0.25.
  That bound governs whether a threshold may be *inherited*, and what the numbers establish is that
  a delta is a poor alarm, not that a large delta is safe.


- `recall uninstall` also removes the desktop app's handoff file from the user config directory
  (`%APPDATA%/RE-call/runtime.json`), which the previous release looked for in the wrong place.
- `--selftest` resolves the embedder only when a model cache already exists, and says which branch
  it took. It was downloading model weights on a cold cache.

### Fixed

- **A desktop upload no longer silently shrinks the corpus.** Carried-forward files that live
  outside the upload staging directory (which is every file a wizard install indexed) are kept and
  the build reader is widened to reach them. Only a file whose bytes are genuinely gone is dropped,
  and the count is now named in the upload's own message rather than passing unmentioned.
- **A failed `docker compose down` is reported.** With the docker daemon unreachable, `recall
  uninstall` recorded no failure, printed `Removed N item(s).`, and deleted the stack file naming
  the containers that were still running — which then made them unnameable by the tool that left
  them. The stack file and `wizard.json` are now kept whenever the teardown fails, so the uninstall
  can be retried.
- **An uninstall no longer overwrites the install-time backup of the MCP client config.** It writes
  its own under a name that is never reused, and copies the source file's mode rather than creating
  it at the umask default, since that file carries bearer tokens.
- **A volume the stack declared `external` is never removed.** A fallback that derived the
  historical volume name could reinstate exactly the volume that had just been excluded as not
  ours. The fallback now applies only to a legacy stack that declares no volumes at all.
- **A failure inside `build()` no longer strands a generation.** The desktop upload's cleanup path
  called only `abandon`, which refuses any state but `ready`, so failures before `validate()` left
  a full copy of the corpus that `gc` could not collect.
- **A blank or relative data folder is refused** rather than silently becoming the process's
  working directory, and the terminal interview reports it as a refusal instead of a traceback.
- **`corpus_version` may no longer begin with `desktop-`.** That prefix decides which generations
  the desktop upload path abandons; a wizard corpus carrying it would have been reclaimed.



* ⛔ **The graphical installer could not open its own window, and 0.9.7 shipped that way.**
  `install_main` called `run_window(InstallerWindow(...))`. Python evaluates an argument before the
  call, so the window was constructed before `run_window` had created the `QApplication` it needs.
  Qt answers a widget built with no application by printing `QWidget: Must construct a QApplication
  before a QWidget` and aborting the process on its fatal handler: one line of stderr, no window,
  no dialog, exit `0xC0000409`. Every copy of the 0.9.7 installer did this, on every machine.
  `run_app` had the identical call shape, so the main desktop window carried the same defect.

  `run_window` now takes a **factory** rather than a window, so the application is created first and
  the ordering cannot be decided at a call site. Passing an already-built widget raises a `TypeError`
  naming the mistake instead of killing the process.

* ⛔ **The self-test that exists to prove the bundle runs was green for that build, and now cannot
  be.** `--selftest` constructed its own `QApplication` and then its own window, in the correct
  order, which is not the order the entry point used. It rehearsed a launch sequence nothing ships.
  Every desktop test in the suite had the same shape, and not by carelessness: once any test in a
  pytest session creates a `QApplication` it cannot be unmade, so no in-process test can observe a
  process that has none. The self-test now builds its window through `application_and_window`, the
  same code the entry point orders its launch with, and `tests/test_desktop_launch_order.py` adds a
  check that starts the real entry point in a **fresh interpreter** and asserts on the exit code,
  which is the only thing here that reproduces a double-click.


## [0.9.7]

### Added

* **Production promotion works, gated on certification.** `generation promote` under
  `RECALL_ENV=production` used to refuse outright with "unavailable in production until
  certification gates land". Those gates have landed: promotion now succeeds for a generation whose
  published calibration certified and is still bound to this pipeline and corpus, and refuses every
  other status by name (MISSING, DRAFT, UNCERTIFIED, STALE each need a different action from
  whoever hit it). `--unsafe-development-promotion` is refused there rather than ignored, so the
  development escape hatch cannot be carried into production by habit.

* **A desktop upload into a production tenant can reach CERTIFIED.** The gate above had no
  reachable door on that path: nothing in `generation_ingest` produced a calibration, so every
  upload ended built-and-validated but never live, with the CLI as the only route to a live corpus.
  It now calibrates and publishes before promoting, using the same query-set generator the
  installer uses. Measured end to end against a production tenant: a ten-file corpus certifies and
  goes live; a one-file corpus reports "no certifiable query set could be generated from 1
  chunk(s)" and stays ready — the corpus's own reason, not the gate's generic one.

### Changed

* ⛔ **`generation rollback` never refuses on certification grounds, and this reverses a gate that
  shipped for one release.** Rollback is the incident path, and a gate that blocks recovery
  precisely when recovery is needed trades a visible degradation for an invisible workaround. Two
  ways the refusal bit, both certain rather than hypothetical: `forget()` rewrites the corpus
  fingerprint of every generation of a tenant, so one erasure request left no rollback target ever
  again; and every generation an existing install is serving was promoted under `development` and
  has no published calibration, so upgrading would have removed rollback from all of them.

  The invariant is kept by REPORTING instead of preventing. `generation_rolled_back` now records
  the target's resolved calibration status and an optional `provisional_reason`, so a recovery that
  downgrades a tenant from certified to provisional is visible. Reasoning:
  `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`.

* **The certification gate follows the SERVING environment, not the build one.** `GenerationManager`
  takes `serving_environment`, defaulting to `environment`. The wizard is why: it builds every
  corpus under `development`, because a production build demands a verifiable embedder identity a
  bundled model does not have, then serves those tenants with `RECALL_ENV=production`. Keyed on the
  build environment, the gate ran on no tenant an install creates.

* **One definition of a pipeline identity.** `generation_ingest` assembled its own, hardcoding
  `provider="fastembed"` for every embedder and spelling out a chunker identity with an empty
  configuration. A generation built by the wizard and one built by a desktop upload therefore
  carried different pipeline fingerprints for the same pipeline, which is what makes a published
  calibration resolve STALE. Chunk boundaries are unchanged; only the recorded identity moves.

* **The `fastembed` extra is bounded to `<1`.** `embedder_artifact_path` walks a private attribute
  chain to find a model's own snapshot directory, and does so defensively — a release that renames
  it returns `None` rather than raising, which makes the pipeline identity unverified, which makes
  production builds refuse. An install that silently stops accepting uploads, with nothing pointing
  at the dependency that moved. Verified reachable on fastembed 0.8.0.

### Fixed

* **The certification gate decided from a different transaction than the one it authorised.** It
  opened its own connection while `promote` held the tenant and generation rows `FOR UPDATE`, so a
  concurrent `forget()` could invalidate the calibration between the verdict and the commit, and the
  connection was acquired while holding a lock. Measured: it is **not** a deadlock, which is what it
  looks like — a plain `SELECT` does not wait on `FOR UPDATE` under MVCC. Now resolved on the
  caller's transaction; a competing fingerprint rewrite waits for the promotion rather than racing
  it.

* **The artifact digest cache vouched for bytes that had changed.** Keyed by path with no
  invalidation, the first answer stood for the life of the process, so a re-download or a swapped
  model file kept a provenance claim no bytes on disk supported. Now keyed by file count, total size
  and newest mtime alongside the digest. This detects staleness, not tampering: someone able to
  write into the model directory can also set mtimes.

* **A generated stack could serve one version while every file on disk claimed another.** Adding a
  project inherits the existing stack's image tag so the new corpus runs the same recall as its
  siblings, but the Dockerfile was regenerated at the running version. Measured on a 0.9.1 stack
  under a 0.9.6 wizard: tag `recall-wizard:0.9.1`, Dockerfile `recall-rag==0.9.6`. Compose reuses a
  tag rather than building it, so the container would have started 0.9.1 in silence. The Dockerfile
  now follows the tag.

* **A refused upload no longer loses the previous upload's files or leaks its corpus.** A new build
  seeded its manifest from the ACTIVE generation, and a refused promotion never advances that
  pointer, so each upload dropped every earlier un-promoted upload's files. Measured against a real
  database: upload #2 reported 1 file where it should have reported 3. Builds it supersedes are now
  abandoned so `gc` can reclaim them.

* **Six operator documents claimed production promotion was blocked.** Each was true when written
  and falsified by the gates landing, and each was the document somebody reads before deciding what
  a command will do: `PRODUCTION.md`, `MIGRATIONS.md`, `CALIBRATION.md`, `GENERATIONS.md`,
  `FIRST_CALIBRATION.md`, `ENVIRONMENT.md`.

## [0.9.6]

### Added

* **The `documents` extra is now published.** It has existed in `pyproject.toml` for some time but
  no release ever carried it, so `pip install "recall-rag[documents]"` installed nothing and said
  so only in a warning:

      WARNING: recall-rag 0.9.5 does not provide the extra 'documents'
      --- pip exit: 0 ---

  pip exits **zero** for an extra a release does not provide, so every install of it succeeded and
  silently omitted `pypdf`, `pdfplumber`, `python-docx`, `openpyxl`, `python-pptx`, `xlrd`,
  `beautifulsoup4` and `python-oxmsg`. Anyone who installed it got a recall that accepted `.pdf`,
  `.docx`, `.xlsx` and `.pptx` files and extracted nothing from them, which reads as recall being
  bad at documents rather than as a missing dependency.

  Found while giving the Windows wizard's generated Docker stack an image it could build: the
  Dockerfile pins the running version, and a post-install import check turned the silent omission
  into a build failure.

### Changed

* ⚠️ **BREAKING, and it costs one full re-index: the incremental skip guard now keys on the whole
  embedding identity rather than the profile id.** `index._index_fingerprint` hashed
  `embedding_profile_id(embedder)`, which is ONE field of an `EmbeddingProfile`, so any two
  embedders sharing an id were the same embedder as far as the skip guard was concerned. It now
  hashes `EmbeddingProfile.fingerprint()`, bringing it into line with `recall/cache.py`, which has
  always keyed on the whole profile and whose docstring gives the reason: "The ID alone is not an
  identity".

  The hole was reachable. A 384-dimension corpus and a 1024-dimension corpus produced **equal**
  index fingerprints for the same file, so swapping the model left every file skipped, every vector
  stale, and the run reporting success with a skipped count. Ten identity fields now reach the
  fingerprint that did not before: model name, artifact digest, dimension, both encoder modes,
  normalization, instruction version, chunker version, context version and dependencies.

  **What you have to do.** Every `index_fingerprint` already stored in chunk metadata is now
  stale, so the next `recall index` over an existing corpus re-reads, re-chunks and **re-embeds
  every file, once**. Measured: the default `FastEmbedEmbedder()` moves from `a93f4428…` to
  `1832d370…`. There is no cache to soften this on the shipped paths, because nothing in `recall`
  or `recall_mcp` passes an `EmbeddingCache` to the `Indexer`; only the benchmarks do. Budget the
  re-embed on a metered embedder accordingly, and prefer to run it deliberately rather than
  discovering it inside an unattended job. Nothing is lost if you do not: the corpus keeps serving
  its existing vectors until it is re-indexed, since `index_fingerprint` is only ever compared to
  itself.

  **New re-index triggers, deliberately.** `dependencies` carries the inference-library version and
  the ONNX execution providers, so a fastembed upgrade or a CPU-to-CUDA move now re-fingerprints
  the corpus. That is the trade `EmbeddingProfile.fingerprint` already makes for the cache, for the
  same stated reason: a runtime change is free to move the last bits of a vector, and neither a
  cache nor a skip guard can tell. The two agree now instead of disagreeing.

  **`ContextPolicy.max_tokens` is now covered too**, in the same change and for the same reason.
  It selects a different rung of `contextual_passages`' degradation ladder, so two policies
  differing only in it build different passages and used to hash equal. It had been carried as a
  known gap on one ground only, that closing it re-fingerprints every corpus, and that cost is
  being paid here regardless; deferring it again would have charged a second full re-embed later
  for a one-term change. No shipped path sets it (`context_policy_for_profile` leaves it unset),
  so this widens the identity without moving any shipped corpus further than the paragraph above
  already does.

  ⚠️ `ContextPolicy.tokenizer` is deliberately **not** covered, and that is a decision rather than
  an oversight. It changes the passage exactly as `max_tokens` does, but a callable has no identity
  stable across processes: `__qualname__` collides for closures and lambdas and `id()` differs
  every run. An unstable term is far worse than a missing one, because the fingerprint would differ
  from itself and re-embed the whole corpus on every single run, silently and permanently. Closing
  it needs a caller-supplied stable tokenizer identity. A test pins the current behaviour so the
  limit is recorded rather than assumed.

### Fixed

* **An embedder built without a registered profile claimed `bge-small-symmetric-v1` whatever model
  it actually was.** `FastEmbedEmbedder.__init__` minted that literal (or `bge-small-asymmetric-v1`)
  unconditionally on the no-identity path, so the id was a label rather than a claim. Measured
  2026-08-18: a `fastembed:BAAI/bge-large-en-v1.5` embedder reported `dim=1024` under
  `profile_id='bge-small-symmetric-v1'`, whose registry entry is 384-dimensional, and a production
  corpus of 8,716 chunks had stored that pairing in its chunk metadata. The fallback id is now
  derived from the model name, dimension and encoder modes
  (`unregistered__BAAI__bge-large-en-v1.5__1024__symmetric`) unless the embedder genuinely is the
  model the legacy literal names, at the width the registry declares for it. The `/` is replaced
  because a profile id is interpolated into a result filename by
  `recall.eval.promotion.run.ArmConfig.key`, the same reason `SparseProfile` already did this.

  The embedding cache was never affected: `EmbeddingProfile.fingerprint` already covers
  `model_name` and `dimension`, so the two models keyed apart despite sharing an id. What was
  affected is `recall.index._index_fingerprint`, which hashes the profile id alone with no
  dimension term of its own. A bge-small corpus and a bge-large corpus therefore produced the
  **same** fingerprint for the same file, so the incremental skip guard treated a model swap as a
  no-op and left stale vectors in place. Verified by execution before the fix: the 384-dimension
  and 1024-dimension fingerprints were equal.

  ⚠️ **Scope of the change, which is narrower than it looks.** The default `FastEmbedEmbedder()`
  is bge-small at 384 and keeps `bge-small-symmetric-v1`, so its index fingerprints are
  byte-identical to before (verified) and no default corpus needs re-indexing. Registered
  enterprise profiles never reach this path at all. Only a corpus indexed with an *unregistered*
  model changes id, and for those the change is the repair: the next `recall index` run sees a
  different fingerprint and re-embeds, replacing vectors whose recorded provenance was false.
  Search keeps working in the meantime, because the stored `embedding_profile` metadata is
  reported as a diagnostic and never compared at read time. One thing to re-do deliberately: a
  calibration file fitted for such a corpus was written under the wrong id and will stop
  resolving, which is correct (a bge-small-keyed threshold does not transfer to bge-large cosines)
  but shows up as "uncalibrated" until it is re-fitted.

* **`recall generation build` recorded an overlap the chunker never used, and correcting it moves
  the pipeline fingerprint.** The chunker clamps overlap to `max_chars // 4`; the generation's
  `ChunkerIdentity` recorded what was asked for. So the record described a pipeline that did not
  run, and it was reachable with default arguments: `--max-chars 200` with the default overlap of
  80 chunked at 50 and recorded 80. Two configurations producing byte-identical chunks therefore
  fingerprinted differently, and a calibration binds to that fingerprint.

  ⚠️ **This is a deliberate break on rebuild, in exactly one region.** The recorded value changes
  only when `overlap > max_chars // 4`, which at the default overlap means any `--max-chars` below
  320. The default 800/80 is unchanged. Generations already in a database are immutable and
  unaffected, but rebuilding such a corpus with identical flags now yields a different
  `pipeline_fingerprint`, which costs the cross-generation chunk reuse keyed on that column and
  the binding of any calibration measured against the old generation. Re-run
  `recall calibration calibrate --publish` against the new generation. The pre-upgrade record was
  false, so there is no version of this that is both correct and non-breaking. `recall index` is
  unaffected: it takes neither flag and builds no pipeline identity.

* **The MCP server now honours `RECALL_TRUST_MODE`.** `docs/USING_WITH_CLAUDE.md` has told users to
  set it since the document was written, and it did nothing: the variable appeared nowhere in
  `recall_mcp`, and `search_memory` and `evidence_memory` were both called without `policy=`, so the
  service applied its strict default. Following the documented first-run path therefore produced
  `INDEX_NOT_READY` on every `recall_search` against a freshly indexed corpus, with the one
  documented remedy inert. The CLI honoured the same variable throughout, which is what let the gap
  survive: the same setting worked in one entry point and was silently ignored in the other.
  Strict remains the default, a misspelling such as `developmnet` still stays strict, and a relaxed
  server now logs a warning at every start rather than degrading quietly.

### Added

* `recall extract run|show --status-vocabulary W,X,Y` lets a corpus that states status in its
  own words, not the shipped memo set, be measured without every such claim being refused at a
  batch rung. It does not widen what `recall rewrite` may write: the write path still extracts
  under the shipped vocabulary and `route_relation` still refuses anything outside it.
* Promoted deterministic answer slot selection to a supported optional evidence path. Public
  `AnswerSlot` and `EvidencePolicy` exports, plus the LangChain and LlamaIndex evidence adapters,
  can require multiple answer components and abstain with `answer_slot_gap` when one is absent.
  Existing retrieval ordering remains the default. Beam selection and reasoning remain opt in.

## [0.9.5] (2026-08-15)

### Added

* Added model backed truth extraction: `recall/truth_extraction/`, turning memo prose into
  structured, quoted claims behind a refusing validation ladder. Off unless
  `RECALL_TRUTH_EXTRACTION=1`, runs on the ingest path only, never the query path. The
  extraction engine is a port with two implementations, a deterministic rules reference and an
  OpenAI compatible model engine (`pip install "recall-rag[extract]"`); whatever an engine returns
  clears the same ladder, so a model gains no ability to skip a rung.
* Added `recall extract run|show`, which reads a corpus and writes nothing, and
  `recall rewrite plan|apply|reject|verify`, which declares reviewed claims in corpus
  frontmatter. `recall rewrite apply` is a dry run by default and requires `--reviewer` and
  `--note` as argparse requirements, so the named human gate fires before any code runs.
* Added the `recall_rewrite_plan` MCP tool, read only. There is deliberately no
  `recall_rewrite_apply`: the MCP client is the model, and a reviewer id it can type is a field
  rather than a person. `recall_reasoning_proposals` and `recall reasoning proposals` gain
  `include_extracted`, defaulting to off so existing behaviour is byte identical.
* Added `recall extract run --cache PATH`, a persistent SQLite extraction cache, so re-ingesting
  an unchanged memo does not re-pay the engine for it. Entries are keyed on engine identity,
  engine revision, prompt revision, the file, its body and the corpus names, so an answer
  produced under one engine is never served for another. A path that is not a usable cache is
  refused before any engine call, a corrupt row is a miss and is re-paid, and a failed write is
  counted and reported rather than discarding the files already extracted. `--cache` was briefly
  a boolean because an earlier version accepted a PATH and ignored it; the flag came back when
  the persistence did. See [docs/archive/EXTRACTION_CACHE_DESIGN.md](docs/archive/EXTRACTION_CACHE_DESIGN.md).

### Fixed

* Fixed `recall extract run` aborting a whole corpus on one filename that is not valid UTF-8.
  A POSIX name arrives as a lone surrogate through `Path.glob`'s surrogateescape, and it raised
  twice: once hashing the cache key, which is computed for every file whether or not a cache is
  in use, and again printing the report, because reconfiguring stdout's encoding resets its
  error handler to strict. The first discarded every file already extracted; the second threw
  away a completed extraction at the last step, exiting 1 with empty output. Such a name is now
  reported with its bad bytes escaped.

### Changed

* **BREAKING: `PROPOSAL_SCHEMA_VERSION` moved from 1 to 2**, which rewrites **every** `ip_`
  proposal id in existence, including the checked in `results/reasoning_session3_proposals.json`.
  Version 2 adds `declares_validity` and `declares_status` to `ProposedRelation`, because a
  document asserting something about ITSELF is not a relation between two documents and forcing
  it into `references` would put a false relation into an audit record. The bump is the point:
  an id minted under a vocabulary that could not express validity must not be mistaken for one
  minted under a vocabulary that can. Anyone holding stored `ip_` ids must re-derive them.

### Fixed

* MTRAG Task B and C generation no longer scores an answer that the token ceiling cut off.
  `benchmarks/mtrag/generation.py` sent `--max-tokens` (512 by default) and never read
  `finish_reason`, so a truncated completion was written to the submission and judged as if the
  system had produced it. It now raises `CompletionTruncated`, unretried because the same ceiling
  cuts every further attempt, and the existing per-task quarantine keeps the task out of the
  submission and in the failures log.

### Added

* `recall setup` gains an optional reasoning arm step, asked after the entailment judge question
  and before the CLAUDE.md scaffold question. Answering yes writes four new environment
  variables: `RECALL_REASONING`, `RECALL_REASONING_MODEL`, `RECALL_REASONING_BASE_URL`, and
  `RECALL_REASONING_API_KEY`. Answering no writes `RECALL_REASONING=0` and nothing else, so
  "switched off" and "never configured" stay distinguishable in `.env`. The shipped reasoning
  tools do not read these variables yet; this writes the settings for a port the reasoning arm
  will use once it is built. See
  [docs/archive/REASONING_MODEL_SELECTION_DESIGN.md](docs/archive/REASONING_MODEL_SELECTION_DESIGN.md).

### Fixed

* Corrected `recall setup`'s refusal message for an embedder whose vector width conflicts with a
  table that already holds data. It previously pointed at a remedy that failed identically to the
  original problem. It now stops and tells you to choose an embedder matching the existing
  schema, or point setup at a fresh table name or database.

## [0.9.4] (2026-08-12)

Released straight from 0.9.2. 0.9.3 is deliberately skipped and will never exist.

Most of this release came from walking the documented quickstart on a clean machine as a new
user would, which found five defects on the path every new user takes.

### Added

* `recall setup` offers bge-base (768 dims) and bge-large (1024 dims) beside bge-small, each
  gated on having room for its own weights rather than on the shared download floor.
* `recall setup` refuses an embedder whose vector width does not match the table it will write
  to, naming the schema command that fixes it. Previously the mismatch surfaced on the first
  write, after the model had downloaded and the corpus had been read.
* `recall setup` lists retrieval options this machine cannot run yet, marked `(not installed
  yet)`, and prints what to install when one is chosen. They were previously hidden, which made
  the feature look absent and left no way to ask for it.
* `recall setup` scaffolds a `CLAUDE.md` and a `memory/` directory for the project, and indexes
  that directory once it exists.
* Added provider execution metadata for reasoning diagnostics and benchmark artifacts, including
  provider id, model id, model revision when available, token counts, latency, and monetary cost
  when providers expose it.
* Added a reviewed inference proposal promotion workflow with separate promoted fact records.
* Added experimental reasoning release notes covering opt in use, provider neutrality, citation
  constraints, CLI and MCP migration notes, serialized fields, limitations, and evaluation posture.

### Changed

* PostgreSQL 18 compliance is now declared and tested: local Docker uses
  `pgvector/pgvector:pg18`, CI runs schema migrations on PostgreSQL 16, 17, and 18, and the main
  integration job runs on PostgreSQL 18.
* Ported the MCP server to MCP Python SDK 2.x and raised the `mcp` extra floor to `mcp>=2,<3`.
* Raised the development Ruff range to `ruff>=0.16,<0.17` while keeping the prior lint baseline
  explicit in `pyproject.toml`.
* The shipped calibration sample now holds twenty answerable and twenty unanswerable queries.
  It previously held fourteen and five, below the certification floor, so following the
  documented path produced an explicitly uncertified threshold.
* The quickstart no longer assumes a clone. The compose file is inline, so `pip install` alone is
  enough to follow it, and the sample corpus that ships beside `recall/eval/queries.json` is now
  pointed at rather than left for the reader to discover.

### Fixed

* The quickstart's schema command failed on every fresh database. It passed a custom `--table`,
  but global migrations must be applied through the default target first, so a new user following
  the README exactly got `SchemaTooOld` from inside the library.
* The terminal video renderer writes its GIF before attempting the MP4, so a missing optional
  dependency no longer discards an asset that never needed it, and reports which file it did not
  write and why.

## [0.9.2] (2026-08-10)

### Added

- Added official MCP Registry metadata in `server.json`.
- Added the PyPI ownership marker required by the MCP Registry.
- Added the `recall-mcp` console script for registry clients using `uvx`.

## [0.9.1] - 2026-08-10

### Fixed

- Rebuilt the package description from the current GitHub README so PyPI no longer shows the stale
  pre-cleanup product copy or the incorrect MIT license sentence from the old buyer table.

## [0.9.0] - 2026-08-09

### Added

- Multi-query fusion through `HybridRetriever.search_fused(query, history, k, source)`.
- Reachable evidence boundary through package exports, CLI `--evidence`, MCP `recall_evidence`,
  and LangChain and LlamaIndex adapter methods.
- Request-time retrieval budgets, overload refusals, stage timings, and profile-sized MCP worker
  pools.
- FastEmbed resolved-provider reporting and provider-aware profile fingerprints.

### Changed

- Fast and quality retrieval profiles now carry separate concurrency and queue budgets.
- The quality reranker is pinned by artifact digest and refuses mismatched local trees.
- FastEmbed profile fingerprints changed, so profile-bound calibrations should be re-fitted.

### Fixed

- Restored package entry points, README sections, dependency declarations, and release smoke
  coverage that had been dropped during a merge.
- Split duplicate benchmark helper modules so pytest no longer collects colliding test names.

Full release detail: [docs/archive/CHANGELOG_FULL.md](docs/archive/CHANGELOG_FULL.md).
