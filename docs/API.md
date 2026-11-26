# Supported API

This page names the public integration surface. Anything outside these paths may support
benchmarks, migrations, or experiments, and can change more freely.

Compatibility and upgrade rules for this surface are in [COMPATIBILITY.md](COMPATIBILITY.md).

## Python

| Surface | Import | Purpose |
|---|---|---|
| Trust search | `recall.trust.trusted_search` | Return verdicts, confidence, provenance, and abstention state. |
| Reasoning | `recall.reasoning.reason` | Run explicit opt-in reasoning from trusted retrieval, bounded provider ports, graph projections, and citation validation. |
| Reasoning graph | `recall.reasoning_graph.build_reasoning_graph` | Derive immutable, generation-bound authored and semantic graph projections for reasoning and proposal inspection. |
| Embeddings | `recall.embeddings.resolve_embedder` | Construct supported embedding backends from configuration. |
| Multimodal provenance | `recall.multimodal.build_media_ref`, `project_media_evidence` | Validate bounded media references and authorize original object references in the isolated multimodal tenant. |
| Generation store | `recall.generation_store.GenerationStore` | Serve immutable, tenant-scoped generations. |
| pgvector store | `recall.store.PgVectorStore` | Local indexing and retrieval over PostgreSQL plus pgvector. |
| Related evidence | `recall.related.trusted_related` | Opt in, independently trusted source, ordinal, supersession, or `caused_by` related evidence, returning at most 50 items. Supersession evaluates the complete matching lineage before limiting returned items; source, ordinal, and `caused_by` relations also limit candidates. `caused_by` follows the regression edge between audit records in both directions. |
| Audit records | `recall.records.lookup_records`, `admit_record` | Exact lookup of the `recall_record` objects memos carry (schema `recall/schemas/recall_record.v1.json`), by anchor, id or full dump, each returned exactly as authored with its anchor match; and the index-time admission that keeps a valid record and drops an invalid one with a warning naming the file. No embedding, calibration or abstention. |
| Source security | `recall.security_policy.SourceSecurityPolicy` | Enforce tenant bound source authorization, operation purposes, classification, egress, and redaction at indexing and serving boundaries. |
| Current state | `recall.current_state.project_current_state` | Pure, generation-bound authored state projection. |
| Query routing | `recall.query_class.classify_query` and `route_query` | Versioned deterministic query classes and shadow routing decisions. |
| LangChain | `recall.integrations.langchain.RecallRetriever` | Use RE-call as a LangChain retriever. |
| LlamaIndex | `recall.integrations.llamaindex.RecallRetriever` | Use RE-call as a LlamaIndex retriever. |
| Claude Agent SDK | `recall_agent.RecallAgentMemory` | In-process SDK tools, SessionStart digest, and `ClaudeAgentOptions` assembly for Claude Agent SDK apps ([USING_WITH_AGENT_SDK.md](USING_WITH_AGENT_SDK.md)). |
| Serving JSON | `recall_mcp.service.serving_json` | The one renderer every serving surface (MCP server, Agent SDK tools) uses, so results are byte-identical across transports. |
| Errors | `recall.errors.RecallError` | Common base of every deliberate recall/recall_mcp exception. Each family also keeps its historical built-in base (`RuntimeError` or `ValueError`), so existing handlers keep working. |

The expected application pattern is to call `trusted_search`, check `result.abstained`, and answer
only from returned hits whose verdict and provenance satisfy the caller's policy.

## Unknown-term vocabulary check

`trusted_search` checks entity-like query tokens when the store supports a global vocabulary
lookup. Lite supports it through FTS5, and a PostgreSQL generation store through the active
generation's full-text index (`tsv`, its own configuration); the legacy PostgreSQL route does not.
The heuristic selects tokens of at
least three characters containing a letter and a digit, all-capital names, or an inner capital.
Plain words are not checked. An absent token demotes otherwise `ok` hits to `unknown_term` and
the result abstains. Other verdicts and raw candidate scores retain their existing meaning.

Python results expose `diagnostics.unknown_term_check` and `diagnostics.unknown_terms`. Search
and evidence JSON, including in-process agent responses, expose those two fields directly:

| State | Meaning |
|---|---|
| `checked` | The vocabulary check completed. `unknown_terms` lists absent query tokens. |
| `unavailable` | The store lacks this capability, or a constrained request cannot safely use its global vocabulary. No lookup of excluded sources occurs and verdicts are unchanged. |
| `disabled` | `RECALL_UNKNOWN_TERM_GATE` disabled the optional check. |
| `not_run` | No check was reached or an older payload omitted these additive fields. |

An unavailable vocabulary check is not evidence that a term was absent. Calibration and
`trust_state` still govern trust; this check does not replace them. Source, folder, facet and
authorization constraints report `unavailable` until scoped vocabulary checking is supported.
`RECALL_UNKNOWN_TERM_GATE` defaults to `1`; `0`, `false`, `no`, or `off` disables it, ignoring case
and surrounding whitespace. An explicitly supplied `env` mapping controls the request even when
the process environment differs. See [ENVIRONMENT.md](ENVIRONMENT.md).

## Command Line

Every `recall` subcommand, in `recall --help` order. `tests/test_api_doc_drift.py` diffs this
table against the registered parsers, so a command cannot ship undocumented or linger here after
removal.

| Command | Purpose |
|---|---|
| `recall setup` | Guided local setup: embedder and reranker choices and optional CLAUDE.md/memory scaffolding. Without `--lite` it does not calibrate; `recall wizard` does. With `--lite` it calibrates the store it builds. Refuses a flag it would not apply: `--embedder` (the interview chooses it), `--tenant` other than `default`, and `--memory`, `--store` and `--no-register` without `--lite`; with `--lite`, `--tenant` and `--table` other than their defaults and a typed `sqlite:///` `--serving-dsn` (`--dsn`). |
| `recall wizard` | Build, calibrate, publish, and promote a generation, interactively or from a saved config with `--headless --config` ([WIZARD.md](WIZARD.md)). Ordinary `recall setup` configures the installation and neither calibrates nor builds a generation. |
| `recall uninstall` | Remove what setup installed: MCP registrations, hooks, and optionally the database stack. |
| `recall doctor` | Diagnose an install end to end and change nothing: interpreter, package, console scripts on PATH, embedder backend, Docker, database, pgvector, schema, whether the configured table and tenant actually hold chunks, calibration, and the Claude Code registration. Prints the repair command for each problem. `--json` for machines. Exits non-zero only when something is blocked, so a missing calibration does not fail a script. |
| `recall quickstart` | From a fresh `pip install` to a real answer: start a throwaway PostgreSQL, index the bundled 22-document demo corpus into `quickstart_chunks`/`quickstart`, answer three queries, and print the values the Claude Code plugin asks for. `--remove` destroys it. Calibrates nothing and registers nothing. |
| `recall schema` | Apply, inspect, and plan PostgreSQL schema migrations (`status`, `plan`, `apply`, `grants`), and rewrite the chunk and fact tables after erasure (`compact`, PostgreSQL 19 or later). |
| `recall manifest` | Build and verify index manifests (`create`, `inventory`, `verify`). |
| `recall lite` | The one-file local store (`recall setup --lite`). `refresh` indexes what changed in the memo folder under the store's write lane and recalibrates; `--check` reports what a refresh would find and writes nothing. The MCP server runs the same refresh on its own when `RECALL_INDEX_ROOT` names the folder: at start, and before a search at most every `RECALL_LITE_REFRESH_INTERVAL` seconds (30), waiting up to `RECALL_LITE_REFRESH_WAIT` (3); `RECALL_LITE_AUTO_REFRESH=0` turns it off. |
| `recall generation` | Immutable generation lifecycle (`build`, `validate`, `promote`, `abandon`, `rollback`, `list`, `gc`, `stated-facts`: record the facts a generation's memos state, the active one by default; `rescue`: build the atomic rescue artifact a generation serves with, the active one by default, which `promote` and `rollback` do themselves under an active rescue). |
| `recall graph` | Inspect (`status`) or rebuild (`rebuild`) the deterministic Evidence Graph V1 without changing chunks or generation identity. On PostgreSQL `rebuild` takes `--generation`; on a lite store the graph belongs to the file's one corpus, so `--generation` is optional (see [LITE.md](LITE.md#see-how-memos-link)). |
| `recall index` | Index a folder of supported documents or code. |
| `recall forget` | Permanently erase indexed sources; the right-to-erasure path. Facts whose only evidence they were are retracted, and the values the sources supplied are erased from the fact ledger and the evidence cards (rewritten in place under fresh ids); the replay receipts naming them are deleted. `--vacuum` also compacts a lite file. |
| `recall withdraw` | Erase indexed sources that were a mistake, as `recall forget` does (fact values included), and give back what they superseded: their `supersedes:` claims become void and the chain is spliced; facts whose only evidence they were are withdrawn, which also undoes the supersessions those facts made. Previews by default, `--yes` to act; served on a lite DSN. |
| `recall search` | Query an indexed corpus through the trust layer. |
| `recall scopes` | List the folders or facets a search can be filtered by, with their sizes. |
| `recall reasoning` | Inspect projections (`projection`), proposals (`proposals`), queries (`query`), traces (`trace`), audits (`audit`), and opt-in reasoning without changing ordinary retrieval behavior. Query and trace accept `--graph-expansion auto|off|one-hop`; `auto` uses bounded global one-hop expansion for nonempty queries. |
| `recall extract` | Extract structured truth claims from memo prose (`run`, `show`). Reads only; writes nothing. Off unless `RECALL_TRUTH_EXTRACTION=1`. |
| `recall rewrite` | Review extracted claims and model-judged supersession pairs (`plan`, `apply`, `reject`, `verify`) and declare accepted ones in corpus frontmatter. Dry run by default; `--reviewer` and `--note` are required. The supersession arbiter is off unless `RECALL_SUPERSESSION_ARBITER=1`. |
| `recall dashboard` | Open a local review page (127.0.0.1 only) for supersession proposals: agent reports from `recall_report_stale` (the local sidecar, and with a connected corpus database also the `stale_report` rows of the tenants named by `--reports-tenant`, defaulting to the SQLite file's tenant or `default` for PostgreSQL, the tenant every RE-call surface opens, mapped to this folder's memos and their quotes recovered from them) and cached arbiter judgements. Accepting writes `supersedes:` into the newer memo exactly as `recall rewrite apply` does; rejecting records the claim so it is not proposed again. Also a reader that renders each memo with what replaced it and what links to it, a 3D memory graph, a person-only editor for `supersedes`, validity dates and status, recent activity and memory health, all from the memo files. Without a DSN file it automatically reads a SQLite store from `RECALL_SERVING_DSN`, then `RECALL_DSN`. `--db-dsn-file` takes precedence and is required for PostgreSQL through a SELECT-only role (optionally with `--tunnel HOST`; SQLite rejects a tunnel). Scoped database reads set the selected tenant context for row-level security. Tenant inventory requires an authorized cross-tenant reader or a role with an explicit `recall.tenant_id` session default; a missing context is reported as unavailable. The corpus database is read over a read-only session: tenants with their embedder, size and calibration, generations, the searches recorded by the decision ledger (`RECALL_DECISION_LEDGER=1`), and a Control page counting, per memory, how often search returned it beside what agents reported with `recall_report_use`. A home page lists what needs attention; a top bar finds memos and switches corpus. The corpus picker lists the tenants that exist, the default one first; `--tenants NAME[=LABEL],...` (or `RECALL_DASHBOARD_TENANTS`) puts your own main tenants first, with labels, and only a pinned tenant that is missing is shown as not built. With the database it also draws any corpus's graph from the database (`/graph?corpus=TENANT`: supersessions, authored links and Python imports between files, plus files that share an uncommon name) and opens one file's page from it (`/corpus/source`): its state, what replaces it, its relations, imports and chunks. A Configuration page shows what RE-call runs with on this machine, in each RE-call MCP server registered for the project, and (with `--config-host HOST`) on a remote host, each report produced by `python -m recall_mcp.config_report` where the values live, so a value is withheld there when its key is flagged secret or named like a credential, and the credential shapes it recognises (URL userinfo, credential query parameters, keyword passwords, authorization values) are replaced in the values it shows; a credential in another shape is shown. Calls no model. |
| `recall demo` | Index the sample corpus and run example searches. |
| `recall code` | Index RE-call source code and run example code searches. |
| `recall lint` | Validate memo frontmatter and corpus shape. |
| `recall check` | Validate one memo, optionally in strict mode. |
| `recall calibrate` | Fit an abstention threshold from a labeled query file (legacy single-shot form). |
| `recall calibration` | Calibration artifact lifecycle (`calibrate`, `carry-forward`, `drift`, `auto`, `list`, `show`, `export`, `import`), the reported near-miss certificate (`near-miss`), and the served versus exact maximum diagnostic (`served-gap`). |
| `recall provenance` | Apply structured facts through the deterministic provenance controller, inspect the current fact projection, rebuild the materialized current facts from the ledger, or erase every fact value of a tenant (`apply`, `current`, `rebuild`, `erase`). |
| `recall backup` | Inspect, create, verify, and restore encrypted Aurora backups through the explicit `status`, `create`, `verify`, and `restore` workflows. |
| `recall secret` | `verify`: verify that every running ECS task reports the intended nonsecret secret version identifiers. |
| `recall idempotency` | `reconcile`: record an operator verified mutation result after a replay result was lost. |
| `recall records` | Look up audit records by exact key, with no embedder: `--anchors FILE` (a JSON list of `{file, symbol, code_sha}`, with `--repo`), `--id`, or `--all` for a complete dump of the tenant's records. `--repo` is required with `--anchors`; with `--id` or `--all`, only records of this repository. `--kind` filters, `--json` prints the envelope the `recall_records` MCP tool returns. Each record carries its `recall_record` exactly as authored, its anchor match (`exact`, `symbol_changed`, `file_only`), and its supersession and `caused` edges; superseded and expired records are listed and marked (`superseded_by`, `expired`), never hidden. Applies the source policy (`RECALL_SOURCE_POLICY_FILE`) as `recall search` does: a record in a denied file is left out, and a denied successor reads `<withheld>` (so does one whose file cannot be placed for the policy, a path two roots hold). A record a `supersedes:` reference names among several files reads `<ambiguous>`. Served on a lite store. A fourth form, `--select-fix FILE`, selects one fix (see [Verified fix selection](#verified-fix-selection)): it requires `--repo`, refuses `--kind`, always prints its own JSON decision (so `--json` changes nothing), and exits non-zero on a request over 4 MiB or one that fails validation. |
| `recall-enterprise` | Manage generation routing and readiness for production deployments. |
| `recall repair-checks` | `prompt`: prepare a check selection prompt; `resolve`: validate the chosen checks; `feedback`: package the observations for a repair retry. See [Repair check selection](#repair-check-selection). |

### Snapshot-checked SQLite rules

`recall search "account opening date" --sqlite-snapshot analytics.sqlite`
returns the evidence bundle and rendered prompt for the local SQLite file
that the caller will query. Use the returned `prompt` as the model input.
The command uses the existing retrieval trust gate and checks the selected
field notes within a read-only SQLite transaction.

A field note begins with a `table.column` line. Put a date representation on
its own line as `Format: YYYY-MM-DD` or `Format: YYMMDD`. The verifier also
accepts `in the form YYMMDD` on its own line. A note with a single supported
declaration is checked against all non-null values. A checked item carries the
format observed in the snapshot in its own `snapshot_format` field, which only
the verifier sets; the note text never states it. The prompt's system message
then tells the model what the field means and that it takes precedence over a
format stated in the text; a library caller renders it with
`recall.evidence.render_snapshot_evidence_prompt(bundle)`. If the values use the other
supported format, the declaration is removed from the prompt text. This derived
text retains the source chunk ID for traceability. The indexed text stays intact.

Field descriptions that make no format claim retain their wording when their
table and column exist. Any `Format:` or `in the form` line is a format claim,
and so is any date pattern such as `DD/MM/YYYY` or `%d/%m/%Y` in the body. A
note whose claims are not one supported declaration is omitted. A `YYMMDD` value must
also be a valid date. The check establishes physical representation and field
existence. Business definitions and units are sourced from the original notes.
Notes without a field header, notes for missing fields, and date claims that
cannot be resolved from a uniform non-empty column are omitted, as is a note
whose column scan does not finish within the two-second budget shared by the
whole bundle. Table and column names match without regard to ASCII case, as
SQLite resolves them. The bundle
abstains with reason code `no_verified_database_rules` if it has no eligible
notes. The JSON includes `snapshot_rule_checks` by chunk ID, with the observed
format when known. Each check has one status:

| Status | Note |
|---|---|
| `field_present` | kept unchanged; it makes no format claim |
| `format_verified` | kept with `snapshot_format`; the declaration matches the snapshot |
| `format_corrected` | kept with `snapshot_format`; the declaration is removed from the text |
| `not_a_field_note` | omitted; no `table.column` header line, or nothing after it |
| `field_missing` | omitted; no such ordinary table (not a view or virtual table) and column |
| `unsupported_format_claim` | omitted; its claims are not one supported declaration |
| `unverified_format` | omitted; the column is empty, not uniformly one supported format, or its scan ran out of budget |

Pass a local SQLite snapshot with the same contents the analytics query will
use. It is read in one read transaction that holds a shared lock until the
check ends, so the check sees one consistent state of the file. The reader
waits for a writer's lock with a 0.25 second busy timeout. While the check
runs, a writer to a rollback-journal snapshot cannot commit. It waits up to its
own busy timeout: if the check ends first, the commit goes through; otherwise
it fails with "database is locked". For a
WAL-mode snapshot, an existing `-wal` file is part of the state checked, as it
is for the analytics query; it can hold committed data, so do not delete it. If
the `-wal` or `-shm` file does not exist, the read creates it beside the
database, and it remains until a read-write connection to the database closes.
In a directory the caller cannot write to, a WAL-mode snapshot missing either
file cannot be opened, and the check fails. This option is available from the
CLI and
`recall.sqlite_rules.verify_sqlite_rules(bundle, snapshot)`. The CLI output
uses the canonical text and cannot be combined with `--locale`.

### Snapshot-checked SQL planning

Python applications can opt in to a two-call workflow with one model callback.
The callback takes a system prompt and a user message and returns text.

```python
from pathlib import Path

from recall.evidence import build_evidence_bundle
from recall.sqlite_planning import generate_sql_from_evidence

result = generate_sql_from_evidence(
    build_evidence_bundle(trusted_result),
    Path("analytics.sqlite"),
    generator,
    code_fields={"account.frequency", "loan.status"},
)
```

The first call proposes fields, joins, filters, and a measure. RE-call checks
the plan in the SQLite snapshot before the second call writes SQL. The same
question, schema, and up to three eligible field notes appear in both calls.
The optional `code_fields` allowlist uses exact `table.column` names. The
second call can receive short lists of observed values for those fields.
Membership and join overlap checks establish facts about the stored data.
The writer decides how those facts apply to the question.

A plan that parses is sent to the second call with the result of each check,
including fields that do not exist and joins that share no stored value. A
malformed plan, or a SQLite error while checking it, sends the original inputs
instead. `used_checked_plan` says whether the second call received the checked
plan, not whether every check passed. When evidence is unavailable or the
snapshot cannot supply the schema and notes, the workflow returns a
`reason_code` before calling the model. Callback exceptions propagate to the
caller. Successful calls return either `sql` or a `clarification`. When the
second reply is neither a clarification nor one read-only `SELECT` or `WITH`
statement that compiles, the result carries `reason_code`
`invalid_read_only_sql` and no SQL.

RE-call compiles SQL under a read-only authorizer on the same transaction
used for the checks. Run the returned statement against that same version of
the snapshot. Query execution and the choice of model provider belong to the
calling application. The read transaction stays open across both model calls.
In rollback journal mode a process writing to the snapshot file waits, or
fails with `database is locked`, until the workflow returns. In WAL mode writes
proceed, but no checkpoint can move past the workflow's read, so the WAL file
grows and a `FULL`, `RESTART` or `TRUNCATE` checkpoint waits. Use a copy of the
database that no process writes to.

### Verified fix selection

`recall records --repo owner/name --select-fix request.json` takes an explicit shortlist
request from a trusted local agent or runner. Use `-` for standard input. The pilot is
opt-in through this flag. Normal retrieval continues through `recall search` and the
existing agent integrations.

The command rechecks the requested IDs against one current record snapshot and the
configured source policy. It excludes an exact record ID when that record has an active
supersession, including when the successor is outside the request, withheld by policy,
or the relationship is marked ambiguous. Any successor must qualify as a candidate in
its own right. The selector ranks current proved bugs within the exact pre-fix revision.
A candidate is excluded before ranking unless its record is a current, unexpired bug of
tier `proved` whose stored `fix_commit` equals the candidate's `fix_commit`, and the
candidate's `pre_fix_revision` equals the request's `revision`. Positive lexical matches use the bug summary, test node and assertion text. A file match moves
ahead of other positive matches; `file_hints` are repository-relative paths, and `./` or
backslash separators are normalised before the comparison. At most three test nodes are
returned for verification.

The request contains `report`, a full 40-character Git `revision`, optional `file_hints`,
and `candidates`. Each candidate has `id`, `fix_commit`, `pre_fix_revision`, `test_node`,
optional `assertion_text`, `symptom_groups`, `failure_marker`, and `repair`. A symptom group
lists 1 to 8 alternative phrases, and a candidate has 2 to 8 groups. Every group must match
the report to establish the symptom. Phrases are compared as lowercased alphanumeric
tokens, so case and punctuation are ignored. For example,
`[["duplicate", "repeated"], ["question row", "question rows"]]`. The request is strict:
unknown keys are refused and candidate IDs must be unique.
Candidate metadata is supplied alongside retrieved IDs. The command does not infer that
metadata for existing memories.

Omit `observations` to obtain the shortlist. After a test run, submit observations with
`revision`, `test_node`, `outcome` (`fail`, `pass`, `error`, or `skipped`), and
`diagnostic`, the trusted runner's extracted failure diagnostic. Serving requires that
exactly one eligible candidate, shortlisted or not, matches the symptom, that it is in the
shortlist, and that no other eligible candidate shares its test node (`ambiguous_test_node`).
It also requires exactly one observation of any outcome for that test node and revision: a
second one, even a pass, abstains as `ambiguous_observation`. That observation must be a
`fail` whose diagnostic equals the candidate's marker. The caller is
responsible for measuring the clean checkout revision and supplying the actual diagnostic.
The command neither runs tests nor applies fixes.

The JSON response contains `decision`, `reason`, `shortlist` (entries of `id` and
`test_node`), and the `tenant_id` and `generation_id` of the snapshot it was read from
(`generation_id` is null when no snapshot was read). Only `serve` adds one `fix`, with keys
`id`, `source`, `source_commit`, `verification` (the test node), and `repair`.

| `decision` | `reason` | When |
| --- | --- | --- |
| `serve` | `observed_symptom` | every condition above holds |
| `abstain` | `ambiguous_records` | a requested ID matches more than one visible record |
| `abstain` | `superseded_fix` | the shortlist is empty and a requested record was excluded by supersession |
| `abstain` | `no_relevant_candidates` | the shortlist is empty otherwise |
| `abstain` | `ambiguous_symptom` | more than one eligible candidate matches the symptom |
| `abstain` | `symptom_not_correlated` | no eligible candidate matches, or the match is not shortlisted |
| `abstain` | `ambiguous_test_node` | another eligible candidate shares the match's test node |
| `abstain` | `missing_observation` | no observation for that test node and revision |
| `abstain` | `ambiguous_observation` | more than one observation for that test node and revision |
| `abstain` | `failure_not_matched` | the observation is not a `fail` with the candidate's marker |
| `unavailable` | `store_missing` | the lite store file does not exist; it is not created |
| `unavailable` | `snapshot_unavailable` | the backend has no single-snapshot read (a legacy PostgreSQL table) |

The ordinary records forms continue to return audit records. This pilot supports lite and
generation-backed stores.

### Repair check selection

Use this opt-in workflow when a repair attempt still fails and the runner can offer a
menu of relevant checks. The CLI works locally with either store configuration. It
builds JSON for the caller's selector, validates the selected IDs and packages the
runner's observations. The caller supplies the model, executes checks on the same
failed edit, and keeps the request fixed through the three phases.

```sh
recall repair-checks prompt request.json
recall repair-checks resolve request.json selection.json
recall repair-checks feedback request.json selection.json observations.json
```

The request describes the task, the current failure, the failed edit, and a menu of
2 to 16 checks with unique IDs and both roles represented. Each description states
what to inspect and which behavior is expected. IDs have up to 64 ASCII letters,
digits, underscores or hyphens and begin with a letter or digit. For example,

```json
{
  "task": "Preserve a note's date when changing its title.",
  "failure": "The date disappears after reloading.",
  "failed_edit": "Save the title from the editor.",
  "checks": [
    {"id": "date", "role": "diagnostic", "description": "Reopen the renamed note and check its saved date."},
    {"id": "undated", "role": "preservation", "description": "Rename an undated note and verify it stays undated."}
  ]
}
```

When RE-call returns citable evidence for a distinct earlier repair, the caller can
add `prior_procedure` with the exact `source` and a `procedure` describing the relevant
way to distinguish outcomes. Verify that the source is current and that the procedure
applies. Keep the current fix and answers to the menu out of the procedure. Omit the
field when no applicable source is available. The CLI accepts this provenance from the
caller. The normal retrieval trust checks happen before constructing the request.

Give the selector the prompt's `instructions`, `input` and `response_schema`. Supply
its JSON response, for example `{"diagnostic":"date","preservation":"undated"}`, to
`resolve` before running the chosen checks. For an abstention, use JSON `null`.
An invalid response exits with an error, and the caller can continue with an abstention.

For `feedback`, supply one observation per chosen ID with `id`, `outcome`
(`pass`, `fail`, `error` or `skipped`), and optional `detail`. Use an empty array
for abstention. Each output check contains its ID, role, description and observed
result. Give that output to the repair agent alongside its current task and failed
edit. The prior procedure is scoped to selection.

Each input is UTF-8 JSON limited to 512 KiB. Use `-` for at most one input per
invocation. Text fields are limited to 4096 characters and the failed edit to 65536.
Unknown fields, duplicate JSON keys, wrong-role choices and mismatched observations
are rejected.

## MCP

### Health endpoints

The HTTP deployment exposes `GET /livez`, `GET /readyz`, and `GET /startupz`. `livez` is
dependency free and returns `200` while the process is running. `startupz` returns `200` after
startup completed, otherwise `503`. `readyz` returns `200` only when the database, schema, pgvector,
RLS, and active generation checks pass. Its `calibration` entry reflects the enterprise startup
readiness gate, which runs once at startup; `readyz` does not re-evaluate calibration, and strict
trust still refuses an answer whose calibration is not certified. Redis limiter state is reported but does not
make reads unready, because reads have a bounded local fallback and mutations fail closed.

Mutating MCP tools require an idempotency key on HTTP deployments for writes, forget, and admin
operations. Most tools expose it as `idempotency_key`; `recall_apply_fact` exposes the same value
as `request_id`. PostgreSQL records the completed response before Redis stores its replay cache,
so a repeated mutation key returns the original response even when Redis lost the response write.
The receipt is bound to the idempotency key, tool, canonical request arguments, and the
authenticated principal, so reusing one key for a different operation, different arguments, or
from another principal of the same tenant is rejected as `idempotency_conflict`. A replay is
still a request and spends one unit of the tool's budget. Receipts and reservations written
before principal binding hold the unbound fingerprint, so a key in flight across that upgrade
cannot be replayed and needs a new key. The
`recall_apply_fact` tool calls this public key `request_id`; other mutating tools call it
`idempotency_key`. If neither receipt is available, the server returns
`reconciliation_required` and does not execute the mutation again.

An operator resolves that state only after checking the mutation's side effect in its owning
system. Save the exact JSON response, then run:

```bash
recall --tenant TENANT idempotency reconcile \
  --key KEY --operation TOOL --fingerprint FINGERPRINT \
  --result-file verified-result.json --confirm RECONCILE_IDEMPOTENCY
```

The command writes only the PostgreSQL durable receipt. It does not clear the Redis reservation and
does not execute the mutation, so a later retry replays the verified response without risking a
second side effect. Without the confirmation value it prints a dry run.

`readyz` runs one shared control-plane check and a deterministic, bounded sample of tenant stores.
The JSON response includes `checks.control_plane`, `checks.tenant_probes` (the number of stores
probed), `checks.rate_limiter`, and `failures`. Tenant inventory counts are kept out of this
unauthenticated endpoint; the full configured count and probe limit are emitted in startup logs.
The probe count is not the configured tenant count, so a `200` response is representative process
readiness only and does not prove that every configured tenant is correctly provisioned. The
`tenants` check is a deprecated alias of `tenant_probes`: it reports the probe count, not the
tenant count. Perform a
separate exhaustive tenant validation when that guarantee is required.
Durable receipts are operational replay records rather than long-term audit history. A tenant's
expired rows are pruned whenever that tenant records a new receipt, so only tenants that stop
writing keep them. The table forces row level security, so a scheduled prune must set the tenant
for each tenant it cleans, for example:

```sql
SELECT set_config('recall.tenant_id', '<tenant>', false);
DELETE FROM recall_idempotency_receipts
WHERE tenant_id = '<tenant>' AND expires_at <= now();
```

Without the tenant setting the statement matches no rows and reports nothing.

The MCP server is `python -m recall_mcp.server`. Every registered tool, in `tools/list` order;
the same drift test diffs this table against the `@mcp.tool` registrations:

| Tool | Purpose |
|---|---|
| `recall_search` | Search trusted memory. Each hit's `indexed_via` names the channel its source first entered the index through (`cli_index`, `recall_index`, `recall_ingest`, `recall_remember`, `folder_refresh`), kept across a re-index; null for a source indexed before it was recorded or by a generation build. See [SECURITY_MODEL.md](SECURITY_MODEL.md). |
| `recall_evidence` | Return evidence for a query. |
| `recall_current_facts` | Read the current structured fact projection. Each fact names its memos in `sources`; values that disagree with no supersession between their memos are listed under `contested`. `served_state` is `ok`, `degraded` (a check against the served memos failed and `served_lookups_failed` names it: after `admission` or `content` stated facts may be missing, after `presence` or `supersession` a listed fact may be retired or no longer served) or `unchecked` (a store with no such lookup). A lite store records the facts its memos state in `recall_graph` frontmatter whenever it indexes them, from the folders it refreshes, or the prefixes `RECALL_STATED_FACT_SOURCES` names (`none` turns it off). A PostgreSQL tenant records them only where that variable names places (a URI such as `s3://bucket/corpus/memory`, an absolute path, or a path under `RECALL_INDEX_ROOT`, matched against each memo's source), when a generation is promoted or rolled back, when the legacy table is indexed, and on `recall generation stated-facts`. Set the variable in the process that promotes and in the one that serves, since the rule is applied again when facts are read; with it on, a memo it admits that supersedes another memo also retires that memo's applied facts. Uploads never state. With `subject` (and optionally `predicate`) it lists only that subject's facts and contested groups, matched ignoring case and spacing, plus `similar_subjects`: subjects whose names contain or are contained in it, so a writer can reuse an existing key's exact spelling. |
| `recall_apply_fact` | Apply a reviewed structured fact through the provenance controller. Scope `recall:fact_write`, which `recall:write` does not imply. |
| `recall_related` | Retrieve independently trusted structural related evidence: `source`, `ordinal`, `supersession`, or `caused_by` (the regression edge between audit records, both directions). |
| `recall_records` | Look up audit records (bugs, false positives, lessons, retractions, and code anchors in a code tenant) by exact key: `repo` with up to 200 `anchors`, or `ids`, optionally filtered by `kinds`. Each record is returned exactly as authored, with each anchor's match (`exact`, `symbol_changed`, `file_only`), its validity and supersession (the envelope's `supersedes` is the front matter reference as written; `superseded_by` names the successor by its root-relative file, its absolute source when two roots hold that path; `recall_record.supersedes` is a record id), whether it has `expired` (computed at the lookup's time: a `valid_until` date lasts through that day, UTC), and the reverse `caused` edge. In a code tenant the chunks sharing one `code_anchor` (a split symbol, a module's module-level chunks) are one record per distinct `(repo, file, symbol, code_sha)`: `chunk_id` is the lowest-ordinal chunk and `chunk_ids` lists all of them (a memo's record lists its chunks the same way), the record's `id` is that chunk id, and its `name` and `description` are null; a code anchor with a null `repo` never matches a `repo` query. Under the caller's source policy a record in a denied file is left out and never named in `caused`, and a successor the caller may not read, or whose file cannot be placed for the policy, is reported in `superseded_by` as `<withheld>`. A record a `supersedes:` reference names among several files is reported as `<ambiguous>`, never as current. No embedding, calibration or abstention; more than 500 matches is refused, never truncated. Read only. |
| `recall_current_state` | Inspect a deterministic authored current state projection. |
| `recall_reasoning_query` | Run an explicit opt-in reasoning query over trusted retrieval. `graph_expansion` defaults to `auto`, which uses bounded global one-hop expansion for every nonempty query. `off` and `one_hop` remain explicit overrides. Precision admission diagnostics and a policy fingerprint are additive response fields. Legacy `expand_retrieval` remains available when configured. |
| `recall_query_construction_challenge` | Start or continue bounded query construction with an original-model challenge, deterministic candidate controls, and generation-bound trusted retrieval. |
| `recall_reasoning_projection` | Inspect the generation-bound reasoning graph projection. |
| `recall_reasoning_proposals` | Inspect inference proposals as review candidates. |
| `recall_rewrite_plan` | Report which key a proposal would declare, in which file. Writes nothing. |
| `recall_report_stale` | Report that one memory used in a task is replaced by another, with a verbatim quote from each. Queues the claim for a person to review (`recall dashboard`); edits no memo and changes no verdict. When both memos are files under `RECALL_INDEX_ROOT` it goes to the `.recall` sidecar there; when the store is database-backed and both are in the served generation, with the quotes verbatim in the served text where the store can show it, it is (also, or only) one `stale_report` row per claim per day in `recall_audit_events`, which a dashboard on another machine reads. That row keeps a fingerprint of each quote, never the memo text. The result's `recorded_in` says where it went. Needs `recall:write`. With source policy enabled it also needs `recall:read`, validates quotes against eligible served rows, and uses the audit ledger. |
| `recall_report_use` | After a task that searched memory, report what the memories did for it: `effect` (helped, no_difference, misled), the memories `used` and any found `wrong`, and optionally whether the task succeeded. Every memory named must exist in the served generation. Appends one `use_report` row to the audit ledger, counted as agent-reported on the dashboard's Control page; changes no memo, verdict or ranking. Needs `recall:write`. With source policy enabled it also needs `recall:read` and resolves only eligible served sources. |
| `recall_reasoning_audit` | Report reasoning integration state and diagnostics. |
| `recall_index` | Index allowed files beneath `RECALL_INDEX_ROOT`. |
| `recall_tenants` | Return the tenant scopes visible to this caller (the full inventory needs `recall:admin`). |
| `recall_ingest` | Upload bounded source files and index them, debiting the tenant's byte quota. Optional `source_namespace` (1 to 64 ASCII letters, digits, underscores or hyphens) gives legacy memory uploads stable paths: a matching name replaces its source, while files omitted from a partial batch remain. Legacy namespaced replacement refuses when source security policies are configured; use reviewed source manifests for that configuration. Optional per-file `source_key` (a lowercase SHA-256 root identity) qualifies the first name component to separate projects in a shared tenant. Generation uploads keep immutable staging. Updated hosted hooks require a server exposing this parameter. |
| `recall_job_status` | Return the state of one ingest job, scoped to the caller's tenant. |
| `recall_calibration_status` | Return the latest calibration artifact bound to the caller's generation. |
| `recall_calibration_run` | Fit corpus calibration. PostgreSQL creates a draft for the active generation. Lite activates completed fits immediately: a certified fit governs serving, and a completed uncertified fit replaces the prior calibration and blocks strict answers. Attempts that cannot complete retain the prior artifact. Lite has no separate publish step. |
| `recall_calibration_publish` | Publish one certified calibration artifact. Requires `recall:admin`: publication changes what the whole tenant serves. Under an active atomic rescue it rebuilds the generation's artifact; a failed rebuild is reported in `atomic_rescue` and the publication stands. |
| `recall_forget` | Erase indexed source material, including its staged upload files, retract the facts whose only evidence it was, and erase the values it supplied from the fact ledger (`facts_redacted`). Whatever a forgotten memo superseded stays retired, in the document graph and the fact ledger alike. |
| `recall_withdraw` | For a memo that was a mistake: erase it as `recall_forget` does (fact values included), then give back what it superseded. Its `supersedes:` claims become void and whatever superseded it supersedes what it superseded instead (withdrawing B in A -> B -> C leaves A retired by C); each fact it alone supported gets a `withdrawn` event, which also undoes that fact's supersession. A memo forgotten or pruned earlier is withdrawn by the name it had. Scope `recall:forget`. |
| `recall_inventory` | List sources with their content SHA-256. Generation storage hashes source bytes. Legacy storage hashes Markdown as UTF-8 after BOM removal, universal-newline normalization and NUL removal; other files use raw bytes. These digests differ from indexing fingerprints. Hosted hooks upload normalized Markdown to agree with both backends. Not in the `read` or `search` presets. |
| `recall_resolve_reference` | Before writing `supersedes:` into a memo file, ask which memo a reference names, by search's own rule: `resolved` (with `write`, the shortest form that names it alone), `ambiguous` (several memos share the name; write the one meant with its folder, from `candidates`), `dangling` or `self`. With a source security policy it resolves among the sources the caller may read. Writes nothing. `recall_index` also lists, in `ambiguous_references`, every stored reference that names more than one memo. This receipt omits the advisory when source policy is configured. Use `recall_resolve_reference` for an individual reference. |
| `recall_stats` | Report counters and operational state. |
| `recall_remember` | Assistant surface, lite only (`RECALL_ASSISTANT_MEMOS` names a memo folder inside `RECALL_INDEX_ROOT`): write ONE fact as one memo, with `valid_from`, `valid_until`, `supersedes` (only a memo the assistant wrote: of this folder by its memo id, or of another assistant's memo folder under the same `RECALL_INDEX_ROOT` by the id search names it by, such as `agent-b/note-0001`; written into the memo as the target's path under the index root) and an optional structured `fact`, and index it before returning `{memo_id}` (a legacy call raises if indexing fails; with a receipt the failure is reported as `index_state`, see [Memo corrections](MEMO_CORRECTIONS.md)). Optional `request_id`, `expected_revision` and `include_receipt` provide recoverable corrections as described below. With `include_receipt` and a `fact`, the receipt's `fact_key` names the fact's key, its `state` (`current`, `contested`, `not_in_force`, `not_single_valued` or `unchecked`), the values `in_force` with their memos, and the `proposals` rule `deterministic.stated_key_successor` makes (each `requires_review`): of the memos stating one single-valued key, each supersedes the one of the latest earlier `valid_from`, whatever the value. Optional `settle` writes that proposal into the new memo as its `supersedes`, when the operator has set `RECALL_ASSISTANT_SETTLE_SUCCESSORS` (otherwise the call is refused): only when the writer named no `supersedes`, the caller has the read scope, no source security policy is configured, the memo is dated, RE-call's predicate table calls the predicate single-valued (spaces and hyphens in it read as underscores, so `phone number` is the table's `phone_number`), the one earlier memo it would supersede is the only memo holding the key, and that memo is an assistant memo file with no successor yet, indexed or not. The answer's `settle` is `{"settled": <memo id>}`, `{"settled": null}` (nothing to settle) or `{"declined": <reason>}` (`writer_named_supersedes`, `no_fact`, `unchecked`, `invalid`, `not_single_in_table`, `not_single_valued`, `undated`, `several_older_memos`, `other_memos_hold_key`, `not_assistant_memo`, `target_has_successor`, `target_missing`), and a declined call is written without the edge. Each settled memo is listed under `settled` in the folder's authorship ledger. With the surface on, `recall_search`, `recall_evidence`, `recall_current_facts` and `recall_apply_fact` answer in a compact view naming each hit by its memo id; the `assistant` preset serves exactly these six tools. Unset, none of this is registered. Scope `recall:write`. |
| `recall_forget_memo` | Assistant surface, lite only: forget one memo of the memo folder by its id, retracting the facts whose only evidence it was and erasing their values from the fact ledger, then delete its file. `{deleted: 0}` for an id that names no memo there. Scope `recall:forget`. |

For an evaluation that replays simulated days, `RECALL_EVAL_CLOCK=<file>` (lite only) makes the
server read "now" from that file, one ISO 8601 time with an offset, re-read on every call: search
verdicts, row write times, fact application and the current facts follow it, while a caller that
passes its own time keeps it. A missing or offset-less time fails the call rather than falling back
to the wall clock (`recall.eval_clock`).

Search's `abstained`, `decision_state`, and `evidence.citable` describe present-state support.
For a question about an earlier time, the optional `historical` field lists individually relevant,
certified superseded or expired chunk identifiers in `historical.candidates`. Its scope is
`historical_only`, and `requires_validity_check` is always true. Before citing a candidate as a
fact at the requested time, match that time to the hit's `valid_from` and `valid_until` and its
text. If they do not establish that time, describe only the archived assertion, without claiming
it held then. These candidates never expand present-state citations. Setting
`RECALL_SEARCH_EVIDENCE` to `off` (or `0`, `false`, `no`) sets both evidence fields to null and
removes their citation advice.

`recall_search` and `recall_evidence` carry `atomic_rescue_skipped`, an empty string unless
`RECALL_ATOMIC_RESCUE_WHEN_MISSING=skip` let an active rescue answer without its artifact; it then
says why (missing, for another calibration, damaged, or uncalibrated: no calibration in force). See [ENVIRONMENT.md](ENVIRONMENT.md#atomic-rescue).

`recall_search` and `recall_evidence` also accept an optional `locale` argument for presentation
localization. When supplied, the response gains an additive `localized` object containing display
text keyed by `chunk_id`. Canonical hit text, provenance, evidence items, `system_prompt`, and
`user_message` are never translated in place. Localization is disabled unless
`RECALL_TRANSLATION_ENABLED=1` configures a validated HTTPS text endpoint. Provider failures are
fail soft and return canonical values with a fixed warning. Enabling the provider sends selected
retrieved passage text to that endpoint, so deployments with sensitive corpora should use a
self-hosted endpoint and should treat localized values as display data only. The explicit
`RECALL_TRANSLATION_ALLOW_HTTP=1` override permits cleartext HTTP for a deliberately controlled
endpoint and must not be used across an untrusted network.

Both tools also accept optional `scope`, `modality`, and `route_id` fields for request aware
retrieval planning. `RECALL_RETRIEVAL_PLANS_JSON` enables the versioned planner. It selects one
primary physical tenant and, only for an ambiguous request, at most one bounded rescue tenant.
The response carries `retrieval_plan` with the route id, selection reason, allowed tenant set,
per leg limits, latency bound, and selected tenant identities. The planner never compares scores
between tenants. When bounded federation is explicitly enabled, the serving adapter executes the
selected certified legs through isolated stores and adds federation diagnostics to `recall_search`
without changing the compatibility route. `recall_evidence` always runs single tenant, so it
returns no federation diagnostics. An unset plan keeps the existing single tenant search behavior. Invalid
route configuration, unknown route ids, unallowed tenants,
retired generations, and runtime embedding-profile mismatches fail closed; calibration and trust
remain bound to the existing serving trust boundary.

The static README viewer uses these provider locale identifiers: `english`, `italian`, `spanish`,
`french`, `german`, `portuguese`, `chinese_simplified`, `japanese`, `korean`, `russian`, `arabic`,
`hindi`, and `turkish`. Other provider identifiers may be passed to the MCP or CLI presentation
surfaces. An unsupported identifier or provider failure leaves canonical text unchanged and marks
the localized object as a fallback.

Related expansion and structured retrieval explanations are disabled by default. With
`explain=true`, `recall_search` and `recall_evidence` also return a count only memory audit under
`explanation.details.memory_audit`; see [MEMORY_OBSERVABILITY.md](MEMORY_OBSERVABILITY.md). Graph expansion in
reasoning uses bounded global one-hop activation for nonempty queries. Set
`RECALL_ROUTING_MODE=active` only for a preregistered routing experiment. The default `shadow`
mode records the deterministic decision without changing retrieval behavior. See
[Active routing promotion gates](ROUTING_GATES.md) before enabling active mode.
Bounded specialist federation is a separate opt in layer. See
[FEDERATED_RETRIEVAL.md](FEDERATED_RETRIEVAL.md). Its default is `RECALL_FEDERATION_MODE=off`,
which searches only the caller's authenticated tenant; the attached `retrieval_plan` is advisory
in that mode. `shadow` executes a bounded plan without changing
the served control result; `active` is required before a caller may serve the merged result.
`recall_current_state`
defaults to a fail closed maximum of 1000 source records and accepts an explicit `max_records`
bound; use `source` to project one authored lineage when a tenant is larger.

`recall_ingest` accepts a JSON object with `files`, `category`, optional `tenant`, and optional
`idempotency_key` fields. Each `files` entry has `name` and base64 encoded `content_b64` fields.
`category` is one of `documents`, `code`, or `memory` and defaults to `memory`. Upload count,
decoded byte, filename, path, and staging limits are enforced before indexing. HTTP mutations must
provide an idempotency key, and reusing a key with a different category or file payload returns
`idempotency_conflict`; a completed retry returns the original JSON response. The response is the
same bounded indexing result used by `recall_index`, with the staged job id included when the
ingest path creates an asynchronous job. See `docs/ENVIRONMENT.md` for the upload limits.

The CLI accepts the same additive presentation option, for example:

```console
recall search "deployment notes" --locale italian
```

**There is deliberately no `recall_rewrite_apply`.** Nothing reaches corpus metadata without a
named human, and the MCP client is the model: letting it supply a reviewer id and an audit note
would make that gate a formality it satisfies by typing a string, so the gate becomes a field
rather than a person. This surface proposes; a human declares at `recall rewrite apply`.
`recall_mcp/` makes no file write call of any kind, and two tests hold that line, one for a write
call and one for a write import.

`recall_rewrite_plan` hands off a **claim key**, not a proposal id. Its proposals come from the
deterministic rules over the store graph while `recall rewrite apply` resolves ids against the
filesystem extractor, and provider, tenant, generation and pipeline are all hashed into a
proposal id, so those two id spaces are disjoint. Claim keys are generation independent, which is
also why the rejection ledger is keyed by them.

Authentication, tenant isolation, and transport modes are documented in [AUTH.md](AUTH.md).
Reasoning policy and operational behavior are documented in
[REASONING_OPERATIONS.md](REASONING_OPERATIONS.md).

## Stability Boundary

Benchmark harnesses under `benchmarks/`, result builders under `results/`, and experimental code
under `benchmarks/finetune/` are not the library API. They are retained for reproducibility and
evidence review.

## Retrieval eligibility and answerability

Reasoning response decoding preserves these diagnostics and accepts older payloads with the
same unchecked and unmeasured defaults. Retrieval expansion and proof repair retain identical
coverage reports. When merged results disagree about coverage, the aggregate is unmeasured
without a report; its calibration ID is retained only when all merged diagnostics name the same
fit. Read from the store, a report with inconsistent counts, rates, measurement states or
provisional warnings is an `unavailable` diagnostic and does not change evidence admission;
decoding a reasoning response whose diagnostics are inconsistent (a report that disagrees with its
status, or coverage naming another calibration than the response) raises
`EvidenceValidationError`.

Search and evidence responses add `query_support` and `calibration_coverage`. Existing
`decision_state`, `decision`, `abstained`, verdicts, confidence and citation eligibility retain
their behavior. `supported`, `answer` and a citable chunk mean that retrieval trust checks admitted
evidence; they do not establish that the requested detail is stated or that a generated claim is
correct. Callers must establish support for each claim from the eligible passages.

`query_support` is `{ "status": "not_checked", "scope": "returned_evidence", "method": null }`.
No answerability verifier runs. It remains not checked on certified, degraded and empty results.
`validate_answer` still checks structure and citation identity only.

`calibration_coverage` contains the selected `calibration_id`, a `status`, a `method` and a
`report`. When available, `method` is `synthetic_corpus_probes` and `report` preserves the stored
`calibration_id`, the `unknown_entity` and `in_vocabulary` counts, rejection rates, per-class
measurement states, the `provisional_in_vocabulary` warning and `unknown_term_gate` (`checked`,
`unavailable`, `disabled` or `not_run`). These are synthetic diagnostics, not audited labels or
per-query semantic support. `measured` means both probe classes were measured, not that the gate
passed them: a measured report can still be provisional.

Missing measurements, a backend without this optional reader, or a report for another calibration
produce `unmeasured` and a null report. A stored report in which a class has fewer than 10 probes
is `unmeasured` too, and still carries its report. Reader failures produce `unavailable`. A carried calibration
does not inherit near-miss measurements from the parent fit. Coverage is read in the search's
snapshot for the calibration actually selected, without refitting or changing admission.

The Python trusted result and evidence bundle carry the same dataclasses. MCP, the in-process
agent tools, the compact assistant view and CLI report the distinction. Older MCP payloads without
these additive fields remain readable, defaulting to not checked support and unmeasured coverage.

## Assistant memo corrections

See [Memo corrections](MEMO_CORRECTIONS.md) for optional request identities, revision preconditions,
write receipts, conflict behavior and legacy compatibility.
