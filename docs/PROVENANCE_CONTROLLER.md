# Deterministic provenance controller

The provenance controller is the write authorization boundary for structured RE-call facts. It is
not the existing retrieval trust gate and it is not the human-reviewed corpus rewrite path.

## Contract

`recall_evidence` returns compact `EvidenceCard` records. A card contains a stable `card_id`, the
source and source digest, valid time, recorded time, tenant and generation lineage, trust and
calibration identity, rank, authored links, and deterministic structured facts. The model may cite
the card ID, but it cannot supply or alter any of those fields at application time.

`recall_apply_fact` accepts one `AtomicFact`, card IDs, and a request ID. It re-resolves each card
from the durable tenant-scoped card projection, checks the active tenant and generation, checks the
current source digest, and requires a certified `ok` card with a structured fact matching the
requested claim. Arbitrary prose is not sufficient for an automatic write.

`recall_current_facts` and `recall provenance current` expose the ledger projection. They return
only facts that are asserted, not superseded, and valid at the requested point in time.

For an offline local corpus, `recall provenance apply --sqlite-path PATH --source-root ROOT`
uses the SQLite card and fact stores. `--source-root` is mandatory in this mode; each card source
is resolved beneath that root and hashed again before authorization. PostgreSQL remains the default
CLI backend and the only backend used by the MCP service.

The low level ledger adapters also require a single use permit minted by the controller. A direct
ledger assertion without that capability is refused, even if the caller can import the adapter.

Before application, the service also rebuilds source-derived card fields from the current
generation. A changed source identity, validity window, structured fact, authored link, or
malformed source record changes the card identity and fails closed. Rank and calibrated trust are
retrieval-time fields bound to the immutable card projection. They are not recomputed from a card
ID alone because rank requires the original retrieval context; a changed generation must instead
produce a fresh evidence search.

## Conflict policy

Facts are canonicalized with stable JSON and Unicode NFC normalization. Two facts conflict when
namespace, subject, predicate, and context match, object values differ, and validity intervals
overlap. Duplicate facts are idempotent. A conflicting fact is accepted only when an authored
supersession link names the prior fact, prior card, prior chunk, or prior source. Model-inferred
links are never authorization evidence.

Predicates are single-valued by default. A future policy may explicitly allow a multi-valued
predicate; multiplicity is never inferred from retrieval output.

## Ledger

`recall_fact_ledger_events` is append-only and tenant-isolated. It stores asserted, superseded,
rejected, and abstained events, including immutable evidence-card snapshots. Current facts are a
projection over the event stream, so generation garbage collection cannot remove the provenance of
an applied fact.

PostgreSQL is the production adapter. `SQLiteFactLedger` is the durable local adapter and
`SQLiteEvidenceCardStore` provides the matching durable local card projection. `InMemoryFactLedger`
and `EvidenceCardStore` are deterministic test adapters. Evidence cards are persisted in
`recall_evidence_cards`; applied facts retain card snapshots in
`recall_fact_ledger_events`. Database application serializes on the canonical conflict key and
uses the same tenant boundary as the chunk store.

`recall_fact_materialization_outbox` stores an immutable snapshot of every authorized event that
has a downstream materializer. Delivery state is separate from the append-only fact ledger, with
short leases, retry attempts, and failure diagnostics. `MaterializationRecovery.run_once()` claims
bounded pending work and retries it without re-running fact authorization. A process crash after
claiming is recoverable when the lease expires. The controller enqueues and claims the event before
calling the materializer, so a successful replay never creates a second fact event.

If a process crashes after the ledger commit but before the outbox insert, a recovery job calls
`MaterializationRecovery.reconcile(ledger.events)`. It re-enqueues only immutable asserted events in
canonical order and then runs the same bounded delivery pass. This closes the delivery gap without
granting the recovery job authority to assert facts.

The preregistered deterministic fixture is runnable with
`python -m benchmarks.provenance_controller_eval --out results/provenance_controller_eval.json`.
It exercises current, stale, changed, cross-lineage, unsupported, supersession, duplicate,
fresh-search, outage, materialization, and concurrent-conflict cases and reports the safety and
latency metrics declared in `docs/preregistrations/2026-09-04-provenance-controller.md`.

The PostgreSQL controller path uses `apply_assertion_with_outbox`, which commits the fact event and
outbox snapshot in the same transaction. Reconciliation remains necessary for legacy or external
writers and for the SQLite adapter, whose standalone local connections cannot share a transaction.

For production least privilege, generate serving grants with
`recall schema grants --role recall_server --strict`, generate separate controller grants with
`recall schema grants --role recall_fact_writer --controller`, and set `RECALL_FACT_WRITE_DSN` on
the controller process. In strict mode the serving role receives read-only ledger and outbox
access, so a raw SQL connection using the serving DSN cannot append a fact event. The default
serving grant retains legacy single-role compatibility; strict mode is required for this database
level boundary. Migration `0022_provenance_protected_append.sql` removes raw `INSERT` from the
isolated controller role as well. Its ledger and outbox appends go through owner-controlled
`SECURITY DEFINER` functions with a tenant-context check, while outbox delivery updates remain
available to the controller worker. The functions are revoked from `PUBLIC` and exposed only by
the generated controller grants. This protects the table boundary; the controller role remains a
trusted deployment credential and must be kept separate from model-facing serving credentials.

The headless wizard also accepts an optional `fact_write_dsn` config key. When it is present, the
wizard verifies that it reaches the same database through a distinct controller role, requires the
serving and migration connections to be distinct as well, applies both strict serving grants and
controller grants over the migration connection, creates the controller login role when it is
absent, and carries the controller DSN into each registered MCP server. Existing roles are never
silently altered, and a role that cannot log in is refused. The key is excluded from resumable
corpus identity, so changing credentials rewires the deployment without rebuilding corpus
generations. The interactive question surface offers this as an optional field for existing
PostgreSQL installs; leaving it blank preserves legacy single-role behavior. Operators can also add
it to a saved wizard JSON or use the grant CLI for an explicitly managed deployment.

## Failure behavior

The controller makes at most one fresh search. The query is generated from canonical fact fields,
not supplied by the model. If refreshed cards still cannot support the write, the controller records
a stable refusal or abstention code and performs no assertion. Ledger or source-store failure is
not reported as evidence absence.

The existing `PromotedFact` and `apply_rewrite` path remains human reviewed and unchanged. It is a
separate mechanism for corpus metadata rewrites, not an alternate route to structured fact
application.

## Downstream materialization

Library callers that maintain a second fact store may provide a `FactMaterializer` together with a
durable materialization outbox. The controller invokes it only after the append-only ledger event
and outbox row exist, and reports `MATERIALIZATION_UNAVAILABLE` rather than claiming success when
the writer fails. Materializers must be idempotent by event ID. Replaying the same request reuses
the existing ledger event and skips an already applied outbox row. `MaterializationRecovery` is the
bounded replay path for failed or interrupted downstream writes.
