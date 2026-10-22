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

Cards without structured support are not classified as usable write evidence, even when their
retrieval hit was otherwise trusted. The evidence path omits those cards, and the controller treats
an absent or unusable support record as `UNSUPPORTED_CLAIM` before any ledger append.

`cards_from_trusted_result` preserves each `ok` hit's original retrieval rank while omitting hits
without usable structured support. `recall_current_facts` and `recall provenance current` expose the ledger projection. They return
only facts that are asserted, valid at the requested point in time, and not hidden by a successor
then.

A successor hides only the assertions recorded before it (so a value asserted again after its
successor is current again), and only while the successor's own fact holds: a successor dated
from next quarter leaves today's value current, and once a bounded successor expires the old value
is current again (the window rule, owner decision 2026-10-21). What hides a successor hides what
it hid, each inside its own window. A claim conflicts with a value only where that value is in
force inside the claim's window, judged as if the claim were already recorded, and the controller
and every ledger refuse on that one rule (`unresolved_conflicts`). A predicate is single-valued
unless the fact declares a `cardinality` (`set` or `event`) or the predicate is a known set-valued
or event predicate (`DEFAULT_PREDICATE_CARDINALITY`, for example `email`); two values of a
set-valued or event predicate never conflict.

For an offline local corpus, `recall provenance apply --sqlite-path PATH --source-root ROOT`
uses the SQLite card and fact stores. `--source-root` is mandatory in this mode; each card source
is resolved beneath that root and hashed again before authorization. PostgreSQL remains the default
CLI backend. The MCP service uses PostgreSQL, or, on a lite `sqlite:///` store, the ledger and cards
kept in that file (`recall.lite.facts`), with the same event ids and projection.

The low level ledger adapters also require a single use permit minted by the controller. A direct
ledger assertion without that capability is refused, even if the caller can import the adapter.

Before application, the service also rebuilds source-derived card fields from the current
generation. A changed source identity, validity window, structured fact, authored link, or
malformed source record changes the card identity and fails closed. Rank and calibrated trust are
retrieval-time fields bound to the immutable card projection. They are not recomputed from a card
ID alone because rank requires the original retrieval context; a changed generation must instead
produce a fresh evidence search.

The decision code vocabulary is fixed: `APPLIED`, `DUPLICATE`, `CARD_NOT_FOUND`, `CARD_TAMPERED`,
`SOURCE_CHANGED`, `VALIDITY_EXPIRED`, `VALIDITY_NOT_STARTED`, `GENERATION_MISMATCH`,
`LINEAGE_MISMATCH`, `TRUST_UNAVAILABLE`, `UNSUPPORTED_CLAIM`,
`CONTRADICTION_WITHOUT_SUPERSESSION`, `FRESH_SEARCH_UNAVAILABLE`, `FRESH_SEARCH_INSUFFICIENT`,
`LEDGER_UNAVAILABLE`, `MATERIALIZATION_UNAVAILABLE`, `REQUEST_ID_REUSED` and
`REQUEST_ID_RESERVED`.

A request id names one application, a fact on a set of cards. Sent again with the same fact and
cards it is answered with the event it recorded, as `DUPLICATE`; sent with another fact, or the same
fact on other cards, it is refused `REQUEST_ID_REUSED`. An id of a form the ledger mints for its own
events (ending `:supersession`, or starting `retraction:` or `stated:`) is refused
`REQUEST_ID_RESERVED` before the ledger is read. Neither refusal is recorded: the id already names
another event, or belongs to one the ledger will write.

The MCP request shape is `{claim: AtomicFact, evidence_card_ids: string[], request_id: string,
writer: string}`. The response shape is `{allowed: boolean, decision_code: string, request_id:
string, fact_id: string, retried: boolean, detail: string, event_id: string|null,
evidence_card_ids: string[]}`. `AtomicFact` contains `namespace`, `subject`, `predicate`,
`object`, optional object `context`, and optional ISO 8601 `valid_from` and `valid_until`.
`EvidenceCard` contains its canonical `card_id`, chunk and source identity, source digest,
validity and index timestamps, tenant and generation IDs, pipeline and corpus fingerprints,
calibration status, trust state, verdict, confidence, rank, authored links, support references,
structured facts, and schema version. Card payloads are server created and are not accepted as
authorization input.

## Conflict policy

Facts are canonicalized with stable JSON and Unicode NFC normalization. Two facts conflict when
namespace, subject, predicate, and context match, object values differ, validity intervals
overlap, and they are not both of one multi-valued cardinality. A fact repeating one already in
force is a DUPLICATE when it adds nothing; one that names a fact to supersede which the earlier
assertion did not is recorded anew. A conflicting fact is accepted only when an authored
supersession link names the prior fact, prior card, prior chunk, or prior source, or when what
it names already hides the prior value wherever the new fact holds. Model-inferred links are
never authorization evidence.

Predicates are single-valued by default. A fixed table makes a few multi-valued
(`DEFAULT_PREDICATE_CARDINALITY`: `email`, `phone`, `alias` and similar are `set`; `visited`,
`attended` and similar are `event`), and a fact may declare its own `cardinality`. Two values
conflict unless both are `set` or both are `event`, so declaring `set` cannot slip a second
value past a single-valued one. Multiplicity is never inferred from retrieval output.

## Ledger

`recall_fact_ledger_events` is append-only and tenant-isolated. It stores asserted, superseded,
rejected, and abstained events, including immutable evidence-card snapshots. Current facts are a
projection over the event stream, so generation garbage collection cannot remove the provenance of
an applied fact.

PostgreSQL is the production adapter. `SQLiteFactLedger` is the durable local adapter and
`SQLiteEvidenceCardStore` provides the matching durable local card projection. On a lite
`sqlite:///` store the MCP service uses `LiteFactLedger` and `LiteEvidenceCardStore`
(`recall.lite.facts`), which keep the same tables in the store's own file and mint PostgreSQL's
event ids. `InMemoryFactLedger`
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
`MaterializationRecovery.reconcile(ledger.events)`. It re-enqueues the immutable events of the types
the materializer names in its `materializes` attribute (asserted events when it names none) in
canonical order and then runs the same bounded delivery pass. This closes the delivery gap without
granting the recovery job authority to assert facts.

The preregistered deterministic fixture is runnable with
`python -m benchmarks.provenance_controller_eval --out results/provenance_controller_eval.json`.
It exercises current, stale, changed, cross-lineage, absent support, supersession, duplicate,
fresh-search, outage, materialization, and concurrent-conflict cases and reports the safety and
latency metrics declared in `docs/preregistrations/2026-09-04-provenance-controller.md`.

The PostgreSQL controller path uses `apply_assertion_with_outbox`, which commits the fact event and
outbox snapshot in the same transaction. Reconciliation remains necessary for legacy or external
writers and for the SQLite adapter, whose standalone local connections cannot share a transaction.

For production least privilege, generate serving grants with
`recall schema grants --role recall_server --strict`, generate separate controller grants with
`recall schema grants --role recall_fact_writer --controller`, and set `RECALL_FACT_WRITE_DSN` on
the controller process. The serving role always receives read-only ledger, card, and outbox
access, so a raw SQL connection using the serving DSN cannot append a fact event. Migration
`0026_provenance_protected_append.sql` removes raw `INSERT` from the
isolated controller role as well. Its ledger and outbox appends go through owner-controlled
`SECURITY DEFINER` functions with a tenant-context check, while outbox delivery updates remain
available to the controller worker. The functions are revoked from `PUBLIC` and exposed only by
the generated controller grants. This protects the table boundary; the controller role remains a
trusted deployment credential and must be kept separate from model-facing serving credentials.

The headless wizard also accepts an optional `fact_write_dsn` config key. When it is present, the
wizard verifies that it reaches the same database through a distinct controller role, requires the
serving and migration connections to be distinct as well, applies read-only serving grants and
controller grants over the migration connection, creates the controller login role when it is
absent, and carries the controller DSN into each registered MCP server. Existing roles are never
silently altered, and a role that cannot log in is refused. The key is excluded from resumable
corpus identity, so changing credentials rewires the deployment without rebuilding corpus
generations. The interactive question surface offers this as an optional field for existing
PostgreSQL installs; leaving it blank preserves legacy single-role behavior. Operators can also add
it to a saved wizard JSON or use the grant CLI for an explicitly managed deployment.

## Materialized current facts

A read used to decode and resolve the tenant's whole history, and a write's conflict check did so
twice more. Migration 0030 adds three derived tables
(`recall.fact_current`): `recall_fact_current_events` files every ledger event under the conflict
key it bears on, `recall_fact_current` holds one row per effective interval of each applied,
unretracted assertion (its own window less the windows of the successors that hide it, so a
retroactive correction splits an open-ended fact into two rows), and `recall_fact_current_state`
marks a tenant whose tables may not answer. They hold ids and hashes only, never a fact value. The
rows of a key are computed by the ledger's own rule functions over that key's events, so the tables
are always rebuildable from the ledger.

`current()` and the conflict check (`ProvenanceController` through `conflict_history`, and each
ledger's `apply_assertion` under the key's lock) read the tables only when, in one snapshot, they
file every event the ledger holds for the tenant and every id they name still resolves; otherwise
they replay the history, which is the definition the tables must equal. A lag falls back to replay
rather than failing closed because the replay can never answer differently, and a lag is routine:
an older build writing the same database feeds nothing. `PostgresFactLedger` and `LiteFactLedger`
materialize what each append wrote inside its own transaction, in a savepoint, so a materialization
failure leaves the tables behind and never fails the append. `recall provenance rebuild` rebuilds a
tenant from its ledger; `--reconcile` files only what the tables lack (through the outbox and
`MaterializationRecovery` on PostgreSQL); `--check` reports. An event that names a fact of another
conflict key, which no write path produces, puts the tenant in replay until a rebuild.

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
