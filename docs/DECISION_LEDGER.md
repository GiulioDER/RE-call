# The decision ledger

Opt-in, append-only records of what the trust layer decided and why, one row per search, in the
tenant's existing audit table. Off by default; nothing about retrieval, trust, evidence, or MCP
behaviour changes when it is off, and nothing about verdicts, ordering, refusals, or the
decision to abstain changes when it is on.

## The gap it closes

`trusted_search` already assembles a complete account of every decision at the moment it makes
one: the query that triggered it, every hit's verdict, cosine and calibrated confidence, the
calibration artifact and generation that governed the thresholds, the demoted hits that lost, and
the reason it abstained when nothing won. Until now that account lived exactly as long as the
`TrustedResult` object. The aggregate counters in `recall.observability` answer *how often* the
system abstains; nothing answered *why it abstained on Tuesday*, or which calibration was in
force when it confirmed something that later proved stale.

That second question is the operationally expensive one. Studying calibration drift today means
replaying history against old generations; a decision ledger accumulates the real decisions, with
the thresholds and identities that governed each one, as a side effect of serving.

## Witness, not enforcer

The trust layer is the enforcer: strict mode refuses, degraded mode demotes, and the gate sits
above retrieval so a refusal leaks nothing. The ledger only records what that boundary returned.
Three consequences are load-bearing, and tested:

- **A ledger failure never fails the search.** Writes are best-effort: a failed insert costs one
  counter increment (`recall_ledger_write_failures_total`) and one warning per failure kind per
  process, never an exception. `tests/test_decision_ledger.py` breaks the store on purpose and
  asserts the search result is unaffected.
- **A refusal is recorded and still raised.** A `TrustRefusal` is a decision too — whether the
  strict gate refused or a dependency fault ended the call in any mode — and the
  ledger appends a `search_refusal` record and re-raises the exception unchanged.
- **Records are appended after the decision is complete.** The write happens once, on the final
  outcome (after entailment demotion, after successor promotion), so the ledger cannot become a
  step inside the mechanism it documents.

## What a record holds, and what it never holds

Records land in `recall_audit_events` — the same append-only, RLS-isolated table the write side
(generation and calibration lifecycle) has used since migration 0008 — under two event types:

| `event_type` | When |
|---|---|
| `search_decision` | the call completed: answered or abstained |
| `search_refusal` | the search raised `TrustRefusal` before retrieval: the strict-mode gate, or a `DEPENDENCY_UNAVAILABLE` fault in any mode |

The JSONB payload (`record_version: 1`) carries:

- **the trigger**: the query, bounded at 2,000 characters with the true length and a truncation
  flag recorded beside it, plus `k`;
- **the outcome**: `answered`, `abstained`, or `refused`, with the abstention reason or the
  `TrustFailureCode`;
- **the authority that governed it**: `calibration_id`, `calibration_status`, `generation_id`,
  `pipeline_fingerprint`, `corpus_fingerprint`, `query_set_digest`, `trust_state`,
  `failure_code`;
- **the evidence, winners and losers**: every hit's chunk id, source, file, ord, cosine,
  confidence, verdict, validity window, successor, and last-index timestamp (`indexed_at`; the
  first-write axis, `first_indexed_at`, is not recorded in record version 1). The demoted
  hits are recorded on purpose — a ledger showing only supporting evidence is justification,
  not audit;
- **both time axes**: `valid_time` (the `now` the verdicts were judged against) and
  `known_as_of` (the transaction-time replay instant, non-null exactly when the record is about
  a reconstruction of the past), with `created_at` stamped by the database on the row.

**Chunk text never enters a record.** References only, so enabling the ledger cannot create a
second, unguarded copy of the corpus. The query *is* recorded — it is the trigger, and the
record is unreadable without it. That deliberately differs from `TrustRefusal`, which excludes
the query because an exception leaks into every log line that touches it; a ledger row lands
only in a tenant-isolated table behind serving credentials.

## Enabling it

Per process, for the CLI `search` command and the MCP service:

```bash
RECALL_DECISION_LEDGER=1 python -m recall.cli search "your question"
```

A malformed value warns once and stays off: raising would turn a typo in an env var into a
refusal of every search, which is enforcement, and the one thing the witness must not do.

Per call, from the typed API:

```python
from recall.decision_ledger import DecisionLedger
from recall.trust import trusted_search

result = trusted_search(store, embedder, query, ledger=DecisionLedger(store))
```

## Reading it back

```sql
SELECT created_at, actor, payload->>'outcome' AS outcome,
       payload->>'failure_code' AS failure_code, payload->>'query' AS query
FROM recall_audit_events
WHERE event_type IN ('search_decision', 'search_refusal')
ORDER BY created_at DESC LIMIT 20;
```

The table is tenant-isolated by row-level security, so read through a connection whose
`recall.tenant_id` GUC is set, exactly as for every other table here.

## Limits, stated rather than implied

- **Best-effort means missing rows are possible.** A dead database, a schema predating
  migration 0008, or a revoked grant loses records silently apart from the counter and the
  first warning. If the audit trail is compliance-mandatory rather than operational, this
  mechanism is not sufficient on its own.
- **No cryptographic custody.** Append-only is enforced by the code surface (the store exposes
  no update or delete for audit events) and by grants, not by hashes or signatures. A superuser
  can rewrite history.
- **The table grows without bound while the ledger is on.** One row per search, typically a few
  KB of JSONB plus two index entries, in the same `recall_audit_events` table the generation and
  calibration lifecycle writes to. The library ships no retention mechanism on purpose (append is
  the only verb on its surface), so retention is the operator's, outside the library — for
  example, on a schedule:

  ```sql
  DELETE FROM recall_audit_events
  WHERE event_type IN ('search_decision', 'search_refusal')
    AND created_at < now() - interval '90 days';
  ```

  Watch `recall_ledger_records_total` against `recall_ledger_write_failures_total`: together they
  are the write rate and the loss rate, and the failure counter alone cannot tell you either.
- **It witnesses observation, not omniscience.** A record proves what was retrieved, judged,
  and refused under which authority — not that the corpus contained everything relevant. If the
  world changed and nothing indexed the change, the ledger holds a flawless account of a
  decision that was already wrong. That gap belongs to indexing and drift detection, not to the
  record.

The record's shape follows the "reasoning ledger record" pattern described by K. W. Alger
(dev.to, 2026): trigger first-class, authority provenance, losing evidence preserved, bitemporal
axes, and mechanisms (revalidation, retrieval obligations) kept outside the record itself.
