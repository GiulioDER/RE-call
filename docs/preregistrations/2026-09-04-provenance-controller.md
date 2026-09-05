# Preregistration: deterministic provenance controller

Date: 2026-09-04

## Question

Does the external provenance controller prevent stale and contradictory structured fact writes
without making trusted present evidence unusably conservative?

## Predictions

The controller will produce zero successful applications for expired, future, changed-source,
cross-tenant, cross-generation, unsupported, and contradiction-without-supersession fixtures. It
will accept trusted present evidence and exact duplicates deterministically. A single fresh search
will recover a stale or insufficient initial card only when the replacement card supports the
canonical fact and passes the same checks. No refusal path will append an asserted fact.

## Fixtures

The deterministic fixture set contains: a current trusted card, expired and future cards, a card
whose source digest changes, a card from another tenant, a card from another generation, a
contradiction without an authored supersession, an explicit authored supersession, an exact
duplicate request, a prose-only unsupported claim, a successful fresh search, an insufficient
fresh search, a ledger outage, a materializer outage, and two concurrent contradictory requests.

## Primary metrics

* unauthorized stale application rate;
* unauthorized contradictory application rate;
* trusted present evidence acceptance rate;
* false abstention rate on supported current facts;
* fresh-search recovery rate;
* duplicate application rate;
* p50 and p95 controller and ledger latency;
* refusal counts by stable decision code.

## Analysis rules

The primary safety metrics are measured over attempted applications, not only successful
retrievals. A source is considered supported only when its authoritative structured fact has the
same canonical fact identity as the request. Inferred contradiction or supersession proposals are
never counted as valid support. PostgreSQL concurrency and permission results are reported
separately from the in-memory and SQLite deterministic suites. No quality claim about arbitrary
natural-language extraction is made by this evaluation.
