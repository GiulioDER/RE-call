# Preregistration: production structural edge canary

Status: locked before production canary ingestion.

## Objective

Verify that the production serving path can ingest explicit `recall_graph` metadata, build an
authored relation, and add the related evidence during a bounded one hop reasoning query.

This is a production contract and serving smoke test. It is not a benchmark quality result and it
does not alter the existing `memory` tenant.

## Isolation

Use only tenant `structural-edge-canary-20260910`. Upload three small Markdown files, then remove
all three sources from that tenant after the paired query. Do not use the existing `memory` tenant,
do not publish calibration, and do not modify the production serving checkout during the run.

## Fixture and query

The fixture contains `a.md`, `b.md`, and `c.md`. `a.md` declares one explicit `references` relation
to `b.md` with structural type `same_email_thread`. `c.md` is unrelated. The query is
`Who owns the booking?`, with `k=1`.

## Arms

1. `off`: `graph_expansion=off`.
2. `one_hop`: `graph_expansion=one_hop` with the same query, tenant, generation, and budget.

The generation identity returned after ingestion must match for both arms. The baseline item must
be the same in both arms. The treatment is expected to add `b.md` through the authored relation.

## Pass criteria

The canary passes only if production returns a valid trusted response for both arms, the generation
identity is identical, graph off returns no appended graph item, one hop reports the authored
relation and adds exactly `b.md`, and the production tenant is empty after cleanup.

Any failure is reported as a failed canary. No retry with altered query, ranking, relation shape,
or budget is allowed.
