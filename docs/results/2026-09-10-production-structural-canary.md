# Production structural edge canary result

Date: 2026-09-10

Status: inconclusive, failed closed before the preregistered effect test.

## Scope

The canary used only tenant `structural-edge-canary-20260910` and the locked fixture from
`docs/preregistrations/2026-09-10-production-structural-canary.md`. The fixture contained
`a.md`, `b.md`, and unrelated `c.md`. The query was `Who owns the booking?`, with `k=1`.

Production serving checkout at measurement time:

```text
bccf4d80d7a92701d10ffbfee9cf3ed3ab2eadeb
```

The canary generation was `gen_ed341f43ef39462f9664d706443f0d53`, using certified calibration
`cal_3f0b889388224140bd19263dc238d0d4`, threshold `0.447`, and calibration checksum
`7e30e62cfd852e75a76e0196ac73ad08fab2124dd5966b29fe40f93e8f65cb05`.

## Paired result

Both preregistered reasoning arms returned the same server error before producing a response:

```text
Error executing tool recall_reasoning_query: 'ReasoningPlan' object has no attribute 'steps'
```

The `off` and `one_hop` arms therefore produced no valid trusted evidence, no comparable graph
diagnostics, and no production effect estimate. I did not retry with altered query, ranking,
relation shape, or budget.

## Diagnostic retrieval check

After the failed paired query, ordinary production `recall_search` succeeded against the canary
tenant and returned `a.md`, `b.md`, and low-confidence `c.md`. This only confirms that ingestion and
ordinary retrieval worked. It does not measure structural edge use because `b.md` was already a
dense or sparse retrieval hit.

The production related-expansion path failed independently with:

```text
Error executing tool recall_search: column "id" does not exist
LINE 1: SELECT id, source, text, metadata FROM recall_chunks_v1 WHER...
```

This indicates that the deployed structural expansion path is incompatible with the production
`recall_chunks_v1` schema at this checkout.

## Cleanup

Inventory before cleanup contained exactly the three staged canary sources. `recall_forget` removed
3 chunks, 3 sources, and 3 staged upload files, with no sources not found. A follow-up inventory in
the same isolated tenant returned an empty entry list. The existing `memory` tenant was not touched.

## Conclusion

This is not evidence for or against the value of structural edges in production. The canary exposed
two production blockers that must be repaired before rerunning the locked measurement:

1. reasoning response instrumentation accesses `ReasoningPlan.steps`, while the deployed plan
   exposes its execution trace through a different field;
2. related structural expansion queries a non-existent `recall_chunks_v1.id` column.

The offline LoCoMo result remains the only valid effect measurement so far.
