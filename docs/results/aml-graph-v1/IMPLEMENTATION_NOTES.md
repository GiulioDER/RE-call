# AML grounded graph sidecar

Status: implemented and contract tested, not promoted as the AML submission default.

## Purpose

The `G1_grounded_graph` variant tests whether the existing source anchored `gpt-4o-mini`
compiler can improve raw evidence ordering through explicit corpus relations without allowing
generated records to replace raw memory or widen the candidate set.

## Add path

1. Raw message segments are stored in the normal user tenant.
2. Compiler v3 may create typed records using server generated evidence anchors.
3. The server resolves accepted evidence spans back to raw chunk IDs.
4. It emits authored `references` relations from the compiled record to each overlapping raw
   evidence chunk. The model does not choose either relation endpoint.
5. Compiled records and relation metadata are stored in a tenant derived graph sidecar. They do
   not compete with raw records in the primary dense or lexical candidate pool.
6. A compiler error, an empty grounded result, or an unsupported span stores raw memory only.

## Search path

1. The frozen raw dense plus lexical search produces the baseline top 100.
2. The graph sidecar is searched independently with the same query.
3. Only authored `references` edges with matching subject, target file, source session, ordinal,
   and character overlap are admitted.
4. The first eight raw results are immutable.
5. At most two related raw items already present in the baseline top 100 may move to ranks 9 and
   10. No new item can enter the result set.
6. Superseded compiled seeds are ignored for current queries and remain available for explicitly
   historical queries.

## Fail safe behavior

No sidecar hits, no eligible relation, malformed relation data, a cross session edge, unavailable
sidecar storage, or unavailable sidecar corpus status returns the original raw hit order and
scores. Search exposes content free activation, fallback, relation, candidate, promotion, invalid
relation, order change, and membership change headers.

## Selection boundary

This implementation makes the graph measurable. It does not establish an AML Task Solve gain and
does not authorize an official Full run. Any quality measurement needs a committed
preregistration, the exact frozen endpoint identity, and the existing retrieval plus executable
promotion gates.
