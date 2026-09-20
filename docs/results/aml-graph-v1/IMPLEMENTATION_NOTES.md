# AML grounded graph sidecar

Status: implemented, endpoint tested, and selected for the next official AML compatibility Smoke.

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

This implementation makes the graph measurable and eligible for a frozen AML endpoint submission.
The current official Open-source Methods checklist requires `gpt-4o-mini` during Add and Search,
memory-evidence-only Search output, sample isolation, a passing platform Smoke, and a version that
matches the declared submission. `G1_grounded_graph` satisfies the locally inspectable model,
output, isolation, and version-binding conditions. It does not establish an AML Task Solve gain or
an official AML result. Full remains blocked until the platform compatibility Smoke passes for the
exact frozen deployment and the remaining official checklist and promotion gates pass.

Official source checked 2026-09-20: the
[Cycle 2 documentation](https://agentmemoryleaderboard.ai/docs) defines the Open-source Methods
model rule, evidence-only Search boundary, sample isolation requirement, fixed-version review,
Smoke gate, and Full gate.
