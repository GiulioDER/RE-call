---
name: re-call-fp-run-20261021-9f8e7d-bug-003
description: "RE-call BUG-003 (false positive): Pool._lease can hand the same connection to two callers."
valid_from: 2026-10-21
recall_record: {"anchors":[{"code_sha":"279118d17555dada7c93533ef1fec82b01dc39b01b1df0a59bc6f2633524aabc","file":"recall/pool.py","symbol":"Pool._lease"},{"code_sha":"966827d6d28e3468e4df8c23952cb1a8c94dcbd0749bca7f92e7979870887f3c","file":"recall/store.py","symbol":"Store._with_retry_shared"}],"auditor":"bug-auditor","category":"concurrency","claim":"Pool._lease can hand the same connection to two callers.","date":"2026-10-21","evidence_refs":["recall/pool.py:88","recall/store.py:412"],"finding_id":"BUG-003","fp_reason":"not_real","high_stakes":false,"id":"RE-call:fp:run-20261021-9f8e7d:BUG-003","kind":"false_positive","priority":"P2","project":"RE-call","repo":"GiulioDER/RE-call","run_id":"run-20261021-9f8e7d","severity":"P2","v":1,"verify_source":"llm"}
metadata:
  node_type: memory
  type: project
  originSessionId: cca-bugs
  modified: 2026-10-21T00:00:00Z
---

# RE-call BUG-003: a false positive CCA accepted

## The claim

Pool._lease can hand the same connection to two callers.

## Why it is not a defect

Evidence: `recall/pool.py:88`, `recall/store.py:412`

## Where

* `recall/pool.py` `Pool._lease`
* `recall/store.py` `Store._with_retry_shared`

Run `run-20261021-9f8e7d`, auditor `bug-auditor`, P2 P2, verified by llm; accepted by the L6 architect.
