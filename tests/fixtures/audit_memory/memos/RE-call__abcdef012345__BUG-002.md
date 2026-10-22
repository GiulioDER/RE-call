---
name: re-call-abcdef012345-bug-002
description: "RE-call BUG-002 (proved): The liveness verdict is now taken inside the lease; the earlier fix moved the read too late."
valid_from: 2026-08-05
supersedes: RE-call__fedcba987654__P2-4
recall_record: {"anchors":[{"code_sha":"966827d6d28e3468e4df8c23952cb1a8c94dcbd0749bca7f92e7979870887f3c","file":"recall/store.py","symbol":"Store._with_retry_shared"},{"code_sha":"279118d17555dada7c93533ef1fec82b01dc39b01b1df0a59bc6f2633524aabc","file":"recall/pool.py","symbol":"Pool._lease"}],"auditor":"bug-auditor","bug_id":"RE-call:abcdef012345:BUG-002","category":"concurrency","caused_by":["RE-call:0123456789ab:BUG-001"],"date":"2026-08-05","evidence":{"confirmed_l25":"evidenced","fixed":"evidenced","l6_approved":"evidenced","red_proof":"evidenced"},"files":["recall/store.py","recall/pool.py"],"finding_id":"BUG-002","fix_commit":"abcdef0123456789abcdef0123456789abcdef01","id":"RE-call:abcdef012345:BUG-002","kind":"bug","ordinal":1,"project":"RE-call","repo":"GiulioDER/RE-call","run_id":"run-20261020-1a2b3c","severity":"P1","sources":["git:abcdef0123456789abcdef0123456789abcdef01"],"summary":"The liveness verdict is now taken inside the lease; the earlier fix moved the read too late.","supersedes":"RE-call:fedcba987654:P2-4","tier":"proved","v":1}
metadata:
  node_type: memory
  type: project
  originSessionId: cca-bugs
  modified: 2026-10-21T00:00:00Z
---

# RE-call BUG-002: a bug CCA fixed

## The bug

The liveness verdict is now taken inside the lease; the earlier fix moved the read too late.

## The fix

Commit `abcdef0123456789abcdef0123456789abcdef01` in `GiulioDER/RE-call`, 2026-08-05. Files changed by that commit:

* `recall/store.py`
* `recall/pool.py`

## Where

* `recall/store.py` `Store._with_retry_shared`
* `recall/pool.py` `Pool._lease`

## Caused by

* `RE-call:0123456789ab:BUG-001`

## The evidence

Tier: **proved**.

| Condition | Value |
|---|---|
| confirmed_l25 | evidenced |
| fixed | evidenced |
| l6_approved | evidenced |
| red_proof | evidenced |

Sources: `git:abcdef0123456789abcdef0123456789abcdef01`
