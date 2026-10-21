---
name: re-call-fedcba987654-p2-4
description: "RE-call P2-4 (historical): The retry read conn.closed after the pool took it back."
valid_from: 2026-08-03
recall_record: {"anchors":[{"code_sha":"966827d6d28e3468e4df8c23952cb1a8c94dcbd0749bca7f92e7979870887f3c","file":"recall/store.py","symbol":"Store._with_retry_shared"}],"auditor":"bug-auditor","bug_id":"RE-call:fedcba987654:P2-4","category":"concurrency","caused_by":[],"date":"2026-08-03","evidence":{"confirmed_l25":"claimed","fixed":"evidenced","l6_approved":"claimed","red_proof":"absent"},"files":["recall/store.py"],"finding_id":"P2-4","fix_commit":"fedcba9876543210fedcba9876543210fedcba98","id":"RE-call:fedcba987654:P2-4","kind":"bug","ordinal":1,"project":"RE-call","repo":"GiulioDER/RE-call","run_id":"run-20261020-1a2b3c","severity":"P2","sources":["git:fedcba9876543210fedcba9876543210fedcba98"],"summary":"The retry read conn.closed after the pool took it back.","supersedes":null,"tier":"historical","v":1}
metadata:
  node_type: memory
  type: project
  originSessionId: cca-bugs
  modified: 2026-10-21T00:00:00Z
---

# RE-call P2-4: a bug CCA fixed

## The bug

The retry read conn.closed after the pool took it back.

## The fix

Commit `fedcba9876543210fedcba9876543210fedcba98` in `GiulioDER/RE-call`, 2026-08-03. Files changed by that commit:

* `recall/store.py`

## Where

* `recall/store.py` `Store._with_retry_shared`

## The evidence

Tier: **historical**.

| Condition | Value |
|---|---|
| confirmed_l25 | claimed |
| fixed | evidenced |
| l6_approved | claimed |
| red_proof | absent |

Sources: `git:fedcba9876543210fedcba9876543210fedcba98`
