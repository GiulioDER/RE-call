---
name: re-call-lesson-6a3ae35062b9
description: "RE-call lesson: A connection's liveness must be judged inside the lease that owns it."
valid_from: 2026-10-21
recall_record: {"accepted_at":"2026-10-21","accepted_by":"owner","anchors":[{"code_sha":null,"file":"recall/pool.py","symbol":"Pool._lease"},{"code_sha":null,"file":"recall/store.py","symbol":null}],"date":"2026-10-21","id":"RE-call:lesson:6a3ae35062b9","kind":"lesson","members":["RE-call:0123456789ab:BUG-001","RE-call:abcdef012345:BUG-002","RE-call:fedcba987654:P2-4"],"pattern":"A read of conn.closed or conn.info after the pool may have taken the connection back.","project":"RE-call","repo":"GiulioDER/RE-call","rule":"A connection's liveness must be judged inside the lease that owns it.","v":1}
metadata:
  node_type: memory
  type: project
  originSessionId: cca-bugs
  modified: 2026-10-21T00:00:00Z
---

# RE-call: a lesson from 3 bugs

## The rule

A connection's liveness must be judged inside the lease that owns it.

## What to look for

A read of conn.closed or conn.info after the pool may have taken the connection back.

## Where

* `recall/pool.py` `Pool._lease`
* `recall/store.py` (module level)

## The bugs it comes from

* `RE-call:0123456789ab:BUG-001`
* `RE-call:abcdef012345:BUG-002`
* `RE-call:fedcba987654:P2-4`

Accepted by owner on 2026-10-21.
