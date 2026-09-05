# Safety core mutation testing preregistration

Registered 2026-09-06, before collecting mutation results.

## Scope

The run targets four safety modules only:

* `recall/trust.py`
* `recall_mcp/server.py`
* `recall/generations.py`
* `recall/provenance_controller.py`

The committed harness is `scripts/safety_core_mutations.py`. It applies one explicit, plausible
source mutation at a time, runs the named tests in a fresh subprocess, restores the original bytes
in a `finally` block, and refuses duplicate or missing anchors. A mutation that makes the test
process fail is killed only when the baseline for that test command was green and the mutant did
not merely collect zero tests or time out.

## Predicted mutation families

The registry covers these decisions, without claiming to exhaust every operator in the files:

* Trust: transaction time visibility, ambiguous supersession, supersession precedence, confidence
  thresholding, calibrated versus fallback evaluation, and corpus controlled reference quoting.
* MCP server: verifier wiring, requested tenant binding, scope enforcement, rate limiting, and
  returning the authorised tenant store.
* Generations: production identity checks, production calibration checks, ready state checks,
  active generation retirement, and garbage collection retention.
* Provenance controller: trust availability, source revalidation, contradiction authorization,
  decision allowlisting, and materialization failure handling.

## Prediction

I predict that the established boundary tests will kill most registered mutations. I expect any
survivors to cluster around guards whose negative cases are not represented by the current fixtures,
especially card tenant or generation lineage mismatches and generation rollback edge cases. I will
report survivors as findings, not convert them into a passing score.

## Re-measurement

From the repository root, after the registration commit:

```powershell
python scripts/safety_core_mutations.py
```

The database-backed portion must run with a disposable `RECALL_TEST_DSN` created by
`scripts/session-db.sh up`. No mutation result is part of this preregistration until appended in a
later commit; the prediction above remains unchanged.
