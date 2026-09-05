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

## Results

Measured 2026-09-06 after commit `7501c808`, with `RECALL_TEST_DSN` set to the disposable session
database created by `scripts/session-db.sh up`, using:

```powershell
python scripts/safety_core_mutations.py
```

| Module | Killed | Survived | Inconclusive | Stale | Total |
|---|---:|---:|---:|---:|---:|
| `recall/trust.py` | 5 | 1 | 0 | 0 | 6 |
| `recall_mcp/server.py` | 4 | 1 | 0 | 0 | 5 |
| `recall/generations.py` | 4 | 1 | 0 | 0 | 5 |
| `recall/provenance_controller.py` | 3 | 2 | 0 | 0 | 5 |
| Total | 16 | 5 | 0 | 0 | 21 |

The five survivors are:

* `trust.py`: calibrated input is ignored by the selected tests when the uncalibrated fallback is
  forced.
* `server.py`: an explicit requested tenant is not covered by the selected tool authorization
  calls.
* `generations.py`: no selected test promotes a generation that is not ready.
* `provenance_controller.py`: no selected test supplies an untrusted card, and the contradiction
  guard is masked by the in memory ledger's independent contradiction check.

Every mutation was restored by bytes. The harness reported matching SHA 256 digests for all four
targets, and the worktree was clean after the run. The result is a test strength measurement, not
a claim that the five surviving behaviours are safe.
