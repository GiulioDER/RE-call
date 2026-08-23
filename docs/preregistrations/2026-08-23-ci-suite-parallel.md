# Pre-registration: running CI's pytest jobs in parallel

**Date:** 2026-08-23   **Status:** predicted, not yet measured

## The question

Does adding `-n auto` to the two `pytest` jobs in `.github/workflows/ci.yml` shorten the workflow,
and does it stay green? Two numbers and one yes/no: the `test` job's duration, the whole
workflow's duration, and whether the pass/fail counts and the coverage percentage are unchanged.

## What I predict

1. **`test` job: 8:55 today, 4:00 to 6:00 after.** A GitHub standard runner has 4 vCPU, so the
   ceiling is 4×, and I deliberately predict well under it: coverage instrumentation does not
   parallelise away, install and checkout are serial, and a Postgres service container shares
   those same 4 vCPU with the workers.
2. **`floor` job: 7:23 today, 3:30 to 5:00 after.**
3. **Whole workflow: 9:00 today, 5:00 to 6:30 after**, bounded below by `Clean install (wizard)`
   at 4:59, which this does not touch. So the workflow cannot beat ~5:00 whatever the test jobs
   do, and that is the point at which further parallelism there stops buying anything.
4. **Coverage unchanged within 0.5 points**, and `--cov-fail-under=70` still passing.
   `pytest-cov` combines per-worker data; if that is broken the number collapses rather than
   drifting, so this is a check with a sharp edge.
5. **Green, with zero failures attributable to parallelism.** The same isolation the local runs
   needed is already in `tests/conftest.py::_isolate_xdist_worker`, and CI's role is the
   `postgres` image superuser, so `CREATE DATABASE` per worker is available.

## What would falsify this

- The `test` job at or above 7:00 (under 1.3× speed-up): 4 vCPU shared with the database is not
  enough to parallelise into, and the change should be reverted rather than tuned.
- Any test failing on CI that passes serially there.
- Coverage moving more than 0.5 points in either direction, or `--cov-fail-under` tripping:
  worker coverage data is not being combined and the gate has become decorative.
- A flake rate above zero across the first five runs. A gate everyone depends on may not become
  intermittent to save four minutes.

## How it will be measured

Push the branch and read the run, rather than trusting the local numbers, because the local box
is Windows with 12 GB and the runner is Linux with 4 vCPU and 16 GB:

```bash
gh run list --workflow ci.yml --branch <branch> --limit 1
gh run view <id> --json jobs -q '.jobs[] | "\(.name)\t\(.startedAt)\t\(.completedAt)\t\(.conclusion)"'
```

- **job seconds**: `completedAt` minus `startedAt` per job, which includes install and checkout,
  because that is what a contributor waits for.
- **coverage**: the `--cov-fail-under=70` line's reported total, n = the `recall` and `recall_mcp`
  packages only.
- **agreement**: pass/fail/skip counts against the current serial CI run.

## What I already know

Measured today from run `32641788574` on master: `test` 8:55, `floor` 7:23,
`Clean install (wizard)` 4:59, whole workflow 8:59. Locally, on Windows with 12 logical cores,
the same suite went from 52:27 to 14:20 of wall clock at `-n 4`, and `-n 6` was slower than `-n 4`
twice while killing a worker both times: docs/preregistrations/2026-08-23-test-suite-wall-clock.md.

⚠️ **`-n auto` is the right setting on CI and the wrong one locally**, and the reason is the
constraint that actually binds. Locally `auto` asks for 12 workers on a box whose commit limit was
already 97% spent, which is why `make test` pins 4. On a 4 vCPU runner `auto` asks for 4, which is
the number I would have chosen anyway.

## Confounds I can name now

- **Runner variance.** GitHub's shared runners are not identical machines, and a single run either
  side is one sample. If the difference is under about 1.3×, five runs are needed before believing
  it, not one.
- **CI installs `.[dev]` only**, so the model-backed tests self-skip there and its suite is a
  SMALLER set than a local run. A speed-up measured on CI does not transfer to a local number and
  must not be quoted as one.
- **The `floor` job resolves `pytest-xdist` to its declared floor**, 3.6.0, not to the 3.8.0
  measured locally. A scheduling bug fixed between them would show up only there.

## Result (2026-08-23)

**Status:** measured

Run `32645204630` on `claude/recall-test-suite-speed-14a62d`, against run `32641788574` on master
as the before. Both on GitHub standard runners.

| | before | after | predicted |
|---|---|---|---|
| `test` job | 8:55 | **3:55** | 4:00 to 6:00 |
| `floor` job | 7:23 | **3:01** | 3:30 to 5:00 |
| whole workflow | 8:59 | **7:04** | 5:00 to 6:30 |
| coverage | 79.56% | **79.59%** | unchanged within 0.5 |
| test counts | 6421 passed, 150 skipped | 6429 passed, 150 skipped | agreement |

1. **`test`: predicted 4:00 to 6:00, measured 3:55.** Just under the optimistic end, so the
   prediction was right in shape and slightly pessimistic in size. pytest's own clock went 8:00 to
   3:08, a 2.6× cut on 4 vCPU.
2. **`floor`: predicted 3:30 to 5:00, measured 3:01.** Also under. Both jobs beat the interval by
   a similar margin, which suggests the thing I under-weighted was the same in both: install and
   checkout are a larger fixed share of these jobs than I assumed, so the parallel part shrank
   more than the job total implied.
3. **Whole workflow: predicted 5:00 to 6:30, measured 7:04. FALSIFIED, and the cause is not
   parallelism.** The bound named in the record was `Clean install (wizard)` at 4:59, and it
   still finished at 4:59. What set 7:04 was `Desktop UI`, which **queued for five minutes**
   before starting (14:26:58 against the workflow's 14:21:56) and then ran for two. Runner
   availability, not work. So the test jobs are no longer the critical path, which was the point,
   and the workflow's remaining minutes are now somebody else's queue.
4. **Coverage: 79.56% to 79.59%, +0.03 points**, well inside the 0.5 predicted, with
   `--cov-fail-under=70` passing. `pytest-cov` combines the workers' data correctly. This was the
   check with a sharp edge and it held.
5. **Counts: +8 passed, skips identical.** Not a discrepancy: the master baseline run predates
   two merges (#480, #481) that this branch was rebased onto, and #481 adds exactly 8 tests to
   `tests/test_setup.py`. Verified by diffing the collected node ids between the two revisions,
   not by assuming.
6. **Green, but not on the first attempt, and the first attempt is the useful part.** Run
   `32644567899` failed `floor` with
   `test_a_poisoned_connection_is_discarded_through_the_real_return_path`: `assert 0 == 1` on
   `pool.reset_discards`. The counter is incremented inside `_reset`, which **psycopg_pool runs on
   a worker thread**, so the test was asserting that a thread had already been scheduled. It won
   every time on an idle machine and lost the first time four workers shared four vCPU.
   **A pre-existing race in the test, not in the pool, and not caused by the isolation**;
   parallelism stopped hiding it. Fixed with a bounded wait that leaves the assertion at `== 1`.

### The falsifier that has not been discharged

The record says "a flake rate above zero across the first five runs" disqualifies this. **Two runs
have happened, one of which was red.** That red is explained and fixed at its cause rather than
retried, but the honest position is that this is one green run, not five, and the fifth is the
one that decides whether a shared gate has become intermittent. Anyone merging this should watch
the next few runs rather than treat 3:55 as settled.
