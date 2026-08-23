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
