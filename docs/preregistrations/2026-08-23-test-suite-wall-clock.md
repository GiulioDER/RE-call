# Pre-registration: cutting the test suite's wall clock

**Date:** 2026-08-23   **Status:** predicted, not yet measured

## The question

Does (a) removing the `voyageai` import from collection and (b) running `tests/` under
`pytest-xdist` cut the full suite's wall clock on this workstation, and by how much, without
changing which tests pass?

Two numbers and one yes/no: seconds of collection, seconds of full-suite wall clock, and whether
the pass/fail/skip counts are identical to the serial run.

## What I predict

1. **Collection.** Serial collection is **154.1s** today (measured below). Making the three
   `voyageai` test modules import it lazily removes **80 to 95s** of it, leaving collection at
   **55 to 75s**. This is the confident one: the cost is a single measured import chain.
2. **Serial wall clock, voyage fix only.** Baseline is 30 to 40 minutes. The fix removes ~90s,
   so **a 3 to 6% cut, and nothing more**. It is not a meaningful lever on its own; it matters
   because collection is paid once **per xdist worker**.
3. **Parallel wall clock.** With the fix and `-n 6` on 12 logical cores, the suite runs in
   **11 to 16 minutes**, i.e. a **2.5× to 3.5× speed-up**, not the 6× the worker count suggests.
   I deliberately predict well under the ceiling: this box is at its commit limit (29.2 GB
   committed of a 30.1 GB limit, 1.28 GB available, measured today), four other session Postgres
   containers are already running, and both the database and the disk are shared by the workers.
   `-n 12` I predict is **no better than `-n 6` and possibly worse**, because of memory.
4. **Correctness.** Going parallel exposes shared state. I predict **5 to 40 tests** fail under
   `-n 6` that pass serially, concentrated on the shared default `chunks` table
   (`tests/conftest.py::restore_default_chunks_table`, dropped and rebuilt by
   `tests/test_wizard_database.py`) and on the cluster-wide unprivileged role in
   `unprivileged_dsn`. Per-worker database isolation is predicted to take that to **zero**.

## What would falsify this

- Collection after the fix still above 100s: the voyage chain was not the cost.
- Parallel wall clock above 24 minutes (under 1.7× speed-up): parallelism is not the lever here
  and the suite is bound by something shared (the single Postgres container, or the disk).
- Any test that passes serially and fails under `-n 6` after the isolation work: the isolation is
  incomplete and the speed-up is not free.
- Skip count differing from the serial run by more than the 34 documented skips: a worker that
  quietly lost its database looks exactly like a fast run.

## How it will be measured

One database container of this checkout's own, started with `scripts/session-db.sh up`, and
three runs on the same commit, back to back, nothing else of mine running:

```bash
python -m pytest tests/ -q --durations=50            # serial baseline
python -m pytest tests/ -q -p no:cacheprovider -n 6  # parallel
python -m pytest tests/ -q --collect-only            # collection only, before and after
```

Metric names, because each has a different denominator:

- **collection seconds**: pytest's own "collected in Ns" line, n = 6563 tests.
- **wall clock seconds**: `time` around the whole invocation, n = 6563 tests, INCLUDING
  collection and interpreter start, because that is what a developer waits for.
- **agreement**: the triple (passed, failed, skipped) compared between serial and parallel.

## What I already know

Measured today, on this worktree at `ec6ab9a0`, before writing this record:

- Collection alone is **154.10s** for 6563 tests; a second profiled run gave 113.6s of collect
  with ~27s of interpreter and conftest start on top.
- One module, `tests/test_embeddings_retry_after.py`, accounts for **86.9s** of that, and **94.4s**
  when collected alone. `python -X importtime` attributes it to `voyageai` (74.8s cumulative),
  which imports `langchain_text_splitters`, which imports `transformers` (31.4s) and `torch`.
  The module's own comment already names this cost and accepts it; three modules do the same.
- `pytest-xdist` is **not installed** in this environment.
- `os.cpu_count()` is 12. Memory as above.

`CLAUDE.md` records the suite at **37:13, 39:20 and 29:45** on 2026-08-20 at 6088 tests, with the
30 slowest tests accounting for 511s of 1785s (29%) and the remainder averaging 0.29s. That is the
finding this record builds on: there is no single hot test to fix, so the lever has to be
parallelism rather than surgery.

## Confounds I can name now

- **This machine is loaded and not by me alone.** Four other session containers are up and other
  Claude sessions are working in other worktrees. A 10-minute spread across three identical runs
  is already documented. Any speed-up under ~1.3× is inside that noise, which is why the
  falsification threshold above is set at 1.7×.
- **A hosted-model or network test can fail for reasons unrelated to this change.**
  `tests/test_entailment.py::test_qnli_judge_separates_answering_from_adjacent_text` downloads
  from HuggingFace and has already failed once on DNS. It must be read as a network failure, not
  as a parallelism regression.
- **Warm caches.** The first run of the day pays `__pycache__` compilation and the fastembed model
  load. The baseline is therefore run first and the parallel run second, which biases IN FAVOUR of
  the parallel run. If the two are close, that ordering is the reason, and the runs get swapped.
- **`-n` changes ordering, not just timing.** A test that passes serially because a neighbour ran
  first will fail under xdist for a reason that was always a latent bug, and counting it against
  parallelism would be wrong.
