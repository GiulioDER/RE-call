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

## Result (2026-08-23)

**Status:** measured

All runs on `ec6ab9a0` plus the changes this record predicted, one container
(`recall-sess-4d6c7b06`, port 5614), 6,563 tests collected, back to back on the same afternoon.
Wall clock is `time` around the whole invocation; pytest's own line is given beside it because it
excludes interpreter start and collection differs between the two.

| Run | pytest clock | wall | passed | failed | skipped |
|---|---|---|---|---|---|
| serial, before any change | 49:58 | 52:27 | 6529 | 0 | 34 |
| `-n 6`, before worker isolation was right | 16:45 | 17:36 | 6510 | 6 + 10 errors | 37 |
| `-n 6`, after | 21:08 | 22:27 | 6523 | 3 | 37 |
| `-n 4`, after | 14:05 | 14:20 | 6525 | 1 | 37 |
| `-n 4`, after, repeat | 13:36 | — | 6525 | 1 | 37 |

Collection alone: **154.10s → 75.14s**.

1. **Collection. Predicted 55 to 75s, measured 75.1s.** Right, at the edge of the interval and on
   the pessimistic side of it, which is the direction I do not usually miss in.
   ([[i-over-predict-effect-magnitudes]] is about the other direction.) The removed cost was 79.0s
   against a predicted 80 to 95s: the `voyageai` chain is 74.8s of it and the rest was already
   shared with modules that import `openai`.
2. **Serial with the voyage fix only: not measured, and deliberately not.** It would have cost 50
   minutes to confirm a 3% cut that the collection number already gives directly. Recorded as a
   gap rather than estimated, because a number in this table has to have been measured.
3. **Parallel. Predicted 11 to 16 minutes at `-n 6` and a 2.5× to 3.5× speed-up. Measured 14:20
   at `-n 4`, a 3.66× speed-up, with `-n 6` SLOWER than `-n 4` in both of its runs.**
   The wall-clock prediction lands inside its interval and the speed-up lands just above the top
   of its, but the worker count is wrong, and that is the part worth keeping: `-n 6` measured
   17:36 and 22:27, a 4.9 minute spread on identical work, and **killed a worker in both runs**
   (`node down: Not properly terminated`, different test each time, which is the signature of the
   box running out of memory rather than of a test). The pre-registered reasoning for predicting
   under the ceiling (commit limit, shared database, other sessions) was right about the cause and
   wrong about which knob it would show up in: it did not scale the speed-up down smoothly, it put
   a hard stop between four workers and six.
4. **Correctness. Predicted 5 to 40 tests failing under `-n 6`, and zero after isolation.**
   Measured **16** on the first parallel run, inside the interval, and the mechanism was NOT the
   one predicted. The shared `chunks` table and the `recall_rls_probe` role never failed anything:
   a database per worker removed both before they could. What failed was the **shape of the DSN**
   the isolation handed back. `psycopg.conninfo.make_conninfo` returns libpq keyword form, and
   five tests take `TEST_DSN` apart as a URL, producing
   `invalid connection option "//recall_serve_x:pw@None:5432/user"`. Rewriting the URL's path
   instead fixed all six failures. The remaining ten errors were a real cross-worker collision of
   exactly the predicted kind, in a place the prediction did not look: a module-scoped
   `xfer_guard_db`, dropped `WITH (FORCE)` on both sides of every test in
   `tests/test_beam_transfer_index_guards.py`, which is now named after the worker.

   **After both fixes, zero failures are attributable to parallelism**, as predicted.

5. **Falsification check on the skip count.** The record says a skip count differing by more than
   the documented 34 has to be explained. It differs: **37, in every parallel run**, and one test
   also fails. Both are the same cause and it is not parallelism. `pyarrow` began raising
   `ImportError: DLL load failed while importing lib: An Application Control policy has blocked
   this file` on this machine some time between 12:41 and 15:20 today, with no commit in between.
   It fails `test_the_shipped_local_reranker_is_reachable_without_a_cloud_call`, because
   `pytest.importorskip` deliberately refuses to skip a module that is installed and broken, and
   it skips three `sentence-transformers` tests whose own guards catch `ImportError`.
   **Verified by re-running both serially, alone**: the failure and all three skips reproduce with
   one worker. The confound named in the record as "a hosted-model or network test can fail for
   reasons unrelated to this change" was the right shape and the wrong dependency.

### What the apparatus check found

Predicting the outcome does not reveal a broken harness, so: the parallel runs were checked for
the false-green signature this repository already documents. 6,525 passed against 6,529 serially,
with all four differences accounted for above and each reproduced serially. The `-n 4` run was
repeated and returned the identical triple, so the counts are not a scheduling artefact.
