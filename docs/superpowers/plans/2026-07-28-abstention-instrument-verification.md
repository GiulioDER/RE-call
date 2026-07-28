# Abstention Instrument Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every eval result artifact carry proof of the corpus it was measured against, then inventory which published abstention claims still describe the shipped pipeline and re-measure the cheap stale ones.

**Architecture:** Three phases from [the design spec](../specs/2026-07-28-abstention-instrument-verification-design.md). Phase 0 adds an exact post-condition (`store.count() == stats.chunks`) and carries the row count into every result JSON. Phase 1 writes an inventory table. Phase 2 re-measures the two cheap stale claims and puts the expensive one to an explicit decision. **This plan stops at the end of Phase 2** — Phase 3 (the written verdict that gates the next cycle) is deliberately out of scope and gets its own plan.

**Tech Stack:** Python 3.11+, pytest, psycopg 3, pgvector, Postgres via Docker Compose.

## Global Constraints

- Import name is `recall`; distribution name is `recall-rag`. Never `pip install recall`.
- Python floor is `>=3.11` (`pyproject.toml`).
- Postgres-backed tests need Docker: run `make db-up` before `make test`. Tests decorated `@requires_db` skip without it — **a skipped test is not a passing test**; check the summary line for `skipped`, not just `failed`.
- Conventional Commits (`feat:`, `test:`, `fix:`, `docs:`, `chore:`).
- **This cycle adds no capability.** No new signal, no threshold change, no abstention-policy change. Do not touch `recall/trust.py`, `recall/calibration.py` or `recall/guards.py`.
- **Do not re-run #103.** Its numbers are sound and `rerank_modern` alone costs 3.8 hours.
- **Do not back-fill row counts into existing artifacts.** A count reconstructed after the fact is a claim, not a measurement.
- Other sessions share the `~/Documents/recall` clone. This plan runs in the worktree `~/Documents/recall-abstention-spec` on branch `docs/abstention-instrument-verification`. Re-read `origin/master` before trusting any line number below — "already merged" is a race, not an error.
- Measurement steps run on a metered, shared box. Run them **one at a time** and confirm each reached a terminal state before starting the next. Poll every terminal state and say which one fired; exit code 0 is not a measurement.

## File Structure

| file | responsibility | phase |
|---|---|---|
| `recall/eval/provenance.py` *(new)* | one function producing the `{corpus_rows, table, tenant, git_sha}` block every runner embeds | 0 |
| `recall/eval/locomo.py` | post-condition assert in `run_conversation`; provenance in the report | 0 |
| `recall/eval/locomo_abstention.py` | provenance in the report | 0 |
| `recall/eval/locomo_entailment_sweep.py` | provenance in the report | 0 |
| `tests/test_eval_provenance.py` *(new)* | pure-function tests for the provenance block | 0 |
| `tests/test_locomo_corpus_postcondition.py` *(new)* | the post-condition, verified on its detection path | 0 |
| `results/INSTRUMENT_STATUS.md` *(new)* | the inventory table | 1 |
| `results/locomo_rerank/PREDICTION-*.md` pattern | one prediction per Phase 2 measurement | 2 |

---

### Task 1: The corpus post-condition — catch a *concurrent* second writer

`run_conversation` already refuses to index over an existing tenant (`recall/eval/locomo.py:274-282`). That guard is a **pre**-condition: it reads `store.count()` before indexing. The failure it was written for was two *concurrent* launchers — and a second writer that lands **during** indexing passes the pre-check and still doubles the corpus.

`Indexer.index_path` returns `IndexStats` with an exact `chunks` field (`recall/index.py:298-302`) and `run_conversation:285` discards it. Comparing it to `store.count()` afterwards closes the window.

**Files:**
- Create: `tests/test_locomo_corpus_postcondition.py`
- Modify: `recall/eval/locomo.py:285` (the `Indexer(...).index_path(...)` call) and the `run_conversation` return dict
- Read for context: `recall/eval/locomo.py:237-300`, `recall/index.py:296-330`, `tests/conftest.py`

**Interfaces:**
- Consumes: `Indexer.index_path(path) -> IndexStats` with field `chunks: int`; `PgVectorStore.count() -> int` (tenant-scoped, `recall/store.py:1451`).
- Produces: `run_conversation(...)` return dict gains key `corpus_rows: int`. Task 2 reads it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_locomo_corpus_postcondition.py`:

```python
"""`run_conversation` must verify the corpus it just built, not just the one it found.

The shipped guard at recall/eval/locomo.py:274 is a PRE-condition: it reads store.count()
before indexing and refuses if rows are already there. That catches a sequential re-run.

It does not catch what actually happened on 2026-07-27: two CONCURRENT launchers. The second
one passes the pre-check (the table was empty when it looked) and writes while the first is
still indexing. Both finish, every tenant holds its corpus twice, nothing errors, and every
depth of the curve comes in ~0.05 low — plausible, self-consistent and wrong.

The post-condition is exact and cheap: Indexer.index_path returns IndexStats.chunks, which is
how many chunks THIS call wrote. On a fresh tenant the tenant-scoped row count must equal it.
Anything else means someone else wrote here.
"""
from __future__ import annotations

import pytest

from recall.embeddings import HashingEmbedder
from recall.eval.locomo import run_conversation
from recall.types import Chunk

from tests.conftest import requires_db

DIM = 64

_CONVERSATION = {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:00 pm on 8 May, 2023",
    "session_1": [
        {"speaker": "Caroline", "dia_id": "D1:1", "text": "I finally adopted a greyhound."},
        {"speaker": "Melanie", "dia_id": "D1:2", "text": "I signed up for a pottery class."},
    ],
}
_QA = [{"question": "What did Caroline adopt?", "category": 1, "evidence": ["D1:1"]}]


@requires_db
def test_run_conversation_reports_the_rows_it_indexed(make_store, tmp_path):
    """The happy path: the count is reported, and it is the count that was written."""
    store = make_store(DIM)
    res = run_conversation(
        _CONVERSATION, _QA,
        store=store, embedder=HashingEmbedder(dim=DIM), k=5, corpus_dir=tmp_path / "corpus",
    )
    assert res["corpus_rows"] > 0, "the run reported no corpus at all"
    assert res["corpus_rows"] == store.count(), (
        "corpus_rows must be the tenant's actual row count, not a number carried from elsewhere"
    )


@requires_db
def test_a_concurrent_writer_fails_the_run_instead_of_depressing_it(
    make_store, tmp_path, monkeypatch
):
    """A second writer landing DURING indexing must fail the run, not skew it.

    Simulated by writing extra rows into the same tenant from inside index_path, which is
    exactly the window the pre-condition cannot see.
    """
    store = make_store(DIM)
    embedder = HashingEmbedder(dim=DIM)

    from recall.index import Indexer

    real_index_path = Indexer.index_path

    def racing_index_path(self, *args, **kwargs):
        stats = real_index_path(self, *args, **kwargs)
        # The "other launcher", arriving after the pre-check and before the post-check.
        store.upsert(
            [Chunk(id="intruder", source="intruder.md", text="another run wrote this", metadata={})],
            [[0.1] * DIM],
        )
        return stats

    monkeypatch.setattr(Indexer, "index_path", racing_index_path)

    with pytest.raises(RuntimeError, match="corpus_rows|row count|indexed"):
        run_conversation(
            _CONVERSATION, _QA,
            store=store, embedder=embedder, k=5, corpus_dir=tmp_path / "corpus",
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
make db-up && python -m pytest tests/test_locomo_corpus_postcondition.py -v
```

Expected: both FAIL — the first with `KeyError: 'corpus_rows'`, the second with `DID NOT RAISE`.

If either is *skipped*, Docker is not up. A skipped test is not a passing test — fix Docker and re-run before continuing.

- [ ] **Step 3: Implement the post-condition**

In `recall/eval/locomo.py`, replace line 285:

```python
    Indexer(store, embedder).index_path(corpus_dir)
```

with:

```python
    stats = Indexer(store, embedder).index_path(corpus_dir)

    # Post-condition, and NOT a restatement of the pre-check above.
    #
    # The pre-check reads the table BEFORE indexing, so it catches a sequential re-run and
    # nothing else. On 2026-07-27 the failure was two CONCURRENT launchers: the second passed
    # the pre-check on an empty table and wrote while the first was still indexing. Both
    # finished, every tenant held its corpus twice, and nothing errored.
    #
    # `stats.chunks` is how many chunks THIS call wrote. On a tenant the pre-check just proved
    # empty, the tenant-scoped count must equal it. A larger count means another writer is in
    # this tenant, and the run is measuring a corpus nobody described.
    indexed = store.count()
    if not allow_existing and indexed != stats.chunks:
        raise RuntimeError(
            f"tenant {store.tenant!r} holds {indexed} chunk(s) after indexing but this run wrote "
            f"{stats.chunks}. Another writer is in this table CONCURRENTLY — the pre-check cannot "
            f"see one that arrives mid-index. Every hit@k from this corpus would be depressed "
            f"without erroring. Drop the table and re-run alone; take a lock if two runs must "
            f"share a host (see scripts/run_locomo_arms.sh)."
        )
```

Then add the count to the `run_conversation` return dict. Find the `return {` that ends `run_conversation` and add as its first key:

```python
        "corpus_rows": indexed,
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_locomo_corpus_postcondition.py -v
```

Expected: PASS, 2 passed, 0 failed, 0 skipped.

- [ ] **Step 5: Verify the guard is not vacuous**

The second test already exercises the detection path — but confirm the *first* one would notice a wrong value.

⚠️ **Use an editor for both edits. Do not `git checkout` this file** — Step 3's implementation is uncommitted, so a checkout would discard it and you would silently re-run Step 4 against the original code.

1. Edit `recall/eval/locomo.py` and change `"corpus_rows": indexed,` to `"corpus_rows": 999,`.
2. Run:

```bash
python -m pytest tests/test_locomo_corpus_postcondition.py -v
```

Expected: `test_run_conversation_reports_the_rows_it_indexed` FAILS on the `== store.count()` assertion. If it PASSES, the test is not reading the field it claims to and must be fixed before continuing.

3. Edit the same line back to `"corpus_rows": indexed,` and re-run the command above. Expected: PASS, 2 passed.

- [ ] **Step 6: Confirm nothing else broke**

```bash
python -m pytest tests/ -q -k "locomo or index or store"
```

Expected: 0 failed. Note the skipped count and confirm it is only the non-DB skips you started with.

- [ ] **Step 7: Commit**

```bash
git add recall/eval/locomo.py tests/test_locomo_corpus_postcondition.py
git commit -m "feat(eval): assert the corpus AFTER indexing, not only before

The shipped pre-check catches a sequential re-run. The 2026-07-27 failure
was two concurrent launchers: the second passed the pre-check on an empty
table and wrote during indexing. IndexStats.chunks vs the tenant row count
closes that window exactly."
```

---

### Task 2: Carry the corpus row count into the LOCOMO result artifact

The verification from Task 1 exists at runtime and dies with the process. `results/locomo_rerank/*.json` prove the problem: #103 genuinely verified 5,882 rows via `scripts/run_locomo_arms.sh`, and its artifacts record embedder, `k`, `candidate_k`, `reranker`, `conversations`, `elapsed_s` — and no row count. The discipline did not survive into the file.

A shared helper, because three runners need the identical block and a copy-pasted one drifts.

**Files:**
- Create: `recall/eval/provenance.py`
- Create: `tests/test_eval_provenance.py`
- Modify: `recall/eval/locomo.py:456-472` (the `run()` report dict)

**Interfaces:**
- Consumes: `run_conversation(...)["corpus_rows"]` from Task 1.
- Produces: `recall.eval.provenance.provenance_block(corpus_rows: int, table: str, tenants: list[str]) -> dict[str, Any]` returning keys `corpus_rows`, `table`, `tenants`, `git_sha`. Tasks 3 consume this exact signature.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_provenance.py`:

```python
"""The block every result artifact embeds so it can be checked later.

No result JSON in results/ records the corpus it was measured against. That makes the
2026-07-27 doubled-corpus failure undetectable retroactively on every published number,
including postfix_pool20.json and postfix_abstention.json.

Pure functions only — this must be testable without a database, because it is the part that
has to work on every runner.
"""
from __future__ import annotations

from recall.eval.provenance import provenance_block


def test_block_carries_the_row_count_and_where_it_came_from():
    b = provenance_block(corpus_rows=5882, table="locomo_chunks", tenants=["locomo-conv-26"])
    assert b["corpus_rows"] == 5882
    assert b["table"] == "locomo_chunks"
    assert b["tenants"] == ["locomo-conv-26"]


def test_git_sha_is_present_so_a_result_names_the_tree_that_made_it():
    """Absent a repo it must degrade to None, never to a wrong or invented sha."""
    b = provenance_block(corpus_rows=1, table="t", tenants=[])
    assert "git_sha" in b
    assert b["git_sha"] is None or isinstance(b["git_sha"], str)


def test_tenants_are_sorted_so_two_runs_of_one_config_produce_equal_blocks():
    """Dict ordering must not make a diff of two identical runs look like a change."""
    a = provenance_block(corpus_rows=2, table="t", tenants=["b", "a"])
    b = provenance_block(corpus_rows=2, table="t", tenants=["a", "b"])
    assert a["tenants"] == b["tenants"] == ["a", "b"]


def test_a_zero_row_corpus_is_representable_not_swallowed():
    """0 is a real, alarming value. It must not be dropped as falsy."""
    b = provenance_block(corpus_rows=0, table="t", tenants=[])
    assert b["corpus_rows"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_eval_provenance.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'recall.eval.provenance'`.

- [ ] **Step 3: Implement**

Create `recall/eval/provenance.py`:

```python
"""What a result artifact must say about the corpus it measured.

No result JSON in this repo records its corpus row count. On 2026-07-27 two concurrent runs
doubled the LOCOMO corpus (11,764 rows against a correct 5,882); every depth came in ~0.05 low
and nothing errored. #103 later verified 5,882 rows in scripts/run_locomo_arms.sh — and the
number went to the runner's stdout, so its own artifacts cannot show it.

A verification that does not reach the artifact protects only the session that ran it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git_sha() -> str | None:
    """Short SHA of the tree that produced this result, or None outside a repo.

    Degrades to None rather than raising or inventing: a result file from a tarball is still a
    result, and a wrong sha is worse than an absent one.

    Anchored to this file's own directory via `cwd`, not the caller's: without it, `git`
    inherits the calling process's working directory, and a run launched from inside some OTHER
    git repo would silently report THAT repo's HEAD — a wrong-but-plausible sha, which is
    exactly the "worse than absent" case above.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def provenance_block(corpus_rows: int, table: str, tenants: list[str]) -> dict[str, Any]:
    """The block every eval result embeds so a later reader can check it.

    `corpus_rows` is the summed tenant-scoped count actually measured, not a configured or
    expected value — an expectation copied into a result proves nothing about the run.
    """
    return {
        "corpus_rows": corpus_rows,
        "table": table,
        "tenants": sorted(tenants),
        "git_sha": _git_sha(),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_eval_provenance.py -v
```

Expected: PASS, 4 passed.

- [ ] **Step 5: Wire it into the LOCOMO report**

In `recall/eval/locomo.py`, add to the imports near the other `recall.eval` imports:

```python
from recall.eval.provenance import provenance_block
```

In `run()`, initialise a `tenants` accumulator alongside `per_conversation`:

```python
    per_conversation: list[dict[str, Any]] = []
    # Captured once per conversation, from the same f-string that opens its PgVectorStore below —
    # not re-derived from `conversations` after the loop. Two independent copies of one formula
    # drift the moment the naming scheme changes in only one of them, and the artifact would
    # silently misreport what was actually measured.
    tenants: list[str] = []
```

The loop already opens each store with `tenant = f"locomo-{sample_id}"`; append it to the accumulator right there instead of re-deriving it later:

```python
            tenant = f"locomo-{sample_id}"
            tenants.append(tenant)
            corpus_dir = workspace / str(sample_id)
```

Immediately before the `return {` at line ~456, add:

```python
    # Summed from what each conversation actually measured (Task 1's post-condition), never
    # from a configured expectation: an expectation copied into a result proves nothing. Hard
    # subscript, not .get(): run_conversation always sets this key, and a silent 0 default would
    # understate the corpus the moment a per-conversation try/except is added and stops raising —
    # exactly the "plausible-but-wrong number" class this module exists to prevent. `tenants` is
    # not re-derived here; it was captured above, in the loop that computed it the first time.
    corpus_rows = sum(res["corpus_rows"] for res in per_conversation)
```

Then add to the returned dict, immediately after the `"benchmark": "LOCOMO",` line:

```python
        **provenance_block(corpus_rows, table, tenants),
```

- [ ] **Step 6: Verify the field reaches a real artifact**

A one-conversation run is enough to prove the wiring; the full run is Phase 2's business.

```bash
python -m recall.eval.locomo --data locomo10.json --embedder fastembed --k 5 --limit 1 --table provcheck_chunks --out /tmp/provcheck.json
```

Then:

```bash
python -c "import json; d=json.load(open('/tmp/provcheck.json')); print({k: d[k] for k in ('corpus_rows','table','tenants','git_sha')})"
```

Expected: `corpus_rows` is a positive integer, `table` is `provcheck_chunks`, `tenants` has one entry, `git_sha` is a short hex string.

If `corpus_rows` is `0`, stop: the sum is reading a key `run_conversation` does not set, and every artifact would carry a confident zero.

- [ ] **Step 7: Commit**

```bash
git add recall/eval/provenance.py tests/test_eval_provenance.py recall/eval/locomo.py
git commit -m "feat(eval): result artifacts record the corpus they measured

#103 verified 5,882 rows and the number reached only the runner's stdout,
so rerank_modern.json cannot show it. The count now travels in the JSON."
```

---

### Task 3: The same block in the two abstention runners

`locomo_abstention.py` and `locomo_entailment_sweep.py` produce the artifacts this whole cycle is about — §9b and §9c. Both build a report dict and write it (`locomo_abstention.py:183` / `:258`, `locomo_entailment_sweep.py:233` / `:296`).

Both index through their own path rather than through `run()`, so they need the block wired at their own report site.

**Files:**
- Modify: `recall/eval/locomo_abstention.py` (imports; report dict at ~183)
- Modify: `recall/eval/locomo_entailment_sweep.py` (imports; report dict at ~233)
- Modify: `tests/test_eval_provenance.py` (add the contract test below)

**Interfaces:**
- Consumes: `provenance_block(corpus_rows, table, tenants)` from Task 2 — same signature, no variant.
- Produces: nothing later tasks import. Phase 1's inventory reads the resulting JSONs by hand.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_provenance.py`:

```python
def test_every_locomo_runner_embeds_the_provenance_block():
    """A runner that writes a result JSON must embed the block.

    Checked by import rather than by running the benchmark: each run costs tens of minutes, so
    a test that ran one would never be run. This catches the realistic regression — a new
    runner, or a refactor that drops the call — which is exactly how the count failed to reach
    #103's artifacts in the first place.
    """
    import inspect

    from recall.eval import locomo, locomo_abstention, locomo_entailment_sweep

    for mod in (locomo, locomo_abstention, locomo_entailment_sweep):
        src = inspect.getsource(mod)
        assert "provenance_block(" in src, (
            f"{mod.__name__} writes a result artifact but does not embed provenance_block — "
            f"its output could not be told apart from a doubled-corpus run"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_eval_provenance.py::test_every_locomo_runner_embeds_the_provenance_block -v
```

Expected: FAIL, naming `recall.eval.locomo_abstention`.

- [ ] **Step 3: Wire `locomo_abstention.py`**

⚠️ **This runner does not index.** `run()` opens `PgVectorStore(..., table="locomo_chunks")` and scores against whatever is already in it — there is no `Indexer` call, so Task 1's post-condition does not apply and must not be pasted in here.

That makes the block *more* load-bearing, not less: for a reader-runner the recorded count is the **only** way to tell that it scored against a doubled corpus. `postfix_abstention.json` — the artifact §9b rests on — was produced by this runner, and nothing in it says whether the table held a clean 5,882 rows or twice that.

Add near the other `recall.eval` imports:

```python
from recall.eval.provenance import provenance_block
```

In `run()`, immediately after `started = time.time()`, add:

```python
    # Counted per tenant as each store is opened. This runner READS an index it did not build,
    # so the count is the only evidence in the artifact of what it actually scored against.
    corpus_rows = 0
    tenants: list[str] = []
```

Inside the `with PgVectorStore(...) as store:` block, immediately after the `retriever = HybridRetriever(store, embedder)` line, add:

```python
            corpus_rows += store.count()
            tenants.append(tenant)
```

Then in the returned dict, immediately after the `"benchmark": "LOCOMO — abstention ablation",` line, add:

```python
        **provenance_block(corpus_rows, "locomo_chunks", tenants),
```

The table name is the literal already hardcoded at the `PgVectorStore(...)` call in this module. If that call is ever parameterised, this must read the same variable — a provenance block naming a different table than the one read is worse than none.

- [ ] **Step 4: Wire `locomo_entailment_sweep.py`**

Same shape, same import. Confirm first whether this runner indexes or reads:

```bash
grep -n "Indexer\|PgVectorStore(\|table=" recall/eval/locomo_entailment_sweep.py | head
```

- **If it reads** (no `Indexer`): apply Step 3's pattern exactly — accumulator before the loop, `corpus_rows += store.count()` and `tenants.append(tenant)` inside the `with` block, `**provenance_block(corpus_rows, <the table literal>, tenants),` after the `"benchmark":` line.
- **If it indexes**: it also needs Task 1's post-condition. Add it there too, then the block.

- [ ] **Step 5: Run the test to verify it passes**

```bash
python -m pytest tests/test_eval_provenance.py -v
```

Expected: PASS, 5 passed.

- [ ] **Step 6: Confirm the runners still start**

Both are long-running, so check they parse and reach their first work rather than running them to completion:

```bash
python -m recall.eval.locomo_abstention --help && python -m recall.eval.locomo_entailment_sweep --help
```

Expected: both print usage and exit 0.

- [ ] **Step 7: Commit**

```bash
git add recall/eval/locomo_abstention.py recall/eval/locomo_entailment_sweep.py tests/test_eval_provenance.py
git commit -m "feat(eval): provenance block in the two abstention runners"
```

---

### Task 4: Phase 1 — the inventory

A table stating, for every abstention claim, whether it is current, stale, or unfalsifiable. Written from artifacts on disk, not from `FINDINGS.md`'s own account of itself.

**Files:**
- Create: `results/INSTRUMENT_STATUS.md`

**Interfaces:**
- Consumes: the artifacts under `results/locomo/` and `results/locomo_rerank/`.
- Produces: the status table Task 5–7 update as each measurement lands.

- [ ] **Step 1: Read the artifacts rather than the write-up**

```bash
ls -la results/locomo/ results/locomo_rerank/
```

```bash
python -c "
import json, glob
for f in sorted(glob.glob('results/locomo*/*.json')):
    d = json.load(open(f))
    print(f, '->', {k: d.get(k) for k in ('benchmark','embedder','k','candidate_k','reranker','conversations','corpus_rows')})
"
```

Record which files exist, and which report `corpus_rows` as `None` — every artifact predating Task 2 will.

- [ ] **Step 2: Write the inventory**

Create `results/INSTRUMENT_STATUS.md`:

```markdown
# Instrument status — which abstention claims are checkable

Written 2026-07-28 against `origin/master` @ 9eb3bc1. Read from the artifacts in `results/`,
not from `FINDINGS.md`'s account of itself.

**current** — measured on the shipped pipeline, artifact retained.
**stale** — measured on a superseded configuration.
**unfalsifiable** — no artifact retained; the claim cannot be checked at any cost short of a re-run.

| claim | status | artifact | notes |
|---|---|---|---|
| §9b LOCOMO abstention, 4 modes | current | `locomo/postfix_abstention.json` | post-#81/#84. `locomo_abstention.py:168` passes `calibration=cal` explicitly, so #101's auto-load bug never reached it |
| §9b abstention with rerank on | **unmeasured** | — | #103 measured the default mode only (0.00, unchanged). The calibrated and judge modes have never been crossed with a reranker |
| §9c entailment ROC sweep | stale | `locomo/postfix_entailment_sweep.log` | the re-run died after 9 conversations; no JSON was written and nothing noticed |
| §10 LongMemEval, all rows | **unfalsifiable** | — | pre-#81/#84; indexes and output discarded. 6h39m to rebuild the merged index alone |
| §7 private-corpus abstention | current, not independently checkable | — | corpus is private |
| §8 PEP abstention | current | — | public corpus and questions; cheap to re-establish |
| every row above | **no row count** | — | no pre-Task-2 artifact records the corpus it measured |

## What this gates

No combined signal, entity-mismatch feature or abstention-policy change is fit against a row
marked **stale** or **unfalsifiable** until that row is re-measured or explicitly demoted.
```

- [ ] **Step 3: Commit**

```bash
git add results/INSTRUMENT_STATUS.md
git commit -m "docs(eval): inventory which abstention claims are checkable"
```

---

### Task 5: Phase 2 — re-measure §9c, and find out why the last attempt died

This repo's own `.gitignore` calls `results/locomo/*.log` transient and says the JSON beside it is the artifact — and for §9c, no JSON was ever written. The only trace, `postfix_entailment_sweep.log`, stops after nine conversations with no summary, but that file is untracked and gitignored by design, so it isn't part of this repository. `FINDINGS.md` records §9c as "not re-measured", which is true and undersells it: it was attempted, it died, and left nothing this repo can check.

**The first question is not the ROC.** A harness that can stop nine tenths of the way through and leave a plausible-looking log is a defect independent of its output.

**Files:**
- Create: `results/locomo/PREDICTION-9c-rerun.md`
- Modify: `results/INSTRUMENT_STATUS.md` (the §9c row)

**Interfaces:**
- Consumes: `python -m recall.eval.locomo_entailment_sweep`, now emitting a provenance block (Task 3).
- Produces: `results/locomo/postfix_entailment_sweep.json` — the artifact §9c currently lacks.

- [ ] **Step 1: Diagnose the previous death before re-running**

```bash
tail -5 results/locomo/postfix_entailment_sweep.log
```

LOCOMO has ten conversations; the log lists nine (`conv-26, 30, 41, 42, 43, 44, 47, 48, 49`). Determine which is missing and whether the runner exits non-zero when a conversation raises:

```bash
grep -n "except\|continue\|raise\|sys.exit\|return 1" recall/eval/locomo_entailment_sweep.py | head -20
```

If a per-conversation exception is caught and the loop continues, that is the defect: a partial sweep is reported as a sweep. Fix it to fail loudly — a partial result must not be indistinguishable from a complete one — and commit that fix **before** the re-run, with its own test.

If instead the process was killed externally (OOM, disconnect), record that in the prediction file and proceed; a killed process leaving no JSON is the correct behaviour.

- [ ] **Step 2: Write the prediction BEFORE running**

Per `docs/RESEARCH_PROTOCOL.md`, committed before the measurement starts.

Create `results/locomo/PREDICTION-9c-rerun.md`:

```markdown
# Prediction — §9c entailment ROC sweep, re-measured post-#81/#84

Written and committed before the run. Scored afterwards for whether it was right, and right
for the right reason.

## What changed since the recorded sweep

#81 (sparse leg no longer ANDs every term) and #84 (`hnsw.ef_search` widening). The recorded
sweep measured an effectively dense-only configuration.

## Prediction

Best separation (adversarial-abstain − answerable false-abstain) stays within **±0.03** of the
recorded 0.240 for `qnli-electra-base` and 0.197 for `qnli-distilroberta`.

## Reasoning

Separation is a property of the JUDGE's ability to tell "Caroline realized X" from "Melanie
realized X", not of which candidates reach it. A better candidate pool hands the judge a
better-retrieved on-topic-but-wrong turn, which scores *higher*, not lower. §9b measured
exactly this: post-fix discrimination came in at 0.154 against a pre-fix 0.157.

## Decision rule, fixed in advance

- Within ±0.03 → §9c's conclusion stands; update the configuration note and mark the row current.
- Outside ±0.03 in either direction → the retrieval fix moved an answerability judgement, which
  contradicts §9b. Do not publish either number until the disagreement is resolved.

## The invariant this run asserts in code

`corpus_rows` in the emitted JSON must equal the count in a clean single run. Recorded here
before the run so it cannot be adjusted to match the result.
```

```bash
git add results/locomo/PREDICTION-9c-rerun.md
git commit -m "docs(eval): prediction for the 9c re-run, before measuring"
```

- [ ] **Step 3: Run it, alone, and poll every terminal state**

Nothing else may touch the LOCOMO tables while this runs.

```bash
python -m recall.eval.locomo_entailment_sweep --data locomo10.json --answerable-sample 40 --out results/locomo/postfix_entailment_sweep.json 2>&1 | tee results/locomo/postfix_entailment_sweep_rerun.log
```

When it returns, check **all three** terminal conditions and say which fired — not just that the command came back:

```bash
echo "exit=$?"; ls -la results/locomo/postfix_entailment_sweep.json; tail -3 results/locomo/postfix_entailment_sweep_rerun.log
```

Required: exit 0, the JSON exists, and the log ends with a summary rather than mid-conversation. Any one missing means the run did not complete — do not read a number from it.

- [ ] **Step 4: Verify the artifact before reading any metric**

```bash
python -c "
import json
d = json.load(open('results/locomo/postfix_entailment_sweep.json'))
print('corpus_rows:', d.get('corpus_rows'), 'git_sha:', d.get('git_sha'), 'conversations:', d.get('conversations'))
assert d.get('conversations') == 10, 'not all ten conversations are in this result'
assert d.get('corpus_rows'), 'no corpus row count — this artifact cannot be checked'
"
```

Expected: no `AssertionError`. If `conversations` is 9, Step 1's defect is not fixed — go back.

- [ ] **Step 5: Score the prediction**

Append to `results/locomo/PREDICTION-9c-rerun.md` under a `## Result` heading: the measured separations, whether the prediction held under the decision rule, and — if it held — whether the *reasoning* was right or it came out right by luck.

- [ ] **Step 6: Update the inventory and commit**

Change the §9c row in `results/INSTRUMENT_STATUS.md` to `current` with the new artifact path.

```bash
git add results/locomo/postfix_entailment_sweep.json results/locomo/postfix_entailment_sweep_rerun.log results/locomo/PREDICTION-9c-rerun.md results/INSTRUMENT_STATUS.md
git commit -m "eval: re-measure the 9c entailment sweep post-#81/#84, with its artifact"
```

---

### Task 6: Phase 2 — re-establish the PEP abstention arm

The inventory needs at least one abstention measurement that is current, public and reproducible end to end. The PEP corpus is the only candidate: corpus, questions and ground truth are all public, and `FINDINGS.md` §8 gives the exact command.

**Files:**
- Create: `results/PREDICTION-peps-abstention.md`
- Create: `results/peps_abstention_2026-07-28.json`
- Modify: `results/INSTRUMENT_STATUS.md` (the §8 row)

**Interfaces:**
- Consumes: `python -m recall.eval.labelled` with the shipped `recall/eval/peps_questions.json`.
- Produces: a retained artifact for the §8 abstention row.

- [ ] **Step 1: Write the prediction and commit it**

Create `results/PREDICTION-peps-abstention.md`:

```markdown
# Prediction — PEP abstention arm, re-measured 2026-07-28

## Prediction

Abstention accuracy stays at **1.00** (11/11) and false-abstain stays within **[0.00, 0.10]**.

## Reasoning

§10c located the boundary: abstention works where the unanswerable queries are genuinely
off-topic and the two cosine distributions are disjoint. The PEP question set is built that
way, and #81/#84 improve retrieval rather than change answerability. If this arm has moved,
the boundary in §10c is wrong and that matters more than the number.

## Decision rule, fixed in advance

- Accuracy 1.00 and false-abstain ≤ 0.10 → §8's abstention row is current; record and move on.
- Anything else → §10c's "far gaps yes" boundary does not hold on a corpus it was drawn from.
  Stop and report; do not fold the number into FINDINGS as a routine update.
```

```bash
git add results/PREDICTION-peps-abstention.md
git commit -m "docs(eval): prediction for the PEP abstention re-measurement"
```

- [ ] **Step 2: Fetch the public corpus**

```bash
git clone --depth 1 https://github.com/python/peps /tmp/peps
```

- [ ] **Step 3: Run it**

```bash
python -m recall.eval.labelled --corpus /tmp/peps/peps --questions recall/eval/peps_questions.json --glob '**/*.rst' > results/peps_abstention_2026-07-28.json
```

- [ ] **Step 4: Check every terminal state**

```bash
echo "exit=$?"; ls -la results/peps_abstention_2026-07-28.json; python -c "
import json; d=json.load(open('results/peps_abstention_2026-07-28.json'))
print({k: d.get(k) for k in ('abstention_accuracy','false_abstain','n')})
"
```

`labelled.py` prints its report to stdout, so a non-zero exit still produces a file — check the exit code and the parse, not the file's existence.

**This artifact will carry no `corpus_rows`,** because `labelled.py` is not wired in this plan (see the self-review note). That is a known gap, not an oversight: record it in the §8 row of `INSTRUMENT_STATUS.md` as *"current, row count not recorded — `labelled.py` wiring deferred"*. An artifact whose limitation is written down is checkable; one whose limitation a reader has to infer is not.

- [ ] **Step 5: Score the prediction and update the inventory**

Append a `## Result` section to `results/PREDICTION-peps-abstention.md` with the measured figures and the verdict under the decision rule. Update the §8 row of `results/INSTRUMENT_STATUS.md`.

- [ ] **Step 6: Commit**

```bash
git add results/peps_abstention_2026-07-28.json results/PREDICTION-peps-abstention.md results/INSTRUMENT_STATUS.md
git commit -m "eval: re-establish the PEP abstention arm on the current pipeline"
```

---

### Task 7: Phase 2 — put §10 to an explicit decision

§10 (LongMemEval) is the strongest negative result in `FINDINGS.md` and is currently unfalsifiable: pre-#81/#84, no artifact retained, 6h39m to rebuild the merged index alone.

**This task does not re-run it.** It writes the decision down and executes whichever branch is chosen, so the outcome is on the record either way.

**Files:**
- Modify: `results/INSTRUMENT_STATUS.md` (the §10 row)
- Modify: `results/FINDINGS.md` §10 (only if DEMOTE is chosen)

**Interfaces:**
- Consumes: the inventory from Task 4.
- Produces: a §10 row that is either `current` or explicitly `unfalsifiable, demoted`.

- [ ] **Step 1: Write the decision, with its cost, and stop for the human**

Append to `results/INSTRUMENT_STATUS.md`:

```markdown
## §10 decision — 2026-07-28

§10's conclusion rests on signal **separability** (AUC ≤ 0.753 against a ~0.90 bar), and
FINDINGS itself argues a better candidate pool does not turn a relevance signal into an
answerability signal. So a re-run is expected to reproduce the conclusion at a cost of 6h39m
for the merged index alone.

- **RE-RUN** — restores a checkable artifact for the document's strongest negative result.
- **DEMOTE** — mark it "measured on a superseded configuration, artifact not retained" and stop
  citing it as load-bearing anywhere in FINDINGS or the README.

Demoting is not a retreat. An unfalsifiable claim cited as evidence is worse than one labelled
as unfalsifiable.

**Decision:** <RE-RUN | DEMOTE>
**Decided by:** <name>, <date>
```

**This step ends the task pending a human decision.** Do not pick a branch autonomously — the
choice trades 6h39m of a metered shared box against the standing of a published claim, and both
are the maintainer's to spend.

- [ ] **Step 2a: If DEMOTE — label the claim where it is made**

In `results/FINDINGS.md` §10, add immediately under the heading:

```markdown
> **⚠️ UNFALSIFIABLE — 2026-07-28.** These runs predate #81/#84 and **no result artifact was
> retained**, so nothing in this section can be checked against this repository. The abstention
> conclusion (§10b) rests on signal separability, which a retrieval fix does not change — but it
> is recorded here as unverifiable rather than merely caveated, and it must not be cited as
> load-bearing evidence until it is re-measured. See `results/INSTRUMENT_STATUS.md`.
```

Then grep for citations of §10 elsewhere and add the same qualifier at each:

```bash
grep -rn "§10\|LongMemEval" README.md results/FINDINGS.md docs/*.md | grep -v INSTRUMENT_STATUS
```

- [ ] **Step 2b: If RE-RUN — schedule it as its own work**

A 6h39m index is not a step inside another task. Record the decision in
`results/INSTRUMENT_STATUS.md`, then stop: the re-run gets its own plan, with a prediction file
and the same three-terminal-state polling as Task 5.

- [ ] **Step 3: Commit**

```bash
git add results/INSTRUMENT_STATUS.md results/FINDINGS.md
git commit -m "docs(eval): record the §10 decision — <RE-RUN|DEMOTE>"
```

---

## Where this plan stops

Phase 3 — the written verdict stating which abstention claims are checkable, and the gate it
imposes on the next cycle — is **deliberately out of scope** and gets its own plan.
`results/INSTRUMENT_STATUS.md` is the artifact Phase 3 will read.

Nothing here fits a new signal, changes a threshold, or alters abstention policy.

## Self-review notes

**Spec coverage.** Phase 0 → Tasks 1–3. Phase 1 → Task 4. Phase 2 → Tasks 5–7. Phase 3 → out of
scope by instruction.

**Deliberately not in this plan.** `recall/eval/labelled.py` and `recall/eval/longmemeval_perq.py`
were listed in the spec's Phase 0 scope and are **not** wired here. `labelled.py` builds its own
index from a corpus directory, so it needs Task 1's post-condition *and* Task 2's block — two
changes, and Task 6 runs it before either lands. `longmemeval_perq.py` scores against a
pre-existing `--master` table whose only consumer (§10) is under decision in Task 7. Both are
named rather than dropped; wiring them is the first item of the follow-up plan, and Task 6's PEP
artifact will lack a row count until then. That gap is recorded in `INSTRUMENT_STATUS.md` rather
than left for a reader to notice.

**A distinction the spec did not draw.** Runners split into *builders* (`locomo.py`,
`labelled.py`) and *readers* (`locomo_abstention.py`, `longmemeval_perq.py`, probably
`locomo_entailment_sweep.py`). Builders get the post-condition; readers cannot have it and get
the block alone. Conflating them would put an assert in a runner with no `IndexStats` to compare
against, which fails at import review rather than silently — but wastes a task.

**Known risk.** Tasks 5 and 6 are measurements on a metered shared box, and another session is
active in the sibling clone. Run them one at a time, take the LOCOMO tables exclusively, and check
all three terminal states before reading any number.
