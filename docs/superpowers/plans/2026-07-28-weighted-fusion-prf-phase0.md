# Phase 0 Diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether leg disagreement (`conf(sparse) > conf(dense)`) selects for retrieval failures, at what rate it fires, and where the gold chunk sits when it does — so the weighted-fusion and PRF phases are built on a measured premise or killed.

**Architecture:** An additive observer. `PgVectorStore.query_sparse` gains an opt-in `ts_rank` return (it currently computes and discards it); `HybridRetriever` gains an optional `probe` callback that fires once per search with both legs' native candidates; the LOCOMO harness threads that probe through the way it already threads `reranker`. All diagnostic logic lives in `recall/eval/`. No fusion change, no second retrieval pass.

**Tech Stack:** Python 3.11+, PostgreSQL 16 + pgvector, psycopg 3, pytest, `uv`, fastembed (`bge-small`).

## Global Constraints

- **No LLM anywhere in the ingest or retrieval path.** Everything here is arithmetic over scores the pipeline already computes.
- **`ScoredChunk.score` keeps its cosine contract.** The trust layer's thresholds and calibrated confidence read it as a cosine. `ts_rank` is returned *alongside*, never in `score`.
- **The default code path stays byte-identical.** `with_rank` defaults `False`; `probe` defaults `None`. A duck-typed store double in `tests/test_advice_injection.py:123` defines `query_sparse(self, query, k, source=None, vec=None)` with no `with_rank`, so the retriever must pass that keyword **only when a probe is attached**.
- **Test DSN:** `RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall"`. Never `:5432` — that container belongs to another worktree and this suite DROPs tables.
- **Run tests with:** `RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest ...`
- **Baseline before starting:** 871 passed, 5 skipped.
- **Predictions are already committed** (`ef68bb1`). Do not edit them after seeing any output.

---

### Task 1: Expose `ts_rank` from `query_sparse`

The blocker. When `vec` is passed — which `HybridRetriever` always does — the outer SELECT overwrites `score` with the dense cosine, so the lexical leg's own score never reaches a caller. `conf(sparse)` computed from today's return would compare dense-cosine decisiveness against dense-cosine decisiveness: no error, a plausible number, a trigger measuring nothing.

**Files:**
- Modify: `recall/store.py` (`query_sparse`, from line 1173)
- Test: `tests/test_store_sparse_rank.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `PgVectorStore.query_sparse(text, k, source=None, vec=None, *, with_rank: bool = False)` returning `list[ScoredChunk]` when `with_rank=False` and `tuple[list[ScoredChunk], list[float]]` when `True`. The float list is `ts_rank` per hit, in the same order as the hits.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store_sparse_rank.py`:

```python
from __future__ import annotations

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from tests.conftest import requires_db


@requires_db
def test_query_sparse_with_rank_returns_ts_rank_not_cosine(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one about caching", encoding="utf-8")
    (tmp_path / "b.md").write_text("indexing decision two", encoding="utf-8")
    (tmp_path / "c.md").write_text("caching appears here once", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    qvec = emb.embed(["caching"])[0]

    hits, ranks = store.query_sparse("caching", k=5, vec=qvec, with_rank=True)

    assert len(ranks) == len(hits)
    assert ranks == sorted(ranks, reverse=True)          # ts_rank order, descending
    assert all(r >= 0.0 for r in ranks)
    assert any(r > 0.0 for r in ranks)                   # a real lexical match scored
    # the whole point: ts_rank is a DIFFERENT quantity from the cosine in `score`
    assert [h.score for h in hits] != ranks


@requires_db
def test_query_sparse_default_return_is_unchanged(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    qvec = emb.embed(["caching"])[0]

    plain = store.query_sparse("caching", k=5, vec=qvec)
    hits, _ = store.query_sparse("caching", k=5, vec=qvec, with_rank=True)

    assert isinstance(plain, list)                        # not a tuple
    assert [h.chunk.id for h in plain] == [h.chunk.id for h in hits]
    assert [h.score for h in plain] == [h.score for h in hits]


@requires_db
def test_query_sparse_with_rank_without_vec_returns_score_as_rank(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one about caching", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)

    hits, ranks = store.query_sparse("caching", k=5, with_rank=True)

    # with no vec, `score` IS ts_rank already — the two must agree exactly
    assert [h.score for h in hits] == ranks
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest tests/test_store_sparse_rank.py -v
```

Expected: FAIL — `TypeError: query_sparse() got an unexpected keyword argument 'with_rank'`.

If `requires_db` is not importable from `tests.conftest`, find its actual definition with `grep -rn "requires_db" tests/conftest.py` and import it from there; every DB test in this suite already uses it.

- [ ] **Step 3: Implement**

In `recall/store.py`, change the signature:

```python
    def query_sparse(
        self,
        text: str,
        k: int,
        source: str | None = None,
        vec: list[float] | None = None,
        *,
        with_rank: bool = False,
    ) -> list[ScoredChunk] | tuple[list[ScoredChunk], list[float]]:
```

Add to the existing docstring, after the first paragraph:

```
        `with_rank=True` additionally returns each hit's `ts_rank` — the lexical leg's OWN
        ranking score, which the `vec` branch otherwise computes in the subquery and discards
        when the outer SELECT replaces `score` with the cosine. `score` is untouched either way:
        the trust layer reads it as a cosine and must keep doing so. Opt-in because callers
        (including duck-typed store doubles in the test suite) implement the 4-argument form.
```

In the `vec is not None` branch, make the rank column conditional and add it to the outer SELECT:

```python
        rank_col = ", rank" if with_rank else ""
        if vec is not None:
            sql = f"""
                {tsquery_cte}
                SELECT id, source, text, metadata, indexed_at,
                       1 - (embedding <=> %(vec)s) AS score{rank_col}
                FROM (
                    SELECT c.id, c.source, c.text, c.metadata, c.indexed_at, c.embedding,
                           ts_rank(c.tsv, q.tsq) AS rank
                    FROM {t} c, q
                    WHERE c.tenant_id = %(tenant)s
                      AND c.tsv @@ q.tsq
                    {where}
                    ORDER BY rank DESC
                    LIMIT %(k)s
                ) top_k
                ORDER BY rank DESC
            """
```

Leave the `else` branch's SQL exactly as it is — there `score` already *is* `ts_rank`.

Replace the final `return self._rows_to_hits(rows)` with:

```python
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        if not with_rank:
            return self._rows_to_hits(rows)
        if vec is not None:
            # `_rows_to_hits` unpacks exactly 6 columns; the rank rides in a 7th.
            return self._rows_to_hits([r[:6] for r in rows]), [float(r[6]) for r in rows]
        return self._rows_to_hits(rows), [float(r[5]) for r in rows]
```

- [ ] **Step 4: Run the new tests, then the full suite**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest tests/test_store_sparse_rank.py -v
```
Expected: 3 passed.

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest -q
```
Expected: 874 passed, 5 skipped. If any previously-passing test fails, the default path was not preserved — fix that before continuing.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/store.py tests/test_store_sparse_rank.py && git commit -m "feat(store): opt-in ts_rank return from query_sparse

The vec branch computes ts_rank in a subquery, orders by it, then the
outer SELECT overwrites score with the dense cosine — so the lexical
leg's own ranking score never reaches a caller. Opt-in with_rank
returns it alongside, leaving score's cosine contract and the default
4-argument call signature untouched.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `leg_confidence()` and its affine-invariance property

The quantity the whole diagnostic rests on. It must be immune to the scale difference between cosine (bounded) and `ts_rank` (unbounded), or the trigger is comparing incomparable numbers.

**Files:**
- Create: `recall/eval/legconf.py`
- Test: `tests/test_leg_confidence.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `recall.eval.legconf.leg_confidence(scores: Sequence[float]) -> float` — the z-score of the maximum within `scores`; `0.0` for empty, single-element, or zero-variance input. Always `>= 0.0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_leg_confidence.py`:

```python
from __future__ import annotations

import pytest

from recall.eval.legconf import leg_confidence


def test_leg_confidence_is_affine_invariant():
    """The load-bearing property: cosine and ts_rank live on different scales, and the
    trigger compares them directly. If conf() were not affine-invariant, that comparison
    would measure the units rather than the decisiveness."""
    base = [0.90, 0.50, 0.40, 0.35, 0.20]
    expected = leg_confidence(base)
    for a, b in [(2.0, 0.0), (0.5, 0.0), (1.0, 10.0), (3.0, -7.5), (1000.0, 42.0)]:
        scaled = [a * s + b for s in base]
        assert leg_confidence(scaled) == pytest.approx(expected, rel=1e-9)


def test_leg_confidence_is_higher_for_a_peaked_leg():
    peaked = [0.9, 0.2, 0.2, 0.2, 0.2]
    flat = [0.5, 0.5, 0.5, 0.5, 0.4]
    assert leg_confidence(peaked) > leg_confidence(flat)


def test_leg_confidence_is_zero_when_flat():
    assert leg_confidence([0.4, 0.4, 0.4, 0.4]) == 0.0


@pytest.mark.parametrize("scores", [[], [0.7]])
def test_leg_confidence_is_zero_without_spread(scores):
    """Empty leg or a single candidate: no spread exists, so there is no decisiveness to
    report. Zero means 'no information', and because the trigger uses a STRICT >, a leg in
    this state can never fire it."""
    assert leg_confidence(scores) == 0.0


def test_leg_confidence_is_never_negative():
    # the max is always >= the mean, so the z-score of the max cannot be negative
    for scores in ([0.1, 0.9], [-5.0, -1.0, -3.0], [0.5, 0.5, 0.51]):
        assert leg_confidence(scores) >= 0.0
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_leg_confidence.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'recall.eval.legconf'`.

- [ ] **Step 3: Implement**

Create `recall/eval/legconf.py`:

```python
"""Per-query, per-leg decisiveness — the quantity the Phase 0 diagnostic measures.

A retrieval leg either found a clear winner or handed back a flat, undifferentiated list.
`leg_confidence` reports which, as the z-score of the leg's top candidate within that leg's
OWN candidate scores.

Why a z-score and not a normalized max: the dense leg scores in cosine (bounded, ~[0, 1]) and
the sparse leg scores in `ts_rank` (unbounded, corpus-dependent). Any statistic that survives
being compared across those two must be invariant to an affine change of units, and a z-score
is. That invariance is asserted in `tests/test_leg_confidence.py`, not assumed here.

This lives under `recall/eval/` deliberately. Nothing in the serving path consumes it yet —
Phase 1 would, if the diagnostic clears its gates. Shipping it into `recall/` before then would
be dead code in the installed package.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def leg_confidence(scores: Sequence[float]) -> float:
    """z-score of the top candidate within `scores`. 0.0 when there is no spread to measure.

    Returns 0.0 for an empty leg, a single candidate, or a perfectly flat leg — all three mean
    "this leg expressed no preference". Never negative: the maximum of a sample is always at
    least its mean.
    """
    n = len(scores)
    if n < 2:
        return 0.0
    mu = sum(scores) / n
    sd = math.sqrt(sum((s - mu) ** 2 for s in scores) / n)
    if sd == 0.0:
        return 0.0
    return (max(scores) - mu) / sd
```

- [ ] **Step 4: Run the tests**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_leg_confidence.py -v
```
Expected: 8 passed (the parametrized case counts twice).

- [ ] **Step 5: Commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/eval/legconf.py tests/test_leg_confidence.py && git commit -m "feat(eval): leg_confidence — affine-invariant per-leg decisiveness

z-score of a leg's top candidate within its own candidates. Affine
invariance is what makes cosine and ts_rank comparable without
normalizing incompatible scales; it is asserted as a property test
rather than assumed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: A `probe` observer on `HybridRetriever`

The diagnostic must observe the *real* pipeline. Reimplementing fusion in the eval harness would measure a copy — the exact apparatus-corruption failure this phase exists to avoid.

**Files:**
- Modify: `recall/retriever.py`
- Test: `tests/test_retriever_probe.py` (create)

**Interfaces:**
- Consumes: `query_sparse(..., with_rank=True)` from Task 1.
- Produces: `recall.retriever.LegProbe` (frozen dataclass, fields below) and a `probe: Callable[[LegProbe], None] | None = None` keyword on `HybridRetriever.__init__`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retriever_probe.py`:

```python
from __future__ import annotations

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.retriever import HybridRetriever, LegProbe
from tests.conftest import requires_db


def _index(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one about caching", encoding="utf-8")
    (tmp_path / "b.md").write_text("indexing decision two", encoding="utf-8")
    (tmp_path / "c.md").write_text("caching appears here once", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    return store, emb


@requires_db
def test_attaching_a_probe_does_not_change_the_result(tmp_path, make_store):
    """The apparatus guarantee in miniature: instrumentation that perturbs the retrieved set
    has broken the thing it was measuring."""
    store, emb = _index(tmp_path, make_store)
    plain = HybridRetriever(store, emb).search("caching", k=5)
    seen: list[LegProbe] = []
    probed = HybridRetriever(store, emb, probe=seen.append).search("caching", k=5)

    assert [h.chunk.id for h in plain.hits] == [h.chunk.id for h in probed.hits]
    assert [h.score for h in plain.hits] == [h.score for h in probed.hits]
    assert plain.gap_warning == probed.gap_warning
    assert len(seen) == 1


@requires_db
def test_probe_reports_sparse_ts_rank_not_cosine(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    seen: list[LegProbe] = []
    HybridRetriever(store, emb, probe=seen.append).search("caching", k=5)
    p = seen[0]

    assert p.query == "caching"
    assert len(p.sparse_ranks) == len(p.sparse)
    assert p.sparse_ranks == sorted(p.sparse_ranks, reverse=True)
    assert [h.score for h in p.sparse] != p.sparse_ranks   # cosine != ts_rank
    assert p.dense and p.fused


@requires_db
def test_probe_fires_once_per_search_with_empty_sparse_when_disabled(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    seen: list[LegProbe] = []
    HybridRetriever(store, emb, use_sparse=False, probe=seen.append).search("caching", k=5)

    assert len(seen) == 1
    assert seen[0].sparse == [] and seen[0].sparse_ranks == []
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest tests/test_retriever_probe.py -v
```
Expected: FAIL — `ImportError: cannot import name 'LegProbe' from 'recall.retriever'`.

- [ ] **Step 3: Implement**

In `recall/retriever.py`, add to the imports:

```python
from collections.abc import Callable
from dataclasses import dataclass
```

Add above `class HybridRetriever`:

```python
@dataclass(frozen=True)
class LegProbe:
    """One search's raw per-leg evidence, handed to an optional observer.

    Exists so a diagnostic can read the legs the REAL pipeline produced. Reconstructing them
    in an eval harness would measure a copy of the retriever rather than the retriever.

    `sparse_ranks` carries `ts_rank` — the sparse leg's own ranking score. It is NOT
    `[h.score for h in sparse]`: those are dense cosines, because `query_sparse` rescores its
    hits against the query vector so lexical-only hits stay comparable downstream.
    """

    query: str
    dense: list[ScoredChunk]        # dense candidates, best-first; score = cosine
    sparse: list[ScoredChunk]       # sparse candidates, best-first by ts_rank; score = cosine
    sparse_ranks: list[float]       # ts_rank per sparse hit, same order as `sparse`
    fused: list[ScoredChunk]        # post-fusion, pre-rerank, pre-truncation
```

Add the parameter to `__init__` (after `use_dense`), and store it:

```python
        use_dense: bool = True,
        probe: Callable[[LegProbe], None] | None = None,
    ) -> None:
```
```python
        self._probe = probe
```

Document it in the class docstring's `Tunables:` block:

```
      probe:         optional observer called once per search with the raw per-leg candidates.
                     Diagnostics only — it cannot influence the result, and the default (None)
                     leaves the query path byte-identical.
```

In `search()`, replace the `sparse = (...)` assignment with:

```python
        sparse_ranks: list[float] = []
        if self._use_sparse:
            if self._probe is not None:
                # `with_rank` is passed ONLY when probing: store doubles in the test suite (and
                # any third-party PgVectorStore-shaped object) implement the 4-argument form.
                sparse, sparse_ranks = self._store.query_sparse(
                    query, k=self._candidate_k, source=source, vec=qvec, with_rank=True
                )
            else:
                sparse = self._store.query_sparse(
                    query, k=self._candidate_k, source=source, vec=qvec
                )
        else:
            sparse = []
```

Then, immediately after the `hits = [...]` list comprehension that builds the fused list and **before** the reranker branch, add:

```python
        if self._probe is not None:
            self._probe(
                LegProbe(
                    query=query,
                    dense=list(dense),
                    sparse=list(sparse),
                    sparse_ranks=list(sparse_ranks),
                    fused=list(hits),
                )
            )
```

- [ ] **Step 4: Run the new tests, then the full suite**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest tests/test_retriever_probe.py -v
```
Expected: 3 passed.

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest -q
```
Expected: 877 passed, 5 skipped. `tests/test_advice_injection.py` in particular must still pass — its fake store has no `with_rank`, which is why the keyword is conditional.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/retriever.py tests/test_retriever_probe.py && git commit -m "feat(retriever): optional LegProbe observer for diagnostics

Lets a diagnostic read the legs the real pipeline produced instead of
reconstructing fusion in the eval harness, which would measure a copy.
Default None leaves the query path byte-identical; with_rank is passed
only when probing so duck-typed store doubles keep working.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Thread the probe through the LOCOMO harness

`run_conversation` builds its own `HybridRetriever` at `recall/eval/locomo.py:288`. Follow the pattern `reranker` already uses.

**Files:**
- Modify: `recall/eval/locomo.py` (`run_conversation` ~line 237, its retriever construction at ~288, and `run` ~line 352)
- Test: `tests/test_eval_locomo_probe.py` (create)

**Interfaces:**
- Consumes: `LegProbe` and the `probe` keyword from Task 3.
- Produces: `probe: Callable[[LegProbe], None] | None = None` keyword on both `run_conversation` and `run`, forwarded to `HybridRetriever`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_locomo_probe.py`:

```python
from __future__ import annotations

import inspect

from recall.eval import locomo


def test_run_conversation_accepts_and_forwards_a_probe():
    sig = inspect.signature(locomo.run_conversation)
    assert "probe" in sig.parameters
    assert sig.parameters["probe"].default is None


def test_run_accepts_a_probe():
    sig = inspect.signature(locomo.run)
    assert "probe" in sig.parameters
    assert sig.parameters["probe"].default is None


def test_run_conversation_passes_probe_to_the_retriever():
    """Guards the wiring itself: a parameter that is accepted and then dropped would leave the
    diagnostic silently collecting nothing."""
    src = inspect.getsource(locomo.run_conversation)
    assert "probe=probe" in src
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_eval_locomo_probe.py -v
```
Expected: FAIL — `AssertionError` on `"probe" in sig.parameters`.

- [ ] **Step 3: Implement**

In `recall/eval/locomo.py`, add the import:

```python
from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever, LegProbe
```
and at the top with the other typing imports:
```python
from collections.abc import Callable
```

Add the parameter to `run_conversation`'s keyword-only block, after `reranker`:

```python
    reranker: Reranker | None = None,
    probe: Callable[[LegProbe], None] | None = None,
    allow_existing: bool = False,
```

Change the retriever construction (line ~288) to:

```python
    retriever = HybridRetriever(
        store, embedder, candidate_k=candidate_k, reranker=reranker, probe=probe
    )
```

Add the same parameter to `run`'s signature after its `reranker` parameter, and forward it at the `run_conversation(...)` call site inside `run`:

```python
    probe: Callable[[LegProbe], None] | None = None,
```
```python
        probe=probe,
```

Note the adversarial arm inside `run_conversation` calls `trusted_search`, not the probed retriever — category 5 is deliberately out of this diagnostic's scope (it has no `evidence` to bucket against).

- [ ] **Step 4: Run the new tests, then the full suite**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_eval_locomo_probe.py -v
```
Expected: 3 passed.

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest -q
```
Expected: 880 passed, 5 skipped.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/eval/locomo.py tests/test_eval_locomo_probe.py && git commit -m "feat(eval): thread an optional probe through the LOCOMO harness

Same pattern reranker already uses. Category 5 stays on trusted_search
and out of scope — it carries no evidence to bucket against.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The diagnostic — Q1, Q2, Q3 and the apparatus check

**Files:**
- Create: `recall/eval/legdiag.py`
- Test: `tests/test_legdiag.py` (create)

**Interfaces:**
- Consumes: `leg_confidence` (Task 2), `LegProbe` (Task 3), the threaded `probe` (Task 4), and `recall.eval.metrics.wilson_ci`.
- Produces: `classify_gold(probe, evidence, k) -> str` returning one of `"a_misranked"`, `"b_unretrieved"`, `"c_absent"`, or `"hit"`; and `build_report(records) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_legdiag.py`. These are pure-function tests over synthetic probes — no database.

```python
from __future__ import annotations

from datetime import datetime, timezone

from recall.eval.legdiag import build_report, classify_gold
from recall.retriever import LegProbe
from recall.types import Chunk, ScoredChunk


def _hit(dia: str, score: float = 0.5) -> ScoredChunk:
    """A hit whose dia id resolves to `dia`.

    `locomo._filename_to_dia_id` is `Path(name).stem.replace("_", ":", 1)` and it reads
    `chunk.metadata["file"]` — there is NO `dia_id` metadata key. Build the filename the way
    the harness does or the classifier silently matches nothing.
    """
    fname = dia.replace(":", "_", 1) + ".md"
    return ScoredChunk(
        chunk=Chunk(id=fname, source=fname, text=dia, metadata={"file": fname}),
        score=score,
        indexed_at=datetime.now(timezone.utc),
    )


def _probe(dense_ids, sparse_ids, fused_ids) -> LegProbe:
    return LegProbe(
        query="q",
        dense=[_hit(c) for c in dense_ids],
        sparse=[_hit(c) for c in sparse_ids],
        sparse_ranks=[1.0 / (i + 1) for i in range(len(sparse_ids))],
        fused=[_hit(c) for c in fused_ids],
    )


def test_classify_hit_when_gold_is_inside_top_k():
    p = _probe(["D1:1", "D1:2"], ["D1:1"], ["D1:1", "D1:2"])
    assert classify_gold(p, ["D1:1"], k=5) == "hit"


def test_classify_a_when_gold_is_in_the_pool_but_below_k():
    fused = [f"D1:{i}" for i in range(1, 9)]        # gold D1:8 sits at rank 8
    p = _probe(fused, [], fused)
    assert classify_gold(p, ["D1:8"], k=5) == "a_misranked"


def test_classify_b_when_gold_is_in_neither_leg():
    p = _probe(["D1:1", "D1:2"], ["D1:3"], ["D1:1", "D1:2", "D1:3"])
    assert classify_gold(p, ["D1:99"], k=5) == "b_unretrieved"


def test_classify_c_when_there_is_no_gold_at_all():
    p = _probe(["D1:1"], [], ["D1:1"])
    assert classify_gold(p, [], k=5) == "c_absent"


def test_report_splits_hit_rate_by_trigger_and_reports_firing_rate():
    records = [
        {"trigger": True, "hit": False, "bucket": "b_unretrieved", "category": 3},
        {"trigger": True, "hit": False, "bucket": "a_misranked", "category": 3},
        {"trigger": True, "hit": True, "bucket": "hit", "category": 1},
        {"trigger": False, "hit": True, "bucket": "hit", "category": 1},
        {"trigger": False, "hit": True, "bucket": "hit", "category": 2},
    ]
    r = build_report(records)

    assert r["q2_firing_rate"]["rate"] == 0.6
    assert r["q2_firing_rate"]["n"] == 5
    assert r["q1_hit_at_k"]["firing"]["rate"] == 1 / 3
    assert r["q1_hit_at_k"]["not_firing"]["rate"] == 1.0
    assert r["q1_hit_at_k"]["delta"] == -2 / 3
    assert r["q3_buckets"]["a_misranked"] == 1
    assert r["q3_buckets"]["b_unretrieved"] == 1
    # every published rate carries an interval
    assert len(r["q2_firing_rate"]["ci"]) == 2


def test_report_handles_an_empty_firing_group():
    records = [{"trigger": False, "hit": True, "bucket": "hit", "category": 1}]
    r = build_report(records)
    assert r["q2_firing_rate"]["rate"] == 0.0
    assert r["q1_hit_at_k"]["firing"]["n"] == 0
    assert r["q1_hit_at_k"]["delta"] is None      # undefined, not zero
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_legdiag.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'recall.eval.legdiag'`.

- [ ] **Step 3: Implement the pure functions**

Create `recall/eval/legdiag.py`:

```python
"""Phase 0 diagnostic — does leg disagreement select for retrieval failures?

Design: docs/superpowers/specs/2026-07-28-weighted-fusion-prf-phase0-design.md
Predictions and kill gates were committed BEFORE this ran. Do not edit them afterwards.

Answers three questions, each with a decision rule fixed in advance:

  Q1  hit@k split on `trigger` — if the firing group is not WORSE, the trigger selects for
      successes and PRF stops here.
  Q2  firing rate — outside 5-50% the trigger needs redesigning.
  Q3  on firing misses, where the gold chunk actually was:
        a_misranked   in the fused pool, below k        -> weighted fusion's job (Phase 1)
        b_unretrieved in neither leg's pool             -> PRF's job (Phase 2); its ceiling
        c_absent      no gold labelled                  -> labelling defect, excluded
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from recall.eval.legconf import leg_confidence
# Reused, not reimplemented. The mapping from a hit to a LOCOMO dialog id lives in
# `_filename_to_dia_id` (stem, first underscore -> colon) and reads `metadata["file"]`. There is
# no `dia_id` key. A local copy of that rule would silently match nothing on drift, which here
# means every miss classifies as "gold was never retrieved" — inflating the PRF ceiling to 100%
# and manufacturing a green light for Phase 2.
from recall.eval.locomo import _retrieved_dia_ids
from recall.eval.metrics import wilson_ci
from recall.retriever import LegProbe

#: `hit@5` and `hit@20` published in FINDINGS §9a, backed by results/locomo/postfix_pool20.json.
EXPECTED_HIT_AT_5 = 0.671
EXPECTED_HIT_AT_20 = 0.855
#: Answerable questions in LOCOMO (categories 1-4). Exact — this is the doubled-corpus check.
EXPECTED_ANSWERABLE_N = 1536
#: Tolerance for the rate asserts. NOT zero: HNSW index builds are nondeterministic (§5b, §6),
#: so demanding equality would fail honest reruns. Wide enough to absorb build noise, far too
#: tight to absorb a structural defect — a doubled corpus moved a headline rate by far more.
HIT_RATE_TOLERANCE = 0.01


def triggered(probe: LegProbe) -> bool:
    """`conf(sparse) > conf(dense)` — the lexical leg was the more decisive one.

    Strict `>`, so a leg with no spread (empty, single candidate, flat) can never fire it:
    `leg_confidence` returns 0.0 there and `conf(dense)` is never negative.
    """
    return leg_confidence(probe.sparse_ranks) > leg_confidence([h.score for h in probe.dense])


def classify_gold(probe: LegProbe, evidence: Sequence[str], k: int) -> str:
    """Where the gold chunk sits relative to what retrieval produced.

    Note `_retrieved_dia_ids` returns DISTINCT dia ids best-rank-first, so slicing the fused
    hits to `k` before mapping (rather than mapping then slicing) is what makes "inside the
    top k" mean the same thing here as it does in `_hit_by_depth`.
    """
    if not evidence:
        return "c_absent"
    gold = set(evidence)
    if gold & set(_retrieved_dia_ids(probe.fused[:k])):
        return "hit"
    pool = set(_retrieved_dia_ids(probe.dense)) | set(_retrieved_dia_ids(probe.sparse))
    return "a_misranked" if gold & pool else "b_unretrieved"


def _mean(flags: list[bool]) -> float:
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def _rate(flags: list[bool]) -> dict[str, Any]:
    if not flags:
        return {"rate": 0.0, "n": 0, "ci": [None, None]}
    lo, hi = wilson_ci(flags)
    return {
        "rate": sum(1 for f in flags if f) / len(flags),
        "n": len(flags),
        "ci": [round(lo, 4), round(hi, 4)],
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Q1/Q2/Q3 from per-question records. Pure — every figure traces to `records`."""
    scored = [r for r in records if r["bucket"] != "c_absent"]
    firing = [r for r in scored if r["trigger"]]
    not_firing = [r for r in scored if not r["trigger"]]

    q1_firing = _rate([r["hit"] for r in firing])
    q1_not = _rate([r["hit"] for r in not_firing])
    delta = (q1_firing["rate"] - q1_not["rate"]) if (firing and not_firing) else None

    buckets: dict[str, int] = {}
    for r in scored:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1

    by_category: dict[int, dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in scored}):
        by_category[cat] = _rate([r["trigger"] for r in scored if r["category"] == cat])

    return {
        "n_scored": len(scored),
        "n_excluded_unlabelled": len(records) - len(scored),
        "q1_hit_at_k": {"firing": q1_firing, "not_firing": q1_not, "delta": delta},
        "q2_firing_rate": _rate([r["trigger"] for r in scored]),
        "q2_firing_rate_by_category": by_category,
        "q3_buckets": buckets,
        "q3_buckets_firing_misses": {
            b: sum(1 for r in firing if r["bucket"] == b)
            for b in ("a_misranked", "b_unretrieved")
        },
    }


def check_apparatus(hit_at_5: float, hit_at_20: float, answerable_n: int) -> None:
    """Fail the run if the instrumented pipeline is not the measured one.

    A corrupted apparatus does not raise — it returns plausible numbers and a manufactured
    finding. Exit code 0 is not a measurement.
    """
    if answerable_n != EXPECTED_ANSWERABLE_N:
        raise RuntimeError(
            f"apparatus: scored {answerable_n} answerable questions, expected "
            f"{EXPECTED_ANSWERABLE_N}. The corpus or the label set is not the one §9a measured."
        )
    for name, got, want in (
        ("hit@5", hit_at_5, EXPECTED_HIT_AT_5),
        ("hit@20", hit_at_20, EXPECTED_HIT_AT_20),
    ):
        if abs(got - want) > HIT_RATE_TOLERANCE:
            raise RuntimeError(
                f"apparatus: {name} reads {got:.4f}, §9a published {want} "
                f"(tolerance {HIT_RATE_TOLERANCE}). Instrumentation changed the retrieved set; "
                f"the diagnostic below would be measuring something else."
            )
```

- [ ] **Step 4: Run the tests**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_legdiag.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/eval/legdiag.py tests/test_legdiag.py && git commit -m "feat(eval): Phase 0 diagnostic — trigger, gold-bucketing, apparatus check

classify_gold splits misses into fusion's job (in the pool, below k) and
PRF's job (in neither pool) — the latter is the ceiling on what a second
retrieval pass could buy. check_apparatus asserts the run reproduces
§9a's published rates within HNSW build noise, and the answerable count
exactly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Add the CLI entry point**

Append to `recall/eval/legdiag.py`:

```python
def main(argv: list[str] | None = None) -> int:
    """Run the LOCOMO diagnostic.

    Calls `run_conversation` per conversation rather than `run`, mirroring `run`'s own loop.
    Not a stylistic choice: `run`'s returned report STRIPS the per-question records
    (``{kk: vv for kk, vv in res.items() if kk != "questions"}``), and this diagnostic needs
    each question's `evidence`, `category` and `hit_by_k` to bucket against. Per-conversation
    probe lists also make the probe/record pairing checkable, which one global list would not.
    """
    import argparse
    import json
    import shutil
    import tempfile
    import uuid
    from pathlib import Path

    import psycopg

    from recall.eval import locomo
    from recall.store import PgVectorStore

    p = argparse.ArgumentParser(description="Phase 0 leg-disagreement diagnostic (LOCOMO)")
    p.add_argument("--data", required=True, type=Path, help="path to locomo10.json")
    p.add_argument("--dsn", default=locomo.DEFAULT_DSN)
    p.add_argument("--embedder", default="fastembed")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="first N conversations only")
    p.add_argument(
        "--table",
        default=None,
        help="table to index into. Default: a uuid-named table, dropped afterwards — so a "
             "rerun never trips run_conversation's existing-rows refusal and never touches a "
             "table anyone else owns.",
    )
    p.add_argument("--out", required=True, type=Path, help="write the JSON report + dump here")
    p.add_argument(
        "--skip-apparatus-check",
        action="store_true",
        help="run without asserting §9a's rates. For debugging only — a report produced with "
             "this flag is not evidence and must not be published.",
    )
    a = p.parse_args(argv)

    depths = [1, 5, 10, 20]
    embedder = locomo._make_embedder(a.embedder)
    conversations = json.loads(a.data.read_text(encoding="utf-8"))
    if a.limit is not None:
        conversations = conversations[: a.limit]

    table = a.table or ("legdiag_" + uuid.uuid4().hex[:8])
    records: list[dict[str, Any]] = []
    workspace = Path(tempfile.mkdtemp(prefix="legdiag-"))
    try:
        for i, conv in enumerate(conversations):
            sample_id = conv.get("sample_id") or f"conv{i}"
            # One tenant per conversation, exactly as `run` does: LOCOMO's conversations are
            # unrelated worlds and dia ids are only unique WITHIN one, so a shared tenant would
            # let a cross-conversation "D1:3" score as a hit.
            probes: list[LegProbe] = []
            with PgVectorStore(
                a.dsn, dim=embedder.dim, tenant=f"locomo-{sample_id}", table=table
            ) as store:
                res = locomo.run_conversation(
                    conv["conversation"],
                    conv.get("qa") or [],
                    store=store,
                    embedder=embedder,
                    k=a.k,
                    corpus_dir=workspace / str(sample_id),
                    ks=depths,
                    candidate_k=a.candidate_k,
                    probe=probes.append,
                )

            # Only answerable, labelled questions reach the probed retriever: category 5 goes to
            # `trusted_search`, and an unlabelled question `continue`s before the search. So these
            # two lists must be equal in length and in order. If they ever diverge, every record
            # below is mis-paired with someone else's legs — fail loudly rather than zip a silent
            # off-by-one into the finding.
            answerable = [q for q in res["questions"] if "evidence" in q]
            if len(answerable) != len(probes):
                raise RuntimeError(
                    f"{sample_id}: {len(answerable)} answerable questions but {len(probes)} "
                    f"probes — records would be mis-paired, refusing to continue"
                )

            for q, probe in zip(answerable, probes, strict=True):
                records.append(
                    {
                        "sample_id": str(sample_id),
                        "question": q["question"],
                        "category": q["category"],
                        "trigger": triggered(probe),
                        "conf_dense": round(leg_confidence([h.score for h in probe.dense]), 4),
                        "conf_sparse": round(leg_confidence(probe.sparse_ranks), 4),
                        "n_dense": len(probe.dense),
                        "n_sparse": len(probe.sparse),
                        "hit": q["hit"],
                        "hit_by_k": {str(d): q["hit_by_k"][d] for d in depths},
                        "bucket": classify_gold(probe, q["evidence"], a.k),
                    }
                )
            print(
                f"  [{i + 1}/{len(conversations)}] {sample_id}: {len(answerable)} scored",
                flush=True,
            )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        if a.table is None:  # only ever drops the table this run created and named
            with psycopg.connect(a.dsn, autocommit=True) as conn:
                conn.execute(f"DROP TABLE IF EXISTS {table}")

    if not a.skip_apparatus_check:
        # Computed from the SAME records the diagnostic buckets, so the check validates the data
        # actually used rather than a parallel aggregate that could agree while these diverge.
        check_apparatus(
            hit_at_5=_mean([r["hit_by_k"]["5"] for r in records]),
            hit_at_20=_mean([r["hit_by_k"]["20"] for r in records]),
            answerable_n=len(records),
        )

    out = {
        "config": {
            "embedder": a.embedder,
            "k": a.k,
            "candidate_k": a.candidate_k,
            "reranker": None,
            "conversations": len(conversations),
        },
        "diagnostic": build_report(records),
        "records": records,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["diagnostic"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Verified against the source while planning** — these were four real bugs in an earlier draft of this file, all of the silent kind, and they are why the plan quotes exact names:

- `run()`'s report strips `questions`, so the diagnostic calls `run_conversation` per conversation.
- `run()`'s signature is `run(data_path, *, dsn, embedder_name, k, limit, keep_corpus, table, ...)` — `limit`, `keep_corpus` and `table` have no defaults.
- `report["depth_curve"]` is keyed by `str(d)` and the rate sits at `curve[str(d)]["overall"]["rate"]`, not `curve[d]["rate"]`. Sidestepped entirely by computing from `hit_by_k`.
- There is no `dia_id` metadata key; the mapping is `_filename_to_dia_id` over `metadata["file"]`.

- [ ] **Step 7: Full suite, then commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest -q && uv run ruff check .
```
Expected: 886 passed, 5 skipped; ruff clean.

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/eval/legdiag.py && git commit -m "feat(eval): legdiag CLI

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Run it, and score the predictions

**Files:**
- Create: `results/legdiag/locomo_phase0.json`
- Create: `results/legdiag/FINDINGS_phase0.md`

- [ ] **Step 1: Run the diagnostic**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_DSN="postgresql://recall:recall@localhost:5434/recall" uv run python -m recall.eval.legdiag --data locomo10.json --dsn "postgresql://recall:recall@localhost:5434/recall" --out results/legdiag/locomo_phase0.json
```

Expected: the apparatus check passes, then the Q1/Q2/Q3 block prints. **If `check_apparatus` raises, stop.** Do not pass `--skip-apparatus-check` to get past it — the raise means the instrumented pipeline is not the one §9a measured, and every number after it would be fiction.

- [ ] **Step 2: Apply the decision rules — before interpreting anything else**

Read them off the spec's table, in this order:

| gate | rule | if it fires |
|---|---|---|
| Q1 | firing-group hit@5 ≥ non-firing | **Stop.** No Phase 2. Write the closure note. |
| Q2 | firing rate outside 5–50% | Redesign the trigger; do not proceed to Phase 1 on this trigger. |
| Q3 | `b_unretrieved` ≈ 0 among firing misses | No Phase 2; the work reduces to weighted fusion alone. |
| Q3 | `a_misranked` ≈ 0 | No measurable Phase 1 gain on this corpus; report it. |

- [ ] **Step 3: Write `results/legdiag/FINDINGS_phase0.md`**

It must contain, in this order: the three answers with CIs; the firing rate overall and per category; **each preregistered prediction quoted from `ef68bb1` with a HIT/MISS verdict and, for hits, whether it was right for the right reason**; which gates fired; and the reproduce command from Step 1. Per `reference-recall-docs-evidence-tier-convention`, no figure ships without its retained artifact — `locomo_phase0.json` is that artifact and is committed alongside.

- [ ] **Step 4: Commit and report**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add results/legdiag/ && git commit -m "results: Phase 0 leg-disagreement diagnostic

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Then report to the user: the three answers, which gates fired, the prediction scorecard, and the resulting go/no-go for Phases 1 and 2. **A null result is the deliverable if that is what the data says.**

---

## Out of scope

Weighted fusion itself, the PRF second pass, the privileged-vs-unprivileged expansion-leg A/B, any embedder or chunking change, and LongMemEval. One variable at a time.

## Not yet run here

The private-46 arm (`recall.eval.labelled`) is in the spec's measurement set but needs the corpus owner's private data and a second probe threading. It is a follow-up task once the LOCOMO arm has cleared its gates — LOCOMO is the arm that decides, because it is public and n=1,536.
