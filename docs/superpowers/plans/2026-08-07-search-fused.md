# search_fused Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `HybridRetriever.search_fused(query, history, k, source)`, which fuses the current turn with a concatenation of prior turns, reproducing the measured `mq_nested2_nogold` arm.

**Architecture:** `search()` and `search_fused()` both delegate to a new private `_retrieve_legs()`. `search()` keeps its exact current behaviour. `search_fused()` runs `_retrieve_legs` twice, fuses each variant's legs (inner RRF), fuses the two variants (outer RRF), caps the pool at 100, reranks once, truncates, then re-scores the returned hits against the query's embedding so every reported cosine is on one basis.

**Tech Stack:** Python 3.12, psycopg, pgvector, sentence-transformers, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-07-multi-query-serving-design.md`. Read it before Task 1.
- **`search()` behaviour must not change.** Its output is consumed by `recall/trust.py` and a dozen eval harnesses.
- **`hit.score` must always be the dense cosine against `query`**, never against the history. `trust.py:536` feeds it to `cal.confidence()`.
- RRF damping constant is **60** at both levels (`_rrf`'s existing default). Do not introduce a second constant.
- **`FUSED_RERANK_POOL_CAP = 100`** and **`FUSED_HISTORY_MAX_CHARS = 4096`**.
- All refusals are `ValueError` and must name the actual and the expected state.
- Every new public store method must be added to BOTH `TIMED_PUBLIC_METHODS` and `STORE_QUERY_LEGS` in `recall/store.py`, or `tests/test_store_query_latency.py` and `tests/test_retrieval_cost_surface.py` go red. This exact omission broke CI on 2026-08-06.
- Run `python -m mypy` before every commit. CI runs bare `mypy` and it is a required check.
- No dashes as punctuation in comments or docstrings; use commas, colons, semicolons, parentheses or full stops.

---

### Task 1: Extract `_retrieve_legs` from `search()`

Pure refactor. No behaviour change. This is the seam that stops the two search methods drifting.

**Files:**
- Modify: `recall/retriever.py:141-246`
- Test: `tests/test_retriever_legs.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `_Legs` dataclass with fields `qvec: list[float]`, `dense: list[ScoredChunk]`, `sparse: list[ScoredChunk]`, `learned: list[ScoredChunk]`, `timings: dict[str, float]`; and `HybridRetriever._retrieve_legs(self, query: str, source: str | None, report_vec: list[float] | None = None) -> _Legs`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retriever_legs.py`:

```python
"""`_retrieve_legs` is the seam `search` and `search_fused` share.

If it drifts from what `search` used to do inline, every existing caller changes behaviour
silently. These tests pin the seam itself rather than the methods built on it.
"""

from __future__ import annotations

from recall.retriever import HybridRetriever, _Legs
from tests.fakes import FakeEmbedder, FakeStore  # see Step 1b if these do not exist


def test_retrieve_legs_reports_sparse_cosines_against_the_query_by_default() -> None:
    """Default behaviour must match what `search` did inline: sparse hits carry the query cosine."""
    store = FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)])
    retriever = HybridRetriever(store, FakeEmbedder(), sparse_backend="lexical")

    legs = retriever._retrieve_legs("what is x", source=None)

    assert isinstance(legs, _Legs)
    assert store.sparse_vec_used == legs.qvec


def test_retrieve_legs_can_report_against_a_different_vector() -> None:
    """`search_fused` needs the HISTORY variant's sparse legs to report the QUERY's cosine.

    Without this the returned scores mix two bases, and `cal.confidence()` in trust.py silently
    receives a cosine measured against a different string.
    """
    store = FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)])
    retriever = HybridRetriever(store, FakeEmbedder(), sparse_backend="lexical")
    other = [0.5] * FakeEmbedder().dim

    legs = retriever._retrieve_legs("history text", source=None, report_vec=other)

    assert store.sparse_vec_used == other
    assert legs.qvec != other  # the leg still RANKS by its own embedding
```

- [ ] **Step 1b: Add the fakes if `tests/fakes.py` lacks them**

Check first: `grep -rn "class FakeStore" tests/`. If a suitable fake exists, import it and skip this step. Otherwise create `tests/fakes.py`:

```python
"""Minimal duck-typed store and embedder for retriever tests. No database."""

from __future__ import annotations

from datetime import UTC, datetime

from recall.types import Chunk, ScoredChunk


class FakeEmbedder:
    dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]


class FakeStore:
    def __init__(
        self,
        dense: list[tuple[str, float]] | None = None,
        sparse: list[tuple[str, float]] | None = None,
        learned: list[tuple[str, float]] | None = None,
    ) -> None:
        self._dense = dense or []
        self._sparse = sparse or []
        self._learned = learned or []
        self.sparse_vec_used: list[float] | None = None
        self.learned_vec_used: list[float] | None = None
        self.cosines_calls: list[tuple[tuple[str, ...], tuple[float, ...]]] = []

    @staticmethod
    def _hits(rows: list[tuple[str, float]]) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=Chunk(id=cid, source="s", text=f"text-{cid}", metadata={}),
                score=score,
            )
            for cid, score in rows
        ]

    def query_dense(self, vector, k, source=None):
        return self._hits(self._dense[:k])

    def query_sparse(self, text, k, source=None, vec=None):
        self.sparse_vec_used = vec
        return self._hits(self._sparse[:k])

    def query_learned_sparse(self, weights, k, profile_id, source=None, vec=None):
        self.learned_vec_used = vec
        return self._hits(self._learned[:k])

    def cosines_for(self, ids, vec):
        self.cosines_calls.append((tuple(ids), tuple(vec)))
        return {cid: 0.77 for cid in ids}

    def newest_indexed_at(self):
        return datetime.now(UTC)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_retriever_legs.py -q`
Expected: FAIL with `ImportError: cannot import name '_Legs'`.

- [ ] **Step 3: Implement `_Legs` and `_retrieve_legs`**

In `recall/retriever.py`, add after the `_rrf` function:

```python
@dataclass(frozen=True)
class _Legs:
    """One query's retrieval legs, before any fusion.

    The seam `search` and `search_fused` share. Extracted rather than duplicated because two
    copies of this pipeline would drift, and the drift would be invisible: both would still
    return plausible hits.
    """

    qvec: list[float]
    dense: list
    sparse: list
    learned: list
    timings: dict[str, float]
```

Add `from dataclasses import dataclass, is_dataclass, replace` to the existing dataclasses import.

Then add the method to `HybridRetriever`, moving the body verbatim out of `search`:

```python
    def _retrieve_legs(
        self, query: str, source: str | None, report_vec: list[float] | None = None
    ) -> _Legs:
        """Run every enabled leg for one query.

        `report_vec` overrides which vector the SPARSE legs report their cosine against, without
        changing what they rank by. `search` leaves it None and gets the pre-existing behaviour.
        `search_fused` passes the QUERY's vector when retrieving for the HISTORY variant, so that
        every returned hit's score is on one basis: `trust.py` feeds `hit.score` to a calibrated
        confidence, and a cosine against a different string would silently mean something else.
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()
        qvec = embed_query(self._embedder, query)
        timings["query_embedding"] = (time.perf_counter() - started) * 1000.0
        reporting = qvec if report_vec is None else report_vec

        started = time.perf_counter()
        dense = (
            self._store.query_dense(qvec, k=self._candidate_k, source=source)
            if self._use_dense
            else []
        )
        timings["dense_retrieval"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        wants_lexical = self._use_sparse and self._sparse_backend in ("lexical", "both")
        sparse = (
            self._store.query_sparse(query, k=self._candidate_k, source=source, vec=reporting)
            if wants_lexical
            else []
        )
        timings["sparse_retrieval"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        learned: list = []
        if self._use_sparse and self._sparse_backend in ("splade", "both"):
            encoder = self._sparse_encoder
            assert encoder is not None  # guaranteed by __init__; re-stated for the type checker
            weights = encoder.encode([query])[0]  # type: ignore[attr-defined]
            if weights:
                learned = self._store.query_learned_sparse(
                    weights,
                    k=self._candidate_k,
                    profile_id=encoder.profile.profile_id,  # type: ignore[attr-defined]
                    source=source,
                    vec=reporting,
                )
        timings["learned_sparse_retrieval"] = (time.perf_counter() - started) * 1000.0
        return _Legs(qvec=qvec, dense=dense, sparse=sparse, learned=learned, timings=timings)
```

Now replace the top of `search` (everything from `timings: dict[str, float] = {}` down to the `learned_sparse_retrieval` timing line) with:

```python
        legs = self._retrieve_legs(query, source)
        timings = dict(legs.timings)
        dense, sparse, learned = legs.dense, legs.sparse, legs.learned
```

Leave the rest of `search` exactly as it is.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_retriever_legs.py tests/test_retrieval_cost_surface.py -q`
Expected: PASS. The cost-surface test proves the stage names are unchanged.

- [ ] **Step 5: Run the full retriever-facing suite and mypy**

Run: `python -m pytest tests/ -q -k "retriev or trust or rerank" && python -m mypy`
Expected: all pass, `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add recall/retriever.py tests/test_retriever_legs.py tests/fakes.py
git commit -m "refactor(retriever): extract _retrieve_legs so search and search_fused cannot drift"
```

---

### Task 2: Add `PgVectorStore.cosines_for`

**Files:**
- Modify: `recall/store.py` (constants near line 78-179, new method near `newest_indexed_at` at line 2007)
- Test: `tests/test_store_cosines_for.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `PgVectorStore.cosines_for(ids: Sequence[str], vec: list[float]) -> dict[str, float]`, returning cosine similarity in [-1, 1] for each id that exists; ids absent from the table are omitted from the mapping. Also `LEG_RESCORE = "rescore"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store_cosines_for.py`:

```python
"""`cosines_for` puts every returned hit's score on one basis.

`search_fused` retrieves with two different query embeddings. Without a way to re-score the
returned hits against the QUERY's embedding, hits surfaced only by the history variant would
carry a cosine measured against the history, and `trust.py` would push that through
`cal.confidence()` as though it meant the same thing.
"""

from __future__ import annotations

import pytest

from recall.store import STORE_QUERY_LEGS, TIMED_PUBLIC_METHODS


def test_cosines_for_is_registered_in_the_timing_surface() -> None:
    """A public store method missing from these tuples drops its timing silently.

    `GenerationStore` made this mistake twice, and it broke CI again on 2026-08-06. The guard is
    a tuple rather than a docstring precisely because docstrings do not fail builds.
    """
    assert "cosines_for" in TIMED_PUBLIC_METHODS
    assert "rescore" in STORE_QUERY_LEGS


@pytest.mark.usefixtures("pg_store")
def test_cosines_for_returns_the_cosine_against_the_given_vector(pg_store) -> None:
    """The value must match what `query_dense` reports for the same chunk and vector."""
    from recall.types import Chunk

    chunks = [Chunk(id="c1", source="s", text="alpha", metadata={})]
    vec = [1.0] + [0.0] * (pg_store.dim - 1)
    pg_store.upsert(chunks, [vec])

    dense = pg_store.query_dense(vec, k=1)
    rescored = pg_store.cosines_for(["c1"], vec)

    assert rescored["c1"] == pytest.approx(dense[0].score, abs=1e-6)


@pytest.mark.usefixtures("pg_store")
def test_cosines_for_omits_ids_that_do_not_exist(pg_store) -> None:
    """An absent id is not a zero. Zero is a real cosine and would look like a poor match."""
    vec = [1.0] + [0.0] * (pg_store.dim - 1)

    assert pg_store.cosines_for(["nope"], vec) == {}


def test_cosines_for_returns_empty_for_no_ids(pg_store) -> None:
    """No round trip for an empty request."""
    vec = [1.0] + [0.0] * (pg_store.dim - 1)

    assert pg_store.cosines_for([], vec) == {}
```

Find the existing Postgres fixture name first: `grep -rn "def pg_store\|@pytest.fixture" tests/conftest.py | head`. If the project's fixture has a different name, use that name throughout instead of `pg_store`.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_store_cosines_for.py -q`
Expected: FAIL with `ImportError` or `AttributeError: 'PgVectorStore' object has no attribute 'cosines_for'`.

- [ ] **Step 3: Add the constant and register it**

In `recall/store.py`, after `LEG_META = "meta"` (line 141):

```python
#: Re-scoring specific ids against a query vector. Its own leg rather than folded into
#: `LEG_META`: it is on the query path only for fused searches, so an operator comparing
#: `search` against `search_fused` needs to see it separately to know what fusion costs.
LEG_RESCORE = "rescore"
```

Change line 158 to:

```python
STORE_QUERY_LEGS = (LEG_DENSE, LEG_SPARSE, LEG_LEARNED_SPARSE, LEG_META, LEG_RESCORE)
```

Change `TIMED_PUBLIC_METHODS` (line 174) to:

```python
TIMED_PUBLIC_METHODS = (
    "query_dense",
    "query_sparse",
    "query_learned_sparse",
    "newest_indexed_at",
    "cosines_for",
)
```

- [ ] **Step 4: Implement the method**

Add next to `newest_indexed_at` in `recall/store.py`:

```python
    def cosines_for(self, ids: "Sequence[str]", vec: list[float]) -> dict[str, float]:
        """Cosine similarity between `vec` and each of `ids`, for ids that exist.

        `search_fused` retrieves with two query embeddings, so a hit surfaced only by the history
        variant carries a cosine against the HISTORY. `hit.score` is not decorative: `trust.py`
        thresholds on it and feeds it to `cal.confidence()`, a calibration fitted on cosines
        against the query. This puts every returned hit back on that one basis.

        Ids that do not exist are OMITTED rather than returned as 0.0. Zero is a real cosine and
        would read as a genuine poor match; absence is a different fact and the caller can tell.

        Subclasses override `_cosines_for`, not this; see `TIMED_PUBLIC_METHODS`.
        """
        if not ids:
            return {}
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_RESCORE):
            return self._cosines_for(ids, vec)

    def _cosines_for(self, ids: "Sequence[str]", vec: list[float]) -> dict[str, float]:
        wanted = list(dict.fromkeys(str(i) for i in ids))

        def _op(conn: "psycopg.Connection") -> dict[str, float]:
            rows = conn.execute(
                f"SELECT id, 1 - (embedding <=> %s::vector) FROM {self._table} "
                f"WHERE tenant_id = %s AND id = ANY(%s)",
                (vec, self._tenant, wanted),
            ).fetchall()
            return {str(row[0]): float(row[1]) for row in rows}

        return self._with_retry(_op)
```

Confirm the embedding column name and the `<=>` usage by reading `_query_dense` first: `grep -n "embedding <=>" recall/store.py`. Use whatever column name and cast that method uses. If `Sequence` is not imported, add `from collections.abc import Sequence` to the imports.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_store_cosines_for.py tests/test_store_query_latency.py tests/test_retrieval_cost_surface.py -q`
Expected: PASS. The latter two prove the timing surface stayed consistent.

- [ ] **Step 6: mypy and commit**

```bash
python -m mypy
git add recall/store.py tests/test_store_cosines_for.py
git commit -m "feat(store): cosines_for, so fused search can put every hit on one basis"
```

---

### Task 3: History concatenation and its budget

**Files:**
- Modify: `recall/retriever.py`
- Test: `tests/test_fused_history.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `FUSED_HISTORY_MAX_CHARS = 4096`, `FUSED_RERANK_POOL_CAP = 100`, and `build_history_query(history: Sequence[str]) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fused_history.py`:

```python
"""The concatenation rule is owned by RE-call, not by callers.

If each caller built its own history string, two installations would send different
concatenations and both would call it the measured configuration. The benchmark's `full` variant
is one specific rule and this reproduces it.
"""

from __future__ import annotations

import pytest

from recall.retriever import FUSED_HISTORY_MAX_CHARS, build_history_query


def test_turns_are_newline_joined_in_order() -> None:
    assert build_history_query(["first", "second"]) == "first\nsecond"


def test_speaker_prefixes_are_stripped() -> None:
    """MTRAG-human prefixes every turn with `|user|: `.

    Left in, the literal token reaches both encoders on every query and depresses the whole run
    with nothing failing. The benchmark stripped it; serving must strip it identically.
    """
    assert build_history_query(["|user|: what is x", "|user|: and y"]) == "what is x\nand y"


def test_a_colon_inside_a_turn_survives() -> None:
    """Only a leading speaker tag is removed. A colon in the question is content."""
    assert build_history_query(["note: this matters"]) == "note: this matters"


def test_blank_turns_are_dropped_not_joined_as_empty_lines() -> None:
    assert build_history_query(["a", "   ", "b"]) == "a\nb"


def test_the_budget_is_4096_matching_the_mcp_cap() -> None:
    assert FUSED_HISTORY_MAX_CHARS == 4096
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_fused_history.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_history_query'`.

- [ ] **Step 3: Implement**

In `recall/retriever.py`, after `SPARSE_BACKENDS`:

```python
#: The most distinct chunks a fused search hands the cross-encoder.
#:
#: 🔑 MEASURED, not chosen for tidiness. Reranking a 547-candidate pool LOST 0.0513 R@100 where
#: reranking 200 GAINED 0.0226: the cross-encoder degrades as the pool widens, which
#: `closed-hypothesis-recall-rerank-pool-interaction-2026-08-05` recorded as "not more to select
#: from, more rope". Fusing two queries roughly doubles the pool, so without this cap the measured
#: gain reverses.
FUSED_RERANK_POOL_CAP = 100

#: Longest history CONCATENATION a fused search accepts, in characters.
#:
#: Matches `recall_mcp.service.MAX_QUERY_CHARS` so the library and the server agree on what an
#: over-long query is. Measured against the concatenation, which is built here and therefore never
#: passes the server's own check on the incoming query.
#:
#: ⚠️ The encoder truncates at 512 tokens regardless, so a history past roughly 2,000 characters
#: is already being clipped by the tokenizer. That clipping was PRESENT IN THE MEASUREMENT, so it
#: is part of the measured system rather than something introduced here. This budget bounds the
#: REQUEST, not the encoded query, and the two are different limits.
FUSED_HISTORY_MAX_CHARS = 4096


def build_history_query(history: "Sequence[str]") -> str:
    """The benchmark's `full` variant: prior turns, newline joined, speaker tags removed.

    Owned here rather than left to callers so that two installations cannot send different
    concatenations and both call it the measured configuration.
    """
    lines = []
    for turn in history:
        for line in str(turn).splitlines():
            stripped = (
                line.split(":", 1)[1].strip()
                if line.startswith("|") and ":" in line
                else line.strip()
            )
            if stripped:
                lines.append(stripped)
    return "\n".join(lines)
```

Add `from collections.abc import Sequence` to the imports if absent.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_fused_history.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: mypy and commit**

```bash
python -m mypy
git add recall/retriever.py tests/test_fused_history.py
git commit -m "feat(retriever): own the history concatenation rule and declare its budget"
```

---

### Task 4: `search_fused` refusals

Refusals first, before any retrieval, so the expensive path is never entered in a state the measurement does not cover.

**Files:**
- Modify: `recall/retriever.py`
- Test: `tests/test_search_fused.py` (create)

**Interfaces:**
- Consumes: `build_history_query`, `FUSED_HISTORY_MAX_CHARS`.
- Produces: `HybridRetriever.search_fused(self, query: str, history: Sequence[str], k: int = 5, source: str | None = None) -> RetrievalResult` (raising only, at this task).

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_fused.py`:

```python
"""`search_fused` refuses rather than degrades.

The measured gain is CONDITIONAL on reranking: raw, this arm is worse by 0.0447 nDCG@5 and
tripped three preregistered ranking vetoes. RE-call ships with the reranker OFF by default, so an
operator enabling fusion without one would silently get a worse system than `search()`.
"""

from __future__ import annotations

import pytest

from recall.retriever import HybridRetriever
from tests.fakes import FakeEmbedder, FakeStore


class _StubReranker:
    def rerank(self, query, hits):
        return list(hits)


def _retriever(reranker=None):
    return HybridRetriever(
        FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)]),
        FakeEmbedder(),
        reranker=reranker,
        sparse_backend="lexical",
    )


def test_fused_search_without_a_reranker_is_refused() -> None:
    """The gain does not exist without reranking, so serving it without one is a worse system."""
    with pytest.raises(ValueError, match="reranker"):
        _retriever().search_fused("q", ["earlier turn"])


def test_empty_history_is_refused_rather_than_silently_becoming_a_single_query() -> None:
    """A caller wanting single-query behaviour should call `search`, not get it by accident."""
    with pytest.raises(ValueError, match="history"):
        _retriever(_StubReranker()).search_fused("q", [])


def test_an_over_budget_history_is_refused_and_names_both_lengths() -> None:
    """Refused, never truncated.

    A truncated history is a configuration the benchmark never tested, served under the measured
    configuration's name. Same principle as `resolve_reranker` refusing an unparseable flag rather
    than reading it as "off".
    """
    with pytest.raises(ValueError, match="4096"):
        _retriever(_StubReranker()).search_fused("q", ["x" * 5000])


def test_k_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="k must be"):
        _retriever(_StubReranker()).search_fused("q", ["earlier"], k=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_search_fused.py -q`
Expected: FAIL with `AttributeError: 'HybridRetriever' object has no attribute 'search_fused'`.

- [ ] **Step 3: Implement the refusals**

Add to `HybridRetriever`:

```python
    def search_fused(
        self,
        query: str,
        history: "Sequence[str]",
        k: int = 5,
        source: str | None = None,
    ) -> RetrievalResult:
        """Retrieve for `query` fused with a concatenation of `history`, then rerank once.

        Reproduces the `mq_nested2_nogold` arm measured on 2026-08-06/07: +0.0084 nDCG@5
        (MiniLM, Holm-significant) and +0.0842 R@100 over single-query search, measured at
        `candidate_k=100` with a reranker on MTRAG-human dev. Other settings are untested rather
        than merely different.

        ⚠️ The gain is CONDITIONAL on reranking. Raw, this arm scores 0.0447 nDCG@5 BELOW
        `search()`. That is why a missing reranker is refused instead of warned about.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if self._reranker is None:
            raise ValueError(
                "search_fused requires a reranker: the fused arm was measured at +0.0084 nDCG@5 "
                "WITH one and at -0.0447 WITHOUT one, so serving it unreranked would be a "
                "measurably worse system than search(). Pass a reranker, or call search()."
            )
        if not history:
            raise ValueError(
                "search_fused requires a non-empty history; call search() for single-query "
                "retrieval rather than passing an empty history and getting it implicitly"
            )
        history_query = build_history_query(history)
        if len(history_query) > FUSED_HISTORY_MAX_CHARS:
            raise ValueError(
                f"history concatenation is {len(history_query)} characters, over the "
                f"{FUSED_HISTORY_MAX_CHARS} budget. It is refused rather than truncated: a "
                f"truncated history is a configuration that was never measured. Send fewer turns."
            )
        if not history_query:
            raise ValueError(
                "history contained no usable text after stripping speaker tags and blank turns"
            )
        raise NotImplementedError("fusion lands in Task 5")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_search_fused.py -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: mypy and commit**

```bash
python -m mypy
git add recall/retriever.py tests/test_search_fused.py
git commit -m "feat(retriever): search_fused refusals, because the gain is conditional on reranking"
```

---

### Task 5: Fusion, pool cap and re-scoring

**Files:**
- Modify: `recall/retriever.py`
- Test: `tests/test_search_fused.py` (extend)

**Interfaces:**
- Consumes: `_retrieve_legs`, `_Legs`, `build_history_query`, `FUSED_RERANK_POOL_CAP`, `store.cosines_for`.
- Produces: a working `search_fused` returning `RetrievalResult`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_fused.py`:

```python
def test_every_returned_score_is_the_cosine_against_the_query() -> None:
    """The reason `cosines_for` exists.

    A hit surfaced only by the HISTORY variant carries a cosine against the history. `trust.py`
    feeds `hit.score` to `cal.confidence()`, a calibration fitted on cosines against the QUERY, so
    a mixed basis produces a confidence that silently means something else.
    """
    store = FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), sparse_backend="lexical"
    )

    result = retriever.search_fused("q", ["earlier turn"], k=5)

    assert result.hits, "expected hits"
    assert all(hit.score == 0.77 for hit in result.hits)  # FakeStore.cosines_for returns 0.77
    assert store.cosines_calls, "cosines_for was never called"
    ids, vec = store.cosines_calls[-1]
    assert set(ids) == {hit.chunk.id for hit in result.hits}
    assert list(vec) == retriever._retrieve_legs("q", source=None).qvec


def test_the_pool_is_capped_before_reranking() -> None:
    """Reranking 547 candidates lost 0.0513 R@100 where 200 gained 0.0226.

    Fusion roughly doubles the pool, so an absent cap is a measured regression, not an
    inefficiency.
    """
    seen: dict[str, int] = {}

    class _CountingReranker:
        def rerank(self, query, hits):
            seen["count"] = len(hits)
            return list(hits)

    rows = [(f"d{i}", 0.5) for i in range(150)]
    store = FakeStore(dense=rows, sparse=[(f"s{i}", 0.4) for i in range(150)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_CountingReranker(),
        candidate_k=150, sparse_backend="lexical",
    )

    retriever.search_fused("q", ["earlier"], k=5)

    assert seen["count"] == 100


def test_gap_warning_uses_the_query_not_the_history() -> None:
    """The honesty guard must answer "does the corpus hold an answer to what the USER asked".

    Letting a strong match on stale earlier context suppress the warning is the guard failing in
    the dangerous direction.
    """
    store = FakeStore(dense=[("a", 0.01)], sparse=[])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), sparse_backend="lexical",
        gap_threshold=0.5,
    )

    assert retriever.search_fused("q", ["earlier"], k=5).gap_warning is True


def test_the_result_carries_the_query_not_the_concatenation() -> None:
    store = FakeStore(dense=[("a", 0.9)], sparse=[])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), sparse_backend="lexical"
    )

    assert retriever.search_fused("the question", ["earlier"], k=5).query == "the question"


def test_diagnostics_report_the_realised_pool_and_the_fusion_stages() -> None:
    """`candidate_pool_size` is the REALISED fused pool, not the configured candidate_k.

    The benchmark learned this distinction the hard way: its `pool_bound()` overstated the
    realised pool by 3x on 307 of 777 queries and no score in the output revealed it.
    """
    store = FakeStore(dense=[("a", 0.9), ("b", 0.8)], sparse=[("c", 0.4)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), candidate_k=50,
        sparse_backend="lexical",
    )

    diagnostics = retriever.search_fused("q", ["earlier"], k=5).diagnostics

    assert diagnostics.reranking_ran is True
    assert diagnostics.candidate_pool_size == 3  # a, b, c: realised, not 50
    assert "history_retrieval" in diagnostics.stage_ms
    assert "outer_fusion" in diagnostics.stage_ms
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_search_fused.py -q`
Expected: FAIL with `NotImplementedError: fusion lands in Task 5`.

- [ ] **Step 3: Implement the fusion**

Replace `raise NotImplementedError("fusion lands in Task 5")` with:

```python
        primary = self._retrieve_legs(query, source)
        started = time.perf_counter()
        secondary = self._retrieve_legs(history_query, source, report_vec=primary.qvec)
        history_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()

        def _inner(legs: _Legs) -> list[str]:
            scores = _rrf(
                [
                    [h.chunk.id for h in legs.dense],
                    [h.chunk.id for h in legs.sparse],
                    [h.chunk.id for h in legs.learned],
                ]
            )
            return sorted(scores, key=lambda cid: scores[cid], reverse=True)

        # Nested, not flat: contrast T1 found the two topologies indistinguishable on R@100 with
        # flat nominally ahead, but the arm that was MEASURED and won is the nested one. Shipping
        # the nominally better arm from a non-significant contrast is reading noise.
        outer = _rrf([_inner(primary), _inner(secondary)])
        ranked_ids = sorted(outer, key=lambda cid: outer[cid], reverse=True)
        realised_pool = len(ranked_ids)
        ranked_ids = ranked_ids[:FUSED_RERANK_POOL_CAP]

        # `primary` first so that where both variants surfaced a chunk, the hit carrying the
        # QUERY's cosine wins.
        by_id: dict[str, object] = {}
        for legs in (primary, secondary):
            for group in (legs.dense, legs.sparse, legs.learned):
                for hit in group:
                    by_id.setdefault(hit.chunk.id, hit)
        dense_score = {h.chunk.id: h.score for h in primary.dense}
        hits = [_rescored(by_id[cid], dense_score.get(cid, by_id[cid].score))  # type: ignore[arg-type]
                for cid in ranked_ids]
        outer_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        # Bound locally so mypy keeps the narrowing from the refusal above. Deliberately NOT an
        # `assert`: asserts are stripped under `python -O`, which would turn the guaranteed
        # ValueError into an AttributeError exactly when optimisation is on.
        reranker = self._reranker
        if reranker is None:  # pragma: no cover - the refusal above already returned
            raise ValueError("search_fused requires a reranker")
        hits = reranker.rerank(query, hits)
        rerank_ms = (time.perf_counter() - started) * 1000.0
        hits = hits[:k]

        # Put every RETURNED hit on one basis. Only the returned ones: <= k rows on a primary key
        # lookup, so the extra round trip is small, and hits below the cut are never reported.
        started = time.perf_counter()
        fresh = self._store.cosines_for([h.chunk.id for h in hits], primary.qvec)
        hits = [_rescored(h, fresh.get(h.chunk.id, h.score)) for h in hits]
        rescore_ms = (time.perf_counter() - started) * 1000.0

        timings = dict(primary.timings)
        timings["history_retrieval"] = history_ms
        timings["outer_fusion"] = outer_ms
        timings["reranking"] = rerank_ms
        timings["rescore"] = rescore_ms

        gap = gap_warning(list(dense_score.values()), self._gap_threshold)
        stale = staleness(
            self._store.newest_indexed_at(), datetime.now(timezone.utc), self._max_age
        )
        return RetrievalResult(
            query=query,
            hits=hits,
            gap_warning=gap,
            staleness=stale,
            diagnostics=RetrievalDiagnostics(
                embedding_profile=embedding_profile_id(self._embedder),
                retrieval_profile=self._retrieval_profile,
                index_generation=self._index_generation,
                candidate_pool_size=realised_pool,
                reranking_ran=True,
                stage_ms={key: round(value, 3) for key, value in timings.items()},
            ),
        )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_search_fused.py -q`
Expected: PASS, 9 tests.

- [ ] **Step 5: Confirm `search()` is untouched**

Run: `python -m pytest tests/ -q -k "retriev or trust or cost_surface or latency"`
Expected: all pass.

- [ ] **Step 6: mypy and commit**

```bash
python -m mypy
git add recall/retriever.py tests/test_search_fused.py
git commit -m "feat(retriever): fused search with the measured pool cap and a single score basis"
```

---

### Task 6: Benchmark parity

The gate that answers *is the served system the measured system?* Without it, `search_fused` can drift from `mq_nested2_nogold` and still cite its numbers.

**Files:**
- Create: `tests/fixtures/fused_parity.json`
- Test: `tests/test_fused_parity.py` (create)

**Interfaces:**
- Consumes: `HybridRetriever.search_fused`, `benchmarks.mtrag.multiquery.fuse_arm`, `POST_HOC_ARMS`.
- Produces: nothing.

- [ ] **Step 1: Generate the fixture**

Run this from the repo root and save the output to `tests/fixtures/fused_parity.json`:

```bash
mkdir -p tests/fixtures
python - <<'PY' > tests/fixtures/fused_parity.json
import json, random
from benchmarks.mtrag.multiquery import POST_HOC_ARMS, fuse_arm

rng = random.Random(20260807)
arm = next(a for a in POST_HOC_ARMS if a.name == "mq_nested2_nogold")
cases = []
for case in range(4):
    legs = {
        variant: {
            "dense": [f"{variant}_d{i}" if rng.random() < 0.6 else f"shared{i}" for i in range(12)],
            "splade": [f"{variant}_s{i}" if rng.random() < 0.6 else f"shared{i}" for i in range(12)],
        }
        for variant in ("last", "full")
    }
    cases.append({"legs": legs, "expected": fuse_arm(arm, legs)})
print(json.dumps({"arm": arm.name, "cases": cases}, indent=2))
PY
```

Verify it is valid and non-trivial: `python -c "import json;d=json.load(open('tests/fixtures/fused_parity.json'));print(d['arm'], len(d['cases']), len(d['cases'][0]['expected']))"`
Expected: `mq_nested2_nogold 4` and an expected list longer than 12.

- [ ] **Step 2: Write the failing test**

Create `tests/test_fused_parity.py`:

```python
"""Is the SERVED system the MEASURED system?

`search_fused` cites +0.0084 nDCG@5 and +0.0842 R@100 from `mq_nested2_nogold`. If the retriever's
outer fusion ever diverges from the benchmark's, those numbers describe something the code no
longer computes, and nothing else in the suite would notice. This is the serving analogue of the
`validate` gate the benchmark itself had to have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.retriever import HybridRetriever
from tests.fakes import FakeEmbedder, FakeStore

FIXTURE = Path(__file__).parent / "fixtures" / "fused_parity.json"


class _PassThroughReranker:
    """Preserves the fused order, so the test compares FUSION rather than reranking."""

    def rerank(self, query, hits):
        return list(hits)


class _ScriptedStore(FakeStore):
    """Returns the fixture's leg rankings, selecting the variant per call.

    ⚠️ Keyed on the query VECTOR, not on a variant set by an earlier call. `_retrieve_legs` calls
    `query_dense` BEFORE `query_sparse`, so a store that selected its variant in `query_sparse`
    and read it in `query_dense` would read a stale value and silently serve the wrong legs.
    """

    def __init__(self, legs: dict, qvec_last: list[float], qvec_full: list[float]) -> None:
        super().__init__()
        self._by_vec = {tuple(qvec_last): legs["last"], tuple(qvec_full): legs["full"]}
        self._by_text: dict[str, dict] = {}

    def bind_text(self, query: str, history: str, legs: dict) -> None:
        self._by_text = {query: legs["last"], history: legs["full"]}

    def _rows(self, ids: list[str]) -> list:
        return self._hits([(cid, 0.5) for cid in ids])

    def query_dense(self, vector, k, source=None):
        return self._rows(self._by_vec[tuple(vector)]["dense"][:k])

    def query_sparse(self, text, k, source=None, vec=None):
        self.sparse_vec_used = vec
        return self._rows(self._by_text[text]["splade"][:k])


@pytest.mark.parametrize("index", range(4))
def test_serving_fusion_reproduces_the_benchmark_arm(index: int) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = data["cases"][index]
    query, history_turn = "the question", "an earlier turn of different length"

    embedder = FakeEmbedder()
    qvec_last = embedder.embed([query])[0]
    qvec_full = embedder.embed([history_turn])[0]
    assert qvec_last != qvec_full, (
        "the fixture needs two distinguishable query vectors; adjust the strings' lengths"
    )

    store = _ScriptedStore(case["legs"], qvec_last, qvec_full)
    store.bind_text(query, history_turn, case["legs"])

    retriever = HybridRetriever(
        store, embedder, reranker=_PassThroughReranker(), sparse_backend="lexical",
        candidate_k=100,
    )

    result = retriever.search_fused(query, [history_turn], k=100)

    assert [h.chunk.id for h in result.hits] == case["expected"][: len(result.hits)]
```

- [ ] **Step 3: Run it**

Run: `python -m pytest tests/test_fused_parity.py -q`

If it fails on **ordering**, the retriever's fusion genuinely differs from the benchmark's: fix `search_fused`, not the test. Do not weaken the assertion to make it pass.

If it fails with `KeyError` on a vector, the two strings embed identically under `FakeEmbedder` (its vector depends on `len(text) % 7`): change one string's length and regenerate nothing, the fixture is independent of the strings.

- [ ] **Step 4: Verify the test can fail**

Temporarily change the outer fusion in `recall/retriever.py` from
`_rrf([_inner(primary), _inner(secondary)])` to `_rrf([_inner(primary)])`, re-run
`python -m pytest tests/test_fused_parity.py -q`, and confirm it goes RED. Then revert.

A parity test that passes when the fusion is removed is worse than no test: this session produced two guards that could not fire, and only a mutation check caught either.

- [ ] **Step 5: Commit**

```bash
python -m mypy
git add tests/fixtures/fused_parity.json tests/test_fused_parity.py
git commit -m "test(retriever): pin served fusion against the measured benchmark arm"
```

---

### Task 7: The degenerate invariant and the documented cap boundary

**Files:**
- Test: `tests/test_search_fused.py` (extend)

**Interfaces:**
- Consumes: `search`, `search_fused`, `FUSED_RERANK_POOL_CAP`.
- Produces: nothing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_fused.py`:

```python
def test_history_identical_to_the_query_matches_plain_search_below_the_cap() -> None:
    """Nested RRF over two copies of one ranking is order-preserving.

    The same structural property the benchmark's 102 turn-1 queries rested on: identical text
    gives identical leg rankings, and fusing a ranking with itself cannot reorder it.
    """
    rows = [(f"d{i}", 0.9 - i * 0.01) for i in range(10)]
    store = FakeStore(dense=rows, sparse=[(f"s{i}", 0.4) for i in range(10)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), candidate_k=20,
        sparse_backend="lexical",
    )

    plain = [h.chunk.id for h in retriever.search("q", k=10).hits]
    fused = [h.chunk.id for h in retriever.search_fused("q", ["q"], k=10).hits]

    assert fused == plain


def test_above_the_cap_the_two_legitimately_diverge() -> None:
    """Asserted rather than hidden.

    `search` reranks its whole pool; `search_fused` caps at 100 because reranking a wider pool was
    measured to LOSE coverage. Above the cap the invariant above does not hold, and pretending it
    did would be an untrue claim in a passing test.
    """
    from recall.retriever import FUSED_RERANK_POOL_CAP

    rows = [(f"d{i}", 0.9) for i in range(FUSED_RERANK_POOL_CAP + 40)]
    store = FakeStore(dense=rows, sparse=[])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(),
        candidate_k=FUSED_RERANK_POOL_CAP + 40, sparse_backend="lexical",
    )

    plain = retriever.search("q", k=1000).hits
    fused = retriever.search_fused("q", ["q"], k=1000).hits

    assert len(plain) > FUSED_RERANK_POOL_CAP
    assert len(fused) == FUSED_RERANK_POOL_CAP
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_search_fused.py -q`
Expected: PASS, 11 tests.

If the first test fails, the outer fusion is perturbing an order it should preserve: fix `search_fused`.

- [ ] **Step 3: Commit**

```bash
python -m mypy
git add tests/test_search_fused.py
git commit -m "test(retriever): pin the degenerate invariant and the cap boundary it stops at"
```

---

### Task 8: Documentation and final verification

**Files:**
- Modify: `README.md` (retrieval section), `CHANGELOG.md`
- Test: full suite

- [ ] **Step 1: Find the right README section**

Run: `grep -n "HybridRetriever\|## Retrieval\|rerank" README.md | head -20` and add the new subsection next to the existing retrieval documentation.

- [ ] **Step 2: Write the README entry**

```markdown
### Multi-query fusion (`search_fused`)

Fuses the current turn with a concatenation of prior turns. Measured on MTRAG-human dev at
`candidate_k=100` with a reranker: **+0.0084 nDCG@5** (Holm-significant) and **+0.0842 R@100**
over single-query `search`, under two cross-encoders 25x apart in size.

```python
result = retriever.search_fused("and what about the deadline?", history=["what is the policy?"])
```

⚠️ **Requires a reranker and refuses without one.** Raw, this arm scores **0.0447 nDCG@5 below**
`search()`; the cross-encoder repairs the ranking damage a concatenated query does. It also costs
roughly **2x the retrieval** plus reranking (~1,050 ms/query on CPU), so it is opt-in: no
`history`, no fusion, and every existing `search()` call is unaffected.

Histories whose concatenation exceeds 4,096 characters are **refused, not truncated**: a truncated
history is a configuration that was never measured.
```

- [ ] **Step 3: Add the CHANGELOG entry**

Follow the file's existing format, under the unreleased heading:

```markdown
- `HybridRetriever.search_fused(query, history, k, source)`: multi-query fusion of the current turn
  with prior turns. +0.0084 nDCG@5 and +0.0842 R@100 measured on MTRAG-human dev. Requires a
  reranker and refuses without one, because the gain is conditional on reranking. Adds
  `PgVectorStore.cosines_for`. `search()` is unchanged.
```

- [ ] **Step 4: Full verification**

```bash
python -m pytest tests/ -q
python -m mypy
python -m pyflakes recall/retriever.py recall/store.py
```

Expected: suite green (Postgres-dependent tests may error locally if no database is running; they run in CI), `Success: no issues found`, and no pyflakes output.

- [ ] **Step 5: Commit and push**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: search_fused, and the reranker condition it enforces"
git push origin claude/handoff-query-diversity-1be84c
```

Push by BRANCH NAME, never `HEAD`: a concurrent session moving refs makes `HEAD` stale, and the push reports success while publishing the wrong commit.

---

## Out of scope

- **MCP exposure.** Adding `history` to a public tool surface with its own auth, limits and query-length contract needs its own spec.
- **LLM reformulations.** Withdrawn on evidence: contrast C3 found the gold rewrite an LLM would approximate contributes nothing once reranked.
- **Flat fusion topology**, tuning `candidate_k`, and any change to `search()`'s behaviour.
