# Selectable SPLADE arm + learned sparse indexing as a library operation: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `benchmarks/store_latency_share.py --sparse-backend splade` actually run against a real SPLADE checkpoint, by shipping corpus-to-sidecar encoding as a supported `recall` operation, and then measure both arms on VPS2.

**Architecture:** One batched encode-and-upsert primitive in `recall/sparse.py`, reached two ways: an optional `sparse_encoder` on `Indexer` (new corpora) and `backfill_learned_sparse` streaming `store.iter_chunks()` (existing corpora, which is every corpus today). A corpus-level `assert_sparse_coverage` is the guard that turns a silent partial write into a refusal. The benchmark gains appendable `--sparse-backend` so both arms run against one store and one encode.

**Tech Stack:** Python 3.12, Postgres + pgvector (`sparsevec`), `transformers` + `torch` (the `sparse` extra), pytest.

**Spec:** `docs/superpowers/specs/2026-08-07-splade-cli-arm-design.md`

## Global Constraints

- **Worktree:** `C:/Users/gde00/Documents/recall-spladecli`, branch `feat/splade-cli-arm`, based on `origin/master` `4983f44`. Do not work in `/Documents/recall`, `/Documents/recall-splade`, or any `/opt/recall-*` tree on VPS2: those belong to other lanes.
- **Test interpreter:** `/c/Users/gde00/Documents/recall/.venv/Scripts/python.exe`, invoked as `python -m pytest` **from the worktree root**, so the local `recall/` package shadows any installed copy. Verified: `import recall` resolves to `C:\Users\gde00\Documents\recall-spladecli\recall\__init__.py`.
- **Test database:** `RECALL_TEST_DSN`, defaulting to the local dev DSN on `localhost:5432`. Tests marked `@requires_db` skip when it is unreachable. That container is shared with other sessions; every test here uses the `make_store` fixture, which creates and drops a uuid-named table.
- **`torch` and `transformers` are imported inside functions, never at module scope** in `recall/sparse.py`, so a lexical-only install never needs the `sparse` extra. Tests use a hand-written deterministic encoder, not a checkpoint: no 500 MB download, no network.
- **`SPARSE_TABLE = "recall_sparse_v1"`**, `SPARSE_DIM = 30522`, `SPARSE_MAX_NONZERO = 1000`, all in `recall/store.py`.
- **No dash as punctuation** in any prose written to this repository by this plan (comments, docstrings, markdown). Hyphens inside identifiers and inside verbatim quotations are fine.
- **Every new guard is shown failing before it is shown passing.** A test written after the change and never run red is a hypothesis, not a guard.

---

### Task 1: `store_sparse_vectors`, the one encode-and-upsert loop

**Files:**
- Modify: `recall/sparse.py` (append after `SpladeEncoder`)
- Test: `tests/test_sparse_indexing.py` (create)

**Interfaces:**
- Consumes: `PgVectorStore.upsert_sparse(profile_id, vectors)`, `SpladeEncoder.profile.profile_id` (both exist).
- Produces: `SparseIndexResult(written: int, empty_ids: list[str])` and `store_sparse_vectors(store, encoder, items, *, batch_size=32, progress=None) -> SparseIndexResult`. Tasks 3, 4 and 7 call it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sparse_indexing.py`:

```python
"""Corpus to learned sparse sidecar, as a library operation rather than two offline scripts."""

from __future__ import annotations

import pytest

from recall.sparse import (
    SparseIndexResult,
    SparseProfile,
    assert_sparse_coverage,
    backfill_learned_sparse,
    store_sparse_vectors,
)
from recall.types import Chunk
from tests.conftest import requires_db

PROFILE_ID = "kw-index-test"


class KeywordSparseEncoder:
    """A real, deterministic encoder: one term id per known word.

    Not a mock of the system under test. It implements the same encoder protocol the production
    path depends on, and is chosen over a SPLADE checkpoint so this file needs no download and
    no network. The store path it drives is the production one.
    """

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self._vocabulary = vocabulary
        self.batches: list[int] = []
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        self.batches.append(len(texts))
        return [
            {self._vocabulary[w]: 1.0 for w in text.lower().split() if w in self._vocabulary}
            for text in texts
        ]


class StubEmbedder:
    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source=f"/c/{cid}.md", text=text, metadata={"file": f"{cid}.md"})


@requires_db
def test_store_sparse_vectors_writes_under_the_encoders_own_profile(make_store) -> None:
    """The profile id is READ OFF the encoder, never passed alongside it.

    Filing vectors under a name a different model produced is the failure the profile column
    exists to prevent, and it produces plausible scores rather than an error.
    """
    store = make_store(64)
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9})

    result = store_sparse_vectors(
        store, encoder, [("a", "aardvark"), ("b", "beta")], batch_size=1
    )

    assert isinstance(result, SparseIndexResult)
    assert result.written == 2
    assert result.empty_ids == []
    assert store.sparse_row_count(PROFILE_ID) == 2
    # batch_size is honoured, not merely accepted: two calls of one text each.
    assert encoder.batches == [1, 1]


@requires_db
def test_a_term_free_chunk_is_skipped_and_named_rather_than_fatal(make_store) -> None:
    """One punctuation-only chunk must not kill a 20,000 chunk index.

    `upsert_sparse` refuses an empty weights mapping outright, and it is right to: the table's
    CHECK requires nnz > 0 and an all-empty run means a broken encoder. So the decision splits by
    level. Here, at the ROW level, the empty vector is skipped and its id recorded. The refusal
    lives at the CORPUS level, in `assert_sparse_coverage`, where an operator can act on it.
    """
    store = make_store(64)
    encoder = KeywordSparseEncoder({"aardvark": 7})

    result = store_sparse_vectors(store, encoder, [("a", "aardvark"), ("b", "!!! ???")])

    assert result.written == 1
    assert result.empty_ids == ["b"]
    assert store.sparse_row_count(PROFILE_ID) == 1


@requires_db
def test_progress_reports_the_running_written_count(make_store) -> None:
    """A silent twenty minute CPU encode is indistinguishable from a hang."""
    store = make_store(64)
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9, "gamma": 11})
    seen: list[int] = []

    store_sparse_vectors(
        store, encoder,
        [("a", "aardvark"), ("b", "beta"), ("c", "gamma")],
        batch_size=2, progress=seen.append,
    )

    assert seen == [2, 3]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_indexing.py -v
```

Expected: collection error, `ImportError: cannot import name 'SparseIndexResult' from 'recall.sparse'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `recall/sparse.py`, after the `SpladeEncoder` class. Add `from collections.abc import Callable, Iterable` to the imports at the top of the file, and `from typing import TYPE_CHECKING, Any, Protocol`.

```python
class SparseEncoderProtocol(Protocol):
    """What the indexing helpers actually need from an encoder.

    Stated as a protocol rather than as `SpladeEncoder`, because the tests drive these helpers
    with a deterministic keyword encoder and that is a feature: it keeps the corpus path testable
    without a 500 MB download, and it keeps `torch` out of a lexical-only install.
    """

    @property
    def profile(self) -> SparseProfile: ...

    def encode(self, texts: list[str]) -> list[dict[int, float]]: ...


@dataclass(frozen=True)
class SparseIndexResult:
    """What one indexing pass wrote, and what it could not write.

    `empty_ids` is not a warning to be discarded. It is the ONLY explanation an operator will get
    for `assert_sparse_coverage` finding fewer sidecar rows than chunks, so it is returned rather
    than logged.
    """

    written: int
    empty_ids: list[str]


def store_sparse_vectors(
    store: Any,
    encoder: SparseEncoderProtocol,
    items: Iterable[tuple[str, str]],
    *,
    batch_size: int = 32,
    progress: Callable[[int], None] | None = None,
) -> SparseIndexResult:
    """Encode `(chunk_id, text)` pairs and write them to the learned sparse sidecar.

    The profile id is read off `encoder.profile`, never taken as a separate argument. Vectors
    filed under a name a different model produced score plausibly instead of failing, which is
    precisely what the profile column exists to prevent, so the caller is not given the chance.

    A chunk that encodes to an EMPTY vector is skipped and its id returned. `upsert_sparse`
    refuses an empty mapping (the table's CHECK requires nnz > 0), and it is right to, but that
    refusal belongs at the corpus level where an operator can act on it: see
    `assert_sparse_coverage`. One term-free passage must not kill a whole index.

    `progress` receives the running written count after each batch, so a caller can print
    something during a CPU encode that takes tens of minutes.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}; nothing would be encoded")

    profile_id = encoder.profile.profile_id
    written = 0
    empty_ids: list[str] = []
    batch: list[tuple[str, str]] = []

    def _flush_batch() -> None:
        nonlocal written
        if not batch:
            return
        vectors = encoder.encode([text for _, text in batch])
        payload: dict[str, dict[int, float]] = {}
        for (chunk_id, _text), weights in zip(batch, vectors, strict=True):
            if weights:
                payload[chunk_id] = weights
            else:
                empty_ids.append(chunk_id)
        if payload:
            written += store.upsert_sparse(profile_id, payload)
        batch.clear()
        if progress is not None:
            progress(written)

    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            _flush_batch()
    _flush_batch()

    return SparseIndexResult(written=written, empty_ids=empty_ids)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_indexing.py -v -k "store_sparse_vectors or term_free or progress"
```

Expected: 3 passed. (The other tests in the file still fail on import until Tasks 2 and 3 land; if collection blocks, comment out the `assert_sparse_coverage` and `backfill_learned_sparse` imports for this step and restore them in Task 2.)

- [ ] **Step 5: Commit**

```bash
git add recall/sparse.py tests/test_sparse_indexing.py
git commit -m "feat(sparse): encode a corpus into the sidecar as a library operation

Until now this existed only as two offline scripts passing a JSONL file
between them. That split is right for rented hardware, where the encoder
must never hold a database credential, and wrong for one process holding
a store.

The empty-vector decision splits by level: skipped at the row level
(one term-free passage must not kill a 20k index), refused at the corpus
level where an operator can act on it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `assert_sparse_coverage`, the guard that has to fire

**Files:**
- Modify: `recall/sparse.py`
- Test: `tests/test_sparse_indexing.py`

**Interfaces:**
- Consumes: `PgVectorStore.sparse_row_count(profile_id)`, `PgVectorStore.count()` (both exist).
- Produces: `assert_sparse_coverage(store, profile_id, *, empty_ids=()) -> None`, raising `SparseCoverageError`. Tasks 4 and 7 call it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sparse_indexing.py`:

```python
@requires_db
def test_coverage_passes_when_every_chunk_is_encoded(make_store) -> None:
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "beta")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9})
    store_sparse_vectors(store, encoder, [("a", "aardvark"), ("b", "beta")])

    assert_sparse_coverage(store, PROFILE_ID)  # must not raise


@requires_db
def test_coverage_refuses_a_half_encoded_corpus(make_store) -> None:
    """"The corpus is half encoded" is a state that exists, and a query over it returns
    plausible thin results rather than an error. So it is refused here, loudly, with both counts
    in the message."""
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "beta")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7})
    store_sparse_vectors(store, encoder, [("a", "aardvark")])

    with pytest.raises(SparseCoverageError, match="1 of 2"):
        assert_sparse_coverage(store, PROFILE_ID)


@requires_db
def test_coverage_names_the_empty_chunks_as_the_explanation(make_store) -> None:
    """A shortfall fully explained by term-free passages still refuses, but says WHY.

    An operator who knows the two missing chunks are punctuation can proceed. One who is told
    only "1 of 2" cannot tell that from a broken encoder.
    """
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "!!!")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7})
    result = store_sparse_vectors(store, encoder, [("a", "aardvark"), ("b", "!!!")])

    with pytest.raises(SparseCoverageError, match="encoded to an empty vector: b"):
        assert_sparse_coverage(store, PROFILE_ID, empty_ids=result.empty_ids)
```

Add `SparseCoverageError` to the imports at the top of the file.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_indexing.py -v -k coverage
```

Expected: collection error, `ImportError: cannot import name 'SparseCoverageError'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `recall/sparse.py`:

```python
class SparseCoverageError(RuntimeError):
    """Fewer sidecar rows than chunks. The retrieval leg would answer, thinly and silently."""


def assert_sparse_coverage(
    store: Any, profile_id: str, *, empty_ids: "Iterable[str]" = ()
) -> None:
    """Refuse a corpus whose sidecar is not complete under `profile_id`.

    This is the corpus-level half of the empty-vector decision made in `store_sparse_vectors`,
    and it is the reason skipping a row there is safe. A partially encoded corpus does not error
    on query: the learned leg simply retrieves from the fraction that exists, and the result is
    indistinguishable from a corpus where those passages genuinely did not match.

    `empty_ids` is what `store_sparse_vectors` returned. It does not suppress the refusal, it
    EXPLAINS it: an operator who can see that the missing chunks were term-free can proceed,
    where "1 of 2" alone cannot be told apart from a broken encoder.
    """
    encoded = store.sparse_row_count(profile_id)
    total = store.count()
    if encoded == total:
        return
    message = (
        f"learned sparse sidecar holds {encoded} of {total} chunks under profile "
        f"{profile_id!r}. A query would retrieve from the encoded fraction and report nothing, "
        f"so no result from this corpus may be quoted."
    )
    named = list(empty_ids)
    if named:
        shown = ", ".join(named[:10])
        more = f" (and {len(named) - 10} more)" if len(named) > 10 else ""
        message += f" {len(named)} chunk(s) encoded to an empty vector: {shown}{more}."
    raise SparseCoverageError(message)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_indexing.py -v -k coverage
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add recall/sparse.py tests/test_sparse_indexing.py
git commit -m "feat(sparse): refuse a half-encoded corpus, and say which chunks were empty

A partially encoded sidecar does not error on query. The learned leg
retrieves from the fraction that exists and the result is
indistinguishable from passages that genuinely did not match.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `backfill_learned_sparse`, for corpora that already exist

**Files:**
- Modify: `recall/sparse.py`
- Test: `tests/test_sparse_indexing.py`

**Interfaces:**
- Consumes: `PgVectorStore.iter_chunks(batch_size)` (exists; server-side cursor, excludes the dense vector), `store_sparse_vectors` from Task 1.
- Produces: `backfill_learned_sparse(store, encoder, *, batch_size=32, progress=None) -> SparseIndexResult`. Task 7 calls it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sparse_indexing.py`:

```python
@requires_db
def test_backfill_encodes_a_corpus_that_was_already_indexed(make_store) -> None:
    """The case that actually matters: every corpus in existence today is already indexed.

    `Indexer` has never written the sidecar, so there is no corpus anywhere whose learned sparse
    vectors were written at index time. The backfill is the only path that can reach them, and it
    is what `benchmarks/store_latency_share.py` calls after `_throwaway_store` has already run.
    """
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "beta")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9})

    result = backfill_learned_sparse(store, encoder)

    assert result.written == 2
    assert_sparse_coverage(store, PROFILE_ID, empty_ids=result.empty_ids)


@requires_db
def test_backfill_is_idempotent(make_store) -> None:
    """`upsert_sparse` is ON CONFLICT DO UPDATE, so a re-run re-encodes rather than duplicating.

    Deliberately NOT resumable: skipping ids already present would need a new
    `store.sparse_ids(profile_id)`, and at the corpus sizes this serves that buys nothing.
    """
    store = make_store(64)
    store.upsert([_chunk("a", "aardvark")], [[0.1] * 64])
    encoder = KeywordSparseEncoder({"aardvark": 7})

    backfill_learned_sparse(store, encoder)
    backfill_learned_sparse(store, encoder)

    assert store.sparse_row_count(PROFILE_ID) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_indexing.py -v -k backfill
```

Expected: `ImportError: cannot import name 'backfill_learned_sparse'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `recall/sparse.py`:

```python
def backfill_learned_sparse(
    store: Any,
    encoder: SparseEncoderProtocol,
    *,
    batch_size: int = 32,
    progress: Callable[[int], None] | None = None,
) -> SparseIndexResult:
    """Encode every chunk already in `store` into the learned sparse sidecar.

    This is the path that reaches corpora indexed before `Indexer` could write the sidecar, which
    is every corpus that exists today. It streams `store.iter_chunks()`, a server-side cursor
    that excludes the dense vector, so a corpus larger than memory is fine.

    IDEMPOTENT, not resumable. `upsert_sparse` is ON CONFLICT DO UPDATE, so re-invoking simply
    re-encodes. Skipping ids already present would need a `store.sparse_ids(profile_id)` this
    store does not have, and at the corpus sizes this serves it would buy nothing. That is a
    decision, not an oversight.
    """
    return store_sparse_vectors(
        store,
        encoder,
        ((chunk.id, chunk.text) for chunk in store.iter_chunks(batch_size=max(batch_size, 1))),
        batch_size=batch_size,
        progress=progress,
    )
```

- [ ] **Step 4: Run the whole file to verify it passes**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_indexing.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add recall/sparse.py tests/test_sparse_indexing.py
git commit -m "feat(sparse): backfill the sidecar for an already-indexed corpus

Every corpus in existence today was indexed before Indexer could write
the sidecar, so the backfill is the only path that reaches them, and it
is what store_latency_share.py needs after _throwaway_store has run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `Indexer(sparse_encoder=...)`, and the test that goes red first

**Files:**
- Modify: `recall/index.py:351-394` (constructor), `recall/index.py:654-713` (`_flush`)
- Test: `tests/test_indexer_sparse_hook.py` (create)

**Interfaces:**
- Consumes: `store_sparse_vectors` (Task 1), `assert_sparse_coverage` (Task 2).
- Produces: `Indexer(store, embedder, ..., sparse_encoder=None)`. No later task depends on it; the benchmark uses the backfill.

**Why the hook goes in `_flush` and not at a call site.** `_flush` is called twice (`index.py:610` and `index.py:617`), and it is where `replace_sources` actually executes. Hooking it there means neither call site can be forgotten independently. `_flush` currently has two `return len(chunks)` statements, one per branch; Step 3 collapses them into one so the sparse write cannot be attached to only one branch.

**Why the second test exists.** `Indexer` already carries an optional secondary write target, `shadow`, and it failed silently once: `b0e74e5` (PR #218), "attaching a shadow to an indexed corpus wrote nothing to the shadow". Every active fingerprint matched, every file hit `continue`, and the shadow write lived past it. The run reported success with a skipped count. Any second write hooked into this loop inherits that shape, so the re-index path is tested explicitly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_indexer_sparse_hook.py`:

```python
"""`Indexer(sparse_encoder=...)`: does the learned sparse sidecar actually get written?

Both assertions below are ROW COUNTS, not call counts. `Indexer`'s other optional secondary
write, `shadow`, passed every structural review and wrote nothing (b0e74e5, PR #218): every
active fingerprint matched, every file was skipped, and the shadow write lived past the
`continue`. The run reported success. A call count would not have caught that; a row count does.
"""

from __future__ import annotations

from pathlib import Path

from recall.index import Indexer
from recall.sparse import SparseProfile, assert_sparse_coverage
from tests.conftest import requires_db

PROFILE_ID = "kw-hook-test"


class KeywordSparseEncoder:
    def __init__(self, vocabulary: dict[str, int]) -> None:
        self._vocabulary = vocabulary
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [
            {self._vocabulary[w]: 1.0 for w in text.lower().split() if w in self._vocabulary}
            for text in texts
        ]


class StubEmbedder:
    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "one.md").write_text("aardvark facts here\n", encoding="utf-8")
    (root / "two.md").write_text("beta prose here\n", encoding="utf-8")
    return root


VOCAB = {"aardvark": 7, "beta": 9, "facts": 11, "prose": 13, "here": 15}


@requires_db
def test_indexing_with_a_sparse_encoder_fills_the_sidecar(make_store, tmp_path) -> None:
    store = make_store(64)
    encoder = KeywordSparseEncoder(VOCAB)

    Indexer(store, StubEmbedder(), sparse_encoder=encoder).index_path(_corpus(tmp_path))

    assert store.count() > 0
    assert store.sparse_row_count(PROFILE_ID) == store.count()
    assert_sparse_coverage(store, PROFILE_ID)


@requires_db
def test_attaching_a_sparse_encoder_to_an_indexed_corpus_fills_the_sidecar(
    make_store, tmp_path
) -> None:
    """The exact sequence the shadow dual-write got wrong.

    Index the corpus with no encoder, then attach one and re-index. Every dense fingerprint still
    matches, so every file is a candidate for `continue`, and a sparse write placed past that
    `continue` would leave the sidecar empty while the run reported success with a skipped count.
    """
    store = make_store(64)
    root = _corpus(tmp_path)
    Indexer(store, StubEmbedder()).index_path(root)
    assert store.sparse_row_count(PROFILE_ID) == 0

    Indexer(store, StubEmbedder(), sparse_encoder=KeywordSparseEncoder(VOCAB)).index_path(root)

    assert store.sparse_row_count(PROFILE_ID) == store.count()
```

- [ ] **Step 2: Run the tests against the un-integrated `Indexer` and watch them go red**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_indexer_sparse_hook.py -v
```

Expected: both FAIL with `TypeError: Indexer.__init__() got an unexpected keyword argument 'sparse_encoder'`. **Record this output.** If either test passes here, stop: it is not testing what it claims to.

- [ ] **Step 3: Add the constructor parameter**

In `recall/index.py`, in `Indexer.__init__`, add the parameter after `shadow`:

```python
        shadow: ShadowIndexTarget | None = None,
        sparse_encoder: "SparseEncoderProtocol | None" = None,
    ) -> None:
```

and store it beside the other fields, next to `self._shadow = shadow`:

```python
        self._shadow = shadow
        #: Optional learned sparse (SPLADE) sidecar write, driven from `_flush`. Deliberately at
        #: the same site as the dense write rather than in the file loop: `shadow` sits in that
        #: loop, past a `continue` whose predicate did not know about it, and wrote nothing for a
        #: period (b0e74e5, PR #218). A second write hooked into that loop inherits the shape.
        self._sparse_encoder = sparse_encoder
```

Add the import at the top of `recall/index.py`:

```python
from recall.sparse import SparseEncoderProtocol, store_sparse_vectors
```

- [ ] **Step 4: Collapse `_flush`'s two returns into one and hook the sparse write**

In `recall/index.py`, in `_flush`, replace:

```python
        if self._shadow is None:
            self._store.replace_sources(sources, chunks, embeddings)
            return len(chunks)
        if shadow_chunks is None or shadow_embedding_texts is None:
```

with:

```python
        if self._shadow is None:
            self._store.replace_sources(sources, chunks, embeddings)
            return self._write_sparse(chunks)
        if shadow_chunks is None or shadow_embedding_texts is None:
```

and replace the final line of `_flush`:

```python
        self._shadow.control_plane.complete_event(
            self._store.tenant, operation_id, len(shadow_chunks)
        )
        return len(chunks)
```

with:

```python
        self._shadow.control_plane.complete_event(
            self._store.tenant, operation_id, len(shadow_chunks)
        )
        return self._write_sparse(chunks)
```

Then add the helper immediately after `_flush`:

```python
    def _write_sparse(self, chunks: list[Chunk]) -> int:
        """Write this batch's learned sparse vectors, and return the batch size.

        Returning the count is what makes this hard to forget. Both of `_flush`'s branches end in
        `return self._write_sparse(chunks)`, so a future branch that returns a bare count is
        visibly different from the two beside it, and a branch that skips the sparse write cannot
        also return the number `_flush` is contracted to return.
        """
        if self._sparse_encoder is not None and chunks:
            store_sparse_vectors(
                self._store,
                self._sparse_encoder,
                [(chunk.id, chunk.text) for chunk in chunks],
            )
        return len(chunks)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_indexer_sparse_hook.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Run the existing indexer suite for regressions**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/ -v -k "index or shadow" 2>&1 | tail -30
```

Expected: no new failures against the pre-change baseline.

- [ ] **Step 7: Commit**

```bash
git add recall/index.py tests/test_indexer_sparse_hook.py
git commit -m "feat(index): optional learned sparse sidecar write during indexing

Hooked in _flush, beside the dense replace_sources, not in the file loop.
The file loop is where 'shadow' sits, past a continue whose predicate did
not know about it, and it wrote nothing for a period (b0e74e5, PR #218)
while the run reported success with a skipped count.

Both tests were run against the un-integrated Indexer and shown to fail.
The second one exercises exactly that sequence: index, then attach an
encoder and re-index, where every dense fingerprint still matches.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `drop_table` must not orphan sidecar rows

**Files:**
- Modify: `recall/store.py:942-967`
- Test: `tests/test_learned_sparse_store.py` (append)

**Interfaces:**
- Consumes: `SPARSE_TABLE` (module constant).
- Produces: nothing new. `drop_table()` gains a side effect.

**Why.** The sidecar is keyed `(tenant_id, chunk_table, profile_id, id)` and has no foreign key to the chunk table, which is a column VALUE rather than a relation. `drop_table` drops the chunk table and cleans the migration ledger and leaves the sidecar rows behind. Under `splade`, every `_throwaway_store` run in `benchmarks/store_latency_share.py` would orphan a uuid-named row set permanently.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_learned_sparse_store.py`:

```python
@requires_db
def test_dropping_the_table_removes_its_sidecar_rows(make_store) -> None:
    """The sidecar has no FOREIGN KEY, so nothing cascades on its behalf.

    `chunk_table` is a column VALUE, not a relation, so a dropped table leaves its sparse rows
    addressable by a name that no longer resolves. Every throwaway eval store would orphan a
    uuid-named row set, permanently, and nothing would ever look for them again.
    """
    store = make_store(64)
    store.upsert([Chunk(id="alpha", source="/c/a.md", text="a", metadata={})], [[0.1] * 64])
    store.upsert_sparse("drop-probe", {"alpha": {7: 1.0}})
    table = store.table
    assert store.sparse_row_count("drop-probe") == 1

    store.drop_table()

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        remaining = conn.execute(
            f"SELECT count(*) FROM {SPARSE_TABLE} WHERE chunk_table = %s", (table,)
        ).fetchone()
    assert remaining is not None and remaining[0] == 0
```

Ensure the file imports `psycopg`, `SPARSE_TABLE` from `recall.store`, `TEST_DSN` and `requires_db` from `tests.conftest`, and `Chunk` from `recall.types`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_learned_sparse_store.py::test_dropping_the_table_removes_its_sidecar_rows -v
```

Expected: FAIL, `assert 1 == 0`.

- [ ] **Step 3: Write the minimal implementation**

In `recall/store.py`, inside `drop_table`'s `_drop` function, add the DELETE inside the same transaction, before the `DROP TABLE`:

```python
        def _drop(conn: "psycopg.Connection") -> None:
            # Disposable eval/test tables may be recreated with the same name. Remove their
            # migration target atomically with the table; otherwise the ledger says every phase
            # is applied and the next explicit ensure_schema() correctly skips all SQL, leaving
            # the requested table absent.
            with conn.transaction():
                # The learned sparse sidecar cannot cascade: its parent is a column VALUE
                # (`chunk_table`), not a relation, so there is no foreign key to fire. Without
                # this DELETE every throwaway store leaves a uuid-named row set addressable by a
                # name that no longer resolves, and nothing ever looks for them again.
                sidecar = conn.execute(f"SELECT to_regclass('{SPARSE_TABLE}')").fetchone()
                if sidecar and sidecar[0]:
                    conn.execute(
                        f"DELETE FROM {SPARSE_TABLE} WHERE chunk_table = %s", (self._table,)
                    )
                conn.execute(f"DROP TABLE IF EXISTS {self._table}")
```

Leave the rest of `_drop` unchanged.

Note the deliberate absence of a `tenant_id` filter: `drop_table` removes the table for every tenant, so scoping the sidecar cleanup to the current tenant would leave other tenants' rows orphaned by the same drop.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_learned_sparse_store.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add recall/store.py tests/test_learned_sparse_store.py
git commit -m "fix(store): drop_table left the dropped table's sparse rows behind

The sidecar keys its parent as a column VALUE, not a relation, so there
is no foreign key to cascade. Every throwaway eval store orphaned a
uuid-named row set permanently, and the splade arm of
store_latency_share.py would have created one per run.

Not scoped to the current tenant: the drop is not either.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: the host quiescence guard

**Files:**
- Create: `recall/eval/hostload.py`
- Test: `tests/test_hostload.py` (create)

**Interfaces:**
- Produces: `read_load_per_core() -> float | None` and `assert_host_quiet(ceiling, *, allow_busy) -> float | None`, raising `HostTooBusyError`. Task 7 calls both.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hostload.py`:

```python
"""The quiescence guard. A latency artifact from a contended host is not a measurement."""

from __future__ import annotations

import pytest

from recall.eval import hostload
from recall.eval.hostload import HostTooBusyError, assert_host_quiet, read_load_per_core


def test_a_busy_host_is_refused(monkeypatch) -> None:
    """Shown FIRING, not shown running.

    VPS2 sat at load 33.7 on 12 cores while this was being designed. That is 2.8 per core, and
    every leg of a latency split measured there would carry queueing delay nobody can attribute.
    """
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: 2.81)

    with pytest.raises(HostTooBusyError, match="2.81"):
        assert_host_quiet(0.30, allow_busy=False)


def test_a_quiet_host_passes_and_returns_the_reading(monkeypatch) -> None:
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: 0.11)

    assert assert_host_quiet(0.30, allow_busy=False) == 0.11


def test_allow_busy_overrides_but_still_returns_the_reading(monkeypatch) -> None:
    """The override does not hide the number. The artifact still gets stamped with it."""
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: 2.81)

    assert assert_host_quiet(0.30, allow_busy=True) == 2.81


def test_an_unavailable_reading_does_not_refuse(monkeypatch) -> None:
    """`os.getloadavg` is Unix only. On Windows this guard CANNOT fire, and that is recorded
    rather than papered over: the field serialises as JSON null and the published artifact comes
    from Linux."""
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: None)

    assert assert_host_quiet(0.30, allow_busy=False) is None


def test_read_load_per_core_is_none_or_a_positive_float() -> None:
    """Runs on whatever host the suite runs on, so it asserts the CONTRACT, not a value."""
    value = read_load_per_core()

    assert value is None or (isinstance(value, float) and value >= 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_hostload.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'recall.eval.hostload'`.

- [ ] **Step 3: Write the minimal implementation**

Create `recall/eval/hostload.py`:

```python
"""Is this host quiet enough to time anything on?

A latency benchmark measures the machine as much as the code. VPS2 was at load average 33.7 on
12 cores when this was written, with several unrelated python processes and four Postgres
backends on it. Every leg of a split measured there carries queueing delay, in an amount nobody
can attribute afterwards, and the artifact would read as a property of the store.

So the reading is taken, recorded, and checked. Recorded even when it passes: an undated,
unloaded latency artifact cannot be compared against itself.
"""

from __future__ import annotations

import os

#: Default ceiling, in load average per core. This is a JUDGEMENT CALL, not a measurement: it
#: leaves roughly seventy percent of cores free, which is where queueing stops being visible in
#: wall-clock in my experience of this box. It is exposed as a flag precisely because it is not
#: derived from anything, and a caller with a measurement should override it.
DEFAULT_MAX_LOAD_PER_CORE = 0.30


class HostTooBusyError(RuntimeError):
    """The host is under load that would be indistinguishable from the cost being measured."""


def read_load_per_core() -> float | None:
    """One minute load average divided by core count, or `None` where that is unknowable.

    `None` is a real answer, not a placeholder. `os.getloadavg` does not exist on Windows, and
    inventing a zero there would turn a guard that cannot fire into a guard that reports all
    clear. The caller records the `None` and the artifact carries JSON null.
    """
    try:
        one_minute = os.getloadavg()[0]
    except (OSError, AttributeError):  # no getloadavg (Windows), or /proc unreadable
        return None
    cores = os.cpu_count() or 1
    return one_minute / cores


def assert_host_quiet(
    ceiling: float = DEFAULT_MAX_LOAD_PER_CORE, *, allow_busy: bool = False
) -> float | None:
    """Return the load per core, refusing above `ceiling` unless `allow_busy`.

    The reading is returned in every case, including when it refuses to be the reason to stop and
    including when the override is set, because the caller stamps it into provenance either way.
    A run that was allowed to proceed on a busy host must still say how busy.
    """
    load = read_load_per_core()
    if load is None or allow_busy or load <= ceiling:
        return load
    raise HostTooBusyError(
        f"host load is {load:.2f} per core, above the {ceiling:.2f} ceiling. Every leg of a "
        f"latency split measured here would carry queueing delay that cannot be attributed "
        f"afterwards, so the artifact would describe the host rather than the store. Wait for "
        f"the box, or pass --allow-busy-host to publish a contended measurement as one."
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_hostload.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add recall/eval/hostload.py tests/test_hostload.py
git commit -m "feat(eval): refuse to publish a latency artifact from a contended host

VPS2 sat at load 33.7 on 12 cores while this was designed. Every leg of a
split measured there carries queueing delay nobody can attribute, and the
artifact reads as a property of the store.

The 0.30 per-core ceiling is a judgement call and is documented as one.
On Windows the reading is None, the guard genuinely cannot fire, and the
artifact carries JSON null rather than an invented zero.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: the benchmark. Flags, wiring, and the docstring that stops being false

**Files:**
- Modify: `benchmarks/store_latency_share.py` (docstring lines 34-42; `LegSplit` ~line 123; `LegSplit(...)` construction ~line 509; `to_markdown` ~line 548; `main()` lines 605-714)
- Test: `tests/test_store_latency_splade_arm.py` (create)

**Interfaces:**
- Consumes: `backfill_learned_sparse`, `assert_sparse_coverage` (Tasks 2 and 3), `assert_host_quiet`, `read_load_per_core`, `DEFAULT_MAX_LOAD_PER_CORE` (Task 6), and `measure(..., sparse_backend=, sparse_encoder=)`, which already exists and is simply never passed.
- Produces: `splits.json` rows carrying `sparse_backend`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store_latency_splade_arm.py`:

```python
"""The splade arm of the latency split, end to end on a tiny corpus.

No checkpoint: a deterministic keyword encoder drives the same production retrieval path, so
this asserts the ARM is selectable and its guards hold, at a size a test suite can afford.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.store_latency_share import measure  # noqa: E402
from recall.sparse import SparseProfile, backfill_learned_sparse  # noqa: E402
from recall.types import Chunk  # noqa: E402
from tests.conftest import requires_db  # noqa: E402

PROFILE_ID = "kw-arm-test"
VOCAB = {"aardvark": 7, "beta": 9, "gamma": 11, "delta": 13}


class KeywordSparseEncoder:
    def __init__(self) -> None:
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [
            {VOCAB[w]: 1.0 for w in text.lower().split() if w in VOCAB} for text in texts
        ]


class StubEmbedder:
    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


@requires_db
def test_the_splade_arm_reports_a_learned_fire_rate_and_no_lexical_leg(make_store) -> None:
    """`splade` REPLACES ts_rank rather than adding to it.

    So under it the LEXICAL leg is the one asserted idle, and its fire rate must read null rather
    than 0.0. Zero is the issue #81 alarm value (a leg present but matching nothing), and a
    healthy splade run must not publish the alarm.
    """
    store = make_store(64)
    chunks = [
        Chunk(id=f"c{i}", source=f"/c/{i}.md", text=text, metadata={"file": f"{i}.md"})
        for i, text in enumerate(["aardvark facts", "beta prose", "gamma notes", "delta text"])
    ]
    store.upsert(chunks, [[0.1] * 64 for _ in chunks])
    encoder = KeywordSparseEncoder()
    backfill_learned_sparse(store, encoder)

    split = measure(
        store, StubEmbedder(),
        [{"query": "aardvark"}, {"query": "beta"}],
        candidate_k=10, reranker=None, n_chunks=store.count(), repeats=1,
        sparse_backend="splade", sparse_encoder=encoder,
    )

    assert split.sparse_backend == "splade"
    assert split.learned_sparse_fire_rate is not None
    assert split.learned_sparse_fire_rate > 0.0
    assert split.sparse_fire_rate is None
    assert split.max_nesting_violation_ms <= 0.0


@requires_db
def test_the_lexical_arm_still_reports_a_null_learned_fire_rate(make_store) -> None:
    """The mirror image, so neither column can be hard-coded to a constant."""
    store = make_store(64)
    chunks = [
        Chunk(id=f"c{i}", source=f"/c/{i}.md", text=text, metadata={"file": f"{i}.md"})
        for i, text in enumerate(["aardvark facts", "beta prose", "gamma notes", "delta text"])
    ]
    store.upsert(chunks, [[0.1] * 64 for _ in chunks])

    split = measure(
        store, StubEmbedder(),
        [{"query": "aardvark"}, {"query": "beta"}],
        candidate_k=10, reranker=None, n_chunks=store.count(), repeats=1,
        sparse_backend="lexical",
    )

    assert split.sparse_backend == "lexical"
    assert split.learned_sparse_fire_rate is None
    assert split.sparse_fire_rate is not None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_store_latency_splade_arm.py -v
```

Expected: FAIL, `AttributeError: 'LegSplit' object has no attribute 'sparse_backend'`.

- [ ] **Step 3: Add `sparse_backend` to `LegSplit`**

In `benchmarks/store_latency_share.py`, in the `LegSplit` dataclass, insert after `reranked: bool`:

```python
    candidate_k: int
    reranked: bool
    #: Which sparse backend this row measured: `lexical`, `splade` or `both`. Present because one
    #: invocation now sweeps several, and two rows that differ only in this would otherwise be
    #: indistinguishable in `splits.json` while measuring different pipelines.
    sparse_backend: str
    repeats: int
```

In the `LegSplit(...)` construction inside `measure`, add the matching argument after `reranked=`:

```python
        candidate_k=candidate_k,
        reranked=reranker is not None,
        sparse_backend=sparse_backend,
        repeats=repeats,
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_store_latency_splade_arm.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Add the backend column to the markdown table**

In `to_markdown`, change the header rows to:

```python
    rows = [
        "| chunks | cand_k | backend | rerank | total | embed | dense | sparse | learned | "
        "splade enc | meta | fusion | rerank | **store** | resid | **store share** | "
        "sparse fire | learned fire |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
```

and in the row f-string, insert `s.sparse_backend` immediately after the `candidate_k` cell so the column order matches the header. Read the existing f-string and place the new cell between the `cand_k` value and the `rerank` value.

- [ ] **Step 6: Add the CLI flags**

In `main()`, after `ap.add_argument("--rerank", action="store_true")`, add:

```python
    ap.add_argument(
        "--sparse-backend", action="append", dest="sparse_backends",
        choices=["lexical", "splade", "both"],
        help="repeatable; each value is swept against the SAME store and the same encode "
             "(default: lexical). `splade` REPLACES the ts_rank leg rather than adding to it.",
    )
    ap.add_argument("--sparse-model", default=None, help="learned sparse checkpoint")
    ap.add_argument(
        "--sparse-revision", default=None,
        help="pin the checkpoint revision. Without it artifact_digest can silently fall back to "
             "'unpinned' and the run is not reproducible.",
    )
    ap.add_argument("--sparse-top-k", type=int, default=None, help="prune budget; pgvector caps at 1000")
    ap.add_argument(
        "--accept-noncommercial-license", action="store_true",
        help="required for the naver checkpoints (cc-by-nc-sa-4.0); RE-call itself is MIT",
    )
    ap.add_argument(
        "--max-load-per-core", type=float, default=DEFAULT_MAX_LOAD_PER_CORE,
        help="refuse to measure above this 1-minute load average per core",
    )
    ap.add_argument(
        "--allow-busy-host", action="store_true",
        help="measure anyway, and stamp the artifact with the load it was measured under",
    )
```

Add the imports at the top of the file:

```python
from recall.eval.hostload import DEFAULT_MAX_LOAD_PER_CORE, assert_host_quiet, read_load_per_core
from recall.sparse import assert_sparse_coverage, backfill_learned_sparse
```

- [ ] **Step 7: Wire the flags into the run**

In `main()`, after the existing `--queries`/`--repeats` validation and before `warn_if_insecure_dsn(args.dsn)`, add:

```python
    sparse_backends = args.sparse_backends or ["lexical"]
    wants_learned = any(b in ("splade", "both") for b in sparse_backends)
    # Before the corpus build, not after: the operator should not pay for indexing and a CPU
    # encode to be told the host was never quiet enough to publish from.
    load_before = assert_host_quiet(args.max_load_per_core, allow_busy=args.allow_busy_host)
```

Replace the `emb = _make_embedder(args.embedder)` line's neighbourhood with the encoder construction:

```python
    emb = _make_embedder(args.embedder)

    sparse_encoder = None
    if wants_learned:
        # Imported HERE, not at module scope: torch and transformers are the `sparse` extra, and
        # a lexical-only run must not require them to be installed at all.
        from recall.sparse import DEFAULT_MODEL, HNSW_MAX_NONZERO, SpladeEncoder

        sparse_encoder = SpladeEncoder.from_pretrained(
            args.sparse_model or DEFAULT_MODEL,
            top_k=args.sparse_top_k or HNSW_MAX_NONZERO,
            revision=args.sparse_revision,
            accept_noncommercial_license=args.accept_noncommercial_license,
        )
        print(f"learned sparse encoder: {sparse_encoder.profile.fingerprint()}")
```

Inside the `with _throwaway_store(...) as store:` block, after the `print(f"  indexed {n_chunks} chunks ...")` line, add the backfill:

```python
        if sparse_encoder is not None:
            print(f"  encoding {n_chunks} chunks to the learned sparse sidecar ...")
            t_sparse = time.perf_counter()
            sparse_result = backfill_learned_sparse(
                store, sparse_encoder,
                progress=lambda done: print(f"    {done}/{n_chunks}", flush=True),
            )
            assert_sparse_coverage(
                store, sparse_encoder.profile.profile_id, empty_ids=sparse_result.empty_ids
            )
            print(f"  encoded in {time.perf_counter() - t_sparse:.1f}s")
```

Replace the `configs` comprehension and the measure loop with:

```python
        configs = [
            (backend, ck, rr)
            for backend in sparse_backends
            for ck in candidate_ks
            for rr in ([None, reranker] if reranker else [None])
        ]
        for backend, ck, rr in configs:
            print(f"  measuring backend={backend} candidate_k={ck} rerank={'yes' if rr else 'no'} ...")
            splits.append(
                measure(
                    store, emb, queries, candidate_k=ck, reranker=rr,
                    n_chunks=n_chunks, repeats=args.repeats,
                    sparse_backend=backend,
                    sparse_encoder=sparse_encoder if backend in ("splade", "both") else None,
                )
            )
```

After the `with` block closes and before `splits.json` is written, add the post-run reading:

```python
    # Sampled AFTER the timed phase as well. A box that was quiet at the start can be busy by the
    # end of a forty minute run, and a pre-run reading alone would certify a window that closed.
    # This one cannot un-measure the run, so it is a note rather than a refusal, and `notes`
    # already drives the exit code.
    load_after = read_load_per_core()
    if (
        not args.allow_busy_host
        and load_after is not None
        and load_after > args.max_load_per_core
    ):
        for split in splits:
            split.notes.append(
                f"host load rose to {load_after:.2f} per core during the run (ceiling "
                f"{args.max_load_per_core:.2f}); these figures include contention"
            )
```

In the `_provenance` dict, add the two readings:

```python
                "_provenance": {
                    "generation": "post-#81/#84",
                    "status": "current",
                    "superseded_by": None,
                    "backs": ["store latency share — the Redis-port decision"],
                    "host_load_per_core_before": load_before,
                    "host_load_per_core_after": load_after,
                    "host_load_ceiling_per_core": args.max_load_per_core,
                    "host_load_override": args.allow_busy_host,
```

And in the top-level artifact body, beside `"repeats": args.repeats,`, add:

```python
                "sparse_backends": sparse_backends,
                "sparse_profile": (
                    None if sparse_encoder is None
                    else {
                        "profile_id": sparse_encoder.profile.profile_id,
                        "model_name": sparse_encoder.profile.model_name,
                        "artifact_digest": sparse_encoder.profile.artifact_digest,
                        "top_k": sparse_encoder.profile.top_k,
                        "fingerprint": sparse_encoder.profile.fingerprint(),
                    }
                ),
```

- [ ] **Step 8: Rewrite the docstring paragraph that is now false**

In `benchmarks/store_latency_share.py`, replace the paragraph at lines 34-42 (the one beginning "**The learned sparse arm.**") with:

```
**The learned sparse arm.** `--sparse-backend` selects it, and is repeatable: every value listed
is swept against the SAME store and the same corpus encode, so two arms differ only in the leg
under test rather than in the machine, the corpus or the hour. Each guard here has a learned-leg
counterpart: the per-query denominator, the nesting cross-check, the fire rate floor, and the
attribution itself. Selecting `splade` encodes the corpus into the learned sparse sidecar first
(`recall.sparse.backfill_learned_sparse`) and refuses to measure a corpus that did not fully
encode. Note that `splade` REPLACES the ts_rank leg rather than adding to it, so under it the
LEXICAL leg is the one asserted idle, and its fire rate reads `n/a` rather than 0%.

⚠️ Artifacts under `results/store_latency/` generated before 2026-08-07 predate the learned arm
and carry none of its fields. Nothing was asserted about the learned leg when they were produced,
and they were produced on a different host, so they must not be read beside a learned-arm row.
```

Also update the `Usage:` block at the end of the docstring:

```
Usage:
    python benchmarks/store_latency_share.py --embedder fastembed --filler 20000 \
        --candidate-k 20 --candidate-k 250 --repeats 3 \
        --sparse-backend lexical --sparse-backend splade
```

- [ ] **Step 9: Verify the whole file still behaves, including the lexical path**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_store_latency_splade_arm.py -v && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/ -k "store_latency or store_query_latency" -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 10: Smoke-run the CLI on a tiny corpus with a real checkpoint**

This is the first execution of the arm the docstring said could not be selected. It downloads `prithivida/Splade_PP_en_v1` (apache-2.0) on first use.

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe benchmarks/store_latency_share.py --embedder hashing --queries 5 --repeats 1 --candidate-k 10 --sparse-backend lexical --sparse-backend splade --allow-busy-host --out /c/Users/gde00/AppData/Local/Temp/claude/splade_smoke
```

Expected: two rows in the printed table, the `splade` row showing a learned fire rate and `n/a` in the sparse fire column, the `lexical` row the reverse. `--allow-busy-host` is correct here: this is a smoke test, not an artifact.

- [ ] **Step 11: Commit**

```bash
git add benchmarks/store_latency_share.py tests/test_store_latency_splade_arm.py
git commit -m "feat(bench): the splade arm is selectable, and both arms sweep one store

--sparse-backend is repeatable, so every arm is measured against the same
store, the same corpus and the same encode. Two arms that differ only in
the backend now differ only in the backend.

Selecting splade backfills the sidecar first and refuses a corpus that
did not fully encode. The host quiescence guard refuses before the corpus
build and notes a load that rose during the run.

Removes the docstring paragraph saying the CLI cannot select this arm.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: register the new refusals in the guard ablation sweep

**Files:**
- Modify: `scripts/ablate_store_latency_guards.py`

**Interfaces:**
- Consumes: the tests from Tasks 2, 6 and 7.
- Produces: nothing importable.

**Why.** That script's own docstring records the cost of a stale entry: the rule prints SKIP and silently stops being exercised, which is the same class of defect the rule exists to catch. A new guard that is not in the sweep is a guard nobody will ever mutate.

- [ ] **Step 1: Read the existing `ABLATIONS` structure**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && sed -n '45,140p' scripts/ablate_store_latency_guards.py
```

Each entry is `(label, file, old, new, test_that_must_fail)`. `old` must match the file byte for byte, so copy it from the source rather than retyping it.

- [ ] **Step 2: Add three entries**

Append to the `ABLATIONS` list, using the exact current text of each guard as `old`:

```python
    (
        "the sparse coverage guard refuses a partial sidecar",
        SPARSE,
        "    if encoded == total:\n        return",
        "    if encoded <= total:\n        return",
        "test_coverage_refuses_a_half_encoded_corpus",
    ),
    (
        "the host load guard refuses a busy host",
        HOSTLOAD,
        "    if load is None or allow_busy or load <= ceiling:\n        return load",
        "    return load",
        "test_a_busy_host_is_refused",
    ),
    (
        "the benchmark passes the selected backend to measure()",
        BENCH,
        "                    sparse_backend=backend,",
        "                    sparse_backend=\"lexical\",",
        "test_the_splade_arm_reports_a_learned_fire_rate_and_no_lexical_leg",
    ),
```

Add the three path constants beside the existing `STORE` / `OBS` / `GEN`:

```python
SPARSE = ROOT / "recall" / "sparse.py"
HOSTLOAD = ROOT / "recall" / "eval" / "hostload.py"
BENCH = ROOT / "benchmarks" / "store_latency_share.py"
```

The last entry's test lives in a different file from `TESTS`, so check how the script invokes the named test and, if it hardcodes `TESTS`, extend the tuple with a per-entry test path rather than leaving the entry pointing at a file that does not contain its test. An entry whose test cannot be found prints SKIP, which is exactly the silent failure this script documents.

- [ ] **Step 3: Run the sweep**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe scripts/ablate_store_latency_guards.py 2>&1 | tail -30
```

Expected: every entry reports the named test going RED under its mutation, and no entry reports SKIP. A SKIP is a failure of this task, not a pass.

- [ ] **Step 4: Confirm the tree is clean after the sweep**

The script mutates `recall/*.py` in place while it runs.

```bash
cd /c/Users/gde00/Documents/recall-spladecli && git status --porcelain
```

Expected: only `scripts/ablate_store_latency_guards.py` modified. If any other file is dirty, the sweep did not restore it: `git checkout -- <that file>` before committing.

- [ ] **Step 5: Commit**

```bash
git add scripts/ablate_store_latency_guards.py
git commit -m "test(ablation): register the three new refusals in the guard sweep

A guard that is not in this sweep is a guard nobody will ever mutate.
This file's own docstring records what a missing or stale entry costs: it
prints SKIP and silently stops being exercised.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8b: `--sparse-device`, and a GPU asked for but unusable must say so

**Files:**
- Modify: `recall/sparse.py`
- Modify: `benchmarks/store_latency_share.py` (the argparse block and encoder construction from Task 7)
- Test: `tests/test_sparse_device.py` (create)

**Interfaces:**
- Consumes: `SpladeEncoder.from_pretrained(..., device=...)`, which already exists and already defaults to `"cuda" if torch.cuda.is_available() else "cpu"`.
- Produces: `SPARSE_DEVICES`, `DeviceReport`, `SparseDeviceError`, `device_refusal(...)`, `inspect_sparse_device(requested, required_vram_mb=...)`, `resolve_sparse_device(requested, required_vram_mb=...)`.

**Measured facts this task exists to handle, established 2026-08-07 on the local box.** The GPU is an **NVIDIA GeForce GTX 1070 Ti, 8192 MiB**, driver 582.66. `Splade_PP_en_v1` is BERT-base, roughly 110M parameters, so fp32 inference at batch 32 needs well under 2 GB: VRAM is not the constraint. The installed torch is **`2.12.1+cpu`**, with `torch.version.cuda` reporting `None` and `device_count` 0, so CUDA is unreachable because of the wheel and not the hardware. The card is Pascal, **compute capability 6.1**, and recent PyTorch CUDA wheels have been dropping older architectures. Whether the current wheel still ships `sm_61` is **not asserted anywhere in this task**: it is read off `torch.cuda.get_arch_list()` at runtime, which is the only honest way to answer it.

**Why refusal and not fallback.** `recall/sparse.py` already documents this silence for rented hardware: `from_pretrained` loads to CPU, so a rented GPU "would sit idle while the corpus encoded on the instance's CPU. Nothing would error. The vectors would be correct, the run would be ~100x slower, and the only symptom is the bill." Locally it is worse, because there is no bill to notice. An explicit `--sparse-device cuda` is a statement about the run, and a silent fallback makes it a false one.

**Why the device reaches provenance.** `learned_sparse_encode_ms_mean` is a transformer forward pass. Its value on CPU and on GPU are measurements of different things, so an artifact that does not record which one it was cannot be compared against another.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sparse_device.py`:

```python
"""Asking for a GPU and silently getting a CPU is the failure this file exists to exclude.

`device_refusal` is a PURE function over the facts, so every branch below is shown firing on a
box with no CUDA build and no GPU. The thin collector that reads those facts off torch is the
only part not covered here, and it is the part with no logic in it.
"""

from __future__ import annotations

import pytest

from recall.sparse import DeviceReport, SparseDeviceError, device_refusal, resolve_sparse_device

#: A GTX 1070 Ti as measured on this box: Pascal, compute capability 6.1, 8 GB.
PASCAL = (6, 1)
PASCAL_ARCHES = ("sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")
NO_PASCAL_ARCHES = ("sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100")


def test_a_cpu_only_wheel_is_refused_by_name() -> None:
    """The exact local condition: torch 2.12.1+cpu, hardware present and unreachable.

    Reported as a WHEEL problem, not as "no GPU". Those need different fixes, and telling someone
    with a working card that they have no GPU sends them to the wrong one.
    """
    refusal = device_refusal(
        cuda_build=None, device_count=0, capability=None,
        arch_list=(), free_vram_mb=None, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "CPU-only" in refusal


def test_a_cuda_wheel_with_no_visible_device_is_refused() -> None:
    refusal = device_refusal(
        cuda_build="12.8", device_count=0, capability=None,
        arch_list=("sm_80",), free_vram_mb=None, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "no CUDA device" in refusal


def test_a_card_whose_architecture_the_wheel_dropped_is_refused() -> None:
    """The Pascal trap, and the reason this reads the arch list rather than assuming it.

    A wheel built without sm_61 does not politely decline. Naming the architecture and listing
    what the wheel does carry is what turns that into an actionable message.
    """
    refusal = device_refusal(
        cuda_build="12.8", device_count=1, capability=PASCAL,
        arch_list=NO_PASCAL_ARCHES, free_vram_mb=8192, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "sm_61" in refusal
    assert "sm_80" in refusal


def test_insufficient_free_vram_is_refused_with_both_numbers() -> None:
    refusal = device_refusal(
        cuda_build="12.8", device_count=1, capability=PASCAL,
        arch_list=PASCAL_ARCHES, free_vram_mb=512, required_vram_mb=2048,
    )

    assert refusal is not None
    assert "512" in refusal and "2048" in refusal


def test_a_usable_card_produces_no_refusal() -> None:
    """The positive control. Without it every assertion above passes on a function that always
    refuses, which would read as a working guard while blocking every GPU run."""
    assert device_refusal(
        cuda_build="12.8", device_count=1, capability=PASCAL,
        arch_list=PASCAL_ARCHES, free_vram_mb=8192, required_vram_mb=2048,
    ) is None


def test_requesting_cuda_explicitly_raises_rather_than_falling_back(monkeypatch) -> None:
    import recall.sparse as sparse

    monkeypatch.setattr(
        sparse, "inspect_sparse_device",
        lambda requested, required_vram_mb=2048: DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None,
            refusal="torch is a CPU-only build",
        ),
    )

    with pytest.raises(SparseDeviceError, match="CPU-only"):
        resolve_sparse_device("cuda")


def test_auto_falls_back_without_raising(monkeypatch) -> None:
    """`auto` means "use it if it is there", so a refusal is information, not an error."""
    import recall.sparse as sparse

    monkeypatch.setattr(
        sparse, "inspect_sparse_device",
        lambda requested, required_vram_mb=2048: DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None,
            refusal="torch is a CPU-only build",
        ),
    )

    assert resolve_sparse_device("auto") == "cpu"


def test_requesting_cpu_never_consults_cuda() -> None:
    """Asking for CPU on a box with no torch at all must still work.

    Routed through the collector, this would import torch in order to decide it did not need
    torch.
    """
    assert resolve_sparse_device("cpu") == "cpu"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_device.py -v
```

Expected: collection error, `ImportError: cannot import name 'DeviceReport' from 'recall.sparse'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `recall/sparse.py`:

```python
#: What BERT-base fp32 inference needs with headroom for activations at batch 32. Stated as a
#: number rather than computed, and exposed as a parameter, because a caller running a larger
#: checkpoint or a bigger batch has a different answer and should not have to edit this file.
DEFAULT_REQUIRED_VRAM_MB = 2048

SPARSE_DEVICES = ("auto", "cpu", "cuda")


class SparseDeviceError(RuntimeError):
    """A device was asked for by name and cannot be used."""


@dataclass(frozen=True)
class DeviceReport:
    """Everything the device decision was made from, kept so it can be printed and stamped.

    `learned_sparse_encode_ms_mean` is a transformer forward pass, so its value on CPU and on GPU
    are measurements of different things. An artifact that does not record which one it was cannot
    be compared against another, which is why this whole object reaches provenance rather than
    only the resolved string.
    """

    requested: str
    resolved: str
    torch_cuda_build: str | None
    device_name: str | None
    capability: tuple[int, int] | None
    supported_architectures: tuple[str, ...]
    free_vram_mb: int | None
    refusal: str | None


def device_refusal(
    *,
    cuda_build: str | None,
    device_count: int,
    capability: tuple[int, int] | None,
    arch_list: tuple[str, ...],
    free_vram_mb: int | None,
    required_vram_mb: int = DEFAULT_REQUIRED_VRAM_MB,
) -> str | None:
    """Why CUDA cannot be used, or `None` when it can.

    A PURE function over the facts, deliberately. Every branch is then reachable from a test on a
    box with no GPU and no CUDA build, which is the only way this guard gets shown FIRING rather
    than shown running. The collector that gathers these facts has no logic in it.

    The checks are ordered so each names its own fix. A CPU-only wheel and an absent card need
    different actions, and telling someone with a working card that they have no GPU sends them
    to the wrong one.
    """
    if cuda_build is None:
        return (
            "torch is a CPU-only build (torch.version.cuda is None), so no GPU is reachable "
            "regardless of what hardware is present. Install a CUDA build of torch."
        )
    if device_count < 1:
        return (
            f"torch is built against CUDA {cuda_build} but reports no CUDA device. The driver, "
            f"the container's device mapping or CUDA_VISIBLE_DEVICES is where to look."
        )
    if capability is not None:
        arch = f"sm_{capability[0]}{capability[1]}"
        if arch_list and arch not in arch_list:
            return (
                f"this card is compute capability {capability[0]}.{capability[1]} ({arch}) and "
                f"the installed torch was not built for it. It carries {', '.join(arch_list)}. "
                f"A wheel without the architecture does not decline politely, so this refuses "
                f"here instead. Install a torch build listing {arch}, or pass --sparse-device cpu."
            )
    if free_vram_mb is not None and free_vram_mb < required_vram_mb:
        return (
            f"only {free_vram_mb} MiB of VRAM is free and this needs about {required_vram_mb} "
            f"MiB. Encoding would fail partway through a corpus rather than here."
        )
    return None


def inspect_sparse_device(
    requested: str = "auto", required_vram_mb: int = DEFAULT_REQUIRED_VRAM_MB
) -> DeviceReport:
    """Read the device facts off torch and apply `device_refusal` to them.

    `requested="cpu"` short-circuits before importing torch. Otherwise asking for CPU on a box
    with no torch would import torch in order to decide it did not need torch.
    """
    if requested == "cpu":
        return DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None, refusal=None,
        )

    try:
        import torch
    except ImportError:
        return DeviceReport(
            requested=requested, resolved="cpu", torch_cuda_build=None, device_name=None,
            capability=None, supported_architectures=(), free_vram_mb=None,
            refusal="torch is not installed; the learned sparse path needs the `sparse` extra",
        )

    cuda_build = torch.version.cuda
    device_count = torch.cuda.device_count() if cuda_build else 0
    name = None
    capability = None
    free_vram_mb = None
    arch_list: tuple[str, ...] = ()
    if device_count:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        arch_list = tuple(torch.cuda.get_arch_list())
        # `mem_get_info` returns (free, total) in bytes. FREE rather than total: another process
        # holding the card is the common case on a shared box, and total would say yes to a card
        # with nothing left to give.
        free_bytes, _total = torch.cuda.mem_get_info(0)
        free_vram_mb = int(free_bytes // (1024 * 1024))

    refusal = device_refusal(
        cuda_build=cuda_build, device_count=device_count, capability=capability,
        arch_list=arch_list, free_vram_mb=free_vram_mb, required_vram_mb=required_vram_mb,
    )
    return DeviceReport(
        requested=requested, resolved="cpu" if refusal else "cuda",
        torch_cuda_build=cuda_build, device_name=name, capability=capability,
        supported_architectures=arch_list, free_vram_mb=free_vram_mb, refusal=refusal,
    )


def resolve_sparse_device(
    requested: str = "auto", required_vram_mb: int = DEFAULT_REQUIRED_VRAM_MB
) -> str:
    """The device string for `SpladeEncoder.from_pretrained`, refusing a named GPU it cannot use.

    `auto` means "use it if it is there", so a refusal is information and the answer is `cpu`.
    `cuda` is a STATEMENT about the run, and answering `cpu` to it would make that statement false
    while producing correct vectors roughly a hundred times more slowly, with nothing to show for
    it. See the note on `SpladeEncoder.device`.
    """
    if requested not in SPARSE_DEVICES:
        raise ValueError(f"device must be one of {SPARSE_DEVICES}, got {requested!r}")
    report = inspect_sparse_device(requested, required_vram_mb=required_vram_mb)
    if requested == "cuda" and report.refusal:
        raise SparseDeviceError(f"--sparse-device cuda was requested but {report.refusal}")
    return report.resolved
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -m pytest tests/test_sparse_device.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Confirm the real collector agrees with the measured facts of this box**

Not an assertion in the suite: that would encode one machine into the test file. Run it once and read it.

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe -c "
from recall.sparse import inspect_sparse_device
print(inspect_sparse_device('auto'))
"
```

Expected on this box, given `torch 2.12.1+cpu`: `resolved='cpu'` with a refusal naming the CPU-only build. Anything else means the collector disagrees with `torch.version.cuda is None`, and that must be understood before going further.

- [ ] **Step 6: Wire the flag into the benchmark**

Add to the module-scope imports in `benchmarks/store_latency_share.py`. `SPARSE_DEVICES` is needed at argparse time, before the lazy block, and it is a plain tuple of strings that pulls in no torch:

```python
from recall.sparse import SPARSE_DEVICES
```

Add the flag beside the other sparse flags from Task 7:

```python
    ap.add_argument(
        "--sparse-device", choices=list(SPARSE_DEVICES), default="auto",
        help="`cuda` REFUSES if the GPU is unusable rather than falling back; `auto` falls back "
             "to CPU and prints what it chose",
    )
```

Replace the encoder construction added in Task 7 Step 7 with:

```python
    sparse_encoder = None
    sparse_device_report = None
    if wants_learned:
        # Imported HERE, not at module scope: torch and transformers are the `sparse` extra, and
        # a lexical-only run must not require them to be installed at all.
        from recall.sparse import (
            DEFAULT_MODEL, HNSW_MAX_NONZERO, SpladeEncoder,
            inspect_sparse_device, resolve_sparse_device,
        )

        sparse_device_report = inspect_sparse_device(args.sparse_device)
        device = resolve_sparse_device(args.sparse_device)
        print(f"learned sparse device: {device} (requested {args.sparse_device})")
        if sparse_device_report.refusal:
            print(f"  GPU not used: {sparse_device_report.refusal}")
        sparse_encoder = SpladeEncoder.from_pretrained(
            args.sparse_model or DEFAULT_MODEL,
            top_k=args.sparse_top_k or HNSW_MAX_NONZERO,
            revision=args.sparse_revision,
            accept_noncommercial_license=args.accept_noncommercial_license,
            device=device,
        )
        print(f"learned sparse encoder: {sparse_encoder.profile.fingerprint()}")
```

In the artifact body, beside `"sparse_profile"`, add:

```python
                "sparse_device": (
                    None if sparse_device_report is None
                    else {
                        "requested": sparse_device_report.requested,
                        "resolved": sparse_device_report.resolved,
                        "device_name": sparse_device_report.device_name,
                        "torch_cuda_build": sparse_device_report.torch_cuda_build,
                        "capability": (
                            None if sparse_device_report.capability is None
                            else list(sparse_device_report.capability)
                        ),
                        "free_vram_mb": sparse_device_report.free_vram_mb,
                        "refusal": sparse_device_report.refusal,
                    }
                ),
```

- [ ] **Step 7: Verify the refusal fires through the real entry point**

`torch` here is a CPU-only build, so this must refuse. This is the guard shown firing through the CLI, not through a monkeypatch.

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe benchmarks/store_latency_share.py --embedder hashing --queries 3 --repeats 1 --candidate-k 10 --sparse-backend splade --sparse-device cuda --allow-busy-host --out /c/Users/gde00/AppData/Local/Temp/claude/splade_device_probe; echo "exit=$?"
```

Expected: a non-zero exit with `SparseDeviceError: --sparse-device cuda was requested but torch is a CPU-only build ...`, raised **before** the corpus is generated. If it generates a corpus first, the resolve call is in the wrong place: move it above the corpus build.

- [ ] **Step 8: Verify `auto` still runs, on CPU, and says so**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe benchmarks/store_latency_share.py --embedder hashing --queries 3 --repeats 1 --candidate-k 10 --sparse-backend splade --sparse-device auto --allow-busy-host --out /c/Users/gde00/AppData/Local/Temp/claude/splade_device_auto
```

Expected: `learned sparse device: cpu (requested auto)` followed by the refusal line, then a normal run. Confirm `splits.json` carries `sparse_device.resolved == "cpu"` and a non-null `refusal`.

- [ ] **Step 9: Add the device refusal to the ablation sweep**

Append to `ABLATIONS` in `scripts/ablate_store_latency_guards.py`, alongside the entries from Task 8:

```python
    (
        "an explicitly requested cuda device is refused when unusable",
        SPARSE,
        "    if requested == \"cuda\" and report.refusal:",
        "    if False and report.refusal:",
        "test_requesting_cuda_explicitly_raises_rather_than_falling_back",
    ),
```

Run the sweep and confirm no entry prints SKIP:

```bash
cd /c/Users/gde00/Documents/recall-spladecli && /c/Users/gde00/Documents/recall/.venv/Scripts/python.exe scripts/ablate_store_latency_guards.py 2>&1 | tail -30
```

- [ ] **Step 10: Commit**

```bash
git add recall/sparse.py benchmarks/store_latency_share.py scripts/ablate_store_latency_guards.py tests/test_sparse_device.py
git commit -m "feat(sparse): --sparse-device, and a named GPU that cannot be used refuses

from_pretrained already preferred CUDA when available. What was missing
was a way to ASK for it and a way to be told no.

sparse.py already records what the silent version costs on rented
hardware: correct vectors, ~100x slower, and the only symptom is the
bill. Locally there is not even a bill, so an explicit --sparse-device
cuda refuses and names which check failed: CPU-only wheel, no visible
device, an architecture the wheel was not built for, or insufficient
free VRAM. 'auto' still falls back, and now prints what it chose.

device_refusal is pure over the facts, so all four branches are shown
firing on a box with no CUDA build. The local card is a GTX 1070 Ti
(Pascal, sm_61) behind a torch 2.12.1+cpu wheel, which is exactly the
first refusal.

The resolved device reaches provenance: learned_sparse_encode_ms_mean is
a transformer forward pass, so CPU and GPU runs are not comparable and an
artifact that does not say which it was cannot be read beside another.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: run both arms on VPS2

**Files:**
- Create on VPS2: `/opt/recall-splade-cli` (checkout), `/opt/recall-splade-cli/.venv`
- Create locally after the run: `results/store_latency/splits.json`, `results/store_latency/SPLIT.md`

**Interfaces:**
- Consumes: everything above, merged and pushed so the VPS2 checkout can fetch it.

**Host facts, established 2026-08-07 and not to be assumed:** 12 cores, 47 GB, **no GPU**, Postgres 17.10 with pgvector **0.8.2**, `prithivida/Splade_PP_en_v1` already in `/root/.cache/huggingface`. `/opt/recall-beam*` and the `recall_splade` database belong to the beam lane and must not be touched.

- [ ] **Step 1: Push the branch**

```bash
cd /c/Users/gde00/Documents/recall-spladecli && git push -u origin feat/splade-cli-arm
```

- [ ] **Step 2: Create an isolated checkout and venv on VPS2**

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'git clone --branch feat/splade-cli-arm https://github.com/GiulioDER/RE-call.git /opt/recall-splade-cli && cd /opt/recall-splade-cli && python3 -m venv .venv && .venv/bin/pip install -q -e ".[fastembed,sparse]" && .venv/bin/python -c "import torch,transformers,fastembed;print(torch.__version__, transformers.__version__)"'
```

Expected: a torch and transformers version pair. If the clone needs credentials, copy the worktree up with `scp -r` instead.

- [ ] **Step 3: Verify the pgvector nonzero ceiling on 0.8.2 rather than assuming 0.8.4's**

`HNSW_MAX_NONZERO = 1000` was measured on pgvector 0.8.4. This host runs 0.8.2. An over-budget vector is rejected at INSERT, so a wrong constant means a 20,000 chunk load dies partway through with an arbitrary prefix committed.

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cd /opt/recall-splade-cli && sudo -u postgres psql -d postgres -tAc "SELECT extversion FROM pg_extension WHERE extname='"'"'vector'"'"'"'
```

Then, in a scratch database, create a `sparsevec(30522)` column with an HNSW index and INSERT one vector with 1000 nonzeros and one with 1001. Expected: the first succeeds, the second raises. Record both outcomes. If 1001 succeeds, the constant is conservative and the run is unaffected; if 1000 fails, stop and reduce `--sparse-top-k` before the long run.

- [ ] **Step 4: Verify `artifact_digest` does not silently degrade under transformers 5.x**

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cd /opt/recall-splade-cli && .venv/bin/python -c "
from recall.sparse import SpladeEncoder
e = SpladeEncoder.from_pretrained(\"prithivida/Splade_PP_en_v1\")
print(\"digest:\", e.profile.artifact_digest)
print(\"fingerprint:\", e.profile.fingerprint())
"'
```

If the digest prints `unpinned`, transformers 5 no longer exposes `model.config._commit_hash`, and the run in Step 7 **must** pass `--sparse-revision`. Resolve the revision from the HuggingFace model page and record it in the artifact either way.

- [ ] **Step 5: Create a dedicated database**

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'sudo -u postgres psql -tAc "CREATE DATABASE recall_storelat OWNER sentiment" && sudo -u postgres psql -d recall_storelat -tAc "CREATE EXTENSION IF NOT EXISTS vector"'
```

Do not use `recall_splade`, `recall_bench`, or any other existing database. Each belongs to a lane.

- [ ] **Step 6: Take a measured encode rate before committing to the full run**

Encode 1000 passages and read the reported rate, so the full-run duration is a projection from a measurement rather than a guess. `scripts/encode_sparse.py --limit` already reports `rate_per_s` and `projected_hours_for_366479`.

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cd /opt/recall-splade-cli && nohup .venv/bin/python benchmarks/store_latency_share.py --dsn "$RECALL_STORELAT_DSN" --embedder fastembed --filler 500 --queries 10 --repeats 1 --candidate-k 20 --sparse-backend lexical --sparse-backend splade --allow-busy-host --out /var/tmp/storelat_probe > /var/tmp/storelat_probe.log 2>&1 &' && echo launched
```

Read `/var/tmp/storelat_probe.log` for the encode wall-clock, and scale it to 20,000 chunks. If the projection exceeds two hours, reduce `--filler` and say so in the artifact rather than quietly running a smaller corpus.

- [ ] **Step 7: Wait for the host to be quiet, then launch the real run under nohup**

Check the load first. It was 33.7 per 12 cores on 2026-08-07; the guard refuses above 0.30 per core, which is a 1-minute load average of about 3.6 on this box.

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cat /proc/loadavg'
```

When it is under 3.6, launch:

```bash
ssh -i ~/.ssh/contabo_sentiment root@100.91.148.25 'cd /opt/recall-splade-cli && nohup .venv/bin/python benchmarks/store_latency_share.py --dsn "$RECALL_STORELAT_DSN" --embedder fastembed --filler 20000 --queries 100 --repeats 3 --candidate-k 20 --candidate-k 250 --sparse-backend lexical --sparse-backend splade --sparse-device cpu --sparse-revision "<resolved in Step 4>" --out /var/tmp/storelat_run > /var/tmp/storelat_run.log 2>&1 &' && echo launched
```

`--sparse-device cpu` is passed **explicitly** even though VPS2 has no GPU and `auto` would resolve the same way. The artifact then records a device the operator chose rather than one the box happened to have, and a future run of the same command on a GPU box produces a comparable number instead of a silently faster one.

**No `--rerank`.** On a CPU box the cross-encoder dominates the wall-clock, and its absence inflates the store's share, which biases toward "porting the store is worth it". Step 9 records that.

- [ ] **Step 8: Collect the artifact**

```bash
scp -i ~/.ssh/contabo_sentiment root@100.91.148.25:/var/tmp/storelat_run/splits.json /var/tmp/storelat_run/SPLIT.md /c/Users/gde00/Documents/recall-spladecli/results/store_latency/
```

Before committing, verify in `splits.json` that `host_load_per_core_before` and `host_load_per_core_after` are both present and both below the ceiling, that `sparse_profile.artifact_digest` is not `"unpinned"`, and that every `splade` row has a non-null `learned_sparse_fire_rate` and a null `sparse_fire_rate`.

- [ ] **Step 9: Add the caveats to `results/ARTIFACTS.md` and commit**

Record, beside the artifact index entry: the host and its load readings, that there is no GPU so the SPLADE query encode is CPU-bound and dominates the learned bracket, that the corpus is synthetic and therefore measures the regime where the store is cheap by construction (commit `9a5165b` measured sparse median 496 ms on a real 72k corpus), and that the run carries no reranker, which inflates the store share in the direction that favours porting.

```bash
cd /c/Users/gde00/Documents/recall-spladecli && git add results/store_latency/splits.json results/store_latency/SPLIT.md results/ARTIFACTS.md
git commit -m "bench: the splade arm, measured, on one store beside its lexical control

First artifact from the arm the docstring said could not be selected.
Both arms swept against one store, one corpus and one encode, so they
differ in the backend and nothing else.

Caveats travel with the numbers: no GPU so the query encode is CPU and
dominates the learned bracket; synthetic corpus so the sparse leg is not
representative (9a5165b measured 496 ms median on a real 72k corpus); no
reranker, which inflates the store share toward 'porting is worth it'.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Library primitive → Task 1. Empty-vector split → Tasks 1 and 2. Coverage guard → Task 2. Backfill and its idempotence note → Task 3. `Indexer` hook in `_flush`, tested on row counts and on the re-index skip path → Task 4. `drop_table` sidecar leak → Task 5. Quiescence guard, both samples, per-core ceiling, Windows `null` → Tasks 6 and 7. Flags, one-invocation sweep, `LegSplit.sparse_backend`, `measure()` wiring, docstring rewrite, provenance fields → Task 7. Ablation registry → Task 8. Device selection, GPU capability refusal, device in provenance → Task 8b (added 2026-08-07 after the spec was approved; the spec has a matching addendum). VPS2 isolation, pgvector 0.8.2 probe, `_commit_hash` probe, dedicated database, `nohup`, no `--rerank` → Task 9.

**Known gaps, stated rather than hidden.** Task 7 Step 5 and Task 8 Step 2 describe an edit against text the implementer must read from the file first, because the exact current bytes of that f-string and of the `ABLATIONS` invocation are what the edit must match, and transcribing them here would create a second copy free to drift. Both steps say so and say where to look. Task 9 Step 3 describes a probe rather than giving its SQL inline, because the scratch table name depends on the database created in Step 5.

**Type consistency.** `SparseIndexResult(written, empty_ids)` is returned by `store_sparse_vectors` (Task 1) and by `backfill_learned_sparse` (Task 3), and consumed by `assert_sparse_coverage(..., empty_ids=)` (Task 2) and by the benchmark (Task 7). `SparseEncoderProtocol` is defined in Task 1 and referenced by Tasks 3, 4 and 7. `assert_host_quiet` and `read_load_per_core` are defined in Task 6 with the signatures Task 7 calls. `LegSplit.sparse_backend` is added in Task 7 Step 3 and asserted in Task 7 Step 1.
