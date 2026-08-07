# Late-Interaction Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether late-interaction (ColBERT/MaxSim) reranking ranks the 123 MTRAG gold documents that two pooled-pair cross-encoders 25x apart in size both bury.

**Architecture:** A `LateInteractionReranker` implementing the existing 12-line `Reranker` protocol, plus an offloaded scorer that reuses `rerank_offload.cmd_dump`'s existing output files verbatim. Pools stay byte-identical across arms, so the only variable is the score source. Documents are streamed and their token matrices discarded, making peak memory independent of corpus size.

**Tech Stack:** Python 3.11+, `fastembed>=0.3` (already a declared extra, in `uv.lock` at 0.8.0), numpy, pytest. No new dependency is added.

**Spec:** [`docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md`](../specs/2026-08-07-late-interaction-rerank-design.md)

## Global Constraints

Every task's requirements implicitly include this section.

- **`rerank()` REORDERS ONLY.** Every hit keeps its dense cosine `score`. `recall/trust.py:292` thresholds on `hit.score` and `recall/trust.py:536` passes it to `cal.confidence()`. A MaxSim score is an unbounded sum over query tokens in different units. Leaking it corrupts calibrated confidence for every hit.
- **No migration, no sidecar table, no retrieval-side use.** Out of scope per spec. Do not create `recall_late_v1`.
- **No change to `search()` or `search_fused()` defaults.** The reranker is opt-in, constructed by the caller.
- **Permissive licences are `{"mit", "apache-2.0"}` as a SET.** Never `!= "apache-2.0"` — that is `sparse.py:195`'s latent defect and it would refuse the MIT primary arm.
- **`fastembed` must be an optional import.** Guard it exactly as `CrossEncoderReranker` guards `sentence_transformers`, so `import recall.rerank` works without the extra.
- **Use `query_embed` for queries and `passage_embed` for documents.** Never `embed` for both. ColBERT prepends distinct `[Q]`/`[D]` markers and pads queries with `[MASK]`; using one method for both sides silently produces wrong scores that still look like numbers.
- **Do NOT score on MTRAG-UN.** Sealed held-out set. `--split dev` only.
- **Statistics reuse `benchmarks/mtrag/analyse_contrasts.py` unchanged.** Paired bootstrap n>=2000, sign-flip permutation n=5000, Holm at 0.05.
- **Line length 100**, matching the repo. Run **`ruff check .`** before each commit, exactly as CI does, plus **`mypy`** (bare, it reads `pyproject.toml`). `pyflakes` alone is NOT sufficient and this plan proved it: ruff and mypy both went red on a branch pyflakes called clean. Ruff DOES launch on this machine; the note claiming otherwise was stale.
- ⚠️ **This plan's own shape generates `E402`, and every task below inherits the hazard.** Tasks 2, 3, 5, 6 and 7 each say "append to the existing test file", and appending an import block after the first test function is a module-level import not at top of file. Ruff rejects it, and CI's `test` job dies at the **lint step before pytest ever runs**, so a green local pytest tells you nothing about whether CI will pass. When a task says "append": append the TESTS at the bottom, but MERGE the imports into the block at the top.

## Deliberate deviation from the spec, and why it is stronger

The spec places all four licence gates in `benchmarks/mtrag/late_interaction.py`. **Gates 1 and 4 (the registry and the opt-in) move to `recall/rerank.py` instead.**

Reason: a licence is a property of the model, not of one benchmark, and `recall/sparse.py:131` already sets that precedent by putting `KNOWN_MODELS` next to `SpladeEncoder` rather than in a benchmark. Putting them in `recall/` means the **production** constructor is guarded too, not merely the benchmark script. Gates 2 and 3 (the Holm-family refusal and the record tainting) stay in the benchmark, because they are properties of the analysis.

This strictly widens the guard. Nothing in the spec's containment argument weakens.

## File Structure

| file | responsibility |
|---|---|
| `recall/rerank.py` (modify) | `maxsim()`, `LateInteractionReranker`, `LATE_INTERACTION_MODELS`, `PERMISSIVE_LICENCES`, `late_interaction_licence()` |
| `tests/test_late_interaction_rerank.py` (create) | reranker unit tests, incl. the score-preservation test |
| `benchmarks/mtrag/late_interaction.py` (create) | `LateArm`, `LATE_ARMS`, `holm_family()`, streaming `score`/`validate` CLI |
| `tests/test_bench_late_interaction.py` (create) | arm registry, Holm-family refusal, record tainting |
| `benchmarks/mtrag/buried_gold_power.py` (create) | Family B precondition: derive the 123 and compute minimum detectable shift |
| `tests/test_buried_gold_power.py` (create) | power arithmetic pinned against a hand-computed case |

---

### Task 1: MaxSim, as a pure function

The scoring metric is pinned independently of any model download, so a wrong `LateInteractionReranker` cannot hide behind a wrong metric.

**Files:**
- Modify: `recall/rerank.py` (append after `CrossEncoderReranker`)
- Test: `tests/test_late_interaction_rerank.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float`. Both arrays are 2-D, shape `(n_tokens, dim)`, L2-normalised by the encoder. Returns the sum over query tokens of the max dot product against any document token.

- [ ] **Step 1: Write the failing test**

Create `tests/test_late_interaction_rerank.py`:

```python
import numpy as np
import pytest

from recall.rerank import maxsim


def test_maxsim_matches_hand_computed_value():
    # q0 . d0 = 1.0, q0 . d1 = 0.6  -> max 1.0
    # q1 . d0 = 0.0, q1 . d1 = 0.8  -> max 0.8
    # sum = 1.8
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert maxsim(query, doc) == pytest.approx(1.8)


def test_maxsim_is_max_not_mean():
    """The mutation check in G5 relies on these differing. If they ever coincide the gate is
    vacuous, so the difference is pinned here rather than assumed."""
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[1.0, 0.0], [0.6, 0.8]])
    mean_version = float((query @ doc.T).mean(axis=1).sum())
    assert mean_version == pytest.approx(1.2)
    assert maxsim(query, doc) != pytest.approx(mean_version)


def test_maxsim_refuses_empty_document():
    """A document with no tokens cannot be scored. Returning 0.0 would place it mid-ranking,
    which is the same silent-corruption shape `rerank_order` refuses a missing score for."""
    query = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="no tokens"):
        maxsim(query, np.zeros((0, 2)))


def test_maxsim_refuses_empty_query():
    with pytest.raises(ValueError, match="no tokens"):
        maxsim(np.zeros((0, 2)), np.array([[1.0, 0.0]]))


def test_maxsim_refuses_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        maxsim(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_late_interaction_rerank.py -v`
Expected: FAIL, `ImportError: cannot import name 'maxsim' from 'recall.rerank'`

- [ ] **Step 3: Write minimal implementation**

Append to `recall/rerank.py`:

```python
def maxsim(query_tokens: "np.ndarray", doc_tokens: "np.ndarray") -> float:
    """ColBERT late-interaction score: sum over query tokens of the best-matching doc token.

    Both arrays are `(n_tokens, dim)` and L2-normalised by the encoder, so a dot product is a
    cosine. The `max` is the whole point: it keeps per-token evidence instead of pooling the pair
    into one representation, which is the deficiency this experiment exists to test.

    Empty inputs RAISE rather than scoring 0.0, for the reason `rerank_order` raises on a missing
    score: a zero is not a neutral value in a ranking, it silently places the item mid-pool.
    """
    if query_tokens.shape[0] == 0:
        raise ValueError("query has no tokens")
    if doc_tokens.shape[0] == 0:
        raise ValueError("document has no tokens")
    if query_tokens.shape[1] != doc_tokens.shape[1]:
        raise ValueError(
            f"dimension mismatch: query is {query_tokens.shape[1]}-d, "
            f"document is {doc_tokens.shape[1]}-d"
        )
    return float((query_tokens @ doc_tokens.T).max(axis=1).sum())
```

Add at the top of `recall/rerank.py`, inside the existing `TYPE_CHECKING` guard if one exists, otherwise create it:

```python
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - numpy arrives with the fastembed extra
    import numpy as np
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_late_interaction_rerank.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add recall/rerank.py tests/test_late_interaction_rerank.py
git commit -m "feat(rerank): maxsim, pinned against a hand-computed value"
```

---

### Task 2: The licence registry and its refusals

**Files:**
- Modify: `recall/rerank.py`
- Test: `tests/test_late_interaction_rerank.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LATE_INTERACTION_MODELS: dict[str, str]` mapping checkpoint to licence; `PERMISSIVE_LICENCES: frozenset[str]`; `late_interaction_licence(model_name: str, *, accept_noncommercial_license: bool = False) -> str` which raises on an unknown checkpoint or a non-permissive one without opt-in, and otherwise returns the licence string; `DEFAULT_LATE_INTERACTION_MODEL: str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_late_interaction_rerank.py`:

```python
from recall.rerank import (
    DEFAULT_LATE_INTERACTION_MODEL,
    LATE_INTERACTION_MODELS,
    PERMISSIVE_LICENCES,
    late_interaction_licence,
)


def test_mit_is_permissive():
    """The load-bearing correction to `sparse.py:195`, which gates on `!= "apache-2.0"` and would
    therefore refuse the MIT primary arm under its own guard."""
    assert "mit" in PERMISSIVE_LICENCES
    assert late_interaction_licence("colbert-ir/colbertv2.0") == "mit"


def test_apache_is_permissive():
    assert late_interaction_licence("answerdotai/answerai-colbert-small-v1") == "apache-2.0"


def test_default_model_is_permissive():
    assert LATE_INTERACTION_MODELS[DEFAULT_LATE_INTERACTION_MODEL] in PERMISSIVE_LICENCES


def test_noncommercial_refused_without_optin():
    with pytest.raises(ValueError, match="cc-by-nc-4.0"):
        late_interaction_licence("jinaai/jina-colbert-v2")


def test_noncommercial_allowed_with_optin():
    assert late_interaction_licence(
        "jinaai/jina-colbert-v2", accept_noncommercial_license=True
    ) == "cc-by-nc-4.0"


def test_unknown_checkpoint_refused():
    """An unrecorded licence is exactly what this check exists to prevent, so an unknown model
    raises even though it might be perfectly permissive."""
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        late_interaction_licence("some/unrecorded-colbert")


def test_unknown_checkpoint_refused_even_with_optin():
    """The opt-in waives the LICENCE check, not the REGISTRY check."""
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        late_interaction_licence("some/unrecorded-colbert", accept_noncommercial_license=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_late_interaction_rerank.py -v`
Expected: FAIL, `ImportError: cannot import name 'LATE_INTERACTION_MODELS'`

- [ ] **Step 3: Write minimal implementation**

Append to `recall/rerank.py`:

```python
#: Licence per late-interaction checkpoint, mirroring `recall.sparse.KNOWN_MODELS`.
#:
#: An unrecorded checkpoint RAISES rather than defaulting to permissive: an unrecorded licence is
#: exactly what this check exists to prevent.
LATE_INTERACTION_MODELS: dict[str, str] = {
    "colbert-ir/colbertv2.0": "mit",
    "answerdotai/answerai-colbert-small-v1": "apache-2.0",
    # Capacity diagnostic ONLY (~560M against the 110M default). Non-commercial, so it is refused
    # without an explicit opt-in and it may never contribute to a shipping decision. See the
    # preregistration's monotonicity rule.
    "jinaai/jina-colbert-v2": "cc-by-nc-4.0",
}

#: Licences compatible with RE-call's own MIT distribution for commercial use.
#:
#: ⚠️ A SET, not an equality test. `recall/sparse.py:195` gates on `license_id != "apache-2.0"`,
#: which would refuse an MIT checkpoint. That is latent there (no MIT entry in `KNOWN_MODELS`) and
#: would be fatal here, because the DEFAULT model below is MIT and would be refused by its own
#: guard. `sparse.py` is deliberately left alone: its defect cannot fire.
PERMISSIVE_LICENCES = frozenset({"mit", "apache-2.0"})

DEFAULT_LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"


def late_interaction_licence(
    model_name: str, *, accept_noncommercial_license: bool = False
) -> str:
    """The checkpoint's licence, refusing unknown or non-permissive ones.

    The opt-in waives the LICENCE check only. An unrecorded checkpoint raises either way, because
    the point of the registry is that no licence goes unrecorded.
    """
    licence = LATE_INTERACTION_MODELS.get(model_name)
    if licence is None:
        raise ValueError(
            f"unknown late-interaction model {model_name!r}; known models are "
            f"{sorted(LATE_INTERACTION_MODELS)}. Record it in LATE_INTERACTION_MODELS with its "
            f"licence first — an unrecorded licence is exactly what this check exists to prevent."
        )
    if licence not in PERMISSIVE_LICENCES and not accept_noncommercial_license:
        raise ValueError(
            f"{model_name} is licensed {licence}, which is not compatible with RE-call's MIT "
            f"distribution for commercial use. Pass accept_noncommercial_license=True to use it "
            f"anyway (benchmark reproduction only — it may not contribute to a shipping "
            f"decision), or keep the default {DEFAULT_LATE_INTERACTION_MODEL}."
        )
    return licence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_late_interaction_rerank.py -v`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add recall/rerank.py tests/test_late_interaction_rerank.py
git commit -m "feat(rerank): licence registry for late-interaction checkpoints

Gates on a permissive SET, not sparse.py:195's `!= apache-2.0`, which would
have refused the MIT primary arm under its own guard."
```

---

### Task 3: LateInteractionReranker

The encoder is injected, mirroring `SpladeEncoder.__init__`, so every behavioural test runs without downloading 0.44 GB of weights.

**Files:**
- Modify: `recall/rerank.py`
- Test: `tests/test_late_interaction_rerank.py`

**Interfaces:**
- Consumes: `maxsim`, `late_interaction_licence`, `DEFAULT_LATE_INTERACTION_MODEL` (Tasks 1-2).
- Produces: `LateInteractionReranker(encoder: Any, *, model_name: str)` with `.rerank(query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]`, `.model_name: str`, `.licence: str`, and classmethod `from_pretrained(model_name: str = DEFAULT_LATE_INTERACTION_MODEL, *, accept_noncommercial_license: bool = False, cache_dir: str | None = None, threads: int | None = None) -> LateInteractionReranker`. The injected encoder must expose `query_embed(list[str])` and `passage_embed(list[str])`, each returning an iterable of `(n_tokens, dim)` arrays.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_late_interaction_rerank.py`:

```python
from recall.rerank import LateInteractionReranker, Reranker
from recall.types import Chunk, ScoredChunk


class _FakeEncoder:
    """Returns pre-set token matrices by text. Records which method each text went through, so a
    test can prove queries use `query_embed` and documents use `passage_embed`."""

    def __init__(self, table: dict[str, list[list[float]]]) -> None:
        self._table = table
        self.query_calls: list[str] = []
        self.passage_calls: list[str] = []

    def query_embed(self, texts):
        texts = list(texts)
        self.query_calls.extend(texts)
        return [np.array(self._table[t]) for t in texts]

    def passage_embed(self, texts):
        texts = list(texts)
        self.passage_calls.extend(texts)
        return [np.array(self._table[t]) for t in texts]


def _hit(cid: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, source="f", text=text), score=score)


def _reranker(table):
    return LateInteractionReranker(_FakeEncoder(table), model_name="colbert-ir/colbertv2.0")


def test_satisfies_the_reranker_protocol():
    assert isinstance(_reranker({}), Reranker)


def test_reorders_by_maxsim():
    table = {
        "q": [[1.0, 0.0]],
        "far": [[0.0, 1.0]],   # maxsim 0.0
        "near": [[1.0, 0.0]],  # maxsim 1.0
    }
    hits = [_hit("far", "far", 0.9), _hit("near", "near", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["near", "far"]


def test_preserves_dense_cosine_score():
    """THE load-bearing invariant. trust.py:292 thresholds on `score` and trust.py:536 feeds it to
    cal.confidence(). A MaxSim value is an unbounded sum in different units; leaking it into
    `score` would corrupt calibrated confidence for every hit. Same hazard rerank.py:84 documents
    for the cross-encoder."""
    table = {
        "q": [[1.0, 0.0]],
        "far": [[0.0, 1.0]],
        "near": [[1.0, 0.0]],
    }
    hits = [_hit("far", "far", 0.9), _hit("near", "near", 0.1)]
    out = _reranker(table).rerank("q", hits)
    by_id = {h.chunk.id: h.score for h in out}
    assert by_id == {"far": 0.9, "near": 0.1}
    assert sorted(h.score for h in out) == sorted(h.score for h in hits)


def test_preserves_indexed_at_and_first_indexed_at():
    from datetime import datetime, timezone

    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = datetime(2025, 1, 1, tzinfo=timezone.utc)
    table = {"q": [[1.0, 0.0]], "a": [[1.0, 0.0]]}
    hits = [
        ScoredChunk(
            chunk=Chunk(id="a", source="f", text="a"),
            score=0.5,
            indexed_at=stamp,
            first_indexed_at=first,
        )
    ]
    out = _reranker(table).rerank("q", hits)
    assert out[0].indexed_at == stamp
    assert out[0].first_indexed_at == first


def test_uses_query_embed_for_query_and_passage_embed_for_documents():
    """ColBERT prepends distinct [Q]/[D] markers and pads queries with [MASK]. Using `embed` for
    both sides produces wrong scores that still look like numbers."""
    table = {"q": [[1.0, 0.0]], "a": [[1.0, 0.0]]}
    encoder = _FakeEncoder(table)
    LateInteractionReranker(encoder, model_name="colbert-ir/colbertv2.0").rerank(
        "q", [_hit("a", "a", 0.5)]
    )
    assert encoder.query_calls == ["q"]
    assert encoder.passage_calls == ["a"]


def test_empty_hits_returns_empty():
    assert _reranker({}).rerank("q", []) == []


def test_ties_preserve_input_order():
    table = {"q": [[1.0, 0.0]], "a": [[1.0, 0.0]], "b": [[1.0, 0.0]]}
    hits = [_hit("a", "a", 0.1), _hit("b", "b", 0.2)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["a", "b"]


def test_output_is_a_permutation_of_input():
    table = {"q": [[1.0, 0.0]], "a": [[0.0, 1.0]], "b": [[1.0, 0.0]]}
    hits = [_hit("a", "a", 0.1), _hit("b", "b", 0.2)]
    out = _reranker(table).rerank("q", hits)
    assert len(out) == len(hits)
    assert {h.chunk.id for h in out} == {"a", "b"}


def test_records_its_licence():
    rr = _reranker({})
    assert rr.model_name == "colbert-ir/colbertv2.0"
    assert rr.licence == "mit"


def test_construction_refuses_an_unregistered_checkpoint():
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        LateInteractionReranker(_FakeEncoder({}), model_name="some/unrecorded")


def test_unscoreable_document_sorts_last_instead_of_aborting_the_batch():
    """`maxsim` refuses a zero-token document, and for one document that is right. For a BATCH it
    is not: raising would break reranking for every hit in the request over one malformed chunk.
    Last is not mid-pool, so the original objection to scoring 0.0 is still honoured."""
    table = {"q": [[1.0, 0.0]], "empty": [], "ok": [[1.0, 0.0]], "weak": [[0.0, 1.0]]}
    hits = [_hit("empty", "empty", 0.9), _hit("weak", "weak", 0.5), _hit("ok", "ok", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["ok", "weak", "empty"]


def test_unscoreable_documents_keep_their_input_order_among_themselves():
    table = {"q": [[1.0, 0.0]], "e1": [], "e2": [], "ok": [[1.0, 0.0]]}
    hits = [_hit("e1", "e1", 0.9), _hit("e2", "e2", 0.5), _hit("ok", "ok", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["ok", "e1", "e2"]


def test_unscoreable_document_still_keeps_its_dense_cosine():
    """The reorder-only invariant must hold for the salvaged case too."""
    table = {"q": [[1.0, 0.0]], "empty": [], "ok": [[1.0, 0.0]]}
    hits = [_hit("empty", "empty", 0.9), _hit("ok", "ok", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert {h.chunk.id: h.score for h in out} == {"empty": 0.9, "ok": 0.1}


def test_empty_query_still_raises():
    """Deliberately NOT salvaged. With no query tokens there is no evidence to rank anything by,
    so unlike a single bad document there is no partial ordering worth returning."""
    table = {"q": [], "ok": [[1.0, 0.0]]}
    with pytest.raises(ValueError, match="query has no tokens"):
        _reranker(table).rerank("q", [_hit("ok", "ok", 0.1)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_late_interaction_rerank.py -v`
Expected: FAIL, `ImportError: cannot import name 'LateInteractionReranker'`

- [ ] **Step 3: Write minimal implementation**

Append to `recall/rerank.py`:

```python
class LateInteractionReranker:
    """Reorder hits by ColBERT-style MaxSim. Requires `pip install recall[fastembed]`.

    The encoder is INJECTED rather than loaded in `__init__`, mirroring `SpladeEncoder`, so the
    scoring path is testable against fake token matrices without a 0.44 GB download.
    `from_pretrained` is the production constructor.

    ⚠️ Queries go through `query_embed` and documents through `passage_embed`. ColBERT prepends
    distinct `[Q]`/`[D]` markers and pads the query side with `[MASK]` tokens, so using one method
    for both sides yields wrong scores that still look like plausible numbers.
    """

    def __init__(self, encoder: object, *, model_name: str) -> None:
        # Validates the checkpoint even on the injected path: a test or a benchmark that fakes the
        # encoder must not be able to fake its way past the licence registry.
        self.licence = late_interaction_licence(model_name, accept_noncommercial_license=True)
        self.model_name = model_name
        self._encoder = encoder

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_LATE_INTERACTION_MODEL,
        *,
        accept_noncommercial_license: bool = False,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> "LateInteractionReranker":
        """Load `model_name`, refusing an unknown or non-permissive checkpoint."""
        late_interaction_licence(
            model_name, accept_noncommercial_license=accept_noncommercial_license
        )
        try:
            from fastembed import LateInteractionTextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "LateInteractionReranker requires: pip install recall[fastembed]"
            ) from exc
        encoder = LateInteractionTextEmbedding(
            model_name=model_name, cache_dir=cache_dir, threads=threads
        )
        return cls(encoder, model_name=model_name)

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        qtokens = list(self._encoder.query_embed([query]))[0]  # type: ignore[attr-defined]
        # Validated UP FRONT, not left to `maxsim`. `maxsim` only runs for documents that have
        # tokens, so a batch in which every document is unscoreable would skip the query check
        # entirely and return an unranked order for a query carrying no evidence at all.
        if qtokens.shape[0] == 0:
            raise ValueError("query has no tokens")
        texts = [h.chunk.text for h in hits]
        dtokens = list(self._encoder.passage_embed(texts))  # type: ignore[attr-defined]
        # A document that encodes to zero tokens cannot be scored, and `maxsim` refuses it. That
        # refusal is right for ONE document, because 0.0 is not a neutral score, it lands the item
        # mid-pool. It is wrong for a BATCH: raising here would abort reranking for every hit in
        # the request over one malformed chunk, which is worse than the outcome the refusal exists
        # to prevent. Ranking such a document LAST satisfies the original objection (last is not
        # mid) without letting one bad chunk break every query that retrieves it.
        #
        # The QUERY side deliberately still raises, via `maxsim`: if the query encodes to nothing
        # there is no evidence to rank ANY document by, so there is no salvageable ordering.
        scores = [
            float("-inf") if d.shape[0] == 0 else maxsim(qtokens, d) for d in dtokens
        ]
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        # Reorder ONLY — each hit keeps its dense cosine `score`, `indexed_at` and
        # `first_indexed_at`. Identical to CrossEncoderReranker.rerank and for the identical
        # reason: `recall.trust` reads `score` as a cosine.
        return [hits[i] for i in order]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_late_interaction_rerank.py -v`
Expected: PASS, 27 passed

- [ ] **Step 5: Verify the whole suite is still green**

Run: `python -m pytest tests/test_rerank.py tests/test_late_interaction_rerank.py -q`
Expected: PASS, no regressions in the existing reranker tests

- [ ] **Step 6: Commit**

```bash
git add recall/rerank.py tests/test_late_interaction_rerank.py
git commit -m "feat(rerank): LateInteractionReranker, reorder-only like its cross-encoder sibling

hit.score stays the dense cosine because trust.py:292 thresholds on it and
trust.py:536 feeds it to cal.confidence(). Pinned by a test, not a comment."
```

---

### Task 4: Benchmark arms, and the Holm family that refuses a non-deployable one

**Files:**
- Create: `benchmarks/mtrag/late_interaction.py`
- Test: `tests/test_bench_late_interaction.py`

**Interfaces:**
- Consumes: `LATE_INTERACTION_MODELS`, `PERMISSIVE_LICENCES` (Task 2).
- Produces: frozen dataclass `LateArm(name: str, checkpoint: str)` with properties `.licence -> str` and `.deployable -> bool`; module constant `LATE_ARMS: tuple[LateArm, ...]`; `holm_family(arms: Sequence[LateArm]) -> tuple[str, ...]` which raises `ValueError` if any arm is non-deployable; `arm_record(arm: LateArm) -> dict[str, object]` returning `{"arm", "checkpoint", "licence", "deployable"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bench_late_interaction.py`:

```python
import pytest

from benchmarks.mtrag.late_interaction import (
    LATE_ARMS,
    LateArm,
    arm_record,
    holm_family,
)


def _by_name(name: str) -> LateArm:
    return next(a for a in LATE_ARMS if a.name == name)


def test_arms_are_frozen_before_any_score():
    """Arms are declared in code, as SPARSE_ARMS is, so they cannot be edited after seeing a
    number without it showing up in the diff."""
    assert isinstance(LATE_ARMS, tuple)
    assert [a.name for a in LATE_ARMS] == ["li_colbertv2", "li_answerai", "li_jina"]


def test_permissive_arms_are_deployable():
    assert _by_name("li_colbertv2").deployable is True
    assert _by_name("li_answerai").deployable is True


def test_jina_is_not_deployable():
    arm = _by_name("li_jina")
    assert arm.licence == "cc-by-nc-4.0"
    assert arm.deployable is False


def test_holm_family_accepts_deployable_arms():
    assert holm_family([_by_name("li_colbertv2"), _by_name("li_answerai")]) == (
        "li_colbertv2",
        "li_answerai",
    )


def test_holm_family_refuses_a_non_deployable_arm():
    """THE containment gate. The verdict that gates the follow-on project is computed from a
    family li_jina cannot mechanically enter. A refusal, not a docstring."""
    with pytest.raises(ValueError, match="li_jina"):
        holm_family([_by_name("li_colbertv2"), _by_name("li_jina")])


def test_holm_family_refuses_even_a_lone_non_deployable_arm():
    with pytest.raises(ValueError, match="li_jina"):
        holm_family([_by_name("li_jina")])


def test_arm_record_carries_the_taint():
    """Numbers get lifted out of these archives into later documents. A lifted number must arrive
    with its licence attached rather than as a bare float."""
    assert arm_record(_by_name("li_jina")) == {
        "arm": "li_jina",
        "checkpoint": "jinaai/jina-colbert-v2",
        "licence": "cc-by-nc-4.0",
        "deployable": False,
    }


def test_every_arm_checkpoint_is_registered():
    from recall.rerank import LATE_INTERACTION_MODELS

    for arm in LATE_ARMS:
        assert arm.checkpoint in LATE_INTERACTION_MODELS


def test_arm_with_an_unregistered_checkpoint_raises_on_licence():
    """LATE_ARMS is frozen, so this branch is unreachable today. It is tested because the failure
    it prevents is a future arm added without a matching registry entry, which would otherwise
    reach `deployable` and be answered from a licence that does not exist."""
    with pytest.raises(ValueError, match="unregistered checkpoint"):
        LateArm("li_future", "some/unrecorded").licence


def test_holm_family_returns_every_name_when_handed_a_single_use_iterator():
    """Reading the argument twice would exhaust the iterator on the `blocked` scan, and the return
    would then be an empty tuple: silent omission of arms the caller believed were included.

    ⚠️ Every arm here is DEPLOYABLE, and that is the whole point. With a blocked arm in the
    iterator the function raises during the first read and never reaches the second, so that
    version of this test passes against the unmaterialised implementation and proves nothing. The
    first draft of this test made exactly that mistake. Verified by mutation: deleting
    `arms = list(arms)` fails this test and only this test.
    """
    assert holm_family(iter([_by_name("li_colbertv2"), _by_name("li_answerai")])) == (
        "li_colbertv2",
        "li_answerai",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'benchmarks.mtrag.late_interaction'`

- [ ] **Step 3: Write minimal implementation**

Create `benchmarks/mtrag/late_interaction.py`:

```python
"""Late-interaction (ColBERT/MaxSim) reranking arms for MTRAG-human dev.

Preregistration: `docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md`.

🔑 Why this reuses `rerank_offload.cmd_dump` rather than re-running retrieval. Pool width alone
moves reranker results here (`closed-hypothesis-recall-rerank-pool-interaction-2026-08-05`: the
same MiniLM got WORSE as the pool widened, entire 95% CI below threshold). Scoring the same frozen
pools means identical pools, identical tie rule and identical metrics, with the score source as the
only variable.

⚠️ `li_jina` is cc-by-nc-4.0 and its effect is declared MONOTONE in the preregistration: it can
strengthen a null or weaken a positive claim, and it can never support a decision to build the
follow-on project. `holm_family` enforces that by refusing it, rather than trusting a reader.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from recall.rerank import LATE_INTERACTION_MODELS, PERMISSIVE_LICENCES


@dataclass(frozen=True)
class LateArm:
    """One late-interaction arm. Frozen, and declared before any score exists."""

    name: str
    checkpoint: str

    @property
    def licence(self) -> str:
        licence = LATE_INTERACTION_MODELS.get(self.checkpoint)
        if licence is None:
            raise ValueError(
                f"arm {self.name!r} names unregistered checkpoint {self.checkpoint!r}; record it "
                f"in recall.rerank.LATE_INTERACTION_MODELS with its licence first"
            )
        return licence

    @property
    def deployable(self) -> bool:
        return self.licence in PERMISSIVE_LICENCES


#: Frozen before any score was observed, per the project's preregistration standard.
LATE_ARMS: tuple[LateArm, ...] = (
    LateArm("li_colbertv2", "colbert-ir/colbertv2.0"),
    LateArm("li_answerai", "answerdotai/answerai-colbert-small-v1"),
    LateArm("li_jina", "jinaai/jina-colbert-v2"),
)


def holm_family(arms: Sequence[LateArm]) -> tuple[str, ...]:
    """The arm names forming one Holm-corrected family, refusing any non-deployable arm.

    This is the containment gate. The verdict that gates the follow-on project must not be
    computable from a family containing a non-commercial checkpoint, so the impossibility is
    mechanical rather than editorial.
    """
    # Materialised before it is read twice. A single-use iterator would be exhausted by the
    # blocked scan below and the return would then be an empty tuple, which is silent omission:
    # exactly what this gate exists to prevent. The annotation says Sequence, but a gate that
    # degrades to "quietly pass" on a type violation is not a gate.
    arms = list(arms)
    blocked = [a.name for a in arms if not a.deployable]
    if blocked:
        raise ValueError(
            f"non-deployable arms cannot enter a Holm family: {blocked}. Their licences "
            f"({[a.licence for a in arms if not a.deployable]}) make them diagnostic only, and "
            f"the preregistration fixes their effect as monotone: they may strengthen a null or "
            f"weaken a positive claim, never support a build decision. Report them separately."
        )
    return tuple(a.name for a in arms)


def arm_record(arm: LateArm) -> dict[str, object]:
    """The identity block stamped onto every emitted row, so a lifted number keeps its taint."""
    return {
        "arm": arm.name,
        "checkpoint": arm.checkpoint,
        "licence": arm.licence,
        "deployable": arm.deployable,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/mtrag/late_interaction.py tests/test_bench_late_interaction.py
git commit -m "bench(mtrag): late-interaction arms, and a Holm family that refuses the nc arm

Containment is a refusal: holm_family raises on a non-deployable arm, so the
verdict gating the follow-on project cannot include jina-colbert-v2."
```

---

### Task 5: The streaming scorer

Peak memory must not scale with corpus size. Documents are encoded once, scored against only the queries that reference them, and their token matrices discarded.

**Files:**
- Modify: `benchmarks/mtrag/late_interaction.py`
- Test: `tests/test_bench_late_interaction.py`

**Interfaces:**
- Consumes: `LateArm`, `arm_record` (Task 4); `maxsim` (Task 1).
- Produces: `score_stream(encoder, queries: dict[str, str], docs: Iterable[tuple[str, str]], pairs: dict[str, set[str]], batch_size: int = 32) -> Iterator[dict]` yielding `{"qid", "doc_id", "score"}`. `pairs` maps `doc_id -> {qid, ...}`, which is the inverted form of `pairs.jsonl` and is what makes the stream single-pass over documents.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bench_late_interaction.py`:

```python
import numpy as np

from benchmarks.mtrag.late_interaction import score_stream


class _FakeEncoder:
    def __init__(self, table):
        self._table = table
        self.passage_batches: list[list[str]] = []

    def query_embed(self, texts):
        return [np.array(self._table[t]) for t in texts]

    def passage_embed(self, texts):
        texts = list(texts)
        self.passage_batches.append(texts)
        return [np.array(self._table[t]) for t in texts]


def test_score_stream_emits_only_requested_pairs():
    table = {"qa": [[1.0, 0.0]], "qb": [[0.0, 1.0]], "d1": [[1.0, 0.0]], "d2": [[0.0, 1.0]]}
    rows = list(
        score_stream(
            _FakeEncoder(table),
            queries={"qa": "qa", "qb": "qb"},
            docs=[("d1", "d1"), ("d2", "d2")],
            pairs={"d1": {"qa"}, "d2": {"qa", "qb"}},
        )
    )
    assert {(r["qid"], r["doc_id"]) for r in rows} == {("qa", "d1"), ("qa", "d2"), ("qb", "d2")}


def test_score_stream_computes_maxsim():
    table = {"qa": [[1.0, 0.0]], "d1": [[1.0, 0.0]]}
    rows = list(
        score_stream(
            _FakeEncoder(table),
            queries={"qa": "qa"},
            docs=[("d1", "d1")],
            pairs={"d1": {"qa"}},
        )
    )
    assert rows == [{"qid": "qa", "doc_id": "d1", "score": pytest.approx(1.0)}]


def test_score_stream_encodes_each_document_exactly_once():
    """The point of streaming. A document referenced by many queries is encoded once, not once
    per pair, which is what makes this cheaper than 241,270 cross-encoder forward passes."""
    table = {"qa": [[1.0, 0.0]], "qb": [[0.0, 1.0]], "d1": [[1.0, 0.0]]}
    encoder = _FakeEncoder(table)
    list(
        score_stream(
            encoder,
            queries={"qa": "qa", "qb": "qb"},
            docs=[("d1", "d1")],
            pairs={"d1": {"qa", "qb"}},
        )
    )
    assert [t for batch in encoder.passage_batches for t in batch] == ["d1"]


def test_score_stream_batches_documents():
    table = {"qa": [[1.0, 0.0]], **{f"d{i}": [[1.0, 0.0]] for i in range(5)}}
    encoder = _FakeEncoder(table)
    list(
        score_stream(
            encoder,
            queries={"qa": "qa"},
            docs=[(f"d{i}", f"d{i}") for i in range(5)],
            pairs={f"d{i}": {"qa"} for i in range(5)},
            batch_size=2,
        )
    )
    assert [len(b) for b in encoder.passage_batches] == [2, 2, 1]


def test_score_stream_skips_documents_with_no_pairs():
    table = {"qa": [[1.0, 0.0]], "d1": [[1.0, 0.0]], "unused": [[1.0, 0.0]]}
    encoder = _FakeEncoder(table)
    rows = list(
        score_stream(
            encoder,
            queries={"qa": "qa"},
            docs=[("d1", "d1"), ("unused", "unused")],
            pairs={"d1": {"qa"}},
        )
    )
    assert [r["doc_id"] for r in rows] == ["d1"]
    assert [t for batch in encoder.passage_batches for t in batch] == ["d1"]


def test_score_stream_refuses_an_unknown_query_id():
    """A pair naming a query the caller did not supply is a dump/scorer mismatch, and scoring it
    as anything at all would fabricate a number. G3 depends on this raising."""
    table = {"qa": [[1.0, 0.0]], "d1": [[1.0, 0.0]]}
    with pytest.raises(KeyError, match="ghost"):
        list(
            score_stream(
                _FakeEncoder(table),
                queries={"qa": "qa"},
                docs=[("d1", "d1")],
                pairs={"d1": {"ghost"}},
            )
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: FAIL, `ImportError: cannot import name 'score_stream'`

- [ ] **Step 3: Write minimal implementation**

Append to `benchmarks/mtrag/late_interaction.py`:

```python
from collections.abc import Iterable, Iterator
from typing import Any

from recall.rerank import maxsim


def score_stream(
    encoder: Any,
    queries: dict[str, str],
    docs: Iterable[tuple[str, str]],
    pairs: dict[str, set[str]],
    batch_size: int = 32,
) -> Iterator[dict]:
    """Score every requested `(qid, doc_id)` pair, streaming the documents.

    🔑 This is the design decision that removes the GPU rental. A cross-encoder runs one forward
    pass PER PAIR (241,270 of them on 2026-08-07). Late interaction encodes the two sides
    independently, so each document is encoded ONCE and MaxSim'd against only the queries that
    reference it.

    Document token matrices are discarded after each batch. Materialising them would cost roughly
    7 GB at 128 dims (unique docs x ~180 tokens x 128 floats), and holding them buys nothing:
    peak memory here is independent of corpus size.

    `pairs` maps doc_id -> {qid}, the INVERTED form of `pairs.jsonl`. Inverting it is what makes a
    single pass over documents possible.

    A pair naming an unknown query raises: it means the dump and the scorer disagree, and any
    score emitted for it would be fabricated.
    """
    qids = list(queries)
    qmatrices = dict(zip(qids, encoder.query_embed([queries[q] for q in qids]), strict=True))
    for qid, matrix in qmatrices.items():
        if matrix.shape[0] == 0:
            raise ValueError(f"query {qid!r} has no tokens")

    batch: list[tuple[str, str]] = []

    def _flush() -> Iterator[dict]:
        if not batch:
            return
        matrices = list(encoder.passage_embed([text for _, text in batch]))
        for (doc_id, _), dmatrix in zip(batch, matrices, strict=True):
            for qid in sorted(pairs[doc_id]):
                if qid not in qmatrices:
                    raise KeyError(
                        f"pair references unknown query {qid!r} for document {doc_id!r}; the dump "
                        f"and the scorer disagree, and any score emitted here would be fabricated"
                    )
                # `-inf`, not a raise, and this MUST match `LateInteractionReranker.rerank`.
                # A zero-token document sorts last there rather than aborting the batch, and the
                # validate gate reranks a pool locally and compares it against these offloaded
                # scores. If this path raised instead, the gate would find the live reranker
                # ranking a document that has no offloaded score at all, and `rerank_order`
                # refuses a candidate with no score. The two paths agree or the offload is not a
                # substitute for the real reranker.
                score = (
                    float("-inf")
                    if dmatrix.shape[0] == 0
                    else maxsim(qmatrices[qid], dmatrix)
                )
                yield {"qid": qid, "doc_id": doc_id, "score": score}
        batch.clear()

    for doc_id, text in docs:
        if doc_id not in pairs:
            continue  # no query asked for this document; encoding it would be wasted work
        batch.append((doc_id, text))
        if len(batch) >= batch_size:
            yield from _flush()
    yield from _flush()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: PASS, 18 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/mtrag/late_interaction.py tests/test_bench_late_interaction.py
git commit -m "bench(mtrag): streaming MaxSim scorer, encoding each document once

Peak memory independent of corpus size: materialising doc token matrices would
cost ~7 GB at 128 dims and buys nothing."
```

---

### Task 6: The CLI, with G3 completeness and G5 mutation check

**Files:**
- Modify: `benchmarks/mtrag/late_interaction.py`
- Test: `tests/test_bench_late_interaction.py`

**Interfaces:**
- Consumes: `score_stream`, `arm_record`, `LATE_ARMS` (Tasks 4-5).
- Produces: `load_pairs_inverted(path: Path) -> dict[str, set[str]]`; `assert_complete(pairs: dict[str, set[str]], scored: dict[str, set[str]]) -> None` raising on any unscored pair; `main(argv: list[str] | None = None) -> int` with subcommands `score` and `validate`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bench_late_interaction.py`:

```python
import json
from pathlib import Path

from benchmarks.mtrag.late_interaction import assert_complete, load_pairs_inverted


def test_load_pairs_inverted(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"qid": "q1", "doc_id": "d1"}\n'
        '{"qid": "q2", "doc_id": "d1"}\n'
        '{"qid": "q1", "doc_id": "d2"}\n',
        encoding="utf-8",
    )
    assert load_pairs_inverted(path) == {"d1": {"q1", "q2"}, "d2": {"q1"}}


def test_load_pairs_inverted_ignores_blank_lines(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"qid": "q1", "doc_id": "d1"}\n\n', encoding="utf-8")
    assert load_pairs_inverted(path) == {"d1": {"q1"}}


def test_assert_complete_passes_when_every_pair_is_scored():
    # No assert: `assert_complete` returns None and signals success by NOT raising, so the call
    # completing IS the assertion. `assert f(...) is None` would read as a test that checks
    # nothing, which is worse than no assert at all.
    assert_complete({"d1": {"q1"}}, {"d1": {"q1"}})


def test_assert_complete_raises_on_a_missing_score():
    """G3. A missing score does NOT raise on its own — it sinks the document to the bottom of the
    ranking. That is the `ef_search` failure shape: `_query_learned_sparse` returned 6 of 100 and
    no test caught it, a timing anomaly did. So counts are asserted, never assumed."""
    with pytest.raises(ValueError, match="1 pair"):
        assert_complete({"d1": {"q1", "q2"}}, {"d1": {"q1"}})


def test_assert_complete_raises_on_a_wholly_unscored_document():
    with pytest.raises(ValueError, match="1 pair"):
        assert_complete({"d1": {"q1"}, "d2": {"q1"}}, {"d1": {"q1"}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: FAIL, `ImportError: cannot import name 'assert_complete'`

- [ ] **Step 3: Write minimal implementation**

Append to `benchmarks/mtrag/late_interaction.py`:

```python
import argparse
import json
import sys
import time
from pathlib import Path


def load_pairs_inverted(path: Path) -> dict[str, set[str]]:
    """Read `pairs.jsonl` as doc_id -> {qid}, the form `score_stream` needs."""
    pairs: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.setdefault(str(row["doc_id"]), set()).add(str(row["qid"]))
    return pairs


def assert_complete(pairs: dict[str, set[str]], scored: dict[str, set[str]]) -> None:
    """G3: every requested pair received a score.

    A missing score does not raise on its own. `rerank_order` raises when it later meets an
    unscored candidate, but only for candidates in a pool — a pair dropped before that reaches
    nothing that checks it. So the count is asserted here, at the point the scorer claims it is
    done, rather than assumed downstream.
    """
    missing = [
        (doc_id, qid)
        for doc_id, qids in pairs.items()
        for qid in qids
        if qid not in scored.get(doc_id, set())
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} pair(s) received no score, e.g. {missing[:3]}. A missing score is not "
            f"a zero: it would sink that document to the bottom of the ranking silently."
        )


def _resolve_arm(name: str, accept_noncommercial: bool) -> LateArm:
    """The named arm, refusing a non-deployable one without an explicit opt-in.

    Shared by `score` and `validate`. `validate` used to pass
    `accept_noncommercial_license=not arm.deployable`, which handed itself the waiver precisely
    for the arms the gate exists to withhold it from. The preregistration's containment gate
    requires an EXPLICIT opt-in on every entry point, not only the convenient one, so the check
    lives in one place that both commands must go through.
    """
    arm = next((a for a in LATE_ARMS if a.name == name), None)
    if arm is None:
        raise SystemExit(f"unknown arm {name!r}; known arms are {[a.name for a in LATE_ARMS]}")
    if not arm.deployable and not accept_noncommercial:
        raise SystemExit(
            f"{arm.name} is licensed {arm.licence} and needs --accept-noncommercial. It is a "
            f"capacity diagnostic only and may not contribute to a shipping decision."
        )
    return arm


def cmd_score(args: argparse.Namespace) -> int:
    from recall.rerank import LateInteractionReranker

    out = args.output_dir.resolve()
    arm = _resolve_arm(args.arm, args.accept_noncommercial)

    reranker = LateInteractionReranker.from_pretrained(
        arm.checkpoint, accept_noncommercial_license=args.accept_noncommercial
    )

    queries: dict[str, str] = {}
    with (out / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                queries[str(row["qid"])] = str(row["text"])

    pairs = load_pairs_inverted(out / "pairs.jsonl")

    def _docs():
        with (out / "docs.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    yield str(row["doc_id"]), str(row["text"])

    import fastembed

    header = {
        "_header": True,
        **arm_record(arm),
        # G4: the artifact records the encoder identity, so a venv change is detectable rather
        # than silent. The SPLADE run's standing warning is to re-verify after any venv change.
        "fastembed_version": fastembed.__version__,
        "queries": len(queries),
        "pairs_requested": sum(len(v) for v in pairs.values()),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    scored: dict[str, set[str]] = {}
    started = time.perf_counter()
    with args.scores.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for n, row in enumerate(
            score_stream(reranker._encoder, queries, _docs(), pairs, args.batch_size), 1
        ):
            scored.setdefault(row["doc_id"], set()).add(row["qid"])
            fh.write(json.dumps(row) + "\n")
            if n % 20000 == 0:
                print(json.dumps({
                    "event": "progress", "arm": arm.name, "pairs": n,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }), flush=True)

    assert_complete(pairs, scored)
    print(json.dumps({
        "event": "score_done", **arm_record(arm),
        "pairs_scored": sum(len(v) for v in scored.values()),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score", help="score the dumped pairs with a late-interaction model")
    p.add_argument("--output-dir", type=Path, required=True, help="the rerank_offload dump dir")
    p.add_argument("--scores", type=Path, required=True, help="jsonl to write")
    p.add_argument("--arm", required=True, choices=[a.name for a in LATE_ARMS])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--accept-noncommercial",
        action="store_true",
        help="required for cc-by-nc checkpoints; diagnostic use only",
    )
    p.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: PASS, 32 passed

- [ ] **Step 5: Check for unused imports**

Run: `python -m pyflakes benchmarks/mtrag/late_interaction.py recall/rerank.py`
Expected: no output

- [ ] **Step 6: Commit**

```bash
git add benchmarks/mtrag/late_interaction.py tests/test_bench_late_interaction.py
git commit -m "bench(mtrag): score CLI with the G3 completeness assertion

A missing score is not a zero, it sinks the document silently. Counts are
asserted at the point the scorer claims it is done."
```

---

### Task 7: The validate gate (G2), and proving it can fail (G5)

A gate that cannot fail is exactly as useless as one that cannot fire, and this line of work has already produced two of the latter.

**Files:**
- Modify: `benchmarks/mtrag/late_interaction.py`
- Test: `tests/test_bench_late_interaction.py`

**Interfaces:**
- Consumes: `score_stream`, `LATE_ARMS` (Tasks 4-5); `rerank_order`, `ORDER_EXACT_K`, `EVAL_K`, `SCORE_TOLERANCE` from `benchmarks.mtrag.rerank_offload`.
- Produces: `compare_orderings(local: list[str], offloaded: list[str], local_by_id: dict[str, float], task_id: str) -> tuple[dict | None, bool]` in `rerank_offload.py`; `validate_sample(reranker, rows: list[dict], docs: dict[str, str], scores: dict[str, dict[str, float]]) -> dict` returning `{"verdict", "sampled", "max_score_delta", "score_tolerance", "scores_within_tolerance", "failures", "deep_tie_count"}`; a `validate` subcommand on `main`.

**⚠️ Steps 1-5 refactor EXISTING code before any new code is written.** Without it `validate_sample` would be a near-copy of `rerank_offload.cmd_validate`'s three-branch comparison cascade, and the two definitions of "mismatch" would drift. Only the cascade is shared. Each caller still computes its own local scores and still drives the REAL reranker, which is what the gate exists to check.

The refactor touches `cmd_validate` only, never `cmd_dump` or `cmd_apply`, so the G1 reproduction path in Task 10 is untouched.

- [ ] **Step 1: Write the failing test for the extracted helper**

Append to `tests/test_rerank_offload.py`:

```python
from benchmarks.mtrag.rerank_offload import compare_orderings


def test_compare_orderings_reports_no_failure_when_orders_match():
    failure, tie = compare_orderings(
        local=["a", "b", "c"],
        offloaded=["a", "b", "c"],
        local_by_id={"a": 3.0, "b": 2.0, "c": 1.0},
        task_id="t1",
    )
    assert failure is None
    assert tie is False


def test_compare_orderings_flags_a_top_k_order_difference():
    failure, tie = compare_orderings(
        local=["a", "b", "c"],
        offloaded=["b", "a", "c"],
        local_by_id={"a": 3.0, "b": 2.0, "c": 1.0},
        task_id="t1",
    )
    assert failure == {"task_id": "t1", "why": "top-10 order differs"}
    assert tie is False


def test_compare_orderings_reports_a_deep_tie_rather_than_a_failure():
    """Past the metric cutoffs a swap is a near-tie and is information, not failure. The first
    version of this gate demanded exact ordering over the whole pool and COULD NOT PASS: CUDA and
    CPU do not produce bit-identical floats."""
    local = [f"d{i}" for i in range(120)]
    offloaded = local[:100] + [local[101], local[100]] + local[102:]
    failure, tie = compare_orderings(
        local=local,
        offloaded=offloaded,
        local_by_id={c: float(1000 - i) for i, c in enumerate(local)},
        task_id="t1",
    )
    assert failure is None
    assert tie is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rerank_offload.py -v -k compare_orderings`
Expected: FAIL, `ImportError: cannot import name 'compare_orderings'`

- [ ] **Step 3: Extract the helper and route `cmd_validate` through it**

Add to `benchmarks/mtrag/rerank_offload.py`, immediately above `cmd_validate`:

```python
def compare_orderings(
    local: list[str],
    offloaded: list[str],
    local_by_id: dict[str, float],
    task_id: str,
) -> tuple[dict | None, bool]:
    """Compare one query's local ordering against its offloaded one.

    Returns `(failure_or_None, is_deep_tie)`. Shared by the cross-encoder and late-interaction
    validate gates: they differ in how `local_by_id` is computed, not in what counts as a
    mismatch, and two copies of that definition would drift apart.

    What this guarantees, and deliberately no more: order is exact where metrics are cut, and the
    top-`EVAL_K` SET is exact so Recall@100 is unaffected. Deeper swaps are near-ties, reported as
    information — see SCORE_TOLERANCE above for why demanding more is a gate that cannot pass.
    """
    if local[:ORDER_EXACT_K] != offloaded[:ORDER_EXACT_K]:
        return {"task_id": task_id, "why": f"top-{ORDER_EXACT_K} order differs"}, False
    if set(local[:EVAL_K]) != set(offloaded[:EVAL_K]):
        return {"task_id": task_id, "why": f"top-{EVAL_K} set differs"}, False
    if local != offloaded:
        return None, True
    return None, False
```

Then replace the three-branch cascade inside `cmd_validate`'s loop. It currently reads:

```python
        # (a) top-of-ranking ORDER, where every reported metric is cut.
        if local[:ORDER_EXACT_K] != offloaded[:ORDER_EXACT_K]:
            failures.append({"task_id": row["task_id"], "why": f"top-{ORDER_EXACT_K} order differs"})
        # (b) top-100 SET, which is what Recall@100 counts (order within it does not matter).
        elif set(local[:EVAL_K]) != set(offloaded[:EVAL_K]):
            failures.append({"task_id": row["task_id"], "why": f"top-{EVAL_K} set differs"})
        elif local != offloaded:
            rank = next(i for i, (a, b) in enumerate(zip(local, offloaded, strict=True)) if a != b)
            gap = abs(local_by_id[local[rank]] - local_by_id[offloaded[rank]])
            ties.append({"task_id": row["task_id"], "rank": rank, "score_gap": gap})
```

Replace with:

```python
        failure, is_tie = compare_orderings(local, offloaded, local_by_id, row["task_id"])
        if failure is not None:
            failures.append(failure)
        elif is_tie:
            rank = next(i for i, (a, b) in enumerate(zip(local, offloaded, strict=True)) if a != b)
            gap = abs(local_by_id[local[rank]] - local_by_id[offloaded[rank]])
            ties.append({"task_id": row["task_id"], "rank": rank, "score_gap": gap})
```

The rank/gap detail stays at this call site: it is this command's reporting, not part of the shared decision.

- [ ] **Step 4: Run the existing offload tests to prove the refactor changed no behaviour**

Run: `python -m pytest tests/test_rerank_offload.py -v`
Expected: PASS, the three new `compare_orderings` tests plus every pre-existing test.

**If a pre-existing test fails, the refactor is wrong. Revert and redo it. Do not adjust the test.**

- [ ] **Step 5: Commit the refactor on its own**

```bash
git add benchmarks/mtrag/rerank_offload.py tests/test_rerank_offload.py
git commit -m "refactor(mtrag): extract the ordering comparison both validate gates need

Behaviour-preserving; the pre-existing offload tests pass unchanged. The
late-interaction gate needs the same cascade over different local scores, and
two copies would drift."
```

- [ ] **Step 6: Write the failing test for validate_sample**

Append to `tests/test_bench_late_interaction.py`:

```python
from benchmarks.mtrag.late_interaction import validate_sample
from recall.rerank import LateInteractionReranker


def _live_reranker(table):
    return LateInteractionReranker(_FakeEncoder(table), model_name="colbert-ir/colbertv2.0")
```

⚠️ **Multi-token documents are load-bearing in the fixture below, and this is easy to get wrong.**
With one token per document `mean` and `max` coincide, the G5 mutation becomes invisible, and the
test that proves the gate can fail would itself fail. `multi` carries two tokens precisely so the
mutation flips the ORDER, which is what `validate_sample` compares:

| | multi | mid | far | resulting order |
|---|---|---|---|---|
| `max` (correct) | 1.0 | 0.6 | 0.0 | `[multi, mid, far]` |
| `mean` (mutated) | 0.5 | 0.6 | 0.0 | `[mid, multi, far]` |

Verified numerically before this plan was written.

```python
_TABLE = {
    "q": [[1.0, 0.0]],
    "multi": [[1.0, 0.0], [0.0, 1.0]],  # maxsim 1.0, but MEAN 0.5
    "mid": [[0.6, 0.8]],                # maxsim 0.6, and mean 0.6
    "far": [[0.0, 1.0]],                # maxsim 0.0
}
_ROWS = [{"task_id": "t1", "query": "q", "candidates": ["far", "mid", "multi"]}]
_DOCS = {"far": "far", "mid": "mid", "multi": "multi"}


def test_validate_matches_when_offloaded_scores_agree():
    scores = {"t1": {"far": 0.0, "mid": 0.6, "multi": 1.0}}
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, scores)
    assert report["verdict"] == "MATCH"
    assert report["max_score_delta"] < 1e-9


def test_validate_mismatches_when_the_offloaded_order_is_wrong():
    """G2's whole purpose. An offloaded ordering that merely looks reasonable produces a
    publishable nDCG that RE-call itself would never compute."""
    scores = {"t1": {"far": 9.0, "mid": 0.6, "multi": 1.0}}
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, scores)
    assert report["verdict"] == "MISMATCH"
    assert report["failures"]


def test_the_mutation_fixture_actually_mutates():
    """Guards the guard below. If mean and max ever coincide on this fixture, the G5 test passes
    vacuously and proves nothing — which is the exact failure mode G5 exists to prevent."""
    query = np.array(_TABLE["q"])
    for doc in ("far", "mid", "multi"):
        tokens = np.array(_TABLE[doc])
        assert float((query @ tokens.T).max(axis=1).sum()) == pytest.approx(
            {"far": 0.0, "mid": 0.6, "multi": 1.0}[doc]
        )
    assert float((query @ np.array(_TABLE["multi"]).T).mean(axis=1).sum()) == pytest.approx(0.5)


def test_validate_detects_the_mean_for_max_mutation():
    """G5, as an automated test rather than a manual ritual. Scores computed with `mean` instead
    of `max` must make the gate go RED. If this passes, the gate is vacuous."""
    query = np.array(_TABLE["q"])
    mutated = {
        "t1": {
            doc: float((query @ np.array(_TABLE[doc]).T).mean(axis=1).sum())
            for doc in ("far", "mid", "multi")
        }
    }
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, mutated)
    assert report["verdict"] == "MISMATCH"


def test_validate_reports_the_worst_score_delta():
    scores = {"t1": {"far": 0.0, "mid": 0.6, "multi": 1.0004}}
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, scores)
    assert report["max_score_delta"] == pytest.approx(0.0004, abs=1e-9)


def test_validate_sample_names_a_task_with_no_offloaded_scores():
    """An incomplete input must not be reported as a MISMATCH: that verdict means the
    offloaded ordering disagreed with the live reranker, which was never measured here."""
    with pytest.raises(ValueError, match="no offloaded scores for task"):
        validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, {})


def test_validate_sample_names_the_candidates_it_is_missing():
    partial = {"t1": {"far": 0.0}}
    with pytest.raises(ValueError, match="missing offloaded scores for 2 of 3"):
        validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, partial)
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: FAIL, `ImportError: cannot import name 'validate_sample'`

- [ ] **Step 8: Write minimal implementation**

Append to `benchmarks/mtrag/late_interaction.py`:

```python
from benchmarks.mtrag.rerank_offload import (
    SCORE_TOLERANCE,
    compare_orderings,
    rerank_order,
)


def validate_sample(
    reranker: Any,
    rows: list[dict],
    docs: dict[str, str],
    scores: dict[str, dict[str, float]],
) -> dict:
    """G2: require the offloaded ordering to match the real reranker on a sample.

    The comparison cascade and the tolerance are `rerank_offload`'s, shared rather than
    re-derived. That module already learned that demanding exact ordering over a whole pool is a
    gate that CANNOT PASS, because CUDA and CPU do not produce bit-identical floats and near-ties
    swap for reasons unrelated to correctness. Only the LOCAL SCORING differs here: MaxSim over
    independently encoded sides, rather than a cross-encoder's joint forward pass. The definition
    of a mismatch is identical, and two copies of it would drift.
    """
    from recall.types import Chunk, ScoredChunk

    failures: list[dict] = []
    ties = 0
    worst_delta = 0.0
    for row in rows:
        candidates = row["candidates"]
        hits = [
            ScoredChunk(chunk=Chunk(id=c, source="s", text=docs[c], metadata={}), score=0.0)
            for c in candidates
        ]
        # An incomplete scores file is an INPUT error, not a MISMATCH: MISMATCH means the
        # offloaded ordering disagreed with the live reranker, and reporting a truncated run as a
        # measured disagreement would archive it as evidence of something that was never measured.
        # So this raises with what is actually missing, rather than a bare KeyError naming one key.
        if row["task_id"] not in scores:
            raise ValueError(
                f"no offloaded scores for task {row['task_id']!r}: the scores file is incomplete "
                f"for this pool. Re-run `score`, which asserts completeness before it exits."
            )
        offloaded_scores = scores[row["task_id"]]
        unscored = [c for c in candidates if c not in offloaded_scores]
        if unscored:
            raise ValueError(
                f"task {row['task_id']!r} is missing offloaded scores for {len(unscored)} of "
                f"{len(candidates)} candidates, e.g. {unscored[:3]}. The scores file is "
                f"incomplete. Re-run `score`, which asserts completeness before it exits."
            )

        qtokens = list(reranker._encoder.query_embed([row["query"]]))[0]
        dtokens = list(reranker._encoder.passage_embed([h.chunk.text for h in hits]))
        local_by_id = {
            c: maxsim(qtokens, d) for c, d in zip(candidates, dtokens, strict=True)
        }
        worst_delta = max(
            worst_delta, max(abs(offloaded_scores[c] - local_by_id[c]) for c in candidates)
        )

        local = [h.chunk.id for h in reranker.rerank(row["query"], hits)]
        offloaded = rerank_order(candidates, offloaded_scores)

        failure, is_tie = compare_orderings(local, offloaded, local_by_id, row["task_id"])
        if failure is not None:
            failures.append(failure)
        elif is_tie:
            ties += 1

    within = worst_delta < SCORE_TOLERANCE
    return {
        "verdict": "MATCH" if not failures and within else "MISMATCH",
        "sampled": len(rows),
        "max_score_delta": worst_delta,
        "score_tolerance": SCORE_TOLERANCE,
        "scores_within_tolerance": within,
        "failures": failures[:5],
        "deep_tie_count": ties,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    from recall.rerank import LateInteractionReranker

    out = args.output_dir.resolve()
    arm = _resolve_arm(args.arm, args.accept_noncommercial)
    reranker = LateInteractionReranker.from_pretrained(
        arm.checkpoint, accept_noncommercial_license=args.accept_noncommercial
    )

    docs: dict[str, str] = {}
    with (out / "docs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                docs[str(row["doc_id"])] = str(row["text"])

    scores: dict[str, dict[str, float]] = {}
    with args.scores.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_header"):
                print(json.dumps({"event": "scores_header", **row}), flush=True)
                continue
            scores.setdefault(str(row["qid"]), {})[str(row["doc_id"])] = float(row["score"])

    pool_path = next(iter(sorted((out / "pools").glob("*.jsonl"))))
    rows = []
    with pool_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    report = validate_sample(reranker, rows[: args.sample], docs, scores)
    print(json.dumps({"event": "validate", "arm": arm.name, **report}, indent=2), flush=True)
    return 0 if report["verdict"] == "MATCH" else 1
```

Register the subcommand inside `main`, immediately before `args = parser.parse_args(argv)`:

```python
    v = sub.add_parser("validate", help="require the offloaded ordering to match the real reranker")
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--scores", type=Path, required=True)
    v.add_argument("--arm", required=True, choices=[a.name for a in LATE_ARMS])
    v.add_argument("--sample", type=int, default=20)
    v.add_argument(
        "--accept-noncommercial",
        action="store_true",
        help="required for cc-by-nc checkpoints; diagnostic use only",
    )
    v.set_defaults(func=cmd_validate)
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_late_interaction.py -v`
Expected: PASS, 34 passed

- [ ] **Step 10: Run the full suite for regressions**

Run: `python -m pytest tests/test_rerank.py tests/test_rerank_offload.py tests/test_late_interaction_rerank.py tests/test_bench_late_interaction.py -q`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add benchmarks/mtrag/late_interaction.py tests/test_bench_late_interaction.py
git commit -m "bench(mtrag): the G2 validate gate, and a test proving it can fail

G5's mutation check is automated rather than a manual ritual: scores computed
with mean instead of max must turn the gate red. Two guards in this line of
work already shipped unable to fire."
```

---

### Task 8: Family B power precondition

**This task BLOCKS freezing Family B.** Its output decides whether D1 and D2 are preregistered with p-values or demoted to descriptive diagnostics. Per the spec, if the cell is underpowered the demotion is automatic and is not a judgement call made after seeing the counts.

**Files:**
- Create: `benchmarks/mtrag/buried_gold_power.py`
- Test: `tests/test_buried_gold_power.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `mcnemar_power(n: int, p_control: float, p_treatment: float, rho: float, alpha: float = 0.05, trials: int = 20000, seed: int = 0) -> float`; `minimum_detectable_shift(n: int, p_control: float, rho: float, target_power: float = 0.80, alpha: float = 0.05) -> float | None` returning the largest treatment bury-rate still detectable reaching `target_power`, or `None` if no rate in `[0, p_control]` does.

- [ ] **Step 1: Write the failing test**

Create `tests/test_buried_gold_power.py`:

```python
import pytest

from benchmarks.mtrag.buried_gold_power import mcnemar_power, minimum_detectable_shift


def test_no_effect_gives_power_near_alpha():
    """With treatment == control the test should reject at roughly its nominal rate. This is the
    sanity check that the simulation is a test and not a rubber stamp."""
    power = mcnemar_power(n=123, p_control=0.73, p_treatment=0.73, rho=0.5, trials=20000, seed=1)
    assert power < 0.10


def test_large_effect_is_well_powered():
    power = mcnemar_power(n=123, p_control=0.73, p_treatment=0.35, rho=0.5, trials=20000, seed=1)
    assert power > 0.95


def test_power_increases_with_effect_size():
    small = mcnemar_power(n=123, p_control=0.73, p_treatment=0.68, rho=0.5, trials=20000, seed=1)
    large = mcnemar_power(n=123, p_control=0.73, p_treatment=0.50, rho=0.5, trials=20000, seed=1)
    assert large > small


def test_power_increases_with_n():
    small_n = mcnemar_power(n=40, p_control=0.73, p_treatment=0.55, rho=0.5, trials=20000, seed=1)
    large_n = mcnemar_power(n=400, p_control=0.73, p_treatment=0.55, rho=0.5, trials=20000, seed=1)
    assert large_n > small_n


def test_minimum_detectable_shift_is_below_control():
    mds = minimum_detectable_shift(n=123, p_control=0.73, rho=0.5)
    assert mds is not None
    assert 0.0 <= mds < 0.73


def test_minimum_detectable_shift_is_actually_powered():
    mds = minimum_detectable_shift(n=123, p_control=0.73, rho=0.5)
    assert mcnemar_power(n=123, p_control=0.73, p_treatment=mds, rho=0.5, trials=20000, seed=2) >= 0.78


def test_tiny_n_may_be_unpowered_at_any_effect():
    """The case the spec's demotion rule exists for. `None` means no shift in range is detectable,
    and Family B becomes descriptive with no p-value attached."""
    assert minimum_detectable_shift(n=3, p_control=0.73, rho=0.5, target_power=0.99) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_buried_gold_power.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'benchmarks.mtrag.buried_gold_power'`

- [ ] **Step 3: Write minimal implementation**

Create `benchmarks/mtrag/buried_gold_power.py`:

```python
"""Family B power precondition: can the 123-document cell detect anything?

Preregistration: `docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md`.

🔑 This BLOCKS freezing Family B, and the standard it serves is specific:
`feedback-check-the-deciding-cell-has-power-2026-08-06` records a prior session that built three
guards which could not fire, could not pass, or rested on n=8. A design whose deciding cell cannot
resolve the effect it is looking for produces a null that means nothing.

The design is PAIRED: the same 123 gold documents are ranked by the control (MiniLM buries 90) and
by the treatment, so the relevant test is McNemar's on the discordant pairs. `rho` is the
tetrachoric-style association between the two rankers' bury decisions. It is NOT a free parameter
to tune until the answer is pleasant: it is estimated from the MiniLM/BGE agreement in the
2026-08-07 archive (90 and 91 of the same 123), and reported alongside the result.

Power is simulated rather than derived from a closed form, because the closed forms for McNemar
disagree at small discordant counts and the exact binomial test is what will actually be run.
"""

from __future__ import annotations

import random
from math import comb


def _binom_two_sided_p(b: int, n_discordant: int) -> float:
    """Exact two-sided binomial p at p=0.5, which is McNemar's exact test."""
    if n_discordant == 0:
        return 1.0
    k = min(b, n_discordant - b)
    tail = sum(comb(n_discordant, i) for i in range(k + 1)) / (2 ** n_discordant)
    return min(1.0, 2.0 * tail)


def mcnemar_power(
    n: int,
    p_control: float,
    p_treatment: float,
    rho: float,
    alpha: float = 0.05,
    trials: int = 20000,
    seed: int = 0,
) -> float:
    """Simulated power of McNemar's exact test on `n` paired binary outcomes.

    `p_control` and `p_treatment` are bury RATES (lower is better for the treatment). `rho` is the
    probability that the treatment simply copies the control's decision, which models two rankers
    agreeing on the easy cases; the remainder is drawn independently. That is a deliberately
    simple association model, and its value is reported with the result rather than hidden.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    rng = random.Random(seed)
    rejections = 0
    for _ in range(trials):
        b = c = 0  # b: control buried, treatment did not. c: the reverse.
        for _ in range(n):
            control = rng.random() < p_control
            if rng.random() < rho:
                treatment = control
            else:
                treatment = rng.random() < p_treatment
            if control and not treatment:
                b += 1
            elif treatment and not control:
                c += 1
        if _binom_two_sided_p(b, b + c) < alpha:
            rejections += 1
    return rejections / trials


def minimum_detectable_shift(
    n: int,
    p_control: float,
    rho: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
) -> float | None:
    """The largest treatment bury-rate still detectable at `target_power`, or None if none is.

    Scans downward from `p_control` in steps of 0.01. `None` is the answer the spec's demotion
    rule keys on: it means no shift in range is detectable, so Family B carries no p-value.
    """
    rate = p_control
    while rate >= 0.0:
        if mcnemar_power(n, p_control, rate, rho, alpha, trials=4000, seed=7) >= target_power:
            return round(rate, 4)
        rate = round(rate - 0.01, 4)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_buried_gold_power.py -v`
Expected: PASS, 7 passed. Allow up to ~2 minutes; these are simulations.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/mtrag/buried_gold_power.py tests/test_buried_gold_power.py
git commit -m "bench(mtrag): Family B power precondition, which blocks freezing it

n=123 with 90 buried is a paired binary design. If no shift in range reaches
0.80 power, Family B is demoted to descriptive with no p-value, and that
consequence is fixed before the counts are visible."
```

---

### Task 9: Run the precondition and record its answer in the spec

This is the gate between building and measuring. It touches VPS2 and produces the number that decides Family B's status.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md` (the "Power" section only)

- [ ] **Step 1: Recover the buried-gold cell from the archive**

On VPS2 (`root@100.91.148.25`, key `~/.ssh/contabo_sentiment`), read
`/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/`. Recover three things and record where each came from:

1. the 123 gold document ids reachable only via SPLADE,
2. how many each reranker buries below rank 10 (expected: MiniLM 90, BGE 91),
3. the equal-reranking width used for `mq_nested2_nogold`, which Family C's V1 needs.

⚠️ `pkill -f <pat>` matches its own ssh command line and kills the shell running it. Use `[p]at`.
⚠️ Heredocs over ssh mangle quotes. Use `scp` for anything non-trivial. `scp` takes `-P`, `ssh` takes `-p`.

- [ ] **Step 2: Estimate rho from the two rerankers' agreement**

MiniLM buries 90 and BGE buries 91 of the same 123. Compute the actual 2x2 agreement table over the two rankers' bury decisions, and derive `rho` as the observed copy-rate. Do not assume 0.5.

- [ ] **Step 3: Compute the minimum detectable shift**

Run:

```bash
python -c "from benchmarks.mtrag.buried_gold_power import minimum_detectable_shift as m; print(m(n=123, p_control=90/123, rho=RHO))"
```

substituting the `rho` from Step 2.

⚠️ **A preliminary figure, computed while writing this plan, so nobody is surprised by it.** At a
placeholder `rho=0.5` the answer is **0.46**, verified at 0.8095 power on an independent seed.

Read what that means before running the real thing: the control buries 90/123 = **0.7317**, so
reaching 80% power needs the treatment to bury only **~57 of 123**, a rescue of roughly **33
documents**. Family B can detect a large effect and nothing subtler.

Two consequences the implementer should carry into Step 4:

1. **`rho` moves this a lot, and not in the intuitive direction.** Higher agreement between the two
   rankers means *fewer discordant pairs*, and McNemar's test sees only discordant pairs, so a
   higher `rho` gives **less** power, not more. MiniLM and BGE burying 90 and 91 of the same 123
   suggests agreement is high, so the real `rho` is plausibly well above 0.5 and the real minimum
   detectable shift correspondingly harsher.
2. **This is a preliminary number and must not be quoted as the result.** It uses a placeholder
   `rho` and the archive has not been read yet. Step 4 records the real one.

- [ ] **Step 4: Append the answer to the spec's Power section**

Add under "Power: does the deciding cell have any?", without editing anything above it:

```markdown
### Precondition result, computed YYYY-MM-DD before the arms ran

| | |
|---|---|
| n | 123 |
| control bury rate (MiniLM) | 90/123 = 0.7317 |
| rho, estimated from MiniLM/BGE agreement | <value> |
| minimum detectable bury rate at 0.80 power | <value or None> |
| **Family B status** | **PREREGISTERED / DEMOTED TO DESCRIPTIVE** |

Family C's V1 width, read from the archive: <value>.
```

If the minimum detectable shift is `None`, or if it is so close to 0.7317 that only an implausibly large rescue is detectable, write **DEMOTED TO DESCRIPTIVE** and state that D1 and D2 carry no p-value. That follows from the rule fixed before the number was seen.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md
git commit -m "docs(spec): Family B power precondition result, before any arm ran"
```

---

### Task 10: Execute the gates and the arms

**Files:** none modified. This produces artifacts.

- [ ] **Step 1: G1, reproduction**

Re-run `rr_minilm` through `rerank_offload` on the existing dump and require 0.7603 R@100 / 0.3769 nDCG@5 to four significant figures.

**If G1 fails, STOP.** Nothing else is read. It means the path is wrong, not that reproduction is hard: the 2026-08-07 run already reproduced these across different hardware and a different code path.

- [ ] **Step 2: G4, encoder identity**

Record the `fastembed` version and checkpoint in every scores header (already automatic via `cmd_score`). Confirm the same venv scores queries and documents. Re-verify after any venv change.

- [ ] **Step 3: Score the two deployable arms**

```bash
python -m benchmarks.mtrag.late_interaction score --output-dir DUMP --scores scores_colbertv2.jsonl --arm li_colbertv2
```

```bash
python -m benchmarks.mtrag.late_interaction score --output-dir DUMP --scores scores_answerai.jsonl --arm li_answerai
```

G3 runs automatically at the end of each. Record wall-clock, which tests prediction P4 (no GPU rental needed).

- [ ] **Step 4: G2, validate each arm**

```bash
python -m benchmarks.mtrag.late_interaction validate --output-dir DUMP --scores scores_colbertv2.jsonl --arm li_colbertv2 --sample 20
```

Expected: `"verdict": "MATCH"`. **A MISMATCH stops the run.**

- [ ] **Step 5: Score the diagnostic arm**

```bash
python -m benchmarks.mtrag.late_interaction score --output-dir DUMP --scores scores_jina.jsonl --arm li_jina --accept-noncommercial
```

Then validate it the same way. Its numbers are diagnostic only and may not enter a Holm family, which `holm_family` enforces.

- [ ] **Step 6: Compute the contrasts**

Run `benchmarks/mtrag/analyse_contrasts.py` unchanged for Family A (C1, C2, C3), and separately for Family C (V1) on the `mq_nested2_nogold` pools at the width recovered in Task 9.

Report every figure with its CI. A point estimate is not a result.

⚠️ **The Holm family MUST be built by `benchmarks.mtrag.late_interaction.holm_family`, not by
hand.** It is the gate that makes `li_jina` mechanically unable to enter a shipping-relevant
family, and until this step calls it, it is protecting nothing. Passing the arm list to it is the
precondition for reporting any Holm-corrected contrast, not a stylistic preference.

- [ ] **Step 7: Apply the decision rule**

C1 >= +0.010 nDCG@5 and Holm-significant within Family A, with no veto tripped (R@100 or nDCG@10 regression whose CI excludes zero). Do not renegotiate the bar, it was copied from the 2026-08-06 preregistration precisely so it could not be.

---

### Task 11: Append RESULTS and archive

**Files:**
- Modify: `docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md` (append below a `# RESULTS` heading only)

- [ ] **Step 1: Append the RESULTS section**

Nothing above the `# RESULTS` line may be edited. Include the contrast table with CIs and Holm flags, the gate outcomes, and how each of P1 to P4 landed, including where they were wrong.

State the capacity reading in the bounded phrasing the spec fixes: the spread here is 5x against the primary and 17x at its widest, not the 25x that closed the reranker lever, so a shared null licenses "capacity does not appear to be the binding constraint over 33M to 560M" and nothing stronger.

- [ ] **Step 2: Archive on VPS2**

Write to `/var/lib/recall-benchmarks/YYYY-MM-DD-mtrag-late-interaction/` with a SHA256 manifest and a `NOTE.md` carrying the caveats, matching the 2026-08-06 and 2026-08-07 archives.

⚠️ Check liveness of the JOB, not one of its stages. A tarball was once hashed mid-write because the encoder had exited while `tar` was still running.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md
git commit -m "bench(mtrag): late-interaction results, and what they say about the pooled-pair claim"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: containment gates 1 and 4 → Task 2 (moved to `recall/` and flagged); gates 2 and 3 → Task 4; architecture and the `hit.score` constraint → Task 3; streaming → Task 5; G1 → Task 10 Step 1; G2 → Task 7; G3 → Task 6; G4 → Task 6 header plus Task 10 Step 2; G5 → Task 7 as an automated test; Family B power precondition → Tasks 8 and 9; Family C's V1 width → Task 9 Step 1 and Task 10 Step 6; decision rule → Task 10 Step 7; deliverable → Task 11.

**Two gaps found and closed while reviewing.** G5 was written in the spec as a manual ritual ("replace max with mean, confirm G2 goes red, revert"), which is exactly the kind of check that quietly stops being run. Task 7 Step 1 makes it a permanent test instead. And the spec required V1's width to come from the archive but named no task that recovers it, so Task 9 Step 1 now recovers it alongside the 123.

**One deliberate deviation, flagged in its own section above:** licence gates 1 and 4 live in `recall/rerank.py` rather than the benchmark, which guards the production constructor too.

**Type consistency.** `maxsim(query_tokens, doc_tokens)` is called identically in Tasks 3, 5 and 7. `LateInteractionReranker(encoder, *, model_name=...)` is constructed identically in Tasks 3 and 7. `score_stream`'s `pairs` is `doc_id -> {qid}` in Tasks 5, 6 and its producer `load_pairs_inverted`. `arm_record` returns the same four keys in Tasks 4 and 6.

**One known coupling, stated rather than hidden.** Tasks 6 and 7 reach `reranker._encoder`, a private attribute. `rerank_offload.cmd_validate` sets this precedent by reaching `reranker._model` for the same reason: the offload needs the raw encoder, not the reranking wrapper. Keeping the wrapper's public surface to `rerank()` is worth more than avoiding the private access in two benchmark call sites.
