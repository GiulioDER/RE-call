# Weighted Fusion (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Weight reciprocal-rank fusion by each leg's per-query decisiveness, and measure it against §9a's two published baselines.

**Architecture:** `leg_confidence` graduates from `recall/eval/` to a new `recall/fusion.py` alongside a `weighted_rrf`. `HybridRetriever` gains an opt-in `fusion=` selector defaulting to today's behaviour. The LOCOMO harness gains a matching flag. Nothing changes for existing users until a default is flipped, which this plan does not do.

**Tech Stack:** Python 3.11+, PostgreSQL 16 + pgvector, psycopg 3, pytest, `uv`, fastembed (`bge-small`).

**Preregistration:** [`2026-07-28-weighted-fusion-phase1-design.md`](../specs/2026-07-28-weighted-fusion-phase1-design.md), committed `583359f` before implementation. Do not edit its predictions.

## Global Constraints

- **No LLM anywhere** in the ingest or retrieval path. This is arithmetic over scores already computed.
- **The default path must stay byte-identical.** `fusion` defaults to the current RRF. Equal confidence must reduce *exactly* to today's ordering — that is a test, not an assumption.
- **Zero fitted constants.** No temperature, no per-corpus α, no threshold. A constant here would need re-fitting per corpus by someone with a labelled set they do not have (§2).
- **Test DSN:** `RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall"` — never `:5432`, that container belongs to another worktree and this suite DROPs tables.
- **All three CI gates green before every commit** (`.github/workflows/ci.yml` lines 48/60/83):
  ```
  cd /c/Users/gde00/Documents/recall-fusion-prf && uv run ruff check . && uv run mypy && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest -q
  ```
- **State at plan start:** 903 passed, 4 skipped; ruff clean; mypy clean. Absolute counts are informational; the binding check is zero failures and no previously-passing test broken.
- `locomo10.json` is in the worktree root and **gitignored**. Do not commit it.

---

### Task 1: `recall/fusion.py` — `leg_confidence` and `weighted_rrf`

**Files:**
- Create: `recall/fusion.py`
- Modify: `recall/eval/legconf.py` (re-export, keep `more_decisive`)
- Test: `tests/test_fusion.py` (create)

**Interfaces:**
- Produces: `recall.fusion.leg_confidence(scores: Sequence[float]) -> float` (moved verbatim) and `recall.fusion.weighted_rrf(rankings: Sequence[Sequence[str]], weights: Sequence[float] | None = None, k: int = 60) -> dict[str, float]`.
- `recall/eval/legconf.py` keeps importing `leg_confidence` from the new home so `legdiag` and its tests are untouched.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fusion.py`:

```python
from __future__ import annotations

import pytest

from recall.fusion import leg_confidence, weighted_rrf


def _rrf_reference(rankings, k=60):
    """The shipped unweighted formula, inlined so the equivalence test cannot drift with it."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def test_equal_weights_preserve_the_shipped_ORDER():
    """The backward-compatibility guarantee. Equal weights scale every fused score by the same
    constant, which cannot reorder anything — so today's behaviour is a special case of the new
    function, and shipping this cannot silently change what existing users get."""
    a = ["d1", "d2", "d3", "d4"]
    b = ["d3", "d1", "d9", "d2"]
    ref = _rrf_reference([a, b])
    got = weighted_rrf([a, b], weights=[0.5, 0.5])

    assert set(got) == set(ref)
    order = lambda s: sorted(s, key=lambda c: (-s[c], c))
    assert order(got) == order(ref)


def test_weights_none_is_uniform():
    a, b = ["d1", "d2"], ["d2", "d3"]
    assert weighted_rrf([a, b]) == pytest.approx(weighted_rrf([a, b], weights=[0.5, 0.5]))


def test_a_heavier_leg_pulls_its_own_ranking_up():
    """The whole point: when one leg is trusted more, its ordering should dominate the prefix."""
    dense = ["d_top", "x", "y"]
    sparse = ["s_top", "x", "y"]
    dense_heavy = weighted_rrf([dense, sparse], weights=[0.9, 0.1])
    sparse_heavy = weighted_rrf([dense, sparse], weights=[0.1, 0.9])

    assert dense_heavy["d_top"] > dense_heavy["s_top"]
    assert sparse_heavy["s_top"] > sparse_heavy["d_top"]


def test_a_zero_weight_leg_contributes_nothing():
    a, b = ["d1", "d2"], ["d3"]
    got = weighted_rrf([a, b], weights=[1.0, 0.0])
    assert got["d3"] == 0.0
    assert got["d1"] > 0.0


def test_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        weighted_rrf([["a"], ["b"]], weights=[1.0])


def test_empty_rankings_give_empty_scores():
    assert weighted_rrf([[], []], weights=[0.5, 0.5]) == {}
```

- [ ] **Step 2: Run and confirm failure**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_fusion.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'recall.fusion'`.

- [ ] **Step 3: Implement**

Create `recall/fusion.py`. Move `leg_confidence` **verbatim** from `recall/eval/legconf.py` (body unchanged), and write a module docstring that carries its provenance correctly:

```python
"""Rank fusion, and the per-query leg decisiveness that weights it.

`leg_confidence` reports whether a retrieval leg found a clear winner or handed back a flat,
undifferentiated list: the z-score of the leg's top candidate within that leg's OWN candidates.
It is **affine-invariant**, which is what lets a cosine leg and a `ts_rank` leg be compared
without normalizing incompatible scales. That invariance is asserted as a property test.

Provenance, stated precisely because it is easy to misread: this function was built for the
Phase 0 diagnostic, and Phase 0's *trigger* — `more_decisive`, "is sparse more decisive than
dense" used to PREDICT retrieval failure — was falsified
(`results/legdiag/FINDINGS_phase0.md`). What was falsified is that use. `leg_confidence` itself
measures what it claims to, and Phase 1 uses it for a different job: deciding **which leg's
ranking should dominate this query's prefix**, not predicting whether the query will fail.
Phase 0 is in fact evidence for this use — the leg it identified as more decisive was the one
whose queries scored HIGHER (hit@5 0.708 vs 0.616).
"""
```

Then add:

```python
def weighted_rrf(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Fuse best-first id rankings into one score map, weighting each ranking.

    Each id accrues ``w_L / (k + rank)`` from every ranking it appears in. `k` (default 60, the
    standard RRF damping constant — unrelated to the caller's result-count `k`) softens the top
    ranks so no single ranking dominates outright.

    `weights` defaults to uniform, which reproduces the ORDER of the unweighted formula exactly:
    a common factor scales every score and cannot reorder them. The returned dict is UNSORTED;
    callers sort by value descending.
    """
    if weights is None:
        weights = [1.0 / len(rankings)] * len(rankings) if rankings else []
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(rankings)} rankings but {len(weights)} weights")
    scores: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
    return scores
```

In `recall/eval/legconf.py`, replace the `leg_confidence` definition with a re-export and correct the two now-false statements in its docstring — it currently says *"Nothing in the serving path consumes it"* and *"Phase 1 does not consume this"*, both of which this task makes untrue:

```python
from recall.fusion import leg_confidence  # re-exported: Phase 1 moved it to the serving path

__all__ = ["leg_confidence", "more_decisive"]
```

Keep `more_decisive` here — it is diagnostic-only and *is* the falsified thing. Its FALSIFIED note stays, scoped to itself.

- [ ] **Step 4: Verify all three gates**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run pytest tests/test_fusion.py tests/test_leg_confidence.py tests/test_legdiag.py -v
```
Expected: all pass — `test_leg_confidence.py` and `test_legdiag.py` must pass **unchanged**, proving the re-export preserved the contract.

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run ruff check . && uv run mypy && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest -q
```

- [ ] **Step 5: Commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/fusion.py recall/eval/legconf.py tests/test_fusion.py && git commit -m "feat(fusion): weighted_rrf, and leg_confidence graduates to the serving path

Equal weights reproduce the shipped ORDER exactly — asserted against an
inlined reference rather than against the shipped function, so the two
cannot drift together.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Wire WRRF into the retriever and the LOCOMO harness

**Files:**
- Modify: `recall/retriever.py`
- Modify: `recall/eval/locomo.py`
- Test: `tests/test_retriever_fusion.py` (create)

**Interfaces:**
- Consumes: `recall.fusion.weighted_rrf`, `recall.fusion.leg_confidence`.
- Produces: `HybridRetriever(..., fusion: str = "rrf")` accepting `"rrf"` | `"wrrf"`; `run_conversation(..., fusion: str = "rrf")`; a `--fusion` CLI flag on `recall.eval.locomo`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retriever_fusion.py`:

```python
from __future__ import annotations

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.retriever import HybridRetriever
from tests.conftest import requires_db


def _index(tmp_path, make_store):
    for i in range(12):
        (tmp_path / f"doc{i}.md").write_text(
            f"caching decision {i} about retrieval and indexing", encoding="utf-8"
        )
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    return store, emb


@requires_db
def test_default_fusion_is_unchanged(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    assert HybridRetriever(store, emb)._fusion == "rrf"


@requires_db
def test_wrrf_returns_the_same_candidate_SET_as_rrf(tmp_path, make_store):
    """Weighting reorders; it must never add or drop a candidate. If the sets differ, the
    weighting has changed retrieval rather than ranking, and every downstream comparison
    between the two arms would be confounded."""
    store, emb = _index(tmp_path, make_store)
    a = HybridRetriever(store, emb, candidate_k=20).search("caching retrieval", k=10)
    b = HybridRetriever(store, emb, candidate_k=20, fusion="wrrf").search("caching retrieval", k=10)
    assert {h.chunk.id for h in a.hits} == {h.chunk.id for h in b.hits}


@requires_db
def test_wrrf_keeps_score_as_the_dense_cosine(tmp_path, make_store):
    """The trust layer reads `score` as a cosine. Fusion weights must not leak into it."""
    store, emb = _index(tmp_path, make_store)
    r = HybridRetriever(store, emb, fusion="wrrf").search("caching retrieval", k=5)
    assert all(-1.0 <= h.score <= 1.0 for h in r.hits)


@requires_db
def test_unknown_fusion_is_rejected(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    with pytest.raises(ValueError):
        HybridRetriever(store, emb, fusion="bogus")
```

- [ ] **Step 2: Run and confirm failure**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest tests/test_retriever_fusion.py -v
```
Expected: FAIL — `TypeError: ... unexpected keyword argument 'fusion'`.

- [ ] **Step 3: Implement**

In `recall/retriever.py`, import from the new module and add the selector to `__init__`, validating it eagerly:

```python
        fusion: str = "rrf",
```
```python
        if fusion not in ("rrf", "wrrf"):
            raise ValueError(f"fusion must be 'rrf' or 'wrrf', got {fusion!r}")
        self._fusion = fusion
```

Document it in the class docstring's `Tunables:` block:

```
      fusion:        'rrf' (default, shipped) weights both legs equally; 'wrrf' weights each leg
                     by its per-query decisiveness (recall.fusion.leg_confidence). Equal
                     decisiveness makes the two identical.
```

In `search()`, replace the `fused = _rrf(...)` call. Compute the weights from each leg's **own native scores** — dense cosines, and the sparse leg's `ts_rank` where available:

```python
        dense_ranking = [h.chunk.id for h in dense]
        sparse_ranking = [h.chunk.id for h in sparse]
        if self._fusion == "wrrf":
            # Each leg is scored on its OWN units; `leg_confidence` is affine-invariant, which is
            # what makes a cosine leg and a ts_rank leg comparable. `sparse_ranks` is only
            # populated when probing, so fall back to the sparse hits' cosines otherwise — both
            # are that leg's ordering evidence, and the z-score is scale-free either way.
            c_dense = leg_confidence([h.score for h in dense])
            c_sparse = leg_confidence(sparse_ranks or [h.score for h in sparse])
            total = max(c_dense, 0.0) + max(c_sparse, 0.0)
            weights = (
                [max(c_dense, 0.0) / total, max(c_sparse, 0.0) / total] if total > 0 else [0.5, 0.5]
            )
        else:
            weights = [0.5, 0.5]
        fused = weighted_rrf([dense_ranking, sparse_ranking], weights=weights)
```

Delete the now-unused module-level `_rrf` **only if nothing else references it** — check with `grep -rn "_rrf" recall/ tests/` first and report what you find. If tests reference it, keep it and have it delegate to `weighted_rrf` with uniform weights rather than duplicating the formula.

In `recall/eval/locomo.py`, thread `fusion: str = "rrf"` through `run_conversation` (forward it to `HybridRetriever`) and through `run`, and add a CLI flag:

```python
    p.add_argument(
        "--fusion", default="rrf", choices=["rrf", "wrrf"],
        help="rrf = shipped equal-weight fusion; wrrf = weighted by per-query leg decisiveness",
    )
```
Record it in the report dict beside `candidate_k` so an artifact says which arm produced it.

- [ ] **Step 4: Verify all three gates**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && uv run ruff check . && uv run mypy && RECALL_TEST_DSN="postgresql://recall:recall@localhost:5434/recall" uv run pytest -q
```

- [ ] **Step 5: Commit**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && git add recall/retriever.py recall/eval/locomo.py tests/test_retriever_fusion.py && git commit -m "feat(retriever): opt-in wrrf fusion, threaded through the LOCOMO harness

Default stays 'rrf'. The candidate SET is asserted identical between the
two arms — weighting must reorder, never retrieve differently, or the
arm comparison is confounded.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Run the four arms and score the predictions

**Files:**
- Create: `results/wrrf/` (four JSON reports + logs)
- Create: `results/wrrf/FINDINGS_phase1.md`

- [ ] **Step 1: Run all four arms**

```bash
cd /c/Users/gde00/Documents/recall-fusion-prf && for arm in "rrf 20 A" "wrrf 20 B" "rrf 100 C" "wrrf 100 D"; do set -- $arm; uv run python -m recall.eval.locomo --data locomo10.json --dsn "postgresql://recall:recall@localhost:5434/recall" --k 5 --k-curve 1,5,10,20 --candidate-k $2 --fusion $1 --out results/wrrf/arm_$3_${1}_pool$2.json; done
```

- [ ] **Step 2: Apply the apparatus check BEFORE reading any WRRF number**

Arm A must reproduce **hit@5 0.671 ± 0.01** and arm C **0.596 ± 0.01**, both at n=1,536 exactly. If either fails, the run is not comparable to §9a — **stop and report**, do not interpret.

- [ ] **Step 3: Apply the decision rules from the spec, in order**

| gate | rule | consequence |
|---|---|---|
| D vs C | D ≤ C | do not ship; publish the null |
| B vs A | B < A − 0.02 | do not ship regardless of D |
| stratified | helps Q1–Q3 of \|conf gap\|, hurts Q4 | do not ship as default; report the interaction |

- [ ] **Step 4: Write `results/wrrf/FINDINGS_phase1.md`**

Must contain: the four arms with Wilson CIs; paired McNemar A↔B and C↔D; Δ(WRRF − RRF) **stratified by |conf gap| quartile** (specified in advance, so it must be reported whether or not it is flattering); **each preregistered prediction quoted from `583359f` with a HIT/MISS verdict**; which gates fired; and the reproduce commands. Per the evidence-tier convention, no figure without its retained artifact.

- [ ] **Step 5: Commit and report**

Report to the user: the four numbers, which gates fired, the prediction scorecard, and the ship/no-ship verdict. **A null ships as a null.**
