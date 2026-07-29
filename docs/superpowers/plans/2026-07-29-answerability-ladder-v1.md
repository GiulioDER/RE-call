# The Answerability Ladder v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen excision manifest and run the RE-call arm on LOCOMO, ending at the H1
gate — does abstention performance actually vary with excision distance, or is the curve flat?

**Architecture:** A DB-free builder turns LOCOMO into a manifest of paired instances at controlled
excision distances, using a standalone BM25 index to order what gets removed. Frozen doc-id lists
go into the manifest so every lab excises identically. A runner ingests each ring into RE-call and
records abstain-or-answer per instance. A DB-free scorer produces the 2×2 per ring and the λ-priced
curve. Builder and scorer never touch Postgres; only the runner does.

**Tech Stack:** Python ≥3.11, stdlib only for builder/scorer (no new dependencies), pytest,
psycopg + pgvector for the runner arm only.

**Spec:** `docs/superpowers/specs/2026-07-29-answerability-ladder-design.md` (commit `bbb0ea0`).

## Global Constraints

- **Branch:** `bench/answerability-ladder-design` (already exists, spec committed at `bbb0ea0`).
- **No new runtime dependencies.** Builder and scorer are stdlib-only, matching `recall/eval/metrics.py`
  and `recall/eval/gap_study.py` — "so the analysis runs in the offline test suite and its arithmetic
  can be read without trusting a library."
- **`from __future__ import annotations`** at the top of every new module — repo-wide convention.
- **Determinism is a tested property, not an aspiration.** No `set` iteration order, no `dict` ordering
  assumptions, no unseeded randomness anywhere in the builder. BM25 ties break by `doc_id` ascending.
- **Every new module under `benchmarks/` states its prior-work search in the docstring**, per
  `benchmarks/EXPERIMENT-CONVENTION.md`. The line for every module in this plan is:
  `Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals measured, all failed; this measures the axis they were measured on, not a seventh signal.`
- **Lettered section citations (`§9h`) must be registered** in `tests/test_findings_crossrefs.py::EXPECTED`
  or the suite fails. Prefer not citing lettered sections in new code.
- **`recall/eval/bm25.py`'s formula must not be forked.** One scoring function, two callers.
- **Python:** run as `python -m pytest` from `~/Documents/recall`.
- **Do not run the runner arm (Task 9) without Postgres up:** `make db-up`.
- **Commit-type note:** use `bench(...)` rather than `eval(...)` as a conventional-commit scope in
  this repo's hooks environment — a `PreToolUse` security hook pattern-matches `eval(` and blocks
  the write.

---

## File Structure

| File | Responsibility |
|---|---|
| `recall/eval/bm25.py` (modify) | Gains `BM25Index`, a DB-free scoring core; `BM25Retriever` becomes a thin store-backed wrapper over it |
| `benchmarks/ladder/__init__.py` (create) | Package marker |
| `benchmarks/ladder/adapter.py` (create) | `Document`, `Response`, `MemorySystem` protocol — the third-party boundary |
| `benchmarks/ladder/manifest.py` (create) | `Instance`, serialisation, canonical digest |
| `benchmarks/ladder/rings.py` (create) | `RingSpec`, excision-ring construction from gold ids + BM25 order |
| `benchmarks/ladder/sources/__init__.py` (create) | Package marker |
| `benchmarks/ladder/sources/locomo.py` (create) | LOCOMO → documents + `SourceQuestion`s |
| `benchmarks/ladder/build.py` (create) | CLI: sources + ring spec → frozen manifest |
| `benchmarks/ladder/invariants.py` (create) | The artefact assertions |
| `benchmarks/ladder/systems/recall_system.py` (create) | RE-call reference adapter |
| `benchmarks/ladder/run.py` (create) | Runner: manifest + system → responses JSONL |
| `benchmarks/ladder/score.py` (create) | 2×2 per ring, λ-pricing, H1 flat-curve test |
| `benchmarks/ladder/report.py` (create) | Prints the curve and the H1 verdict; exits 1 on FAIL |
| `benchmarks/PREREGISTRATION-ladder.md` (create) | Predictions + ring widths, committed before the builder runs |
| `tests/test_ladder_*.py` (create) | One test module per unit above |

---

### Task 1: Pre-registration — fix the free parameters before the builder exists

The spec commits to fixing ring widths before the builder runs. Ring widths chosen after seeing a
curve are how a curve gets the shape its author wanted. Corpus *statistics* may be inspected (sizes
are not outcomes); no retrieval or abstention result may be.

**Files:**
- Create: `benchmarks/PREREGISTRATION-ladder.md`
- Create: `scripts/ladder_corpus_stats.py`

**Interfaces:**
- Consumes: nothing
- Produces: `RING_WIDTHS = (0, 4, 16, 64)` and the `RING_MAX` sentinel, referenced by Task 4's
  `RingSpec` and Task 6's builder CLI default. The widths below are provisional until Step 1 runs;
  if the median cluster is smaller than 64, replace 64 with the median cluster size rounded down to
  a power of four, and record the substitution in the pre-registration.

- [ ] **Step 1: Write the corpus-statistics script**

```python
"""Corpus statistics for the Answerability Ladder pre-registration — sizes only, never outcomes.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Ring widths must be fixed before the builder runs, and a width wider than the corpus is not a
ring — it is d=max wearing a different number. So the widths are derived from cluster sizes, which
are a property of the data and not of any result. Nothing here reads a question, an answer, or a
retrieval score.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def conversation_turn_counts(path: Path) -> list[int]:
    """Turns per LOCOMO conversation — the cluster size a d=max excision would remove."""
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: list[int] = []
    for sample in data:
        conversation = sample.get("conversation", {})
        turns = 0
        for key, value in conversation.items():
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list):
                turns += sum(1 for t in value if t.get("dia_id"))
        counts.append(turns)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, default=Path("locomo10.json"))
    args = parser.parse_args(argv)

    counts = conversation_turn_counts(args.locomo)
    print(f"conversations: {len(counts)}")
    print(
        f"turns per conversation: min={min(counts)} "
        f"median={statistics.median(counts)} max={max(counts)}"
    )
    print(f"total turns: {sum(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it and record the numbers**

Run: `python scripts/ladder_corpus_stats.py --locomo locomo10.json`
Expected: three lines of counts. Copy the exact output into the pre-registration in Step 3.
If `locomo10.json` is missing, fetch it first:
`curl -sLO https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`

- [ ] **Step 3: Write the pre-registration**

Create `benchmarks/PREREGISTRATION-ladder.md`. Paste the Step 2 output verbatim into the "Corpus
statistics" section. Everything else below is fixed text:

```markdown
# Pre-registration — the Answerability Ladder

Written **before** the builder exists. The git history of this file is the evidence.
Design: `docs/superpowers/specs/2026-07-29-answerability-ladder-design.md`.

## Corpus statistics (sizes only — inspected before fixing widths, as permitted)

<paste the exact output of `python scripts/ladder_corpus_stats.py` here>

## Fixed parameters

- **Ring widths** `d in {0, 4, 16, 64}` plus `d = max` (whole conversation). Powers of four so the
  ladder spans two orders of magnitude in five levels; the top rung is capped at the median
  conversation size, because a ring wider than its cluster is d=max under another name.
- **Saturation rule:** d=max excises every turn of the conversation the gold turn belongs to.
- **Lambda in {1, 3, 10}**, where lambda = cost(false answer) / cost(false abstention). lambda=1
  reproduces BEAM's implicit weighting. Choosing lambda after seeing results is forbidden by this
  file.
- **Tie-breaking:** equal BM25 scores rank by `doc_id` ascending.
- **Instances per question:** one per ring level, all paired to the same answerable original.

## Predictions, committed now

- **P1 (H1).** Correct-abstain rate rises with d. Specifically, correct-abstain at d=max exceeds
  d=0 by **more than 0.15**, with a bootstrap 95 % CI on the paired difference excluding zero.
- **P2 (H2).** The d=0 rung reproduces the adversarial regime: RE-call's correct-abstain at d=0 is
  **below 0.25**, consistent with 0.00/446 on LOCOMO's own category-5 adversarials and 0.467 on
  BEAM's abstention category.
- **P3.** False-abstain on the *answerable* originals exceeds **0.30**, consistent with the 0.481
  measured on LongMemEval per-question.
- **P4 (H3).** Rebuilding rings with a random-within-cluster neighbour function preserves the sign
  and rough magnitude of the P1 difference. If it does not, BM25 is a confound and the curve does
  not ship.

## What falsifies the benchmark

P1 failing is not a bad result — it is the **kill condition**. A flat curve means the axis is a
fiction, and no comparative arm (Mem0, BEIR) is run. Money is spent only after P1 passes.

## Known to cut against us

- RE-call false-abstains at 0.481 while retrieval hit@5 is 0.970 (LongMemEval, per-question).
- On BEAM's abstention category we score 0.467 against Mem0's 0.533; false-abstain 9.6 % vs 4.1 %.
- Six candidate abstention signals were already measured and all failed (dense cosine AUC 0.753
  ships and sits at its ROC ceiling). This benchmark does not reopen that question.
- LOCOMO's `evidence` labels share an annotation pass with its answer key, 6.4 % of which is wrong.
  Using evidence for excision is safer, not safe.

## What v1 does NOT measure

Whether an *answered* question was answered **correctly**. v1 has no judge, by design and by
budget. On the answerable arm, "answered" is scored as success, which makes every v1 accuracy
figure an **upper bound**. The write-up must say so wherever a number appears.
```

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/PREREGISTRATION-ladder.md scripts/ladder_corpus_stats.py
git commit -m "prereg(ladder): fix ring widths, lambda and four predictions before the builder exists

Ring widths chosen after seeing a curve are how a curve gets the shape its author wanted, so they
are fixed here, derived from cluster sizes (a property of the data, not of any result).
P1 is the kill condition: a flat curve means the axis is a fiction and no paid arm runs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Extract a DB-free BM25 scoring core

`recall/eval/bm25.py` is the right implementation — dependency-free on purpose so the baseline
cannot silently change between releases of a third-party package. But `BM25Retriever` imports
`PgVectorStore` and scores over indexed chunks, and the builder must run with no database. Extract
the core; do **not** fork the formula.

**Files:**
- Modify: `recall/eval/bm25.py` (add `BM25Index`; rewrite `BM25Retriever.__init__` to delegate)
- Test: `tests/test_ladder_bm25_index.py`

**Interfaces:**
- Consumes: `tokenize`, `K1`, `B` from `recall/eval/bm25.py`
- Produces:
  - `BM25Index(docs: Iterable[tuple[str, str]], k1: float = K1, b: float = B)`
  - `BM25Index.doc_ids -> list[str]` (corpus order)
  - `BM25Index.score(query: str) -> list[float]` (corpus order)
  - `BM25Index.rank(query: str) -> list[tuple[str, float]]` (all docs, score desc, ties by doc_id asc)
  - `len(index) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_bm25_index.py`:

```python
"""The DB-free BM25 core the ladder builder ranks with.

The builder must run with no Postgres, and the ring order it produces is frozen into a released
manifest — so ties cannot break by dict insertion order, or two builds of the same corpus produce
two different benchmarks. These tests pin the ranking property, the IDF property, and determinism.
"""
from __future__ import annotations

from recall.eval.bm25 import BM25Index

DOCS = [
    ("d1", "the cache was replaced with a read-through cache"),
    ("d2", "retry policy uses exponential backoff"),
    ("d3", "the cache warms on deploy"),
]


def test_ranks_the_document_containing_the_query_terms_first():
    index = BM25Index(DOCS)
    assert index.rank("backoff")[0][0] == "d2"


def test_a_term_in_every_document_cannot_decide_the_ranking():
    """IDF drives a term appearing in every document to near-zero weight."""
    index = BM25Index([("a", "shared term"), ("b", "shared term"), ("c", "shared term")])
    scores = index.score("shared")
    assert len(set(scores)) == 1


def test_ties_break_by_doc_id_ascending_not_insertion_order():
    forward = BM25Index([("z", "same text here"), ("a", "same text here")])
    reverse = BM25Index([("a", "same text here"), ("z", "same text here")])
    assert [d for d, _ in forward.rank("same")] == ["a", "z"]
    assert [d for d, _ in reverse.rank("same")] == ["a", "z"]


def test_unknown_term_scores_everything_zero():
    index = BM25Index(DOCS)
    assert index.score("xyzzy") == [0.0, 0.0, 0.0]


def test_empty_corpus_scores_nothing_rather_than_dividing_by_zero():
    index = BM25Index([])
    assert len(index) == 0
    assert index.score("anything") == []


def test_doc_ids_are_corpus_order():
    assert BM25Index(DOCS).doc_ids == ["d1", "d2", "d3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_bm25_index.py -q`
Expected: FAIL — `ImportError: cannot import name 'BM25Index' from 'recall.eval.bm25'`

- [ ] **Step 3: Add `BM25Index` to `recall/eval/bm25.py`**

Insert immediately after the `tokenize` function, before `class BM25Retriever`:

```python
class BM25Index:
    """Okapi BM25 over `(doc_id, text)` pairs, with no database and no store type.

    The formula lives here and `BM25Retriever` delegates to it. Two copies of a scoring function
    is how a baseline and the thing it anchors quietly stop agreeing — see this module's docstring
    on why the formula is written out rather than imported from a package.

    Ties break by `doc_id` ascending. That is not tidiness: `rank()` output is frozen into a
    released benchmark manifest, and a tie broken by insertion order would make two builds of the
    same corpus into two different benchmarks.
    """

    def __init__(self, docs: Iterable[tuple[str, str]], k1: float = K1, b: float = B) -> None:
        self._k1 = k1
        self._b = b
        self._doc_ids: list[str] = []
        self._len: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        df: Counter[str] = Counter()

        for doc_id, text in docs:
            tokens = tokenize(text)
            tf = Counter(tokens)
            i = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            self._len.append(len(tokens))
            df.update(tf.keys())
            for term, freq in tf.items():
                self._postings.setdefault(term, []).append((i, freq))

        n = len(self._doc_ids)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    def score(self, query: str) -> list[float]:
        """Per-document BM25 score for `query`, in corpus order."""
        terms = tokenize(query)
        scores = [0.0] * len(self._doc_ids)
        if not terms or not self._avgdl:
            return scores
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, f in self._postings.get(term, ()):
                norm = 1.0 - self._b + self._b * (self._len[i] / self._avgdl)
                scores[i] += idf * (f * (self._k1 + 1.0)) / (f + self._k1 * norm)
        return scores

    def rank(self, query: str) -> list[tuple[str, float]]:
        """Every document, best first. Ties by `doc_id` ascending — see the class docstring."""
        scored = list(zip(self._doc_ids, self.score(query)))
        return sorted(scored, key=lambda ds: (-ds[1], ds[0]))
```

Add `from collections.abc import Iterable` to the module imports if it is not already present.

- [ ] **Step 4: Rewrite `BM25Retriever.__init__` to delegate**

Replace `BM25Retriever.__init__` (the loop building `_postings`/`_idf`) and its `score` method with:

```python
    def __init__(self, store: PgVectorStore, k1: float = K1, b: float = B) -> None:
        self._chunks: list[Chunk] = list(store.iter_chunks())
        # One formula, two callers. The chunk id is positional here because `search` maps back by
        # index; `BM25Index` only needs the ids to be unique and orderable.
        self._index = BM25Index(
            ((str(i), chunk.text) for i, chunk in enumerate(self._chunks)), k1=k1, b=b
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def score(self, query: str) -> list[float]:
        """Per-chunk BM25 score for `query`, in corpus order."""
        return self._index.score(query)
```

Delete the now-dead `self._len`, `self._postings`, `self._idf`, `self._avgdl`, `self._k1` and
`self._b` attributes from `BM25Retriever`. Leave `search()` unchanged — it already calls
`self.score(query)`.

- [ ] **Step 5: Run both test suites**

Run: `python -m pytest tests/test_ladder_bm25_index.py -q && python -m pytest tests/test_eval_bm25.py -q`
Expected: the new module PASSES. `test_eval_bm25.py` passes, or skips DB-marked tests if Postgres
is down — run `make db-up` first and re-run to confirm the delegation did not change behaviour.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add recall/eval/bm25.py tests/test_ladder_bm25_index.py
git commit -m "refactor(bm25): extract a DB-free scoring core, one formula and two callers

The ladder builder must rank with no Postgres, and BM25Retriever scored over store chunks. A
second copy of the formula is how a baseline and the thing it anchors quietly stop agreeing, so
BM25Retriever now delegates instead.

Ties break by doc_id ascending because rank() output is frozen into a released manifest: a tie
broken by insertion order would make two builds of one corpus into two different benchmarks.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The manifest — schema, canonical serialisation, digest

The manifest **is** the benchmark. A third party who distrusts the builder must be able to read it,
verify its digest, and never run our code.

**Files:**
- Create: `benchmarks/ladder/__init__.py`
- Create: `benchmarks/ladder/manifest.py`
- Test: `tests/test_ladder_manifest.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MANIFEST_VERSION: str = "1.0"`
  - `Instance` frozen dataclass: `instance_id, corpus, source_question_id, question, label, ring, excised_doc_ids: tuple[str, ...], gold_doc_ids: tuple[str, ...], pair_id`
  - `LABEL_ANSWERABLE = "answerable"`, `LABEL_UNANSWERABLE = "unanswerable"`
  - `RING_MAX: int = -1` (sentinel for "whole cluster")
  - `instance_to_dict(inst: Instance) -> dict`, `instance_from_dict(d: Mapping) -> Instance`
  - `manifest_digest(instances: Sequence[Instance]) -> str`
  - `write_manifest(path: Path, instances: Sequence[Instance], *, ring_widths: Sequence[int], corpus_hashes: Mapping[str, str]) -> str` (returns the digest)
  - `read_manifest(path: Path) -> tuple[list[Instance], dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_manifest.py`:

```python
"""The manifest is the released artifact — these tests are its contract with a stranger.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

A reader who distrusts our builder must be able to verify the digest and read the instances
without running our code. So the digest must be stable across instance ordering and across a
write/read round trip, and it must CHANGE when any scored field changes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    MANIFEST_VERSION,
    Instance,
    instance_from_dict,
    instance_to_dict,
    manifest_digest,
    read_manifest,
    write_manifest,
)


def _inst(instance_id: str = "i1", ring: int = 0, **kw) -> Instance:
    base = dict(
        instance_id=instance_id,
        corpus="locomo",
        source_question_id="locomo_0_qa3",
        question="When did Caroline go to the support group?",
        label=LABEL_UNANSWERABLE,
        ring=ring,
        excised_doc_ids=("D1:3",),
        gold_doc_ids=("D1:3",),
        pair_id="p1",
    )
    base.update(kw)
    return Instance(**base)


def test_round_trips_through_dict_unchanged():
    inst = _inst()
    assert instance_from_dict(instance_to_dict(inst)) == inst


def test_digest_is_stable_across_instance_order():
    a, b = _inst("i1"), _inst("i2", ring=4)
    assert manifest_digest([a, b]) == manifest_digest([b, a])


def test_digest_changes_when_an_excised_id_changes():
    before = manifest_digest([_inst()])
    after = manifest_digest([_inst(excised_doc_ids=("D1:4",))])
    assert before != after


def test_digest_changes_when_the_label_changes():
    before = manifest_digest([_inst()])
    after = manifest_digest([_inst(label=LABEL_ANSWERABLE)])
    assert before != after


def test_write_then_read_preserves_instances_and_digest(tmp_path: Path):
    instances = [_inst("i1"), _inst("i2", ring=4)]
    path = tmp_path / "manifest.jsonl"
    digest = write_manifest(
        path, instances, ring_widths=[0, 4], corpus_hashes={"locomo": "abc123"}
    )
    read_back, header = read_manifest(path)
    assert read_back == instances
    assert header["digest"] == digest
    assert header["manifest_version"] == MANIFEST_VERSION
    assert header["ring_widths"] == [0, 4]
    assert header["corpus_hashes"] == {"locomo": "abc123"}


def test_read_rejects_a_manifest_whose_digest_does_not_match_its_body(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_inst("i1")], ring_widths=[0], corpus_hashes={})
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"D1:3"', '"D9:9"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        read_manifest(path)


def test_an_unknown_label_is_refused_at_construction():
    with pytest.raises(ValueError, match="label"):
        _inst(label="maybe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/__init__.py`:

```python
"""The Answerability Ladder — a public benchmark whose x-axis is excision distance.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Design: docs/superpowers/specs/2026-07-29-answerability-ladder-design.md
Pre-registration: benchmarks/PREREGISTRATION-ladder.md
"""
```

Create `benchmarks/ladder/manifest.py`:

```python
"""The released artifact: instances, frozen excision doc-id lists, and a digest over both.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

The manifest is the benchmark. Everything else in this package is replaceable plumbing, and a
reader who distrusts the builder must be able to verify this file and read it without running our
code. That imposes two properties the tests pin:

- The digest is over a **canonical** rendering (sorted keys, sorted instances), so it does not
  depend on the order the builder happened to emit.
- `read_manifest` **recomputes** the digest and refuses a body that does not match its header. A
  manifest that silently tolerated an edited excision list would be the same failure shape as an
  artifact with no provenance: a plausible answer with no signal that it is wrong.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MANIFEST_VERSION = "1.0"

LABEL_ANSWERABLE = "answerable"
LABEL_UNANSWERABLE = "unanswerable"
_LABELS = frozenset({LABEL_ANSWERABLE, LABEL_UNANSWERABLE})

#: Ring sentinel for "excise the whole cluster". Not a width, so it cannot be confused with one.
RING_MAX = -1

#: Ring sentinel for the answerable original, which excises nothing. Distinct from RING_MAX so the
#: scorer can never group an original into a rung.
RING_ORIGINAL = -2

_FIELDS = (
    "instance_id",
    "corpus",
    "source_question_id",
    "question",
    "label",
    "ring",
    "excised_doc_ids",
    "gold_doc_ids",
    "pair_id",
)


@dataclass(frozen=True)
class Instance:
    """One question at one excision distance, paired to its own answerable original."""

    instance_id: str
    corpus: str
    source_question_id: str
    question: str
    label: str
    ring: int
    excised_doc_ids: tuple[str, ...]
    gold_doc_ids: tuple[str, ...]
    pair_id: str

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise ValueError(f"label must be one of {sorted(_LABELS)}, got {self.label!r}")
        if not isinstance(self.excised_doc_ids, tuple) or not isinstance(self.gold_doc_ids, tuple):
            raise TypeError("doc-id collections must be tuples — they are hashed and frozen")


def instance_to_dict(inst: Instance) -> dict:
    return {
        "instance_id": inst.instance_id,
        "corpus": inst.corpus,
        "source_question_id": inst.source_question_id,
        "question": inst.question,
        "label": inst.label,
        "ring": inst.ring,
        "excised_doc_ids": list(inst.excised_doc_ids),
        "gold_doc_ids": list(inst.gold_doc_ids),
        "pair_id": inst.pair_id,
    }


def instance_from_dict(d: Mapping) -> Instance:
    missing = [f for f in _FIELDS if f not in d]
    if missing:
        raise ValueError(f"instance is missing fields: {missing}")
    return Instance(
        instance_id=d["instance_id"],
        corpus=d["corpus"],
        source_question_id=d["source_question_id"],
        question=d["question"],
        label=d["label"],
        ring=int(d["ring"]),
        excised_doc_ids=tuple(d["excised_doc_ids"]),
        gold_doc_ids=tuple(d["gold_doc_ids"]),
        pair_id=d["pair_id"],
    )


def _canonical(inst: Instance) -> str:
    return json.dumps(instance_to_dict(inst), sort_keys=True, ensure_ascii=False)


def manifest_digest(instances: Sequence[Instance]) -> str:
    """SHA-256 over the canonical rendering of every instance, order-independent."""
    h = hashlib.sha256()
    for line in sorted(_canonical(i) for i in instances):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_manifest(
    path: Path,
    instances: Sequence[Instance],
    *,
    ring_widths: Sequence[int],
    corpus_hashes: Mapping[str, str],
) -> str:
    """Write header line + one JSON object per instance. Returns the digest."""
    digest = manifest_digest(instances)
    header = {
        "manifest_version": MANIFEST_VERSION,
        "digest": digest,
        "n_instances": len(instances),
        "ring_widths": list(ring_widths),
        "corpus_hashes": dict(corpus_hashes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n")
        for inst in instances:
            fh.write(_canonical(inst) + "\n")
    return digest


def read_manifest(path: Path) -> tuple[list[Instance], dict]:
    """Read and VERIFY. A body that does not match its header digest is refused, not repaired."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    header = json.loads(lines[0])
    instances = [instance_from_dict(json.loads(line)) for line in lines[1:] if line.strip()]
    actual = manifest_digest(instances)
    if actual != header.get("digest"):
        raise ValueError(
            f"{path}: body digest {actual} does not match header digest {header.get('digest')}. "
            f"The manifest has been edited since it was written; rebuild it rather than trusting it."
        )
    return instances, header
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_manifest.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/__init__.py benchmarks/ladder/manifest.py tests/test_ladder_manifest.py
git commit -m "feat(ladder): the manifest — canonical digest, and a read that refuses an edited body

The manifest IS the benchmark: a reader who distrusts the builder must verify it without running
our code. So the digest is order-independent, and read_manifest recomputes it and refuses a
mismatch rather than repairing it — a manifest that tolerated an edited excision list would be a
plausible answer with no signal that it is wrong.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Ring construction — the excision ladder

**Files:**
- Create: `benchmarks/ladder/rings.py`
- Test: `tests/test_ladder_rings.py`

**Interfaces:**
- Consumes: `BM25Index` (Task 2), `RING_MAX` (Task 3)
- Produces:
  - `RingSpec` frozen dataclass: `widths: tuple[int, ...]`
  - `build_rings(index: BM25Index, question: str, gold_doc_ids: Sequence[str], cluster_doc_ids: Sequence[str], spec: RingSpec) -> dict[int, tuple[str, ...]]`
  - `random_rings(question: str, gold_doc_ids: Sequence[str], cluster_doc_ids: Sequence[str], spec: RingSpec, *, seed: int) -> dict[int, tuple[str, ...]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_rings.py`:

```python
"""Ring construction — the x-axis, and the two ways it could quietly become circular.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Two properties carry the design. Rings must NEST (d=4 removes everything d=0 removed, plus more),
or "distance" is not a distance. And gold must always be excised at every level, or an
"unanswerable" instance is answerable and the label is a lie.
"""
from __future__ import annotations

from benchmarks.ladder.manifest import RING_MAX
from benchmarks.ladder.rings import RingSpec, build_rings, random_rings
from recall.eval.bm25 import BM25Index

CLUSTER = [f"d{i}" for i in range(10)]
DOCS = [
    ("d0", "caroline attended the lgbtq support group on may seventh"),
    ("d1", "caroline mentioned the support group again"),
    ("d2", "the support group meets weekly"),
    ("d3", "melanie ran a charity race"),
    ("d4", "melanie trained for the race"),
    ("d5", "the weather was cold"),
    ("d6", "they discussed dinner plans"),
    ("d7", "a new job application"),
    ("d8", "the cat needed a vet"),
    ("d9", "holiday travel arrangements"),
]
SPEC = RingSpec(widths=(0, 2, 4))
QUESTION = "when did caroline go to the support group"


def test_ring_zero_excises_exactly_the_gold_documents():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert rings[0] == ("d0",)


def test_rings_nest_so_distance_is_a_distance():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert set(rings[0]) <= set(rings[2]) <= set(rings[4]) <= set(rings[RING_MAX])


def test_ring_widths_add_that_many_neighbours():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert len(rings[2]) == 3  # 1 gold + 2 neighbours
    assert len(rings[4]) == 5  # 1 gold + 4 neighbours


def test_ring_max_excises_the_whole_cluster():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert set(rings[RING_MAX]) == set(CLUSTER)


def test_gold_is_excised_at_every_level_including_max():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert all("d0" in ids for ids in rings.values())


def test_neighbours_come_only_from_the_cluster():
    outside = DOCS + [("other0", "caroline support group elsewhere")]
    rings = build_rings(BM25Index(outside), QUESTION, ["d0"], CLUSTER, SPEC)
    assert all(d in CLUSTER for d in rings[4])


def test_a_width_wider_than_the_cluster_saturates_rather_than_erroring():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, RingSpec(widths=(0, 500)))
    assert set(rings[500]) == set(CLUSTER)


def test_excised_ids_are_sorted_so_two_builds_agree():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    for ids in rings.values():
        assert list(ids) == sorted(ids)


def test_random_rings_are_reproducible_from_their_seed():
    a = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=7)
    b = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=7)
    c = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=8)
    assert a == b
    assert a != c


def test_random_rings_obey_the_same_nesting_and_gold_rules():
    rings = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=7)
    assert set(rings[0]) <= set(rings[2]) <= set(rings[4])
    assert all("d0" in ids for ids in rings.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_rings.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.rings'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/rings.py`:

```python
"""Excision rings: gold, then a widening ring of BM25 neighbours, then the whole cluster.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

The x-axis is a COUNT of excised documents, not a similarity score. BM25 only decides the ORDER in
which neighbours are removed; the resulting id lists are then frozen into the manifest, so every
lab excises identically no matter what embedder it runs. That is what keeps the axis non-circular:
a system under test never computes its own distances.

BM25 deciding the order is still a choice, and `random_rings` exists to price it — P4 in the
pre-registration. If the curve's shape depends on which neighbour function ordered the removal,
the curve is measuring BM25 and does not ship as an answerability result.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.ladder.manifest import RING_MAX
from recall.eval.bm25 import BM25Index


@dataclass(frozen=True)
class RingSpec:
    """Ring widths, fixed in `benchmarks/PREREGISTRATION-ladder.md` before the builder ran.

    A width is a count of NEIGHBOURS excised on top of gold, so width 0 is "gold only".
    """

    widths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.widths:
            raise ValueError("a ladder with no rungs is not a ladder")
        if any(w < 0 for w in self.widths):
            raise ValueError("ring widths are counts and cannot be negative")
        if list(self.widths) != sorted(set(self.widths)):
            raise ValueError("widths must be strictly increasing — rings nest")


def _assemble(
    gold: Sequence[str],
    ordered_neighbours: Sequence[str],
    spec: RingSpec,
    cluster: Sequence[str],
) -> dict[int, tuple[str, ...]]:
    gold_set = set(gold)
    rings: dict[int, tuple[str, ...]] = {}
    for width in spec.widths:
        # Saturates rather than erroring: a width wider than the cluster is d=max under another
        # name, and refusing it would make the ladder's top rung depend on conversation length.
        taken = ordered_neighbours[:width]
        rings[width] = tuple(sorted(gold_set | set(taken)))
    rings[RING_MAX] = tuple(sorted(gold_set | set(cluster)))
    return rings


def build_rings(
    index: BM25Index,
    question: str,
    gold_doc_ids: Sequence[str],
    cluster_doc_ids: Sequence[str],
    spec: RingSpec,
) -> dict[int, tuple[str, ...]]:
    """Ring level -> excised doc ids. Neighbours are drawn from the cluster only, BM25 order."""
    gold_set = set(gold_doc_ids)
    cluster_set = set(cluster_doc_ids)
    ordered = [
        doc_id
        for doc_id, _ in index.rank(question)
        if doc_id in cluster_set and doc_id not in gold_set
    ]
    return _assemble(gold_doc_ids, ordered, spec, cluster_doc_ids)


def random_rings(
    question: str,
    gold_doc_ids: Sequence[str],
    cluster_doc_ids: Sequence[str],
    spec: RingSpec,
    *,
    seed: int,
) -> dict[int, tuple[str, ...]]:
    """The P4 robustness arm: same widths, neighbours drawn at random within the cluster.

    `question` is unused and kept in the signature on purpose, so this is a drop-in for
    `build_rings` at the call site rather than a second code path in the builder.
    """
    gold_set = set(gold_doc_ids)
    ordered = sorted(d for d in cluster_doc_ids if d not in gold_set)
    random.Random(seed).shuffle(ordered)
    return _assemble(gold_doc_ids, ordered, spec, cluster_doc_ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_rings.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/rings.py tests/test_ladder_rings.py
git commit -m "feat(ladder): excision rings — nesting, gold always removed, saturation not error

The x-axis is a count of excised documents; BM25 only orders the removal, and the resulting id
lists are frozen into the manifest so no system under test computes its own distances.

random_rings is not a convenience: BM25 ordering is itself a choice, and P4 prices it. If the
curve's shape depends on which neighbour function ordered the removal, the curve is measuring
BM25 and does not ship.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: LOCOMO source loader

**Files:**
- Create: `benchmarks/ladder/sources/__init__.py`
- Create: `benchmarks/ladder/sources/locomo.py`
- Test: `tests/test_ladder_source_locomo.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `SourceQuestion` frozen dataclass: `question_id: str, question: str, gold_doc_ids: tuple[str, ...], cluster_id: str`
  - `SourceCorpus` frozen dataclass: `documents: tuple[tuple[str, str], ...], questions: tuple[SourceQuestion, ...], cluster_members: dict[str, tuple[str, ...]], content_hash: str`
  - `load_locomo(path: Path) -> SourceCorpus`

Doc ids are LOCOMO `dia_id`s namespaced by `sample_id` (`"conv-0/D1:3"`), because ids are unique
only **within** a conversation — `recall/eval/locomo.py:415` records that exact trap. Cluster = one
conversation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_source_locomo.py`:

```python
"""LOCOMO -> documents + questions, with the id-collision trap the existing runner already hit.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

`dia_id` is unique only WITHIN a conversation (recall/eval/locomo.py:415). Loading ten
conversations into one id space without namespacing would silently make "D1:3" from conversation
0 and conversation 7 the same document — and the excision would remove the wrong turn while every
count still looked right.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ladder.sources.locomo import load_locomo

SAMPLE = [
    {
        "sample_id": "conv-0",
        "conversation": {
            "session_1_date_time": "7 May 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Caroline", "text": "I went to the support group."},
                {"dia_id": "D1:2", "speaker": "Melanie", "text": "How was it?"},
            ],
        },
        "qa": [
            {
                "question": "When did Caroline go?",
                "answer": "7 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {"question": "An adversarial one", "adversarial_answer": "no", "category": 5},
        ],
    },
    {
        "sample_id": "conv-1",
        "conversation": {
            "session_1_date_time": "1 Jan 2023",
            "session_1": [{"dia_id": "D1:1", "speaker": "Ann", "text": "Different conversation."}],
        },
        "qa": [{"question": "Who spoke?", "answer": "Ann", "evidence": ["D1:1"], "category": 1}],
    },
]


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return path


def test_doc_ids_are_namespaced_by_conversation(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    ids = [doc_id for doc_id, _ in corpus.documents]
    assert "conv-0/D1:1" in ids
    assert "conv-1/D1:1" in ids
    assert len(ids) == len(set(ids))


def test_gold_ids_are_namespaced_to_match_the_documents(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    q = next(q for q in corpus.questions if q.question_id == "conv-0/qa0")
    assert q.gold_doc_ids == ("conv-0/D1:1",)


def test_adversarial_category_five_questions_are_dropped(tmp_path: Path):
    """They are unanswerable by ANNOTATION, not by excision — a different construction."""
    corpus = load_locomo(_write(tmp_path))
    assert all("adversarial" not in q.question.lower() for q in corpus.questions)
    assert len(corpus.questions) == 2


def test_cluster_members_group_turns_by_conversation(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    assert corpus.cluster_members["conv-0"] == ("conv-0/D1:1", "conv-0/D1:2")
    assert corpus.cluster_members["conv-1"] == ("conv-1/D1:1",)


def test_document_text_carries_speaker_and_date(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    text = dict(corpus.documents)["conv-0/D1:1"]
    assert "Caroline" in text and "7 May 2023" in text and "support group" in text


def test_content_hash_changes_when_a_turn_changes(tmp_path: Path):
    before = load_locomo(_write(tmp_path)).content_hash
    altered = json.loads(json.dumps(SAMPLE))
    altered[0]["conversation"]["session_1"][0]["text"] = "changed"
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    assert load_locomo(path).content_hash != before


def test_questions_without_evidence_are_dropped_not_silently_ungolded(tmp_path: Path):
    altered = json.loads(json.dumps(SAMPLE))
    altered[1]["qa"][0].pop("evidence")
    path = tmp_path / "noevi.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    corpus = load_locomo(path)
    assert all(q.gold_doc_ids for q in corpus.questions)
    assert len(corpus.questions) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_source_locomo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.sources'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/sources/__init__.py`:

```python
"""Source-corpus loaders. Each returns a `SourceCorpus` and knows nothing about rings."""
```

Create `benchmarks/ladder/sources/locomo.py`:

```python
"""LOCOMO -> documents, questions and clusters, with ids namespaced per conversation.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Two deliberate exclusions, both of which would otherwise contaminate the ladder:

- **Category 5 is dropped.** Those 446 questions are unanswerable by ANNOTATION — an event
  attributed to the wrong speaker — not by excision. Mixing two constructions of "unanswerable"
  into one axis would make the axis mean two things. They remain valuable as an EXTERNAL check on
  H2 (RE-call scores 0.00/446 on them) and are scored separately, never as a rung.
- **Questions with no `evidence`** are dropped rather than kept with an empty gold set: a question
  with nothing to excise is answerable at every rung, which would flatten the very curve H1 tests.

`dia_id` is unique only within a conversation (see `recall/eval/locomo.py:415`), so every id is
namespaced `"{sample_id}/{dia_id}"`. Without this, "D1:3" from two conversations collide and the
builder excises the wrong turn while every count still looks right.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ADVERSARIAL_CATEGORY = 5


@dataclass(frozen=True)
class SourceQuestion:
    question_id: str
    question: str
    gold_doc_ids: tuple[str, ...]
    cluster_id: str


@dataclass(frozen=True)
class SourceCorpus:
    documents: tuple[tuple[str, str], ...]
    questions: tuple[SourceQuestion, ...]
    cluster_members: dict[str, tuple[str, ...]]
    content_hash: str


def _turn_text(turn: dict[str, Any], session_date: str) -> str:
    speaker = turn.get("speaker", "unknown")
    text = turn.get("text", "")
    return f"{speaker} ({session_date}): {text}"


def _sessions(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )


def _hash(documents: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for doc_id, text in sorted(documents):
        h.update(doc_id.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_locomo(path: Path) -> SourceCorpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    documents: list[tuple[str, str]] = []
    questions: list[SourceQuestion] = []
    cluster_members: dict[str, tuple[str, ...]] = {}

    for sample in data:
        sample_id = sample["sample_id"]
        conversation = sample.get("conversation", {})
        members: list[str] = []
        for key in _sessions(conversation):
            turns = conversation[key]
            if not isinstance(turns, list):
                continue
            date = conversation.get(f"{key}_date_time", "unknown date")
            for turn in turns:
                dia_id = turn.get("dia_id")
                if not dia_id:
                    continue
                doc_id = f"{sample_id}/{dia_id}"
                documents.append((doc_id, _turn_text(turn, date)))
                members.append(doc_id)
        cluster_members[sample_id] = tuple(members)

        for i, qa in enumerate(sample.get("qa", [])):
            if qa.get("category") == _ADVERSARIAL_CATEGORY:
                continue
            evidence = [e for e in (qa.get("evidence") or []) if isinstance(e, str)]
            if not evidence:
                continue
            questions.append(
                SourceQuestion(
                    question_id=f"{sample_id}/qa{i}",
                    question=qa["question"],
                    gold_doc_ids=tuple(f"{sample_id}/{e}" for e in evidence),
                    cluster_id=sample_id,
                )
            )

    return SourceCorpus(
        documents=tuple(documents),
        questions=tuple(questions),
        cluster_members=cluster_members,
        content_hash=_hash(documents),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_source_locomo.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/sources/ tests/test_ladder_source_locomo.py
git commit -m "feat(ladder): LOCOMO source loader — namespaced ids, category 5 excluded

dia_id is unique only WITHIN a conversation, so ten conversations in one id space would make
'D1:3' collide across two of them and excise the wrong turn while every count still looked right.
Ids are namespaced per sample_id.

Category 5 is dropped from the ladder: those questions are unanswerable by ANNOTATION, not by
excision, and mixing two constructions of 'unanswerable' into one axis makes the axis mean two
things. They are scored separately as an external H2 check.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The builder — source + rings → frozen manifest

**Files:**
- Create: `benchmarks/ladder/build.py`
- Test: `tests/test_ladder_build.py`

**Interfaces:**
- Consumes: `load_locomo`/`SourceCorpus` (Task 5), `RingSpec`/`build_rings`/`random_rings` (Task 4),
  `Instance`/`write_manifest`/`RING_MAX`/`RING_ORIGINAL`/labels (Task 3), `BM25Index` (Task 2)
- Produces:
  - `build_instances(corpus: SourceCorpus, spec: RingSpec, *, corpus_name: str, random_seed: int | None = None) -> list[Instance]`
  - `main(argv: list[str] | None = None) -> int`

Every question yields `len(spec.widths) + 2` instances: one answerable original (nothing excised),
one per width, and one at `RING_MAX`. All share a `pair_id`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_build.py`:

```python
"""The builder: one question -> a paired family of instances, and the same manifest every time.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Determinism is the load-bearing property. The manifest is released and cited; if two builds of one
corpus differ, there is no benchmark, only a run. So the digest is asserted equal across rebuilds
rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ladder.build import build_instances, main
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    manifest_digest,
    read_manifest,
)
from benchmarks.ladder.rings import RingSpec
from benchmarks.ladder.sources.locomo import load_locomo
from tests.test_ladder_source_locomo import SAMPLE

SPEC = RingSpec(widths=(0, 1))


def _corpus(tmp_path: Path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return load_locomo(path)


def test_every_question_yields_one_answerable_original_plus_one_per_rung(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_instances(corpus, SPEC, corpus_name="locomo")
    # 2 widths + RING_MAX + 1 answerable original = 4 per question
    assert len(instances) == len(corpus.questions) * 4


def test_the_answerable_original_excises_nothing(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    originals = [i for i in instances if i.label == LABEL_ANSWERABLE]
    assert originals
    assert all(i.excised_doc_ids == () for i in originals)


def test_every_unanswerable_instance_excises_its_gold(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    for inst in instances:
        if inst.label == LABEL_UNANSWERABLE:
            assert set(inst.gold_doc_ids) <= set(inst.excised_doc_ids)


def test_a_family_shares_one_pair_id(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    by_question: dict[str, set[str]] = {}
    for inst in instances:
        by_question.setdefault(inst.source_question_id, set()).add(inst.pair_id)
    assert all(len(pairs) == 1 for pairs in by_question.values())


def test_instance_ids_are_unique(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    ids = [i.instance_id for i in instances]
    assert len(ids) == len(set(ids))


def test_ring_max_instance_is_present_for_every_question(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_instances(corpus, SPEC, corpus_name="locomo")
    at_max = [i for i in instances if i.ring == RING_MAX]
    assert len(at_max) == len(corpus.questions)


def test_two_builds_of_the_same_corpus_produce_the_same_digest(tmp_path: Path):
    corpus = _corpus(tmp_path)
    a = build_instances(corpus, SPEC, corpus_name="locomo")
    b = build_instances(corpus, SPEC, corpus_name="locomo")
    assert manifest_digest(a) == manifest_digest(b)


def test_the_random_arm_differs_from_the_bm25_arm(tmp_path: Path):
    corpus = _corpus(tmp_path)
    bm25 = build_instances(corpus, RingSpec(widths=(0, 1)), corpus_name="locomo")
    rand = build_instances(corpus, RingSpec(widths=(0, 1)), corpus_name="locomo", random_seed=7)
    assert manifest_digest(bm25) != manifest_digest(rand)


def test_cli_writes_a_readable_manifest(tmp_path: Path):
    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(SAMPLE), encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    assert main(["--locomo", str(src), "--out", str(out), "--widths", "0,1"]) == 0
    instances, header = read_manifest(out)
    assert instances
    assert header["ring_widths"] == [0, 1]
    assert "locomo" in header["corpus_hashes"]
```

Note: `test_the_random_arm_differs_from_the_bm25_arm` can fail spuriously on a corpus where the
random shuffle happens to reproduce the BM25 order. `SAMPLE` has a 2-turn and a 1-turn
conversation, so if it does fail, widen `SAMPLE`'s first conversation to four turns rather than
weakening the assertion.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.build'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/build.py`:

```python
"""Build the frozen manifest: source corpus + ring spec -> paired instances.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Every question becomes a FAMILY: one answerable original with nothing excised, one instance per
ring width, and one at RING_MAX. They share a `pair_id`, which is what lets the scorer pair each
unanswerable instance against its own original — the design that defended the Mem0 head-to-head
and that differences out LOCOMO's shared annotation error.

Determinism is not a nicety here. This manifest is released and cited; two builds that disagree
mean there is no benchmark, only a run. Everything is sorted, and the only randomness is the
explicitly seeded P4 arm.

Usage::

    python -m benchmarks.ladder.build --locomo locomo10.json --out results/ladder/manifest.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    RING_ORIGINAL,
    Instance,
    write_manifest,
)
from benchmarks.ladder.rings import RingSpec, build_rings, random_rings
from benchmarks.ladder.sources.locomo import SourceCorpus, load_locomo
from recall.eval.bm25 import BM25Index


def build_instances(
    corpus: SourceCorpus,
    spec: RingSpec,
    *,
    corpus_name: str,
    random_seed: int | None = None,
) -> list[Instance]:
    """One family per question: the answerable original, each rung, and RING_MAX."""
    index = BM25Index(corpus.documents)
    instances: list[Instance] = []

    for question in corpus.questions:
        pair_id = f"{corpus_name}/{question.question_id}"
        cluster = corpus.cluster_members.get(question.cluster_id, ())

        instances.append(
            Instance(
                instance_id=f"{pair_id}#original",
                corpus=corpus_name,
                source_question_id=question.question_id,
                question=question.question,
                label=LABEL_ANSWERABLE,
                ring=RING_ORIGINAL,
                excised_doc_ids=(),
                gold_doc_ids=question.gold_doc_ids,
                pair_id=pair_id,
            )
        )

        if random_seed is None:
            rings = build_rings(index, question.question, question.gold_doc_ids, cluster, spec)
        else:
            rings = random_rings(
                question.question, question.gold_doc_ids, cluster, spec, seed=random_seed
            )

        for level in sorted(rings, key=lambda r: (r == RING_MAX, r)):
            instances.append(
                Instance(
                    instance_id=f"{pair_id}#d{level}",
                    corpus=corpus_name,
                    source_question_id=question.question_id,
                    question=question.question,
                    label=LABEL_UNANSWERABLE,
                    ring=level,
                    excised_doc_ids=rings[level],
                    gold_doc_ids=question.gold_doc_ids,
                    pair_id=pair_id,
                )
            )

    return instances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Answerability Ladder manifest.")
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--out", type=Path, required=True, help="manifest output path (.jsonl)")
    parser.add_argument(
        "--widths",
        default="0,4,16,64",
        help="comma-separated ring widths, fixed in PREREGISTRATION-ladder.md",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="build the P4 robustness arm with random-within-cluster neighbours instead of BM25",
    )
    args = parser.parse_args(argv)

    spec = RingSpec(widths=tuple(int(w) for w in args.widths.split(",")))
    corpus = load_locomo(args.locomo)
    instances = build_instances(corpus, spec, corpus_name="locomo", random_seed=args.random_seed)
    digest = write_manifest(
        args.out,
        instances,
        ring_widths=list(spec.widths),
        corpus_hashes={"locomo": corpus.content_hash},
    )
    print(f"wrote {len(instances)} instances to {args.out}")
    print(f"digest {digest}")
    print(f"corpus locomo {corpus.content_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_build.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Build the real manifest and prove it is deterministic**

Run:
`python -m benchmarks.ladder.build --locomo locomo10.json --out results/ladder/manifest.jsonl --widths 0,4,16,64`
Expected: `wrote N instances`, a digest, and a corpus hash. Record all three in the commit message.

Then:
`python -m benchmarks.ladder.build --locomo locomo10.json --out /tmp/m2.jsonl --widths 0,4,16,64`
Expected: **the same digest line.** If it differs, stop — a nondeterministic builder cannot ship a
manifest, and the cause must be found before any arm is run.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/build.py tests/test_ladder_build.py results/ladder/manifest.jsonl
git commit -m "feat(ladder): the builder, and the manifest it freezes

Every question becomes a family — the answerable original, one instance per rung, and RING_MAX —
sharing a pair_id, which is what lets the scorer pair each unanswerable instance against its own
original and difference out LOCOMO's shared annotation error.

Determinism is asserted, not assumed: two builds of one corpus must produce one digest, or there
is no benchmark, only a run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The adapter boundary

**Files:**
- Create: `benchmarks/ladder/adapter.py`
- Test: `tests/test_ladder_adapter.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Document` frozen dataclass: `doc_id: str, text: str`
  - `Response` frozen dataclass: `answer: str | None, cited_ids: tuple[str, ...] = (), tokens: int = 0`, property `abstained -> bool`
  - `MemorySystem` Protocol: `name: str`, `ingest(docs: Iterable[Document]) -> None`, `indexed_doc_ids() -> frozenset[str]`, `query(question: str) -> Response`

`indexed_doc_ids()` exists for invariant 1 (Task 8) — without it there is no way to prove a system
really dropped the excised turns, and a system that cached across rings would pass every rung while
looking like a strong result.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_adapter.py`:

```python
"""The third-party boundary. An interface with one implementation is a class, not a protocol.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.
"""
from __future__ import annotations

from collections.abc import Iterable

from benchmarks.ladder.adapter import Document, MemorySystem, Response


class _Fake:
    name = "fake"

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    def ingest(self, docs: Iterable[Document]) -> None:
        self._docs = {d.doc_id: d.text for d in docs}

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset(self._docs)

    def query(self, question: str) -> Response:
        term = question.split()[0]
        hit = next((i for i, t in sorted(self._docs.items()) if term in t), None)
        if hit is None:
            return Response(answer=None)
        return Response(answer=self._docs[hit], cited_ids=(hit,))


def test_a_minimal_implementation_satisfies_the_protocol():
    system: MemorySystem = _Fake()
    system.ingest([Document("d1", "alpha text")])
    assert system.indexed_doc_ids() == {"d1"}


def test_none_answer_is_the_abstention():
    assert Response(answer=None).abstained is True
    assert Response(answer="something").abstained is False


def test_an_empty_string_is_an_answer_not_an_abstention():
    """A system returning '' has answered badly, not declined. Conflating them would score a
    broken generator as a well-calibrated one."""
    assert Response(answer="").abstained is False


def test_query_returns_an_abstention_when_nothing_matches():
    system = _Fake()
    system.ingest([Document("d1", "alpha text")])
    assert system.query("zulu something").abstained is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_adapter.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.adapter'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/adapter.py`:

```python
"""The boundary a third party implements to be scored by this benchmark.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Three methods and two dataclasses, deliberately. Anything richer would encode assumptions about
how a memory system is built, and the benchmark's whole claim is that it scores the OUTCOME rather
than the mechanism.

`answer=None` **is** the abstention. An empty string is an answer, badly given — conflating the two
would score a broken generator as a well-calibrated one.

`indexed_doc_ids()` is not optional plumbing: it is how invariant 1 proves a system really dropped
the excised turns. A system that cached across rings would otherwise pass every rung and look like
a strong result rather than a broken harness.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Response:
    answer: str | None
    cited_ids: tuple[str, ...] = ()
    tokens: int = 0

    @property
    def abstained(self) -> bool:
        return self.answer is None


@runtime_checkable
class MemorySystem(Protocol):
    name: str

    def ingest(self, docs: Iterable[Document]) -> None:
        """Replace this system's corpus with `docs`. Must not retain anything from a prior call."""

    def indexed_doc_ids(self) -> frozenset[str]:
        """Every doc id currently retrievable. Read by invariant 1."""

    def query(self, question: str) -> Response:
        """Answer, or abstain with `answer=None`."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_adapter.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/adapter.py tests/test_ladder_adapter.py
git commit -m "feat(ladder): the adapter boundary — and an abstention that is not the empty string

answer=None is the abstention; an empty string is an answer badly given, and conflating them
would score a broken generator as a well-calibrated one.

indexed_doc_ids() is not plumbing: it is how invariant 1 proves a system really dropped the
excised turns. A system that cached across rings would otherwise pass every rung and look like a
strong result rather than a broken harness.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: The invariants

**Files:**
- Create: `benchmarks/ladder/invariants.py`
- Test: `tests/test_ladder_invariants.py`

**Interfaces:**
- Consumes: `Instance`, `manifest_digest`, `LABEL_ANSWERABLE` (Task 3)
- Produces:
  - `InvariantViolation(RuntimeError)`
  - `assert_excised_absent(instance: Instance, indexed: frozenset[str]) -> None`
  - `assert_ring_zero_has_survivors(instance: Instance, indexed: frozenset[str], cluster: Sequence[str]) -> None`
  - `assert_originals_were_answered(answered: Mapping[str, bool], instances: Sequence[Instance]) -> None`
  - `assert_manifest_digest(instances: Sequence[Instance], expected: str) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_invariants.py`:

```python
"""The artefact assertions. Exit code 0 is not a measurement.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Each of these catches a failure that would otherwise produce a PLAUSIBLE NUMBER rather than an
error — which is the only kind of failure worth writing an assertion for.
"""
from __future__ import annotations

import pytest

from benchmarks.ladder.invariants import (
    InvariantViolation,
    assert_excised_absent,
    assert_manifest_digest,
    assert_originals_were_answered,
    assert_ring_zero_has_survivors,
)
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_ORIGINAL,
    Instance,
    manifest_digest,
)


def _inst(**kw) -> Instance:
    base = dict(
        instance_id="i1",
        corpus="locomo",
        source_question_id="q1",
        question="when?",
        label=LABEL_UNANSWERABLE,
        ring=0,
        excised_doc_ids=("c/D1:3",),
        gold_doc_ids=("c/D1:3",),
        pair_id="p1",
    )
    base.update(kw)
    return Instance(**base)


def _original(instance_id: str = "o1") -> Instance:
    return _inst(
        instance_id=instance_id,
        label=LABEL_ANSWERABLE,
        ring=RING_ORIGINAL,
        excised_doc_ids=(),
    )


def test_excised_absent_passes_when_the_system_really_dropped_them():
    assert_excised_absent(_inst(), frozenset({"c/D1:1", "c/D1:2"}))


def test_excised_absent_raises_when_a_system_cached_across_rings():
    with pytest.raises(InvariantViolation, match="still indexed"):
        assert_excised_absent(_inst(), frozenset({"c/D1:3"}))


def test_ring_zero_needs_surviving_neighbours():
    cluster = ["c/D1:1", "c/D1:2", "c/D1:3"]
    assert_ring_zero_has_survivors(_inst(), frozenset({"c/D1:1", "c/D1:2"}), cluster)


def test_ring_zero_with_no_survivors_is_secretly_ring_max():
    with pytest.raises(InvariantViolation, match="d=max"):
        assert_ring_zero_has_survivors(_inst(), frozenset(), ["c/D1:3"])


def test_non_zero_rings_are_not_subject_to_the_survivor_rule():
    assert_ring_zero_has_survivors(_inst(ring=16), frozenset(), ["c/D1:3"])


def test_originals_answered_passes_when_at_least_one_was_answered():
    assert_originals_were_answered({"o1": True}, [_original()])


def test_originals_all_abstained_means_the_questions_are_broken_not_hard():
    with pytest.raises(InvariantViolation, match="broken"):
        assert_originals_were_answered({"o1": False}, [_original()])


def test_manifest_digest_mismatch_is_refused():
    instances = [_inst()]
    assert_manifest_digest(instances, manifest_digest(instances))
    with pytest.raises(InvariantViolation, match="digest"):
        assert_manifest_digest(instances, "deadbeef")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_invariants.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.invariants'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/invariants.py`:

```python
"""Assertions on the ARTEFACT, not on the process. Exit code 0 is not a measurement.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Every check here exists because its failure mode produces a plausible number rather than an error,
and a plausible number with no signal that it is wrong is the failure this repo keeps paying for.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from benchmarks.ladder.manifest import LABEL_ANSWERABLE, Instance, manifest_digest


class InvariantViolation(RuntimeError):
    """A measured artefact contradicts something the design guarantees."""


def assert_excised_absent(instance: Instance, indexed: frozenset[str]) -> None:
    """Invariant 1: the system really dropped what this rung excises.

    A system that cached across rings would pass every rung, and the curve would look like a
    strong result instead of a broken harness.
    """
    leaked = sorted(set(instance.excised_doc_ids) & indexed)
    if leaked:
        raise InvariantViolation(
            f"{instance.instance_id}: {len(leaked)} excised documents are still indexed "
            f"({leaked[:5]}). The system retained state across rings; its ingest must replace, "
            f"not merge."
        )


def assert_ring_zero_has_survivors(
    instance: Instance, indexed: frozenset[str], cluster: Sequence[str]
) -> None:
    """Invariant 2: at d=0 the topic must survive, or d=0 is d=max wearing a different number."""
    if instance.ring != 0:
        return
    survivors = (set(cluster) - set(instance.excised_doc_ids)) & indexed
    if not survivors:
        raise InvariantViolation(
            f"{instance.instance_id}: nothing from its cluster survived d=0, so this rung is "
            f"d=max under another name. Drop the instance rather than scoring it."
        )


def assert_originals_were_answered(
    answered: Mapping[str, bool], instances: Sequence[Instance]
) -> None:
    """Invariant 3: a question no system answers with its gold present is broken, not hard."""
    originals = [i for i in instances if i.label == LABEL_ANSWERABLE]
    if not originals:
        return
    unanswered = [i.instance_id for i in originals if not answered.get(i.instance_id, False)]
    if len(unanswered) == len(originals):
        raise InvariantViolation(
            f"all {len(originals)} answerable originals were abstained on. Those questions are "
            f"broken, not hard, and cannot anchor a pair. Check ingest before reading any curve."
        )


def assert_manifest_digest(instances: Sequence[Instance], expected: str) -> None:
    """Invariant 4: the instances being scored are the instances that were published."""
    actual = manifest_digest(instances)
    if actual != expected:
        raise InvariantViolation(
            f"manifest digest {actual} != expected {expected}. The manifest changed between build "
            f"and scoring; results computed against it are not results for the published benchmark."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_invariants.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/invariants.py tests/test_ladder_invariants.py
git commit -m "feat(ladder): four invariants on the artefact, because exit code 0 is not a measurement

Each exists because its failure mode produces a PLAUSIBLE NUMBER rather than an error: a system
caching across rings passes every rung and looks strong; a d=0 with no survivors is d=max under
another name; originals nobody answers are broken questions, not hard ones.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: The runner and the RE-call adapter

**Files:**
- Create: `benchmarks/ladder/run.py`
- Create: `benchmarks/ladder/systems/__init__.py`
- Create: `benchmarks/ladder/systems/recall_system.py`
- Test: `tests/test_ladder_run.py`

**Interfaces:**
- Consumes: `MemorySystem`/`Document` (Task 7), `read_manifest`/`Instance`/`RING_ORIGINAL` (Task 3),
  invariants (Task 8), `load_locomo` (Task 5)
- Produces:
  - `run(manifest_path: Path, system: MemorySystem, out_path: Path, *, documents: Mapping[str, str], cluster_members: Mapping[str, Sequence[str]], resume: bool = True) -> int`
  - `RecallSystem(dsn: str)` implementing `MemorySystem`
  - Output JSONL rows: `{"instance_id", "system", "abstained", "cited_ids", "tokens"}`

The runner groups instances by their excised set so each distinct corpus state is ingested **once**,
not once per instance — LOCOMO is ~2 000 turns and re-ingesting per instance would be hours of
embedding for no information.

- [ ] **Step 1: Write the failing test (runner logic only, no database)**

Create `tests/test_ladder_run.py`:

```python
"""The runner: ingest once per distinct corpus state, and let invariants stop a bad run early.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

These tests use a fake MemorySystem on purpose. The Postgres-backed adapter is exercised by the
real run; what needs pinning here is the runner's own logic, which is where a silent defect would
cost a whole overnight job.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from benchmarks.ladder.adapter import Document, Response
from benchmarks.ladder.invariants import InvariantViolation
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_ORIGINAL,
    Instance,
    write_manifest,
)
from benchmarks.ladder.run import run

DOCS = {f"c/D1:{i}": f"turn {i} about the support group" for i in range(1, 5)}


class _Fake:
    name = "fake"

    def __init__(self, *, leak: bool = False) -> None:
        self._docs: dict[str, str] = {}
        self.ingest_calls = 0
        self._leak = leak

    def ingest(self, docs: Iterable[Document]) -> None:
        self.ingest_calls += 1
        incoming = {d.doc_id: d.text for d in docs}
        # A leaking system MERGES instead of replacing — exactly what invariant 1 exists to catch.
        self._docs = {**self._docs, **incoming} if self._leak else incoming

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset(self._docs)

    def query(self, question: str) -> Response:
        if not self._docs:
            return Response(answer=None)
        first = sorted(self._docs)[0]
        return Response(answer=self._docs[first], cited_ids=(first,), tokens=10)


def _manifest(tmp_path: Path) -> Path:
    instances = [
        Instance(
            instance_id="p1#original", corpus="locomo", source_question_id="q1",
            question="when?", label=LABEL_ANSWERABLE, ring=RING_ORIGINAL,
            excised_doc_ids=(), gold_doc_ids=("c/D1:1",), pair_id="p1",
        ),
        Instance(
            instance_id="p1#d0", corpus="locomo", source_question_id="q1",
            question="when?", label=LABEL_UNANSWERABLE, ring=0,
            excised_doc_ids=("c/D1:1",), gold_doc_ids=("c/D1:1",), pair_id="p1",
        ),
        Instance(
            instance_id="p2#d0", corpus="locomo", source_question_id="q2",
            question="who?", label=LABEL_UNANSWERABLE, ring=0,
            excised_doc_ids=("c/D1:1",), gold_doc_ids=("c/D1:1",), pair_id="p2",
        ),
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, instances, ring_widths=[0], corpus_hashes={"locomo": "x"})
    return path


def test_writes_one_row_per_instance(tmp_path: Path):
    out = tmp_path / "responses.jsonl"
    run(_manifest(tmp_path), _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["instance_id"] for r in rows} == {"p1#original", "p1#d0", "p2#d0"}


def test_ingests_once_per_distinct_excision_set_not_once_per_instance(tmp_path: Path):
    system = _Fake()
    run(_manifest(tmp_path), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members={"c": tuple(DOCS)})
    # Two distinct states: nothing excised, and {c/D1:1} excised.
    assert system.ingest_calls == 2


def test_abstention_is_recorded_as_a_boolean(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    run(_manifest(tmp_path), _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    rows = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
    }
    assert isinstance(rows["p1#d0"]["abstained"], bool)


def test_a_system_that_merges_instead_of_replacing_is_stopped(tmp_path: Path):
    with pytest.raises(InvariantViolation, match="still indexed"):
        run(_manifest(tmp_path), _Fake(leak=True), tmp_path / "r.jsonl", documents=DOCS,
            cluster_members={"c": tuple(DOCS)})


def test_resume_skips_instances_already_written(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    manifest = _manifest(tmp_path)
    run(manifest, _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    system = _Fake()
    run(manifest, system, out, documents=DOCS, cluster_members={"c": tuple(DOCS)}, resume=True)
    assert system.ingest_calls == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_run.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.run'`

- [ ] **Step 3: Write the runner**

Create `benchmarks/ladder/run.py`:

```python
"""Run one MemorySystem over a manifest: ingest per distinct corpus state, record abstain-or-answer.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Instances are grouped by their EXCISED SET, not iterated one by one. LOCOMO is ~2 000 turns; a
re-ingest per instance would spend hours of embedding to produce a corpus state it already had.

Rows are flushed per instance and `--resume` skips by instance id, so a run that dies overnight
resumes without re-paying. Resume does NOT verify the system config — resuming across a config
change silently mixes two arms into one artifact.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.ladder.adapter import Document, MemorySystem
from benchmarks.ladder.invariants import (
    assert_excised_absent,
    assert_originals_were_answered,
    assert_ring_zero_has_survivors,
)
from benchmarks.ladder.manifest import RING_ORIGINAL, Instance, read_manifest


def _already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["instance_id"])
    return done


def _cluster_of(instance: Instance, cluster_members: Mapping[str, Sequence[str]]) -> Sequence[str]:
    # Doc ids are "{cluster_id}/{dia_id}", so the gold id names its own cluster.
    if not instance.gold_doc_ids:
        return ()
    cluster_id = instance.gold_doc_ids[0].split("/", 1)[0]
    return cluster_members.get(cluster_id, ())


def run(
    manifest_path: Path,
    system: MemorySystem,
    out_path: Path,
    *,
    documents: Mapping[str, str],
    cluster_members: Mapping[str, Sequence[str]],
    resume: bool = True,
) -> int:
    """Returns the number of instances scored in this invocation."""
    instances, _header = read_manifest(manifest_path)
    done = _already_done(out_path) if resume else set()

    by_state: dict[tuple[str, ...], list[Instance]] = {}
    for inst in instances:
        if inst.instance_id in done:
            continue
        by_state.setdefault(tuple(sorted(inst.excised_doc_ids)), []).append(inst)

    answered_originals: dict[str, bool] = {}
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for excised, group in sorted(by_state.items()):
            dropped = set(excised)
            keep = [Document(d, t) for d, t in sorted(documents.items()) if d not in dropped]
            system.ingest(keep)
            indexed = system.indexed_doc_ids()
            for inst in sorted(group, key=lambda i: i.instance_id):
                assert_excised_absent(inst, indexed)
                assert_ring_zero_has_survivors(inst, indexed, _cluster_of(inst, cluster_members))
                response = system.query(inst.question)
                if inst.ring == RING_ORIGINAL:
                    answered_originals[inst.instance_id] = not response.abstained
                fh.write(
                    json.dumps(
                        {
                            "instance_id": inst.instance_id,
                            "system": system.name,
                            "abstained": response.abstained,
                            "cited_ids": list(response.cited_ids),
                            "tokens": response.tokens,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                fh.flush()
                written += 1

    if answered_originals:
        assert_originals_were_answered(answered_originals, instances)
    return written


def main(argv: list[str] | None = None) -> int:
    from benchmarks.ladder.sources.locomo import load_locomo
    from benchmarks.ladder.systems.recall_system import RecallSystem

    parser = argparse.ArgumentParser(description="Run a system over the Answerability Ladder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_locomo(args.locomo)
    system = RecallSystem(args.dsn)
    n = run(
        args.manifest,
        system,
        args.out,
        documents=dict(corpus.documents),
        cluster_members=corpus.cluster_members,
        resume=not args.no_resume,
    )
    print(f"scored {n} instances into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_run.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Write the RE-call adapter**

Create `benchmarks/ladder/systems/__init__.py`:

```python
"""Reference `MemorySystem` implementations."""
```

Create `benchmarks/ladder/systems/recall_system.py`. **Before writing it, read
`recall/eval/locomo.py:160-243`** (`write_conversation_corpus` and `index_conversation`) for the
exact store/embedder/indexer wiring this repo uses, and mirror it — a bespoke in-memory loader
would measure a code path no user runs, which is the reason that file writes files to disk. The
adapter must:

1. **Replace, not merge.** Each `ingest` call starts from an empty store (drop/recreate or truncate
   the table namespace) so invariant 1 can pass. If it merges, Task 8's assertion will stop the run
   at the first rung — that is the assertion working, not a bug to route around.
2. Write each `Document` to a temp directory as `<sanitised doc_id>.md` and route it through
   `Indexer.index_path`, matching `write_conversation_corpus`'s reasoning. Keep the doc_id ↔
   filename mapping invertible; `_dia_id_to_filename` / `_filename_to_dia_id` in
   `recall/eval/locomo.py:125-138` are the pattern to follow (note ids here also contain `/`).
3. `query` calls the shipped retriever **at shipped defaults** and returns `Response(answer=None)`
   when RE-call abstains, else `Response(answer=<top chunk text>, cited_ids=(<doc ids>,),
   tokens=<measured>)`.
4. `indexed_doc_ids()` reads `store.iter_chunks()` and maps chunk sources back to doc ids with the
   inverse of step 2.

**RE-call runs at shipped defaults.** Any tuned variant is a separately labelled arm and never the
headline — `SUITE-DESIGN.md` rule 4.

- [ ] **Step 6: Run the real arm**

```bash
cd ~/Documents/recall
make db-up
python -m benchmarks.ladder.run \
  --manifest results/ladder/manifest.jsonl \
  --locomo locomo10.json \
  --out results/ladder/responses_recall.jsonl \
  --dsn "$RECALL_DSN"
```

Expected: `scored N instances`. If an `InvariantViolation` fires, **stop and fix the cause** — it is
reporting that the artefact contradicts the design, which is the one thing that must never be
worked around.

- [ ] **Step 7: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/run.py benchmarks/ladder/systems/ tests/test_ladder_run.py results/ladder/responses_recall.jsonl
git commit -m "feat(ladder): the runner and the RE-call adapter, ingesting once per corpus state

Instances are grouped by excised set rather than iterated: LOCOMO is ~2000 turns and a re-ingest
per instance would spend hours of embedding to rebuild a state it already had.

Invariants run inside the loop, so a system that merges instead of replacing stops the run at the
first rung instead of producing a full, plausible, wrong curve.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: The scorer — 2×2 per ring, λ-pricing, and the H1 gate

**Files:**
- Create: `benchmarks/ladder/score.py`
- Test: `tests/test_ladder_score.py`

**Interfaces:**
- Consumes: `Instance`/`LABEL_*` (Task 3), runner JSONL rows (Task 9)
- Produces:
  - `Cell` frozen dataclass: `correct_abstain: int, false_answer: int, false_abstain: int, answered_answerable: int`
  - `confusion_by_ring(instances: Sequence[Instance], abstained: Mapping[str, bool]) -> dict[int, Cell]`
  - `correct_abstain_rate(cell: Cell) -> float`
  - `lambda_cost(cell: Cell, lam: float) -> float`
  - `paired_difference_ci(instances, abstained, low: int, high: int, *, seed: int = 0, iterations: int = 10_000) -> tuple[float, float, float]`
  - `h1_verdict(difference: float, ci_low: float, ci_high: float, *, threshold: float = 0.15) -> str`

**Naming is load-bearing.** The field is `answered_answerable`, not `correct_answer`: v1 has no
judge, so answering an answerable question is scored as success **without verifying the content**.
Calling it `correct_answer` would bake an overclaim into every downstream number.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_score.py`:

```python
"""2x2 per rung, lambda-pricing, and the H1 gate that can kill the benchmark.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

`answered_answerable` is deliberately not called `correct_answer`: v1 has no judge, so answering
an answerable question counts as success WITHOUT the content being checked. Every v1 accuracy is
therefore an upper bound, and the field name is where that is enforced.
"""
from __future__ import annotations

from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    RING_ORIGINAL,
    Instance,
)
from benchmarks.ladder.score import (
    Cell,
    confusion_by_ring,
    correct_abstain_rate,
    h1_verdict,
    lambda_cost,
    paired_difference_ci,
)


def _i(instance_id: str, label: str, ring: int, pair_id: str) -> Instance:
    return Instance(
        instance_id=instance_id, corpus="locomo", source_question_id=pair_id,
        question="q", label=label, ring=ring,
        excised_doc_ids=("g",) if label == LABEL_UNANSWERABLE else (),
        gold_doc_ids=("g",), pair_id=pair_id,
    )


def test_unanswerable_abstention_is_a_correct_abstain():
    cells = confusion_by_ring([_i("a", LABEL_UNANSWERABLE, 0, "p1")], {"a": True})
    assert cells[0].correct_abstain == 1
    assert cells[0].false_answer == 0


def test_unanswerable_answered_is_a_false_answer():
    cells = confusion_by_ring([_i("a", LABEL_UNANSWERABLE, 0, "p1")], {"a": False})
    assert cells[0].false_answer == 1


def test_answerable_abstention_is_a_false_abstain():
    cells = confusion_by_ring([_i("o", LABEL_ANSWERABLE, RING_ORIGINAL, "p1")], {"o": True})
    assert cells[RING_ORIGINAL].false_abstain == 1


def test_answerable_answered_counts_as_answered_not_correct():
    cells = confusion_by_ring([_i("o", LABEL_ANSWERABLE, RING_ORIGINAL, "p1")], {"o": False})
    assert cells[RING_ORIGINAL].answered_answerable == 1
    assert not hasattr(cells[RING_ORIGINAL], "correct_answer")


def test_instances_with_no_recorded_response_are_skipped_not_counted_as_abstentions():
    """A missing row is missing data. Counting it as an abstention would flatter a crashed run."""
    assert confusion_by_ring([_i("a", LABEL_UNANSWERABLE, 0, "p1")], {}) == {}


def test_lambda_weights_a_false_answer_more_heavily_as_lambda_rises():
    cell = Cell(correct_abstain=0, false_answer=1, false_abstain=1, answered_answerable=0)
    assert lambda_cost(cell, 10.0) > lambda_cost(cell, 1.0)


def test_lambda_one_weights_the_two_errors_equally():
    a = Cell(correct_abstain=0, false_answer=2, false_abstain=0, answered_answerable=0)
    b = Cell(correct_abstain=0, false_answer=0, false_abstain=2, answered_answerable=0)
    assert lambda_cost(a, 1.0) == lambda_cost(b, 1.0)


def test_correct_abstain_rate_is_over_unanswerable_instances_only():
    cell = Cell(correct_abstain=3, false_answer=1, false_abstain=5, answered_answerable=5)
    assert correct_abstain_rate(cell) == 0.75


def test_h1_passes_only_when_the_ci_excludes_zero_and_the_gap_is_large():
    assert h1_verdict(0.30, 0.20, 0.40) == "PASS"
    assert h1_verdict(0.30, -0.05, 0.60) == "FAIL"   # CI includes zero
    assert h1_verdict(0.05, 0.01, 0.09) == "FAIL"    # significant but below threshold


def test_paired_difference_uses_only_questions_present_at_both_rungs():
    instances = [
        _i("p1#d0", LABEL_UNANSWERABLE, 0, "p1"),
        _i("p1#dmax", LABEL_UNANSWERABLE, RING_MAX, "p1"),
        _i("p2#d0", LABEL_UNANSWERABLE, 0, "p2"),  # no d=max partner
    ]
    abstained = {"p1#d0": False, "p1#dmax": True, "p2#d0": False}
    diff, low, high = paired_difference_ci(instances, abstained, 0, RING_MAX, iterations=200)
    assert diff == 1.0  # the one paired question flips 0 -> 1
    assert low <= diff <= high
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_score.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.score'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/score.py`:

```python
"""The 2x2 per rung, the lambda-priced curve, and the H1 gate.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Stdlib only, matching `recall/eval/metrics.py` and `recall/eval/gap_study.py`: the arithmetic that
decides whether this benchmark lives should be readable without trusting a library.

`answered_answerable` is NOT `correct_answer`. v1 has no judge, so answering an answerable question
is scored as success without the content being verified. Every accuracy computed from this module
is an UPPER BOUND, and the field name is where that is enforced rather than remembered.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchmarks.ladder.manifest import LABEL_ANSWERABLE, LABEL_UNANSWERABLE, Instance


@dataclass(frozen=True)
class Cell:
    correct_abstain: int
    false_answer: int
    false_abstain: int
    answered_answerable: int


def confusion_by_ring(
    instances: Sequence[Instance], abstained: Mapping[str, bool]
) -> dict[int, Cell]:
    """Ring -> 2x2.

    Instances with no recorded response are SKIPPED, not counted as abstentions: a missing row is
    missing data, and scoring it as an abstention would flatter a crashed run into looking
    well-calibrated.
    """
    counts: dict[int, list[int]] = {}
    for inst in instances:
        if inst.instance_id not in abstained:
            continue
        cell = counts.setdefault(inst.ring, [0, 0, 0, 0])
        did_abstain = abstained[inst.instance_id]
        if inst.label == LABEL_UNANSWERABLE:
            cell[0 if did_abstain else 1] += 1
        elif inst.label == LABEL_ANSWERABLE:
            cell[2 if did_abstain else 3] += 1
    return {ring: Cell(*vals) for ring, vals in counts.items()}


def correct_abstain_rate(cell: Cell) -> float:
    """Over unanswerable instances only. Returns 0.0 when there are none."""
    n = cell.correct_abstain + cell.false_answer
    return cell.correct_abstain / n if n else 0.0


def lambda_cost(cell: Cell, lam: float) -> float:
    """Expected cost with cost(false answer) = lam x cost(false abstention).

    lam=1 reproduces BEAM's implicit weighting. Fixed in PREREGISTRATION-ladder.md before results.
    """
    return lam * cell.false_answer + cell.false_abstain


def _paired_flags(
    instances: Sequence[Instance], abstained: Mapping[str, bool], low: int, high: int
) -> tuple[list[int], list[int]]:
    """Per pair_id, the abstention flag at each of two rungs — pairs present at BOTH only."""
    at: dict[int, dict[str, int]] = {low: {}, high: {}}
    for inst in instances:
        if inst.ring in at and inst.instance_id in abstained:
            at[inst.ring][inst.pair_id] = int(abstained[inst.instance_id])
    shared = sorted(set(at[low]) & set(at[high]))
    return [at[low][p] for p in shared], [at[high][p] for p in shared]


def paired_difference_ci(
    instances: Sequence[Instance],
    abstained: Mapping[str, bool],
    low: int,
    high: int,
    *,
    seed: int = 0,
    iterations: int = 10_000,
) -> tuple[float, float, float]:
    """Mean paired difference in correct-abstain (high rung minus low), with a bootstrap 95 % CI.

    Paired because the same question appears at both rungs; an unpaired test would attribute
    question difficulty to the rung.
    """
    lo_flags, hi_flags = _paired_flags(instances, abstained, low, high)
    n = len(lo_flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    deltas = [h - lo for lo, h in zip(lo_flags, hi_flags)]
    observed = sum(deltas) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(iterations))
    return observed, means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def h1_verdict(
    difference: float, ci_low: float, ci_high: float, *, threshold: float = 0.15
) -> str:
    """P1 from the pre-registration: difference > 0.15 AND the bootstrap CI excludes zero.

    FAIL is the kill condition, not a disappointing result: a flat curve means the axis is a
    fiction, and no paid comparative arm runs.
    """
    excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
    return "PASS" if difference > threshold and excludes_zero else "FAIL"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_score.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures. `tests/test_findings_crossrefs.py` must still pass — if a new module
cited a lettered section like `§9h`, register it in `EXPECTED` with a substring of that section's
current title.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/score.py tests/test_ladder_score.py
git commit -m "feat(ladder): 2x2 per rung, lambda-priced curve, and the H1 kill gate

The field is answered_answerable, not correct_answer: v1 has no judge, so answering an answerable
question counts as success without the content being checked. Every v1 accuracy is an upper bound
and the field name is where that is enforced rather than remembered.

A missing response row is skipped, not scored as an abstention — counting it would flatter a
crashed run into looking well-calibrated.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: The H1 verdict, published whichever way it falls

**Files:**
- Create: `benchmarks/ladder/report.py`
- Create: `results/ladder/H1_VERDICT.md`
- Test: `tests/test_ladder_report.py`

**Interfaces:**
- Consumes: `read_manifest`/`RING_MAX`/`RING_ORIGINAL` (Task 3), all of `score.py` (Task 10)
- Produces: `main(argv: list[str] | None = None) -> int` — prints the curve, the 2×2 per rung, the
  λ-costs and the H1 verdict; returns 1 on FAIL

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder_report.py`:

```python
"""The report prints the verdict it computed — including FAIL.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

A report that can only render a PASS is not a report. `test_a_flat_curve_prints_fail...` is the
test that matters here: it exercises the DETECTION path, not the green path.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ladder.manifest import LABEL_UNANSWERABLE, RING_MAX, Instance, write_manifest
from benchmarks.ladder.report import main


def _setup(tmp_path: Path, *, flat: bool) -> tuple[Path, Path]:
    instances = []
    rows = []
    for i in range(40):
        for ring in (0, RING_MAX):
            iid = f"p{i}#d{ring}"
            instances.append(
                Instance(
                    instance_id=iid, corpus="locomo", source_question_id=f"q{i}",
                    question="q", label=LABEL_UNANSWERABLE, ring=ring,
                    excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id=f"p{i}",
                )
            )
            abstained = False if flat else (ring == RING_MAX)
            rows.append({"instance_id": iid, "system": "recall", "abstained": abstained,
                         "cited_ids": [], "tokens": 0})
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, instances, ring_widths=[0], corpus_hashes={})
    responses = tmp_path / "responses.jsonl"
    responses.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return manifest, responses


def test_a_separating_curve_prints_pass(tmp_path: Path, capsys):
    manifest, responses = _setup(tmp_path, flat=False)
    assert main(["--manifest", str(manifest), "--responses", str(responses)]) == 0
    assert "H1: PASS" in capsys.readouterr().out


def test_a_flat_curve_prints_fail_and_says_the_benchmark_is_dead(tmp_path: Path, capsys):
    manifest, responses = _setup(tmp_path, flat=True)
    assert main(["--manifest", str(manifest), "--responses", str(responses)]) == 1
    out = capsys.readouterr().out
    assert "H1: FAIL" in out
    assert "kill condition" in out.lower()


def test_the_report_prints_every_ring_it_has_data_for(tmp_path: Path, capsys):
    manifest, responses = _setup(tmp_path, flat=False)
    main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "d=0" in out and "d=max" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ladder_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.ladder.report'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/ladder/report.py`:

```python
"""Print the ladder: 2x2 per rung, lambda costs, and the H1 verdict — PASS or FAIL.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Exits 1 on FAIL, so a scheduled run cannot report success merely by finishing. A FAIL here is not
a bug: it is the pre-registered kill condition, and it means no paid comparative arm runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.ladder.manifest import RING_MAX, RING_ORIGINAL, read_manifest
from benchmarks.ladder.score import (
    confusion_by_ring,
    correct_abstain_rate,
    h1_verdict,
    lambda_cost,
    paired_difference_ci,
)

LAMBDAS = (1.0, 3.0, 10.0)


def _ring_label(ring: int) -> str:
    if ring == RING_MAX:
        return "d=max"
    if ring == RING_ORIGINAL:
        return "original"
    return f"d={ring}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the Answerability Ladder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    args = parser.parse_args(argv)

    instances, header = read_manifest(args.manifest)
    abstained: dict[str, bool] = {}
    for line in args.responses.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            abstained[row["instance_id"]] = bool(row["abstained"])

    cells = confusion_by_ring(instances, abstained)

    print(f"manifest digest {header['digest']}  n_instances={header['n_instances']}")
    print(f"scored responses: {len(abstained)}")
    print()
    print(
        f"{'rung':<10}{'n':>6}{'corr-abst':>11}{'false-ans':>11}{'false-abst':>12}"
        + "".join(f"{'L=' + str(int(lam)):>8}" for lam in LAMBDAS)
    )
    for ring in sorted(cells, key=lambda r: (r == RING_MAX, r)):
        cell = cells[ring]
        n = (
            cell.correct_abstain
            + cell.false_answer
            + cell.false_abstain
            + cell.answered_answerable
        )
        print(
            f"{_ring_label(ring):<10}{n:>6}{correct_abstain_rate(cell):>11.3f}"
            f"{cell.false_answer:>11}{cell.false_abstain:>12}"
            + "".join(f"{lambda_cost(cell, lam):>8.1f}" for lam in LAMBDAS)
        )

    print()
    diff, low, high = paired_difference_ci(instances, abstained, 0, RING_MAX)
    verdict = h1_verdict(diff, low, high)
    print(f"H1 paired delta(correct-abstain), d=max - d=0: {diff:+.3f} [{low:+.3f}, {high:+.3f}]")
    print(f"H1: {verdict}")
    if verdict == "FAIL":
        print(
            "\nThis is the pre-registered kill condition, not a disappointing result. A flat curve "
            "means excision distance is not the axis this benchmark claimed it was. Do NOT run the "
            "Mem0 arm, and publish this."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ladder_report.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Produce the real verdict**

```bash
cd ~/Documents/recall
python -m benchmarks.ladder.report \
  --manifest results/ladder/manifest.jsonl \
  --responses results/ladder/responses_recall.jsonl \
  | tee results/ladder/H1_VERDICT.txt
```

Then write `results/ladder/H1_VERDICT.md`: paste the table verbatim, state the verdict, and check
each of P1–P3 from the pre-registration against what was measured — **including the ones that
missed.** A prediction that missed is the most informative line in the file.

- [ ] **Step 6: Run the P4 robustness arm — only if H1 PASSED**

```bash
cd ~/Documents/recall
python -m benchmarks.ladder.build --locomo locomo10.json --out results/ladder/manifest_random.jsonl --widths 0,4,16,64 --random-seed 7
python -m benchmarks.ladder.run --manifest results/ladder/manifest_random.jsonl --locomo locomo10.json --out results/ladder/responses_recall_random.jsonl --dsn "$RECALL_DSN"
python -m benchmarks.ladder.report --manifest results/ladder/manifest_random.jsonl --responses results/ladder/responses_recall_random.jsonl | tee results/ladder/H1_VERDICT_random.txt
```

If the sign or rough magnitude of the paired delta differs from the BM25 arm, **P4 has failed**:
BM25 is a confound, the curve is measuring the neighbour function, and it does not ship as an
answerability result. Record that in `H1_VERDICT.md` and stop.

- [ ] **Step 7: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/ladder/report.py tests/test_ladder_report.py results/ladder/
git commit -m "bench(ladder): the H1 verdict, published whichever way it fell

The report exits 1 on FAIL so a scheduled run cannot report success by finishing, and the FAIL
path is the one covered by test — a report that can only render a PASS is not a report.

H1_VERDICT.md checks every pre-registered prediction including the ones that missed; a prediction
that missed is the most informative line in the file.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Out of scope for this plan (plan 2)

Deferred deliberately, each with the reason:

- **The Mem0 arm.** Costs API money at ingest; OpenRouter credits are exhausted. Gated on H1 PASS.
- **The BEIR generality arm.** Reuses `recall/eval/beir.py`; adds nothing to the H1 kill decision.
- **LongMemEval as a second anchor.** Same structure via `answer_session_ids`; one anchor answers H1.
- **Answer-quality judging (nugget-based).** Requires first reproducing the 62.81 % judge-acceptance
  finding on our own dumps. Until then every v1 accuracy is an upper bound and is labelled as one.
- **H2 external check** against LOCOMO category 5 (0.00/446) and BEAM's abstention category (0.467).
- **Extraction of `benchmarks/ladder/` into an installable package** so third parties need not
  install `recall-rag` to score their own system.
- **The write-up and public release** of the manifest.

## Self-review notes

Checked against the spec, section by section:

| Spec section | Covered by |
|---|---|
| 1 — claim, H1/H2/H3 | Task 1 (P1–P4), Task 10 (`h1_verdict`), Task 11 (verdict + P4 arm) |
| 2 — four units | Tasks 3 (manifest), 6 (builder), 7 (adapter), 10 (scorer) |
| 2.1 — adapter boundary | Task 7 |
| 3 — excision ladder | Task 4 |
| 3.1 — BM25 decoupling | Task 2 |
| 4 — scoring, λ, no headline scalar | Task 10 |
| 4.1 — nugget-based answer quality | **Deferred to plan 2**, stated above and enforced meanwhile by the `answered_answerable` field name |
| 5 — five invariants | Task 8 (1–4), Task 11 Step 6 (5, the ring-robustness arm) |
| 6 — how we lose | Task 1 "Known to cut against us"; Task 11 publishes the verdict either way |
| 7 — v1 scope | This plan is v1 minus the deferred arms listed above |
| 8 — honest risks | 8.1 in Task 5 (evidence-based excision, category 5 excluded); 8.2/8.3 deferred; 8.4 accepted in Task 10 (no headline scalar); 8.5 and 8.6 unchanged, documentation-only |

Two gaps, stated rather than hidden:

1. **Invariant 5 (ring robustness) runs as a manual step in Task 11, not as an automated
   assertion.** It compares two full runs, so it cannot be an in-process assert — but that also
   means nothing fails loudly if it is skipped. Plan 2 should add a check that refuses to publish a
   curve with no recorded P4 result.
2. **Task 9 Step 5 (the RE-call adapter) is specified as requirements plus a pointer to the code to
   mirror, not as complete code.** Every other step in this plan shows the code. This one does not,
   because the store/embedder/indexer wiring must match `recall/eval/locomo.py` exactly and
   inventing it here would risk shipping a second indexing path — the specific failure that file's
   own docstring warns against. The implementer must read that file first.
