# PEPs Truth-Extraction Labelled Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen, publishable labelled set for prose supersession extraction from `python/peps`, so that precision and recall become measurable for the first time.

**Architecture:** PEP headers (`Superseded-By:` / `Replaces:`) are the gold label, authored years ago by people who have never seen this project, so the positive class costs no labelling and carries no labelling bias. A pure parsing module turns 733 `.rst` into header edges, prose markers and restatement counts; four thin CLIs on top of it emit a census artifact, a frozen gold manifest, a blind adjudication pack, and a trust query set. Freezing reuses `recall/eval/promotion/manifest.py` unchanged; blinding reuses the shuffle-and-separate-key shape of `benchmarks/labelling/build_beam_labelling.py`.

**Tech Stack:** Python 3.11+, stdlib only (`re`, `csv`, `json`, `hashlib`, `pathlib`). No model, no database, no network at run time. `recall.lint.CLOSURE_MARKERS`, `recall.eval.promotion.manifest`, `recall.eval.provenance`.

---

## Measured before planning — read this first

I cloned `python/peps` at HEAD `5981b2a292610104eb30735423504c52fe454650` and ran the census
probes before writing this plan. Three numbers in the spec did not survive contact, and one of
them changes what this set can be used for.

| quantity | spec said | measured | note |
|---|---|---|---|
| `n_files` (`.rst`) | 733 | **733** | prereg still accurate at this SHA; 0 `.md`, no Markdown migration yet |
| `n_header_edges` | 55–95 | **47** | below the predicted floor |
| `n_prose_marker_files` | — | **209** | matches `CLOSURE_MARKERS` in the body |
| `n_marker_without_header` | "the 60 vs 2 analogue" | **175** | this is the adjudication pool |
| `n_restated_in_prose` | the recall ceiling | **8 of 47 = 17.0%** | the finding that matters |
| adjudicable rows from that pool | — | **38 rows from 30 files** | the remaining 145 name no target |

The adjudication pool is 175 files but **38 adjudicable rows**. Only 30 of the 175 name a
candidate target in the same sentence as the marker. The other 145 carry a bare marker with no
target, which `fix.py`'s "a fix is proposed only when the target is PROVABLE" rule reports as
needing a human rather than guessing at — they are counted in the census and excluded from
adjudication, because there is no candidate pair to adjudicate.

**Correction, recorded rather than quietly overwritten.** This row first read 54 rows from 32
files. That figure was measured with the pre-fix sentence splitter, before Task 1 was corrected to
stop unwrapping newlines across paragraph boundaries. Sixteen of those 54 were spurious pairs
built by gluing a marker in one paragraph to a PEP reference in another — the exact
false-positive channel Task 1 removed. Under the shipped splitter the count is 38 from 30 files,
and the sixteen that vanished were junk that would otherwise have gone in front of an
adjudicator as real work.

**The 17% ceiling is the headline result, and it is exactly what step 1 exists to surface.** Of 47
authored header edges, only 8 are restated in prose with the marker and the partner PEP's number
in the same sentence. A perfect prose extractor scores recall 8/47 = 0.170 against the header
denominator. Bracketing confirms it is a corpus fact and not a windowing artifact: the count is 8
at both sentence and paragraph windows, and rises to 26/47 only under whole-body co-occurrence,
which pairs a marker in one section with a PEP reference in another and is precisely the loose
matching `fix.py`'s docstring records as producing garbage. **The tight window is the ceiling; the
whole-body number is an upper bracket no precision-respecting extractor should reach.**

The load-bearing consequence, which must be stated in the artifact: **the positive class usable
for prose-extraction recall is 8, not 47.** Wilson on n=8 is about as uninterpretable as the n=4
this plan's trust set exists to fix. So this set measures **precision** well (38 adjudicable
candidate pairs, good power) and **recall** poorly (n=8), and the census is what makes that
visible instead of letting a 17% recall read as a model failure.

The 47 edges remain fully sound for the downstream trust query set, where the label is the
supersession relation itself rather than whether prose states it.

## Global Constraints

- Tests live flat as `tests/test_<area>_<subject>.py`. No new test subdirectories.
- A boundary someone could violate gets a `tests/test_*_contract.py` whose docstring enumerates the properties, one test per property, each written so a plausible wrong implementation fails.
- Anything writing an artifact gets a validator **at the write site**, one `pytest.raises(match=<field>)` per rejection path, one non-over-rejection test, and one test that the write site actually calls the validator. Pattern: `benchmarks/artifact_contract.py`.
- `python -m ruff check .` must be clean. Bare `ruff` on this machine is a stale 0.6.9 — always `python -m ruff`. **Never** run `ruff format`.
- mypy is run **scoped**: `python -m mypy benchmarks/labelling/truth_extraction`. Whole-repo `python -m mypy .` already fails on master with a pre-existing duplicate-module error between `results/mtrag_taskA_dev/paired_rerank_test.py` and `results/mtrag_generation/scripts/paired_rerank_test.py` (last touched in `f37111f`, unrelated to this work). Do not try to fix it here, and do not treat it as a regression.
- Never inject CRLF. Use the Edit tool, or `newline="\n"` when scripting a write. Every artifact writer in this plan opens with `newline="\n"` so bytes do not depend on the freezing OS.
- No new dependencies. Stdlib only.
- No test may pass silently when the PEPs corpus is absent. Corpus-dependent tests `pytest.skip` with an explicit reason; committed-artifact cross-checks always run.
- Every guard must be mutated and watched go red before it is claimed to work. Report test output, not assertions about test output.
- `RECALL_DSN` stays unset. Nothing here touches a database.

## Corpus acquisition (prerequisite, not a task)

The 733 `.rst` are **not** vendored; they are referenced by SHA, per `docs/EVIDENCE.md:242`.

```bash
git clone --depth 1 https://github.com/python/peps /tmp/peps && git -C /tmp/peps rev-parse HEAD
```

Every CLI below takes `--peps-dir /tmp/peps/peps` (note the nested `peps/` subdirectory that
holds the files). Set `RECALL_PEPS_DIR` to the same path so the corpus-dependent tests run rather
than skip.

## File Structure

| file | responsibility |
|---|---|
| `benchmarks/labelling/truth_extraction/__init__.py` | package marker, empty |
| `benchmarks/labelling/truth_extraction/peps_header.py` | pure parsing: RFC822 headers, body split, edge extraction, reference detection, restatement windows. No writes. |
| `benchmarks/labelling/truth_extraction/artifact_contract.py` | `validate_census(payload)` — the write-site validator |
| `benchmarks/labelling/truth_extraction/census.py` | census counts + `_provenance`, writes `results/truth_extraction/census.json` |
| `benchmarks/labelling/truth_extraction/build_gold.py` | freezes `gold.manifest.jsonl` via `manifest.write_manifest` |
| `benchmarks/labelling/truth_extraction/build_adjudication.py` | blind `adjudication.csv` + separate `adjudication_key.json` |
| `benchmarks/labelling/truth_extraction/build_trust_queries.py` | writes `recall/eval/peps_trust_queries.json` |
| `benchmarks/labelling/truth_extraction/fixtures/*.md` | 4 transplanted negatives from `fix.py`'s private failures |
| `tests/test_truth_extraction_peps_header.py` | parsing unit tests |
| `tests/test_truth_extraction_contract.py` | the boundary contract |
| `tests/test_truth_extraction_census.py` | row-count guard, digest verification, blinding assertions |
| `tests/test_truth_extraction_fixtures.py` | the four transplanted fixtures behave as negatives |

---

### Task 1: PEP header parsing (pure, no I/O)

**Files:**
- Create: `benchmarks/labelling/truth_extraction/__init__.py`
- Create: `benchmarks/labelling/truth_extraction/peps_header.py`
- Test: `tests/test_truth_extraction_peps_header.py`

**Interfaces:**
- Consumes: `recall.lint.CLOSURE_MARKERS`
- Produces:
  - `split_header(text: str) -> tuple[str, str]`
  - `header_fields(head: str) -> dict[str, str]`
  - `pep_refs(text: str) -> set[str]` — zero-padded stems, e.g. `{"pep-0287"}`
  - `Edge` frozen dataclass: `.superseded: str`, `.successor: str` (both zero-padded stems)
  - `edges_from_fields(stem: str, fields: Mapping[str, str]) -> set[Edge]`
  - `sentences(body: str) -> list[str]` — newline-joined before splitting
  - `restates(body: str, partner: str) -> str | None` — the evidence sentence, or None

- [ ] **Step 1: Write the failing test**

Create `tests/test_truth_extraction_peps_header.py`:

```python
"""Parsing PEP RFC822 headers and detecting prose restatements of a header edge."""
from __future__ import annotations

from benchmarks.labelling.truth_extraction.peps_header import (
    Edge,
    edges_from_fields,
    header_fields,
    pep_refs,
    restates,
    sentences,
    split_header,
)

HEAD = """PEP: 216
Title: Docstring Format
Status: Superseded
Superseded-By: 287

Abstract
========
Body text here.
"""


def test_split_header_cuts_at_first_blank_line():
    head, body = split_header(HEAD)
    assert "Superseded-By: 287" in head
    assert "Abstract" in body
    assert "Superseded-By" not in body


def test_split_header_without_blank_line_yields_empty_body():
    head, body = split_header("PEP: 1\nTitle: X")
    assert head == "PEP: 1\nTitle: X"
    assert body == ""


def test_header_fields_parses_rfc822_continuations():
    fields = header_fields("Replaces: 245,\n  246\nTitle: T")
    assert fields["Replaces"] == "245, 246"
    assert fields["Title"] == "T"


def test_pep_refs_accepts_all_three_citation_forms():
    assert pep_refs("see :pep:`287` and PEP 292 and pep-0435") == {
        "pep-0287", "pep-0292", "pep-0435",
    }


def test_pep_refs_zero_pads_so_pep_5_and_pep_0005_are_one_document():
    assert pep_refs("PEP 5") == pep_refs("PEP 0005") == {"pep-0005"}


def test_edges_from_superseded_by_points_away_from_this_pep():
    assert edges_from_fields("pep-0216", {"Superseded-By": "287"}) == {
        Edge(superseded="pep-0216", successor="pep-0287")
    }


def test_edges_from_replaces_points_toward_this_pep():
    # `Replaces:` is active voice — the edge's SUCCESSOR is the document declaring it.
    # Inverting this would demote the live PEP beneath the one it replaced.
    assert edges_from_fields("pep-0440", {"Replaces": "386"}) == {
        Edge(superseded="pep-0386", successor="pep-0440")
    }


def test_edges_from_multivalued_replaces_yields_one_edge_each():
    assert edges_from_fields("pep-3124", {"Replaces": "245, 246"}) == {
        Edge(superseded="pep-0245", successor="pep-3124"),
        Edge(superseded="pep-0246", successor="pep-3124"),
    }


def test_sentences_joins_hard_wrapped_lines_before_splitting():
    # RST hard-wraps prose. Splitting on newlines would cut this restatement in half.
    got = sentences("It has been\nsuperseded by :pep:`287`. Next one.")
    assert got[0].strip() == "It has been superseded by :pep:`287`."


def test_sentences_never_glues_across_a_blank_line():
    # A paragraph with no terminal punctuation must not run into the next section. Gluing here
    # is whole-body co-occurrence arriving through the back door.
    got = sentences("Heading\n\nThis is deprecated\n\nSee :pep:`287` for formatting.")
    assert not any("deprecated" in s and "287" in s for s in got)


def test_sentences_keeps_a_trailing_fragment_with_no_terminator():
    # pep-0634's real restatement ends in a colon. Dropping it loses a true positive.
    got = sentences("It replaces :pep:`622`, which is hereby split in three parts:")
    assert any(":pep:`622`" in s for s in got)


def test_restates_finds_an_unterminated_restatement():
    body = "It replaces :pep:`622`, which is hereby split in three parts:"
    assert restates(body, "pep-0622") is not None


def test_restates_returns_the_evidence_sentence():
    assert restates("It has been superseded by :pep:`287`.", "pep-0287") == (
        "It has been superseded by :pep:`287`."
    )


def test_restates_requires_marker_and_partner_in_the_SAME_sentence():
    # Marker in one sentence, reference in another: whole-body co-occurrence, which is the
    # loose matching fix.py records as producing garbage. Must not count.
    body = "This is deprecated. Unrelatedly, see :pep:`287` for formatting."
    assert restates(body, "pep-0287") is None


def test_restates_returns_none_when_partner_absent():
    assert restates("It has been superseded by :pep:`999`.", "pep-0287") is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_truth_extraction_peps_header.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'benchmarks.labelling.truth_extraction'`.

- [ ] **Step 3: Write the implementation**

Create `benchmarks/labelling/truth_extraction/__init__.py` as an empty file.

Create `benchmarks/labelling/truth_extraction/peps_header.py`:

```python
"""Parse PEP RFC822 headers and detect prose restatements of a header edge.

The header is the gold label. `Superseded-By:` and `Replaces:` describe the SAME relation from
opposite ends, so both are normalised to one `Edge(superseded, successor)` and deduplicated —
otherwise a pair that declares the relation at both ends counts twice and inflates the
denominator every recall number is measured against.

Direction is the hazard this module exists to get right, and it is the same one `fix.py:24-30`
documents for memos: `Replaces:` is active voice, so the declaring document is the SUCCESSOR;
`Superseded-By:` is passive, so the declaring document is the one being replaced. Inverting
either would declare the live document stale and demote it beneath the one it replaced.

Pure and file-free, so the direction rule is testable on strings alone.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from recall.lint import CLOSURE_MARKERS

#: A PEP header block is RFC822: it ends at the first blank line.
_HEADER_END = re.compile(r"\n\s*\n")

#: How a PEP is cited in prose. Three forms, all of which appear in the corpus:
#:   :pep:`287`   the Sphinx role, the modern convention
#:   PEP 287      plain prose
#:   pep-0287     a file-name-shaped reference
_REF = re.compile(r"(?::pep:`|PEP\s*|pep-)(\d{1,4})", re.IGNORECASE)


def split_header(text: str) -> tuple[str, str]:
    """``(header_block, body)``. A file with no blank line is all header and no body."""
    match = _HEADER_END.search(text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end() :]


def header_fields(head: str) -> dict[str, str]:
    """RFC822 fields, folding continuation lines into the field they continue.

    `Replaces: 245,\\n  246` is one field with two values. Reading the continuation as a new
    field would silently drop PEP 246 from the gold set — a missing positive, which is invisible
    in a precision number and lowers recall for a reason that has nothing to do with the model.
    """
    fields: dict[str, str] = {}
    key: str | None = None
    for line in head.split("\n"):
        if not line.strip():
            continue
        if line[0].isspace() and key is not None:
            fields[key] += " " + line.strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            fields[key] = value.strip()
    return fields


def pep_refs(text: str) -> set[str]:
    """Every PEP cited in `text`, as zero-padded stems.

    Zero-padding is what makes `PEP 5`, `PEP 0005` and `pep-0005` one document rather than three.
    """
    return {f"pep-{int(m.group(1)):04d}" for m in _REF.finditer(text)}


@dataclass(frozen=True, order=True)
class Edge:
    """One supersession relation, normalised so both header spellings produce one object."""

    superseded: str
    successor: str


def edges_from_fields(stem: str, fields: Mapping[str, str]) -> set[Edge]:
    """The edges one PEP's headers declare. `stem` is this PEP, zero-padded."""
    edges: set[Edge] = set()
    for number in re.findall(r"\d+", fields.get("Superseded-By", "")):
        edges.add(Edge(superseded=stem, successor=f"pep-{int(number):04d}"))
    for number in re.findall(r"\d+", fields.get("Replaces", "")):
        edges.add(Edge(superseded=f"pep-{int(number):04d}", successor=stem))
    return edges


def sentences(body: str) -> list[str]:
    """Sentences, unwrapping hard line breaks WITHIN a paragraph but never across one.

    Three properties, each of which was measured to matter on this corpus:

    RST wraps prose at column ~79, so a restatement routinely spans two lines. Splitting on
    newlines cut `"It has been\\nsuperseded by :pep:`287`."` in half and lost the reference.

    Unwrapping every newline instead — including blank lines — glues a paragraph to the heading
    and body that follow it, so a marker in one section can pair with a reference in another.
    That is the whole-body co-occurrence this module exists to exclude, arriving through the back
    door. Paragraphs are therefore split first and unwrapped individually.

    The `[^.!?]+` alternative keeps a trailing fragment that has no terminator. Without it,
    `pep-0634`'s `"It replaces :pep:`622`, which is hereby split in three parts:"` — a real
    restatement, ending in a colon — is silently discarded. Before this was fixed the edge was
    still counted, but only because the blank-line gluing ran the fragment into the next
    paragraph: two defects cancelling, on one edge, by luck.
    """
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        flat = re.sub(r"\s*\n\s*", " ", paragraph).strip()
        if flat:
            out.extend(re.findall(r"[^.!?]*[.!?]|[^.!?]+", flat))
    return out


def restates(body: str, partner: str) -> str | None:
    """The sentence stating this edge in prose, or None.

    Requires a closure marker AND the partner's reference in the SAME sentence. Whole-body
    co-occurrence — a marker in one section, the reference in another — is not a restatement;
    on this corpus it more than triples the count (8 to 26 of 47) by pairing text that has no
    relation, which is the failure `fix.py`'s docstring records for looser matching.
    """
    for sentence in sentences(body):
        if CLOSURE_MARKERS.search(sentence) and partner in pep_refs(sentence):
            return sentence
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_truth_extraction_peps_header.py -q
```

Expected: `15 passed`.

- [ ] **Step 5: Mutate each direction guard and watch it go red**

This is the guard that would silently invert the supersession graph. Prove it can fail.

In `peps_header.py`, temporarily swap the `Replaces` branch to
`Edge(superseded=stem, successor=f"pep-{int(number):04d}")`, then run:

```bash
python -m pytest tests/test_truth_extraction_peps_header.py -q
```

Expected: `test_edges_from_replaces_points_toward_this_pep` and
`test_edges_from_multivalued_replaces_yields_one_edge_each` FAIL. Revert.

Then temporarily change `sentences` to `return body.split(".")` and re-run. Expected:
`test_sentences_joins_hard_wrapped_lines_before_splitting` and
`test_restates_returns_the_evidence_sentence` FAIL.

Then temporarily change `sentences` back to the naive single-pass form —
`return re.findall(r"[^.!?]*[.!?]", re.sub(r"\s*\n\s*", " ", body))` — and re-run. Expected:
`test_sentences_never_glues_across_a_blank_line`,
`test_sentences_keeps_a_trailing_fragment_with_no_terminator` and
`test_restates_finds_an_unterminated_restatement` FAIL. This is the form that was shipped first
and it must be provably red. Revert, re-run, confirm `15 passed`.

- [ ] **Step 6: Lint and commit**

```bash
python -m ruff check benchmarks/labelling/truth_extraction tests/test_truth_extraction_peps_header.py && python -m mypy benchmarks/labelling/truth_extraction
```

```bash
git add benchmarks/labelling/truth_extraction/__init__.py benchmarks/labelling/truth_extraction/peps_header.py tests/test_truth_extraction_peps_header.py && git commit -m "feat(truth-extraction): parse PEP headers into normalised supersession edges"
```

---

### Task 2: Census artifact + write-site validator

**Files:**
- Create: `benchmarks/labelling/truth_extraction/artifact_contract.py`
- Create: `benchmarks/labelling/truth_extraction/census.py`
- Test: `tests/test_truth_extraction_contract.py`

**Interfaces:**
- Consumes: Task 1's `split_header`, `header_fields`, `edges_from_fields`, `restates`, `Edge`; `recall.eval.provenance.generated_at`, `.model_stack`; `recall.lint.CLOSURE_MARKERS`
- Produces:
  - `validate_census(payload: Mapping[str, object]) -> None` — raises `ValueError`
  - `Census` frozen dataclass with `.n_files`, `.n_header_edges`, `.n_prose_marker_files`, `.n_marker_without_header`, `.n_restated_in_prose`, `.edges: tuple[Edge, ...]`, `.restatements: dict[str, str]`, `.marker_without_header: tuple[str, ...]`, `.file_digests: dict[str, str]`
  - `compute_census(peps_dir: Path) -> Census`
  - `census_payload(census: Census, *, peps_sha: str, clone_date: str, recall_commit: str, invocation: str) -> dict`
  - `write_census(path: Path, payload: Mapping[str, object]) -> None`

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_truth_extraction_contract.py`:

```python
"""The census artifact boundary.

Properties, one test each:
  1. A payload missing `_provenance` is refused.
  2. A `_provenance` missing `peps_sha` is refused — the artifact would name no corpus version.
  3. A `_provenance` missing `recall_commit` is refused.
  4. A census whose `n_header_edges` disagrees with `len(edges)` is refused, because the two
     are the same fact written twice and a reader cannot tell which one is the typo.
  5. A census whose `n_restated_in_prose` disagrees with `len(restatements)` is refused.
  6. A census claiming more restatements than header edges is refused — the ceiling cannot
     exceed 100%.
  7. A ceiling of EXACTLY 100% is accepted. A corpus that restates every edge is legitimate,
     and refusing it would be a validator rejecting its own best possible input.
  8. A well-formed payload is NOT rejected.
  9. The write site calls the validator.
 10. The writer emits LF regardless of platform.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.labelling.truth_extraction.artifact_contract import validate_census
from benchmarks.labelling.truth_extraction.census import write_census


def _ok() -> dict:
    return {
        "n_files": 733,
        "n_header_edges": 2,
        "n_prose_marker_files": 209,
        "n_marker_without_header": 175,
        "n_restated_in_prose": 1,
        "edges": [
            {"superseded": "pep-0216", "successor": "pep-0287"},
            {"superseded": "pep-0386", "successor": "pep-0440"},
        ],
        "restatements": {"pep-0216->pep-0287": "It has been superseded by :pep:`287`."},
        "marker_without_header": ["pep-0001"],
        "file_digests": {"pep-0216.rst": "a" * 64},
        "_provenance": {
            "peps_sha": "5981b2a292610104eb30735423504c52fe454650",
            "clone_date": "2026-08-11",
            "recall_commit": "439717b",
            "generated_at": "2026-08-11T12:00:00+00:00",
            "model_stack": {},
            "invocation": "python -m benchmarks.labelling.truth_extraction.census ...",
        },
    }


def test_missing_provenance_is_refused():
    payload = _ok()
    del payload["_provenance"]
    with pytest.raises(ValueError, match="_provenance"):
        validate_census(payload)


def test_provenance_without_peps_sha_is_refused():
    payload = _ok()
    del payload["_provenance"]["peps_sha"]
    with pytest.raises(ValueError, match="peps_sha"):
        validate_census(payload)


def test_provenance_without_recall_commit_is_refused():
    payload = _ok()
    del payload["_provenance"]["recall_commit"]
    with pytest.raises(ValueError, match="recall_commit"):
        validate_census(payload)


def test_edge_count_disagreeing_with_edge_list_is_refused():
    payload = _ok()
    payload["n_header_edges"] = 3
    with pytest.raises(ValueError, match="n_header_edges"):
        validate_census(payload)


def test_restated_count_disagreeing_with_restatements_is_refused():
    payload = _ok()
    payload["n_restated_in_prose"] = 2
    with pytest.raises(ValueError, match="n_restated_in_prose"):
        validate_census(payload)


def test_ceiling_above_one_hundred_percent_is_refused():
    # Two restatements against one edge: more edges stated in prose than exist in the headers,
    # which means the restatement detector matched something outside the gold set.
    payload = _ok()
    payload["n_header_edges"] = 1
    payload["edges"] = [{"superseded": "pep-0216", "successor": "pep-0287"}]
    payload["restatements"] = {
        "pep-0216->pep-0287": "It has been superseded by :pep:`287`.",
        "pep-0386->pep-0440": "supersedes :pep:`386` even for metadata v1.",
    }
    payload["n_restated_in_prose"] = 2
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_census(payload)


def test_ceiling_of_exactly_one_hundred_percent_is_accepted():
    # A corpus where EVERY header edge is also stated in prose is a legitimate corpus, not a
    # malformed artifact. This pins the comparison at `>` and not `>=`: the loose form refuses a
    # perfect corpus, which is a validator rejecting the best possible input.
    payload = _ok()
    payload["n_header_edges"] = 1
    payload["edges"] = [{"superseded": "pep-0216", "successor": "pep-0287"}]
    validate_census(payload)  # exactly 1 restatement, exactly 1 edge — must not raise


def test_well_formed_payload_is_accepted():
    validate_census(_ok())  # must not raise


def test_write_site_calls_the_validator(tmp_path: Path):
    payload = _ok()
    del payload["_provenance"]["peps_sha"]
    with pytest.raises(ValueError, match="peps_sha"):
        write_census(tmp_path / "census.json", payload)
    assert not (tmp_path / "census.json").exists(), "refused payload must not be written"


def test_writer_emits_lf_not_crlf(tmp_path: Path):
    path = tmp_path / "census.json"
    write_census(path, _ok())
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8"))["n_files"] == 733
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_truth_extraction_contract.py -q
```

Expected: collection error, `ModuleNotFoundError: ... artifact_contract`.

- [ ] **Step 3: Write the validator**

Create `benchmarks/labelling/truth_extraction/artifact_contract.py`:

```python
"""Validation for the truth-extraction census artifact, applied at the write site.

The census is the artifact every later recall number is read against. Its counts and its lists
are the same facts written twice — a summary a reader quotes and a body a reader recomputes from.
If they disagree, the artifact is not merely wrong, it is unfalsifiable: nothing in it says which
of the two is the typo. So the disagreement is refused at write time, when it costs nothing.

Pattern follows `benchmarks/artifact_contract.py`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

_REQUIRED_PROVENANCE = ("peps_sha", "clone_date", "recall_commit", "generated_at", "invocation")


def validate_census(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent, attributable census."""
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("census payload requires a _provenance block")
    for field in _REQUIRED_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"census _provenance requires {field}")

    edges = payload.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("census edges must be an array")
    if payload.get("n_header_edges") != len(edges):
        raise ValueError(
            f"n_header_edges {payload.get('n_header_edges')!r} disagrees with "
            f"{len(edges)} entries in edges"
        )

    restatements = payload.get("restatements")
    if not isinstance(restatements, Mapping):
        raise ValueError("census restatements must be an object")
    if payload.get("n_restated_in_prose") != len(restatements):
        raise ValueError(
            f"n_restated_in_prose {payload.get('n_restated_in_prose')!r} disagrees with "
            f"{len(restatements)} entries in restatements"
        )

    # The recall ceiling is a proportion of the header edges. A value above 1.0 means the
    # restatement detector matched something that is not in the gold set at all.
    if len(restatements) > len(edges):
        raise ValueError(
            f"n_restated_in_prose ({len(restatements)}) cannot exceed n_header_edges "
            f"({len(edges)}) — the recall ceiling cannot exceed 100%"
        )


__all__ = ["validate_census"]
```

- [ ] **Step 4: Write the census module**

Create `benchmarks/labelling/truth_extraction/census.py`:

```python
"""Census of supersession evidence in `python/peps`. No model, no human judgement.

This runs before any arm and is arm-independent, so it belongs in the preregistration's
`## Already measured` section rather than among its predictions.

The number it exists to publish is `n_restated_in_prose / n_header_edges`: the fraction of
authored header edges that the body ALSO states in prose. That ratio is the hard ceiling on
recall for any extractor that reads prose, because an edge no sentence states cannot be found by
reading sentences. Measured at `5981b2a`: **8 of 47, or 17.0%.** Publish it, or every recall
number below it reads as a model failure when it is a corpus fact.

`restates` requires the marker and the partner reference in ONE sentence. Whole-body
co-occurrence would report 26 of 47 instead, by pairing a marker in one section with a reference
in another — the loose matching `recall/fix.py` records as producing garbage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from benchmarks.labelling.truth_extraction.artifact_contract import validate_census
from benchmarks.labelling.truth_extraction.peps_header import (
    Edge,
    edges_from_fields,
    header_fields,
    restates,
    split_header,
)
from recall.eval.provenance import generated_at, model_stack
from recall.lint import CLOSURE_MARKERS


@dataclass(frozen=True)
class Census:
    n_files: int
    n_header_edges: int
    n_prose_marker_files: int
    n_marker_without_header: int
    n_restated_in_prose: int
    edges: tuple[Edge, ...]
    restatements: dict[str, str]
    marker_without_header: tuple[str, ...]
    file_digests: dict[str, str]


def compute_census(peps_dir: Path) -> Census:
    """Count everything the labelled set is built from. Reads files; decides nothing."""
    files = sorted(peps_dir.glob("pep-*.rst"))
    if not files:
        raise SystemExit(
            f"no pep-*.rst under {peps_dir} — pass the nested 'peps/' directory of the clone, "
            f"not the repository root"
        )

    bodies: dict[str, str] = {}
    digests: dict[str, str] = {}
    edges: set[Edge] = set()
    marker_files: list[str] = []

    for path in files:
        raw = path.read_bytes()
        digests[path.name] = hashlib.sha256(raw).hexdigest()
        head, body = split_header(raw.decode("utf-8", errors="replace"))
        bodies[path.stem] = body
        edges |= edges_from_fields(path.stem, header_fields(head))
        if CLOSURE_MARKERS.search(body):
            marker_files.append(path.stem)

    # An edge is restated if EITHER end states it. The successor's body saying "replaces PEP 386"
    # is as much a prose statement of the relation as the predecessor's "superseded by PEP 440".
    restatements: dict[str, str] = {}
    for edge in sorted(edges):
        for holder, partner in (
            (edge.superseded, edge.successor),
            (edge.successor, edge.superseded),
        ):
            sentence = restates(bodies.get(holder, ""), partner)
            if sentence:
                restatements[f"{edge.superseded}->{edge.successor}"] = sentence.strip()
                break

    in_an_edge = {e.superseded for e in edges} | {e.successor for e in edges}
    without_header = tuple(sorted(s for s in marker_files if s not in in_an_edge))

    return Census(
        n_files=len(files),
        n_header_edges=len(edges),
        n_prose_marker_files=len(marker_files),
        n_marker_without_header=len(without_header),
        n_restated_in_prose=len(restatements),
        edges=tuple(sorted(edges)),
        restatements=restatements,
        marker_without_header=without_header,
        file_digests=digests,
    )


def census_payload(
    census: Census,
    *,
    peps_sha: str,
    clone_date: str,
    recall_commit: str,
    invocation: str,
) -> dict:
    """The committed artifact: counts, bodies, and the provenance that makes them checkable."""
    ceiling = census.n_restated_in_prose / census.n_header_edges if census.n_header_edges else 0.0
    return {
        "n_files": census.n_files,
        "n_header_edges": census.n_header_edges,
        "n_prose_marker_files": census.n_prose_marker_files,
        "n_marker_without_header": census.n_marker_without_header,
        "n_restated_in_prose": census.n_restated_in_prose,
        "recall_ceiling": round(ceiling, 4),
        "edges": [{"superseded": e.superseded, "successor": e.successor} for e in census.edges],
        "restatements": census.restatements,
        "marker_without_header": list(census.marker_without_header),
        "file_digests": census.file_digests,
        "_provenance": {
            "peps_sha": peps_sha,
            "clone_date": clone_date,
            "recall_commit": recall_commit,
            "generated_at": generated_at(),
            "model_stack": model_stack(),
            "invocation": invocation,
            "note": (
                "Arm-independent: no model and no human judgement. recall_ceiling is the "
                "fraction of authored header edges also stated in prose, and is the hard upper "
                "bound on recall for any prose extractor."
            ),
        },
    }


def write_census(path: Path, payload: Mapping[str, object]) -> None:
    """Validate, then write. A payload that fails validation leaves no file behind."""
    validate_census(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True,
                        help="the nested peps/ directory of a python/peps clone")
    parser.add_argument("--peps-sha", required=True, help="git rev-parse HEAD of that clone")
    parser.add_argument("--clone-date", required=True, help="ISO date the clone was taken")
    parser.add_argument("--out", type=Path, default=Path("results/truth_extraction/census.json"))
    args = parser.parse_args()

    recall_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    census = compute_census(args.peps_dir)
    payload = census_payload(
        census,
        peps_sha=args.peps_sha,
        clone_date=args.clone_date,
        recall_commit=recall_commit,
        invocation=" ".join(["python", "-m", "benchmarks.labelling.truth_extraction.census",
                             *sys.argv[1:]]),
    )
    write_census(args.out, payload)
    print(f"{args.out}")
    print(f"  n_files                 {census.n_files}")
    print(f"  n_header_edges          {census.n_header_edges}")
    print(f"  n_prose_marker_files    {census.n_prose_marker_files}")
    print(f"  n_marker_without_header {census.n_marker_without_header}")
    print(f"  n_restated_in_prose     {census.n_restated_in_prose}"
          f"  <- recall ceiling {payload['recall_ceiling']:.1%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_truth_extraction_contract.py -q
```

Expected: `10 passed`.

- [ ] **Step 6: Mutate the validator and watch each guard go red**

**Every** guard gets mutated, not only the interesting ones. A guard nobody has watched fail is
a guard nobody has tested. Seven mutations, each reverted before the next:

1. Comment out the `n_header_edges` comparison → `test_edge_count_disagreeing_with_edge_list_is_refused` FAILS.
2. Comment out the `restatements` comparison → `test_restated_count_disagreeing_with_restatements_is_refused` FAILS.
3. Comment out the ceiling check → `test_ceiling_above_one_hundred_percent_is_refused` FAILS.
4. Make `write_census` write before validating → `test_write_site_calls_the_validator` FAILS on the `not ... exists()` assertion.
5. Drop the `isinstance(provenance, Mapping)` check (return early instead of raising) → `test_missing_provenance_is_refused` FAILS.
6. Remove `"peps_sha"` from `_REQUIRED_PROVENANCE` → `test_provenance_without_peps_sha_is_refused` FAILS.
7. Remove `"recall_commit"` from `_REQUIRED_PROVENANCE` → `test_provenance_without_recall_commit_is_refused` FAILS.

Then loosen the ceiling comparison from `>` to `>=` and confirm
`test_ceiling_of_exactly_one_hundred_percent_is_accepted` FAILS. This is the over-rejection
guard: `>=` refuses a corpus that restates every edge, which is a validator rejecting its own
best possible input. Revert and confirm `10 passed`.

- [ ] **Step 7: Generate the real census**

```bash
python -m benchmarks.labelling.truth_extraction.census --peps-dir /tmp/peps/peps --peps-sha "$(git -C /tmp/peps rev-parse HEAD)" --clone-date 2026-08-11
```

Expected, at SHA `5981b2a`: `n_files 733`, `n_header_edges 47`, `n_prose_marker_files 209`,
`n_marker_without_header 175`, `n_restated_in_prose 8`, recall ceiling `17.0%`. **If any count
differs, the corpus moved — record the new SHA and update this plan's table rather than editing
the numbers to match.**

- [ ] **Step 8: Lint and commit**

```bash
python -m ruff check benchmarks tests && python -m mypy benchmarks/labelling/truth_extraction
```

```bash
git add benchmarks/labelling/truth_extraction/artifact_contract.py benchmarks/labelling/truth_extraction/census.py tests/test_truth_extraction_contract.py results/truth_extraction/census.json && git commit -m "feat(truth-extraction): census with a 17% prose recall ceiling on 47 header edges"
```

---

### Task 3: Transplanted fixtures — the bridge to the private corpus

**Files:**
- Create: `benchmarks/labelling/truth_extraction/fixtures/reported_speech.md`
- Create: `benchmarks/labelling/truth_extraction/fixtures/hedged.md`
- Create: `benchmarks/labelling/truth_extraction/fixtures/partial_scope_claim.md`
- Create: `benchmarks/labelling/truth_extraction/fixtures/partial_scope_scope.md`
- Test: `tests/test_truth_extraction_fixtures.py`

**Interfaces:**
- Consumes: `recall.fix.extract_edges`, `recall.frontmatter.parse_frontmatter`
- Produces: four `.md` files whose correct label is **negative**, i.e. `extract_edges` must
  return no reference for any of them.

These reproduce the four private-corpus failures quoted verbatim in `fix.py` — reported speech
(`fix.py:134`), hedging (`fix.py:156`), and partial scope twice (`fix.py:170`). Publishing the
error modes is how a private finding becomes publicly checkable without publishing the corpus.

- [ ] **Step 1: Write the fixture files**

`fixtures/reported_speech.md` — the subject of the marker is another document:

```markdown
---
valid_from: 2026-07-17
---
# Trust layer deployment notes

First annotations: LRP closure memo supersedes [[project_lrp_maker_2026-06-24]]

This memo narrates that relation; it does not declare one of its own.
```

`fixtures/hedged.md` — the author declined to commit, and said *augments* when asked:

```markdown
---
valid_from: 2026-06-22
---
# CI green constraints, revisited

Supersedes/augments [[feedback_ci_green_constraints_2026-06-22]]
```

`fixtures/partial_scope_claim.md` — a claim inside the target, not the target:

```markdown
---
valid_from: 2026-07-14
---
# Maker attribution correction

Supersedes the *inferred* "maker" claim in [[curate_wallets_2026-07-14]]
```

`fixtures/partial_scope_scope.md` — the target's scope, not the target:

```markdown
---
valid_from: 2026-07-18
---
# Abstention scope narrowed

Supersedes the scope in [[project_recall_abstention_2026-07-18]]
```

- [ ] **Step 2: Write the test**

Create `tests/test_truth_extraction_fixtures.py`:

```python
"""The four transplanted negatives reproduce fix.py's measured false positives.

Each is a labelled NEGATIVE: a sentence a naive extractor reads as declaring a supersession
edge, which on the private corpus was wrong on human review. They are the publishable half of a
private finding — the error MIX transfers even though the corpus cannot.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.fix import extract_edges
from recall.frontmatter import parse_frontmatter
from recall.lint import CLOSURE_MARKERS

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "labelling" / "truth_extraction" / "fixtures"
)

CASES = [
    ("reported_speech.md", "the marker's subject is another document"),
    ("hedged.md", "the author wrote Supersedes/augments and meant augments"),
    ("partial_scope_claim.md", "supersedes a claim inside the target, not the target"),
    ("partial_scope_scope.md", "supersedes the scope in the target, not the target"),
]


def test_all_four_fixtures_exist():
    assert sorted(p.name for p in FIXTURES.glob("*.md")) == sorted(n for n, _ in CASES)


@pytest.mark.parametrize(("name", "why"), CASES)
def test_fixture_is_a_negative(name: str, why: str):
    _, body = parse_frontmatter((FIXTURES / name).read_text(encoding="utf-8"))
    active, passive = extract_edges(body)
    assert not active and not passive, f"{name} must be refused: {why}"


@pytest.mark.parametrize(("name", "_why"), CASES)
def test_fixture_would_tempt_a_naive_extractor(name: str, _why: str):
    # Guards the guard: a fixture that carried no marker at all would pass the test above for
    # the wrong reason, and the set would silently stop covering its error mode.
    _, body = parse_frontmatter((FIXTURES / name).read_text(encoding="utf-8"))
    assert CLOSURE_MARKERS.search(body), f"{name} carries no closure marker — it tests nothing"
```

- [ ] **Step 3: Run the tests**

```bash
python -m pytest tests/test_truth_extraction_fixtures.py -q
```

Expected: `9 passed`. If a `test_fixture_is_a_negative` case FAILS, that is a real finding about
`fix.py` and must be reported, not fixed by editing the fixture.

- [ ] **Step 4: Mutate and watch red**

Edit `fixtures/hedged.md` to read `Supersedes [[feedback_ci_green_constraints_2026-06-22]]`
(drop the `/augments`). Re-run. Expected: `test_fixture_is_a_negative[hedged.md-...]` FAILS,
proving the fixture tests the hedge and not merely the file's existence. Revert.

Then blank the body of `reported_speech.md` and re-run. Expected:
`test_fixture_would_tempt_a_naive_extractor` FAILS. Revert, confirm `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/labelling/truth_extraction/fixtures tests/test_truth_extraction_fixtures.py && git commit -m "test(truth-extraction): transplant fix.py's four private false positives as public fixtures"
```

---

### Task 4: Freeze the gold manifest

**Files:**
- Create: `benchmarks/labelling/truth_extraction/build_gold.py`
- Create (generated): `benchmarks/labelling/truth_extraction/gold.manifest.jsonl`
- Test: extends `tests/test_truth_extraction_census.py` (created here)

**Interfaces:**
- Consumes: Task 2's `compute_census`; `recall.eval.promotion.manifest.{FrozenQuestion, question_input_hash, write_manifest, read_manifest, file_digest}`
- Produces: `build_gold_questions(census, peps_dir) -> list[FrozenQuestion]`

**Design note, and it is load-bearing.** `FrozenQuestion.input_hash` covers
`(question_id, corpus, query, expected_relevance_labels)`. For an extraction item the *input* is
the source document's prose, so `query` is the **source PEP's body text**. That makes
`input_hash` cover the exact prose an extractor will read, and `FrozenQuestion.verify` then fails
loudly on the item that drifted if upstream edits the PEP. The body itself is never stored:
`question_to_dict` writes only the id, corpus, hash and labels.

`expected_relevance_labels` holds the successor's **file name** (`pep-0287.rst`), matching
`PepsAdapter.label_kind = "source"`. The four transplanted fixtures are frozen with `()` labels.
`manifest.py:71-76` warns that `()` means *unanswerable* and moves an item into the safety set —
which is exactly right here: a negative genuinely has no edge, and must never be scored as a
retrieval miss.

- [ ] **Step 1: Write the failing test**

Create `tests/test_truth_extraction_census.py`:

```python
"""The committed artifacts agree with each other and with the census that produced them.

Properties:
  1. The gold manifest's positive row count equals the census `n_header_edges`.
  2. The manifest verifies against its own digest (read_manifest refuses a mismatch).
  3. Every positive carries exactly one successor label; every fixture negative carries none.
  3b. Every positive's label is the SUCCESSOR named in its own question_id, not the superseded
      document. Without this the suite cannot detect a wholesale direction inversion.
  4. Corpus-dependent recomputation runs only when RECALL_PEPS_DIR is set, and SKIPS loudly
     otherwise rather than passing vacuously.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from recall.eval.promotion.manifest import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
CENSUS = REPO_ROOT / "results" / "truth_extraction" / "census.json"
GOLD = REPO_ROOT / "benchmarks" / "labelling" / "truth_extraction" / "gold.manifest.jsonl"


@pytest.fixture(scope="module")
def census() -> dict:
    return json.loads(CENSUS.read_text(encoding="utf-8"))


def test_manifest_verifies_against_its_digest():
    # read_manifest recomputes and refuses a mismatch; reaching this line means it matched.
    questions, header = read_manifest(GOLD)
    assert header["digest"]
    assert questions


def test_positive_row_count_equals_census_header_edges(census: dict):
    questions, _ = read_manifest(GOLD)
    positives = [q for q in questions if q.expected_relevance_labels]
    assert len(positives) == census["n_header_edges"]


def test_every_positive_has_exactly_one_successor_label():
    questions, _ = read_manifest(GOLD)
    for question in questions:
        if question.expected_relevance_labels:
            assert len(question.expected_relevance_labels) == 1
            assert question.expected_relevance_labels[0].endswith(".rst")


def test_every_positive_label_is_the_SUCCESSOR_not_the_superseded():
    # `question_id` is "<superseded>-><successor>". Labelling a positive with the SUPERSEDED PEP
    # would make the gold set assert that the live document is the stale one — the inversion the
    # trust layer exists to prevent, baked into the labels every later number is scored against.
    #
    # This test exists because without it the suite cannot detect that inversion: a
    # `build_gold_questions` that swapped the two ends would still emit 47 positives, each with
    # exactly one `.rst` label, and pass every other test unchanged. Counting rows and checking a
    # suffix says nothing about direction.
    questions, _ = read_manifest(GOLD)
    positives = [q for q in questions if q.expected_relevance_labels]
    assert positives, "no positives in the manifest — this test would pass vacuously"
    for question in positives:
        superseded, _, successor = question.question_id.partition("->")
        assert successor, f"{question.question_id} is not an '<a>-><b>' identity"
        assert question.expected_relevance_labels[0] == f"{successor}.rst"


def test_fixture_negatives_are_frozen_with_no_labels():
    questions, _ = read_manifest(GOLD)
    negatives = [q for q in questions if not q.expected_relevance_labels]
    assert len(negatives) == 4
    assert all(q.corpus == "fix-transplant" for q in negatives)


def test_recall_ceiling_is_published_and_below_one(census: dict):
    # The number this set exists to publish. If it ever reads 1.0, the detector is matching
    # something that is not in the gold set.
    assert 0.0 < census["recall_ceiling"] < 1.0
    assert census["n_restated_in_prose"] <= census["n_header_edges"]


def test_census_recomputes_from_the_corpus(census: dict):
    peps_dir = os.environ.get("RECALL_PEPS_DIR")
    if not peps_dir:
        pytest.skip(
            "RECALL_PEPS_DIR unset — clone python/peps and point it at the nested peps/ dir. "
            "This test is SKIPPED, not passed: the corpus-dependent counts are unverified."
        )
    from benchmarks.labelling.truth_extraction.census import compute_census

    recomputed = compute_census(Path(peps_dir))
    assert recomputed.n_files == census["n_files"]
    assert recomputed.n_header_edges == census["n_header_edges"]
    assert recomputed.n_prose_marker_files == census["n_prose_marker_files"]
    assert recomputed.n_marker_without_header == census["n_marker_without_header"]
    assert recomputed.n_restated_in_prose == census["n_restated_in_prose"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_truth_extraction_census.py -q
```

Expected: FAIL — `gold.manifest.jsonl` does not exist yet.

- [ ] **Step 3: Write the builder**

Create `benchmarks/labelling/truth_extraction/build_gold.py`:

```python
"""Freeze the gold positives and the transplanted negatives into one manifest.

The positives cost no labelling: every one is an edge a PEP author declared in a machine-readable
header, years ago, with no knowledge of this project. That is what removes labelling bias from
the recall denominator — the denominator was not chosen by anyone measuring against it.

Freezing reuses `recall/eval/promotion/manifest.py` unchanged. Its guarantee is the one this set
needs: a label edited after seeing an extractor's results changes `input_hash`, which changes the
body, which changes the digest, so `read_manifest` refuses the file rather than repairing it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.labelling.truth_extraction.census import compute_census
from benchmarks.labelling.truth_extraction.peps_header import split_header
from recall.eval.promotion.manifest import (
    FrozenQuestion,
    file_digest,
    question_input_hash,
    write_manifest,
)

CORPUS = "peps"
#: The four transplanted private failures live in their own corpus namespace: they are not PEPs,
#: and an aggregate that mixed them into the PEPs arm would report a precision over two corpora.
FIXTURE_CORPUS = "fix-transplant"
#: Anchored to this module, not the cwd — the freeze must produce the same manifest wherever
#: it is invoked from.
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_gold_questions(peps_dir: Path) -> list[FrozenQuestion]:
    """One FrozenQuestion per header edge, plus one per transplanted negative."""
    census = compute_census(peps_dir)
    questions: list[FrozenQuestion] = []

    for edge in census.edges:
        source = peps_dir / f"{edge.superseded}.rst"
        # The extractor's INPUT is the superseded PEP's prose, so that is what the hash covers.
        # The body is hashed, never stored: question_to_dict writes only id, corpus, hash, labels.
        _, body = _split(source)
        labels = (f"{edge.successor}.rst",)
        questions.append(FrozenQuestion(
            question_id=f"{edge.superseded}->{edge.successor}",
            corpus=CORPUS,
            input_hash=question_input_hash(
                question_id=f"{edge.superseded}->{edge.successor}",
                corpus=CORPUS,
                query=body,
                expected_relevance_labels=labels,
            ),
            expected_relevance_labels=labels,
        ))

    for path in sorted(FIXTURES.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        # Empty labels: manifest.py:71-76 reads () as UNANSWERABLE and scores it on the safety
        # axis. For a labelled negative that is correct — there is no edge to hit, and scoring it
        # as a retrieval miss would charge a true refusal as a failure.
        questions.append(FrozenQuestion(
            question_id=path.stem,
            corpus=FIXTURE_CORPUS,
            input_hash=question_input_hash(
                question_id=path.stem,
                corpus=FIXTURE_CORPUS,
                query=body,
                expected_relevance_labels=(),
            ),
            expected_relevance_labels=(),
        ))
    return questions


def _split(path: Path) -> tuple[str, str]:
    return split_header(path.read_text(encoding="utf-8", errors="replace"))


#: Repo root, from this module's location. Every path below is anchored to it, so the freeze
#: produces the same manifest regardless of the directory it is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    # Taken explicitly, exactly as `census.py` takes it, and NOT derived by running
    # `git -C <peps_dir>/.. rev-parse HEAD`. Git discovers repositories upwards: if the clone is
    # ever laid out differently from the assumed nested `peps/peps`, that command succeeds
    # against some enclosing repository and records an unrelated commit as the corpus
    # provenance. A frozen artifact that silently names the wrong corpus is the precise failure
    # the freeze exists to prevent, and it would be undetectable after the fact.
    parser.add_argument("--peps-sha", required=True, help="git rev-parse HEAD of that clone")
    parser.add_argument(
        "--out", type=Path,
        default=_REPO_ROOT / "benchmarks" / "labelling" / "truth_extraction" / "gold.manifest.jsonl",
    )
    args = parser.parse_args()

    questions = build_gold_questions(args.peps_dir)
    corpus_hashes = {
        "peps_sha": args.peps_sha,
        "census": file_digest(_REPO_ROOT / "results" / "truth_extraction" / "census.json"),
    }
    digest = write_manifest(args.out, questions, corpus_hashes=corpus_hashes)
    positives = sum(1 for q in questions if q.expected_relevance_labels)
    print(f"{args.out}\n  {positives} positives + {len(questions) - positives} negatives")
    print(f"  digest {digest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Freeze the manifest**

```bash
python -m benchmarks.labelling.truth_extraction.build_gold --peps-dir /tmp/peps/peps --peps-sha "$(git -C /tmp/peps rev-parse HEAD)"
```

Expected: `47 positives + 4 negatives`, and a digest line. The SHA must equal the one already
recorded in `results/truth_extraction/census.json` — if it does not, the corpus moved between the
two freezes and both must be regenerated together, not reconciled by hand.

- [ ] **Step 5: Run tests to verify they pass**

```bash
RECALL_PEPS_DIR=/tmp/peps/peps python -m pytest tests/test_truth_extraction_census.py -q
```

Expected: `7 passed`. Then run **without** the env var and confirm `6 passed, 1 skipped` — a
corpus-dependent check that passes with no corpus is a guard that cannot fail.

- [ ] **Step 6: Mutate the freeze and watch red**

Append a byte to `gold.manifest.jsonl` (e.g. duplicate its final line), then run
`python -m pytest tests/test_truth_extraction_census.py -q`. Expected:
`test_manifest_verifies_against_its_digest` FAILS with `read_manifest`'s "has been edited since
it was frozen" message. Restore by re-running Step 4.

- [ ] **Step 7: Lint and commit**

```bash
python -m ruff check benchmarks tests && python -m mypy benchmarks/labelling/truth_extraction
```

```bash
git add benchmarks/labelling/truth_extraction/build_gold.py benchmarks/labelling/truth_extraction/gold.manifest.jsonl tests/test_truth_extraction_census.py && git commit -m "feat(truth-extraction): freeze 47 gold positives and 4 transplanted negatives"
```

---

### Task 5: Blind adjudication pack

**Files:**
- Create: `benchmarks/labelling/truth_extraction/build_adjudication.py`
- Create (generated): `benchmarks/labelling/truth_extraction/adjudication.csv`
- Create (generated): `benchmarks/labelling/truth_extraction/adjudication_key.json`
- Test: extends `tests/test_truth_extraction_contract.py`

**Interfaces:**
- Consumes: Task 2's `compute_census`; Task 1's `sentences`, `pep_refs`, `CLOSURE_MARKERS`
- Produces: `build_rows(census, peps_dir, seed, limit) -> tuple[list[dict], dict]`

The 175 `marker_without_header` PEPs are where `fix.py`'s four false positives lived. The
adjudicator sees the evidence sentence and the candidate target, never which rule or model
surfaced it. Shape follows `build_beam_labelling.py`: fixed-seed shuffle, mapping to a separate
key file, `_csv_safe` against spreadsheet formula injection, and blank meaning *undecidable* per
`score_beam_labels.py:29` rather than *no*.

**The full pool is adjudicated: no cap.** `--limit` exists and is applied *after* the shuffle, so
that a capped subset would stay a uniform sample rather than a prefix, but it is left unset. The
pool is 175 files and **38 rows**: only 30 files name a candidate target in the marker's
sentence. The other 145 carry a marker with no target named, and `fix.py`'s rule is that an
unprovable target is reported for a human, never guessed — there is no candidate pair to put in
front of an adjudicator, so they are counted in the census and excluded here.

- [ ] **Step 1: Add the blinding properties to the contract test**

First add `import csv` and `import os` to the file's existing import block, and extend its module
docstring property list with:

```
 10. The blind CSV exposes no arm / model / judge / score / rule / system column.
 11. The un-blinding key is a separate file, and every CSV item has an entry in it.
 12. The verdict column ships blank — blank is "undecidable", per score_beam_labels.py:29.
 13. Every row names a candidate target; the unprovable-target class is excluded, not guessed.
 14. The committed row count recomputes from the corpus (skips loudly without RECALL_PEPS_DIR).
```

Then append to `tests/test_truth_extraction_contract.py`:

```python
_TE = Path(__file__).resolve().parents[1] / "benchmarks" / "labelling" / "truth_extraction"
CSV_PATH = _TE / "adjudication.csv"
KEY_PATH = _TE / "adjudication_key.json"


def _csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_blind_csv_leaks_no_arm_model_or_judge_column():
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    leaky = [c for c in header
             if any(t in c.lower() for t in ("arm", "model", "judge", "score", "rule", "system"))]
    assert not leaky, f"blind CSV exposes {leaky} — the adjudicator would see what produced the row"


def test_the_key_is_a_separate_file_from_the_csv():
    assert KEY_PATH.exists() and KEY_PATH != CSV_PATH
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    assert key, "key file is empty — nothing could be un-blinded after labelling"


def test_every_csv_item_has_a_key_entry():
    items = {row["item"] for row in _csv_rows()}
    assert items == set(json.loads(KEY_PATH.read_text(encoding="utf-8")))


def test_verdict_column_ships_blank():
    rows = _csv_rows()
    assert rows and all(r["your_verdict_Y_or_N"] == "" for r in rows)


def test_every_row_names_a_candidate_target():
    # The pool is 175 FILES but only the 30 that name a target in the marker's sentence are
    # adjudicable. A row with an empty target would be asking a human to guess at an unprovable
    # one, which is the class fix.py refuses to guess at rather than the class it adjudicates.
    assert all(r["candidate_target"].startswith("pep-") for r in _csv_rows())


def test_row_count_recomputes_from_the_corpus():
    peps_dir = os.environ.get("RECALL_PEPS_DIR")
    if not peps_dir:
        pytest.skip(
            "RECALL_PEPS_DIR unset — the committed row count is UNVERIFIED against the corpus. "
            "Clone python/peps and point it at the nested peps/ dir."
        )
    from benchmarks.labelling.truth_extraction.build_adjudication import build_rows

    rebuilt, _ = build_rows(Path(peps_dir), seed=0, limit=None)
    assert len(rebuilt) == len(_csv_rows())
```

- [ ] **Step 2: Run and watch it fail**

```bash
python -m pytest tests/test_truth_extraction_contract.py -q
```

Expected: 6 FAILs / errors — the CSV does not exist.

- [ ] **Step 3: Write the builder**

Create `benchmarks/labelling/truth_extraction/build_adjudication.py`:

```python
"""Build a BLIND adjudication set over prose markers that no header confirms.

The 175 PEPs carrying a closure marker with no corresponding header edge are the PEPs analogue of
the 60-versus-2 gap on the private corpus, and they are where `fix.py`'s four measured false
positives lived. A negative label here is a human judgement, so it is made blind: the adjudicator
sees the evidence sentence and the candidate target and nothing about what surfaced them.

Blank is data. `score_beam_labels.read_verdict` reads an empty cell as *undecidable* and EXCLUDES
it, rather than counting it against whichever arm happened to be labelled. An adjudicator who
cannot tell should leave the cell empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from benchmarks.labelling.truth_extraction.census import compute_census
from benchmarks.labelling.truth_extraction.peps_header import pep_refs, sentences, split_header
from recall.lint import CLOSURE_MARKERS

#: Characters a spreadsheet executes as a formula rather than displaying. Same defence as
#: `build_beam_labelling._csv_safe`: these cells are third-party text, not author-written.
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: str) -> str:
    return "'" + value if value and value[0] in _FORMULA_LEAD else value


def build_rows(
    peps_dir: Path, *, seed: int, limit: int | None
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    census = compute_census(peps_dir)
    candidates: list[dict[str, str]] = []

    for stem in census.marker_without_header:
        _, body = split_header((peps_dir / f"{stem}.rst").read_text(
            encoding="utf-8", errors="replace"))
        for sentence in sentences(body):
            if not CLOSURE_MARKERS.search(sentence):
                continue
            refs = sorted(pep_refs(sentence) - {stem})
            # No target named in the sentence is not a negative — it is unprovable, and
            # `fix.py` reports that class rather than guessing at it. Excluded from adjudication.
            for target in refs:
                candidates.append({
                    "source_pep": stem,
                    "candidate_target": target,
                    "evidence_sentence": sentence.strip(),
                })

    random.Random(seed).shuffle(candidates)
    if limit:
        candidates = candidates[:limit]

    rows: list[dict[str, str]] = []
    key: dict[str, dict[str, str]] = {}
    for i, cand in enumerate(candidates, 1):
        key[str(i)] = dict(cand)
        rows.append({
            "item": str(i),
            "evidence_sentence": cand["evidence_sentence"],
            "candidate_target": cand["candidate_target"],
            "your_verdict_Y_or_N": "",
        })
    return rows, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap items; applied AFTER the shuffle so the subset stays uniform")
    parser.add_argument(
        "--out", type=Path,
        default=Path("benchmarks/labelling/truth_extraction/adjudication"),
    )
    args = parser.parse_args()

    rows, key = build_rows(args.peps_dir, seed=args.seed, limit=args.limit)
    if not rows:
        raise SystemExit("no candidates selected")

    csv_path = args.out.with_suffix(".csv")
    key_path = args.out.parent / (args.out.name + "_key.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is required by the csv module (it does its own line-ending handling), and
    # lineterminator="\n" then stops it emitting CRLF on Windows. `.gitattributes` normalises to
    # LF on commit either way — `judge_labelling.csv` is committed at 0 CRLF despite
    # `build_beam_labelling.py` writing the default — but a working-tree file whose bytes depend
    # on the OS that wrote it is the thing the freeze discipline exists to prevent.
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item", "evidence_sentence", "candidate_target", "your_verdict_Y_or_N"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows([{k: _csv_safe(v) for k, v in row.items()} for row in rows])
    with key_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(key, indent=1, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"{len(rows)} items\n  {csv_path}\n  {key_path}   <- do NOT open until labelling is done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the pack — the whole pool, no cap**

```bash
python -m benchmarks.labelling.truth_extraction.build_adjudication --peps-dir /tmp/peps/peps --seed 0
```

Expected: `38 items`. If it prints 175, the builder is emitting one row per file rather than one
per `(source, sentence, target)` candidate, and 145 of them have no target to adjudicate. If it
prints 54, it is using the pre-fix sentence splitter that glues across paragraph boundaries.

- [ ] **Step 5: Run tests to verify they pass**

```bash
RECALL_PEPS_DIR=/tmp/peps/peps python -m pytest tests/test_truth_extraction_contract.py -q
```

Expected: `16 passed`. Then run without the env var and confirm `15 passed, 1 skipped` — the
row-count recomputation must skip loudly, not pass vacuously.

- [ ] **Step 6: Mutate the blinding and watch red**

Three mutations, each reverted and regenerated after it is watched go red.

1. Add `"judge_score"` to the CSV `fieldnames` and emit it on every row. Regenerate, re-run.
   Expected: `test_blind_csv_leaks_no_arm_model_or_judge_column` FAILS naming `judge_score`.
2. Write the key's contents into the CSV as a `source_pep` column instead of to the key file.
   Expected: `test_every_csv_item_has_a_key_entry` FAILS.
3. Pre-fill `your_verdict_Y_or_N` with `"Y"`. Expected: `test_verdict_column_ships_blank` FAILS.

Regenerate with the committed builder and confirm `16 passed`.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/labelling/truth_extraction/build_adjudication.py benchmarks/labelling/truth_extraction/adjudication.csv benchmarks/labelling/truth_extraction/adjudication_key.json tests/test_truth_extraction_contract.py && git commit -m "feat(truth-extraction): blind adjudication pack over 175 unconfirmed prose markers"
```

---

### Task 6: Trust query set

**Files:**
- Create: `benchmarks/labelling/truth_extraction/build_trust_queries.py`
- Create (generated): `recall/eval/peps_trust_queries.json`
- Test: extends `tests/test_truth_extraction_census.py`

**Interfaces:**
- Consumes: Task 2's `compute_census`; Task 1's `header_fields`, `split_header`
- Produces: `build_queries(peps_dir, *, n_abstain, seed) -> list[dict]`

**Two decisions worth stating.** First, this is a **new file**, not an edit to
`recall/eval/queries.json`: that file's ids reference the synthetic memo corpus at
`recall/eval/corpus`, and appending PEP rows would produce a query set no single corpus can
serve. It uses the same schema, which is what the spec asks for.

Second, query text is the superseded PEP's `Title:` header, **mechanically**, never hand
authored. All 733 PEPs carry a Title, so this needs no judgement and no exceptions. It is a
weaker query than a real user question, and the artifact must say so.

Abstain rows come from PEPs with `Status:` Withdrawn or Rejected and no successor — 188 are
available, sampled with a fixed seed. 47 successor rows + 20 abstain = **67 queries**, inside the
40–70 target, against a shipped set of 6 whose successor arm is n=4.

- [ ] **Step 1: Add the test**

Append to `tests/test_truth_extraction_census.py`:

```python
TRUST = REPO_ROOT / "recall" / "eval" / "peps_trust_queries.json"


def test_trust_set_is_between_40_and_70_queries():
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    assert 40 <= len(rows) <= 70, f"{len(rows)} queries — Wilson needs the shipped n=4 fixed"


def test_trust_set_matches_the_shipped_queries_schema():
    shipped = json.loads((REPO_ROOT / "recall" / "eval" / "queries.json").read_text(
        encoding="utf-8"))
    shipped_trust_keys = {k for e in shipped if e.get("trust") for k in e}
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    for row in rows:
        assert set(row) == shipped_trust_keys, f"{row['id']} does not match the shipped schema"


def test_successor_rows_have_a_successor_and_abstain_rows_do_not():
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    for row in rows:
        if row["expect"] == "successor":
            assert row["successor_ids"] and row["stale_ids"]
        else:
            assert row["expect"] == "abstain" and row["successor_ids"] == []


def test_successor_row_count_equals_census_header_edges(census: dict):
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    successors = [r for r in rows if r["expect"] == "successor"]
    assert len(successors) == census["n_header_edges"]


def test_stale_and_successor_are_not_inverted(census: dict):
    # `stale_ids` must hold the SUPERSEDED document and `successor_ids` the SUCCESSOR. Swapping
    # them scores a system correct exactly when it prefers the stale document — the failure the
    # trust layer exists to prevent, written into the labels every later number is graded on.
    #
    # This test exists because the suite was measured blind to it: an inverted builder still
    # emits 67 rows, 47 of them `successor`, with matching schema keys and non-empty
    # `successor_ids`, and passes every other test in this file. Counting rows and checking
    # shape says nothing about direction.
    #
    # The comparison is against `census.json`'s edge list, which is independently frozen and
    # whose own direction was verified against the PEP headers.
    edges = {(e["superseded"], e["successor"]) for e in census["edges"]}
    rows = json.loads(TRUST.read_text(encoding="utf-8"))

    def stem(chunk_id: str) -> str:
        name = chunk_id.rsplit(":", 1)[0]
        return name[:-4] if name.endswith(".rst") else name

    seen = set()
    for row in (r for r in rows if r["expect"] == "successor"):
        pair = (stem(row["stale_ids"][0]), stem(row["successor_ids"][0]))
        assert pair in edges, (
            f"{row['id']}: {pair[0]} -> {pair[1]} is not a census edge. "
            f"Reversed pair present in census: {(pair[1], pair[0]) in edges}"
        )
        seen.add(pair)
    assert seen == edges, "successor rows do not cover the census edges exactly"
```

- [ ] **Step 2: Run and watch it fail**

```bash
python -m pytest tests/test_truth_extraction_census.py -q
```

Expected: 4 FAILs — `peps_trust_queries.json` does not exist.

- [ ] **Step 3: Write the builder**

Create `benchmarks/labelling/truth_extraction/build_trust_queries.py`:

```python
"""The downstream trust query set: does retrieval prefer the successor over the stale document?

Each `(superseded, successor)` header edge is a natural `(stale_ids, successor_ids)` row. The
shipped set in `recall/eval/queries.json` has 6 trust rows, of which only 4 expect a successor,
and a Wilson interval on n=4 is uninterpretable.

A NEW file rather than an edit to `queries.json`: that file's ids address the synthetic memo
corpus under `recall/eval/corpus`, so appending PEP rows would build a query set no single corpus
can serve. The schema is identical.

Query text is the superseded PEP's `Title:`, taken mechanically. All 733 carry one, so this needs
no judgement and has no exceptions. It is a weaker probe than a real user question — a title is
what the document is called, not what someone would ask — and the artifact says so.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from benchmarks.labelling.truth_extraction.census import compute_census
from benchmarks.labelling.truth_extraction.peps_header import header_fields, split_header


def _fields(peps_dir: Path, stem: str) -> dict[str, str]:
    head, _ = split_header((peps_dir / f"{stem}.rst").read_text(
        encoding="utf-8", errors="replace"))
    return header_fields(head)


def build_queries(peps_dir: Path, *, n_abstain: int, seed: int) -> list[dict]:
    census = compute_census(peps_dir)
    rows: list[dict] = []

    for i, edge in enumerate(census.edges, 1):
        title = _fields(peps_dir, edge.superseded).get("Title", "")
        rows.append({
            "id": f"pt{i:02d}",
            "query": title.lower(),
            "trust": True,
            "expect": "successor",
            "stale_ids": [f"{edge.superseded}.rst:0"],
            "successor_ids": [f"{edge.successor}.rst:0"],
        })

    in_an_edge = {e.superseded for e in census.edges} | {e.successor for e in census.edges}
    abstain_pool = []
    for path in sorted(peps_dir.glob("pep-*.rst")):
        if path.stem in in_an_edge:
            continue
        fields = _fields(peps_dir, path.stem)
        if fields.get("Status") in {"Withdrawn", "Rejected"} and not fields.get("Superseded-By"):
            abstain_pool.append((path.stem, fields.get("Title", "")))

    random.Random(seed).shuffle(abstain_pool)
    for i, (stem, title) in enumerate(abstain_pool[:n_abstain], len(rows) + 1):
        rows.append({
            "id": f"pt{i:02d}",
            "query": title.lower(),
            "trust": True,
            "expect": "abstain",
            "stale_ids": [f"{stem}.rst:0"],
            "successor_ids": [],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    parser.add_argument("--n-abstain", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("recall/eval/peps_trust_queries.json"))
    args = parser.parse_args()

    rows = build_queries(args.peps_dir, n_abstain=args.n_abstain, seed=args.seed)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    successors = sum(1 for r in rows if r["expect"] == "successor")
    print(f"{args.out}\n  {len(rows)} queries ({successors} successor / "
          f"{len(rows) - successors} abstain)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate and test**

```bash
python -m benchmarks.labelling.truth_extraction.build_trust_queries --peps-dir /tmp/peps/peps --n-abstain 20
```

Expected: `67 queries (47 successor / 20 abstain)`.

```bash
RECALL_PEPS_DIR=/tmp/peps/peps python -m pytest tests/test_truth_extraction_census.py -q
```

Expected: `12 passed` (7 from Task 4 plus 5 here).

- [ ] **Step 5: Mutate and watch red**

Regenerate with `--n-abstain 40` (yielding 87 rows) and re-run. Expected:
`test_trust_set_is_between_40_and_70_queries` FAILS. Regenerate with `20`.

Then the direction mutation, which is the one that matters. Swap `stale_ids` and `successor_ids`
in the builder's successor-row block, regenerate to a SCRATCH path, and point the tests at it.
Expected: `test_stale_and_successor_are_not_inverted` FAILS on the first row, reporting
`Reversed pair present in census: True`, while
`test_successor_row_count_equals_census_header_edges` and the schema and shape tests all still
PASS. That contrast is the evidence: it was measured that the suite without this test passes an
inverted set unchanged. Restore and confirm the committed file is byte-identical.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/labelling/truth_extraction/build_trust_queries.py recall/eval/peps_trust_queries.json tests/test_truth_extraction_census.py && git commit -m "feat(truth-extraction): 67-query PEPs trust set, replacing an n=4 successor arm"
```

---

### Task 7: Publish — ARTIFACTS.md and the preregistration

**Files:**
- Modify: `results/ARTIFACTS.md`
- Modify: `benchmarks/archive/preregistrations/PREREGISTRATION-peps-rerank-pool.md`

- [ ] **Step 1: Add the ARTIFACTS.md entry**

Append a section under `## The artifacts`, matching the existing table style:

```markdown
### Truth-extraction labelled set

| artifact | backs |
|---|---|
| `results/truth_extraction/census.json` | the 17.0% prose recall ceiling; counts recomputable from `python/peps` at the recorded SHA |
| `benchmarks/labelling/truth_extraction/gold.manifest.jsonl` | 47 gold positives (authored PEP headers) + 4 transplanted negatives, frozen |
| `benchmarks/labelling/truth_extraction/adjudication.csv` | the blind negative-adjudication pack, 38 rows from the 30 of 175 marker-without-header PEPs that name a target; `adjudication_key.json` un-blinds it and must not be opened until labelling is finished |
| `recall/eval/peps_trust_queries.json` | 67 trust queries (47 successor / 20 abstain), replacing a shipped successor arm of n=4 |

**What this set measures well, and what it does not.** Of 47 authored header edges, only **8** are
restated in prose with the marker and the partner PEP in the same sentence. A perfect prose
extractor therefore scores recall **8/47 = 0.170** against the header denominator, and the usable
positive class for recall is **8, not 47** — an n on which a Wilson interval is about as
uninterpretable as the n=4 this set's trust arm was built to fix. Precision is the axis this set
measures with power, from the 38 adjudicable candidate pairs drawn from those markers.

**PEPs are not memos.** They cite each other as `PEP 3106`, not `[[wikilink]]`, and they are
written under an editorial process a personal memo corpus does not have. A precision measured
here does not transfer to a memo corpus. What transfers is the **error mix**, which the four
transplanted fixtures in `benchmarks/labelling/truth_extraction/fixtures/` make checkable: they
reproduce, verbatim, the reported speech, hedging and two partial-scope failures measured on the
private 792-memo corpus and quoted in `recall/fix.py`.
```

- [ ] **Step 2: Add the census to the preregistration's `## Already measured`**

The census is arm-independent — no model, no human judgement — so it lands before any prediction
is frozen. Add to `benchmarks/archive/preregistrations/PREREGISTRATION-peps-rerank-pool.md`:

```markdown
## Already measured (arm-independent, frozen before predictions)

Census of supersession evidence over the same 733 `.rst`, at `python/peps` SHA
`5981b2a292610104eb30735423504c52fe454650`. No model, no human judgement, so it constrains no arm
and is not a prediction: `results/truth_extraction/census.json`.

| quantity | value |
|---|---|
| `n_files` | 733 |
| `n_header_edges` | 47 |
| `n_prose_marker_files` | 209 |
| `n_marker_without_header` | 175 |
| `n_restated_in_prose` | 8 |
| **recall ceiling** | **17.0%** |
```

- [ ] **Step 3: Full verification**

```bash
python -m ruff check . && python -m mypy benchmarks/labelling/truth_extraction recall
```

```bash
RECALL_PEPS_DIR=/tmp/peps/peps python -m pytest tests/test_truth_extraction_peps_header.py tests/test_truth_extraction_contract.py tests/test_truth_extraction_census.py tests/test_truth_extraction_fixtures.py -q
```

Expected: `52 passed` (15 peps_header + 16 contract + 12 census + 9 fixtures). Then run the full
suite in the background — it takes ~12 minutes here, not
the ~3 the pyproject claims:

```bash
python -m pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add results/ARTIFACTS.md benchmarks/archive/preregistrations/PREREGISTRATION-peps-rerank-pool.md && git commit -m "docs(truth-extraction): publish the census, the 17% ceiling, and the memo-transfer caveat"
```

---

## Self-review against the spec

| spec requirement | where |
|---|---|
| Use PEPs, do not hand author memos | Tasks 1–2, 6. Only the 4 transplanted fixtures are authored, and they are quoted from `fix.py`. |
| Census: `n_files`, `n_header_edges`, `n_prose_marker_files`, `n_marker_without_header`, `n_restated_in_prose` | Task 2, `Census` dataclass and `census_payload` |
| Census lands in the prereg's `## Already measured` | Task 7 Step 2 |
| Positives free from headers, "55 to 95" expected | Task 4. **Measured 47** — recorded, not adjusted. |
| Negatives blindly adjudicated, reuse existing machinery | Task 5, following `build_beam_labelling.py`'s shuffle + separate key |
| Blank is data | Task 5 docstring; `score_beam_labels.read_verdict` unchanged |
| Four transplanted fixtures | Task 3 |
| Freeze with `manifest.py` unchanged, `newline="\n"` | Task 4; every writer opens with `newline="\n"` |
| Trust set 40–70 queries in `queries.json`'s schema | Task 6, 67 queries |
| `_provenance` with `generated_at()`/`model_stack()`, PEPs SHA, clone date, RE-call commit, per-file digests, counts, exact invocation | Task 2, `census_payload` |
| Listed in `results/ARTIFACTS.md` in the same change | Task 7 |
| State the PEPs-versus-memos caveat | Task 7 Step 1 |
| Done when: census recomputed from committed files, digest verified, CSV has no arm/model/judge column, key is separate | Task 4 test + Task 5 test |

**Known gap, deliberate.** The "recompute the census from the committed files" test splits in
two: cross-checks between committed artifacts (manifest row count vs `n_header_edges`, trust
successor count vs `n_header_edges`) always run, while the counts that need the 733 `.rst` skip
loudly when `RECALL_PEPS_DIR` is unset. Vendoring 733 files to make them always-run was rejected;
a test that passes with no corpus would be a guard that cannot fail.
