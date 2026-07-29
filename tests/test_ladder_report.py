"""The report prints the verdict it computed — including FAIL.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

A report that can only render a PASS is not a report. `test_a_flat_curve_prints_fail...` is the
test that matters here: it exercises the DETECTION path, not the green path.

Task 11 AMENDMENT coverage: `d=max` ingests an EMPTY corpus (RING_MAX excises the whole cluster,
and ingest scope is one conversation), so the pre-registered `d=0` vs `d=max` contrast can PASS on
a system that only ever abstains when its index is empty. The tests below pin the qualifications
this report must print alongside the pre-registered verdict: `n` on every comparison, per-rung
surviving-document counts, every adjacent pairwise contrast plus the widest non-empty one, the
machine-derived qualification line, and the bge-small disclosure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.ladder.build import build_instances
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    RING_ORIGINAL,
    Instance,
    write_manifest,
)
from benchmarks.ladder.report import main
from benchmarks.ladder.rings import RingSpec
from benchmarks.ladder.sources.locomo import load_locomo

# A conversation with enough turns that widths 0/4/16 all differ from each other and from d=max,
# so the "adjacent pairwise contrast" and "widest non-empty contrast" logic has real rungs to walk
# instead of degenerating to a single comparison. 20 turns, one gold turn per question.
_TURNS = [
    {"dia_id": f"D1:{n}", "speaker": "Caroline" if n % 2 else "Melanie", "text": f"turn {n}"}
    for n in range(1, 21)
]
WIDE_CONVO = [
    {
        "sample_id": "conv-0",
        "conversation": {"session_1_date_time": "7 May 2023", "session_1": _TURNS},
        "qa": [
            {
                "question": f"question {n}",
                "answer": "x",
                "evidence": [f"D1:{n}"],
                "category": 2,
            }
            for n in range(1, 21)
        ],
    }
]


def _write_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(WIDE_CONVO), encoding="utf-8")
    return path


def _setup(tmp_path: Path, *, flat: bool) -> tuple[Path, Path]:
    """The brief's original fixture: 40 synthetic paired questions at d=0 and d=max only."""
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


def _write_responses(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _write_manifest_from_instances(tmp_path: Path, name: str, instances) -> Path:
    path = tmp_path / name
    write_manifest(path, instances, ring_widths=[0, 4, 16, 64], corpus_hashes={"locomo": "x"})
    return path


# ---------------------------------------------------------------------------
# Baseline (Task 11 brief, Step 1)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Amendment: n on every comparison
# ---------------------------------------------------------------------------


def test_the_headline_contrast_prints_its_n(tmp_path: Path, capsys):
    manifest, responses = _setup(tmp_path, flat=False)
    main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "H1 (pre-registered)" in out
    # the headline line itself carries n= — a 2-of-300 overlap must be visible right where the
    # verdict is printed, not only in a table above it.
    h1_lines = [line for line in out.splitlines() if line.startswith("H1 (pre-registered)")]
    assert h1_lines and "n=40" in h1_lines[0]


# ---------------------------------------------------------------------------
# Amendment: surviving-document counts per rung, from the manifest + load_locomo
# ---------------------------------------------------------------------------


def test_surviving_document_counts_are_printed_when_a_corpus_is_given(tmp_path: Path, capsys):
    corpus_path = _write_corpus(tmp_path)
    corpus = load_locomo(corpus_path)
    spec = RingSpec(widths=(0, 4, 16))
    instances = build_instances(corpus, spec, corpus_name="locomo")
    manifest = _write_manifest_from_instances(tmp_path, "manifest.jsonl", instances)
    rows = [
        {"instance_id": i.instance_id, "system": "recall",
         "abstained": (i.ring == RING_MAX), "cited_ids": [], "tokens": 0}
        for i in instances
    ]
    responses = _write_responses(tmp_path, "responses.jsonl", rows)

    main(["--manifest", str(manifest), "--responses", str(responses), "--corpus", str(corpus_path)])
    out = capsys.readouterr().out

    # cluster has 20 turns; the answerable original excises nothing -> 20 surviving docs. Use the
    # FIRST line starting with each label — that is the ring table row; later sections (the
    # supplementary pairwise contrasts) reuse the same labels inside comparison rows like
    # "original vs d=0 ... FAIL", which must not shadow the table row in this lookup.
    lines: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        key = line.split()[0]
        lines.setdefault(key, line)
    assert "original" in lines
    assert lines["original"].split()[-1] == "20"
    # RING_MAX excises the whole cluster -> the confound, visible in the table itself.
    assert lines["d=max"].split()[-1] == "0"


def test_surviving_document_counts_are_skipped_without_a_corpus_argument(tmp_path: Path, capsys):
    manifest, responses = _setup(tmp_path, flat=False)
    main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "surv" not in out.lower() or "SKIPPED" in out


# ---------------------------------------------------------------------------
# Amendment: every adjacent pairwise contrast, catching ValueError per-pair
# ---------------------------------------------------------------------------


def test_a_pair_with_no_overlap_prints_n_equals_zero_instead_of_aborting(tmp_path: Path, capsys):
    """Two rungs can each have data while sharing no pair_id — the per-pair guard must catch
    that ValueError and keep printing the rest of the report, not blow up the whole run."""
    instances = []
    rows = []
    # 20 pairs answer only at d=0, 20 different pairs answer only at d=4 -> d=0 vs d=4 shares none.
    for i in range(20):
        iid = f"p{i}#d0"
        instances.append(
            Instance(instance_id=iid, corpus="locomo", source_question_id=f"q{i}",
                     question="q", label=LABEL_UNANSWERABLE, ring=0,
                     excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id=f"p{i}")
        )
        rows.append({"instance_id": iid, "system": "recall", "abstained": False,
                     "cited_ids": [], "tokens": 0})
    for i in range(20, 40):
        iid = f"p{i}#d4"
        instances.append(
            Instance(instance_id=iid, corpus="locomo", source_question_id=f"q{i}",
                     question="q", label=LABEL_UNANSWERABLE, ring=4,
                     excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id=f"p{i}")
        )
        rows.append({"instance_id": iid, "system": "recall", "abstained": True,
                     "cited_ids": [], "tokens": 0})
    # RING_MAX pairs, shared with the d=0 pairs, so the headline contrast still has data.
    for i in range(20):
        iid = f"p{i}#dmax"
        instances.append(
            Instance(instance_id=iid, corpus="locomo", source_question_id=f"q{i}",
                     question="q", label=LABEL_UNANSWERABLE, ring=RING_MAX,
                     excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id=f"p{i}")
        )
        rows.append({"instance_id": iid, "system": "recall", "abstained": True,
                     "cited_ids": [], "tokens": 0})

    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, instances, ring_widths=[0, 4], corpus_hashes={})
    responses = _write_responses(tmp_path, "responses.jsonl", rows)

    rc = main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "n=0 (no paired data)" in out
    # the report kept going and still printed the headline verdict, not just the failed pair.
    assert "H1:" in out


def test_the_headline_contrast_with_no_shared_data_propagates_instead_of_printing(tmp_path: Path):
    """The headline (d=0 vs d=max) is the pre-registered verdict; per the brief, a verdict with no
    data must not be printable — so ValueError from the headline contrast is NOT caught."""
    instances = [
        Instance(instance_id="p1#d0", corpus="locomo", source_question_id="q1",
                 question="q", label=LABEL_UNANSWERABLE, ring=0,
                 excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id="p1"),
        Instance(instance_id="p2#dmax", corpus="locomo", source_question_id="q2",
                 question="q", label=LABEL_UNANSWERABLE, ring=RING_MAX,
                 excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id="p2"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, instances, ring_widths=[0], corpus_hashes={})
    rows = [
        {"instance_id": "p1#d0", "system": "recall", "abstained": False, "cited_ids": [], "tokens": 0},
        {"instance_id": "p2#dmax", "system": "recall", "abstained": True, "cited_ids": [], "tokens": 0},
    ]
    responses = _write_responses(tmp_path, "responses.jsonl", rows)

    with pytest.raises(ValueError, match="no question appears at BOTH"):
        main(["--manifest", str(manifest), "--responses", str(responses)])


# ---------------------------------------------------------------------------
# Amendment: the machine-checked qualification line
# ---------------------------------------------------------------------------

_QUALIFICATION_LINE = 'the axis as built prices "is anything indexed at all", not answerability'


def _confound_setup(tmp_path: Path) -> tuple[Path, Path]:
    """40 paired questions at d=0, d=64 and d=max.

    Abstains ONLY at d=max, never at d=0 or d=64: `d=0 vs d=max` separates (the pre-registered
    PASS), but `d=0 vs d=64` — the widest contrast whose corpus is not empty — stays perfectly
    flat. That is the confound this qualification line exists to name.
    """
    instances = []
    rows = []
    for i in range(40):
        for ring in (0, 64, RING_MAX):
            iid = f"p{i}#d{ring}"
            instances.append(
                Instance(
                    instance_id=iid, corpus="locomo", source_question_id=f"q{i}",
                    question="q", label=LABEL_UNANSWERABLE, ring=ring,
                    excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id=f"p{i}",
                )
            )
            abstained = ring == RING_MAX
            rows.append({"instance_id": iid, "system": "recall", "abstained": abstained,
                         "cited_ids": [], "tokens": 0})
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, instances, ring_widths=[0, 64], corpus_hashes={})
    responses = _write_responses(tmp_path, "responses.jsonl", rows)
    return manifest, responses


def test_the_qualification_line_appears_when_the_headline_passes_but_the_widest_contrast_is_flat(
    tmp_path: Path, capsys
):
    manifest, responses = _confound_setup(tmp_path)
    rc = main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "H1: PASS" in out
    assert _QUALIFICATION_LINE in out
    assert rc == 0


def test_the_qualification_line_is_absent_when_the_widest_contrast_also_separates(
    tmp_path: Path, capsys
):
    """Same shape, but this time abstention ALSO separates at d=64 -> no confound to report."""
    instances = []
    rows = []
    for i in range(40):
        for ring in (0, 64, RING_MAX):
            iid = f"p{i}#d{ring}"
            instances.append(
                Instance(
                    instance_id=iid, corpus="locomo", source_question_id=f"q{i}",
                    question="q", label=LABEL_UNANSWERABLE, ring=ring,
                    excised_doc_ids=("g",), gold_doc_ids=("g",), pair_id=f"p{i}",
                )
            )
            abstained = ring in (64, RING_MAX)
            rows.append({"instance_id": iid, "system": "recall", "abstained": abstained,
                         "cited_ids": [], "tokens": 0})
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, instances, ring_widths=[0, 64], corpus_hashes={})
    responses = _write_responses(tmp_path, "responses.jsonl", rows)

    main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "H1: PASS" in out
    assert _QUALIFICATION_LINE not in out


def test_the_qualification_line_is_absent_when_the_headline_itself_fails(tmp_path: Path, capsys):
    manifest, responses = _setup(tmp_path, flat=True)
    main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "H1: FAIL" in out
    assert _QUALIFICATION_LINE not in out


def test_the_qualification_line_is_absent_when_there_is_no_widest_non_empty_rung(
    tmp_path: Path, capsys
):
    """The brief's own baseline fixture only has d=0 and d=max — nothing wider than d=0 to
    contrast against, so the line must not fire even though the headline PASSes."""
    manifest, responses = _setup(tmp_path, flat=False)
    main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "H1: PASS" in out
    assert _QUALIFICATION_LINE not in out


# ---------------------------------------------------------------------------
# Amendment: the bge-small disclosure
# ---------------------------------------------------------------------------


def test_the_report_discloses_the_uncalibrated_bge_small_floor(tmp_path: Path, capsys):
    manifest, responses = _setup(tmp_path, flat=False)
    main(["--manifest", str(manifest), "--responses", str(responses)])
    out = capsys.readouterr().out
    assert "bge-small" in out
    assert "0.50" in out
    assert "calibrat" in out.lower()
