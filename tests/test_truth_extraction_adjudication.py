"""The blind adjudication pack, and the properties that make its labels usable as evidence.

38 candidate `(sentence, target)` pairs drawn from the 175 PEPs whose prose closes a decision that
no `Superseded-By:` / `Replaces:` header confirms. They are the precision instrument for the
pre-registration: the 47 gold edge questions can only measure recall, and that is capped at 3 by
how PEP authors write.

⚠️ The pre-registration lives on `claude/truth-extraction-prereg`, not on this branch, so
`results/truth_extraction/PREREGISTRATION-prose-extraction.md` does not resolve here. Read it with
`git show claude/truth-extraction-prereg:results/truth_extraction/PREREGISTRATION-prose-extraction.md`.
That gap is how a wrong citation of its decision rule survived review once already.

Blinding is the whole value. The CSV shows the sentence and the candidate target and withholds
which PEP the sentence came from, so the adjudicator judges the LANGUAGE rather than what they
already know about a given PEP. `adjudication_key.json` holds the mapping and stays a separate
file. If the two were ever merged, the labels would silently become a measurement of the
adjudicator's familiarity with the corpus.

The two files hold the same sentences in DIFFERENT forms, deliberately: the CSV carries the
`_csv_safe` injection-defended form, the key carries the raw sentence. Comparing them raw reads
the defence as damage — it did, and a "repair" stripped the apostrophe off four cells before
anything caught it. Every comparison here goes through `_csv_safe`, and
`test_the_key_stores_raw_sentences` guards the mirror image, which is repairing the KEY to match
the CSV.

Follows `tests/test_beam_blind_labelling.py`, which pins the same properties for the BEAM pack.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from benchmarks.labelling.truth_extraction.build_adjudication import _FORMULA_LEAD, _csv_safe

PACK = Path(__file__).resolve().parent.parent / "benchmarks" / "labelling" / "truth_extraction"
CSV_PATH = PACK / "adjudication.csv"
KEY_PATH = PACK / "adjudication_key.json"

VERDICTS = {"Y", "N"}

#: Named rather than written inline. A shell heredoc ate this escape three times in one session,
#: writing the decoded characters into the source where a bytes literal cannot hold them.
BOM = b"\xef\xbb\xbf"

#: The adjudication, frozen: item, the SENTENCE it judged, its target, and the verdict.
#:
#: A tally cannot tell a reversed column from the real one. An item-keyed digest of the verdicts
#: alone cannot tell a swap of the EVIDENCE from the real one either: exchange two rows' sentences
#: in both the CSV and the key, leave the verdict column untouched, and every label now answers
#: the wrong question while the digest is unchanged. Mutation confirmed that passed. Binding the
#: sentence in is what makes this pin the adjudication rather than just its right-hand column.
ANSWERED_DIGEST = "f2b5d6969b6515b146a459a98bf8ae1efbaa5157caf50c257ab632d0569d27a3"

#: The row the adjudicator could not decide. Named, because the argument for keeping a blank is
#: that THIS candidate is undecidable, not that some unspecified one is.
UNDECIDABLE_ITEM = "7"

#: P1 and P2 predict 0.80 precision. Precision is `TP / proposals` and `TP <= min(P, n)`, so with
#: fewer than 8 positives the primary prediction is unreachable at the smallest non-underpowered
#: proposal count (10), and the experiment could only ever falsify, never confirm.
#:
#: NOT the pre-registration's underpowered clause, which reads "fewer than 10 PROPOSALS in either
#: arm, or Wilson half width above 0.30". Positives cannot move either term: an arm proposing on
#: 10 rows of a pack with zero positives scores 0.00 at n=10, which is decisive rather than
#: underpowered. An earlier version of this constant cited that clause and was wrong about it.
MIN_POSITIVES = 8


def _raw() -> bytes:
    return CSV_PATH.read_bytes()


def _rows() -> list[dict[str, str]]:
    # `utf-8-sig`: a spreadsheet's "CSV UTF-8" writes a BOM, which lands on the first header field
    # and turns every `row["item"]` into a KeyError three tests deep. The BOM is still REPORTED,
    # by the byte guard below, rather than silently absorbed here.
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _key() -> dict[str, dict[str, str]]:
    return json.loads(KEY_PATH.read_text(encoding="utf-8-sig"))


def test_the_file_is_shaped_before_anything_reads_it_as_a_pack() -> None:
    """A mangled file must fail with a sentence, not a KeyError raised inside another test.

    A spreadsheet round trip damages this file in two ways that used to surface as raw
    `KeyError: 'item'`, `AttributeError: NoneType has no strip`, and `TypeError: sequence item 3`
    across five tests: a BOM on the header, and a dropped trailing field on the blank row.
    """
    raw = _raw()
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM means a spreadsheet rewrote this file"
    assert b"\r" not in raw, "CRLF: this repo is eol=lf and the builder writes lineterminator=lf"
    assert raw.endswith(b"\n"), "no trailing newline"

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        parsed = list(csv.reader(handle))
    assert parsed[0] == ["item", "evidence_sentence", "candidate_target", "your_verdict_Y_or_N"]
    for line, row in enumerate(parsed[1:], start=2):
        assert len(row) == 4, f"line {line} has {len(row)} fields, not 4"

    # The key is edited by the same tools and was not covered: a BOM there raised
    # JSONDecodeError from inside four unrelated tests, none of which named the cause.
    key_raw = KEY_PATH.read_bytes()
    assert not key_raw.startswith(BOM), "the key file carries a BOM"
    assert key_raw.endswith(b"\n"), "the key file has no trailing newline"

    # Bytes were guarded and STRUCTURE was not, which left the same hole one level in: a key
    # rewritten as a list, or an entry missing a field, still parsed as JSON and then surfaced as
    # `TypeError: list indices must be integers` or a bare `KeyError` from inside whichever test
    # happened to subscript it first. The three keys are named here because every consumer below
    # subscripts all three.
    key = _key()
    assert isinstance(key, dict), f"the key must be an object keyed by item, not {type(key).__name__}"
    malformed = {
        item: sorted(entry) if isinstance(entry, dict) else type(entry).__name__
        for item, entry in key.items()
        if not isinstance(entry, dict)
        or set(entry) != {"source_pep", "candidate_target", "evidence_sentence"}
    }
    assert not malformed, f"key entries are not (source_pep, candidate_target, evidence_sentence): {malformed}"


def test_the_committed_verdicts_are_frozen() -> None:
    """The labels themselves, pinned item by item.

    Before this existed the suite asserted only shape: reversing the entire verdict column,
    swapping two verdicts, or filling in the deliberate blank all left it green, while the commit
    message claimed the verdicts were "pinned separately". They were not pinned by anything.
    """
    rows = _rows()
    # Row ORDER is asserted separately, below, and the digest is taken over items sorted
    # numerically. A spreadsheet "sort by sentence" is a routine thing to do to a file handed to
    # a human, and it changes no label: it must fail with its own message rather than with one
    # telling the reader to re-adjudicate.
    ordered = sorted(rows, key=lambda r: int(r["item"]))
    joined = "".join(
        f"{r['item']}="
        f"{hashlib.sha256(r['evidence_sentence'].encode('utf-8')).hexdigest()[:12]}:"
        f"{r['candidate_target']}="
        f"{r['your_verdict_Y_or_N'].strip()};"
        for r in ordered
    )
    assert hashlib.sha256(joined.encode("utf-8")).hexdigest() == ANSWERED_DIGEST, (
        "the adjudication changed: a verdict, or the sentence a verdict answers. If that is "
        "deliberate, re-adjudicate and update the digest in the same commit; do not edit in place"
    )


def test_the_rows_are_in_item_order() -> None:
    """Separate from the digest, so a reorder reports itself instead of reading as a relabel."""
    items = [r["item"] for r in _rows()]
    assert items == [str(i) for i in range(1, len(items) + 1)], "rows are not in item order"


def test_every_verdict_is_yes_no_or_deliberately_blank() -> None:
    """Blank is a third value, not a missing one."""
    bad = [
        (row["item"], repr(row["your_verdict_Y_or_N"]))
        for row in _rows()
        if row["your_verdict_Y_or_N"].strip() and row["your_verdict_Y_or_N"].strip() not in VERDICTS
    ]
    assert not bad, f"verdicts must be Y, N, or empty: {bad}"


def test_no_verdict_hides_behind_whitespace() -> None:
    padded = [
        (row["item"], repr(row["your_verdict_Y_or_N"]))
        for row in _rows()
        if row["your_verdict_Y_or_N"] != row["your_verdict_Y_or_N"].strip()
    ]
    assert not padded, f"verdicts carry stray whitespace: {padded}"


def test_the_undecidable_row_stayed_undecidable() -> None:
    """Both directions, because the docstring's claim needs a guard in each.

    A pack with 38 of 38 filled suggests guessing to complete the sheet, which is the bias the
    blank exists to avoid. An earlier version asserted `decided + blanks == len(rows)`, which is a
    tautology: `decided` was defined as `len(rows) - len(blanks)` on the line above.
    """
    rows = _rows()
    blanks = [r["item"] for r in rows if not r["your_verdict_Y_or_N"].strip()]
    assert blanks, "38 of 38 filled: a guess to complete the sheet is the bias blank exists for"
    assert blanks == [UNDECIDABLE_ITEM], f"a different row went blank: {blanks}"
    assert len(rows) - len(blanks) > 0, "a pack with no decided rows cannot measure precision"


def test_the_positive_class_can_support_the_primary_prediction() -> None:
    positives = [r for r in _rows() if r["your_verdict_Y_or_N"].strip() == "Y"]
    assert len(positives) >= MIN_POSITIVES, (
        f"only {len(positives)} positives; below {MIN_POSITIVES} the 0.80 precision P1 and P2 "
        f"predict is unreachable at the smallest non-underpowered proposal count"
    )


def test_the_csv_withholds_which_pep_each_sentence_came_from() -> None:
    """The blinding, asserted against the key rather than by reading column names."""
    assert "source_pep" not in _rows()[0], "the CSV must not carry the source column"
    key = _key()
    leaked = [
        row["item"]
        for row in _rows()
        if key[row["item"]]["source_pep"] in ",".join(row.values())
    ]
    assert not leaked, f"rows leak their source PEP into the blind sheet: {leaked}"


def test_the_csv_names_no_arm_model_or_judge() -> None:
    """Substring tokens, not exact names, so `arm_id` fails this too."""
    header = [column.lower() for column in _rows()[0]]
    leaky = [
        column
        for column in header
        if any(token in column for token in ("arm", "model", "judge", "score", "system"))
    ]
    assert not leaky, f"blind sheet exposes {leaky}"


def test_the_key_carries_the_mapping_the_csv_withholds() -> None:
    key = _key()
    rows = _rows()
    assert set(key) == {row["item"] for row in rows}, "the key must cover exactly the CSV's items"
    for row in rows:
        entry = key[row["item"]]
        # `_csv_safe`, not a raw compare. The CSV carries the injection-defended form and the key
        # carries the raw sentence, so they are SUPPOSED to differ on any cell opening with a
        # formula character. A raw compare reads that defence as corruption — it did.
        assert _csv_safe(entry["evidence_sentence"]) == row["evidence_sentence"], row["item"]
        assert _csv_safe(entry["candidate_target"]) == row["candidate_target"], row["item"]
        # Shape only, and the message says so. A well-formed but WRONG stem still passes:
        # verifying attribution needs the corpus, and the only corpus-gated test here skips
        # without RECALL_PEPS_DIR. Claiming more than this checks is how the previous version
        # read as protection it did not provide.
        assert re.fullmatch(r"pep-\d{4}", entry["source_pep"]), (
            f"item {row['item']}: source_pep {entry['source_pep']!r} is not a well-formed PEP "
            f"stem. This does NOT verify it is the right one"
        )


def test_the_key_is_a_separate_file() -> None:
    assert CSV_PATH.is_file() and KEY_PATH.is_file()
    assert CSV_PATH != KEY_PATH


def test_the_key_stores_raw_sentences() -> None:
    """The mirror of the repair that already went wrong once.

    The defence guard below derives its defended set FROM the key, so prefixing an apostrophe onto
    a key entry — "repairing" the key to match the CSV instead of the CSV to match the key — drops
    that item out of the set and is invisible to every other test. Mutation confirmed a
    single-cell key corruption passed the whole suite.
    """
    offenders = [
        item
        for item, entry in _key().items()
        if entry["evidence_sentence"].startswith("'") or entry["candidate_target"].startswith("'")
    ]
    assert not offenders, (
        f"items {offenders} carry a leading apostrophe in the KEY. The key stores the RAW "
        f"sentence; an apostrophe here means a repair went the wrong way"
    )


def test_the_formula_injection_defence_survives_a_round_trip_through_a_spreadsheet() -> None:
    """The guard a spreadsheet silently removes, and that a raw key-compare mistakes for damage.

    `_csv_safe` prefixes `'` to any cell opening with `= + - @`, tab or CR, because these are
    third-party PEP sentences and a spreadsheet would execute them as formulas. A spreadsheet
    reads that apostrophe as its own text-force marker and DROPS it on save, so a labelling round
    trip strips the defence from exactly the cells that need it. It happened on this pack, four
    cells, and the first repair removed the defence rather than restoring it.
    """
    rows = _rows()
    key = _key()
    defended = [
        row["item"]
        for row in rows
        if key[row["item"]]["evidence_sentence"][:1] in _FORMULA_LEAD
    ]
    assert defended, "no cell needs the defence; this test would be vacuous on this pack"
    for item in defended:
        cell = next(r["evidence_sentence"] for r in rows if r["item"] == item)
        assert cell.startswith("'"), (
            f"item {item} lost its formula-injection prefix, which is what a spreadsheet "
            f"round trip does to it"
        )


@pytest.mark.parametrize("column", ["item", "evidence_sentence", "candidate_target"])
def test_labelling_did_not_edit_the_evidence(column: str) -> None:
    """The sentence and target are the question. Editing them changes what was adjudicated."""
    key = _key()
    for row in _rows():
        if column == "item":
            assert row["item"] in key
        else:
            assert row[column] == _csv_safe(key[row["item"]][column]), (
                f"item {row['item']}: {column} changed"
            )
