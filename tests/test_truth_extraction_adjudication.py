"""The blind adjudication pack, and the properties that make its labels usable as evidence.

38 candidate `(sentence, target)` pairs drawn from the 175 PEPs whose prose closes a decision that
no `Superseded-By:` / `Replaces:` header confirms. They are the precision instrument for
`results/truth_extraction/PREREGISTRATION-prose-extraction.md`: the 47 gold edge questions can only
measure recall, and that is capped at 3 by how PEP authors write.

Blinding is the whole value. The CSV shows the sentence and the candidate target and withholds
which PEP the sentence came from, so the adjudicator judges the LANGUAGE rather than what they
already know about a given PEP. `adjudication_key.json` holds the mapping and stays a separate
file. If the two were ever merged, the labels would silently become a measurement of the
adjudicator's familiarity with the corpus.

Follows `tests/test_beam_blind_labelling.py`, which pins the same properties for the BEAM pack.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.labelling.truth_extraction.build_adjudication import (
    _FORMULA_LEAD,
    _csv_safe,
)

PACK = Path(__file__).resolve().parent.parent / "benchmarks" / "labelling" / "truth_extraction"
CSV_PATH = PACK / "adjudication.csv"
KEY_PATH = PACK / "adjudication_key.json"

VERDICTS = {"Y", "N"}
#: The pre-registration's decision rule calls anything under 10 predictions UNDERPOWERED. The
#: labelled positives bound what any arm can score against, so a pack that fell below this could
#: not support a precision claim at all.
MIN_POSITIVES = 5


def _rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _key() -> dict[str, dict[str, str]]:
    return json.loads(KEY_PATH.read_text(encoding="utf-8"))


def test_every_verdict_is_yes_no_or_deliberately_blank() -> None:
    """Blank is a third value, not a missing one.

    An undecidable row is excluded from the denominator rather than guessed at, so a stray `?`,
    `maybe`, or a typo'd `y ` must fail here rather than be silently read as one of the two.
    """
    bad = [
        (row["item"], repr(row["your_verdict_Y_or_N"]))
        for row in _rows()
        if row["your_verdict_Y_or_N"].strip() and row["your_verdict_Y_or_N"].strip() not in VERDICTS
    ]
    assert not bad, f"verdicts must be Y, N, or empty: {bad}"


def test_no_verdict_hides_behind_whitespace() -> None:
    """`\" Y\"` and `\"Y\"` must not be two different things to whoever scores this."""
    padded = [
        (row["item"], repr(row["your_verdict_Y_or_N"]))
        for row in _rows()
        if row["your_verdict_Y_or_N"] != row["your_verdict_Y_or_N"].strip()
    ]
    assert not padded, f"verdicts carry stray whitespace: {padded}"


def test_the_blank_count_is_visible_rather_than_silently_dropped() -> None:
    """How many rows were undecidable is itself a finding, so it must be countable from the file.

    A pack where every row is filled is the suspicious one: it suggests guessing to complete the
    sheet, which is the bias the blank exists to avoid. This asserts the count is derivable, not
    that it is any particular number.
    """
    rows = _rows()
    blanks = [row["item"] for row in rows if not row["your_verdict_Y_or_N"].strip()]
    decided = len(rows) - len(blanks)
    assert decided + len(blanks) == len(rows), "every row is either decided or blank"
    assert decided > 0, "a pack with no decided rows cannot measure precision"


def test_the_positive_class_can_support_a_precision_claim() -> None:
    """Guards against a pack that is technically labelled and statistically empty."""
    positives = [r for r in _rows() if r["your_verdict_Y_or_N"].strip() == "Y"]
    assert len(positives) >= MIN_POSITIVES, (
        f"only {len(positives)} positives; below {MIN_POSITIVES} no arm's precision on this pack "
        f"could clear the pre-registration's underpowered clause"
    )


def test_the_csv_withholds_which_pep_each_sentence_came_from() -> None:
    """The blinding itself, asserted against the key rather than by reading column names.

    Checking only the header would pass on a pack that leaked the source PEP inside a cell.
    """
    assert "source_pep" not in _rows()[0], "the CSV must not carry the source column"
    key = _key()
    leaked = [
        row["item"]
        for row in _rows()
        if key[row["item"]]["source_pep"] in ",".join(row.values())
    ]
    assert not leaked, f"rows leak their source PEP into the blind sheet: {leaked}"


def test_the_csv_names_no_arm_model_or_judge() -> None:
    """A label conditioned on which system proposed the row measures the adjudicator's priors."""
    header = set(_rows()[0])
    forbidden = {"arm", "model", "judge", "system", "proposer", "provider", "verdict_model"}
    assert not (header & forbidden), f"blind sheet exposes {sorted(header & forbidden)}"


def test_the_key_carries_the_mapping_the_csv_withholds() -> None:
    key = _key()
    rows = _rows()
    assert set(key) == {row["item"] for row in rows}, "the key must cover exactly the CSV's items"
    for row in rows:
        entry = key[row["item"]]
        # `_csv_safe`, not a raw compare. The CSV carries the injection-defended form and the key
        # carries the raw sentence, so they are SUPPOSED to differ on any cell opening with a
        # formula character. A raw compare here would read that defence as corruption — it did,
        # and the "repair" stripped the apostrophe off four cells before this was caught.
        assert _csv_safe(entry["evidence_sentence"]) == row["evidence_sentence"], row["item"]
        assert _csv_safe(entry["candidate_target"]) == row["candidate_target"], row["item"]
        assert entry["source_pep"], f"item {row['item']} has no source PEP in the key"


def test_the_key_is_a_separate_file() -> None:
    """Two files, so un-blinding is an act rather than an accident."""
    assert CSV_PATH.is_file() and KEY_PATH.is_file()
    assert CSV_PATH != KEY_PATH


@pytest.mark.parametrize("column", ["item", "evidence_sentence", "candidate_target"])
def test_labelling_did_not_edit_the_evidence(column: str) -> None:
    """The sentence and target are the question. Editing them changes what was adjudicated.

    Compared against the key, which is the independent record of both, so this fails whether the
    CSV drifted or was rebuilt from a different sample.
    """
    key = _key()
    for row in _rows():
        if column == "item":
            assert row["item"] in key
        else:
            want = _csv_safe(key[row["item"]][column])
            assert row[column] == want, f"item {row['item']}: {column} changed"


def test_the_formula_injection_defence_survives_a_round_trip_through_a_spreadsheet() -> None:
    """The guard a spreadsheet silently removes, and that a raw key-compare mistakes for damage.

    `build_adjudication._csv_safe` prefixes `'` to any cell opening with `= + - @`, tab or CR,
    because these are third-party PEP sentences and a spreadsheet would execute them as formulas.
    A spreadsheet reads that apostrophe as its own text-force marker and drops it on save, so a
    labelling round trip strips the defence from exactly the cells that need it.

    This is not hypothetical: it happened on this pack, four cells, and the first attempt to
    reconcile the file with the key removed the defence rather than restoring it.
    """
    rows = _rows()
    key = _key()
    # Imported rather than restated: a hand-copied lead-character set here could drift from the
    # builder's, and the drift would be invisible — this test would keep passing while the cells
    # the builder actually defends changed underneath it.
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
