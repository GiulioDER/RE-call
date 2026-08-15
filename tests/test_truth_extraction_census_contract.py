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
  7. A ceiling of EXACTLY 100% is accepted. A corpus that restates every edge is legitimate, and
     refusing it would be a validator rejecting its own best possible input.
  8. A well-formed payload is NOT rejected.
  9. The write site calls the validator.
  10. The writer emits LF regardless of platform.
  11. The blind CSV exposes no arm / model / judge / score / rule / system column.
  12. The un-blinding key is a separate file, and every CSV item has an entry in it.
  13. The BUILDER emits a blank verdict column, so a rebuild never carries a previous round's
      verdicts forward. Asserted by running it, not by grepping its source. Blank means
      "undecidable" and is excluded from the denominator, per
      `benchmarks/labelling/score_beam_labels.read_verdict` (named, not line-numbered: a line
      number in prose goes stale on the next edit, and this one already had). The committed
      pack's own labels are pinned in `tests/test_truth_extraction_adjudication.py`.
  13b. The builder applies `_csv_safe` when it WRITES, which nothing covered.
  14. Every row names a candidate target; the unprovable-target class is excluded, not guessed.
  15. The committed row count recomputes from the corpus (skips loudly without RECALL_PEPS_DIR).
  16. The PEPs SHA format validator accepts a real 40-hex SHA and refuses malformed input.
"""
from __future__ import annotations

import csv
import json
import os
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
    # perfect corpus, which is a validator rejecting its own best possible input.
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


def test_validate_peps_sha_format_accepts_a_real_sha():
    from benchmarks.labelling.truth_extraction.census import _validate_peps_sha_format

    _validate_peps_sha_format("5981b2a292610104eb30735423504c52fe454650")  # must not raise


def test_validate_peps_sha_format_rejects_a_malformed_sha():
    from benchmarks.labelling.truth_extraction.census import _validate_peps_sha_format

    with pytest.raises(ValueError, match="40-character"):
        _validate_peps_sha_format("not-a-sha")


_TE = Path(__file__).resolve().parents[1] / "benchmarks" / "labelling" / "truth_extraction"
CSV_PATH = _TE / "adjudication.csv"
KEY_PATH = _TE / "adjudication_key.json"


#: ⚠️ `utf-8-sig` on BOTH files, here as well as in `tests/test_truth_extraction_adjudication.py`.
#: "CSV UTF-8" writes a BOM, and the same tools edit the key. The named byte guard that REPORTS a
#: BOM lives in the other file; these readers only have to survive one, so that a BOM is
#: diagnosed once rather than raising `KeyError: 'item'` and `JSONDecodeError` from inside
#: whichever test happened to open the file first. Moving one reader and not the other left
#: exactly that hole open, one file over, on the round that set out to close it.
def _csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_blind_csv_leaks_no_arm_model_or_judge_column():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    leaky = [
        column
        for column in header
        if any(token in column.lower() for token in ("arm", "model", "judge", "score", "rule", "system"))
    ]
    assert not leaky, f"blind CSV exposes {leaky} — the adjudicator would see what produced the row"


def test_the_key_is_a_separate_file_from_the_csv():
    assert KEY_PATH.exists() and KEY_PATH != CSV_PATH
    key = json.loads(KEY_PATH.read_text(encoding="utf-8-sig"))
    assert key, "key file is empty — nothing could be un-blinded after labelling"


def test_every_csv_item_has_a_key_entry():
    items = {row["item"] for row in _csv_rows()}
    assert items == set(json.loads(KEY_PATH.read_text(encoding="utf-8-sig")))


def test_the_builder_emits_a_blank_verdict_column(tmp_path):
    """Asserted on the BUILDER, not on the committed pack, because the pack is now labelled.

    This used to read `all(row["your_verdict_Y_or_N"] == "")` over the committed CSV, which was
    right while the sheet was waiting for an adjudicator and became false the moment one filled it
    in. Deleting it would have lost the property that still matters: a REBUILD must hand the next
    adjudicator an empty column, never carry a previous round's verdicts forward, which is how a
    labelling exercise quietly turns into a measurement of the last labelling exercise.

    Asserted by RUNNING the builder, not by grepping its source. The first version of this test
    matched the literal `"your_verdict_Y_or_N": ""` in `build_adjudication.py`, which was both
    too weak and too brittle: a builder rewritten to carry a previous round's verdicts forward
    passed as long as the literal survived in a comment, while three benign reformattings of a
    CORRECT builder (single quotes, a named constant, a line break before the value) turned it
    red. `build_rows` runs on a two-file synthetic corpus in about a second, so there is no
    reason to assert on bytes.

    The committed pack's own verdicts are pinned separately, in
    `tests/test_truth_extraction_adjudication.py::test_the_committed_verdicts_are_frozen`.
    """
    from benchmarks.labelling.truth_extraction.build_adjudication import build_rows

    corpus = tmp_path / "peps"
    corpus.mkdir()
    (corpus / "pep-0001.rst").write_text(
        "PEP: 1\nTitle: One\nStatus: Active\n\n\nThis PEP is superseded by :pep:`2`.\n",
        encoding="utf-8",
    )
    (corpus / "pep-0002.rst").write_text(
        "PEP: 2\nTitle: Two\nStatus: Active\n\n\nNothing to see.\n", encoding="utf-8"
    )

    rows, key = build_rows(corpus, seed=0, limit=None)

    assert rows, "the synthetic corpus must yield a candidate, or this test proves nothing"
    assert all(row["your_verdict_Y_or_N"] == "" for row in rows), (
        "a rebuild must hand the next adjudicator an empty column"
    )
    assert set(key) == {row["item"] for row in rows}


def test_the_builder_applies_the_injection_defence_when_it_writes(tmp_path):
    """The defence at BUILD time, asserted DOWNSTREAM of the code that applies it.

    `_csv_safe` is applied where the pack is serialised, not in `build_rows`, so removing it left
    the whole suite green while the pack shipped raw `+`-leading cells for a spreadsheet to
    execute.

    The first attempt to close that gap reimplemented the `DictWriter` block in this test's own
    body and therefore asserted on its own copy: deleting `_csv_safe` from the real writer still
    passed. A guard that reimplements what it guards is testing itself. The block now lives in
    `write_pack`, which `main` calls and this test calls, so there is one definition and the
    assertion sits downstream of it.
    """
    import csv as _csv

    from benchmarks.labelling.truth_extraction.build_adjudication import build_rows, write_pack

    rows, key = build_rows(_formula_corpus(tmp_path), seed=0, limit=None)
    assert rows, "the fixture must yield a candidate, or this test proves nothing"

    csv_path, key_path = write_pack(rows, key, tmp_path / "pack")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        parsed = list(_csv.DictReader(handle))
    defended = [r for r in parsed if r["evidence_sentence"].startswith("'")]
    assert defended, (
        "a sentence opening with a formula character must reach the CSV with its apostrophe"
    )
    # The key is the un-blinding record and must keep the RAW sentence: the two files hold the
    # same text in different forms on purpose, and a defence applied to both would make the
    # CSV/key comparison unable to see a spreadsheet stripping it.
    stored = json.loads(key_path.read_text(encoding="utf-8"))
    assert not any(e["evidence_sentence"].startswith("'") for e in stored.values()), (
        "the key must store the raw sentence, not the injection-defended form"
    )


@pytest.mark.parametrize(
    "out",
    ["adjudication", "adjudication.v2", "round2.2026-08-15", "pack.csv"],
)
def test_the_two_halves_of_a_pack_always_share_a_stem(tmp_path, out: str):
    """One stem, two files. They were derived by two incompatible rules.

    `out.with_suffix(".csv")` replaces the last dotted component while `out.name + "_key.json"`
    appends to the whole name. So `--out round2.2026-08-15` wrote `round2.csv` beside
    `round2.2026-08-15_key.json`: halves sharing no stem, so the obvious sibling lookup finds
    nothing. And `--out adjudication.v2` resolved its CSV to `adjudication.csv`, which is the
    committed, human-labelled pack.
    """
    from benchmarks.labelling.truth_extraction.build_adjudication import pack_paths

    csv_path, key_path = pack_paths(tmp_path / out)
    assert csv_path.suffix == ".csv"
    assert key_path.name == csv_path.name[: -len(".csv")] + "_key.json", (
        f"{csv_path.name} and {key_path.name} do not name one pack"
    )


def test_a_rebuild_refuses_to_replace_an_existing_pack(tmp_path):
    """`main`'s default --out IS the committed pack, and a rebuild silently truncated it.

    Reproduced before this guard: running the builder from the repo root took the labelled CSV
    from 5808 bytes to 108 and its 37 verdicts to 0, exit code 0, output "1 items". The
    row-count test tells an operator to set RECALL_PEPS_DIR and rebuild, which is that exact
    invocation. Git tracked the file, so it was recoverable. That is luck, not a design.
    """
    from benchmarks.labelling.truth_extraction.build_adjudication import build_rows, write_pack

    rows, key = build_rows(_formula_corpus(tmp_path), seed=0, limit=None)
    out = tmp_path / "pack"
    csv_path, _ = write_pack(rows, key, out)
    labelled = csv_path.read_text(encoding="utf-8").replace(",\n", ",Y\n")
    csv_path.write_text(labelled, encoding="utf-8", newline="")

    with pytest.raises(SystemExit, match="--force"):
        write_pack(rows, key, out)
    assert csv_path.read_text(encoding="utf-8") == labelled, "the verdicts were overwritten"

    write_pack(rows, key, out, force=True)  # and --force still works
    assert csv_path.read_text(encoding="utf-8") != labelled


@pytest.mark.parametrize("when", ["staging", "replacing"])
@pytest.mark.parametrize("failure", [OSError("no space left on device"), KeyboardInterrupt()])
@pytest.mark.parametrize("preexisting", [False, True], ids=["fresh", "over-a-labelled-pack"])
def test_a_failed_key_write_never_leaves_a_mismatched_pack(
    tmp_path, monkeypatch, failure: BaseException, preexisting: bool, when: str
):
    """Item numbers restart at 1 in every build.

    So a new CSV beside a previous run's key is not a broken pack, it is a pack whose un-blinding
    record attributes every sentence to the WRONG source PEP, and nothing about it looks wrong.

    ⚠️ Both parameters exist because the first version of this test had neither, and each hid a
    defect that destroyed adjudicated work:

    - It ran on a FRESH path only, so `assert not key_path.exists()` could not fail (the key was
      never written) and the `--force` path was untested. On that path the rollback deleted the
      human-labelled CSV outright: `os.replace` had already overwritten it, and the unlink then
      removed the only remaining copy, leaving the OLD key behind.
    - It raised only `OSError`, so `except Exception` looked correct. Ctrl-C during the key write
      is the likeliest interruption there is, and it skipped the rollback entirely.
    """
    import benchmarks.labelling.truth_extraction.build_adjudication as builder

    rows, key = builder.build_rows(_formula_corpus(tmp_path), seed=0, limit=None)
    out = tmp_path / "pack"
    csv_path, key_path = builder.pack_paths(out)

    before: bytes | None = None
    if preexisting:
        builder.write_pack(rows, key, out)
        # Labelled, so a loss is visible as a loss rather than as an identical rebuild.
        csv_path.write_text(
            csv_path.read_text(encoding="utf-8").replace(",\n", ",Y\n"),
            encoding="utf-8",
            newline="",
        )
        before = csv_path.read_bytes()
        assert b"Y" in before, "the fixture must carry a verdict, or a loss is invisible"

    # ⚠️ BOTH injection points, because they reach different guards and each version of this test
    # had only one. Failing at STAGING never replaces the CSV, so it exercises the scratch-file
    # cleanup and leaves the rollback untouched — three separate mutations of the rollback were
    # green under it. Failing at the second REPLACE is the state that matters: the CSV landed and
    # the key did not.
    if when == "staging":
        real_stage = builder._stage

        def _fail_stage(path, text):
            if path.name.endswith("_key.json"):
                raise failure
            return real_stage(path, text)

        monkeypatch.setattr(builder, "_stage", _fail_stage)
    else:
        real_replace = builder.os.replace

        def _fail_replace(src, dst):
            if str(dst).endswith("_key.json"):
                raise failure
            return real_replace(src, dst)

        monkeypatch.setattr(builder.os, "replace", _fail_replace)

    with pytest.raises(type(failure)):
        builder.write_pack(rows, key, out, force=preexisting)
    monkeypatch.undo()

    if preexisting:
        assert csv_path.read_bytes() == before, "the labelled pack was destroyed by a rebuild"
    else:
        assert not csv_path.exists(), "a blind CSV was left with no key that can un-blind it"
        assert not key_path.exists()
    # And no scratch file survives either. A fixed `<target>.tmp` name was shared by two
    # concurrent builds, so one run's rename could commit the other run's bytes.
    assert not list(tmp_path.glob("*.tmp")), f"staging files left behind: {list(tmp_path.iterdir())}"


def test_two_builds_of_one_pack_do_not_share_a_staging_file(tmp_path):
    """A fixed `<target>.tmp` is shared, so one run's rename commits the other run's bytes.

    That produces a CSV beside the wrong key, which is the state the whole write path exists to
    prevent, and it needs no crash to happen: two builds of the same `--out` are enough.
    """
    from benchmarks.labelling.truth_extraction.build_adjudication import _stage

    target = tmp_path / "pack.csv"
    first, second = _stage(target, "a"), _stage(target, "b")
    try:
        assert first != second, "two builds of one pack share a staging file"
        assert first.read_text(encoding="utf-8") == "a", "the second build overwrote the first"
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


@pytest.mark.parametrize("out", ["", ".", "C:/"])
def test_an_out_that_names_no_file_is_refused_with_a_sentence(out: str):
    """And before the census runs, which is why `main` calls `pack_paths` first.

    `Path("").name` is empty and `with_name` then raises `ValueError: WindowsPath('.') has an
    empty name` — after a walk of 733 files, and with no mention of `--out`.
    """
    from benchmarks.labelling.truth_extraction.build_adjudication import pack_paths

    with pytest.raises(SystemExit, match="names no file"):
        pack_paths(Path(out))


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_is_refused_rather_than_reinterpreted(tmp_path, limit: int):
    """`if limit:` read 0 as "no cap" and -1 as `candidates[:-1]`, both silently."""
    from benchmarks.labelling.truth_extraction.build_adjudication import build_rows

    with pytest.raises(ValueError, match="at least 1"):
        build_rows(_formula_corpus(tmp_path), seed=0, limit=limit)


def test_the_builder_refuses_a_key_that_does_not_cover_the_rows(tmp_path):
    """The invariant every reader assumes, asserted where the pack is MADE."""
    from benchmarks.labelling.truth_extraction.build_adjudication import build_rows, write_pack

    rows, key = build_rows(_formula_corpus(tmp_path), seed=0, limit=None)
    key.pop(rows[0]["item"])
    with pytest.raises(ValueError, match="exactly the CSV's items"):
        write_pack(rows, key, tmp_path / "pack")


@pytest.mark.parametrize(
    ("item", "int_parses"),
    [
        ("", False),
        # 128 non-ASCII characters satisfy `str.isdigit()` and still raise from `int()`. This is
        # the case the first version of the guard missed, so it passed and the digest test then
        # produced the bare `ValueError: invalid literal for int()` the guard exists to prevent.
        ("³", False),
        ("1.0", False),
        # Stricter than `int()` on purpose: `int(" 1")` succeeds, but a padded item cell is
        # spreadsheet damage, and the digest sorts on these.
        (" 1", True),
    ],
)
def test_a_malformed_item_cell_is_named_by_the_shape_guard(item: str, int_parses: bool):
    """Applied as DATA, because the committed pack carries no such cell.

    Deleting the guard alone changes nothing observable and reads as a survivor that says
    nothing about it: the corruption has to reach the predicate for the predicate to be measured.
    """
    import csv as _csv

    from tests.test_truth_extraction_adjudication import CSV_PATH, malformed_item_cells

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        parsed = list(_csv.reader(handle))
    parsed[3][0] = item

    # THE guard's predicate, imported rather than restated. The first version of this test copied
    # the condition into its own body, so removing `isascii()` from the real guard left it green:
    # a guard's test asserting on its own copy of the guard is the defect this file has now hit
    # three times. Asserting through the real test function instead would need the committed file
    # mutated on disk, and this pack is irreplaceable human work.
    assert malformed_item_cells(parsed[1:]) == [item], (
        f"the shape guard's predicate accepts {item!r}"
    )
    if not int_parses:
        with pytest.raises(ValueError):
            int(item)


def _formula_corpus(tmp_path):
    """A corpus whose marker sentence opens with `-`, so `_csv_safe` has work to do."""
    corpus = tmp_path / "peps"
    corpus.mkdir()
    (corpus / "pep-0001.rst").write_text(
        "PEP: 1\nTitle: One\nStatus: Active\n\n\n- superseded by :pep:`2` in the list above.\n",
        encoding="utf-8",
    )
    (corpus / "pep-0002.rst").write_text(
        "PEP: 2\nTitle: Two\nStatus: Active\n\n\nNothing.\n", encoding="utf-8"
    )
    return corpus


def test_every_row_names_a_candidate_target():
    # The pool is 175 FILES but only the 30 that name a target in the marker's sentence are
    # adjudicable. A row with an empty target would be asking a human to guess at an unprovable
    # one, which is the class fix.py refuses to guess at rather than the class it adjudicates.
    assert all(row["candidate_target"].startswith("pep-") for row in _csv_rows())


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
