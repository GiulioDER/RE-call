"""The one resume mechanism, and the two runners that now share it."""
from __future__ import annotations

import json

import pytest

from recall.eval.resume import LedgerCorrupted, append_rows, ledger_id, read_ledger


def _write(path, rows, *, torn_tail: str | None = None) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if torn_tail is not None:
        text += torn_tail  # no trailing newline: what a SIGKILL mid-write leaves
    path.write_text(text, encoding="utf-8")


def test_a_ledgers_bytes_do_not_depend_on_the_operating_system(tmp_path) -> None:
    """These are committed artifacts; text-mode translation would write CRLF on Windows."""
    path = tmp_path / "ledger.jsonl"
    append_rows(path, [{"question_id": "a"}, {"question_id": "b"}])
    assert b"\r\n" not in path.read_bytes()
    _write(path, [{"question_id": "a"}], torn_tail='{"question_id": "b"')
    read_ledger(path, id_field="question_id", repair_truncated_tail=True)
    assert b"\r\n" not in path.read_bytes()


def test_a_missing_file_is_an_empty_ledger_not_an_error(tmp_path) -> None:
    read = read_ledger(tmp_path / "absent.jsonl", id_field="question_id")
    assert read.rows == () and read.ids == frozenset()


def test_a_truncated_trailing_line_counts_as_not_yet_recorded(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    _write(path, [{"question_id": "a"}, {"question_id": "b"}], torn_tail='{"question_id": "c"')
    read = read_ledger(path, id_field="question_id")
    assert read.ids == {"a", "b"}
    assert read.truncated == (path,)


def test_corruption_in_the_middle_is_loud(tmp_path) -> None:
    """A torn tail has an explanation. A broken line with good lines after it does not."""
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"question_id": "a"}\nNOT JSON\n{"question_id": "c"}\n', encoding="utf-8"
    )
    with pytest.raises(LedgerCorrupted, match="NOT the last line"):
        read_ledger(path, id_field="question_id")


def test_repair_rewrites_the_file_so_the_next_append_is_not_glued_on(tmp_path) -> None:
    """Without the rewrite the next row lands on the end of a partial line and the file stops
    being JSONL for every future reader, not just this run."""
    path = tmp_path / "ledger.jsonl"
    _write(path, [{"question_id": "a"}], torn_tail='{"question_id": "b"')
    read_ledger(path, id_field="question_id", repair_truncated_tail=True)
    append_rows(path, [{"question_id": "c"}])
    assert read_ledger(path, id_field="question_id").ids == {"a", "c"}


def test_reading_without_repair_leaves_the_file_alone(tmp_path) -> None:
    """A reader of another attempt's sidecar must not rewrite it."""
    path = tmp_path / "ledger.jsonl"
    _write(path, [{"question_id": "a"}], torn_tail='{"question_id": "b"')
    before = path.read_bytes()
    read_ledger(path, id_field="question_id", repair_truncated_tail=False)
    assert path.read_bytes() == before


def test_the_first_occurrence_of_an_id_wins_across_several_files(tmp_path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(first, [{"question_id": "x", "score": 1}])
    _write(second, [{"question_id": "x", "score": 2}])
    read = read_ledger([first, second], id_field="question_id")
    assert [row["score"] for row in read.rows] == [1]


def test_a_composite_key_cannot_collide_by_moving_the_boundary(tmp_path) -> None:
    """("ab","c") and ("a","bc") must not share a resume key: one would never be scored."""
    left = ledger_id({"corpus": "ab", "question_id": "c"}, ("corpus", "question_id"))
    right = ledger_id({"corpus": "a", "question_id": "bc"}, ("corpus", "question_id"))
    assert left != right


def test_a_composite_key_separates_the_same_id_in_two_corpora(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    _write(
        path,
        [
            {"corpus": "labelled", "question_id": "q01"},
            {"corpus": "peps", "question_id": "q01"},
        ],
    )
    read = read_ledger(path, id_field=("corpus", "question_id"))
    assert len(read.rows) == 2


def test_the_wrong_id_field_is_refused_rather_than_re_running_everything(tmp_path) -> None:
    """Silently returning an empty ledger would re-run every item and report a clean resume."""
    path = tmp_path / "ledger.jsonl"
    _write(path, [{"instance_id": "a"}])
    with pytest.raises(LedgerCorrupted, match="cannot be resumed"):
        read_ledger(path, id_field="question_id")


def test_ladder_recorded_delegates_and_keeps_its_behaviour(tmp_path) -> None:
    """The ladder's resume contract, asserted through the ladder's own function."""
    from benchmarks.ladder.run import _recorded

    path = tmp_path / "out.jsonl"
    _write(
        path,
        [
            {"instance_id": "i1", "abstained": True},
            {"instance_id": "i2", "abstained": False},
        ],
        torn_tail='{"instance_id": "i3"',
    )
    assert _recorded(path) == {"i1": True, "i2": False}
    # repaired in place, so the caller's append-mode reopen is safe
    assert path.read_text(encoding="utf-8").endswith('"instance_id": "i2"}\n')


def test_beam_already_done_now_survives_the_torn_tail_it_was_guaranteed_to_meet(tmp_path) -> None:
    """These sidecars are read after an interruption, which is exactly what tears a tail."""
    from benchmarks.beam.run import _already_done

    path = tmp_path / "partial.jsonl"
    _write(path, [{"question_id": "q1"}], torn_tail='{"question_id": "q2"')
    rows, done = _already_done([path])
    assert done == {"q1"} and len(rows) == 1


def test_beam_already_done_does_not_rewrite_another_attempts_sidecar(tmp_path) -> None:
    from benchmarks.beam.run import _already_done

    path = tmp_path / "partial.jsonl"
    _write(path, [{"question_id": "q1"}], torn_tail='{"question_id": "q2"')
    before = path.read_bytes()
    _already_done([path])
    assert path.read_bytes() == before
