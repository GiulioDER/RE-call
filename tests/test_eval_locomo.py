"""The LOCOMO adapter's pure data-shaping logic.

The benchmark run itself needs a live pgvector and is exercised by hand (see the module
docstring); these tests pin the parts that turn LOCOMO's JSON into a corpus and its evidence ids
into a score, because a silent bug there does not crash — it reports a wrong hit@k that looks
plausible. Every assertion here guards a specific way the mapping could be quietly wrong.
"""
from __future__ import annotations

from recall.eval.locomo import (
    ADVERSARIAL_CATEGORY,
    _dia_id_to_filename,
    _filename_to_dia_id,
    _rate,
    _turn_document,
    write_conversation_corpus,
)


class _FakeChunk:
    def __init__(self, file: str) -> None:
        self.metadata = {"file": file}


class _FakeHit:
    def __init__(self, file: str) -> None:
        self.chunk = _FakeChunk(file)


def test_dia_id_round_trips_through_filename() -> None:
    # The colon is stripped for Windows/NTFS safety, then must map back exactly — a lossy round
    # trip would score a correct retrieval as a miss against the D-prefixed evidence id.
    for dia in ("D1:3", "D10:27", "D5:1"):
        assert _filename_to_dia_id(_dia_id_to_filename(dia)) == dia


def test_filename_has_no_colon() -> None:
    # The reason the round trip exists at all: a raw colon is a reserved char on Windows.
    assert ":" not in _dia_id_to_filename("D1:3")


def test_turn_document_carries_speaker_and_date() -> None:
    # Speaker and date are frequently the answer (who/when questions), so they must reach the
    # indexed body, not sit in metadata the embedder never sees.
    turn = {"speaker": "Caroline", "text": "I went to the support group.", "dia_id": "D1:3"}
    doc = _turn_document(turn, "7 May 2023")
    assert "Caroline" in doc
    assert "7 May 2023" in doc
    assert "support group" in doc


def test_turn_document_includes_image_caption() -> None:
    turn = {"speaker": "Mel", "text": "Look!", "blip_caption": "a sunrise over water"}
    doc = _turn_document(turn, "2022")
    assert "sunrise over water" in doc


def test_write_corpus_creates_one_file_per_turn(tmp_path) -> None:
    conversation = {
        "speaker_a": "Caroline",
        "speaker_b": "Mel",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "hi"},
            {"speaker": "Mel", "dia_id": "D1:2", "text": "hey"},
        ],
        "session_2_date_time": "2 Jan 2023",
        "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "back again"},
        ],
    }
    n = write_conversation_corpus(conversation, tmp_path)
    assert n == 3
    assert (tmp_path / "D1_1.md").exists()
    assert (tmp_path / "D2_1.md").exists()
    # The date metadata keys must NOT become documents.
    assert not (tmp_path / "session_1_date_time.md").exists()


def test_write_corpus_skips_turns_without_dia_id(tmp_path) -> None:
    # A turn with no id cannot be scored against evidence, so it must not silently become a file
    # that inflates the turn count.
    conversation = {
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "hi"},
            {"speaker": "Mel", "text": "no id here"},
        ],
    }
    assert write_conversation_corpus(conversation, tmp_path) == 1


def test_write_corpus_orders_sessions_numerically(tmp_path) -> None:
    # session_10 must sort after session_2, not lexically before it. Ordering only affects which
    # date a turn gets if ids repeated across sessions; the guard is cheap and the bug is subtle.
    conversation = {f"session_{i}_date_time": f"day {i}" for i in range(1, 12)}
    for i in range(1, 12):
        conversation[f"session_{i}"] = [{"speaker": "A", "dia_id": f"D{i}:1", "text": f"s{i}"}]
    n = write_conversation_corpus(conversation, tmp_path)
    assert n == 11
    assert "day 10" in (tmp_path / "D10_1.md").read_text(encoding="utf-8")


def test_rate_reports_n_and_ci() -> None:
    r = _rate([True, True, False, True])
    assert r["n"] == 4
    assert r["rate"] == 0.75
    assert r["ci95"][0] <= 0.75 <= r["ci95"][1]


def test_rate_empty_is_nan_not_crash() -> None:
    r = _rate([])
    assert r["n"] == 0
    assert r["rate"] != r["rate"]  # NaN


def test_adversarial_category_is_five() -> None:
    # A guard against a silent renumbering: category 5 is the adversarial split the abstention
    # arm scores. If this constant drifts, abstention would be measured on answerable questions.
    assert ADVERSARIAL_CATEGORY == 5
