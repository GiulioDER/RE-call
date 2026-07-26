"""The LOCOMO adapter's pure data-shaping logic.

The benchmark run itself needs a live pgvector and is exercised by hand (see the module
docstring); these tests pin the parts that turn LOCOMO's JSON into a corpus and its evidence ids
into a score, because a silent bug there does not crash — it reports a wrong hit@k that looks
plausible. Every assertion here guards a specific way the mapping could be quietly wrong.
"""
from __future__ import annotations

from recall.eval.locomo import (
    ADVERSARIAL_CATEGORY,
    _depths,
    _dia_id_to_filename,
    _filename_to_dia_id,
    _hit_by_depth,
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


# --- the depth curve: one retrieval, scored at every k ---------------------------------------

def test_depth_curve_finds_evidence_that_arrives_below_the_shallow_cutoff() -> None:
    # The whole point of the curve: an evidence turn ranked 8th is a miss at k=5 and a hit at
    # k=10. If this collapsed to one value the curve would be reporting a constant.
    hits = [_FakeHit(_dia_id_to_filename(f"D1:{i}")) for i in range(1, 11)]

    by_k = _hit_by_depth(hits, ["D1:8"], [1, 5, 10])

    assert by_k == {1: False, 5: False, 10: True}


def test_depth_truncates_chunks_not_the_dialog_ids_derived_from_them() -> None:
    # THE inflation trap. Six chunks occupy the top of the ranking but map to only two turns, so
    # the evidence turn is the THIRD distinct id while sitting at chunk rank 7. Slicing the id
    # list to 5 would include it and score a hit at k=5 that a k=5 retrieval never returned.
    # Densely-chunked corpora would inflate most, which is exactly where it would be believed.
    hits = [_FakeHit(_dia_id_to_filename("D1:1")) for _ in range(3)]
    hits += [_FakeHit(_dia_id_to_filename("D1:2")) for _ in range(3)]
    hits += [_FakeHit(_dia_id_to_filename("D1:9"))]

    by_k = _hit_by_depth(hits, ["D1:9"], [5, 7])

    assert by_k[5] is False, "sliced the projection instead of the ranking"
    assert by_k[7] is True


def test_depth_curve_is_monotone_in_k() -> None:
    # A deeper retrieval is a superset of a shallower one, so hit@k can never fall as k rises.
    # A non-monotone curve means the truncation is wrong, not that the corpus is unusual.
    hits = [_FakeHit(_dia_id_to_filename(f"D1:{i}")) for i in range(1, 21)]

    rates = [_hit_by_depth(hits, ["D1:15"], [d])[d] for d in (1, 3, 5, 10, 20)]

    assert rates == sorted(rates), "hit@k fell as k rose"


def test_depth_curve_agrees_with_a_single_depth_scoring() -> None:
    # The curve's k row must equal what a run issued at that k alone would score — the coherence
    # check `run()` asserts on the real data, pinned here on data that cannot drift.
    hits = [_FakeHit(_dia_id_to_filename(f"D1:{i}")) for i in range(1, 11)]

    curve = _hit_by_depth(hits, ["D1:4"], [1, 4, 10])
    alone = _hit_by_depth(hits, ["D1:4"], [4])

    assert curve[4] == alone[4]


def test_any_of_several_evidence_turns_counts_as_a_hit() -> None:
    # LOCOMO annotates multi-turn questions with several evidence ids and the metric is "was the
    # evidence retrieved", not "were all of them" — scoring it as all-of would report a stricter
    # metric than the label supports.
    hits = [_FakeHit(_dia_id_to_filename("D1:2"))]

    assert _hit_by_depth(hits, ["D1:7", "D1:2"], [1])[1] is True


def test_depths_always_includes_the_headline_k() -> None:
    # run_conversation reads hit_by_k[k]; if k were ever absent from the scored depths the
    # headline would be missing (before, a fallback silently reported a deeper depth instead).
    # _depths folds k in unconditionally, and also dedupes and sorts.
    assert _depths([10, 20], 5) == [5, 10, 20]      # k absent from the curve list -> still scored
    assert _depths([5, 5, 10], 5) == [5, 10]        # duplicates collapse
    assert _depths(None, 5) == [5]                  # no curve -> just the headline
    assert _depths([20, 3, 1], 5) == [1, 3, 5, 20]  # sorted ascending
