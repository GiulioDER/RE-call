"""The run harness: `run_arm` scoring, and `_load`'s LOCOMO -> question-list translation.

Everything here is offline. `run_arm` is driven with a fake system and a fake completer (no DB,
no network, no spend), and `_load` is driven with a small inline LOCOMO-shaped fixture written to
a temp file rather than the real `locomo10.json` — so the assertions below pin the translation
rules (which questions survive, how `adversarial` is derived, how ids are formed) instead of
restating whatever the shipped dataset happens to contain.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks import run as run_module
from benchmarks.llm import OpenRouterLLM
from benchmarks.pipeline import Outcome
from benchmarks.run import _load, main, run_arm
from benchmarks.systems import MemorySystem


class _Sys:
    """Retrieves nothing for the adversarial question, a real memory for the answerable one."""

    name = "fake"

    def __init__(self) -> None:
        self.ingested: list[dict[str, Any]] = []

    def ingest(self, conversation: dict[str, Any]) -> None:
        self.ingested.append(conversation)

    def retrieve(self, question: str) -> str:
        return "" if "penguin" in question else "rate limit is 500 rps"


def _completer(system: str, user: str) -> str:
    if "Correct?" in user:
        return "YES"
    if "<memories>\n\n</memories>" in user:  # empty context -> the generator must refuse
        return "NO_ANSWER"
    return "500 rps"


def test_run_arm_produces_outcomes_and_aggregate() -> None:
    questions = [
        {
            "question_id": "1",
            "category": "cat1",
            "adversarial": False,
            "question": "rps?",
            "answer": "500",
        },
        {
            "question_id": "2",
            "category": "cat5-adversarial",
            "adversarial": True,
            "question": "penguins on mars?",
            "answer": "",
        },
    ]
    outcomes, agg = run_arm(_Sys(), _completer, questions)
    assert len(outcomes) == 2
    assert all(isinstance(o, Outcome) for o in outcomes)
    assert agg["answerable_accuracy"]["n"] == 1
    assert agg["answerable_accuracy"]["rate"] == 1.0
    assert agg["adversarial_abstention"]["n"] == 1
    assert agg["adversarial_abstention"]["rate"] == 1.0


def test_run_arm_records_context_and_answer_for_the_results_artifact() -> None:
    questions = [
        {
            "question_id": "1",
            "category": "cat1",
            "adversarial": False,
            "question": "rps?",
            "answer": "500",
        },
    ]
    outcomes, _agg = run_arm(_Sys(), _completer, questions)
    assert outcomes[0].context == "rate limit is 500 rps"
    assert outcomes[0].answer == "500 rps"
    assert outcomes[0].correct is True


def _fixture() -> list[dict[str, Any]]:
    """Two LOCOMO-shaped items: the field names are the ones `locomo10.json` really uses.

    Categories 1-4 carry `answer`; category 5 carries `adversarial_answer` and (usually) no
    `answer` at all. `qa` entries that a run cannot score are included deliberately so the skip
    rules are covered: no gold answer, a blank question, and an out-of-range category.
    """
    return [
        {
            "sample_id": "conv-a",
            "conversation": {"speaker_a": "Alice", "speaker_b": "Bob"},
            "qa": [
                {
                    "question": "What did Alice research?",
                    "answer": "adoption agencies",
                    "evidence": ["D1:2"],
                    "category": 1,
                },
                # a numeric gold answer: LOCOMO really does ship a handful of these
                {"question": "How many siblings?", "answer": 3, "category": 2},
                {
                    "question": "What did Bob realise after his charity race?",
                    "adversarial_answer": "self-care is important",
                    "evidence": ["D2:3"],
                    "category": 5,
                },
                {"question": "Unscoreable: no gold answer", "category": 4},
                {"question": "   ", "answer": "blank question", "category": 1},
                {"question": "Category out of range", "answer": "x", "category": 9},
            ],
        },
        {
            "sample_id": "conv-b",
            "conversation": {"speaker_a": "Carol", "speaker_b": "Dan"},
            "qa": [{"question": "Who is Dan?", "answer": "a chef", "category": 3}],
        },
    ]


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "locomo-fixture.json"
    path.write_text(json.dumps(_fixture()), encoding="utf-8")
    return path


def test_load_sets_adversarial_flag_from_category(tmp_path: Path) -> None:
    _convs, questions = _load(_write_fixture(tmp_path))
    flags = {q["question_id"]: q["adversarial"] for q in questions}
    # category 5 and ONLY category 5 is adversarial; every other surviving question is answerable
    assert flags == {
        "conv-a:0": False,
        "conv-a:1": False,
        "conv-a:2": True,
        "conv-b:0": False,
    }
    assert all(isinstance(q["adversarial"], bool) for q in questions)


def test_load_question_ids_are_stable_and_unique(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    _c1, first = _load(path)
    _c2, second = _load(path)
    ids = [q["question_id"] for q in first]
    # stable: the same file yields the same ids, run to run
    assert ids == [q["question_id"] for q in second]
    # unique: the results artifact is keyed by question_id, so a collision would silently merge
    assert len(set(ids)) == len(ids)
    # ids survive the skipped rows: conv-a:2 is the THIRD qa entry, not the third kept one
    assert ids == ["conv-a:0", "conv-a:1", "conv-a:2", "conv-b:0"]


def test_load_skips_questions_a_run_cannot_score(tmp_path: Path) -> None:
    _convs, questions = _load(_write_fixture(tmp_path))
    kept = {q["question"] for q in questions}
    assert "Unscoreable: no gold answer" not in kept
    assert "Category out of range" not in kept
    assert all(q["question"].strip() for q in questions)


def test_load_carries_the_fields_run_question_needs(tmp_path: Path) -> None:
    _convs, questions = _load(_write_fixture(tmp_path))
    by_id = {q["question_id"]: q for q in questions}
    answerable = by_id["conv-a:0"]
    assert answerable["answer"] == "adoption agencies"
    assert answerable["category"] == "cat1"
    assert answerable["sample_id"] == "conv-a"
    # a numeric gold answer is coerced to str for the judge prompt
    assert by_id["conv-a:1"]["answer"] == "3"
    # adversarials have no gold answer to be correct about
    assert by_id["conv-a:2"]["answer"] == ""
    assert by_id["conv-a:2"]["category"] == "cat5-adversarial"


def test_load_slices_conversations_and_their_questions(tmp_path: Path) -> None:
    convs, questions = _load(_write_fixture(tmp_path), limit=1)
    assert [c["sample_id"] for c in convs] == ["conv-a"]
    # the question list must be sliced WITH the conversations, or the harness would score
    # questions about a conversation it never ingested
    assert {q["sample_id"] for q in questions} == {"conv-a"}


def test_load_returns_the_outer_locomo_items_for_ingest(tmp_path: Path) -> None:
    """`ingest()` takes the OUTER item (`sample_id` + nested `conversation`), not the inner one.

    Handing the inner conversation object to `RecallSystem.ingest` would index zero turns and make
    the arm abstain on everything — a spectacular, entirely false abstention score.
    """
    convs, _questions = _load(_write_fixture(tmp_path))
    assert convs[0]["sample_id"] == "conv-a"
    assert convs[0]["conversation"]["speaker_a"] == "Alice"


def test_main_ingests_each_conversation_before_scoring_its_own_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingest and scoring must interleave per conversation, and the dump must be valid JSON.

    Both adapters scope retrieval to the LAST conversation ingested, so a harness that ingested
    every conversation up front would answer all 1,986 questions out of the tenth conversation's
    memory — producing a full, publishable-looking results file that is entirely wrong. This runs
    `main` end-to-end with a fake system and a fake completer (no DB, no network, no spend) and
    pins the ordering plus the artifact's contents.
    """
    events: list[tuple[str, str]] = []

    class _OrderedSys:
        name = "ordered"

        def ingest(self, conversation: dict[str, Any]) -> None:
            events.append(("ingest", str(conversation["sample_id"])))

        def retrieve(self, question: str) -> str:
            events.append(("retrieve", question))
            return "ctx"

    def _fake_build(arm: str, model: str, openrouter_key: str) -> MemorySystem:
        return _OrderedSys()

    def _fake_complete(self: OpenRouterLLM, system: str, user: str) -> str:
        return "YES" if "Correct?" in user else "an answer"

    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-by-the-fake-completer")
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _fake_complete)

    data = _write_fixture(tmp_path)
    out = tmp_path / "results"
    code = main(
        ["--arm", "recall", "--data", str(data), "--conversations", "2", "--out", str(out)]
    )
    assert code == 0

    # conv-a is ingested, then its 3 surviving questions are retrieved, then conv-b is ingested
    assert [i for i, (kind, _) in enumerate(events) if kind == "ingest"] == [0, 4]
    assert events[0] == ("ingest", "conv-a")
    assert events[4] == ("ingest", "conv-b")

    payload = json.loads((out / "recall_openai-gpt-4o-mini_2conv.json").read_text(encoding="utf-8"))
    assert payload["arm"] == "recall"
    assert payload["conversations"] == 2
    assert payload["questions"] == 4
    assert payload["aggregate"]["adversarial_abstention"]["n"] == 1
    by_id = {o["question_id"]: o for o in payload["outcomes"]}
    assert set(by_id) == {"conv-a:0", "conv-a:1", "conv-a:2", "conv-b:0"}
    # the artifact must be re-scorable: question text, retrieved context, answer and verdict
    record = by_id["conv-a:0"]
    assert record["question"] == "What did Alice research?"
    assert record["context"] == "ctx"
    assert record["answer"] == "an answer"
    assert record["correct"] is True
    assert by_id["conv-a:2"]["correct"] is None  # adversarial: unscored by construction


def test_main_dump_has_no_nan_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`NaN` is not valid JSON for any non-Python parser, and empty rate blocks are the risk.

    A single-conversation slice with an adversarial-only category leaves `cat5-adversarial`'s
    answerable sub-block built from an empty list — exactly where a bare `NaN` would appear.
    """

    class _EmptySys:
        name = "empty"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return ""

    def _fake_build(arm: str, model: str, openrouter_key: str) -> MemorySystem:
        return _EmptySys()

    def _fake_complete(self: OpenRouterLLM, system: str, user: str) -> str:
        return "NO_ANSWER"

    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-by-the-fake-completer")
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _fake_complete)

    out = tmp_path / "results"
    main(
        [
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--conversations", "1",
            "--out", str(out),
        ]
    )
    dumped = (out / "recall_openai-gpt-4o-mini_1conv.json").read_text(encoding="utf-8")
    assert "NaN" not in dumped
    payload = json.loads(dumped)
    cat5 = payload["aggregate"]["by_category"]["cat5-adversarial"]
    assert cat5["answerable_accuracy"]["rate"] is None
    assert cat5["adversarial_abstention"]["rate"] == 1.0
