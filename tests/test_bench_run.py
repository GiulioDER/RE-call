"""The run harness: `run_arm` scoring, and `_load`'s LOCOMO -> question-list translation.

Everything here is offline. `run_arm` is driven with a fake system and a fake completer (no DB,
no network, no spend), and `_load` is driven with a small inline LOCOMO-shaped fixture written to
a temp file rather than the real `locomo10.json` — so the assertions below pin the translation
rules (which questions survive, how `adversarial` is derived, how ids are formed) instead of
restating whatever the shipped dataset happens to contain.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from benchmarks import run as run_module
from benchmarks.llm import OpenRouterLLM
from benchmarks.pipeline import GEN_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, Outcome
from benchmarks.run import _load, main, run_arm, validate_openrouter_key
from benchmarks.systems import DEFAULT_K, MemorySystem

#: A shape-valid but obviously fake key. `main` validates the key's SHAPE before doing any work
#: (see `validate_openrouter_key`), so the placeholder these tests used to pass is now rejected —
#: which is the point. The completer is faked in every `main` test, so the value is never sent.
_FAKE_KEY = "sk-or-v1-unused-by-the-fake-completer"

#: Pinned run time, so the tests can name the artifact files exactly. `main` takes `now` as a
#: parameter for this reason — nothing here freezes the process clock.
_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_STAMP_1CONV = "recall_openai-gpt-4o-mini_1conv_20260102T030405Z"
_STAMP_2CONV = "recall_openai-gpt-4o-mini_2conv_20260102T030405Z"


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

    Categories 1-4 carry `answer`; category 5 carries `adversarial_answer` and USUALLY no `answer`
    at all — but not always, and the exception is the whole point of index 4 below. Two real rows
    (`conv-26:167`, `conv-26:178`) carry BOTH, so one fixture row mirrors that shape: without it
    an implementation that derived `adversarial` from `"answer" not in qa` would produce exactly
    the same output as the correct one on this fixture and every assertion would still pass.

    `qa` entries that a run cannot score are included deliberately so the skip rules are covered:
    no gold answer, a blank question, and an out-of-range category. The no-gold-answer row sits at
    index 1 — BETWEEN two kept rows, not trailing — because a trailing skip leaves the kept-row
    index and the source-row index identical, and `question_id` is only meaningful if the two can
    be told apart.
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
                # SKIPPED, and deliberately NOT trailing: id `conv-a:1` must never be handed out,
                # and every id after it must still count from the source list.
                {"question": "Unscoreable: no gold answer", "category": 4},
                # a numeric gold answer: LOCOMO really does ship a handful of these
                {"question": "How many siblings?", "answer": 3, "category": 2},
                {
                    "question": "What did Bob realise after his charity race?",
                    "adversarial_answer": "self-care is important",
                    "evidence": ["D2:3"],
                    "category": 5,
                },
                # The `conv-26:167` shape: category 5 WITH a real `answer` beside the adversarial
                # one. Field presence says "answerable"; the category says adversarial and wins.
                {
                    "question": "Did Bob make the black and white bowl in the photo?",
                    "adversarial_answer": "Yes",
                    "answer": "No",
                    "evidence": ["D3:1"],
                    "category": 5,
                },
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


class _StubSys:
    """Ingests anything, retrieves a constant context. No DB, no network, no spend."""

    name = "stub"

    def ingest(self, conversation: dict[str, Any]) -> None:
        return None

    def retrieve(self, question: str) -> str:
        return "ctx"


def _stub_build(arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed", **_extra: object) -> MemorySystem:
    return _StubSys()


def _patch_recall_stub(monkeypatch: pytest.MonkeyPatch, *, retrieve_returns: str = "ctx") -> None:
    """Make `RecallSystem` behave like `_StubSys` (no DB, no network, no spend).

    `main`'s `isinstance(system, RecallSystem)` guard is now an `assert` (Finding 5 of the
    final-review pass): `arm='recall'` with a bespoke non-`RecallSystem` fake now fails loudly
    instead of silently skipping the ablation preflight, which is the point of that guard. Tests
    below that only care about ordering/config/timestamps/NaN-safety, not about the preflight
    itself, build a real `RecallSystem` and patch its methods at the class level instead of a
    hand-rolled stand-in class.
    """
    from benchmarks.systems import RecallSystem

    def _ingest(self: RecallSystem, conversation: dict[str, Any]) -> None:
        return None

    def _retrieve(self: RecallSystem, question: str) -> str:
        return retrieve_returns

    def _no_op_preflight(
        self: RecallSystem, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
    ) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(RecallSystem, "ingest", _ingest)
    monkeypatch.setattr(RecallSystem, "retrieve", _retrieve)
    monkeypatch.setattr(RecallSystem, "ablation_preflight", _no_op_preflight)


def _recall_stub_build(
    arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed", **_extra: object
) -> MemorySystem:
    """A real `RecallSystem` (satisfies `main`'s isinstance assert), lazy enough to build without
    a database — pair with `_patch_recall_stub` to also stub its instance methods."""
    from benchmarks.systems import RecallSystem

    return RecallSystem("postgresql://x/y", embedder_name="hashing", k=k)


def _stub_complete(self: OpenRouterLLM, system: str, user: str) -> str:
    """Judge says YES, generator answers. Patched over `OpenRouterLLM.complete`: nothing is sent."""
    return "YES" if "Correct?" in user else "an answer"


def test_load_sets_adversarial_flag_from_category(tmp_path: Path) -> None:
    _convs, questions, _skipped = _load(_write_fixture(tmp_path))
    flags = {q["question_id"]: q["adversarial"] for q in questions}
    # category 5 and ONLY category 5 is adversarial; every other surviving question is answerable
    assert flags == {
        "conv-a:0": False,
        "conv-a:2": False,
        "conv-a:3": True,
        "conv-a:4": True,
        "conv-b:0": False,
    }
    assert all(isinstance(q["adversarial"], bool) for q in questions)


def test_load_derives_adversarial_from_category_not_from_field_presence(tmp_path: Path) -> None:
    """`conv-a:4` is category 5 AND carries a real `answer` — the `conv-26:167` shape.

    This is the row the cheap heuristic gets wrong. `adversarial = "answer" not in qa` would call
    it answerable, hand its `"No"` to the judge as gold, and score the system's correct refusal as
    a wrong answer — silently deflating the abstention column in a published artifact. Deriving
    the flag from the category is what makes the row come out adversarial with no gold at all.
    """
    _convs, questions, _skipped = _load(_write_fixture(tmp_path))
    both = {q["question_id"]: q for q in questions}["conv-a:4"]
    assert both["question"] == "Did Bob make the black and white bowl in the photo?"
    assert both["adversarial"] is True
    # the gold handed to the judge is empty, NOT the row's "No": an adversarial has nothing to be
    # correct about, and `run_question` never calls the judge for one
    assert both["answer"] == ""
    assert both["category"] == "cat5-adversarial"


def test_load_question_ids_are_stable_and_unique(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    _c1, first, _s1 = _load(path)
    _c2, second, _s2 = _load(path)
    ids = [q["question_id"] for q in first]
    # stable: the same file yields the same ids, run to run
    assert ids == [q["question_id"] for q in second]
    # unique: the results artifact is keyed by question_id, so a collision would silently merge
    assert len(set(ids)) == len(ids)
    # The id counts over the SOURCE `qa` list, not over the kept rows. `conv-a:1` is skipped (no
    # gold answer) and its number is burned, so every surviving id still joins straight back to
    # `locomo10.json` — which is the only reason a published record can be checked against the
    # dataset. Numbering over the kept rows would yield conv-a:0,1,2,3 here: the same ids, sliding
    # silently onto different source rows, and re-sliding whenever a skip rule changes.
    assert ids == ["conv-a:0", "conv-a:2", "conv-a:3", "conv-a:4", "conv-b:0"]


def test_load_skips_questions_a_run_cannot_score(tmp_path: Path) -> None:
    _convs, questions, _skipped = _load(_write_fixture(tmp_path))
    kept = {q["question"] for q in questions}
    assert "Unscoreable: no gold answer" not in kept
    assert "Category out of range" not in kept
    assert all(q["question"].strip() for q in questions)


def test_load_reports_how_many_rows_it_dropped_and_why(tmp_path: Path) -> None:
    """A published n that is not 1,540/446 must be explainable from the artifact, not guessed at.

    The loader silently dropped rows for three different reasons. Counted, they are a footnote;
    uncounted, a reader comparing this run's n against LOCOMO's documented split cannot tell "the
    dataset has defects we skipped" from "the harness loses questions".
    """
    _convs, questions, skipped = _load(_write_fixture(tmp_path))
    assert skipped["total"] == 3
    assert skipped["by_reason"] == {
        "no_gold_answer": 1,
        "blank_question": 1,
        "category_not_scored": 1,
    }
    # and the books balance: kept + skipped accounts for every `qa` row in the fixture
    assert len(questions) + skipped["total"] == sum(len(c["qa"]) for c in _fixture())


def test_load_requires_a_sample_id(tmp_path: Path) -> None:
    """A LOCOMO item with no `sample_id` stops the run instead of merging into `bench-None`."""
    path = tmp_path / "no-sample-id.json"
    path.write_text(
        json.dumps([{"conversation": {}, "qa": [{"question": "q", "answer": "a", "category": 1}]}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample_id"):
        _load(path)


def test_load_carries_the_fields_run_question_needs(tmp_path: Path) -> None:
    _convs, questions, _skipped = _load(_write_fixture(tmp_path))
    by_id = {q["question_id"]: q for q in questions}
    answerable = by_id["conv-a:0"]
    assert answerable["answer"] == "adoption agencies"
    assert answerable["category"] == "cat1"
    assert answerable["sample_id"] == "conv-a"
    # a numeric gold answer is coerced to str for the judge prompt
    assert by_id["conv-a:2"]["answer"] == "3"
    # adversarials have no gold answer to be correct about
    assert by_id["conv-a:3"]["answer"] == ""
    assert by_id["conv-a:3"]["category"] == "cat5-adversarial"


def test_load_slices_conversations_and_their_questions(tmp_path: Path) -> None:
    convs, questions, _skipped = _load(_write_fixture(tmp_path), limit=1)
    assert [c["sample_id"] for c in convs] == ["conv-a"]
    # the question list must be sliced WITH the conversations, or the harness would score
    # questions about a conversation it never ingested
    assert {q["sample_id"] for q in questions} == {"conv-a"}


def test_load_returns_the_outer_locomo_items_for_ingest(tmp_path: Path) -> None:
    """`ingest()` takes the OUTER item (`sample_id` + nested `conversation`), not the inner one.

    Handing the inner conversation object to `RecallSystem.ingest` would index zero turns and make
    the arm abstain on everything — a spectacular, entirely false abstention score.
    """
    convs, _questions, _skipped = _load(_write_fixture(tmp_path))
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
    from benchmarks.systems import RecallSystem

    events: list[tuple[str, str]] = []

    def _ordered_ingest(self: RecallSystem, conversation: dict[str, Any]) -> None:
        events.append(("ingest", str(conversation["sample_id"])))

    def _ordered_retrieve(self: RecallSystem, question: str) -> str:
        events.append(("retrieve", question))
        return "ctx"

    def _no_op_preflight(
        self: RecallSystem, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
    ) -> list[dict[str, Any]]:
        return []

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed", **_extra: object
    ) -> MemorySystem:
        return RecallSystem("postgresql://x/y", embedder_name="hashing", k=k)

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(RecallSystem, "ingest", _ordered_ingest)
    monkeypatch.setattr(RecallSystem, "retrieve", _ordered_retrieve)
    monkeypatch.setattr(RecallSystem, "ablation_preflight", _no_op_preflight)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    data = _write_fixture(tmp_path)
    out = tmp_path / "results"
    code = main(
        ["--arm", "recall", "--data", str(data), "--conversations", "2", "--out", str(out)],
        now=_NOW,
    )
    assert code == 0

    # conv-a is ingested, then its 4 surviving questions are retrieved, then conv-b is ingested
    assert [i for i, (kind, _) in enumerate(events) if kind == "ingest"] == [0, 5]
    assert events[0] == ("ingest", "conv-a")
    assert events[5] == ("ingest", "conv-b")

    payload = json.loads((out / f"{_STAMP_2CONV}.json").read_text(encoding="utf-8"))
    assert payload["arm"] == "recall"
    assert payload["conversations"] == 2
    assert payload["questions"] == 5
    assert payload["aggregate"]["adversarial_abstention"]["n"] == 2
    by_id = {o["question_id"]: o for o in payload["outcomes"]}
    assert set(by_id) == {"conv-a:0", "conv-a:2", "conv-a:3", "conv-a:4", "conv-b:0"}
    # the artifact must be re-scorable: question text, GOLD, retrieved context, answer, verdict
    record = by_id["conv-a:0"]
    assert record["question"] == "What did Alice research?"
    assert record["gold"] == "adoption agencies"
    assert record["context"] == "ctx"
    assert record["answer"] == "an answer"
    assert record["correct"] is True
    assert by_id["conv-a:3"]["correct"] is None  # adversarial: unscored by construction
    assert by_id["conv-a:4"]["correct"] is None  # ditto, despite its stray `answer` field
    # an adversarial has no gold to be correct about; the record says so rather than omitting it
    assert by_id["conv-a:3"]["gold"] == ""
    assert by_id["conv-a:4"]["gold"] == ""

    # the incremental sidecar ends up holding exactly the same records as the final artifact
    lines = (out / f"{_STAMP_2CONV}.partial.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == payload["outcomes"]


def test_main_closes_the_system_it_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`main` builds an arm and must release its handles before returning.

    A `Mem0System` holds exclusive Qdrant locks for as long as it lives (see `Mem0System.close`),
    including one on a machine-global telemetry path that no `run_id` disambiguates. The process
    exits immediately after this, and the OS drops file locks then regardless — so what this really
    buys is that the handles go while the interpreter is still healthy, rather than during
    finalisation, where qdrant-client's own finaliser dies with `ModuleNotFoundError: import of
    msvcrt halted`.

    Scope, stated so nobody reads more into it: this is the SUCCESS path. The write site is now
    wrapped in `try/finally`, so a raise there releases the handles too (see
    `test_a_rejected_cost_claim_still_releases_the_arms_handles`), but that `finally` starts at
    the write site — an exception during ingest or scoring still skips the close and relies on
    process exit. Widening it further is `with _build_system(...) as system:` around the body,
    not another close call somewhere else.
    """
    from benchmarks.systems import RecallSystem

    out = tmp_path / "results"
    closed: list[bool] = []

    class _ClosableSystem(RecallSystem):
        name = "recall"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def ablation_preflight(
            self, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            # Records WHETHER THE ARTIFACT WAS ALREADY WRITTEN, not merely that close ran. The
            # ordering is what makes a failing `close()` harmless: everything is on disk by then,
            # so the worst case is a bad exit status rather than a destroyed run. Asserting only
            # `closed == [...]` is order-blind, and a mutant that moves the close above the write
            # survives it.
            closed.append(any(out.glob("*.json")))

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str,
        embedder: str = "fastembed", **_extra: object,
    ) -> MemorySystem:
        return _ClosableSystem("postgresql://x/y", embedder_name="hashing", k=k)

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    data = _write_fixture(tmp_path)
    code = main(
        ["--arm", "recall", "--data", str(data), "--conversations", "2", "--out", str(out)],
        now=_NOW,
    )

    assert code == 0
    assert closed == [True]  # closed exactly once, and AFTER the artifact was written


def test_a_failing_close_cannot_turn_a_finished_run_into_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown of an already-written artifact must not change the run's verdict.

    By the time the close runs, both the results JSON and the incremental sidecar are on disk and
    their paths have been printed. If a raising `close()` propagated, `main` would never reach
    `return 0`, so a complete run would exit non-zero — and a wrapper or retry loop reading that
    status would re-run it, re-spending LLM credit on work that already succeeded.
    """
    from benchmarks.systems import RecallSystem

    out = tmp_path / "results"

    class _UnclosableSystem(RecallSystem):
        name = "recall"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def ablation_preflight(
            self, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            raise RuntimeError("teardown exploded")

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str,
        embedder: str = "fastembed", **_extra: object,
    ) -> MemorySystem:
        return _UnclosableSystem("postgresql://x/y", embedder_name="hashing", k=k)

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    data = _write_fixture(tmp_path)
    with pytest.warns(UserWarning, match="closing _UnclosableSystem failed"):
        code = main(
            ["--arm", "recall", "--data", str(data), "--conversations", "2", "--out", str(out)],
            now=_NOW,
        )

    assert code == 0                       # the run succeeded, and says so
    assert any(out.glob("*.json"))         # with its artifact intact
    assert any(out.glob("*.partial.jsonl"))


def test_a_failing_close_cannot_fail_the_run_under_warnings_as_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same guarantee, under `-W error`, where the report itself becomes a raise.

    This needs its own test because `pytest.warns` installs `catch_warnings` plus
    `simplefilter("always")`, so it NEUTRALISES warnings-as-errors inside its block. The test
    above therefore cannot see this failure mode however it is run, including under `-W error`:
    the one filter it is closest to is the one it suppresses.
    """
    from benchmarks.systems import RecallSystem

    out = tmp_path / "results"

    class _Unprintable(RuntimeError):
        def __repr__(self) -> str:
            raise ValueError("repr exploded")

        def __str__(self) -> str:
            raise ValueError("str exploded")

    class _UnclosableSystem(RecallSystem):
        name = "recall"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def ablation_preflight(
            self, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            # Both escape routes at once: the handler warns (which raises under `-W error`) and
            # interpolates the exception (whose `__repr__` raises).
            raise _Unprintable()

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str,
        embedder: str = "fastembed", **_extra: object,
    ) -> MemorySystem:
        return _UnclosableSystem("postgresql://x/y", embedder_name="hashing", k=k)

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    data = _write_fixture(tmp_path)
    with warnings.catch_warnings():
        # Narrow, not `simplefilter("error")`: turning EVERY warning into an error would fail this
        # test on any unrelated dependency DeprecationWarning raised anywhere in the run, and that
        # failure would read as a regression of the teardown guard. Only the report itself is
        # promoted, which is the property under test.
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message="benchmarks.run: closing", category=UserWarning)
        code = main(
            ["--arm", "recall", "--data", str(data), "--conversations", "2", "--out", str(out)],
            now=_NOW,
        )

    assert code == 0
    assert any(out.glob("*.json"))
    # The run surviving is half the contract; the breadcrumb surviving is the other half. Without
    # this, replacing the stderr fallback with a bare `pass` — the thing the helper's docstring
    # calls unacceptable — leaves the whole suite green.
    err = capsys.readouterr().err
    assert "closing _UnclosableSystem failed" in err
    assert "<unprintable _Unprintable>" in err  # and the repr fallback produced the detail


def test_main_does_not_require_the_system_to_be_closable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The close is duck-typed, so an arm without one must still run, and run SILENTLY.

    `MemorySystem` is deliberately a three-member protocol and `RecallSystem` has nothing to
    release — it holds a DSN string, not a connection. A `main` that assumed `close()` existed
    would break every such arm.

    `_recall_stub_build` pins the embedder to `hashing`, and that is load-bearing rather than
    copied habit: the real `_build_system` defaults to `fastembed`, an OPTIONAL extra, so letting
    it run makes this pass wherever that extra happens to be installed and fail in the `floor` CI
    job, which installs the declared minimums without extras. That is exactly how it did fail.
    Nothing here is about embedders.
    """
    _patch_recall_stub(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    # On the INSTANCE, which is what `main` probes. A class-level assert alone would still hold if
    # `__init__` ever bound a `self.close`, and the test would stop covering the closeless path
    # without ever going red.
    assert not hasattr(_recall_stub_build("recall", "m", _FAKE_KEY, 5, "stamp"), "close")

    data = _write_fixture(tmp_path)
    # Not merely "does not crash". Dropping the `getattr` guard while KEEPING the surrounding
    # try/except still returns 0, and every closeless arm would then report a teardown failure on
    # every successful run. So assert the breadcrumb is ABSENT.
    #
    # It is asserted on STDERR, and the promotion below is what puts it there. `_warn_teardown_failed`
    # catches `BaseException` around its own `warnings.warn` — deliberately, so `-W error` cannot
    # turn a diagnostic into a failed run — so promoting the warning does not raise out of `main`;
    # it diverts the message to the helper's stderr fallback. Promotion and stderr are therefore
    # one mechanism, not two, and checking either alone is checking half of it:
    #
    #   - promotion + `assert code == 0` alone: the mutant that keeps the try/except and drops
    #     only the `getattr` guard SURVIVES, because the promoted warning is swallowed and nothing
    #     observes it. (The cruder mutant, a bare `system.close()` with no try/except at all, is
    #     caught either way: its AttributeError propagates straight out of `main`.)
    #   - `record=True` + a recorded-warnings assert: works, but then the stderr channel is
    #     unreachable by construction, so a companion stderr assert is inert and kills no mutant.
    #
    # This form has one live assertion that covers both a warn-based reporter (via the fallback)
    # and any future one that writes stderr directly. Everything else stays ignored, so an
    # unrelated warning cannot fail this test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message="benchmarks.run: closing", category=UserWarning)

        # POSITIVE CONTROL, before relying on the breadcrumb's absence. The arming above depends
        # on two production details this test duplicates: the message prefix and the warning
        # category. If either drifts, the promotion stops matching, the fallback is never reached,
        # and the negative assert below passes VACUOUSLY — the test goes inert without going red,
        # which is exactly the failure it was rewritten to escape. So prove the mechanism fires
        # here, in this test, rather than trusting a neighbour to notice.
        run_module._warn_teardown_failed(object(), RuntimeError("probe"))
        armed = capsys.readouterr().err  # also drains, so the assert below sees only `main`
        assert "benchmarks.run: closing" in armed, (
            "the promotion is not armed: the message prefix or warning category has drifted, and "
            "the absence assertion below would prove nothing"
        )

        code = main(
            ["--arm", "recall", "--data", str(data), "--conversations", "2",
             "--out", str(tmp_path / "results")],
            now=_NOW,
        )

    assert code == 0
    assert "benchmarks.run: closing" not in capsys.readouterr().err


def test_main_runs_the_ablation_preflight_once_before_the_first_retrieve_and_stamps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preflight must fire exactly once — right after conv #1's ingest, strictly before its
    first `retrieve` — and its verdicts, the sample size and `--allow-inert-arm` must all land in
    the artifact, so a run that was let through with an inert mechanism cannot read as clean
    afterwards. `RecallSystem`'s own `ingest`/`retrieve`/`ablation_preflight` are monkeypatched off
    the class (no DB, no network) so this is offline like every other `main` test here; the
    `isinstance(system, RecallSystem)` guard in `main` is exactly why a REAL `RecallSystem` has to
    be built rather than the plain `_StubSys` fake the other tests use.
    """
    from benchmarks.systems import RecallSystem

    events: list[str] = []
    calls: list[tuple[list[str], int, str, bool]] = []

    def _fake_ingest(self: RecallSystem, conversation: dict[str, Any]) -> None:
        self._tenant = f"bench-{conversation['sample_id']}"
        events.append(f"ingest:{conversation['sample_id']}")

    def _fake_retrieve(self: RecallSystem, question: str) -> str:
        events.append(f"retrieve:{question}")
        return "ctx"

    def _fake_preflight(
        self: RecallSystem,
        questions: list[str],
        *,
        sample: int,
        metric_class: str,
        allow_inert: bool,
    ) -> list[dict[str, Any]]:
        calls.append((questions, sample, metric_class, allow_inert))
        events.append("preflight")
        return [{"mechanism": "sparse", "verdict": "DIFFERS", "sampled": len(questions), "differing": 1}]

    def _fake_build(
        arm: str,
        model: str,
        openrouter_key: str,
        k: int,
        run_id: str,
        embedder: str = "fastembed",
        **_extra: object,
    ) -> MemorySystem:
        return RecallSystem("postgresql://x/y", embedder_name="hashing", k=k)

    monkeypatch.setattr(RecallSystem, "ingest", _fake_ingest)
    monkeypatch.setattr(RecallSystem, "retrieve", _fake_retrieve)
    monkeypatch.setattr(RecallSystem, "ablation_preflight", _fake_preflight)
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    out = tmp_path / "results"
    code = main(
        [
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--conversations", "2",
            "--ablation-sample", "3",
            "--out", str(out),
        ],
        now=_NOW,
    )
    assert code == 0

    # exactly one preflight call: after conv-a's ingest, before conv-a's first retrieve — and
    # never again for conv-b
    assert events[:2] == ["ingest:conv-a", "preflight"]
    assert events.count("preflight") == 1
    assert events.index("preflight") < events.index("retrieve:What did Alice research?")

    assert len(calls) == 1
    questions, sample, metric_class, allow_inert = calls[0]
    # conv-a's surviving questions ONLY (not conv-b's), --ablation-sample threaded through, "set"
    # DECLARED (LOCOMO reports hit@k) rather than inferred, and the default --allow-inert-arm is
    # False
    assert questions == [
        "What did Alice research?",
        "How many siblings?",
        "What did Bob realise after his charity race?",
        "Did Bob make the black and white bowl in the photo?",
    ]
    assert sample == 3
    assert metric_class == "set"
    assert allow_inert is False

    payload = json.loads((out / f"{_STAMP_2CONV}.json").read_text(encoding="utf-8"))
    assert payload["ablation_preflight"] == {
        "verdicts": [
            {"mechanism": "sparse", "verdict": "DIFFERS", "sampled": 4, "differing": 1},
        ],
        "allow_inert_arm": False,
        "sample": 3,
        "ran": True,
    }


def test_main_stamps_allow_inert_arm_even_when_the_preflight_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mem0 arm never calls `ablation_preflight` (it is RE-call-only), but the artifact must
    still carry the key — an empty `verdicts` list plus the flag values — so a reader never has to
    guess whether a preflight ran from its absence.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    out = tmp_path / "results"
    code = main(
        [
            "--arm", "mem0",
            "--data", str(_write_fixture(tmp_path)),
            "--conversations", "1",
            "--allow-inert-arm",
            "--out", str(out),
        ],
        now=_NOW,
    )
    assert code == 0

    [artifact] = out.glob("mem0_*.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["ablation_preflight"] == {
        "verdicts": [],
        "allow_inert_arm": True,
        "sample": 25,  # the documented default
        "ran": False,
    }


def test_main_records_the_configuration_that_produced_the_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifact must identify its own run: budget, model, prompts, temperature, versions.

    Every number in the file is a function of these settings — change `k`, the generator prompt or
    the Mem0 release and the same code produces different results — so an artifact carrying only
    the arm name and the model string documents a run nobody can reproduce, including its author.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch)
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    out = tmp_path / "results"
    code = main(
        [
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--conversations", "1",
            "--k", "11",
            "--out", str(out),
        ],
        now=_NOW,
    )
    assert code == 0

    payload = json.loads((out / f"{_STAMP_1CONV}.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["k"] == 11
    assert config["model"] == "openai/gpt-4o-mini"
    assert config["temperature"] == 0.0
    assert config["base_url"] == "https://openrouter.ai/api/v1"
    # both prompts VERBATIM: a re-scorer has to be able to reproduce the judge, not paraphrase it
    assert config["gen_system_prompt"] == GEN_SYSTEM_PROMPT
    assert config["judge_system_prompt"] == JUDGE_SYSTEM_PROMPT
    # tolerated as None when the bench extra is not installed — but the key is always present
    assert "mem0ai_version" in config
    # the loader's drops are published beside n, so a headline n is explainable
    assert payload["skipped_questions"]["total"] == 3
    # and the retrieved-context volume is reported, since the arms do not retrieve the same amount
    ctx = payload["aggregate"]["retrieved_context"]
    assert ctx["n"] == 4
    assert ctx["chars"]["mean"] == 3.0  # every stub context is "ctx"
    assert ctx["tokens_approx"]["median"] == 1.0


def test_main_threads_k_to_the_adapters_and_defaults_to_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--k` reaches both adapters, and omitting it keeps the documented default of 5."""
    from benchmarks.systems import RecallSystem

    seen: list[int] = []

    def _recording_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed", **_extra: object
    ) -> MemorySystem:
        seen.append(k)
        return RecallSystem("postgresql://x/y", embedder_name="hashing", k=k)

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch)
    monkeypatch.setattr(run_module, "_build_system", _recording_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    argv = [
        "--arm", "recall",
        "--data", str(_write_fixture(tmp_path)),
        "--conversations", "1",
        "--out", str(tmp_path / "results"),
    ]
    assert main(argv, now=_NOW) == 0
    assert main([*argv, "--k", "17"], now=_NOW + timedelta(seconds=1)) == 0
    assert seen == [DEFAULT_K, 17]
    assert DEFAULT_K == 5  # the default is 5, and it is not to be changed silently


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_main_rejects_a_non_positive_conversation_count(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--conversations 0` sliced the dataset to nothing; a negative sliced from the END.

    Both produced a complete-looking artifact for a slice nobody asked for, so they are rejected
    at the parser rather than measured.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    with pytest.raises(SystemExit):
        main(
            [
                "--arm", "recall",
                "--data", str(_write_fixture(tmp_path)),
                "--conversations", bad,
                "--out", str(tmp_path / "results"),
            ],
            now=_NOW,
        )


def test_main_persists_each_conversation_before_starting_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash part-way through a run must lose at most the conversation in flight.

    Every question already scored has been paid for — a generator call, plus a judge call if it
    was answerable — so a harness that only writes after the last conversation throws that money
    away when conversation 7 of 10 hits a rate limit. Here the fake system raises on the SECOND
    conversation's `ingest`, i.e. after the first conversation is fully scored and before any of
    the second's work exists; the first conversation's outcomes must already be on disk.
    """

    from benchmarks.systems import RecallSystem

    def _crashing_ingest(self: RecallSystem, conversation: dict[str, Any]) -> None:
        if conversation["sample_id"] == "conv-b":
            raise RuntimeError("simulated rate limit part-way through the run")

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch)
    monkeypatch.setattr(RecallSystem, "ingest", _crashing_ingest)
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    out = tmp_path / "results"
    with pytest.raises(RuntimeError):
        main(
            [
                "--arm", "recall",
                "--data", str(_write_fixture(tmp_path)),
                "--conversations", "2",
                "--out", str(out),
            ],
            now=_NOW,
        )

    # the run died, so the aggregate was never written — the sidecar is the only survivor
    assert not (out / f"{_STAMP_2CONV}.json").exists()
    lines = (out / f"{_STAMP_2CONV}.partial.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["question_id"] for r in records] == ["conv-a:0", "conv-a:2", "conv-a:3", "conv-a:4"]
    # and the survivors are full records, not stubs: re-scorable without re-paying
    assert records[0]["question"] == "What did Alice research?"
    assert records[0]["context"] == "ctx"
    assert records[0]["answer"] == "an answer"
    assert records[0]["correct"] is True


def test_main_timestamps_the_filenames_so_a_rerun_cannot_clobber_a_published_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same arm, same model, same slice, twice — two artifacts, not one overwritten one.

    The stem used to be `{arm}_{model}_{N}conv`, so the second run silently replaced a results
    file that cost real money and may already have been linked from the article.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch)
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    out = tmp_path / "results"
    argv = [
        "--arm", "recall",
        "--data", str(_write_fixture(tmp_path)),
        "--conversations", "1",
        "--out", str(out),
    ]
    assert main(argv, now=_NOW) == 0
    assert main(argv, now=_NOW + timedelta(seconds=1)) == 0

    assert sorted(p.name for p in out.glob("*.json")) == [
        f"{_STAMP_1CONV}.json",
        "recall_openai-gpt-4o-mini_1conv_20260102T030406Z.json",
    ]
    # the sidecars are stamped alongside them, so the second run does not append to the first's
    assert len(list(out.glob("*.partial.jsonl"))) == 2


def test_main_dump_has_no_nan_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`NaN` is not valid JSON for any non-Python parser, and empty rate blocks are the risk.

    A single-conversation slice with an adversarial-only category leaves `cat5-adversarial`'s
    answerable sub-block built from an empty list — exactly where a bare `NaN` would appear.
    """

    def _fake_complete(self: OpenRouterLLM, system: str, user: str) -> str:
        return "NO_ANSWER"

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch, retrieve_returns="")
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _fake_complete)

    out = tmp_path / "results"
    main(
        [
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--conversations", "1",
            "--out", str(out),
        ],
        now=_NOW,
    )
    dumped = (out / f"{_STAMP_1CONV}.json").read_text(encoding="utf-8")
    assert "NaN" not in dumped
    payload = json.loads(dumped)
    cat5 = payload["aggregate"]["by_category"]["cat5-adversarial"]
    assert cat5["answerable_accuracy"]["rate"] is None
    assert cat5["adversarial_abstention"]["rate"] == 1.0


# --- key validation -------------------------------------------------------------------------
# Regression tests for a real incident: `export OPENROUTER_API_KEY=...` was pasted verbatim, so
# the variable held the three characters "...". A mere is-it-set check passed, the recall arm ran
# all the way to its first generator call, and only then died on a 401. On the mem0 arm the same
# mistake would fail LATER still — after per-session LLM extraction during ingest had already
# been paid for. These pin the shape check that turns that into an immediate, free failure.


def test_validate_openrouter_key_accepts_a_real_looking_key() -> None:
    key = "sk-or-v1-" + "a" * 64
    assert validate_openrouter_key(key) == key


def test_validate_openrouter_key_rejects_unset_and_blank() -> None:
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="not set"):
            validate_openrouter_key(bad)


def test_validate_openrouter_key_rejects_the_placeholder_that_passed_an_is_set_check() -> None:
    with pytest.raises(ValueError, match="OpenRouter key"):
        validate_openrouter_key("...")


def test_validate_openrouter_key_rejects_embedded_whitespace() -> None:
    # a stray newline or quote from the shell, e.g. a key sourced from a CRLF file
    with pytest.raises(ValueError, match="whitespace"):
        validate_openrouter_key("sk-or-v1-abc\n")


def test_validate_openrouter_key_error_never_echoes_the_key() -> None:
    # the message goes to stderr and into logs; a wrong-prefix key is still a secret
    secret = "sk-XX-supersecretvalue"
    with pytest.raises(ValueError) as excinfo:
        validate_openrouter_key(secret)
    message = str(excinfo.value)
    assert secret not in message
    assert "supersecret" not in message


def test_a_rejected_cost_claim_quarantines_the_run_instead_of_losing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A contract raise must not destroy the artifact it refuses to publish.

    The scored answers themselves are not what is at risk: they are already in the incremental
    sidecar and `benchmarks.salvage --merge-only` rebuilds a runnable artifact from it without
    re-spending. What a discarded artifact costs is the envelope the sidecar does NOT hold, the
    usage block, `provider_metadata`, the config and the ablation preflight, plus a manual
    salvage step. The contract's job is to keep an unauditable artifact out of the PUBLISHED
    filename, not to throw that envelope away.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch)
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    def _reject(payload: dict[str, Any]) -> None:
        raise ValueError("benchmark cost claims require provider_metadata")

    monkeypatch.setattr(run_module, "reject_unauditable_cost_claims", _reject)

    out = tmp_path / "results"
    code = main(
        ["--arm", "recall", "--data", str(_write_fixture(tmp_path)),
         "--conversations", "1", "--out", str(out)],
        now=_NOW,
    )

    # The exact code, not merely non-zero: 1 is what the interpreter itself exits with on an
    # uncaught exception, so a wrapper could not tell "quarantined, salvageable, do NOT re-run"
    # from "crashed mid-run". A mutant returning 2 also collides with argparse's usage error.
    assert code == run_module.QUARANTINE_EXIT == 3
    # Nothing published: the operator's `results/*.json` glob feeds `analyze`, so a quarantined
    # artifact must not sit beside real ones under a name that glob would pick up.
    assert list(out.glob("*.json")) == []
    quarantined = out / "unpublished" / f"{_STAMP_1CONV}.json"
    assert quarantined.exists(), "the artifact was destroyed rather than quarantined"

    recovered = json.loads(quarantined.read_text(encoding="utf-8"))
    assert recovered["arm"] == "recall"
    assert recovered["questions"] == len(recovered["outcomes"]) > 0
    assert recovered["aggregate"]
    # In band marker: handed this file directly, a reader must be able to tell it was refused.
    # Byte-identical to a publishable artifact, it is indistinguishable from one.
    assert recovered["unpublished"] is True
    assert "provider_metadata" in recovered["unpublished_reason"]
    # the incremental sidecar is untouched by the rejection
    assert (out / f"{_STAMP_1CONV}.partial.jsonl").exists()

    # The operator-facing diagnostics are the whole point of the rescue; without asserting them
    # a mutant that quarantines in total silence stays green.
    err = capsys.readouterr().err
    assert str(quarantined) in err
    assert "salvage" in err
    assert "provider_metadata" in err


def test_a_rejected_cost_claim_still_releases_the_arms_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure path releases handles too, and only after the run is safely on disk."""
    from benchmarks.systems import RecallSystem

    out = tmp_path / "results"
    closed: list[bool] = []

    class _ClosableSystem(RecallSystem):
        name = "recall"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def ablation_preflight(
            self, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            # Records whether the quarantined artifact was ALREADY on disk, so a mutant that
            # releases before the rescue write does not survive.
            closed.append((out / "unpublished" / f"{_STAMP_1CONV}.json").exists())

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str,
        embedder: str = "fastembed", **_extra: object,
    ) -> MemorySystem:
        return _ClosableSystem("postgresql://x/y", embedder_name="hashing", k=k)

    def _reject(payload: dict[str, Any]) -> None:
        raise ValueError("benchmark cost claims require provider_metadata")

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)
    monkeypatch.setattr(run_module, "reject_unauditable_cost_claims", _reject)

    code = main(
        ["--arm", "recall", "--data", str(_write_fixture(tmp_path)),
         "--conversations", "1", "--out", str(out)],
        now=_NOW,
    )

    assert code == run_module.QUARANTINE_EXIT
    assert closed == [True]  # released exactly once, and AFTER the rescue write


def test_an_unserialisable_payload_still_releases_the_arms_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`json.dumps` is the other raise at the write site, and it must not skip teardown either."""
    from benchmarks.systems import RecallSystem

    out = tmp_path / "results"
    closed: list[str] = []

    class _ClosableSystem(RecallSystem):
        name = "recall"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def ablation_preflight(
            self, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            closed.append("released")

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str,
        embedder: str = "fastembed", **_extra: object,
    ) -> MemorySystem:
        return _ClosableSystem("postgresql://x/y", embedder_name="hashing", k=k)

    real_payload = run_module._results_payload

    def _unserialisable_payload(*args: object, **kwargs: object) -> dict[str, Any]:
        # A real payload carrying one member json cannot encode. Patching `json.dumps` itself
        # would not test the write site: `json` is the shared module object, so the stub would
        # fire earlier in the run and never reach the code under test.
        payload = real_payload(*args, **kwargs)  # type: ignore[arg-type]
        payload["config"] = {"unserialisable": {object()}}
        return payload

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)
    monkeypatch.setattr(run_module, "_results_payload", _unserialisable_payload)

    with pytest.raises(TypeError):
        main(
            ["--arm", "recall", "--data", str(_write_fixture(tmp_path)),
             "--conversations", "1", "--out", str(out)],
            now=_NOW,
        )

    assert closed == ["released"]


def _quarantine_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    out: Path,
    *,
    reject: bool,
) -> int:
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch)
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)
    if reject:
        def _reject(payload: dict[str, Any]) -> None:
            raise ValueError("benchmark cost claims require provider_metadata")

        monkeypatch.setattr(run_module, "reject_unauditable_cost_claims", _reject)
    return main(
        ["--arm", "recall", "--data", str(_write_fixture(tmp_path)),
         "--conversations", "1", "--out", str(out)],
        now=_NOW,
    )


def test_the_rescue_falls_back_when_the_quarantine_directory_cannot_be_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`mkdir(exist_ok=True)` raises FileExistsError when the path exists as a FILE.

    A rescue that can itself throw away the artifact is not a rescue, so it falls back to a
    sibling name rather than letting the OSError escape.
    """
    out = tmp_path / "results"
    out.mkdir(parents=True)
    (out / "unpublished").write_text("not a directory", encoding="utf-8")

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    fallback = out / f"{_STAMP_1CONV}.unpublished.json.txt"
    assert fallback.exists(), "the artifact was lost when the subdirectory was blocked"
    assert json.loads(fallback.read_text(encoding="utf-8"))["unpublished"] is True
    assert str(fallback) in capsys.readouterr().err
    # `.json.txt`, not `.json`: the fallback has to stay out of the publishable glob just as
    # firmly as the subdirectory does, or the rescue hands `analyze` a refused artifact.
    assert list(out.glob("*.json")) == []


def test_a_failed_published_write_quarantines_the_artifact_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The published write is likelier to fail than the contract is to fire.

    A full disk, a long path or an antivirus lock all raise OSError there, and the encoded bytes
    are already in hand, so losing them would be gratuitous.
    """
    out = tmp_path / "results"
    out.mkdir(parents=True)
    (out / f"{_STAMP_1CONV}.json").mkdir()  # a directory where the artifact wants to go

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=False)

    assert code == run_module.QUARANTINE_EXIT
    quarantined = out / "unpublished" / f"{_STAMP_1CONV}.json"
    assert quarantined.exists(), "a failed publish threw the artifact away"
    recovered = json.loads(quarantined.read_text(encoding="utf-8"))
    assert recovered["unpublished"] is True
    assert recovered["aggregate"]
    assert "could not be written" in capsys.readouterr().err.lower()


class _BrokenStream:
    """A stream that refuses everything, the way a closed pipe does."""

    def write(self, _data: str) -> int:
        raise OSError(32, "Broken pipe")

    def flush(self) -> None:
        raise OSError(32, "Broken pipe")


def test_the_rescue_writes_the_artifact_before_it_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable stderr must not be able to destroy the artifact the rescue exists to save.

    Reporting touches a stream this process does not own. Writing touches a directory it was
    given. Doing them in that order, and guarding the reports, is what makes the docstring's
    "never raises" true rather than aspirational.
    """
    import sys as _sys

    out = tmp_path / "results"
    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)
    assert code == run_module.QUARANTINE_EXIT  # sanity: the healthy-stderr baseline

    out2 = tmp_path / "results2"
    monkeypatch.setattr(_sys, "stderr", _BrokenStream())
    code2 = _quarantine_case(tmp_path, monkeypatch, out2, reject=True)

    assert code2 == run_module.QUARANTINE_EXIT
    assert (out2 / "unpublished" / f"{_STAMP_1CONV}.json").exists()


def test_the_rescue_falls_back_to_stdout_when_no_file_can_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both candidates blocked: the body still has to reach somewhere, and it is still exit 3."""
    out = tmp_path / "results"
    real_write = Path.write_text

    def _refuse(self: Path, data: str, *a: object, **k: object) -> int:
        # `.tmp` too: the rescue writes atomically, so blocking only the final names would
        # block nothing at all.
        if "unpublished" in str(self):
            raise OSError(28, "No space left on device")
        return real_write(self, data, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _refuse)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    captured = capsys.readouterr()
    # stdout also carries the run's progress prose, including a dict repr with single quotes,
    # so anchor on the indented JSON opening rather than on the first brace.
    body = captured.out[captured.out.rindex('{\n  "') :]
    assert json.loads(body)["unpublished"] is True
    assert "stdout" in captured.err


def test_a_partially_written_artifact_is_removed_from_the_published_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_text` opens with mode "w", so the file is truncated before a byte can fail.

    One truncated leftover in the results directory makes every later `analyze` over it die on a
    JSONDecodeError, so the corpse has to go.
    """
    out = tmp_path / "results"
    real_write = Path.write_text

    def _die_midway(self: Path, data: str, *a: object, **k: object) -> int:
        if self.name.startswith(_STAMP_1CONV) and self.parent.name == "results":
            real_write(self, data[:20], encoding="utf-8")  # truncated, like a full disk
            raise OSError(28, "No space left on device")
        return real_write(self, data, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _die_midway)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=False)

    assert code == run_module.QUARANTINE_EXIT
    assert list(out.glob("*.json")) == [], "a truncated artifact was left in the published glob"
    assert list(out.glob("*.tmp")) == [], "the atomic write left its temp behind"
    assert (out / "unpublished" / f"{_STAMP_1CONV}.json").exists()


def test_a_raising_teardown_report_cannot_change_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is wrapped because it runs inside a `finally` that now covers failure paths."""
    from benchmarks.systems import RecallSystem

    out = tmp_path / "results"

    class _ClosableSystem(RecallSystem):
        name = "recall"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def ablation_preflight(
            self, questions: list[str], *, sample: int, metric_class: str, allow_inert: bool
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            raise RuntimeError("teardown exploded")

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str,
        embedder: str = "fastembed", **_extra: object,
    ) -> MemorySystem:
        return _ClosableSystem("postgresql://x/y", embedder_name="hashing", k=k)

    def _report_explodes(subject: object, exc: BaseException) -> None:
        raise SystemError("even the reporter is broken")

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)
    monkeypatch.setattr(run_module, "_warn_teardown_failed", _report_explodes)

    code = main(
        ["--arm", "recall", "--data", str(_write_fixture(tmp_path)),
         "--conversations", "1", "--out", str(out)],
        now=_NOW,
    )

    assert code == 0, "a broken teardown reporter must not change a finished run's verdict"
    assert (out / f"{_STAMP_1CONV}.json").exists()


def test_the_rescue_never_leaks_diagnostics_into_stdout_when_stderr_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sys.stderr` really is None under pythonw, and `print(file=None)` writes to STDOUT.

    Unguarded, every diagnostic line would land on stdout — which is where the last-resort
    artifact body goes — so a reader parsing that stream would find prose wrapped around the
    JSON. `_say` refuses a None stream instead, and that refusal is also what keeps its return
    value honest about whether anything was preserved.
    """
    import sys as _sys

    out = tmp_path / "results"
    monkeypatch.setattr(_sys, "stderr", None)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    assert (out / "unpublished" / f"{_STAMP_1CONV}.json").exists()
    assert "artifact NOT published" not in capsys.readouterr().out


def test_say_refuses_a_stream_that_is_explicitly_none() -> None:
    """`None` must mean "this stream is unusable", never "use the default".

    Conflating the two is how the last resort ends up writing the artifact body to stderr while
    reporting it went to stdout and counting it as preserved.
    """
    assert run_module._say("x", stream=None) is False
    assert run_module._say("x") is True  # no argument still defaults to stderr


def test_say_lets_a_keyboard_interrupt_through() -> None:
    """Swallowing Ctrl+C during the multi-MB last-resort dump would report a total loss.

    A broken stream is an expected failure and is swallowed; an interrupt is the operator
    talking, and must not be turned into "the artifact could not be preserved anywhere".
    """

    class _Interrupting:
        def write(self, _data: str) -> int:
            raise KeyboardInterrupt

        def flush(self) -> None:
            pass

    assert run_module._say("x", stream=_BrokenStream()) is False  # OSError: swallowed
    with pytest.raises(KeyboardInterrupt):
        run_module._say("x", stream=_Interrupting())


def test_a_run_whose_artifact_reaches_nowhere_reports_a_genuine_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 3 claims the work is salvageable. When nothing was saved, that claim is a lie.

    Both file candidates blocked AND stdout unusable: the honest answer is 1.
    """
    import sys as _sys

    out = tmp_path / "results"
    real_write = Path.write_text

    def _refuse(self: Path, data: str, *a: object, **k: object) -> int:
        if "unpublished" in str(self):
            raise OSError(28, "No space left on device")
        return real_write(self, data, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _refuse)
    # None, the real pythonw shape: `print()` to a None stdout is a silent no-op, so the run's
    # own progress lines still work and only the last resort is starved. A broken stream would
    # instead blow up on the first progress print, long before the write site.
    monkeypatch.setattr(_sys, "stdout", None)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == 1, "nothing was preserved, so this is a loss and not a quarantine"


def test_a_partially_written_quarantine_candidate_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpse rule applies to the rescue's own writes too.

    A truncated `<out>/unpublished/<stamp>.json` breaks any `results/**/*.json` reader, and
    `_load_published` parses before it can consult the mark, so the mark cannot rescue it.
    """
    out = tmp_path / "results"
    real_write = Path.write_text

    def _die_midway(self: Path, data: str, *a: object, **k: object) -> int:
        if self.parent.name == "unpublished":
            real_write(self, data[:30], encoding="utf-8")
            raise OSError(28, "No space left on device")
        return real_write(self, data, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _die_midway)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    assert not (out / "unpublished" / f"{_STAMP_1CONV}.json").exists(), "truncated corpse left"
    assert (out / f"{_STAMP_1CONV}.unpublished.json.txt").exists()


def test_a_failed_quarantine_write_does_not_destroy_a_pre_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup that guesses is worse than no cleanup.

    `write_text` truncates at open, so a corpse only exists when the failure comes AFTER open.
    When it comes AT open — an antivirus or indexer lock, EACCES, a read-only mount — the file
    sitting there is intact and somebody else's, and deleting it destroys a good artifact the
    rescue then does not replace.
    """
    out = tmp_path / "results"
    victim = out / "unpublished" / f"{_STAMP_1CONV}.json"
    victim.parent.mkdir(parents=True)
    victim.write_text('{"real": "a previous artifact"}', encoding="utf-8")

    real_write = Path.write_text

    def _locked(self: Path, data: str, *a: object, **k: object) -> int:
        if self.parent.name == "unpublished":
            raise PermissionError(13, "locked by another process")
        return real_write(self, data, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _locked)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    assert victim.read_text(encoding="utf-8") == '{"real": "a previous artifact"}'
    assert (out / f"{_STAMP_1CONV}.unpublished.json.txt").exists()


def test_a_failed_write_leaves_no_temporary_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic writes move the corpse problem into a `.tmp` that must be cleaned up itself."""
    out = tmp_path / "results"
    real_write = Path.write_text

    def _die_midway(self: Path, data: str, *a: object, **k: object) -> int:
        if self.parent.name == "unpublished":
            real_write(self, data[:30], encoding="utf-8")
            raise OSError(28, "No space left on device")
        return real_write(self, data, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _die_midway)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    assert list((out / "unpublished").glob("*")) == [], "a temp or a corpse was left behind"
    assert (out / f"{_STAMP_1CONV}.unpublished.json.txt").exists()


def test_an_interrupt_while_reporting_does_not_lose_a_preserved_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notes print AFTER the artifact is safe, so an interrupt there must not cost exit 3.

    Letting it propagate would hand a wrapper a crash status for a run that was preserved — the
    exact harm narrowing `_say` to `except Exception` was meant to prevent, inverted.
    """
    import sys as _sys

    class _Interrupting:
        def write(self, _data: str) -> int:
            raise KeyboardInterrupt

        def flush(self) -> None:
            pass

    out = tmp_path / "results"
    monkeypatch.setattr(_sys, "stderr", _Interrupting())

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    assert (out / "unpublished" / f"{_STAMP_1CONV}.json").exists()


def test_one_interrupt_while_reporting_still_names_where_the_artifact_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single Ctrl+C must cost one diagnostic line, not all of them.

    Guarding the whole loop instead of each note means one interrupt swallows every note,
    including the only one that tells the operator where the artifact is — silence in place of
    the wrong path the previous round was fixing.
    """
    import sys as _sys

    class _InterruptsOnce:
        def __init__(self) -> None:
            self.seen: list[str] = []
            self.armed = True

        def write(self, data: str) -> int:
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt
            self.seen.append(data)
            return len(data)

        def flush(self) -> None:
            return None

    stream = _InterruptsOnce()
    out = tmp_path / "results"
    monkeypatch.setattr(_sys, "stderr", stream)

    code = _quarantine_case(tmp_path, monkeypatch, out, reject=True)

    assert code == run_module.QUARANTINE_EXIT
    assert any("unpublished  ->" in line for line in stream.seen), (
        "one interrupt swallowed every note, including the artifact's location"
    )


def test_a_second_run_of_the_same_stem_refuses_before_it_spends_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_run_stamp` resolves to the SECOND, so two replicates of one arm can share a whole stem.

    Sharing it is not a cosmetic clash. Both would `os.replace` onto the same
    `<stamp>.json`, so the loser's complete, paid-for measurement is destroyed silently with
    exit 0 and a `full results ->` line naming a file holding the other run's data. Both would
    also APPEND to one `<stamp>.partial.jsonl`, and `salvage --merge-only` would then rebuild a
    single published artifact out of two interleaved runs — which `consistency_report` cannot
    detect, because two replicates agree on arm, model and k.

    The stem is therefore claimed atomically, and the refusal lands before the first token.
    """
    out = tmp_path / "results"
    out.mkdir(parents=True)
    (out / f"{_STAMP_1CONV}.partial.jsonl").write_text("", encoding="utf-8")

    calls: list[str] = []

    def _counting_complete(self: object, system: str, user: str) -> str:
        calls.append(user)
        return _stub_complete(self, system, user)  # type: ignore[arg-type]

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    _patch_recall_stub(monkeypatch)
    monkeypatch.setattr(run_module, "_build_system", _recall_stub_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _counting_complete)

    with pytest.raises(SystemExit, match="already exists"):
        main(
            ["--arm", "recall", "--data", str(_write_fixture(tmp_path)),
             "--conversations", "1", "--out", str(out)],
            now=_NOW,
        )

    assert calls == [], "the refusal must come before a single generator or judge call"
