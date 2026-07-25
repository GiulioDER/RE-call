"""The run harness: `run_arm` scoring, and `_load`'s LOCOMO -> question-list translation.

Everything here is offline. `run_arm` is driven with a fake system and a fake completer (no DB,
no network, no spend), and `_load` is driven with a small inline LOCOMO-shaped fixture written to
a temp file rather than the real `locomo10.json` — so the assertions below pin the translation
rules (which questions survive, how `adversarial` is derived, how ids are formed) instead of
restating whatever the shipped dataset happens to contain.
"""
from __future__ import annotations

import json
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


def _stub_build(arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed") -> MemorySystem:
    return _StubSys()


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
    events: list[tuple[str, str]] = []

    class _OrderedSys:
        name = "ordered"

        def ingest(self, conversation: dict[str, Any]) -> None:
            events.append(("ingest", str(conversation["sample_id"])))

        def retrieve(self, question: str) -> str:
            events.append(("retrieve", question))
            return "ctx"

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed"
    ) -> MemorySystem:
        return _OrderedSys()

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
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


def test_main_records_the_configuration_that_produced_the_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifact must identify its own run: budget, model, prompts, temperature, versions.

    Every number in the file is a function of these settings — change `k`, the generator prompt or
    the Mem0 release and the same code produces different results — so an artifact carrying only
    the arm name and the model string documents a run nobody can reproduce, including its author.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _stub_build)
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
    seen: list[int] = []

    def _recording_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed"
    ) -> MemorySystem:
        seen.append(k)
        return _StubSys()

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
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

    class _CrashingSys:
        name = "crashing"

        def ingest(self, conversation: dict[str, Any]) -> None:
            if conversation["sample_id"] == "conv-b":
                raise RuntimeError("simulated rate limit part-way through the run")

        def retrieve(self, question: str) -> str:
            return "ctx"

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed"
    ) -> MemorySystem:
        return _CrashingSys()

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
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
    monkeypatch.setattr(run_module, "_build_system", _stub_build)
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

    class _EmptySys:
        name = "empty"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return ""

    def _fake_build(
        arm: str, model: str, openrouter_key: str, k: int, run_id: str, embedder: str = "fastembed"
    ) -> MemorySystem:
        return _EmptySys()

    def _fake_complete(self: OpenRouterLLM, system: str, user: str) -> str:
        return "NO_ANSWER"

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(run_module, "_build_system", _fake_build)
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
