"""A generate-only BEAM run must be judgeable LATER, from the artifact alone.

Prior work: searched with ``docs_search(source_type="memory", ...)``.
[[project-recall-beam-bestconfig-blocked-2026-07-28]] records the run these guards exist for: its
runner lived in ``/tmp``, ``/tmp`` was cleared by a reboot, and the 5 questions it had already paid
for became unusable because nothing recorded their configuration. No prior test covered either
behaviour.

The claim under test is a MONEY claim: the answers are the expensive half of a BEAM run, so a
generate-only artifact is only worth producing if the scoring decision can genuinely be deferred.
That is true precisely because BEAM's judge reads ``(question, nugget, answer)`` and never the
retrieved context — which is a property of `get_beam_nugget_judge_prompt`, not a promise, so it is
asserted here rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.beam.dataset import Question
from benchmarks.beam.run import (
    _check_resume_config,
    _config_sidecar_for,
    _run_config,
    _write_run_config,
    aggregate,
    judge_answer,
    score_question,
)

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def _question() -> Question:
    return Question(
        question_id="1M_0_q0_test",
        chat_size="1M",
        conversation_idx=0,
        question_type="information_extraction",
        question="Where did the user study psychology?",
        rubric=["States the user studied at Leiden", "Mentions the year 2019"],
        difficulty="easy",
    )


def _memories() -> list[dict[str, str]]:
    return [
        {"memory": "I did my psychology degree at Leiden.", "created_at": "2019-04-02"},
        {"memory": "Graduated in 2019.", "created_at": "2019-09-11"},
        {"memory": "Unrelated: I like cycling.", "created_at": "2020-01-05"},
    ]


class _Recorder:
    """A Completer that records every prompt it is handed and returns a fixed reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_generate_only_row_can_be_judged_later_without_regeneration() -> None:
    """The whole point: judge the STORED answer, never call the answerer again."""
    answerer = _Recorder("ANSWER: Leiden, in 2019.")
    row = score_question(_question(), _memories(), answerer, None, cutoff=45)

    assert row["judged"] is False
    assert row["judgment"] == "UNJUDGED"
    # CONTRACT CHANGE, finding DAT-009: an unjudged row now carries None, not NaN. `json.dumps`
    # emits a bare `NaN` token that is not valid JSON, and with --no-judge that was every row of
    # an artifact whose purpose is to be read by something other than Python.
    assert row["score"] is None, "an unjudged row carries None (valid JSON), not NaN"
    assert json.dumps(row, allow_nan=False), "the row must be STRICT-JSON serialisable"
    assert len(answerer.prompts) == 1

    # Round-trip through a serialised artifact: the deferred judge gets a FILE, not live objects.
    revived = json.loads(json.dumps(row))
    judge = _Recorder(json.dumps({"score": 1.0, "reason": "stated"}))
    mean, nuggets, errors = judge_answer(
        Question(
            question_id=revived["question_id"],
            chat_size="1M",
            conversation_idx=0,
            question_type=revived["question_type"],
            question=revived["question"],
            rubric=revived["rubric"],
            difficulty=revived["difficulty"],
        ),
        revived["generated_answer"],
        judge,
    )

    assert errors == 0
    assert mean == 1.0
    assert len(nuggets) == 2, "one judge call per rubric nugget, as upstream does"
    # The answerer was NOT called again. This is the assertion that the deferral is free.
    assert len(answerer.prompts) == 1


def test_judge_prompt_never_receives_the_retrieved_context() -> None:
    """Why deferral is free at all. If this fails, a stored answer is no longer judgeable alone."""
    answerer = _Recorder("ANSWER: Leiden, in 2019.")
    row = score_question(_question(), _memories(), answerer, None, cutoff=45)
    judge = _Recorder(json.dumps({"score": 1.0, "reason": "ok"}))
    judge_answer(_question(), row["generated_answer"], judge)

    assert judge.prompts, "the judge must have been called, or this proves nothing"
    for prompt in judge.prompts:
        for memory in _memories():
            assert memory["memory"] not in prompt


def test_generated_row_persists_the_evidence_not_just_its_count() -> None:
    answerer = _Recorder("ANSWER: Leiden.")
    row = score_question(_question(), _memories(), answerer, None, cutoff=2)

    assert row["memories_evaluated"] == 2
    assert [m["memory"] for m in row["memories"]] == [
        "I did my psychology degree at Leiden.",
        "Graduated in 2019.",
    ]
    # `context` is the prompt actually sent, so a later change to the prompt builder cannot
    # reinterpret this artifact.
    assert row["context"] == answerer.prompts[0]
    assert "Leiden" in row["context"]


def test_unjudged_rows_are_not_counted_as_judge_errors() -> None:
    """The failure this guards: 300 deliberately unjudged rows reading as a total judge outage."""
    answerer = _Recorder("ANSWER: something")
    rows = [score_question(_question(), _memories(), answerer, None, cutoff=45) for _ in range(3)]
    metrics = aggregate(rows)

    assert metrics["overall"]["unjudged"] == 3
    assert metrics["overall"]["errors"] == 0
    assert metrics["overall"]["n"] == 0


def test_judge_independent_signals_survive_a_generate_only_run() -> None:
    """Abstention needs no judge, so a generate-only run must still report it."""
    answerer = _Recorder("I don't have enough information to answer this question.")
    row = score_question(_question(), _memories(), answerer, None, cutoff=45)
    metrics = aggregate([row])

    assert row["abstained"] is True
    # The pre-existing block is judge-gated and correctly reports nothing here...
    assert metrics["false_abstain"]["rate"] is None
    # ...while the all-rows block still holds the fact.
    assert metrics["retrieval_all_rows"]["false_abstain"]["rate"] == 1.0
    assert metrics["retrieval_all_rows"]["n"] == 1


def test_aggregate_is_unchanged_for_rows_without_the_judged_field() -> None:
    """Backwards compatibility: artifacts written before `judged` existed must not shift."""
    legacy = [
        {"question_type": "information_extraction", "score": 1.0, "abstained": False, "retrieval_empty": False},
        {"question_type": "abstention", "score": 0.0, "abstained": True, "retrieval_empty": False},
        {"question_type": "information_extraction", "score": float("nan"), "abstained": False, "retrieval_empty": False},
    ]
    metrics = aggregate(legacy)

    assert metrics["overall"]["n"] == 2
    assert metrics["overall"]["errors"] == 1, "a genuine judge error is still an error"
    assert metrics["overall"]["unjudged"] == 0


class _FakeSystem:
    def __init__(self, **described: object) -> None:
        self._described = described

    def describe(self) -> dict[str, object]:
        return dict(self._described)


class _Args:
    chat_size = "1M"
    model = "openai/gpt-5"
    judge_model = None
    no_judge = True
    k = 45
    cutoff = 45
    question_types = None
    # `_run_config` reads this UNGUARDED, deliberately. `getattr(args, "conversations", None)`
    # would let an args object that never carried the flag record "all" and sail through the
    # resume comparison — the silent cross-selection merge the key exists to catch. So the double
    # carries it, and an AttributeError here is the correct failure rather than a defaulted pass.
    conversations = None


def _system(tenant: str, judged: int) -> _FakeSystem:
    return _FakeSystem(
        candidate_k=250,
        embedder={"name": "voyage:voyage-4-large", "model": "voyage-4-large"},
        reranker="local",
        entailment={"guard": "off"},
        table="bench_beam_voyage",
        # Both of these move DURING a run and must not be part of the comparison.
        tenant=tenant,
        entailment_counters={"candidates_judged": judged},
    )


def test_run_config_ignores_state_that_changes_within_a_run() -> None:
    """A guard that fires on per-conversation state would fire on every resume, so be disabled."""
    first = _run_config(_Args(), _system("beam-conv-0", 0))
    later = _run_config(_Args(), _system("beam-conv-9", 4211))

    assert first == later
    assert "tenant" not in first
    assert first["embedder_model"] == "voyage-4-large"
    assert first["candidate_k"] == 250
    assert first["no_judge"] is True
    # Covered, not merely tolerated. `--conversations` decides WHICH rows a run produces, so its
    # absence from this config let a resume across a changed selection merge two populations
    # while `coverage` — which measures only the shortfall — still reported complete.
    assert first["conversations"] == "all"


def test_resume_refuses_when_the_configuration_differs(tmp_path: Path) -> None:
    sidecar = tmp_path / "beam1M_recall_x.partial.jsonl"
    sidecar.write_text("", encoding="utf-8")

    old = _run_config(_Args(), _system("beam-conv-0", 0))
    old["embedder_model"] = "text-embedding-3-small"
    _write_run_config(_config_sidecar_for(sidecar), old)

    current = _run_config(_Args(), _system("beam-conv-0", 0))
    with pytest.raises(SystemExit) as excinfo:
        _check_resume_config([sidecar], current, allow=False)

    message = str(excinfo.value)
    assert "embedder_model" in message
    assert "text-embedding-3-small" in message


def test_resume_refuses_when_the_configuration_is_simply_absent(tmp_path: Path) -> None:
    """'Cannot verify' must not pass as 'verified' — the state the blocked run was left in."""
    sidecar = tmp_path / "beam1M_recall_y.partial.jsonl"
    sidecar.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _check_resume_config([sidecar], _run_config(_Args(), _system("t", 0)), allow=False)

    assert "cannot be compared" in str(excinfo.value)


def test_resume_accepts_a_matching_configuration(tmp_path: Path) -> None:
    """The guard must also be able to PASS, or it trains you to override it."""
    sidecar = tmp_path / "beam1M_recall_z.partial.jsonl"
    sidecar.write_text("", encoding="utf-8")
    current = _run_config(_Args(), _system("beam-conv-0", 0))
    _write_run_config(_config_sidecar_for(sidecar), current)

    _check_resume_config([sidecar], current, allow=False)


def test_override_downgrades_the_refusal_to_a_warning(tmp_path: Path, capsys) -> None:
    sidecar = tmp_path / "beam1M_recall_w.partial.jsonl"
    sidecar.write_text("", encoding="utf-8")

    _check_resume_config([sidecar], _run_config(_Args(), _system("t", 0)), allow=True)

    assert "WARNING" in capsys.readouterr().out


def test_redacted_database_never_leaks_a_credential() -> None:
    """The value is persisted to the sidecar AND the artifact, so it must fail CLOSED.

    The first version used `urlsplit` alone, which returns the whole libpq KEYWORD form as `path`
    — so `password=...` was written verbatim into two files. psycopg accepts that form, so it is
    not a hypothetical input.
    """
    from benchmarks.beam.run import _redacted_database

    keyword = "host=10.0.0.1 port=5432 dbname=recall user=u password=sup3rs3cret"
    assert "sup3rs3cret" not in _redacted_database(keyword)
    assert _redacted_database(keyword) == "10.0.0.1:5432/recall"
    assert "pw" not in _redacted_database("postgresql://u:pw@h:5432/db")
    # Anything unparseable becomes a marker, never the raw string.
    assert _redacted_database("not a dsn at all") == "<unparsed>"
    assert _redacted_database("") == ""


# ---------------------------------------------------------------------------
# Flags the mem0 arm cannot honour. Neither refusal had a test — including the
# `--no-judge` one, which shipped earlier on exactly this reasoning.
# ---------------------------------------------------------------------------


class _Mem0Args:
    """Only the fields the refusal reads. Deliberately minimal: the check must not need a key,
    an artifact, or a populated namespace to deliver a verdict about the flags alone."""

    no_judge = False
    question_types = None
    # Carried so production can read `args.dry_run` UNGUARDED, like its two siblings. Omitting it
    # is what forced a `getattr(..., False)` into the guard — the one accessor that fails OPEN, on
    # the strongest of the three spending refusals. The double adapts to the standard; the
    # standard does not bend to the double.
    dry_run = False


def test_the_mem0_arm_refuses_no_judge_rather_than_judging_anyway() -> None:
    from benchmarks.beam.run import _refuse_flags_this_arm_ignores

    args = _Mem0Args()
    args.no_judge = True
    with pytest.raises(SystemExit, match="(?i)no-judge is meaningless"):
        _refuse_flags_this_arm_ignores(args)


def test_the_mem0_arm_refuses_question_types_rather_than_charging_for_the_full_run() -> None:
    """RED before this change: the flag was accepted and every published answer was judged.

    The cost is the point. `--question-types temporal-reasoning` reads as "judge a subset", and
    the arm has no filter for it, so the operator paid for ~700 judgements believing they had
    asked for a fraction of that.
    """
    from benchmarks.beam.run import _refuse_flags_this_arm_ignores

    args = _Mem0Args()
    args.question_types = "temporal-reasoning"
    with pytest.raises(SystemExit, match="(?i)question-types is not applied"):
        _refuse_flags_this_arm_ignores(args)


def test_the_refusal_is_silent_when_neither_flag_is_set() -> None:
    """A guard that refuses a legitimate invocation is worse than the gap it closes."""
    from benchmarks.beam.run import _refuse_flags_this_arm_ignores

    _refuse_flags_this_arm_ignores(_Mem0Args())


def test_the_mem0_arm_refuses_dry_run_rather_than_paying_the_full_judge() -> None:
    """`--dry-run` promises "make NO LLM call and spend nothing" and this arm never read it.

    RED before this change: its only consumer sits in the RE-call arm past the mem0 branch's
    return, so `--rejudge-mem0 --dry-run` exited 0 having judged every published answer at full
    price. The strongest possible statement of "spend nothing", silently inverted.
    """
    from benchmarks.beam.run import _refuse_flags_this_arm_ignores

    args = _Mem0Args()
    args.dry_run = True
    with pytest.raises(SystemExit, match="(?i)dry-run is not honoured"):
        _refuse_flags_this_arm_ignores(args)


def test_the_refusal_is_delivered_before_the_api_key_is_demanded(monkeypatch, tmp_path) -> None:
    """WHERE the call sits is the change; the three unit tests above cannot see it.

    Moving the call back below the key check restores the original defect and leaves every
    direct-call test green, so this drives the real entry point instead: no key in the
    environment, a nonexistent artifact, and the flag conflict must still be what surfaces.
    """
    import sys

    from benchmarks.beam import run as beam_run

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["run", "--rejudge-mem0", str(tmp_path / "nope.json"), "--no-judge"],
    )

    with pytest.raises(SystemExit) as exc:
        beam_run._main()

    assert "no-judge is meaningless" in str(exc.value), (
        f"the flag conflict must win over the key check and the missing file; got {exc.value!r}"
    )
