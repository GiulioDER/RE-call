"""Pin the BEAM ablation-preflight wiring, and reject a non-positive `--ablation-sample`.

Mirrors `tests/test_bench_run.py::test_main_runs_the_ablation_preflight_once_before_the_first_
retrieve_and_stamps_it`, the LOCOMO test that pins the same guard on that harness. BEAM's control
flow is more fragile than LOCOMO's: it gates the preflight on an `ablation_run` sentinel rather
than `position == 0`, and it interacts with `--resume` skipping an already-scored conversation's
questions — exactly the kind of refactor (moving the sentinel flip, or sampling `pending` instead
of `in_scope`) that would silently break the guard with no red test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.beam import run as beam_run_module
from benchmarks.beam.dataset import Conversation, Question
from benchmarks.beam.run import build_parser
from benchmarks.beam.systems import BeamRecallSystem

#: A shape-valid but obviously fake key. No `.complete()` call is ever made in these tests — the
#: expensive stage (`_run_pool`) is monkeypatched away — so the value is never sent anywhere.
_FAKE_KEY = "sk-or-v1-unused-by-the-fake-pool"


def _conversations() -> tuple[Conversation, Conversation]:
    """Two tiny BEAM conversations: two questions in the first, one in the second.

    Two questions in conv 0 (rather than one) is what lets the resume test below tell "sampled
    from `in_scope`" apart from "sampled from `pending`" — with only one question the two sets
    would always be equal and the assertion would pass for the wrong reason.
    """
    conv0 = Conversation(
        chat_size="1M",
        index=0,
        conversation_id="c0",
        turns=[{"role": "user", "content": "hello", "date": ""}],
        questions=[
            Question(
                question_id="1M_0_q0_single_hop",
                chat_size="1M",
                conversation_idx=0,
                question_type="single_hop",
                question="Q1?",
                rubric=["a"],
            ),
            Question(
                question_id="1M_0_q1_single_hop",
                chat_size="1M",
                conversation_idx=0,
                question_type="single_hop",
                question="Q2?",
                rubric=["a"],
            ),
        ],
    )
    conv1 = Conversation(
        chat_size="1M",
        index=1,
        conversation_id="c1",
        turns=[{"role": "user", "content": "hello again", "date": ""}],
        questions=[
            Question(
                question_id="1M_1_q0_single_hop",
                chat_size="1M",
                conversation_idx=1,
                question_type="single_hop",
                question="Q3?",
                rubric=["a"],
            ),
        ],
    )
    return conv0, conv1


def _scored_row(question: Question) -> dict[str, Any]:
    """A `_run_pool`-shaped scored row, so `aggregate`/`_coverage`/the summary don't choke on it."""
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "question": question.question,
        "rubric": question.rubric,
        "memories_retrieved": 0,
        "memories_evaluated": 0,
        "generated_answer": "stub",
        "abstained": False,
        "retrieval_empty": True,
        "score": 1.0,
        "judgment": "PASS",
        "nugget_scores": [],
        "judge_errors": 0,
        "conversation_idx": question.conversation_idx,
    }


def test_ablation_preflight_fires_once_before_run_pool_and_samples_in_scope_not_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins, at minimum: one preflight call, before the first `_run_pool`, sampling `in_scope`
    (not `pending`), with `self._candidate_k` (not the library default) — and that its result
    lands in `summary["ablation_preflight"]`.

    A REAL `BeamRecallSystem` is built (same reason `test_bench_run.py`'s LOCOMO test builds a
    real `RecallSystem`: only that lets the fake `ablation_preflight` below inspect `self.
    _candidate_k`, which is set by the constructor's own clamp — the exact configuration this
    preflight exists to detect). `ingest` is monkeypatched off the class (no DB, no network); so
    is `ablation_preflight` itself, since it is the call being pinned, not the thing under test —
    `_run_pool` (the expensive stage) is monkeypatched away instead, per Finding 2's instruction
    not to monkeypatch away the guard being tested.

    `--resume` marks conv 0's FIRST question as already scored, so `pending` for conv 0 is only
    the second question while `in_scope` is both — the one case that can tell the two apart.
    """
    conv0, conv1 = _conversations()

    events: list[str] = []
    calls: list[tuple[list[str], int, str, bool, int]] = []

    def _fake_ingest(self: BeamRecallSystem, conversation: Conversation) -> int:
        self._tenant = f"beam-{conversation.chat_size}-{conversation.index}"
        events.append(f"ingest:{conversation.index}")
        return len(conversation.turns)

    def _fake_preflight(
        self: BeamRecallSystem,
        questions: list[str],
        *,
        sample: int,
        metric_class: str,
        allow_inert: bool,
    ) -> list[dict[str, Any]]:
        # self._candidate_k, not recall.retriever.DEFAULT_CANDIDATE_K: this is what BOTH `retrieve`
        # and the REAL `ablation_preflight` use (see the note in `BeamRecallSystem.__init__`), and
        # what a future refactor could silently drop back to the library default without this
        # assertion noticing.
        calls.append((list(questions), sample, metric_class, allow_inert, self._candidate_k))
        events.append("preflight")
        return [{"mechanism": "sparse", "verdict": "DIFFERS", "sampled": len(questions), "differing": 1}]

    def _fake_run_pool(tasks: list[Question], work: Any, workers: int) -> list[dict[str, Any]]:
        events.append(f"run_pool:{len(tasks)}")
        return [_scored_row(q) for q in tasks]

    monkeypatch.setattr(BeamRecallSystem, "ingest", _fake_ingest)
    monkeypatch.setattr(BeamRecallSystem, "ablation_preflight", _fake_preflight)
    monkeypatch.setattr(beam_run_module, "_run_pool", _fake_run_pool)
    monkeypatch.setattr(
        beam_run_module,
        "iter_conversations",
        lambda path, chat_size, indices=None: iter([conv0, conv1]),
    )
    monkeypatch.setattr(beam_run_module, "count_conversations", lambda path, indices=None: 2)
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)

    resume_path = tmp_path / "resume.partial.jsonl"
    resume_path.write_text(
        json.dumps(_scored_row(conv0.questions[0])) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "beam-run",
            "--data", "unused.parquet",
            "--k", "45",
            "--candidate-k", "250",
            "--ablation-sample", "3",
            "--out-dir", str(out_dir),
            "--resume", str(resume_path),
        ],
    )

    beam_run_module._main()

    # Exactly one preflight call: right after conv 0's ingest, strictly before conv 0's
    # `_run_pool` — and never again for conv 1.
    assert events[:2] == ["ingest:0", "preflight"]
    assert events.count("preflight") == 1
    assert events.index("preflight") < events.index("run_pool:1")

    assert len(calls) == 1
    questions, sample, metric_class, allow_inert, candidate_k = calls[0]
    # BOTH of conv 0's questions, even though the first is already "done" via --resume and
    # `pending` therefore holds only the second — the preflight must sample the FULL scope, not
    # the leftover work.
    assert questions == ["Q1?", "Q2?"]
    assert sample == 3
    assert metric_class == "set"
    assert allow_inert is False
    # max(candidate_k=250, k=45) = 250 — the constructor's own clamp, not the library default.
    assert candidate_k == 250

    # conv 0 ingests once; conv 1 ingests once; preflight is strictly before conv 0's run_pool.
    assert events == ["ingest:0", "preflight", "run_pool:1", "ingest:1", "run_pool:1"]

    [artifact] = out_dir.glob("*.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["summary"]["ablation_preflight"] == {
        "verdicts": [
            {"mechanism": "sparse", "verdict": "DIFFERS", "sampled": 2, "differing": 1},
        ],
        "allow_inert_arm": False,
        "sample": 3,
    }


@pytest.mark.parametrize("bad", ["0", "-5"])
def test_ablation_sample_rejects_non_positive_values(bad: str) -> None:
    """`--ablation-sample -5` silently drops the last 5 questions instead of sampling the head
    (`questions[: sample or DEFAULT_SAMPLE]` slices from the end on a negative count); `0` falls
    back to `DEFAULT_SAMPLE=25` with no signal that the requested sample was never honoured. Both
    contradict the flag's own help text ("taken deterministically from the head of the question
    list"), so both are rejected at the parser — the same validator LOCOMO's `--ablation-sample`
    already uses (`benchmarks.run._positive_int`).
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--data", "x.parquet", "--ablation-sample", bad])
