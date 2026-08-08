"""Tasks B and C generation: the properties that cost money or credibility if they slip."""

from __future__ import annotations

import json

import pytest

from benchmarks.mtrag import generation as gen


def _task(task_id: str = "conv1<::>2", answerability: str = "UNANSWERABLE") -> dict:
    """A task shaped like the real release, INCLUDING the label that must not leak.

    `mtrag-human` ships `answerability` inside each turn's `enrichments`. MTRAGEval withheld that
    metadata from participants, and it is exactly what the abstention decision is supposed to
    infer, so the fixture carries it deliberately: a test that omitted it could not catch a leak.
    """
    return {
        "task_id": task_id,
        "conversation_id": task_id.split("<::>")[0],
        "Collection": "clapnq",
        "input": [
            {
                "speaker": "user",
                "text": "where do the arizona cardinals play",
                "enrichments": {"Question Type": ["Factoid"], "answerability": [answerability]},
            },
            {"speaker": "agent", "text": "In Glendale, Arizona."},
            {
                "speaker": "user",
                "text": "do they play outside the US?",
                "enrichments": {"answerability": [answerability]},
            },
        ],
        "contexts": [
            {"document_id": f"d{i}", "score": float(20 - i), "text": f"passage {i}"}
            for i in range(20)
        ],
    }


def test_the_answerability_label_never_reaches_the_prompt() -> None:
    """⛔ The single most important property in this module.

    A correct "I don't know" on an UNANSWERABLE task scores exactly 1.0 on all three metrics at
    once, so a model that can SEE the label scores far above one that has to infer it. That would
    not be a better system, it would be a leaked answer key, and nothing downstream would show it.
    """
    task = _task(answerability="UNANSWERABLE")
    contexts = gen.contexts_for(task, None)

    blob = json.dumps(gen.build_messages(task, contexts))

    assert "UNANSWERABLE" not in blob
    assert "answerability" not in blob
    assert "enrichments" not in blob
    assert "Question Type" not in blob
    # The actual conversation must still be there, or the test would pass on an empty prompt.
    assert "do they play outside the US?" in blob


def test_contexts_are_trimmed_to_the_official_maximum() -> None:
    """`format_checker.py` sets MAX_CONTEXTS = 10 and rejects more. 20 in, 10 out."""
    contexts = gen.contexts_for(_task(), None)

    assert len(contexts) == gen.MAX_CONTEXTS
    assert [c["document_id"] for c in contexts] == [f"d{i}" for i in range(10)], (
        "the trim must keep the TOP of the ranking, not an arbitrary slice"
    )


def test_recall_contexts_do_not_silently_fall_back_to_the_benchmarks_own() -> None:
    """A task RE-call never retrieved must come back empty, not borrowed.

    Falling back would report RE-call numbers for turns RE-call never saw, which is the flattering
    direction and would be invisible in the output.
    """
    task = _task(task_id="missing<::>1")

    contexts = gen.contexts_for(task, {"other<::>1": [{"document_id": "x", "score": 1.0}]})

    assert contexts == []


def test_resume_skips_task_ids_already_written(tmp_path) -> None:
    """Generation is the expensive step; a re-run must not pay for answers it already has."""
    out = tmp_path / "preds.jsonl"
    out.write_text(
        json.dumps({"task_id": "a<::>1", "predictions": [{"text": "hi"}]}) + "\n"
        + "{ truncated line that never finished\n",
        encoding="utf-8",
    )

    done = gen.already_done(out)

    assert done == {"a<::>1"}, (
        "a malformed trailing line, which is what an interrupted write leaves, must be ignored "
        "rather than fatal: that task is simply regenerated"
    )


def test_a_permanent_error_is_not_retried(monkeypatch) -> None:
    """A bad key is not a flaky network, and every attempt is billed."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)

    class AuthenticationError(Exception):
        pass

    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.calls += 1
            raise AuthenticationError("bad key")

    client = Client()
    with pytest.raises(RuntimeError, match="bad key"):
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert client.calls == 1


def test_a_transient_error_is_retried(monkeypatch) -> None:
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)

    class Reply:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]

    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("transient")
            return Reply("  an answer  ")

    client = Client()
    answer = gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert answer == "an answer"
    assert client.calls == 3


def test_the_submission_row_carries_every_required_field() -> None:
    """`format_checker.py` requires the union across modes: rag_taskc is the strictest."""
    task = _task()
    row = gen.submission_row(task, gen.contexts_for(task, None), "an answer")

    for field in ("task_id", "Collection", "input", "contexts", "predictions"):
        assert field in row, field
    assert row["predictions"] == [{"text": "an answer"}]
    for ctx in row["contexts"]:
        assert isinstance(ctx["document_id"], str)
        assert isinstance(ctx["score"], (int, float))


def test_an_oversized_submission_is_refused(tmp_path) -> None:
    """The 20 MB limit is the official one, and RE-call's own Task A files are 127 MB."""
    big = tmp_path / "big.jsonl"
    big.write_bytes(b"x" * (gen.MAX_SUBMISSION_MB * 1024 * 1024 + 1))

    with pytest.raises(RuntimeError, match="above the official"):
        gen.check_submission_size(big)
