"""Crash recovery for a benchmark run: parsing a half-written sidecar, and resuming from it.

Everything here is offline and synthetic. The sidecars are written by hand into `tmp_path` (a
truncated final line cannot be produced by asking the real harness nicely), the dataset is a small
LOCOMO-shaped fixture, and the memory system and the LLM are fakes — no database, no network, no
spend. The assertions pin the properties that make a salvage safe to publish: what a truncated
file costs, what a conflicting duplicate does, which conversations get re-bought, and whether the
resulting artifact admits what it is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks import salvage as salvage_module
from benchmarks.llm import OpenRouterLLM
from benchmarks.pipeline import aggregate
from benchmarks.rejudge import to_outcome
from benchmarks.run import _load
from benchmarks.salvage import (
    conversations_to_score,
    load_partial,
    main,
    merge_records,
    missing_conversations,
    read_partial,
    sample_id_from_question_id,
    scored_conversations,
)
from benchmarks.systems import MemorySystem

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

#: Shape-valid, obviously fake. The completer is patched out in every test that reaches one.
_FAKE_KEY = "sk-or-v1-unused-by-the-fake-completer"

#: The stem `benchmarks.run._run_stamp` would have produced for the run these fixtures pretend
#: crashed: arm `recall`, model `openai/gpt-4o-mini`, a 3-conversation slice.
_STEM = "recall_openai-gpt-4o-mini_3conv_20260724T120442Z"


def _fixture() -> list[dict[str, Any]]:
    """Three LOCOMO-shaped conversations; `conv-a` carries an adversarial question."""
    return [
        {
            "sample_id": "conv-a",
            "conversation": {"speaker_a": "Alice", "speaker_b": "Bob"},
            "qa": [
                {"question": "What did Alice research?", "answer": "adoption", "category": 1},
                {"question": "Penguins on Mars?", "adversarial_answer": "yes", "category": 5},
            ],
        },
        {
            "sample_id": "conv-b",
            "conversation": {"speaker_a": "Carol", "speaker_b": "Dan"},
            "qa": [{"question": "Who is Dan?", "answer": "a chef", "category": 2}],
        },
        {
            "sample_id": "conv-c",
            "conversation": {"speaker_a": "Erin", "speaker_b": "Frank"},
            "qa": [{"question": "Where does Erin live?", "answer": "Turin", "category": 3}],
        },
    ]


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "locomo-fixture.json"
    path.write_text(json.dumps(_fixture()), encoding="utf-8")
    return path


def _record(
    question_id: str,
    *,
    category: str = "cat1",
    adversarial: bool = False,
    answer: str = "an answer",
    abstained: bool = False,
    correct: bool | None = True,
    context: str = "ctx",
    question: str = "q",
    gold: str = "g",
) -> dict[str, Any]:
    """One sidecar record in `benchmarks.run._outcome_record`'s exact shape and key order."""
    return {
        "question_id": question_id,
        "category": category,
        "is_adversarial": adversarial,
        "context": context,
        "answer": answer,
        "abstained": abstained,
        "correct": None if adversarial else correct,
        "question": question,
        "gold": gold,
    }


#: The records a run of the fixture would have written for `conv-a` — one answerable, one
#: adversarial (whose `correct` is None by construction).
def _conv_a_records() -> list[dict[str, Any]]:
    return [
        _record("conv-a:0", question="What did Alice research?", gold="adoption"),
        _record(
            "conv-a:1",
            category="cat5-adversarial",
            adversarial=True,
            answer="NO_ANSWER",
            abstained=True,
            context="",
            question="Penguins on Mars?",
            gold="",
        ),
    ]


def _write_partial(path: Path, records: list[dict[str, Any]], *, truncate: bool = False) -> Path:
    """Write a sidecar. With `truncate`, the last line is cut mid-JSON, as a kill leaves it."""
    body = "".join(json.dumps(record) + "\n" for record in records)
    if truncate:
        body += json.dumps(_record("conv-x:0"))[:37]
    path.write_text(body, encoding="utf-8")
    return path


# --- parsing a half-written sidecar -----------------------------------------------------------


def test_load_partial_drops_only_the_truncated_final_line(tmp_path: Path) -> None:
    """The crash case IS the malformed file: a truncated tail must cost one record, not the file."""
    path = _write_partial(tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}",
                          _conv_a_records(), truncate=True)
    load = read_partial(path)
    assert load.kept == 2
    assert load.discarded_lines == [3]
    assert [r["question_id"] for r in load.records] == ["conv-a:0", "conv-a:1"]
    assert load.provenance() == {"path": str(path), "records": 2, "discarded_lines": 1}
    # the thin wrapper returns the same records
    assert load_partial(path) == load.records


def test_load_partial_keeps_every_line_of_an_intact_sidecar(tmp_path: Path) -> None:
    path = _write_partial(tmp_path / "intact.partial.jsonl", _conv_a_records())
    load = read_partial(path)
    assert load.kept == 2
    assert load.discarded_lines == []


def test_load_partial_refuses_a_malformed_line_that_is_not_the_last(tmp_path: Path) -> None:
    """Mid-file damage is corruption, not a partial write — dropping it would lose a paid record."""
    path = tmp_path / "corrupt.partial.jsonl"
    good = _conv_a_records()
    path.write_text(
        json.dumps(good[0]) + "\n" + '{"question_id": "conv-a:1", trunc\n'
        + json.dumps(good[1]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is NOT the final line"):
        read_partial(path)


def test_load_partial_rejects_a_record_that_breaks_the_outcome_invariant(tmp_path: Path) -> None:
    """`correct` must be None iff adversarial. A record that is not is a corrupt artifact."""
    broken = _record("conv-a:0")
    broken["is_adversarial"] = True  # `correct` is still True
    path = tmp_path / "broken.partial.jsonl"
    path.write_text(json.dumps(broken) + "\n" + json.dumps(_record("conv-b:0")) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="is NOT the final line"):
        read_partial(path)


# --- identity and merging ---------------------------------------------------------------------


def test_sample_id_is_derived_from_the_last_colon() -> None:
    """`question_id` is `f"{sample_id}:{index}"`; a sample_id containing a colon must survive."""
    assert sample_id_from_question_id("conv-26:167") == "conv-26"
    assert sample_id_from_question_id("weird:id:3") == "weird:id"
    for bad in ("conv-a", "conv-a:", ":3", "conv-a:x"):
        with pytest.raises(ValueError, match="sample_id"):
            sample_id_from_question_id(bad)


def test_scored_conversations_reports_the_sample_ids_present() -> None:
    records = [*_conv_a_records(), _record("conv-b:0")]
    assert scored_conversations(records) == {"conv-a", "conv-b"}


def test_merge_records_dedupes_identical_duplicates_silently() -> None:
    first = _conv_a_records()
    second = [_record("conv-a:0", question="What did Alice research?", gold="adoption")]
    merged = merge_records([first, second])
    assert [r["question_id"] for r in merged] == ["conv-a:0", "conv-a:1"]
    assert merged[0] == first[0]


def test_merge_records_raises_on_a_conflicting_duplicate() -> None:
    """Two scorings of one question must not be silently collapsed into one published number."""
    first = [_record("conv-a:0", answer="500 rps", correct=True)]
    second = [_record("conv-a:0", answer="four hundred", correct=False)]
    with pytest.raises(ValueError) as excinfo:
        merge_records([first, second])
    message = str(excinfo.value)
    assert "conv-a:0" in message  # the id is named, so the disagreement can be adjudicated
    assert "answer" in message and "correct" in message


# --- what is left to score --------------------------------------------------------------------


def test_missing_conversations_returns_the_unscored_ones_in_dataset_order(tmp_path: Path) -> None:
    convs, _questions, _skipped = _load(_write_fixture(tmp_path))
    # conv-b is scored; conv-a and conv-c are not, and must come back in the dataset's own order
    missing = missing_conversations(convs, [_record("conv-b:0")])
    assert [c["sample_id"] for c in missing] == ["conv-a", "conv-c"]
    assert missing_conversations(convs, []) == list(convs)


def test_conversations_to_score_includes_one_that_is_merely_short(tmp_path: Path) -> None:
    """A dropped truncated line leaves a conversation PRESENT but one record short.

    Treating presence as completeness would publish an artifact quietly missing that question, so
    the finer-grained walk must return the conversation with exactly its outstanding questions.
    """
    convs, questions, _skipped = _load(_write_fixture(tmp_path))
    records = [_conv_a_records()[0], _record("conv-b:0")]  # conv-a:1 lost to truncation
    work = conversations_to_score(convs, questions, records)
    assert [(c["sample_id"], [q["question_id"] for q in todo]) for c, todo in work] == [
        ("conv-a", ["conv-a:1"]),
        ("conv-c", ["conv-c:0"]),
    ]


# --- the CLI ----------------------------------------------------------------------------------


class _StubSys:
    """Records what it ingested; retrieves a constant. No DB, no network, no spend."""

    name = "stub"

    def __init__(self) -> None:
        self.ingested: list[str] = []

    def ingest(self, conversation: dict[str, Any]) -> None:
        self.ingested.append(str(conversation["sample_id"]))

    def retrieve(self, question: str) -> str:
        return "fresh ctx"


def _stub_complete(self: OpenRouterLLM, system: str, user: str) -> str:
    return "YES" if "Correct?" in user else "a fresh answer"


def _artifact(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def test_merge_only_rebuilds_the_artifact_and_its_aggregate_without_scoring(
    tmp_path: Path,
) -> None:
    """The run scored everything and died before writing the final JSON: no spend, same numbers."""
    records = [*_conv_a_records(), _record("conv-b:0"), _record("conv-c:0")]
    partial = _write_partial(tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", records)
    out = tmp_path / "salvaged.json"

    code = main(
        [
            "--partial", str(partial),
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--out", str(out),
            "--merge-only",
        ]
    )
    assert code == 0

    payload = _artifact(out)
    # schema-identical to a normally-produced artifact, plus the salvage markers
    assert set(payload) == {
        "arm", "model", "config", "conversations", "questions", "skipped_questions",
        "usage", "provider_metadata", "cost_claims", "aggregate", "outcomes", "salvaged",
        # Always present, empty on a clean run: `n` shrinks silently when a question is
        # quarantined, so a reader diffing two artifacts needs the count beside it.
        "dropped",
        "salvage",
    }
    assert payload["arm"] == "recall"
    assert payload["conversations"] == 3
    assert payload["questions"] == 4
    assert [o["question_id"] for o in payload["outcomes"]] == [
        "conv-a:0", "conv-a:1", "conv-b:0", "conv-c:0"
    ]
    # the aggregate is the one `benchmarks.pipeline.aggregate` computes over those same records
    assert payload["aggregate"] == aggregate([to_outcome(r) for r in records])
    assert payload["aggregate"]["answerable_accuracy"]["n"] == 3
    assert payload["aggregate"]["adversarial_abstention"]["n"] == 1
    assert payload["provider_metadata"] == []
    assert payload["cost_claims"] == []
    assert payload["salvage"]["records_rescored"] == 0
    assert payload["salvage"]["questions_still_unscored"] == []
    # nothing was ingested, so the config's per-arm block is empty rather than invented
    assert payload["config"]["system"] == {}
    # and no new sidecar exists: a merge-only salvage writes exactly one file
    assert list(tmp_path.glob("*.salvage.partial.jsonl")) == []


def test_merge_only_says_so_when_the_artifact_is_short_of_a_complete_run(tmp_path: Path) -> None:
    """An artifact whose n silently disagrees with every other run of the slice is the worst case.

    So a merge-only salvage that is short must name the questions it is short OF.
    """
    partial = _write_partial(
        tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", _conv_a_records()
    )
    out = tmp_path / "short.json"
    assert main(
        [
            "--partial", str(partial),
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--out", str(out),
            "--merge-only",
        ]
    ) == 0
    salvage = _artifact(out)["salvage"]
    assert salvage["questions_still_unscored"] == ["conv-b:0", "conv-c:0"]
    assert salvage["conversations_absent_from_partials"] == ["conv-b", "conv-c"]


def test_resume_scores_only_the_missing_conversations_and_marks_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: conv-a is already paid for, so it must not be ingested or scored again."""
    stub = _StubSys()

    def _fake_build(arm: str, model: str, openrouter_key: str, k: int, run_id: str) -> MemorySystem:
        return stub

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(salvage_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    partial = _write_partial(
        tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", _conv_a_records()
    )
    before = partial.read_text(encoding="utf-8")
    out = tmp_path / "resumed.json"

    assert main(
        [
            "--partial", str(partial),
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--k", "5",
            "--out", str(out),
        ]
    ) == 0

    # only the conversations with outstanding questions were ingested, in dataset order
    assert stub.ingested == ["conv-b", "conv-c"]

    payload = _artifact(out)
    assert payload["salvaged"] is True
    assert payload["questions"] == 4
    by_id = {o["question_id"]: o for o in payload["outcomes"]}
    # the salvaged records are byte-identical to what the dead run wrote
    assert by_id["conv-a:0"] == _conv_a_records()[0]
    # and the new ones came from this process
    assert by_id["conv-b:0"]["answer"] == "a fresh answer"
    assert by_id["conv-b:0"]["context"] == "fresh ctx"
    assert by_id["conv-b:0"]["gold"] == "a chef"

    salvage = payload["salvage"]
    assert salvage["records_from_partials"] == 2
    assert salvage["records_rescored"] == 2
    assert salvage["conversations_rescored"] == ["conv-b", "conv-c"]
    assert salvage["merge_only"] is False
    assert salvage["sources"] == [{"path": str(partial), "records": 2, "discarded_lines": 0}]
    assert salvage["questions_still_unscored"] == []

    # the input sidecar is untouched, and the new records went to a NEW one
    assert partial.read_text(encoding="utf-8") == before
    new_sidecar = Path(salvage["new_sidecar"])
    assert new_sidecar != partial
    fresh = [json.loads(line) for line in new_sidecar.read_text(encoding="utf-8").splitlines()]
    assert [r["question_id"] for r in fresh] == ["conv-b:0", "conv-c:0"]


def test_resume_refills_a_conversation_whose_last_record_was_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped truncated line must be re-scored, not silently missing from the artifact.

    The sidecar holds conv-a:0 and a half-written conv-a:1. After the truncated line is discarded,
    conv-a is present but incomplete — so it is re-ingested and its ONE outstanding question is
    re-scored, while the record already paid for is kept.
    """
    stub = _StubSys()

    def _fake_build(arm: str, model: str, openrouter_key: str, k: int, run_id: str) -> MemorySystem:
        return stub

    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)
    monkeypatch.setattr(salvage_module, "_build_system", _fake_build)
    monkeypatch.setattr(OpenRouterLLM, "complete", _stub_complete)

    partial = _write_partial(
        tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}",
        [_conv_a_records()[0]],
        truncate=True,
    )
    out = tmp_path / "refilled.json"
    assert main(
        [
            "--partial", str(partial),
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--out", str(out),
        ]
    ) == 0

    payload = _artifact(out)
    assert stub.ingested == ["conv-a", "conv-b", "conv-c"]
    assert [o["question_id"] for o in payload["outcomes"]] == [
        "conv-a:0", "conv-a:1", "conv-b:0", "conv-c:0"
    ]
    # the kept record was not re-bought; the lost one was
    assert payload["outcomes"][0] == _conv_a_records()[0]
    assert payload["outcomes"][1]["context"] == "fresh ctx"
    salvage = payload["salvage"]
    assert salvage["sources"][0]["discarded_lines"] == 1
    assert salvage["conversations_short_in_partials"] == ["conv-a"]
    assert salvage["records_from_partials"] == 1
    assert salvage["records_rescored"] == 3


def test_resume_refuses_a_partial_produced_by_a_different_arm(tmp_path: Path) -> None:
    """Resuming a `recall` sidecar as `mem0` would publish a silently mixed result."""
    partial = _write_partial(
        tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", _conv_a_records()
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--partial", str(partial),
                "--arm", "mem0",
                "--data", str(_write_fixture(tmp_path)),
                "--out", str(tmp_path / "mixed.json"),
                "--merge-only",
            ]
        )


def test_resume_refuses_a_k_that_the_sibling_artifact_contradicts(tmp_path: Path) -> None:
    """When a sibling artifact exists, k IS discoverable — and a mismatch is a mixed run."""
    partial = _write_partial(
        tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", _conv_a_records()
    )
    (tmp_path / f"{_STEM}.json").write_text(
        json.dumps({"config": {"arm": "recall", "model": "openai/gpt-4o-mini", "k": 10}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--partial", str(partial),
                "--arm", "recall",
                "--k", "5",
                "--data", str(_write_fixture(tmp_path)),
                "--out", str(tmp_path / "mixed.json"),
                "--merge-only",
            ]
        )


def test_provenance_admits_what_could_not_be_checked(tmp_path: Path) -> None:
    """An unverifiable check is stated, never quietly counted as a pass."""
    unnamed = _write_partial(tmp_path / "hand-renamed.partial.jsonl", _conv_a_records())
    out = tmp_path / "unchecked.json"
    assert main(
        [
            "--partial", str(unnamed),
            "--arm", "recall",
            "--data", str(_write_fixture(tmp_path)),
            "--out", str(out),
            "--merge-only",
        ]
    ) == 0
    consistency = _artifact(out)["salvage"]["consistency"]
    assert consistency["requested"] == {"arm": "recall", "model": "openai/gpt-4o-mini", "k": 5}
    assert consistency["verified"] == []
    joined = " ".join(consistency["unverified"])
    assert "neither the arm nor the model could be checked" in joined
    assert "k=5 could not be checked" in joined


def test_resume_refuses_to_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    partial = _write_partial(
        tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", _conv_a_records()
    )
    out = tmp_path / "taken.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--partial", str(partial),
                "--arm", "recall",
                "--data", str(_write_fixture(tmp_path)),
                "--out", str(out),
                "--merge-only",
            ]
        )


def test_resume_refuses_to_append_to_the_partial_it_is_reading(tmp_path: Path) -> None:
    """The input sidecar is the only surviving evidence of the dead run; it is never written to."""
    partial = _write_partial(
        tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", _conv_a_records()
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--partial", str(partial),
                "--arm", "recall",
                "--data", str(_write_fixture(tmp_path)),
                "--out", str(tmp_path / "out.json"),
                "--sidecar", str(partial),
            ]
        )


def test_a_refused_sibling_is_unverified_not_verified_provenance(tmp_path: Path) -> None:
    """Salvage cites a sibling `<stem>.json`'s config as VERIFIED provenance for the artifact it
    publishes, so a refused artifact must not be able to launder its config into a new one."""
    from benchmarks.salvage import consistency_report

    stem = "recall_openai-gpt-4o-mini_1conv_20260102T030405Z"
    partial = tmp_path / f"{stem}.partial.jsonl"
    partial.write_text("", encoding="utf-8")
    (tmp_path / f"{stem}.json").write_text(
        json.dumps(
            {
                "config": {"arm": "recall", "model": "openai/gpt-4o-mini", "k": 5},
                "unpublished": True,
                "unpublished_reason": "benchmark cost claims require provider_metadata",
            }
        ),
        encoding="utf-8",
    )

    report = consistency_report([partial], "recall", "openai/gpt-4o-mini", 5)

    # The FILENAME is still legitimate provenance for arm and model; the refused sibling's
    # config is what must not appear.
    assert not any("(config)" in entry["source"] for entry in report["verified"]), (
        "a refused sibling's config was cited as verified provenance"
    )
    assert any("REFUSED publication" in note for note in report["unverified"])
    # The refusal message carries the sibling's full path; a published artifact must not embed
    # an absolute local path, nor repeat the filename twice.
    assert not any(str(tmp_path) in note for note in report["unverified"])


def test_a_sibling_whose_config_is_not_an_object_is_reported_not_raised(tmp_path: Path) -> None:
    """`config.get(...)` runs outside the try, so a non-mapping config used to crash the CLI."""
    from benchmarks.salvage import consistency_report

    stem = "recall_openai-gpt-4o-mini_1conv_20260102T030405Z"
    partial = tmp_path / f"{stem}.partial.jsonl"
    partial.write_text("", encoding="utf-8")
    (tmp_path / f"{stem}.json").write_text(json.dumps({"config": "recall"}), encoding="utf-8")

    report = consistency_report([partial], "recall", "openai/gpt-4o-mini", 5)

    assert any("not an object" in note for note in report["unverified"])


def _merge_only(partial: Path, out: Path, data: Path) -> int:
    return main(
        ["--partial", str(partial), "--arm", "recall", "--data", str(data),
         "--out", str(out), "--merge-only"]
    )


def test_salvage_quarantines_an_artifact_the_cost_contract_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavioural, not a source-text check.

    Asserting that the contract call appears above the write is satisfied by a call sitting in a
    dead branch — verified: that mutant passed the old version of this test. What matters is
    that a refused payload does not reach `--out`, and that the re-scored work is not thrown
    away: by this point salvage may have re-bought whole conversations.
    """
    records = [*_conv_a_records(), _record("conv-b:0"), _record("conv-c:0")]
    partial = _write_partial(tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", records)
    # A results directory of its own, so the glob assertion below means something.
    out = tmp_path / "results" / "salvaged.json"

    def _poison(*args: object, **kwargs: object) -> dict[str, Any]:
        payload = real_payload(*args, **kwargs)  # type: ignore[arg-type]
        payload["headline"] = "the whole arm for $7.29 of tokens"
        return payload

    real_payload = salvage_module._results_payload
    monkeypatch.setattr(salvage_module, "_results_payload", _poison)

    code = _merge_only(partial, out, _write_fixture(tmp_path))

    assert code == salvage_module.QUARANTINE_EXIT
    assert not out.exists(), "a refused artifact reached the published path"
    # Outside the `results/*.json` glob that feeds `analyze`, exactly as run's quarantine is.
    # Landing beside the published artifacts makes every later `analyze --curve` over that
    # directory abort on the refusal.
    assert list(out.parent.glob("*.json")) == []
    quarantined = out.parent / "unpublished" / f"{out.stem}.json"
    assert quarantined.exists(), "the re-scored work was discarded rather than quarantined"
    recovered = json.loads(quarantined.read_text(encoding="utf-8"))
    assert recovered["unpublished"] is True
    assert "provider_metadata" in recovered["unpublished_reason"]


def test_a_salvage_quarantine_that_cannot_be_written_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rescue that can itself throw the work away is not a rescue — salvage's included.

    Hand-rolling the quarantine meant re-learning that lesson; it goes through run's hardened
    `_quarantine`, which falls back from the subdirectory to a sibling and finally to stdout.
    """
    records = [*_conv_a_records(), _record("conv-b:0"), _record("conv-c:0")]
    partial = _write_partial(tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", records)
    out = tmp_path / "results" / "salvaged.json"
    out.parent.mkdir(parents=True)
    (out.parent / "unpublished").write_text("not a directory", encoding="utf-8")

    real_payload = salvage_module._results_payload

    def _poison(*args: object, **kwargs: object) -> dict[str, Any]:
        payload = real_payload(*args, **kwargs)  # type: ignore[arg-type]
        payload["headline"] = "the whole arm for $7.29 of tokens"
        return payload

    monkeypatch.setattr(salvage_module, "_results_payload", _poison)

    code = _merge_only(partial, out, _write_fixture(tmp_path))

    assert code == salvage_module.QUARANTINE_EXIT
    fallback = out.parent / f"{out.stem}.json.unpublished.json.txt"
    assert fallback.exists() or (out.parent / "salvaged.unpublished.json.txt").exists(), (
        f"the work was lost when the subdirectory was blocked: {list(out.parent.iterdir())}"
    )


def test_a_failed_publish_leaves_no_half_artifact_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Salvage already refuses to overwrite an existing `--out` (see `p.error` on it), so the
    atomic write is not protecting a previous artifact — it is making sure a write that dies
    partway leaves neither a half-written `--out` that reads as a real result, nor its temp."""
    records = [*_conv_a_records(), _record("conv-b:0"), _record("conv-c:0")]
    partial = _write_partial(tmp_path / f"{_STEM}{salvage_module.PARTIAL_SUFFIX}", records)
    out = tmp_path / "salvaged.json"

    real_write = Path.write_text

    def _die_midway(self: Path, data: str, *a: object, **k: object) -> int:
        if self.name.startswith("salvaged.json"):
            real_write(self, data[:20], encoding="utf-8")  # truncated, like a full disk
            raise OSError(28, "No space left on device")
        return real_write(self, data, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _die_midway)

    with pytest.raises(OSError, match="No space left"):
        _merge_only(partial, out, _write_fixture(tmp_path))

    assert not out.exists(), "a half-written artifact was left at the published path"
    assert list(tmp_path.glob("*.tmp")) == [], "the atomic write left its temp behind"


def test_an_unreadable_sibling_does_not_leak_an_absolute_path_either(tmp_path: Path) -> None:
    """The scrubbing has to cover BOTH except branches.

    Compared against the path's PARTS rather than `str(tmp_path)`: an OSError's repr doubles the
    backslashes on Windows, so a substring check against the raw path silently misses a leak
    that is plainly there.
    """
    from benchmarks.salvage import consistency_report

    stem = "recall_openai-gpt-4o-mini_1conv_20260102T030405Z"
    partial = tmp_path / f"{stem}.partial.jsonl"
    partial.write_text("", encoding="utf-8")
    (tmp_path / f"{stem}.json").mkdir()  # unreadable as a file -> OSError, not SystemExit

    report = consistency_report([partial], "recall", "openai/gpt-4o-mini", 5)

    notes = " ".join(report["unverified"])
    assert "unreadable as a results artifact" in notes
    assert tmp_path.name not in notes, f"an absolute path leaked into a published artifact: {notes}"
