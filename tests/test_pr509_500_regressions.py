"""Regression evidence for BUG-001 and BUG-002 from PRs 509 through 500.

Red proof: run these exact nodes on 1cfd9f333 with this test file present, before
repairing prior_art.render/validate and labelled._make_embedder. Unapproved
claims must fail the capability-status assertion, mismatched reviews must fail
the validation assertion, and configured profiles must fail the selected-
embedder assertion. The accepted and environment-unset cases are controls for
unchanged behavior, not separate defect claims. No models or APIs are used.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from recall import embeddings
from recall.eval import labelled, locomo
from tools.prior_art.loader import load_dataset
from tools.prior_art.render import render_files
from tools.prior_art.validate import validate_dataset


@pytest.mark.parametrize(
    ("claim_status", "decision"),
    [
        ("draft", "accepted"),
        ("rejected", "accepted"),
        ("disputed", "accepted"),
        ("accepted", "rejected"),
        ("accepted", "disputed"),
        ("accepted", "accepted"),
    ],
)
def test_prior_art_only_admitted_claims_support_reports(
    tmp_path: Path, claim_status: str, decision: str,
) -> None:
    dataset = deepcopy(load_dataset())
    claim = dataset["claims"][0]
    review = next(r for r in dataset["reviews"] if r["review_id"] == claim["review_id"])
    dataset["claims"] = [claim]
    dataset["reviews"] = [review]
    dataset["systems"] = [s for s in dataset["systems"] if s["system_id"] == claim["system_id"]]
    dataset["report_config"]["target_hypothesis"] = [claim["capability_id"]]
    dataset["root"] = tmp_path
    claim["review_status"] = claim_status
    review["decision"] = decision
    assert validate_dataset(dataset) == []

    render_files(dataset)

    summary = json.loads((tmp_path / "generated_summary.json").read_text())
    matrix = (tmp_path / "generated_matrix.md").read_text()
    gaps = (tmp_path / "generated_gap_report.md").read_text()
    admitted = claim_status == decision == "accepted"
    expected_gap = "emerging" if admitted else "unverified_gap"
    assert summary["capability_status"][claim["capability_id"]] == expected_gap
    combination = summary["target_combination"]["systems"][claim["system_id"]]
    assert combination["status"] == (
        "verified_combination" if admitted else "unverified_combination"
    )
    capability_section = gaps.split(f"### {claim['capability_id']}\n", 1)[1].split("### ", 1)[0]
    assert f"Status: **{expected_gap}**" in capability_section
    system_name = dataset["systems"][0]["display_name"]
    if admitted:
        assert f"| {system_name} | `verified` |" in matrix
        assert claim["claim_id"] not in summary["unresolved_claim_ids"]
    else:
        assert f"| {system_name} | `verified` |" not in matrix
        assert claim["claim_id"] in matrix.split("## Incomplete or unresolved claims", 1)[1]
        assert claim["claim_id"] in summary["unresolved_claim_ids"]


@pytest.mark.parametrize("direction", ["claim", "review"])
def test_prior_art_review_must_match_claim(direction: str) -> None:
    dataset = deepcopy(load_dataset())
    first, second = dataset["claims"][:2]
    if direction == "claim":
        first["review_id"] = second["review_id"]
    else:
        first_review = next(r for r in dataset["reviews"] if r["review_id"] == first["review_id"])
        first_review["claim_id"], dataset["reviews"][1]["claim_id"] = (
            second["claim_id"], first["claim_id"],
        )
    assert any("review_id does not match claim" in error for error in validate_dataset(dataset))


@pytest.mark.parametrize(
    "profile_id", ["bge-small-context-section-v1", "bge-large-context-section-v1", None],
)
def test_labelled_cli_passes_selected_profile_to_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile_id: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("RECALL_EMBED_PROFILE", raising=False)
    if profile_id:
        monkeypatch.setenv("RECALL_EMBED_PROFILE", profile_id)
    legacy = SimpleNamespace(name="legacy-small-raw", dim=384)
    selected = SimpleNamespace(name=profile_id, dim=1024 if profile_id and "large" in profile_id else 384)
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", lambda **kwargs: legacy)
    monkeypatch.setattr(embeddings, "resolve_registered_embedder", lambda *args, **kwargs: selected)
    captured = {}

    def capture_evaluate(dsn, corpus, questions, embedder, **kwargs):
        captured["embedder"] = embedder
        return {"evaluated": True}

    monkeypatch.setattr(labelled, "evaluate", capture_evaluate)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "query": "fixture", "answerable": False}]))
    monkeypatch.setattr(sys, "argv", [
        "labelled", "--corpus", str(tmp_path), "--questions", str(questions),
    ])

    labelled.main()

    assert captured["embedder"] is (selected if profile_id else legacy)
    assert json.loads(capsys.readouterr().out) == {"evaluated": True}
    assert locomo._make_embedder is labelled._make_embedder
