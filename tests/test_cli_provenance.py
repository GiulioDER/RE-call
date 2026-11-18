"""Local SQLite surface tests for structured provenance application."""

from datetime import UTC, datetime
import json

from recall.cli import main
from recall.types import AtomicFact, EvidenceCard, source_content_digest


def test_cli_sqlite_provenance_apply_current_and_source_revalidation(tmp_path, capsys):
    root = tmp_path / "corpus"
    root.mkdir()
    source = root / "owner.md"
    source.write_text("service api is owned by platform", encoding="utf-8")
    claim = AtomicFact(
        namespace="memory",
        subject="service:api",
        predicate="owner",
        object="team:platform",
    )
    card = EvidenceCard(
        card_id="",
        chunk_id="chunk-owner",
        source="owner.md",
        source_digest=source_content_digest(source.read_text(encoding="utf-8")),
        valid_from=None,
        valid_until=None,
        first_indexed_at=datetime(2026, 9, 1, tzinfo=UTC),
        indexed_at=datetime(2026, 9, 1, tzinfo=UTC),
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="calibration-a",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=0.99,
        rank=1,
        structured_facts=(claim,),
    )
    claim_path = tmp_path / "claim.json"
    cards_path = tmp_path / "cards.json"
    claim_path.write_text(json.dumps(claim.to_payload()), encoding="utf-8")
    cards_path.write_text(json.dumps({"cards": [card.to_payload()]}), encoding="utf-8")
    sqlite_path = tmp_path / "provenance.sqlite"
    base = [
        "--tenant", "tenant-a", "provenance", "apply",
        "--claim", str(claim_path), "--cards", str(cards_path),
        "--request-id", "cli-local-1", "--generation", "generation-a",
        "--sqlite-path", str(sqlite_path), "--source-root", str(root),
    ]

    main(base)
    applied = json.loads(capsys.readouterr().out)
    assert applied["allowed"] is True
    assert applied["decision_code"] == "APPLIED"

    main([
        "--tenant", "tenant-a", "provenance", "current",
        "--sqlite-path", str(sqlite_path),
    ])
    current = json.loads(capsys.readouterr().out)
    assert [item["fact"]["object"] for item in current["facts"]] == ["team:platform"]

    source.write_text("service api is owned by security", encoding="utf-8")
    main([
        "--tenant", "tenant-a", "provenance", "apply",
        "--claim", str(claim_path), "--cards", str(cards_path),
        "--request-id", "cli-local-2", "--generation", "generation-a",
        "--sqlite-path", str(sqlite_path), "--source-root", str(root),
    ])
    refused = json.loads(capsys.readouterr().out)
    assert refused["allowed"] is False
    assert refused["decision_code"] == "SOURCE_CHANGED"


def _apply_one(tmp_path, capsys, root, sqlite_path, name, subject, request_id):
    source = root / name
    source.write_text(f"{subject} is owned by platform", encoding="utf-8")
    claim = AtomicFact(namespace="memory", subject=subject, predicate="owner", object="team:platform")
    card = EvidenceCard(
        card_id="", chunk_id=f"chunk-{request_id}", source=name,
        source_digest=source_content_digest(source.read_text(encoding="utf-8")),
        valid_from=None, valid_until=None,
        first_indexed_at=datetime(2026, 9, 1, tzinfo=UTC), indexed_at=datetime(2026, 9, 1, tzinfo=UTC),
        tenant_id="tenant-a", generation_id="generation-a",
        pipeline_fingerprint="p" * 64, corpus_fingerprint="c" * 64,
        calibration_id="calibration-a", calibration_status="certified",
        trust_state="trusted", verdict="ok", confidence=0.99, rank=1, structured_facts=(claim,),
    )
    claim_path = tmp_path / f"{request_id}-claim.json"
    cards_path = tmp_path / f"{request_id}-cards.json"
    claim_path.write_text(json.dumps(claim.to_payload()), encoding="utf-8")
    cards_path.write_text(json.dumps({"cards": [card.to_payload()]}), encoding="utf-8")
    main([
        "--tenant", "tenant-a", "provenance", "apply",
        "--claim", str(claim_path), "--cards", str(cards_path),
        "--request-id", request_id, "--generation", "generation-a",
        "--sqlite-path", str(sqlite_path), "--source-root", str(root),
    ])
    assert json.loads(capsys.readouterr().out)["decision_code"] == "APPLIED"


def test_cli_provenance_current_applies_the_source_policy(tmp_path, capsys, monkeypatch):
    """Under RECALL_SOURCE_POLICY_FILE, `recall provenance current` lists what
    `recall_current_facts` lists: a fact whose only memo the policy denies is not listed, and the
    denied memo is not named. The permitted fact is still listed, so an empty answer cannot pass.

    Red proof (2026-11-18): with `listing_payload(*listing, permitted)` in
    `recall/cli_commands/provenance_cmd.py::_cmd_provenance` reverted to `listing_payload(*listing)`,
    this test failed at the `subjects == ["service:api"]` assertion, listing both facts.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    sqlite_path = tmp_path / "provenance.sqlite"
    _apply_one(tmp_path, capsys, root, sqlite_path, "owner.md", "service:api", "cli-allowed")
    _apply_one(tmp_path, capsys, root, sqlite_path, "secret.md", "service:vault", "cli-denied")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"default_deny": True, "rules": [
        {"prefix": "owner.md", "purposes": ["retrieval"], "principals": [], "redactions": []},
    ]}), encoding="utf-8")
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(policy))

    main(["--tenant", "tenant-a", "provenance", "current", "--sqlite-path", str(sqlite_path)])
    out = capsys.readouterr().out
    current = json.loads(out)
    subjects = [item["fact"]["subject"] for item in current["facts"] + current["contested"]]
    assert subjects == ["service:api"]
    assert "secret.md" not in out
