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
