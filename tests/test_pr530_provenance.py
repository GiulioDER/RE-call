"""PR530 consumer regressions, red against 76a0b36ea.

DAT-001 uses only temporary tables in this checkout's disposable PostgreSQL.
DAT-002 replaces the paid generation transport; no provider request is made.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import psycopg

from scripts import agent_ab_build_gold_set as gold
from scripts import agent_ab_prepare_discoverability_corpus as prep
from tests.conftest import TEST_DSN, requires_db


@requires_db
def test_gold_sheet_uses_the_pinned_corpus(tmp_path, monkeypatch):
    """DAT-001 red: higher-ranked retired/foreign chunks enter the label sheet."""
    conn = psycopg.connect(TEST_DSN)
    conn.execute("CREATE TEMP TABLE recall_generations (tenant_id text, generation_id text)")
    conn.execute("INSERT INTO recall_generations VALUES ('wanted', 'pinned')")
    conn.execute("CREATE TEMP TABLE recall_chunks_v1 (tenant_id text, generation_id text, "
                 "chunk_id text, source_uri text, text text, tsv tsvector)")
    for tenant, generation, chunk, text in [
        ("wanted", "pinned", "correct", "needle chosen-note"),
        ("wanted", "retired", "retired", "needle " * 30 + "retired-decoy"),
        ("other", "pinned", "foreign", "needle " * 100 + "foreign-decoy"),
    ]:
        conn.execute("INSERT INTO recall_chunks_v1 VALUES (%s,%s,%s,%s,%s,to_tsvector('english',%s))",
                     (tenant, generation, chunk, "file:///memo.md", text, text))
    validation, trigger = tmp_path / "validation.json", tmp_path / "trigger.json"
    validation.write_text(json.dumps({"part_a": {"items": [{
        "haiku": True, "sonnet": False, "gemini": True,
        "task_id": "task", "draft_prefix": "needle", "memo": "memo",
    }]}}))
    trigger.write_text(json.dumps({"records": [{
        "population": "positive", "task_id": "task", "draft": "needle",
    }]}))
    out = tmp_path / "out"
    args = SimpleNamespace(validation=str(validation), trigger=str(trigger), dsn=TEST_DSN,
                           out=str(out), tenant="wanted", generation="pinned")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda *_: args)
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)
    try:
        assert gold.main() == 0
    finally:
        conn.close()
    sheet = (out / "gold-set.md").read_text()
    assert "chosen-note" in sheet
    assert "decoy" not in sheet
    key = json.loads((out / "gold-key.json").read_text())
    assert key["tenant"] == "wanted"
    assert key["generation_id"] == "pinned"
    assert key["items"][0]["chunk_id"] == "correct"
    assert key["items"][0]["source_uri"] == "file:///memo.md"


def test_resume_reuses_only_the_same_memo_input(tmp_path, monkeypatch):
    """DAT-002 red: unchanged terms/name retain rewrite for superseded input bytes."""
    memo = tmp_path / "memo.md"
    memo.write_text("# First\n\nOriginal memo body.\n", encoding="utf-8")
    out = tmp_path / "out"
    args = SimpleNamespace(evidence_dsn="offline", out=str(out), workers=1,
                           allow_lossy_reconstruction=False)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda *_: args)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-transport-fixture")

    def rows(_dsn):
        data = memo.read_bytes()
        text = data.decode("utf-8")
        return [(memo.as_uri(), prep.sha256_bytes(data), i, part)
                for i, part in enumerate([text[:10], text[10:]])]

    monkeypatch.setattr(prep, "rows", rows)
    generated = []

    def generate(text, _key):
        generated.append(text)
        return {"title": "Title", "description": "Description", "tasks": ["one", "two", "three"]}

    monkeypatch.setattr(prep, "generate_rewrite", generate)
    assert prep.main() == 0
    assert prep.main() == 0
    assert len(generated) == 1, "unchanged inputs must retain paid completed work"
    memo.write_text("# Second\n\nChanged memo body.\n", encoding="utf-8")
    assert prep.main() == 0
    assert generated == ["# First\n\nOriginal memo body.\n", "# Second\n\nChanged memo body.\n"]
