"""Production memory launchers must match the active Context 4 generation.

Red proof receipt ``context4-memory-launcher-01``: on 2026-09-14 this test failed because
``.mcp.json``, ``scripts/session-mcp.sh``, ``scripts/session_serving_remote.sh``,
``scripts/check_recall_mcp_stdio.py``, and ``scripts/launch_codex_ab_smoke.py`` still selected
``voyage:voyage-4``. A strict search then refused the certified Context 4 generation with
``LINEAGE_MISMATCH`` before retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_recall_mcp_stdio import _command
from scripts.launch_codex_ab_smoke import _codex_home


ROOT = Path(__file__).resolve().parents[1]
CONTEXT4 = "voyage-context:voyage-context-4"
LEGACY = "RECALL_EMBEDDER=voyage:voyage-4"


def test_every_production_memory_launcher_selects_context4(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("RECALL_TEST_EMBEDDER", raising=False)
    monkeypatch.setenv("CODEX_API_KEY", "test-only")

    manifest = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
    manifest_args = " ".join(manifest["mcpServers"]["recall-memory"]["args"])
    _, stdio_args = _command()
    smoke_home = _codex_home(tmp_path / "source", tmp_path, "smoke", with_recall=True)
    smoke_config = (smoke_home / "config.toml").read_text(encoding="utf-8")

    launch_surfaces = [
        (".mcp.json", manifest_args, f"RECALL_EMBEDDER={CONTEXT4}"),
        (
            "check_recall_mcp_stdio.py",
            " ".join(stdio_args),
            f"RECALL_EMBEDDER={CONTEXT4}",
        ),
        ("launch_codex_ab_smoke.py", smoke_config, f"RECALL_EMBEDDER={CONTEXT4}"),
        (
            "session-mcp.sh",
            (ROOT / "scripts/session-mcp.sh").read_text(encoding="utf-8"),
            f'"recall-memory": vps2("memory", "{CONTEXT4}")',
        ),
        (
            "session_serving_remote.sh",
            (ROOT / "scripts/session_serving_remote.sh").read_text(encoding="utf-8"),
            f'EMBEDDER="${{RECALL_SERVING_EMBEDDER:-{CONTEXT4}}}"',
        ),
    ]
    for name, content, expected in launch_surfaces:
        assert expected in content, f"{name} does not select the active Context 4 embedder"
        assert LEGACY not in content, f"{name} still forces the retired Voyage 4 embedder"
