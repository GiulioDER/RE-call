"""The configuration report and the dashboard's Configuration page: what runs, never a secret.

Invariants and the failure each one catches:
- F1 the report withholds every value the schema flags secret (including `RECALL_DSN` and
  `RECALL_REASONING_API_KEY`, flagged by this change), scrubs a password out of any URL it does
  show, keeps a non-secret security flag visible, names unknown `RECALL_*` variables without
  their values, and scrubs a withheld value out of the startup problem it reports.
- F2 a launch command loses every credential: `NAME=value` with a credential name (also inside
  one ssh command string), the argument after `--api-key`, `--api-key value` inline, and a URL
  password; a non-secret `NAME=value` stays readable.
- F3 the page shows this machine and each RE-call server registered for the project (local
  scope and `.mcp.json`), never a secret from any of them, and leaves other MCP servers out.
- F4 a remote report that cannot be produced is a card on the page, not an error page; a host
  that looks like an ssh option is refused before ssh runs.
- F5 report text is escaped on the page.

The placeholder `CHANGEME-SENTINEL` stands in for every secret: a test fails if it appears.

Red proof, 2026-10-08, each mutation alone, failing in its intended assertion (JUnit XML), then
restored byte for byte and green:
- C1 (F1) `withheld` ignoring the schema's `secret` flag: "RECALL_SERVING_DSN's value reached the
  report".
- C2 (F2) the credential-name rule dropped from `withheld`: "GITHUB_TOKEN's value survived". It
  SURVIVED a first version that used `VOYAGE_API_KEY`, which the schema already flags secret, so
  the name rule never had a turn; the test now uses a credential name no schema knows.
- C3 (F1) `redact_value` not scrubbing URL passwords: "a URL setting kept its password".
- C4 (F2) `_INLINE_FLAG` not applied: "--api-key inside a command string kept its value".
- C5 (F3) `_is_recall_server` accepting every server: "another MCP server was reported".
- C6 (F4) the page catching `KeyError` instead of `ConfigUnavailable`: `ConfigUnavailable`
  escaped `handle`, which is the defect itself (an error page where a card belongs).
- C7 (F5) a report value rendered without `_e`: "report text reached the page unescaped".
- C8 (F1) `_scrub_message` not replacing withheld values: "a withheld value survived in the
  startup problem".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.dashboard import config
from recall.dashboard.server import DashboardApp
from recall_mcp.config_report import build_report, redact_argv

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
PLACEHOLDER = "CHANGEME-SENTINEL"


def _entry(report: dict, name: str) -> dict:
    return next(e for e in report["entries"] if e["name"] == name)


def test_the_report_withholds_secrets_and_scrubs_urls() -> None:
    """F1."""
    env = {
        "RECALL_SERVING_DSN": f"postgresql://recall:{PLACEHOLDER}@db/recall",
        "RECALL_DSN": f"postgresql://recall:{PLACEHOLDER}@db/recall",
        "RECALL_REASONING_API_KEY": PLACEHOLDER,
        "RECALL_AUTH_ISSUER_URL": f"https://user:{PLACEHOLDER}@issuer.example",
        "RECALL_ALLOW_INSECURE_DSN": "1",
        "RECALL_TRUST_MOD": "strict",
        "RECALL_TABLE": "bad table",
    }
    report = build_report(env, source="given")
    for name in ("RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_REASONING_API_KEY"):
        entry = _entry(report, name)
        assert entry["withheld"] and entry["state"] == "set" and PLACEHOLDER not in str(entry["value"]), (
            f"{name}'s value reached the report"
        )
    assert _entry(report, "RECALL_AUTH_ISSUER_URL")["value"] == "https://user:***@issuer.example", "a URL setting kept its password"
    assert _entry(report, "RECALL_ALLOW_INSECURE_DSN")["value"] == "1", "a security flag was hidden"
    assert report["unknown_recall_variables"] == ["RECALL_TRUST_MOD"]
    assert report["problem"] and "RECALL_TABLE" in report["problem"]
    assert PLACEHOLDER not in json.dumps(report), "a secret value reached the report"
    from recall_mcp.config_report import _scrub_message

    scrubbed = _scrub_message(f"refused: {PLACEHOLDER} is not valid", {"OPENAI_API_KEY": PLACEHOLDER})
    assert PLACEHOLDER not in scrubbed, "a withheld value survived in the startup problem"


def test_a_launch_command_loses_every_credential() -> None:
    """F2."""
    argv = [
        "ssh", "host",
        f"cd x && GITHUB_TOKEN={PLACEHOLDER}1 RECALL_TENANT=memory python -m recall_mcp.server --api-key {PLACEHOLDER}2",
        "--api-key", f"{PLACEHOLDER}3", f"--token={PLACEHOLDER}4", f"postgresql://a:{PLACEHOLDER}5@h/d",
    ]
    joined = " ".join(redact_argv(argv))
    # GITHUB_TOKEN is in no schema: only the credential-NAME rule can withhold it.
    assert f"{PLACEHOLDER}1" not in joined, "GITHUB_TOKEN's value survived"
    assert f"{PLACEHOLDER}2" not in joined, "--api-key inside a command string kept its value"
    assert PLACEHOLDER not in joined
    assert "RECALL_TENANT=memory" in joined, "a non-secret setting was hidden"


def _registered(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    claude = tmp_path / "claude.json"
    claude.write_text(json.dumps({"projects": {str(project): {"mcpServers": {
        "recall-memory": {
            "command": "python", "args": ["-m", "recall_mcp.server"],
            "env": {"RECALL_DSN": f"sqlite:///{PLACEHOLDER}.db", "RECALL_TENANT": "lite-tenant", "VOYAGE_API_KEY": PLACEHOLDER},
        },
        "other-tool": {"command": "npx", "args": ["some-other-mcp"], "env": {"OTHER": "visible-other"}},
    }}}}), encoding="utf-8")
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {
        "recall-code": {"command": "ssh", "args": ["host", f"OPENAI_API_KEY={PLACEHOLDER} python -m recall_mcp.server"]},
    }}), encoding="utf-8")
    return project, claude


def test_the_page_shows_registered_servers_without_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F3."""
    project, claude = _registered(tmp_path)
    monkeypatch.setenv("RECALL_REASONING_API_KEY", PLACEHOLDER)
    real = config.registered_servers
    monkeypatch.setattr(config, "registered_servers", lambda root: real(root, claude_config=claude))
    page = DashboardApp(tmp_path, port=8765, token="tok", project_root=project).handle("GET", "/config", SIGNED)
    body = page.body.decode()
    assert page.status == 200
    assert "MCP server recall-memory (local scope)" in body and "MCP server recall-code (project scope)" in body
    assert "lite-tenant" in body
    assert "other-tool" not in body and "visible-other" not in body, "another MCP server was reported"
    assert PLACEHOLDER not in body, "a secret reached the configuration page"


def test_a_remote_that_fails_is_a_card_and_a_bad_host_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F4."""
    real_remote = config.remote_report
    with pytest.raises(config.ConfigUnavailable, match="is not a host name"):
        real_remote("-oProxyCommand=calc")

    def unreachable(host: str, command: str = config.DEFAULT_REMOTE_COMMAND) -> dict:
        raise config.ConfigUnavailable("ssh exited with 255")

    monkeypatch.setattr(config, "remote_report", unreachable)
    monkeypatch.setattr(config, "registered_servers", lambda root: [])
    app = DashboardApp(tmp_path, port=8765, token="tok", project_root=tmp_path, config_host="corpus-host")
    page = app.handle("GET", "/config", SIGNED)
    assert page.status == 200
    body = page.body.decode()
    assert "Remote: corpus-host" in body and "Not available: ssh exited with 255" in body


def test_report_text_is_escaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F5."""
    monkeypatch.setattr(config, "registered_servers", lambda root: [])
    monkeypatch.setattr(config, "local_report", lambda: build_report({"RECALL_TENANT": "<script>alert(1)</script>"}, source="given"))
    body = DashboardApp(tmp_path, port=8765, token="tok", project_root=tmp_path).handle("GET", "/config", SIGNED).body.decode()
    assert "<script>alert(1)</script>" not in body, "report text reached the page unescaped"
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
