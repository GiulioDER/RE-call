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
- F6 (CCA on #903) a shown value loses every credential shape the report recognises: token or
  user:password userinfo (an `@` inside the password included), credential query parameters,
  quoted and spaced libpq keyword passwords, `Bearer` values; ordinary URLs stay readable.
- F7 a launch command loses `PGPASSWORD=`, `MYSQL_PWD=`, `X_APIKEY=`, the value after
  `--auth-token` or `--access-token`, `--key=`, and an `Authorization: Bearer` header, while
  `--top-k 5` and non-secret settings stay readable.
- F8 the name rule is anchored: token-count, lifetime and key-prefix settings stay visible; every
  schema key shaped like a DSN, API key, token, secret, password or Redis URL is flagged secret.
- F9 a failed report shows a fixed cause from its exit code, never the child's stderr; an empty
  or spaced host is refused before ssh runs.
- F10 a server launched through ssh or a shell is marked not checked, not valid; unknown
  `RECALL_*` names are not called unread; `?all=1` shows defaults.
Red proofs for F6 to F10 are in each test's docstring.

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
import re
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
    # The whole userinfo goes, user name included (CCA on #903, SEC-001).
    assert _entry(report, "RECALL_AUTH_ISSUER_URL")["value"] == "https://***@issuer.example", "a URL setting kept its password"
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


# ------------------------------------------------------------------ CCA on #903


@pytest.mark.parametrize(
    "value",
    [
        f"https://{PLACEHOLDER}@gateway.example/v1",
        f"https://h.example/translate?key={PLACEHOLDER}&lang=it",
        f"https://h.example/x?access_token={PLACEHOLDER}",
        f"https://b.example/o?X-Amz-Date=1&X-Amz-Signature={PLACEHOLDER}",
        f"postgresql://u:pa@{PLACEHOLDER}@db/recall",
        f"host=db password='a {PLACEHOLDER}' port=5432",
        f"host=db password = {PLACEHOLDER}",
        f"Authorization: Bearer {PLACEHOLDER}",
        f"postgresql://h/db?sslmode=require&sslpassword={PLACEHOLDER}",
        f"host=a;sslpassword={PLACEHOLDER}",
        f"postgresql://u:pa#{PLACEHOLDER}@db/recall",
        f"postgresql://u:pa?{PLACEHOLDER}@db/recall",
        f"https://h.example/x?token=ab#{PLACEHOLDER}",
        f"https://u:p@h.example?token=ab@{PLACEHOLDER}",
        f"https://h.example/x?token=a#Bearer {PLACEHOLDER}",
        f"Authorization: Bearer token=?{PLACEHOLDER}",
        f"Bearer https://u:{PLACEHOLDER}@h.example/x",
        f"Basic password= {PLACEHOLDER}",
        f"Bearer Bearer {PLACEHOLDER}",
        f"?token=https://=;@{PLACEHOLDER}",
        f"h&key=Basic sig=Bearer {PLACEHOLDER}",
    ],
)
def test_a_shown_value_loses_every_credential_shape(value: str) -> None:
    """F6. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: restoring the
    #903 `redact_value` (a `user:password@` rule and an unquoted keyword rule only) failed the first
    eight parameters at "a credential survived in a shown value" (#903 itself redacts the
    `sslpassword` and the `#`/`?` ones). Restored, passes. The readability side
    (`test_ordinary_urls_stay_readable`) failed on `?author=` with a query rule that matched a
    credential word anywhere in the parameter name. The `sslpassword` parameters (found in review)
    failed at "a credential survived in a shown value" without `(?:ssl)?` in the query rule. The
    `#` and `?` parameters (found by the security review of the committed fix) failed the same way
    with `?#` excluded from the userinfo and `#` ending a query value; #903 itself redacted them.
    `https://u:p@...?token=ab@...` (found by the audit of that fix) failed with the userinfo rule
    run before the query rule; the readable `?x=a@b` URLs failed with a userinfo rule that let `?`
    in before a `:`. `?token=a#Bearer ...` (found in review) failed with the authorization rule
    applied only after the span step, and the four `Bearer`/`Basic` parameters after it failed with
    the rules applied one after another instead of as one set of spans. `?token=https://=;@...`
    failed with a single pass of the span step, and `Basic sig=Bearer ...` with authorization
    matches that could not overlap."""
    from recall_mcp.config_report import redact_value

    assert PLACEHOLDER not in redact_value(value), "a credential survived in a shown value"
    shown = build_report({"RECALL_REASONING_BASE_URL": value}, source="given")
    assert PLACEHOLDER not in json.dumps(shown), "a credential survived in a shown value"


def test_ordinary_urls_stay_readable() -> None:
    """F6, the other side: nothing is replaced in a URL that carries no credential."""
    from recall_mcp.config_report import redact_value

    for plain in ("https://api.openai.com/v1", "https://h.example/x?lang=it&top=5", "https://host/path/@user",
                  "https://h.example/x?author=me&design=a&keyword=k", "https://host.example?x=a@b",
                  "postgresql://dbhost?user=bob@corp.example", "https://host.example#me@x"):
        assert redact_value(plain) == plain


def test_a_launch_command_loses_compound_credential_names_and_flags() -> None:
    """F7. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: restoring the
    #903 `_CREDENTIAL_NAME` (a separator required before the word) failed at "PGPASSWORD's value
    survived"; restoring the #903 flag lists failed at "--auth-token kept its value". Restored,
    passes. Bounding the flag word only after review: the suffix-unbounded rule failed at "an
    ordinary flag hid the argument after it" (`--no-auth --port 1` lost `--port`), and a rule
    without `bearer` failed at "a credential survived in the launch command"."""
    argv = [
        "env", f"PGPASSWORD={PLACEHOLDER}1", f"MYSQL_PWD={PLACEHOLDER}2", f"X_APIKEY={PLACEHOLDER}3",
        "RECALL_TRUST_MODE=strict", "srv", "--auth-token", f"{PLACEHOLDER}4", "--top-k", "5",
        "ssh", "h", f"srv --access-token {PLACEHOLDER}5 --key={PLACEHOLDER}6 --log-level info",
        "-H", f"Authorization: Bearer {PLACEHOLDER}7", "--bearer", f"{PLACEHOLDER}8",
        "--no-auth", "--port", "1", "--sort-key", "date",
    ]
    joined = " ".join(redact_argv(argv))
    assert f"{PLACEHOLDER}1" not in joined, "PGPASSWORD's value survived"
    assert f"{PLACEHOLDER}4" not in joined, "--auth-token kept its value"
    assert PLACEHOLDER not in joined, "a credential survived in the launch command"
    assert "RECALL_TRUST_MODE=strict" in joined and "--top-k 5" in joined and "--log-level info" in joined
    assert "--no-auth --port 1" in joined and "--sort-key date" in joined, "an ordinary flag hid the argument after it"


def test_the_name_rule_is_anchored_and_the_schema_flags_credentials() -> None:
    """F8. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: dropping the `$`
    anchor of `_CREDENTIAL_NAME` failed at "a non-secret setting was withheld"; setting
    `secret=False` on RECALL_MIGRATION_DSN failed at "a credential-shaped schema key is not
    flagged secret". Restored, passes."""
    import re

    from recall_mcp.config_report import withheld
    from recall_mcp.settings import ENVIRONMENT_SCHEMA

    for name in ("RECALL_OIDC_MAX_TOKEN_LIFETIME_SECONDS", "RECALL_RATE_LIMIT_KEY_PREFIX",
                 "RECALL_REASONING_ANSWER_MAX_TOKENS", "RECALL_ALLOW_INSECURE_DSN"):
        assert not withheld(name), f"a non-secret setting was withheld: {name}"
    assert withheld("RECALL_RESTORE_VALIDATION_DSN"), "a DSN outside the schema was shown"
    shaped = re.compile(r"_(?:DSN|API_KEY|TOKEN|SECRET|PASSWORD)$|REDIS_URL$")
    unflagged = [s.name for s in ENVIRONMENT_SCHEMA
                 if shaped.search(s.name) and s.name != "RECALL_ALLOW_INSECURE_DSN" and not s.secret]
    assert not unflagged, f"a credential-shaped schema key is not flagged secret: {unflagged}"


def test_a_failed_report_shows_a_cause_never_its_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """F9. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: restoring the
    #903 message (the last stderr line) failed at "the child's stderr reached the message".
    Restored, passes."""
    import subprocess

    def failing(argv, **_kw):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(argv, 255, "", f"./.env: line 3: {PLACEHOLDER}: command not found\n")

    monkeypatch.setattr(config.subprocess, "run", failing)
    with pytest.raises(config.ConfigUnavailable) as caught:
        config.remote_report("corpus-host")
    assert PLACEHOLDER not in str(caught.value), "the child's stderr reached the message"
    assert "ssh could not connect" in str(caught.value)
    for host in ("", "two words"):
        with pytest.raises(config.ConfigUnavailable, match="is not a host name"):
            config.remote_report(host)


def test_a_server_launched_through_ssh_is_not_called_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F10. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: `launcher_of`
    returning None failed at "an ssh-launched server was called valid"; the #903 wording failed at
    "unknown names were called unread". Restored, passes."""
    entry = {"command": "bash", "args": ["-lc", "ssh host '. ~/.env && exec python -m recall_mcp.server'"]}
    server = config.RegisteredServer("recall-memory", "project", entry)
    monkeypatch.setattr(config, "registered_servers", lambda root: [server])
    monkeypatch.setattr(config, "local_report", lambda: build_report({"RECALL_DASHBOARD_CONFIG_HOST": "h"}, source="given"))
    monkeypatch.setattr(config, "_run", lambda argv, stdin=None: build_report({}, source="given", argv=[]))
    body = DashboardApp(tmp_path, port=8765, token="tok", project_root=tmp_path).handle("GET", "/config", SIGNED).body.decode()
    card = body.split("MCP server recall-memory", 1)[1]
    assert "Not checked" in card and "Valid" not in card.split("</div>", 1)[0], "an ssh-launched server was called valid"
    assert "nothing reads them" not in body, "unknown names were called unread"
    everything = DashboardApp(tmp_path, port=8765, token="tok", project_root=tmp_path).handle("GET", "/config?all=1", SIGNED)
    assert "default" in everything.body.decode()
    assert config.launcher_of(["bash", "-lc", "ssh host 'exec python -m recall_mcp.server'"]) == "ssh"
    assert config.launcher_of(["python", "-m", "recall_mcp.server"]) is None


#: #903's own redaction rules, as merged at 8c11a14ae, kept here as the reference: no later change
#: may show a value these two hid.
_REFERENCE_URL_PASSWORD = re.compile(r"(?P<head>[a-z][a-z0-9+.-]*://[^/@\s:]*:)(?P<pw>[^/@\s]+)(?P<at>@)", re.I)
_REFERENCE_KEYWORD_PASSWORD = re.compile(
    r"(?P<head>(?:^|[\s?&;])(?:password|passwd|sslpassword|api_key|apikey|token|secret)=)(?P<pw>[^\s&;]+)", re.I
)


def _reference_redact(value: str) -> str:
    value = _REFERENCE_URL_PASSWORD.sub(lambda m: f"{m.group('head')}***{m.group('at')}", value)
    return _REFERENCE_KEYWORD_PASSWORD.sub(lambda m: f"{m.group('head')}***", value)


def _reference_shapes() -> list[str]:
    """Values built from every shape the reference rules cover, with awkward characters around
    the secret: the URL password rule's three forms, and each keyword in each position."""
    chars = ["", "@", "?", "#", ":", "&", ";", "=", "%", "!", "&token=y", ";key=y", "?sig=y"]
    out = []
    for before in chars:
        for after in chars:
            secret = f"p{before}{PLACEHOLDER}{after}q"
            out += [f"postgresql://u:{secret}@db/recall", f"postgresql://u:{secret}@db", f"redis://:{secret}@h:6379/0"]
    for word in ("password", "passwd", "sslpassword", "api_key", "apikey", "token", "secret", "PASSWORD", "Token"):
        for sep in ("", "?", "&", ";", " "):
            for c in ("", "#", "?", "@", "=", ":", "'", '"', "%"):
                out += [
                    f"x{sep}{word}=v{c}{PLACEHOLDER}", f"{word}=v{c}{PLACEHOLDER}",
                    f"https://h/x?a=1{sep if sep in '&;' else '&'}{word}=v{c}{PLACEHOLDER}",
                    f"host=db {word}=v{c}{PLACEHOLDER} port=1",
                ]
    return out


def test_every_shape_903_redacted_is_still_redacted() -> None:
    r"""F11: parity with #903's own rules. Every generated value they hid is still hidden.

    Failure mode caught: a later narrowing of a regex (the committed fix after #903 excluded `?#`
    from the URL password and ended a query value at `#`) shows credentials #903 withheld, with
    every hand-picked test still green.

    Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: restoring a5399dc96's
    `_URL_USERINFO` (`[^\s/?#]*`), and separately its `_QUERY_CREDENTIAL` value (`[^&\s#;]+`), each
    failed at "a shape #903 redacted is shown again" (141 and 63 shapes); applying the two rules one
    after the other, query first, failed on passwords holding `&token=`, `;key=` or `?sig=`.
    Restored, passes.
    """
    from recall_mcp.config_report import redact_value

    shapes = _reference_shapes()
    assert len(shapes) > 1500
    leaked = [v for v in shapes if PLACEHOLDER not in _reference_redact(v) and PLACEHOLDER in redact_value(v)]
    assert not leaked, f"a shape #903 redacted is shown again ({len(leaked)}): {leaked[:5]}"
