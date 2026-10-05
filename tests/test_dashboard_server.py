"""The dashboard page answers only this machine and this session, and never renders memo text as markup.

Driven through `DashboardApp.handle`, the function the socket handler calls, so no port is opened.

Red proof, 2026-10-05, each mutation alone against `recall/dashboard/server.py`, failing for the stated
reason (JUnit XML), then restored; all six green after:
- S1 the Host check removed: `test_a_request_for_another_host_is_refused`, 303 instead of 403.
- S2 the session cookie not required: `test_the_printed_link_starts_a_session_and_nothing_else_does`,
  200 instead of 403.
- S3 the form token not checked: `test_a_post_without_the_form_token_writes_nothing`, 409 (the
  decision ran and was refused only by the digest) instead of 403.
- S4 `highlight` without escaping: `test_memo_and_report_text_is_escaped_never_rendered`, `<script>`
  in the page.
- S5 report details without escaping: the same test, `<img` in the page.
- S6 the session cookie without HttpOnly: the session test, AssertionError on the cookie.
- S7 the CSP opened to `default-src *`: the escaping test, AssertionError on the header.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest

from recall.dashboard import review
from recall.dashboard.server import COOKIE, DashboardApp, highlight, security_headers
from recall.document import parse_document
from recall.frontmatter import supersedes_targets
from recall.stale_reports import StaleReportQueue, default_queue_path

PORT = 8765
HOST = {"Host": f"127.0.0.1:{PORT}"}
OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "old.md").write_text(f"<script>alert(1)</script> The release {OLD_QUOTE}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}.\n", encoding="utf-8")
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        queue.submit(
            tmp_path,
            stale_source="old.md",
            replacing_source="new.md",
            stale_quote=OLD_QUOTE,
            current_quote=NEW_QUOTE,
            client="mcp:acme",
            task="<img src=x onerror=alert(2)>",
            reported_at=datetime(2026, 10, 5, tzinfo=UTC),
        )
    return tmp_path


@pytest.fixture
def app(corpus: Path) -> DashboardApp:
    return DashboardApp(corpus, port=PORT, token="tok123")


def _signed(**extra: str) -> dict[str, str]:
    return {**HOST, "Cookie": f"{COOKIE}=tok123", **extra}


def _claim(corpus: Path) -> str:
    (item,) = review.build_queue(corpus).items
    return item.claim


def test_a_request_for_another_host_is_refused(app: DashboardApp) -> None:
    for host in ("evil.example:8765", "127.0.0.1:9999", ""):
        response = app.handle("GET", "/?token=tok123", {"Host": host})
        assert response.status == 403
        assert not any(name == "Set-Cookie" for name, _ in response.headers)


def test_the_printed_link_starts_a_session_and_nothing_else_does(app: DashboardApp) -> None:
    assert app.handle("GET", "/", HOST).status == 403
    assert app.handle("GET", "/?token=wrong", HOST).status == 403
    start = app.handle("GET", "/?token=tok123", HOST)
    assert start.status == 303
    headers = dict(start.headers)
    assert headers["Location"] == "/"
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    assert app.handle("GET", "/", _signed()).status == 200


def test_a_post_without_the_form_token_writes_nothing(app: DashboardApp, corpus: Path) -> None:
    form = urlencode({"claim": _claim(corpus), "reviewer": "giulio", "note": "ok", "digest": "x"}).encode()
    response = app.handle("POST", "/accept", _signed(), form)
    assert response.status == 403
    assert supersedes_targets(parse_document((corpus / "new.md").read_text(encoding="utf-8")).meta.get("supersedes")) == ()


def test_memo_and_report_text_is_escaped_never_rendered(app: DashboardApp, corpus: Path) -> None:
    page = app.handle("GET", "/review?" + urlencode({"claim": _claim(corpus)}), _signed()).body.decode()
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<img" not in page and "&lt;img" in page
    assert f"<mark>{NEW_QUOTE}</mark>" in page
    assert "default-src 'none'" in dict(security_headers())["Content-Security-Policy"]


def test_accept_through_the_page_declares_the_edge(app: DashboardApp, corpus: Path) -> None:
    claim = _claim(corpus)
    digest = review.memo_digest(corpus, "new.md")
    form = urlencode({"csrf": "tok123", "claim": claim, "reviewer": "giulio", "note": "confirmed", "digest": digest}).encode()
    response = app.handle("POST", "/accept", _signed(), form)
    assert response.status == 303
    meta = parse_document((corpus / "new.md").read_text(encoding="utf-8")).meta
    assert list(supersedes_targets(meta.get("supersedes"))) == ["old.md"]


def test_highlight_escapes_around_the_mark() -> None:
    assert highlight("a <b> quoted   text here", "quoted text") == "a &lt;b&gt; <mark>quoted   text</mark> here"
