"""The HTTP side of `recall dashboard`: a local page, a person's decisions, nothing else.

Standard library only (`http.server`): a single-user page on this machine needs no framework, and
adding none keeps the package's dependencies where they are.

Safety, each enforced in `DashboardApp.handle` and each covered by a test:

* **This machine only.** The server binds 127.0.0.1, and a request whose `Host` header is not this
  server's own address is refused, which stops a web page elsewhere reaching it by DNS rebinding.
* **A person's decisions.** Every request needs the per-launch token, first from the printed URL,
  then from an HttpOnly, SameSite=Strict cookie. Every POST also carries it as a form field, so a
  form on another site cannot submit one. There is no JSON API to script against.
* **Corpus text is data.** Every string from a memo, a report or a proposal is HTML-escaped before it
  is written, and the Content-Security-Policy allows no script at all.
"""

from __future__ import annotations

import html
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from recall.dashboard import review

COOKIE = "recall_dashboard"
REVIEWER_COOKIE = "recall_reviewer"
MAX_MEMO_CHARS = 20_000
_CSP = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def highlight(text: str, quote: str) -> str:
    """`text` escaped, with the first whitespace-insensitive match of `quote` wrapped in <mark>."""
    words = quote.split()
    if not words:
        return _e(text)
    match = re.search(r"\s+".join(re.escape(word) for word in words), text)
    if match is None:
        return _e(text)
    return _e(text[: match.start()]) + "<mark>" + _e(match.group(0)) + "</mark>" + _e(text[match.end() :])


_STYLE = """
:root { --ink:#1d2024; --muted:#5d6670; --line:#d9dde2; --bg:#fbfbfa; --card:#fff; --accent:#2f6f4e; --warn:#9b3b1b; --mark:#fff1a8; }
@media (prefers-color-scheme: dark) { :root { --ink:#e8eaed; --muted:#a3abb4; --line:#3a3f45; --bg:#16181b; --card:#1f2226; --accent:#6fbf93; --warn:#f08a65; --mark:#5c4f12; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:24px 0 8px; }
a { color:var(--accent); } .muted { color:var(--muted); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:10px 0; }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:12px; } @media (max-width:760px) { .pair { grid-template-columns:1fr; } }
pre { white-space:pre-wrap; word-break:break-word; font:13px/1.45 ui-monospace, monospace; margin:0; max-height:60vh; overflow:auto; }
mark { background:var(--mark); color:inherit; } .tag { font-size:12px; border:1px solid var(--line); border-radius:999px; padding:1px 8px; margin-right:4px; }
label { display:block; font-size:13px; margin:8px 0 2px; } input, textarea { width:100%; font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); }
button { font:inherit; padding:7px 14px; border-radius:6px; border:1px solid var(--line); background:var(--card); color:var(--ink); cursor:pointer; margin-top:10px; }
button.accept { border-color:var(--accent); color:var(--accent); } button.reject { border-color:var(--warn); color:var(--warn); }
.error { border-color:var(--warn); color:var(--warn); } .ok { border-color:var(--accent); }
.diff { font:13px ui-monospace, monospace; color:var(--accent); }
"""


def _page(title: str, body: str) -> bytes:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_STYLE}</style></head><body><main>{body}</main></body></html>"
    ).encode("utf-8")


class DashboardApp:
    """Routing, authentication and rendering, independent of the socket so tests can drive it."""

    def __init__(self, root: Path, *, port: int, token: str | None = None) -> None:
        self.root = root.resolve()
        self.port = port
        self.token = token or secrets.token_urlsafe(32)
        self._hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    # ------------------------------------------------------------------ entry point

    def handle(self, method: str, target: str, headers: Mapping[str, str], body: bytes = b"") -> Response:
        host = headers.get("Host") or headers.get("host") or ""
        if host not in self._hosts:
            return self._error(HTTPStatus.FORBIDDEN, "This page only answers on this machine's own address.")
        url = urlsplit(target)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        cookies = SimpleCookie(headers.get("Cookie") or headers.get("cookie") or "")
        has_cookie = COOKIE in cookies and secrets.compare_digest(cookies[COOKIE].value, self.token)

        if method == "GET" and "token" in query:
            if not secrets.compare_digest(query["token"], self.token):
                return self._error(HTTPStatus.FORBIDDEN, "That link is not this session's. Use the one `recall dashboard` printed.")
            rest = {k: v for k, v in query.items() if k != "token"}
            location = url.path + (("?" + urlencode(rest)) if rest else "")
            return Response(
                HTTPStatus.SEE_OTHER,
                b"",
                (
                    ("Location", location),
                    ("Set-Cookie", f"{COOKIE}={self.token}; HttpOnly; SameSite=Strict; Path=/"),
                ),
            )
        if not has_cookie:
            return self._error(HTTPStatus.FORBIDDEN, "Open the link `recall dashboard` printed to start a session.")

        if method == "GET" and url.path == "/":
            return self._queue_page(query.get("done"))
        if method == "GET" and url.path == "/review":
            return self._review_page(query.get("claim", ""), cookies)
        if method == "POST" and url.path in ("/accept", "/reject"):
            form = {k: v[0] for k, v in parse_qs(body.decode("utf-8", "replace")).items()}
            if not secrets.compare_digest(form.get("csrf", ""), self.token):
                return self._error(HTTPStatus.FORBIDDEN, "This form did not come from this page.")
            return self._decide(url.path[1:], form)
        return self._error(HTTPStatus.NOT_FOUND, "No such page.")

    # ------------------------------------------------------------------ pages

    def _queue_page(self, done: str | None) -> Response:
        queue = review.build_queue(self.root)
        rows = []
        for item in queue.items:
            tags = "".join(f"<span class='tag'>{_e(o)}</span>" for o in item.origins)
            rows.append(
                "<div class='card'>"
                f"{tags}<a href='/review?{_e(urlencode({'claim': item.claim}))}'>"
                f"<strong>{_e(item.replacing)}</strong> replaces <strong>{_e(item.stale)}</strong></a>"
                f"<div class='muted'>{_e(item.stale_quote[:160])}</div>"
                f"<div class='muted'>{_e(item.current_quote[:160])}</div>"
                "</div>"
            )
        notes = "".join(f"<div class='muted'>{_e(note)}</div>" for note in queue.notes)
        flash = f"<div class='card ok'>{_e(done)}</div>" if done else ""
        empty = "" if rows else "<div class='card muted'>Nothing to review.</div>"
        body = (
            "<h1>Review queue</h1>"
            f"<div class='muted'>{_e(self.root)} &middot; {len(rows)} pending</div>"
            f"{flash}{''.join(rows)}{empty}<h2>Sources</h2>{notes or '<div class=muted>agent reports only</div>'}"
        )
        return self._html(_page("RE-call review queue", body))

    def _find(self, claim: str) -> review.QueueItem | None:
        return next((i for i in review.build_queue(self.root).items if i.claim == claim), None)

    def _memo(self, name: str) -> str:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_MEMO_CHARS]

    def _review_page(self, claim: str, cookies: SimpleCookie) -> Response:
        item = self._find(claim)
        if item is None:
            return self._error(HTTPStatus.NOT_FOUND, "That claim is no longer pending.")
        now = datetime.now(UTC)
        try:
            plan = review.preview(self.root, item, now)
            digest = review.memo_digest(self.root, plan.edit_file)
            planned = f"<div class='diff'>{_e(plan.edit_file)}: + {_e(plan.key)}: {_e(plan.value)}</div>"
        except review.ReviewRefused as exc:
            digest, planned = "", f"<div class='card error'>Cannot be applied: {_e(exc)}</div>"
        reviewer = cookies[REVIEWER_COOKIE].value if REVIEWER_COOKIE in cookies else ""
        details = "".join(f"<div class='muted'>{_e(d)}</div>" for d in item.details)
        common = (
            f"<input type='hidden' name='csrf' value='{_e(self.token)}'>"
            f"<input type='hidden' name='claim' value='{_e(item.claim)}'>"
            f"<label>Your name</label><input name='reviewer' value='{_e(reviewer)}' required>"
            "<label>Note (why)</label><textarea name='note' rows='2' required></textarea>"
        )
        accept = (
            f"<form method='post' action='/accept'>{common}"
            f"<input type='hidden' name='digest' value='{_e(digest)}'>"
            "<button class='accept' type='submit'>Accept and write the declaration</button></form>"
            if digest
            else ""
        )
        reject = f"<form method='post' action='/reject'>{common}<button class='reject' type='submit'>Reject</button></form>"
        body = (
            "<div class='muted'><a href='/'>&larr; queue</a></div>"
            f"<h1>{_e(item.replacing)} replaces {_e(item.stale)}?</h1>{details}"
            "<div class='pair'>"
            f"<div class='card'><div class='muted'>older: {_e(item.stale)}</div><pre>{highlight(self._memo(item.stale), item.stale_quote)}</pre></div>"
            f"<div class='card'><div class='muted'>newer: {_e(item.replacing)}</div><pre>{highlight(self._memo(item.replacing), item.current_quote)}</pre></div>"
            "</div>"
            f"<h2>What accepting writes</h2>{planned}"
            "<div class='muted'>The served memory follows after the next index build.</div>"
            f"<div class='pair'><div class='card'>{accept}</div><div class='card'>{reject}</div></div>"
        )
        return self._html(_page("Review a claim", body))

    def _decide(self, action: str, form: Mapping[str, str]) -> Response:
        item = self._find(form.get("claim", ""))
        if item is None:
            return self._error(HTTPStatus.CONFLICT, "That claim is no longer pending.")
        reviewer, note, now = form.get("reviewer", ""), form.get("note", ""), datetime.now(UTC)
        try:
            if action == "accept":
                plan = review.accept(
                    self.root, item, reviewer=reviewer, note=note, shown_digest=form.get("digest", ""), now=now
                )
                done = f"Declared: {plan.edit_file} now supersedes {plan.value}."
            else:
                review.reject(self.root, item, reviewer=reviewer, note=note, now=now)
                done = f"Rejected: {item.replacing} does not replace {item.stale}."
        except review.ReviewRefused as exc:
            return self._error(HTTPStatus.CONFLICT, f"Nothing was written: {exc}")
        cookie = SimpleCookie()
        cookie[REVIEWER_COOKIE] = reviewer
        cookie[REVIEWER_COOKIE]["samesite"] = "Strict"
        cookie[REVIEWER_COOKIE]["path"] = "/"
        return Response(
            HTTPStatus.SEE_OTHER,
            b"",
            (("Location", "/?" + urlencode({"done": done})), ("Set-Cookie", cookie.output(header="").strip())),
        )

    # ------------------------------------------------------------------ helpers

    def _html(self, body: bytes, status: int = HTTPStatus.OK) -> Response:
        return Response(status, body, (("Content-Type", "text/html; charset=utf-8"),))

    def _error(self, status: int, message: str) -> Response:
        page = _page("RE-call dashboard", f"<div class='card error'>{_e(message)}</div><div class='muted'><a href='/'>queue</a></div>")
        return self._html(page, status)


def security_headers() -> tuple[tuple[str, str], ...]:
    return (
        ("Content-Security-Policy", _CSP),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        ("Cache-Control", "no-store"),
    )


def serve(app: DashboardApp) -> ThreadingHTTPServer:
    """A server bound to 127.0.0.1 that hands every request to `app`."""

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(min(length, 1_000_000)) if length else b""
            response = app.handle(method, self.path, dict(self.headers.items()), body)
            self.send_response(response.status)
            for name, value in response.headers + security_headers():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)

        def do_GET(self) -> None:  # noqa: N802  # http.server's name
            self._respond("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._respond("POST")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return  # request lines would carry the token on the first visit

    return ThreadingHTTPServer(("127.0.0.1", app.port), Handler)
