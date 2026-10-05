"""The HTTP side of `recall dashboard`: a local page, a person's decisions, nothing else.

Standard library only (`http.server`): a single-user page on this machine needs no framework, and
adding none keeps the package's dependencies where they are.

Safety, each enforced in `DashboardApp.handle` and each covered by a test:

* **This machine only.** The server binds 127.0.0.1, and a request whose `Host` header is not this
  server's own address is refused, which stops a web page elsewhere reaching it by DNS rebinding.
* **A person's decisions.** Every request needs the per-launch token, first from the printed URL,
  then from an HttpOnly, SameSite=Strict cookie. Every POST also carries it as a form field, so a
  form on another site cannot submit one. The only JSON route is the read-only graph data.
* **Corpus text is data.** Every string from a memo, a report or a proposal is HTML-escaped before it
  is written. The only script is this package's own `static/graph.js`, which builds the page with
  `textContent`; the Content-Security-Policy allows no inline script and nothing from elsewhere.
"""

from __future__ import annotations

import html
import json
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
STATIC = Path(__file__).resolve().parent / "static"
_STATIC_FILES = {"/static/graph.js": ("graph.js", "text/javascript; charset=utf-8")}
_CSP = (
    "default-src 'none'; script-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; "
    "img-src 'self' data:; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


EXCERPT_CHARS = 600


def excerpt(text: str, quote: str) -> str:
    """The lines around the quote, escaped and marked, so the evidence is on screen without scrolling.

    Whole lines, at most about EXCERPT_CHARS either side; an ellipsis line marks what was cut. Falls
    back to the opening of the memo when the quote is not found.
    """
    words = quote.split()
    match = re.search(r"\s+".join(re.escape(word) for word in words), text) if words else None
    newline = "\n"
    if match is None:
        cut = len(text) > EXCERPT_CHARS * 2
        return _e(text[: EXCERPT_CHARS * 2]) + (newline + "…" if cut else "")
    start = text.rfind(newline, 0, max(0, match.start() - EXCERPT_CHARS)) + 1
    end = text.find(newline, min(len(text), match.end() + EXCERPT_CHARS))
    end = len(text) if end == -1 else end
    window = text[start:end]
    head = "…" + newline if start > 0 else ""
    tail = newline + "…" if end < len(text) else ""
    return head + highlight(window, quote) + tail


def highlight(text: str, quote: str) -> str:
    """`text` escaped, with the first whitespace-insensitive match of `quote` wrapped in <mark>."""
    words = quote.split()
    if not words:
        return _e(text)
    match = re.search(r"\s+".join(re.escape(word) for word in words), text)
    if match is None:
        return _e(text)
    return _e(text[: match.start()]) + "<mark>" + _e(match.group(0)) + "</mark>" + _e(text[match.end() :])


# The look follows RE-call's site: a green-black ground, warm cream ink, one amber signal, and
# monospace labels set like an instrument panel. Local fonts only: the page loads nothing from
# the network, which suits a viewer of private memory and keeps the policy above strict.
_STYLE = """
:root {
  --ground:#0e100f; --surface-1:#141714; --surface-2:#191c19; --surface-3:#20241f;
  --ink:#f4f1e8; --ink-soft:#d0d0c6; --ink-muted:#9a9c90; --line:rgba(96,108,96,.38); --line-strong:rgba(120,132,118,.7); --grid:rgba(96,108,96,.11);
  --signal:#d7a52a; --signal-soft:rgba(215,165,42,.14); --rust:#d0603f; --rust-soft:rgba(208,96,63,.14); --sage:#8fb08a;
  --node-current:#efe9d8; --node-superseded:#d0603f; --node-expired:#5d6258; --node-pending:#e3b23c;
  --edge-link:#a9b7a1; --edge-supersedes:#d0603f; --edge-pending:#e3b23c; --mark:rgba(215,165,42,.32);
  --font-display:"Geist","Segoe UI Variable Display","Segoe UI",ui-sans-serif,system-ui,sans-serif;
  --font-mono:"Geist Mono","Cascadia Mono","JetBrains Mono",ui-monospace,Consolas,monospace;
  --ease:cubic-bezier(.16,1,.3,1); color-scheme:dark;
}
@media (prefers-color-scheme: light) {
  :root {
    --ground:#f3efe4; --surface-1:#fbf8f0; --surface-2:#f0ebde; --surface-3:#e7e1d1;
    --ink:#151815; --ink-soft:#3a3f39; --ink-muted:#6c6f64; --line:rgba(70,80,71,.22); --line-strong:rgba(70,80,71,.45); --grid:rgba(70,80,71,.07);
    --signal:#9c720f; --signal-soft:rgba(156,114,15,.12); --rust:#b0462a; --rust-soft:rgba(176,70,42,.1); --sage:#4f7a4a;
    --node-current:#2b302a; --node-superseded:#b0462a; --node-expired:#b7b4a6; --node-pending:#c08a14;
    --edge-link:#6f7f68; --edge-supersedes:#b0462a; --edge-pending:#c08a14; --mark:rgba(192,138,20,.28); color-scheme:light;
  }
}
* { box-sizing:border-box; }
html, body { height:100%; }
body { margin:0; background:var(--ground); color:var(--ink); font:15px/1.55 var(--font-display); -webkit-font-smoothing:antialiased;
  background-image: radial-gradient(1200px 600px at 85% -10%, var(--signal-soft), transparent 60%),
    linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: auto, 56px 56px, 56px 56px; background-attachment: fixed; }
a { color:var(--signal); text-decoration:none; } a:hover { text-decoration:underline; text-underline-offset:3px; }
.shell { display:grid; grid-template-columns:236px minmax(0,1fr); min-height:100vh; }
.side { position:sticky; top:0; height:100vh; display:flex; flex-direction:column; gap:28px; padding:26px 18px;
  background:color-mix(in srgb, var(--surface-1) 88%, transparent); border-right:1px solid var(--line); backdrop-filter:blur(6px); }
.brand { display:flex; flex-direction:column; gap:6px; padding:0 8px; }
.brand b { font-size:22px; letter-spacing:-.02em; font-weight:650; }
.brand b i { font-style:normal; color:var(--signal); }
.brand .rule { height:2px; width:56px; background:linear-gradient(90deg,var(--signal),#f0be4a,var(--signal)); border-radius:2px; }
.eyebrow { font:500 11px/1 var(--font-mono); letter-spacing:.16em; text-transform:uppercase; color:var(--ink-muted); }
nav { display:flex; flex-direction:column; gap:2px; }
nav a { display:flex; align-items:center; gap:10px; padding:9px 10px; border-radius:8px; color:var(--ink-soft); border:1px solid transparent; }
nav a:hover { background:var(--surface-2); text-decoration:none; }
nav a.on { background:var(--surface-3); color:var(--ink); border-color:var(--line); box-shadow:inset 2px 0 0 var(--signal); }
nav .glyph { width:18px; text-align:center; font-family:var(--font-mono); color:var(--signal); }
nav .badge { margin-left:auto; font:600 11px/1 var(--font-mono); padding:4px 7px; border-radius:999px; background:var(--signal-soft); color:var(--signal); }
.side .foot { margin-top:auto; display:flex; flex-direction:column; gap:8px; padding:0 8px; }
.side .foot code { font:12px/1.4 var(--font-mono); color:var(--ink-muted); word-break:break-all; }
.live { display:inline-flex; align-items:center; gap:8px; font:11px var(--font-mono); letter-spacing:.12em; text-transform:uppercase; color:var(--sage); }
.live::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--sage); box-shadow:0 0 0 4px color-mix(in srgb, var(--sage) 20%, transparent); }
main { padding:34px clamp(16px,4vw,48px) 64px; min-width:0; }
.head { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; padding-bottom:18px; margin-bottom:22px; border-bottom:1px solid var(--line); }
.head h1 { margin:8px 0 4px; font-size:clamp(24px,3vw,32px); letter-spacing:-.025em; font-weight:620; }
.head p { margin:0; color:var(--ink-muted); max-width:62ch; }
.card { background:var(--surface-1); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:12px 0;
  animation:rise .55s var(--ease) both; }
.card:nth-child(2){animation-delay:.04s}.card:nth-child(3){animation-delay:.08s}.card:nth-child(4){animation-delay:.12s}.card:nth-child(5){animation-delay:.16s}
@keyframes rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
.claim { display:grid; grid-template-columns:auto 1fr auto; gap:6px 16px; align-items:start; text-decoration:none; color:inherit; }
a.claim:hover { text-decoration:none; border-color:var(--line-strong); background:var(--surface-2); }
.claim .arrow { grid-row:span 2; align-self:center; font:20px var(--font-mono); color:var(--rust); }
.claim .names { font-weight:560; } .claim .names em { font-style:normal; color:var(--ink-muted); font-weight:400; }
.claim .quote { grid-column:2; color:var(--ink-muted); font-size:13.5px; }
.claim .quote s { color:var(--rust); text-decoration-thickness:1px; } .claim .go { grid-row:span 2; align-self:center; color:var(--signal); font:12px var(--font-mono); }
.tag { font:500 10.5px/1 var(--font-mono); letter-spacing:.1em; text-transform:uppercase; border:1px solid var(--line-strong); border-radius:999px; padding:4px 8px; color:var(--ink-soft); }
.tags { display:flex; gap:6px; margin-bottom:6px; }
.muted { color:var(--ink-muted); font-size:13px; }
.notes { margin-top:30px; } .notes div { font:12.5px/1.6 var(--font-mono); color:var(--ink-muted); }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:14px; } @media (max-width:880px) { .pair { grid-template-columns:1fr; } }
.memo-card { padding:0; overflow:hidden; } .memo-card header { display:flex; justify-content:space-between; gap:10px; padding:10px 14px; border-bottom:1px solid var(--line); background:var(--surface-2); }
.memo-card header b { font:12px var(--font-mono); } .older header { box-shadow:inset 3px 0 0 var(--rust); } .newer header { box-shadow:inset 3px 0 0 var(--sage); }
details.full { border-top:1px solid var(--line); } details.full summary { cursor:pointer; padding:9px 14px; font:11px var(--font-mono); letter-spacing:.12em; text-transform:uppercase; color:var(--ink-muted); }
details.full pre { border-top:1px solid var(--line); }
pre { white-space:pre-wrap; word-break:break-word; font:12.5px/1.6 var(--font-mono); margin:0; padding:14px; max-height:58vh; overflow:auto; color:var(--ink-soft); }
mark { background:var(--mark); color:var(--ink); border-radius:3px; padding:0 2px; }
.diff { font:13px var(--font-mono); padding:12px 14px; border-radius:10px; background:var(--surface-2); border:1px dashed var(--line-strong); }
.diff ins { text-decoration:none; color:var(--sage); }
.decide { display:grid; grid-template-columns:1fr 1fr; gap:14px; } @media (max-width:880px) { .decide { grid-template-columns:1fr; } }
label { display:block; font:500 11px var(--font-mono); letter-spacing:.12em; text-transform:uppercase; color:var(--ink-muted); margin:10px 0 6px; }
input[type=text], input:not([type]), textarea, input[type=search] { width:100%; font:inherit; padding:9px 11px; border:1px solid var(--line-strong); border-radius:8px; background:var(--surface-2); color:var(--ink); }
input:focus, textarea:focus, button:focus-visible { outline:2px solid var(--signal); outline-offset:1px; }
button { font:600 13px var(--font-display); padding:10px 16px; border-radius:8px; border:1px solid var(--line-strong); background:var(--surface-2); color:var(--ink); cursor:pointer; margin-top:12px; transition:transform .15s var(--ease), background .15s; }
button:hover { transform:translateY(-1px); }
button.accept { background:var(--signal); color:#141714; border-color:var(--signal); } button.reject { color:var(--rust); border-color:color-mix(in srgb, var(--rust) 60%, transparent); background:var(--rust-soft); }
.error { border-color:var(--rust); background:var(--rust-soft); } .ok { border-color:color-mix(in srgb, var(--sage) 60%, transparent); }
.empty { text-align:center; padding:48px 18px; color:var(--ink-muted); } .empty b { display:block; color:var(--ink); font-size:17px; margin-bottom:4px; }
h2.sec { font:500 11px var(--font-mono); letter-spacing:.16em; text-transform:uppercase; color:var(--ink-muted); margin:28px 0 10px; }
/* graph */
.graph-main { padding-bottom:24px; display:flex; flex-direction:column; height:100vh; }
.toolbar { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; }
.searchbox { position:relative; flex:1 1 280px; max-width:440px; } .searchbox kbd { position:absolute; right:10px; top:50%; transform:translateY(-50%); font:11px var(--font-mono); color:var(--ink-muted); border:1px solid var(--line-strong); border-radius:4px; padding:1px 6px; }
.chip { display:inline-flex; align-items:center; gap:8px; font:12px var(--font-mono); color:var(--ink-soft); border:1px solid var(--line); border-radius:999px; padding:6px 11px; cursor:pointer; user-select:none; background:var(--surface-1); }
.chip input { accent-color:var(--signal); margin:0; } .sw { width:16px; height:0; border-top:2px solid; display:inline-block; }
.sw.link { border-color:var(--edge-link); } .sw.supersedes { border-color:var(--edge-supersedes); } .sw.pending { border-color:var(--edge-pending); border-top-style:dashed; }
.legend { display:flex; flex-wrap:wrap; gap:14px; font:12px var(--font-mono); color:var(--ink-muted); }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
.dot.current { background:var(--node-current); } .dot.superseded { border:1.5px solid var(--node-superseded); } .dot.expired { background:var(--node-expired); } .dot.pending { background:var(--node-pending); box-shadow:0 0 0 3px color-mix(in srgb, var(--node-pending) 25%, transparent); }
#counts { display:flex; flex-wrap:wrap; gap:4px 8px; align-items:baseline; font:12.5px var(--font-mono); margin:14px 0 12px; }
#counts .count-label { color:var(--ink-muted); } #counts strong { color:var(--ink); margin-right:12px; }
#chart-frame { position:relative; flex:1; min-height:420px; border:1px solid var(--line); border-radius:14px; overflow:hidden;
  background: radial-gradient(closest-side at 50% 50%, color-mix(in srgb, var(--surface-3) 70%, transparent), transparent), var(--surface-1);
  opacity:.0; transition:opacity .8s var(--ease); }
#chart-frame.ready { opacity:1; }
#chart { width:100%; height:100%; display:block; cursor:grab; touch-action:none; }
#tip { position:absolute; pointer-events:none; max-width:320px; padding:9px 12px; border-radius:10px; background:color-mix(in srgb, var(--surface-3) 94%, transparent); border:1px solid var(--line-strong); box-shadow:0 12px 30px rgba(0,0,0,.35); font-size:13px; display:flex; flex-direction:column; gap:3px; }
#tip .tip-meta { font:11.5px var(--font-mono); color:var(--ink-muted); }
#panel { position:absolute; top:12px; right:12px; bottom:12px; width:min(360px, calc(100% - 24px)); overflow:auto; padding:16px 18px; border-radius:12px;
  background:color-mix(in srgb, var(--surface-1) 94%, transparent); border:1px solid var(--line-strong); backdrop-filter:blur(8px); }
#panel .panel-empty { color:var(--ink-muted); font-size:13px; margin:0; }
#panel:has(.panel-empty) { bottom:auto; }
.panel-head { display:flex; gap:8px; align-items:center; } .panel-head .close { margin:0 0 0 auto; padding:2px 10px; font-size:18px; line-height:1; }
#panel h2 { font-size:17px; line-height:1.3; margin:12px 0 6px; letter-spacing:-.01em; }
#panel .path { display:block; font:11.5px var(--font-mono); color:var(--ink-muted); word-break:break-all; }
#panel .desc { color:var(--ink-soft); font-size:13.5px; }
.state { font:600 10.5px/1 var(--font-mono); letter-spacing:.1em; text-transform:uppercase; padding:5px 8px; border-radius:999px; }
.state-current { background:color-mix(in srgb, var(--node-current) 14%, transparent); color:var(--ink); }
.state-superseded { background:var(--rust-soft); color:var(--rust); } .state-expired { background:var(--surface-3); color:var(--ink-muted); } .state-pending { background:var(--signal-soft); color:var(--signal); }
.review-link { display:inline-block; margin:6px 0 4px; font:600 13px var(--font-display); }
.links h3 { font:500 10.5px var(--font-mono); letter-spacing:.14em; text-transform:uppercase; color:var(--ink-muted); margin:18px 0 6px; }
.links ul { list-style:none; margin:0; padding:0; } .links li { margin:2px 0; } .links .more { color:var(--ink-muted); font-size:12px; padding:4px 8px; }
.linkbtn { all:unset; display:block; width:100%; box-sizing:border-box; cursor:pointer; padding:5px 8px; border-radius:6px; font-size:13px; color:var(--ink-soft); }
.linkbtn:hover { background:var(--surface-3); color:var(--ink); }
.linkbtn::before { content:""; display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:8px; vertical-align:1px; background:var(--node-current); }
.state-dot-superseded::before { background:transparent; border:1.5px solid var(--node-superseded); } .state-dot-pending::before { background:var(--node-pending); } .state-dot-expired::before { background:var(--node-expired); }
.error-inline { color:var(--rust); }
@media (max-width:760px) { .shell { grid-template-columns:1fr; } .side { position:static; height:auto; flex-direction:row; align-items:center; flex-wrap:wrap; gap:12px; } .side .foot { display:none; } nav { flex-direction:row; } }
@media (prefers-reduced-motion: reduce) { *, *::before { animation:none !important; transition:none !important; } }
"""


def _shell(*, title: str, active: str, root: Path, pending: int, eyebrow: str, heading: str, lede: str, body: str, main_class: str = "", scripts: str = "") -> bytes:
    nav = [
        ("/", "queue", "↯", "Review queue", f"<span class='badge'>{pending}</span>" if pending else ""),
        ("/graph", "graph", "⁘", "Memory graph", ""),
    ]
    links = "".join(
        f"<a href='{href}' class='{'on' if key == active else ''}'><span class='glyph'>{glyph}</span>{label}{extra}</a>"
        for href, key, glyph, label, extra in nav
    )
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_STYLE}</style></head><body><div class='shell'>"
        "<aside class='side'><div class='brand'><b>RE<i>-</i>call</b><span class='rule'></span>"
        "<span class='eyebrow'>Memory dashboard</span></div>"
        f"<nav>{links}</nav>"
        f"<div class='foot'><span class='live'>local only</span><code>{_e(root)}</code></div></aside>"
        f"<main class='{main_class}'><header class='head'><div><span class='eyebrow'>{_e(eyebrow)}</span>"
        f"<h1>{_e(heading)}</h1><p>{_e(lede)}</p></div></header>{body}</main></div>{scripts}</body></html>"
    )
    return page.encode("utf-8")


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
        if method == "GET" and url.path == "/graph":
            return self._graph_page()
        if method == "GET" and url.path == "/api/graph.json":
            from recall.dashboard.graph import build_graph

            payload = json.dumps(build_graph(self.root), ensure_ascii=False).encode("utf-8")
            return Response(HTTPStatus.OK, payload, (("Content-Type", "application/json; charset=utf-8"),))
        if method == "GET" and url.path in _STATIC_FILES:
            name, kind = _STATIC_FILES[url.path]
            return Response(HTTPStatus.OK, (STATIC / name).read_bytes(), (("Content-Type", kind),))
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
                f"<a class='card claim' href='/review?{_e(urlencode({'claim': item.claim}))}'>"
                "<span class='arrow'>⟶</span>"
                f"<div><div class='tags'>{tags}</div><div class='names'>{_e(item.replacing)} <em>replaces</em> {_e(item.stale)}</div></div>"
                "<span class='go'>review →</span>"
                f"<div class='quote'><s>{_e(item.stale_quote[:180])}</s><br>{_e(item.current_quote[:180])}</div>"
                "</a>"
            )
        notes = "".join(f"<div>{_e(note)}</div>" for note in queue.notes) or "<div>agent reports only</div>"
        flash = f"<div class='card ok'>{_e(done)}</div>" if done else ""
        empty = (
            ""
            if rows
            else "<div class='card empty'><b>Nothing to review</b>When an agent reports a memo that another replaces, or the arbiter has judged a pair, it appears here.</div>"
        )
        body = f"{flash}{''.join(rows)}{empty}<h2 class='sec'>Sources</h2><div class='notes'>{notes}</div>"
        return self._html(
            _shell(
                title="RE-call · review queue", active="queue", root=self.root, pending=len(rows),
                eyebrow=f"01 · review · {len(rows)} pending", heading="Review queue",
                lede="Claims that one memo replaces another. Accepting writes the declaration into the newer memo; rejecting keeps the claim from coming back.",
                body=body,
            )
        )

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
            planned = f"<div class='diff'>{_e(plan.edit_file)} <ins>+ {_e(plan.key)}: {_e(plan.value)}</ins></div>"
        except review.ReviewRefused as exc:
            digest, planned = "", f"<div class='card error'>Cannot be applied: {_e(exc)}</div>"
        reviewer = cookies[REVIEWER_COOKIE].value if REVIEWER_COOKIE in cookies else ""
        details = "".join(f"<div class='muted'>{_e(d)}</div>" for d in item.details)
        tags = "".join(f"<span class='tag'>{_e(o)}</span>" for o in item.origins)
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
            else "<p class='muted'>This claim cannot be applied as it stands.</p>"
        )
        reject = f"<form method='post' action='/reject'>{common}<button class='reject' type='submit'>Reject</button></form>"
        body = (
            f"<div class='tags'>{tags}</div>{details}"
            "<div class='pair'>"
            + self._memo_card("older", item.stale, item.stale_quote)
            + self._memo_card("newer", item.replacing, item.current_quote)
            + "</div>"
            f"<h2 class='sec'>What accepting writes</h2>{planned}"
            "<p class='muted'>Search serves the change after the next index build.</p>"
            f"<div class='decide'><div class='card'>{accept}</div><div class='card'>{reject}</div></div>"
        )
        return self._html(
            _shell(
                title="RE-call · review a claim", active="queue", root=self.root, pending=0,
                eyebrow="01 · review · one claim", heading=f"{item.replacing} replaces {item.stale}?",
                lede="Read both memos. The highlighted lines are the evidence the report or the arbiter gave.",
                body="<div class='muted'><a href='/'>← back to the queue</a></div>" + body,
            )
        )

    def _memo_card(self, side: str, name: str, quote: str) -> str:
        text = self._memo(name)
        return (
            f"<div class='card memo-card {side}'><header><b>{_e(name)}</b><span class='tag'>{side}</span></header>"
            f"<pre>{excerpt(text, quote)}</pre>"
            f"<details class='full'><summary>Full memo</summary><pre>{highlight(text, quote)}</pre></details></div>"
        )

    def _graph_page(self) -> Response:
        toggles = "".join(
            f"<label class='chip'><input type='checkbox' id='show-{key}' {'checked' if on else ''}>{swatch}{label}</label>"
            for key, label, swatch, on in (
                ("link", "links", "<span class='sw link'></span>", True),
                ("supersedes", "supersedes", "<span class='sw supersedes'></span>", True),
                ("pending", "pending", "<span class='sw pending'></span>", True),
                ("hubs", "index pages", "", False),
            )
        )
        legend = "".join(
            f"<span><i class='dot {state}'></i>{label}</span>"
            for state, label in (("current", "current"), ("superseded", "superseded"), ("expired", "expired"), ("pending", "pending review"))
        )
        body = (
            "<div class='toolbar'><div class='searchbox'><input type='search' id='search' placeholder='Find a memo' aria-label='Find a memo'><kbd>/</kbd></div>"
            f"{toggles}</div>"
            f"<div class='legend' style='margin-top:12px'>{legend}</div>"
            "<div id='counts' aria-live='polite'><span class='count-label'>loading the corpus…</span></div>"
            "<div id='chart-frame'><canvas id='chart' role='img' aria-label='Memory graph: every memo and the links between them'></canvas>"
            "<div id='tip' hidden></div><aside id='panel'></aside></div>"
        )
        return self._html(
            _shell(
                title="RE-call · memory graph", active="graph", root=self.root, pending=0,
                eyebrow="02 · graph · every memo", heading="Memory graph",
                lede="Each point is a memo. Lines are the links memos make to each other; rust arrows are declared supersessions, amber dashes are claims waiting for review. Drag to move, scroll to zoom, click a point to read it.",
                body=body, main_class="graph-main", scripts="<script src='/static/graph.js' defer></script>",
            )
        )

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
        page = _shell(
            title="RE-call dashboard", active="", root=self.root, pending=0, eyebrow="dashboard",
            heading="Not available", lede="", body=f"<div class='card error'>{_e(message)}</div><p class='muted'><a href='/'>Go to the queue</a></p>",
        )
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
