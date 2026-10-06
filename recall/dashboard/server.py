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
* **The corpus database is read, never written.** The overview and retrieval pages open each
  connection with `default_transaction_read_only=on` (`recall.dashboard.db`), and the tenant a
  person picks must be a plain tenant id before it is queried or remembered in a cookie.
"""

from __future__ import annotations

import html
import json
import re
import secrets
import threading
import time
from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from recall.dashboard import db as dbq
from recall.dashboard import edit, review
from recall.dashboard.db import DashboardDB
from recall.multimodal import MULTIMODAL_TENANT
from recall.truth_extraction.types import STATUS_VOCABULARY

COOKIE = "recall_dashboard"
REVIEWER_COOKIE = "recall_reviewer"
TENANT_COOKIE = "recall_tenant"
DEFAULT_TENANT = "memory"
TENANT_CACHE_SECONDS = 60
# A tenant id as RE-call writes them; anything else in a query or cookie is ignored, never echoed.
TENANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
# The corpora this project serves, first in the switcher, by what they hold.
PRIMARY_TENANTS = (("memory", "memory"), ("re-call-code-gen", "code"), ("re-call-docs", "docs"), (MULTIMODAL_TENANT, "multimodal"))
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
  --cat-1:#efe9d8; --cat-2:#e3b23c; --cat-3:#8fb08a; --cat-4:#d0603f; --cat-5:#7fa7c4; --cat-6:#c49ac0; --cat-7:#b8a37a; --cat-8:#6f7a6c;
  --age-new:#f0be4a; --age-old:#4c544b;
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
    --cat-1:#2b302a; --cat-2:#a87a10; --cat-3:#4f7a4a; --cat-4:#b0462a; --cat-5:#3f6f91; --cat-6:#8a5a86; --cat-7:#7a6a43; --cat-8:#9a9c90;
    --age-new:#a87a10; --age-old:#c9c5b6;
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
.panel-actions { display:flex; flex-direction:column; gap:2px; margin:4px 0 2px; }
dl.values { display:grid; grid-template-columns:max-content 1fr; gap:6px 18px; margin:0; font-size:14px; }
dl.values dt { font:500 11px var(--font-mono); letter-spacing:.12em; text-transform:uppercase; color:var(--ink-muted); padding-top:2px; }
dl.values dd { margin:0; font-family:var(--font-mono); font-size:13px; }
label.check { display:flex; align-items:center; gap:8px; font:13px var(--font-display); letter-spacing:0; text-transform:none; color:var(--ink-soft); margin:6px 0; }
select, input[type=date] { font:inherit; padding:9px 11px; border:1px solid var(--line-strong); border-radius:8px; background:var(--surface-2); color:var(--ink); width:100%; }
.hist { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }
form.inline { display:flex; gap:6px; align-items:center; flex-wrap:wrap; } form.inline input { width:150px; } form.inline button { margin:0; }
pre.diffview { padding:14px; } pre.diffview .add { color:var(--sage); } pre.diffview .del { color:var(--rust); } pre.diffview .ctx { color:var(--ink-muted); }
ul.summary { margin:0; padding-left:18px; }
.lensbar { margin-top:12px; } label.select { display:inline-flex; align-items:center; gap:8px; margin:0; }
label.select select { width:auto; padding:6px 10px; font:12px var(--font-mono); letter-spacing:0; text-transform:none; }
.timebar { position:absolute; left:12px; right:12px; bottom:12px; display:flex; align-items:center; gap:12px; padding:8px 12px;
  border-radius:10px; background:color-mix(in srgb, var(--surface-1) 88%, transparent); border:1px solid var(--line); backdrop-filter:blur(6px); max-width:calc(100% - 396px); }
.timebar button { margin:0; padding:4px 10px; font-size:12px; } .timebar input[type=range] { flex:1; accent-color:var(--signal); }
#time-label { font:12px var(--font-mono); color:var(--ink-soft); min-width:92px; text-align:right; }
@media (max-width:760px) { .timebar { max-width:none; bottom:auto; top:12px; } }
.issues { margin:8px 0 0; padding:0; list-style:none; } .issues li { font-size:12.5px; color:var(--ink-soft); padding:6px 8px; border-left:2px solid var(--rust); background:var(--rust-soft); border-radius:4px; margin:4px 0; }
.issues code { font:11px var(--font-mono); color:var(--rust); display:block; }
.meta-line { font:11.5px var(--font-mono); color:var(--ink-muted); margin:4px 0 0; }
.tiles { display:grid; grid-template-columns:repeat(auto-fill, minmax(190px, 1fr)); gap:12px; }
.tile { display:flex; flex-direction:column; gap:6px; margin:0; } .tile b { font-size:30px; font-weight:620; letter-spacing:-.02em; }
details.card summary { cursor:pointer; } details.card summary b { font-family:var(--font-mono); font-size:13px; }
ul.plain { list-style:none; padding:0; margin:8px 0 0; columns:2 320px; } ul.plain li { padding:3px 0; font-size:13px; break-inside:avoid; }
.event { display:grid; grid-template-columns:96px 1fr auto; gap:14px; align-items:start; margin:8px 0; padding:12px 16px; }
.event time { font:12px var(--font-mono); color:var(--ink-muted); white-space:nowrap; }
.event .tag { justify-self:start; } .kind-accepted .tag, .kind-edit .tag { color:var(--sage); border-color:color-mix(in srgb, var(--sage) 50%, transparent); }
.kind-rejected .tag, .kind-undo .tag { color:var(--rust); border-color:color-mix(in srgb, var(--rust) 50%, transparent); } .kind-report .tag { color:var(--signal); border-color:color-mix(in srgb, var(--signal) 50%, transparent); }
@media (max-width:760px) { .event { grid-template-columns:1fr; } }
.tenantbox { padding:0 8px; } .tenantbox summary { cursor:pointer; display:flex; flex-direction:column; gap:4px; list-style:none; }
.tenantbox summary b { font:13px var(--font-mono); color:var(--signal); } .tenantlist { display:flex; flex-direction:column; margin-top:8px; max-height:40vh; overflow:auto; }
.tenantlist .absent { font:12px var(--font-mono); color:var(--ink-muted); padding:4px 6px; cursor:help; } .tenantlist .sub { margin:10px 6px 4px; }
.tenantlist a { font:12px var(--font-mono); color:var(--ink-soft); padding:4px 6px; border-radius:6px; } .tenantlist a.on, .tenantlist a:hover { background:var(--surface-3); color:var(--ink); text-decoration:none; }
.tenantgrid { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:12px; }
a.tenant { display:flex; flex-direction:column; gap:4px; color:inherit; margin:0; } a.tenant:hover { text-decoration:none; border-color:var(--line-strong); } a.tenant.on { box-shadow:inset 0 0 0 1px var(--signal); }
.mono { font-family:var(--font-mono); } .tile b.small { font-size:15px; overflow-wrap:anywhere; } .ok-text { color:var(--sage); } .warn-text { color:var(--rust); }
table.grid { width:100%; border-collapse:collapse; font-size:13px; } table.grid th { text-align:left; font:500 10.5px var(--font-mono); letter-spacing:.12em; text-transform:uppercase; color:var(--ink-muted); padding:6px 8px; border-bottom:1px solid var(--line); }
table.grid td { padding:7px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
a.search { display:grid; grid-template-columns:120px 1fr auto; gap:14px; align-items:start; color:inherit; margin:8px 0; } a.search:hover { text-decoration:none; border-color:var(--line-strong); }
a.search time { font:12px var(--font-mono); color:var(--ink-muted); } .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.chip { font:11px var(--font-mono); padding:2px 8px; border-radius:999px; border:1px solid var(--line-strong); color:var(--ink-soft); }
.chip.v-ok { color:var(--sage); border-color:color-mix(in srgb, var(--sage) 50%, transparent); } .chip.v-superseded { color:var(--rust); border-color:color-mix(in srgb, var(--rust) 50%, transparent); }
.chip.v-low_confidence { color:var(--signal); border-color:color-mix(in srgb, var(--signal) 50%, transparent); }
.tag.good { color:var(--sage); } .tag.warn { color:var(--rust); } .names.big { font-size:17px; margin-bottom:10px; }
form.searchform { display:flex; gap:8px; margin:16px 0 4px; } form.searchform input { flex:1; } form.searchform button { margin:0; }
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


def _shell(*, title: str, active: str, root: Path, pending: int, eyebrow: str, heading: str, lede: str, body: str, main_class: str = "", scripts: str = "", side: str = "") -> bytes:
    nav = [
        ("/overview", "overview", "◎", "Overview", ""),
        ("/", "queue", "↯", "Review queue", f"<span class='badge'>{pending}</span>" if pending else ""),
        ("/graph", "graph", "⁘", "Memory graph", ""),
        ("/activity", "activity", "≋", "Activity", ""),
        ("/health", "health", "✚", "Memory health", ""),
        ("/retrieval", "retrieval", "⇄", "Retrieval", ""),
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
        f"<nav>{links}</nav>{side}"
        f"<div class='foot'><span class='live'>local only</span><code>{_e(root)}</code></div></aside>"
        f"<main class='{main_class}'><header class='head'><div><span class='eyebrow'>{_e(eyebrow)}</span>"
        f"<h1>{_e(heading)}</h1><p>{_e(lede)}</p></div></header>{body}</main></div>{scripts}</body></html>"
    )
    return page.encode("utf-8")


class DashboardApp:
    """Routing, authentication and rendering, independent of the socket so tests can drive it."""

    def __init__(self, root: Path, *, port: int, token: str | None = None, db: DashboardDB | None = None) -> None:
        self.root = root.resolve()
        self.port = port
        self.token = token or secrets.token_urlsafe(32)
        self.db = db
        self._hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        # Per request, so concurrent requests on the threading server never share a tenant.
        self._request = threading.local()
        self._tenant_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._tenant_lock = threading.Lock()

    # ------------------------------------------------------------------ tenants (database)

    def _current_tenant(self) -> str:
        return getattr(self._request, "tenant", "") or DEFAULT_TENANT

    def _tenants(self) -> list[dict[str, Any]]:
        """Every tenant, cached for a minute: counting chunks for each one takes seconds."""
        if self.db is None:
            return []
        with self._tenant_lock:
            if self._tenant_cache and time.monotonic() - self._tenant_cache[0] < TENANT_CACHE_SECONDS:
                return self._tenant_cache[1]
            found = dbq.tenants(self.db)
            self._tenant_cache = (time.monotonic(), found)
            return found

    def _side(self) -> str:
        if self.db is None:
            return "<div class='tenantbox'><span class='eyebrow'>tenant</span><span class='muted'>no database connected</span></div>"
        current = self._current_tenant()
        path = getattr(self._request, "path", "/overview") or "/overview"
        path = "/retrieval" if path == "/retrieval/event" else path
        try:
            names = [t["tenant"] for t in self._tenants() if t["generation"]]
        except dbq.DatabaseUnavailable:
            return "<div class='tenantbox'><span class='eyebrow'>tenant</span><span class='error-inline'>database unreachable</span></div>"

        def link(name: str, label: str) -> str:
            caption = f"{_e(label)} <span class='muted'>{_e(name)}</span>" if label != name else _e(name)
            return f"<a href='{_e(path)}?{_e(urlencode({'tenant': name}))}' class='{'on' if name == current else ''}'>{caption}</a>"

        main = [
            link(name, label) if name in names
            else f"<span class='absent' title='{_e(name)} has no active generation in this database.'>{_e(label)} <span class='muted'>not built</span></span>"
            for name, label in PRIMARY_TENANTS
        ]
        rest = [link(name, name) for name in names if name not in dict(PRIMARY_TENANTS)]
        others = f"<span class='eyebrow sub'>benchmarks and other projects · {len(rest)}</span>{''.join(rest)}" if rest else ""
        label = dict(PRIMARY_TENANTS).get(current, current)
        return (
            f"<details class='tenantbox'><summary><span class='eyebrow'>tenant</span><b>{_e(label)}</b></summary>"
            f"<div class='tenantlist'>{''.join(main)}{others}</div></details>"
        )

    def _shell(self, **page: Any) -> bytes:
        return _shell(side=self._side(), **page)

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

        chosen = query.get("tenant", "")
        chosen = chosen if TENANT_NAME.fullmatch(chosen) else ""
        remembered = cookies[TENANT_COOKIE].value if TENANT_COOKIE in cookies else ""
        remembered = remembered if TENANT_NAME.fullmatch(remembered) else ""
        self._request.tenant = chosen or remembered or DEFAULT_TENANT
        self._request.path = url.path
        response = self._route(method, url.path, query, cookies, body)
        if chosen:
            cookie = SimpleCookie()
            cookie[TENANT_COOKIE] = chosen
            cookie[TENANT_COOKIE]["samesite"] = "Strict"
            cookie[TENANT_COOKIE]["path"] = "/"
            response = Response(response.status, response.body, response.headers + (("Set-Cookie", cookie.output(header="").strip()),))
        return response

    def _route(self, method: str, path: str, query: Mapping[str, str], cookies: SimpleCookie, body: bytes) -> Response:
        url = urlsplit(path)
        if method == "GET" and url.path == "/overview":
            return self._overview_page()
        if method == "GET" and url.path == "/retrieval":
            return self._retrieval_page(query.get("q", ""))
        if method == "GET" and url.path == "/retrieval/event":
            return self._retrieval_event_page(query.get("id", ""))
        if method == "GET" and url.path == "/":
            return self._queue_page(query.get("done"))
        if method == "GET" and url.path == "/review":
            return self._review_page(query.get("claim", ""), cookies)
        if method == "GET" and url.path == "/activity":
            return self._activity_page()
        if method == "GET" and url.path == "/health":
            return self._health_page()
        if method == "GET" and url.path == "/graph":
            return self._graph_page()
        if method == "GET" and url.path == "/api/graph.json":
            from recall.dashboard.graph import build_graph

            payload = json.dumps(build_graph(self.root), ensure_ascii=False).encode("utf-8")
            return Response(HTTPStatus.OK, payload, (("Content-Type", "application/json; charset=utf-8"),))
        if method == "GET" and url.path in _STATIC_FILES:
            name, kind = _STATIC_FILES[url.path]
            return Response(HTTPStatus.OK, (STATIC / name).read_bytes(), (("Content-Type", kind),))
        if method == "GET" and url.path == "/memo":
            return self._memo_page(query.get("path", ""), query.get("done"), cookies)
        if method == "POST" and url.path in ("/accept", "/reject", "/memo/preview", "/memo/apply", "/memo/undo"):
            fields = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True)
            form = {k: v[0] for k, v in fields.items()}
            if not secrets.compare_digest(form.get("csrf", ""), self.token):
                return self._error(HTTPStatus.FORBIDDEN, "This form did not come from this page.")
            if url.path.startswith("/memo/"):
                return self._memo_action(url.path[len("/memo/"):], form, fields.get("remove", []))
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
            self._shell(
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
            self._shell(
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

    # ------------------------------------------------------------------ memo editing (a person's)

    def _memo_names(self) -> list[str]:
        from recall.dashboard.graph import _memo_files

        return [p.relative_to(self.root).as_posix() for p in _memo_files(self.root)]

    def _memo_page(self, name: str, done: str | None, cookies: SimpleCookie) -> Response:
        try:
            values = edit.current_values(self.root, name)
        except edit.EditRefused as exc:
            return self._error(HTTPStatus.NOT_FOUND, str(exc))
        editor = cookies[REVIEWER_COOKIE].value if REVIEWER_COOKIE in cookies else ""
        token = f"<input type='hidden' name='csrf' value='{_e(self.token)}'>"
        path_field = f"<input type='hidden' name='path' value='{_e(values.name)}'>"
        options = "".join(f"<option value='{_e(n)}'>" for n in self._memo_names() if n != values.name)
        removals = "".join(
            f"<label class='check'><input type='checkbox' name='remove' value='{_e(t)}'> stop superseding <code>{_e(t)}</code></label>"
            for t in values.supersedes
        ) or "<p class='muted'>It supersedes nothing yet.</p>"
        status_options = "".join(
            f"<option value='{_e(s)}'{' selected' if values.status == s else ''}>{_e(s)}</option>" for s in STATUS_VOCABULARY
        )
        current = (
            "<dl class='values'>"
            f"<dt>supersedes</dt><dd>{_e(', '.join(values.supersedes) or 'nothing')}</dd>"
            f"<dt>valid from</dt><dd>{_e(values.valid_from or 'not set')}</dd>"
            f"<dt>valid until</dt><dd>{_e(values.valid_until or 'not set')}</dd>"
            f"<dt>status</dt><dd>{_e(values.status or 'not set')}</dd></dl>"
        )
        form = (
            f"<form method='post' action='/memo/preview' class='card'>{token}{path_field}"
            "<h2 class='sec'>Supersedes</h2>"
            f"{removals}<label>Add: this memo replaces…</label>"
            f"<input name='add' list='memos' placeholder='name of the older memo' autocomplete='off'><datalist id='memos'>{options}</datalist>"
            "<h2 class='sec'>Validity</h2><div class='pair'>"
            f"<div><label>Valid from</label><input type='date' name='valid_from' value='{_e(values.valid_from or '')}'>"
            "<label class='check'><input type='checkbox' name='clear_valid_from'> clear</label></div>"
            f"<div><label>Valid until</label><input type='date' name='valid_until' value='{_e(values.valid_until or '')}'>"
            "<label class='check'><input type='checkbox' name='clear_valid_until'> clear</label></div></div>"
            "<h2 class='sec'>Status</h2>"
            f"<select name='status'><option value=''>keep as it is</option><option value='__clear__'>clear</option>{status_options}</select>"
            "<button type='submit' class='accept'>Preview the change</button></form>"
        )
        rows = []
        for record in edit.history(self.root, values.name, limit=20):
            undo = (
                f"<form method='post' action='/memo/undo' class='inline'>{token}{path_field}"
                f"<input type='hidden' name='edit_id' value='{record.edit_id}'>"
                f"<input name='editor' value='{_e(editor)}' placeholder='your name' required>"
                "<input name='note' placeholder='why undo' required><button type='submit'>Undo</button></form>"
                if record.undone_at is None and not record.summary[0].startswith("undo of edit")
                else f"<span class='muted'>{'undone by ' + _e(record.undone_by) if record.undone_at else ''}</span>"
            )
            rows.append(
                f"<div class='card hist'><div><b>{_e('; '.join(record.summary))}</b>"
                f"<div class='muted'>#{record.edit_id} · {_e(record.editor)} · {_e(record.edited_at[:16].replace('T', ' '))} · {_e(record.note)}</div></div>{undo}</div>"
            )
        flash = f"<div class='card ok'>{_e(done)}</div>" if done else ""
        body = (
            f"<div class='muted'><a href='/graph#memo={_e(values.name)}'>← see it in the graph</a></div>{flash}"
            f"<div class='card'>{current}</div>{form}"
            f"<h2 class='sec'>Changes made here</h2>{''.join(rows) or '<p class=muted>No edits from the dashboard yet.</p>'}"
        )
        return self._html(
            self._shell(
                title="RE-call · edit a memo", active="graph", root=self.root, pending=0,
                eyebrow="02 · graph · edit values", heading=values.name,
                lede="Change what this memo declares: what it replaces, when it holds, its status. You see the exact edit before it is written; every edit is recorded and can be undone.",
                body=body,
            )
        )

    @staticmethod
    def _changes(form: Mapping[str, str], remove: list[str], values: edit.MemoValues) -> dict[str, object]:
        changes: dict[str, object] = {}
        if form.get("add", "").strip():
            changes["add_supersedes"] = (form["add"].strip(),)
        if remove:
            changes["remove_supersedes"] = tuple(remove)
        for key in ("valid_from", "valid_until"):
            if form.get(f"clear_{key}"):
                changes[key] = None
            elif form.get(key, "") and form.get(key) != getattr(values, key):
                changes[key] = form[key]
        status = form.get("status", "")
        if status == "__clear__":
            changes["status"] = None
        elif status:
            changes["status"] = status
        return changes

    def _memo_action(self, action: str, form: Mapping[str, str], remove: list[str]) -> Response:
        name = form.get("path", "")
        now = datetime.now(UTC)
        try:
            if action == "undo":
                edit.undo_edit(self.root, int(form.get("edit_id", "0") or 0), editor=form.get("editor", ""), note=form.get("note", ""), now=now)
                return self._redirect_memo(name, "Undone. The memo is back to how it was before that edit.", form.get("editor", ""))
            values = edit.current_values(self.root, name)
            changes = self._changes(form, remove, values)
            if action == "preview":
                plan = edit.plan_edit(self.root, name, **changes)  # type: ignore[arg-type]
                return self._preview_page(plan, form, remove)
            record = edit.apply_edit(
                self.root, name, editor=form.get("editor", ""), note=form.get("note", ""),
                shown_sha=form.get("shown_sha", ""), now=now, **changes,
            )
            return self._redirect_memo(name, "Saved: " + "; ".join(record.summary) + ". Search serves it after the next index build.", form.get("editor", ""))
        except edit.EditRefused as exc:
            return self._error(HTTPStatus.CONFLICT, f"Nothing was written: {exc}")
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "That edit id is not a number.")

    def _preview_page(self, plan: edit.EditPlan, form: Mapping[str, str], remove: list[str]) -> Response:
        carried = "".join(
            f"<input type='hidden' name='{_e(k)}' value='{_e(form[k])}'>"
            for k in ("path", "add", "valid_from", "valid_until", "clear_valid_from", "clear_valid_until", "status")
            if k in form
        ) + "".join(f"<input type='hidden' name='remove' value='{_e(r)}'>" for r in remove)
        diff = "".join(
            f"<span class='{'add' if line.startswith('+') and not line.startswith('+++') else 'del' if line.startswith('-') and not line.startswith('---') else 'ctx'}'>{_e(line)}</span>\n"
            for line in plan.diff
        )
        body = (
            f"<div class='muted'><a href='/memo?{_e(urlencode({'path': plan.name}))}'>← back without saving</a></div>"
            "<div class='card'><ul class='summary'>" + "".join(f"<li>{_e(s)}</li>" for s in plan.summary) + "</ul></div>"
            f"<pre class='diffview card'>{diff}</pre>"
            f"<form method='post' action='/memo/apply' class='card'><input type='hidden' name='csrf' value='{_e(self.token)}'>"
            f"<input type='hidden' name='shown_sha' value='{_e(plan.before_sha)}'>{carried}"
            "<label>Your name</label><input name='editor' required>"
            "<label>Note (why)</label><textarea name='note' rows='2' required></textarea>"
            "<button class='accept' type='submit'>Write this change</button></form>"
        )
        return self._html(
            self._shell(
                title="RE-call · confirm an edit", active="graph", root=self.root, pending=0,
                eyebrow="02 · graph · confirm", heading=f"Change {plan.name}?",
                lede="This is exactly what will be written. Nothing else in the file moves.", body=body,
            )
        )

    def _redirect_memo(self, name: str, done: str, editor: str) -> Response:
        cookie = SimpleCookie()
        cookie[REVIEWER_COOKIE] = editor
        cookie[REVIEWER_COOKIE]["samesite"] = "Strict"
        cookie[REVIEWER_COOKIE]["path"] = "/"
        return Response(
            HTTPStatus.SEE_OTHER,
            b"",
            (("Location", "/memo?" + urlencode({"path": name, "done": done})), ("Set-Cookie", cookie.output(header="").strip())),
        )

    # ------------------------------------------------------------------ activity and health

    def _memo_link(self, name: str | None) -> str:
        if not name:
            return ""
        return f"<a href='/memo?{_e(urlencode({'path': name}))}'>{_e(name)}</a>"

    def _activity_page(self) -> Response:
        from recall.dashboard.activity import Event, instant, recent_activity

        events = recent_activity(self.root)
        corpus_note = "Index builds, calibrations and searches come from the corpus database and appear once it is connected."
        if self.db is not None:
            tenant = self._current_tenant()
            try:
                for item in dbq.lifecycle_events(self.db, tenant, limit=60):
                    at = item["created_at"].astimezone(UTC).isoformat() if hasattr(item["created_at"], "astimezone") else str(item["created_at"])
                    detail = f"generation {item['generation']}" if item["generation"] else (item["source"] or "")
                    events.append(Event(at, "corpus", f"{tenant}: {str(item['event']).replace('_', ' ')}", detail, item["actor"] or ""))
                corpus_note = f"Corpus events are the {_e(tenant)} tenant's, read from the database; searches are under Retrieval."
            except dbq.DatabaseUnavailable:
                corpus_note = "The corpus database did not answer, so its events are missing from this list."
            events.sort(key=lambda event: instant(event.at), reverse=True)
        rows = []
        for event in events:
            when = event.at[:16].replace("T", " ")
            who = f" · {_e(event.who)}" if event.who else ""
            memo = f"<div class='muted'>{self._memo_link(event.memo)}</div>" if event.memo and event.kind != "changed" else ""
            title = self._memo_link(event.memo) if event.kind == "changed" else _e(event.title)
            rows.append(
                f"<div class='card event kind-{_e(event.kind)}'><span class='tag'>{_e(event.kind)}</span>"
                f"<div><div class='names'>{title}</div>{memo}<div class='muted'>{_e(event.detail)}{who}</div></div>"
                f"<time>{_e(when)}</time></div>"
            )
        body = (
            "".join(rows)
            or "<div class='card empty'><b>Nothing yet</b>Edits, review decisions, agent reports and changed memos appear here.</div>"
        ) + f"<p class='muted'>Memo events come from the files and the dashboard's own records. {corpus_note}</p>"
        return self._html(
            self._shell(
                title="RE-call · activity", active="activity", root=self.root, pending=0,
                eyebrow="03 · activity · newest first", heading="Recent activity",
                lede="What changed in the memory and who changed it: edits made here, review decisions, agent reports, and memos changed on disk.",
                body=body,
            )
        )

    def _health_page(self) -> Response:
        from recall.dashboard.graph import build_graph

        graph = build_graph(self.root, include_queue=True)
        nodes = graph["nodes"]
        counts = graph["counts"]
        tiles = "".join(
            f"<div class='card tile'><span class='eyebrow'>{_e(label)}</span><b>{_e(value)}</b><span class='muted'>{_e(note)}</span></div>"
            for label, value, note in (
                ("memos", counts["memos"], f"{counts['hubs']} of them index pages"),
                ("superseded", counts["superseded"], "declared replaced by a newer memo"),
                ("pending review", counts["pending"], "claims waiting in the queue"),
                ("expired", counts["expired"], "past their valid_until"),
                ("lint issues", counts["issues"], "see below"),
                ("linked only from index pages", counts["isolated"], "nothing else points at them"),
            )
        )
        by_code: dict[str, list[tuple[str, str, str]]] = {}
        for node in nodes:
            for issue in node["issues"]:
                by_code.setdefault(issue["code"], []).append((node["id"], issue["level"], issue["message"]))
        sections: list[str] = []
        for code, found in sorted(by_code.items(), key=lambda item: -len(item[1])):
            items = "".join(
                f"<li>{self._memo_link(name)} <a class='muted' href='/graph#memo={_e(name)}'>graph</a></li>" for name, _level, _msg in found[:60]
            )
            more = f"<li class='muted'>and {len(found) - 60} more</li>" if len(found) > 60 else ""
            sections.append(
                f"<details class='card' {'open' if len(sections) == 0 else ''}><summary><span class='tag'>{_e(found[0][1])}</span> "
                f"<b>{_e(code)}</b> · {len(found)}</summary><p class='muted'>{_e(found[0][2])}</p><ul class='plain'>{items}{more}</ul></details>"
            )
        isolated = [n["id"] for n in nodes if n["isolated"]]
        isolated_list = "".join(f"<li>{self._memo_link(name)}</li>" for name in isolated[:80])
        body = (
            f"<div class='tiles'>{tiles}</div>"
            f"<h2 class='sec'>Lint issues · {counts['issues']}</h2>"
            + ("".join(sections) or "<div class='card muted'>No lint issue found.</div>")
            + f"<h2 class='sec'>Linked only from index pages · {len(isolated)}</h2>"
            + f"<details class='card'><summary>Show the memos nothing else links to</summary><ul class='plain'>{isolated_list}</ul></details>"
            + "<p class='muted'>Lint is <code>recall lint</code> on the memo files. A `closure-marker-unlinked` memo says in prose that it "
            "replaces something without declaring it; open it and add the supersession so search can act on it.</p>"
        )
        return self._html(
            self._shell(
                title="RE-call · memory health", active="health", root=self.root, pending=counts["pending"],
                eyebrow="04 · health · the memo files", heading="Memory health",
                lede="Problems the memory files carry today: broken or ambiguous supersessions, cycles, prose that declares nothing, and memos nothing links to.",
                body=body,
            )
        )

    # ------------------------------------------------------------------ database pages (read-only)

    def _not_connected(self, title: str, active: str, heading: str) -> Response:
        reason = "No database is configured for this dashboard."
        if self.db is not None:
            reason = "The corpus database did not answer."
        body = (
            f"<div class='card error'>{_e(reason)}</div>"
            "<div class='card'><p>Start the dashboard with a read-only connection, for example through an SSH tunnel:</p>"
            "<pre>recall dashboard --root &lt;memo folder&gt; --tunnel &lt;host&gt; --db-dsn-file &lt;file holding the DSN&gt;</pre>"
            "<p class='muted'>The DSN should name a SELECT-only role. The memo pages keep working without it.</p></div>"
        )
        return self._html(self._shell(title=title, active=active, root=self.root, pending=0, eyebrow="database",
                                      heading=heading, lede="", body=body), HTTPStatus.SERVICE_UNAVAILABLE)

    @staticmethod
    def _when(value: Any) -> str:
        return value.strftime("%Y-%m-%d %H:%M") if hasattr(value, "strftime") else (str(value) if value else "")

    def _overview_page(self) -> Response:
        if self.db is None:
            return self._not_connected("RE-call · overview", "overview", "Overview")
        try:
            status = dbq.availability(self.db)
            every = self._tenants()
            tenant = self._current_tenant()
            chosen = next((t for t in every if t["tenant"] == tenant), None)
            history = dbq.generations(self.db, tenant) if chosen else []
            events = dbq.lifecycle_events(self.db, tenant, limit=12) if chosen else []
            recorded = len(dbq.searches(self.db, tenant, limit=500)) if chosen else 0
        except dbq.DatabaseUnavailable:
            return self._not_connected("RE-call · overview", "overview", "Overview")
        migrations = ", ".join(f"{table} {version}" for table, version in status["migrations"] if not table.startswith(("bench_", "comparison_", "locomo_", "recall_aml_", "structural_", "selective_", "voyage4_", "category4_", "chunks_bge")))
        avail = (
            "<div class='tiles'>"
            f"<div class='card tile'><span class='eyebrow'>database</span><b class='ok-text'>answering</b><span class='muted'>PostgreSQL {_e(status['server_version'].split()[0])}</span></div>"
            f"<div class='card tile'><span class='eyebrow'>connected as</span><b class='mono small'>{_e(status['role'])}</b><span class='muted'>{'read-only session' if status['read_only'] else 'NOT read-only'}</span></div>"
            f"<div class='card tile'><span class='eyebrow'>tenants</span><b>{sum(1 for t in every if t['generation'])}</b><span class='muted'>{len(every)} known, the rest without an active generation</span></div>"
            f"<div class='card tile'><span class='eyebrow'>searches recorded</span><b>{recorded}</b><span class='muted'>for {_e(tenant)}; needs RECALL_DECISION_LEDGER=1</span></div>"
            "</div>"
            f"<p class='muted mono'>schema: {_e(migrations or 'unknown')}</p>"
        )
        cards = []
        order = [name for name, _ in PRIMARY_TENANTS]
        for t in sorted(every, key=lambda x: (not x["generation"], order.index(x["tenant"]) if x["tenant"] in order else len(order), -x["chunks"])):
            if not t["generation"]:
                continue
            emb = t["embedder"]
            cal = t["calibration"] or {}
            cert = "certified" if cal.get("certified") else ("not certified" if cal else "no calibration")
            cards.append(
                f"<a class='card tenant {'on' if t['tenant'] == tenant else ''}' href='/overview?{_e(urlencode({'tenant': t['tenant']}))}'>"
                f"<b>{_e(t['tenant'])}</b><span class='mono muted'>{_e(emb.get('model') or '?')} · {_e(emb.get('dimension') or '?')}d</span>"
                f"<span>{t['chunks']:,} chunks · {t['sources']:,} sources</span>"
                f"<span class='{'ok-text' if cal.get('certified') else 'warn-text'}'>{_e(cert)}</span></a>"
            )
        detail = ""
        if chosen:
            emb = chosen["embedder"]
            cal = chosen["calibration"] or {}
            facts = (
                "<dl class='values'>"
                f"<dt>active generation</dt><dd>{_e(chosen['generation'])}</dd>"
                f"<dt>activated</dt><dd>{_e(self._when(chosen['activated_at']))}</dd>"
                f"<dt>embedder</dt><dd>{_e(emb.get('provider'))} · {_e(emb.get('model'))} · {_e(emb.get('dimension'))}d · profile {_e(emb.get('profile_id'))}</dd>"
                f"<dt>calibration</dt><dd>{_e('certified' if cal.get('certified') else 'not certified' if cal else 'none')}"
                f"{' · threshold ' + format(cal['threshold'], '.3f') if cal.get('threshold') is not None else ''}"
                f"{' · separability ' + format(cal['separability'], '.3f') if cal.get('separability') is not None else ''}</dd>"
                f"<dt>size</dt><dd>{chosen['chunks']:,} chunks from {chosen['sources']:,} sources</dd></dl>"
            )
            gens = "".join(
                f"<tr><td class='mono'>{_e(g['generation'][:16])}</td><td><span class='tag'>{_e(g['state'])}</span></td>"
                f"<td>{_e(self._when(g['created_at']))}</td><td>{_e(self._when(g['activated_at']))}</td><td class='muted'>{_e(g['failure_reason'] or '')}</td></tr>"
                for g in history
            )
            evs = "".join(
                f"<tr><td>{_e(self._when(e['created_at']))}</td><td><span class='tag'>{_e(e['event'])}</span></td><td class='muted'>{_e(e['actor'])}</td></tr>"
                for e in events
            )
            detail = (
                f"<h2 class='sec'>{_e(tenant)}</h2><div class='card'>{facts}</div>"
                f"<h2 class='sec'>Generations</h2><div class='card'><table class='grid'><tr><th>generation</th><th>state</th><th>created</th><th>activated</th><th>failure</th></tr>{gens}</table></div>"
                f"<h2 class='sec'>Latest lifecycle events</h2><div class='card'><table class='grid'>{evs}</table></div>"
            )
        body = f"{avail}<h2 class='sec'>Tenants</h2><div class='tenantgrid'>{''.join(cards)}</div>{detail}"
        return self._html(self._shell(
            title="RE-call · overview", active="overview", root=self.root, pending=0,
            eyebrow="00 · overview · read-only", heading="Overview",
            lede="Whether the corpus database answers, and what each tenant holds: its embedder, its calibration, its generations. Read-only; nothing here can change the corpus.",
            body=body,
        ))

    def _memo_for_source(self, source: str | None) -> str | None:
        """The local memo a hit came from, matched by file name, when this dashboard has it."""
        if not source:
            return None
        name = source.replace("\\", "/").rsplit("/", 1)[-1]
        candidate = (self.root / name)
        return name if candidate.is_file() else None

    def _verdict_chips(self, counts: Mapping[str, Any]) -> str:
        return "".join(f"<span class='chip v-{_e(k)}'>{_e(k)} · {_e(v)}</span>" for k, v in sorted(counts.items()))

    def _retrieval_page(self, contains: str) -> Response:
        if self.db is None:
            return self._not_connected("RE-call · retrieval", "retrieval", "Retrieval")
        tenant = self._current_tenant()
        try:
            rows = dbq.searches(self.db, tenant, limit=200, contains=contains)
        except dbq.DatabaseUnavailable:
            return self._not_connected("RE-call · retrieval", "retrieval", "Retrieval")
        answered = sum(1 for r in rows if not r["abstained"])
        hits = [len(r["hits"]) for r in rows]
        superseded = sum(1 for r in rows if any(h["verdict"] == "superseded" for h in r["hits"]))
        tiles = "".join(
            f"<div class='card tile'><span class='eyebrow'>{_e(label)}</span><b>{_e(value)}</b><span class='muted'>{_e(note)}</span></div>"
            for label, value, note in (
                ("searches", len(rows), "most recent 200"),
                ("answered", answered, f"{len(rows) - answered} abstained"),
                ("hits per search", f"{(sum(hits) / len(hits)):.1f}" if hits else "0", "returned to the agent"),
                ("caught a superseded memo", superseded, "a stale version was flagged, not served as current"),
            )
        )
        table = "".join(
            f"<a class='card search' href='/retrieval/event?{_e(urlencode({'id': r['event_id']}))}'>"
            f"<time>{_e(self._when(r['created_at']))}</time>"
            f"<div><div class='names'>{_e(r['query'][:220])}</div><div class='chips'>{self._verdict_chips(r['verdict_counts'])}</div></div>"
            f"<span class='tag {'warn' if r['abstained'] else 'good'}'>{_e('abstained' if r['abstained'] else 'answered')}</span></a>"
            for r in rows
        )
        empty = (
            "<div class='card empty'><b>No searches recorded for this tenant</b>Recording needs RECALL_DECISION_LEDGER=1 on the "
            "server; the RE-call MCP servers launched by session-mcp have it from 2026-10-06. Searches made before that were not kept.</div>"
        )
        search = (
            f"<form method='get' action='/retrieval' class='searchform'><input type='hidden' name='tenant' value='{_e(tenant)}'>"
            f"<input type='search' name='q' value='{_e(contains)}' placeholder='Find a task or query'><button type='submit'>Filter</button></form>"
        )
        body = f"<div class='tiles'>{tiles}</div>{search}{table or empty}"
        return self._html(self._shell(
            title="RE-call · retrieval", active="retrieval", root=self.root, pending=0,
            eyebrow=f"05 · retrieval · {tenant}", heading="Retrieval",
            lede="Every search an agent made: what it asked, what came back, and the verdict on each memory. A result is what was offered; whether the agent used it is not recorded.",
            body=body,
        ))

    def _retrieval_event_page(self, event_id: str) -> Response:
        if self.db is None:
            return self._not_connected("RE-call · a search", "retrieval", "A search")
        tenant = self._current_tenant()
        try:
            found = dbq.search(self.db, tenant, event_id)
        except dbq.DatabaseUnavailable:
            return self._not_connected("RE-call · a search", "retrieval", "A search")
        if found is None:
            return self._error(HTTPStatus.NOT_FOUND, "No such search for this tenant.")
        rows = []
        for i, hit in enumerate(found["hits"], 1):
            memo = self._memo_for_source(hit["source"])
            source = self._memo_link(memo) if memo else _e(hit["source"])
            successor = f"<div class='muted'>superseded by {_e(hit['superseded_by'])}</div>" if hit["superseded_by"] else ""
            confidence = f"{hit['confidence']:.2f}" if isinstance(hit["confidence"], int | float) else ""
            rows.append(
                f"<tr><td>{i}</td><td>{source}{successor}</td><td><span class='chip v-{_e(hit['verdict'])}'>{_e(hit['verdict'])}</span></td>"
                f"<td class='mono'>{_e(confidence)}</td></tr>"
            )
        facts = (
            "<dl class='values'>"
            f"<dt>when</dt><dd>{_e(self._when(found['created_at']))}</dd>"
            f"<dt>by</dt><dd>{_e(found['actor'])}</dd>"
            f"<dt>outcome</dt><dd>{_e('abstained' if found['abstained'] else 'answered')}{' · ' + _e(found['reason']) if found['reason'] else ''}</dd>"
            f"<dt>trust</dt><dd>{_e(found['trust_state'] or '')}{' · ' + _e(found['failure_code']) if found['failure_code'] else ''}</dd>"
            f"<dt>generation</dt><dd class='mono'>{_e(found['generation'] or '')}</dd></dl>"
        )
        body = (
            f"<div class='muted'><a href='/retrieval'>← all searches</a></div>"
            f"<div class='card'><div class='names big'>{_e(found['query'])}</div>{facts}</div>"
            f"<h2 class='sec'>What came back · {len(found['hits'])}</h2>"
            f"<div class='card'><table class='grid'><tr><th>#</th><th>source</th><th>verdict</th><th>confidence</th></tr>{''.join(rows)}</table></div>"
        )
        return self._html(self._shell(
            title="RE-call · a search", active="retrieval", root=self.root, pending=0,
            eyebrow=f"05 · retrieval · {tenant}", heading="One search",
            lede="The memories offered for this task, in order, each with its verdict. Memos this dashboard holds link to their editor.",
            body=body,
        ))

    def _graph_page(self) -> Response:
        toggles = "".join(
            f"<label class='chip'><input type='checkbox' id='show-{key}' {'checked' if on else ''}>{swatch}{label}</label>"
            for key, label, swatch, on in (
                ("link", "links", "<span class='sw link'></span>", True),
                ("supersedes", "supersedes", "<span class='sw supersedes'></span>", True),
                ("pending", "pending", "<span class='sw pending'></span>", True),
                ("hubs", "index pages", "", False),
                ("motion", "motion", "", True),
            )
        )
        legend = "".join(
            f"<span><i class='dot {state}'></i>{label}</span>"
            for state, label in (("current", "current"), ("superseded", "superseded"), ("expired", "expired"), ("pending", "pending review"))
        )
        lenses = "".join(
            f"<option value='{key}'>{label}</option>"
            for key, label in (("state", "state"), ("type", "type"), ("folder", "folder"), ("age", "age"), ("health", "health"))
        )
        body = (
            "<div class='toolbar'><div class='searchbox'><input type='search' id='search' placeholder='Find a memo' aria-label='Find a memo'><kbd>/</kbd></div>"
            f"{toggles}</div>"
            "<div class='toolbar lensbar'>"
            f"<label class='select'>colour by <select id='lens'>{lenses}</select></label>"
            "<label class='select'>focus <select id='focus'><option value='0'>whole memory</option>"
            "<option value='1'>1 hop around the selection</option><option value='2'>2 hops</option><option value='3'>3 hops</option></select></label>"
            f"<div class='legend' id='legend'>{legend}</div></div>"
            "<div id='counts' aria-live='polite'><span class='count-label'>loading the corpus…</span></div>"
            "<div id='chart-frame'><canvas id='chart' role='img' aria-label='Memory graph: every memo and the links between them'></canvas>"
            "<div id='tip' hidden></div><aside id='panel'></aside>"
            "<div class='timebar'><button type='button' id='play' aria-label='Play the memory forward'>▶</button>"
            "<input type='range' id='time' min='0' max='0' value='0' aria-label='Show the memory as of a date'>"
            "<span id='time-label'>today</span></div></div>"
        )
        return self._html(
            self._shell(
                title="RE-call · memory graph", active="graph", root=self.root, pending=0,
                eyebrow="02 · graph · every memo", heading="Memory graph",
                lede="Each star is a memo. Lines are the links memos make to each other; rust arrows are declared supersessions, with light running from the newer memo to the one it replaces; amber dashes are claims waiting for review. Drag to orbit, scroll to zoom, click a star to read it, double-click to reset.",
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
        page = self._shell(
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
