"""The memo reader (/read): a memo's text as a page, safely, with what replaced it.

Invariants and the failure each one catches:
- R1 memo text cannot become markup: raw HTML is escaped, a `javascript:` link stays text, and a
  web link opens with `rel="noopener noreferrer"`.
- R2 a `[[wiki link]]` to a memo of the store opens that memo in the reader; one that names no
  memo is shown as a broken link, never as a link to nowhere.
- R3 a memo another memo declares it supersedes says so at the top and names the newer memo.
- R4 only what a person wrote is shown: the frontmatter and the machine-written derived block
  stay out of the article.
- R5 code is shown as written: formatting markers inside code are not interpreted.
- R6 the reader needs the session, and a name outside the store is not found.

Red proof, 2026-10-07, each mutation alone, failing in the named assertion (JUnit XML), then
restored and green:
- K1 (R1) `recall/dashboard/markdown.py` `_anchor_for` linking any target, not only http(s):
  a `javascript:` href reached the page.
- K2 (R2) an unresolved wiki link rendered as an anchor: no broken-link span.
- K3 (R3) `recall/dashboard/server.py` `_read_page` without the replaced banner: the notice was
  missing.
- K4 (R4) `_read_page` rendering the raw file instead of `human_body`: the frontmatter showed.
- K5 (R5) a code span passed through `inline` instead of being escaped as is: `<strong>` inside
  the code.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pytest

from recall.dashboard.markdown import render
from recall.dashboard.server import DashboardApp

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}

NEW = """---
supersedes: old-port.md
---
# Port

The port is **9090**, see [[old-port]] and [[no-such-memo]].

<script>alert(1)</script> [click](javascript:alert(2)) [docs](https://example.org/a?b=1&c='x')

| key | value |
|---|---|
| port | `9090` |

- one
  - nested
- two

```
**not bold** <b>
```

Inline `**still code**` here.
"""

OLD = "# Old port\n\nThe port was 8080.\n"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "new-port.md").write_text(NEW, encoding="utf-8")
    (tmp_path / "old-port.md").write_text(OLD, encoding="utf-8")
    return tmp_path


def _read(store: Path, name: str, headers: dict[str, str] = SIGNED) -> tuple[int, str]:
    response = DashboardApp(store, port=8765, token="tok").handle("GET", "/read?" + urlencode({"path": name}), headers)
    return response.status, response.body.decode()


def _article(page: str) -> str:
    return page.split("<article class='card prose'>", 1)[1].split("</article>", 1)[0]


def test_memo_text_cannot_become_markup(store: Path) -> None:
    status, page = _read(store, "new-port.md")
    assert status == 200
    article = _article(page)
    assert "<script>" not in article and "&lt;script&gt;alert(1)&lt;/script&gt;" in article
    assert "href='javascript" not in article, "a javascript: link became an anchor"
    assert "href='https://example.org/a?b=1&amp;c=&#x27;x&#x27;' rel='noopener noreferrer'" in article


def test_wiki_links_open_memos_and_unknown_names_are_marked(store: Path) -> None:
    article = _article(_read(store, "new-port.md")[1])
    assert "href='/read?path=old-port.md'" in article
    assert "<span class='broken-link'" in article and "no-such-memo</span>" in article, "an unknown wiki link was not marked broken"


def test_a_replaced_memo_says_so_and_names_the_newer_one(store: Path) -> None:
    _, old = _read(store, "old-port.md")
    assert "This memo has been replaced." in old, "the replaced banner is missing"
    assert "href='/read?path=new-port.md'" in old.split("This memo has been replaced.", 1)[1][:400]
    assert "This memo has been replaced." not in _read(store, "new-port.md")[1]


def test_only_what_a_person_wrote_is_shown(store: Path) -> None:
    (store / "derived.md").write_text(
        "---\nvalid_from: 2026-10-01\n---\n# D\n\nHuman text.\n\n<!-- recall:derived -->\n```recall-derived\nstatus: active\n```\n<!-- /recall:derived -->\n",
        encoding="utf-8",
    )
    article = _article(_read(store, "derived.md")[1])
    assert "Human text." in article
    assert "valid_from" not in article and "supersedes" not in _article(_read(store, "new-port.md")[1]), "frontmatter reached the article"
    assert "recall-derived" not in article


def test_code_is_shown_as_written() -> None:
    html = render("Inline `**x** <b>` and\n\n```\n**y**\n```\n", lambda target: None)
    assert "<code>**x** &lt;b&gt;</code>" in html, "formatting inside a code span was interpreted"
    assert "<pre><code>**y**</code></pre>" in html


def test_structures_render(store: Path) -> None:
    article = _article(_read(store, "new-port.md")[1])
    assert "<table>" in article and "<th>key</th>" in article and "<td><code>9090</code></td>" in article
    assert "<ul><li>one<ul><li>nested</li></ul></li><li>two</li></ul>" in article
    assert "<strong>9090</strong>" in article
    assert "<h2>Port</h2>" not in article, "the title was repeated inside the article"


def test_the_reader_needs_the_session_and_stays_in_the_store(store: Path) -> None:
    assert _read(store, "new-port.md", HOST)[0] == 403
    assert _read(store, "../outside.md")[0] == 404
    assert _read(store, "nope.md")[0] == 404
