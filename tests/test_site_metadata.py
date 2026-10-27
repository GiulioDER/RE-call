"""The homepage states its title and description once, in several places, and they agree.

Invariant: `site/index.html` repeats its title in `<title>` and `og:title`, and its description in
the meta description, `og:description` and every JSON-LD node's `description`; each copy equals
the first. Failure mode caught: one copy is edited (for a release, a rewording) and the others are
not, so search results, link previews and structured data describe the project differently. CCA
CODE-001 of the audit of #952, which added the copies with nothing tying them together.

Red proof, 2026-10-27 on the test host, one copy in `site/index.html` mutated per case, each
restored after (hash-checked), then both tests green:

- `og:title` reworded: `test_the_homepage_title_has_one_value` fails with
  `assert 'RE-call: memory for agents' == 'RE-call: Per...for AI agents'`.
- `og:description` reworded: `test_the_homepage_description_has_one_value` fails with
  `assert {'og:descript...th pgvector.'} == {}`.
- the first JSON-LD node's description reworded: the same test fails with
  `assert {'json-ld Web...th pgvector.'} == {}`.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "site" / "index.html"


class _Head(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self._in_title = False
        self._in_json_ld = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = values.get("name") or values.get("property")
            if key is not None and values.get("content") is not None:
                self.meta[key] = values["content"] or ""
        elif tag == "script" and values.get("type") == "application/ld+json":
            self._in_json_ld = True
            self.json_ld.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_json_ld = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._in_json_ld:
            self.json_ld[-1] += data


def _head() -> _Head:
    parser = _Head()
    parser.feed(INDEX.read_text(encoding="utf-8"))
    return parser


def test_the_homepage_title_has_one_value() -> None:
    head = _head()
    assert head.title.strip()
    assert head.meta["og:title"] == head.title.strip()


def test_the_homepage_description_has_one_value() -> None:
    head = _head()
    description = head.meta["description"]
    assert description
    copies = {"og:description": head.meta["og:description"]}
    for block in head.json_ld:
        for node in json.loads(block).get("@graph", []):
            if "description" in node:
                copies[f"json-ld {node['@type']}"] = node["description"]
    assert len(copies) >= 3
    assert {name: text for name, text in copies.items() if text != description} == {}
