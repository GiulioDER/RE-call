"""Reader regressions for PR888's CCA findings.

Red proof, 2026-10-27: the finding assertions failed against production baseline
1ecb50db5011. The table-lookahead and graph-opener assertions additionally failed
against the first repair, before those SEC-002 gaps were closed. Green command:
``python -m pytest tests/test_dashboard_reader_hardening.py`` on a bounded Linux runner.

The three compatibility controls were independently shown red by mutations:
``memo_files`` excluding all symlinks, ``inline`` percent-decoding wiki names,
and ``inline`` dropping bold formatting. Each failed its intended assertion;
sources were restored byte for byte before the same tests passed. The changed
banner assertion in test_dashboard_reader.py also failed on the baseline server.
The wiki control inspects the article so a sidebar cannot make a broken article
look correct. These controls preserve formatting, internal symlinks and names.
"""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest

from recall.dashboard.markdown import render
from recall.dashboard.server import DashboardApp

SIGNED = {"Host": "127.0.0.1:8765", "Cookie": "recall_dashboard=tok"}


def _read(root: Path, name: str) -> tuple[int, str]:
    response = DashboardApp(root, port=8765, token="tok").handle(
        "GET", "/read?" + urlencode({"path": name}), SIGNED,
    )
    return response.status, response.body.decode()


def test_reader_rejects_outside_symlink_before_graph_read(tmp_path: Path) -> None:
    """SEC-001: an admitted name cannot render an outside symlink target."""
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n\nOUTSIDE_BODY_MARKER", encoding="utf-8")
    (root / "alias.md").symlink_to(outside)
    status, page = _read(root, "alias.md")
    assert status == 404, "SEC-001: outside symlink was admitted"
    assert "OUTSIDE_BODY_MARKER" not in page


def test_outside_symlink_cannot_supply_reader_relationships(tmp_path: Path) -> None:
    """SEC-001: graph metadata must not import outside declarations or titles."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "safe.md").write_text("# Safe\n\nSafe text", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("---\nsupersedes: safe.md\n---\n# OUTSIDE_TITLE_MARKER", encoding="utf-8")
    (root / "alias.md").symlink_to(outside)
    status, page = _read(root, "safe.md")
    assert status == 200
    assert "href='/read?path=alias.md'" not in page, "SEC-001: outside graph declaration reached reader"


def test_in_corpus_symlink_still_reads(tmp_path: Path) -> None:
    """Nonbehavioral control: an internal symlink remains a readable memo."""
    (tmp_path / "memo.md").write_text("# Memo\n\nINSIDE_MARKER", encoding="utf-8")
    (tmp_path / "alias.md").symlink_to(tmp_path / "memo.md")
    status, page = _read(tmp_path, "alias.md")
    assert status == 200 and "INSIDE_MARKER" in page


@pytest.mark.parametrize("name", ["other memo.md", "other#memo.md", "literal%20name.md"])
def test_markdown_url_decodes_filename_once(tmp_path: Path, name: str) -> None:
    """DAT-001: URL path decoding must precede lookup, after fragment removal."""
    (tmp_path / name).write_text("# Target", encoding="utf-8")
    (tmp_path / "source.md").write_text(
        "[target](" + quote(name, safe="") + "?view=1#section)", encoding="utf-8",
    )
    status, page = _read(tmp_path, "source.md")
    assert status == 200
    href = html.escape("/read?" + urlencode({"path": name}), quote=True)
    assert f"href='{href}'>target</a>" in page, "DAT-001: encoded URL lost its memo identity"


def test_wiki_names_keep_literal_percent_sequences(tmp_path: Path) -> None:
    """Nonbehavioral control: wiki names are filenames, not encoded URL paths."""
    (tmp_path / "literal%20name.md").write_text("# Target", encoding="utf-8")
    (tmp_path / "source.md").write_text("[[literal%20name]]", encoding="utf-8")
    _, page = _read(tmp_path, "source.md")
    article = page.split("<article class='card prose'>", 1)[1].split("</article>", 1)[0]
    assert "href='/read?path=literal%2520name.md'" in article


@pytest.mark.parametrize("name", ["100%.md", "literal%20name.md", "hash#name.md"])
def test_reader_graph_fragment_encodes_memo_identity(tmp_path: Path, name: str) -> None:
    """BUG-002: graph.js decodes its fragment once, so the producer encodes once."""
    (tmp_path / name).write_text("# Memo", encoding="utf-8")
    _, page = _read(tmp_path, name)
    assert f"href='/graph#memo={quote(name, safe='')}'" in page, "BUG-002: graph fragment was not encoded"


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fence_prefixed_content_is_preserved(fence: str) -> None:
    """BUG-001: a fence prefix with trailing text cannot close a code block."""
    text = f"{fence}\nfirst\n{fence}not-a-close\nlast\n{fence}\n"
    expected = html.escape(f"first\n{fence}not-a-close\nlast")
    assert f"<pre><code>{expected}</code></pre>" in render(text, lambda _: None), "BUG-001: code content was discarded"


def test_deep_quotes_return_literal_fallback(tmp_path: Path) -> None:
    """BUG-003: parser depth exhaustion must be a page, not a request failure."""
    (tmp_path / "deep.md").write_text(">" * 1500 + " DEPTH_MARKER <b>", encoding="utf-8")
    error = None
    status, page = 0, ""
    try:
        status, page = _read(tmp_path, "deep.md")
    except RecursionError as exc:
        error = exc
    assert error is None, "BUG-003: recursive quotation processing crashed the reader"
    assert status == 200 and "DEPTH_MARKER &lt;b&gt;" in page
    assert "&gt;" in page, "excess nesting must remain visible as literal text"


@pytest.mark.parametrize("text", ["[" * 256 + " <b> **literal**", "ordinary " * 600 + "**literal**"], ids=["delimiter-budget", "length-budget"])
def test_parser_budget_preserves_escaped_literal_text(tmp_path: Path, text: str) -> None:
    """SEC-002: costly fragments fall back before regex parsing, without text loss."""
    (tmp_path / "budget.md").write_text(text, encoding="utf-8")
    status, page = _read(tmp_path, "budget.md")
    assert status == 200
    assert html.escape(text) in page, "SEC-002: fragment work budget did not take literal fallback"


def test_small_markdown_keeps_formatting() -> None:
    """Nonbehavioral control for the parser budget and depth guards."""
    page = render("# Heading\n\n> outer\n>> **inner**\n\n[web](https://example.org)", lambda _: None)
    assert "<h2>Heading</h2>" in page
    assert "<blockquote><p><strong>inner</strong></p></blockquote>" in page
    assert "rel='noopener noreferrer'" in page


@pytest.mark.parametrize("window", ["valid_from: 2099-01-01", "valid_until: 2000-01-01"])
def test_replacement_banner_describes_files_not_search(tmp_path: Path, window: str) -> None:
    """STAKES-001: a filesystem declaration is not an indexed search verdict."""
    (tmp_path / "old.md").write_text("# Old", encoding="utf-8")
    (tmp_path / "new.md").write_text(
        f"---\nsupersedes: old.md\n{window}\n---\n# New", encoding="utf-8",
    )
    _, page = _read(tmp_path, "old.md")
    assert "Another memo declares a replacement." in page, "STAKES-001: missing filesystem-qualified banner"
    assert "search treats this one as out of date" not in page
    assert "current state" not in page


def test_table_lookahead_obeys_parser_work_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-002: a following line must pass the budget before its table regex runs."""
    from recall.dashboard import markdown

    pattern = markdown._TABLE_RULE

    class GuardedPattern:
        def match(self, text: str):
            assert len(text) <= 4096, "SEC-002: table lookahead bypassed the work budget"
            return pattern.match(text)

    monkeypatch.setattr(markdown, "_TABLE_RULE", GuardedPattern())
    text = "a | b\n" + " " * 5000 + "x- <b>"
    page = render(text, lambda _: None)
    assert html.escape(" " * 5000 + "x- <b>") in page


def test_graph_link_scan_does_not_cross_unmatched_openers(tmp_path: Path) -> None:
    """SEC-002: malformed prefix candidates cannot swallow a later valid link."""
    from recall.dashboard.graph import build_graph

    (tmp_path / "source.md").write_text("](" * 256 + "[label](target.md)", encoding="utf-8")
    (tmp_path / "target.md").write_text("# Target", encoding="utf-8")
    graph = build_graph(tmp_path, include_queue=False, lint=False)
    assert {"source": "source.md", "target": "target.md", "kind": "link"} in graph["edges"], "SEC-002: an unmatched prefix swallowed the valid graph link"


def test_graph_empty_description_does_not_capture_another_key(tmp_path: Path) -> None:
    """SEC-002: line-scoped display metadata cannot consume the next key.

    Red proof uses the previous graph._DESCRIPTION production regex, which
    reports 'type: project' as the empty description. The same node passes
    after restricting metadata whitespace to the current line.
    """
    from recall.dashboard.graph import build_graph

    (tmp_path / "memo.md").write_text(
        "---\ndescription:\n\ntype: project\n---\n# Memo", encoding="utf-8",
    )
    node = build_graph(tmp_path, include_queue=False, lint=False)["nodes"][0]
    assert node["description"] == "", "SEC-002: empty description consumed the next metadata key"
    assert node["type"] == "project"
