"""A small, safe Markdown renderer for reading memos on the dashboard.

Standard library only, like the rest of the dashboard. It covers what memos are written with:
headings, paragraphs, bullet and numbered lists (nested by indentation), fenced code, block
quotes, tables, rules, and inline code, bold, italics, links, ``[[wiki links]]`` and bare web
addresses.

Safety is by construction rather than by sanitising afterwards: every piece of memo text passes
through `html.escape` on its way out, and the only tags in the output are the ones built here. A
link becomes an anchor only when `resolve` says what it is: a memo of this store (an in-dashboard
link) or an ``http``/``https`` address (opened with ``rel="noopener noreferrer"``). Anything else,
including ``javascript:`` and ``data:`` addresses, is shown as text.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from urllib.parse import urlencode

#: Given a link target, the memo it names (a path relative to the store), or None.
Resolver = Callable[[str], "str | None"]

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_INLINE = re.compile(
    r"(?P<code>`+)(?P<code_text>.+?)(?P=code)"
    r"|\[\[(?P<wiki>[^\]\n]+)\]\]"
    r"|\[(?P<link_text>[^\]\n]+)\]\((?P<link_url>[^)\s]+)\)"
    r"|\*\*(?P<bold>.+?)\*\*"
    r"|(?<![\w*])\*(?P<em>[^\s*](?:.*?[^\s*])?)\*(?![\w*])"
    r"|(?P<url>https?://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"])"
)
_WEB = re.compile(r"^https?://", re.IGNORECASE)


def memo_href(name: str) -> str:
    return "/read?" + urlencode({"path": name})


def _anchor_for(target: str, label_html: str, resolve: Resolver) -> str:
    memo = resolve(target)
    if memo is not None:
        return f"<a class='memo-link' href='{html.escape(memo_href(memo), quote=True)}'>{label_html}</a>"
    if _WEB.match(target):
        return f"<a href='{html.escape(target, quote=True)}' rel='noopener noreferrer' target='_blank'>{label_html}</a>"
    return label_html


def inline(text: str, resolve: Resolver) -> str:
    """One line or paragraph of memo text as escaped HTML with its inline formatting."""
    out: list[str] = []
    position = 0
    for match in _INLINE.finditer(text):
        out.append(html.escape(text[position: match.start()]))
        position = match.end()
        if match.group("code"):
            out.append(f"<code>{html.escape(match.group('code_text').strip())}</code>")
        elif match.group("wiki"):
            target, _, alias = match.group("wiki").partition("|")
            label = html.escape((alias or target).strip())
            memo = resolve(target.split("#", 1)[0].strip())
            out.append(
                f"<a class='memo-link' href='{html.escape(memo_href(memo), quote=True)}'>{label}</a>" if memo
                else f"<span class='broken-link' title='No memo in this store has that name'>{label}</span>"
            )
        elif match.group("link_text"):
            out.append(_anchor_for(match.group("link_url"), inline(match.group("link_text"), resolve), resolve))
        elif match.group("bold"):
            out.append(f"<strong>{inline(match.group('bold'), resolve)}</strong>")
        elif match.group("em"):
            out.append(f"<em>{inline(match.group('em'), resolve)}</em>")
        else:
            url = match.group("url")
            out.append(_anchor_for(url, html.escape(url), resolve))
    out.append(html.escape(text[position:]))
    return "".join(out)


def _cells(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _list(lines: list[str], resolve: Resolver) -> str:
    """Nested lists from indented items; a line that is not an item continues the previous one."""
    items: list[tuple[int, bool, str]] = []
    for line in lines:
        found = _ITEM.match(line)
        if found:
            items.append((len(found.group(1).expandtabs(4)), found.group(2)[0].isdigit(), found.group(3)))
        elif items:
            indent, ordered, text = items[-1]
            items[-1] = (indent, ordered, f"{text} {line.strip()}")
    out: list[str] = []
    stack: list[tuple[int, str]] = []
    for indent, ordered, text in items:
        tag = "ol" if ordered else "ul"
        while stack and indent < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or indent > stack[-1][0]:
            out.append(f"<{tag}><li>")
            stack.append((indent, tag))
        else:
            out.append("</li><li>")
        out.append(inline(text, resolve))
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def render(text: str, resolve: Resolver, *, skip_title: str | None = None) -> str:
    """A memo body as HTML. `skip_title` drops a first heading equal to it (the page shows it)."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(part.strip() for part in paragraph), resolve)}</p>")
            paragraph.clear()

    first_heading = True
    while index < len(lines):
        line = lines[index]
        fence = _FENCE.match(line)
        if fence:
            flush()
            marker, block = fence.group(1), []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(marker):
                block.append(lines[index])
                index += 1
            index += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(block))}</code></pre>")
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            title = heading.group(2)
            if not (first_heading and skip_title and title.strip() == skip_title.strip()):
                level = min(len(heading.group(1)) + 1, 6)
                out.append(f"<h{level}>{inline(title, resolve)}</h{level}>")
            first_heading = False
            index += 1
            continue
        if _RULE.match(line):
            flush()
            out.append("<hr>")
            index += 1
            continue
        if line.lstrip().startswith(">"):
            flush()
            quoted = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                quoted.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            out.append(f"<blockquote>{render(chr(10).join(quoted), resolve)}</blockquote>")
            continue
        if "|" in line and index + 1 < len(lines) and _TABLE_RULE.match(lines[index + 1]) and "-" in lines[index + 1]:
            flush()
            head = "".join(f"<th>{inline(cell, resolve)}</th>" for cell in _cells(line))
            index += 2
            rows = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append("<tr>" + "".join(f"<td>{inline(cell, resolve)}</td>" for cell in _cells(lines[index])) + "</tr>")
                index += 1
            out.append(f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
            continue
        if _ITEM.match(line):
            flush()
            block = []
            while index < len(lines) and lines[index].strip() and (
                _ITEM.match(lines[index]) or lines[index].startswith((" ", "\t"))
            ):
                block.append(lines[index])
                index += 1
            out.append(_list(block, resolve))
            continue
        if not line.strip():
            flush()
            index += 1
            continue
        paragraph.append(line)
        index += 1
    flush()
    return "".join(out)
