"""Code anchors: the key that joins an audit record to the code it describes.

An anchor is ``(file, symbol, code_sha)``. CCA computes it for every bug, false positive and lesson
it records; the code chunker writes the same triple onto Python code chunks. Two implementations of
one key in two repositories agree only if both reproduce the same test vectors
(``tests/fixtures/audit_memory/anchor_vectors.json``), so this module is the reference those vectors
were checked against, and its rules are the contract:

- A symbol is every function and class ``ast`` reports, dotted by nesting: a method is
  ``Pool._lease``, a nested function ``outer.inner`` (never Python's ``outer.<locals>.inner``).
- A symbol's span runs from its first decorator to its last line.
- A dotted name seen again in one file (a property and its setter) becomes ``name#2``, then ``#3``,
  in source order, so two definitions never share an anchor.
- ``code_sha`` hashes text after CRLF and CR become LF and trailing whitespace is stripped from every
  line, so a Windows checkout and a Linux checkout of one commit produce one anchor.

See ``docs/AUDIT_MEMORY_DESIGN.md``.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True)
class SymbolSpan:
    """One function or class: its dotted name and 1-based inclusive line range."""

    symbol: str
    start_line: int
    end_line: int


def normalize(text: str) -> str:
    """The text every hash is taken over: LF line endings, no trailing whitespace on any line."""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in unified.split("\n"))


def code_sha(text: str) -> str:
    """SHA-256 hex of `normalize(text)` encoded as UTF-8."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def resolve_symbols(source: str) -> list[SymbolSpan]:
    """Every function and class in `source`, parents before children, in source order.

    Raises `SyntaxError` when `source` is not parseable Python: a caller that falls back to a
    file-level anchor must decide that, not receive an empty list that reads as "no symbols".
    """
    tree = ast.parse(normalize(source).removeprefix("﻿"))
    spans: list[SymbolSpan] = []
    seen: dict[str, int] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, _Definition):
                dotted = f"{prefix}{node.name}"
                seen[dotted] = seen.get(dotted, 0) + 1
                name = dotted if seen[dotted] == 1 else f"{dotted}#{seen[dotted]}"
                start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                spans.append(SymbolSpan(name, start, node.end_lineno or node.lineno))
                visit(node.body, f"{dotted}.")
            else:
                # A definition under if/try/with/for/while/match belongs to the enclosing scope.
                for child_body in _statement_bodies(node):
                    visit(child_body, prefix)

    visit(tree.body, "")
    return spans


def _statement_bodies(node: ast.stmt) -> list[list[ast.stmt]]:
    bodies = [getattr(node, "body")] if hasattr(node, "body") else []
    bodies += [h.body for h in getattr(node, "handlers", [])]
    bodies += [getattr(node, f) for f in ("orelse", "finalbody") if hasattr(node, f)]
    bodies += [c.body for c in getattr(node, "cases", [])]
    return [b for b in bodies if isinstance(b, list)]


def enclosing_symbol(source: str, line: int) -> str | None:
    """The innermost symbol whose span contains `line`, or None at module level."""
    containing = [s for s in resolve_symbols(source) if s.start_line <= line <= s.end_line]
    if not containing:
        return None
    return min(containing, key=lambda s: (s.end_line - s.start_line, -s.start_line)).symbol


def symbol_sha(source: str, symbol: str | None) -> str:
    """`code_sha` of `symbol`'s span in `source`, or of the whole file when `symbol` is None.

    Raises `KeyError` when `symbol` is not in `source`.
    """
    if symbol is None:
        return code_sha(source)
    span = {s.symbol: s for s in resolve_symbols(source)}[symbol]
    lines = normalize(source).split("\n")
    return code_sha("\n".join(lines[span.start_line - 1 : span.end_line]))


__all__ = [
    "SymbolSpan",
    "code_sha",
    "enclosing_symbol",
    "normalize",
    "resolve_symbols",
    "symbol_sha",
]
