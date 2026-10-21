"""The anchor reference reproduces the shared test vectors.

``tests/fixtures/audit_memory/anchor_vectors.json`` was generated from hand-written spans, with
hashes taken by an independent regex normaliser, so these tests check `recall.anchors` against a
second statement of the rules rather than against itself. cca-audit carries a byte-identical copy
of the vectors and must pass the same cases.

Red proof (2026-10-21, `recall/anchors.py`, run on the deployment host inside the test bounds):
- `normalize` with ``.replace("\\r", "\\n")`` removed (a lone CR not unified):
  `test_crlf_does_not_change_sha` failed on ``code_sha("a\\rb\\r")``.
- `resolve_symbols` naming every repeat plainly (no ``#N``):
  `test_property_setter_gets_a_suffix` failed on the symbol list.
- `normalize` without ``rstrip``: `test_trailing_whitespace_does_not_change_sha` failed on the
  hash comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.anchors import code_sha, enclosing_symbol, resolve_symbols, symbol_sha

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "audit_memory" / "anchor_vectors.json").read_text(
        encoding="utf-8"
    )
)
BY_NAME = {case["name"]: case for case in VECTORS}


@pytest.mark.parametrize("case", VECTORS, ids=[c["name"] for c in VECTORS])
def test_vectors_reproduce(case: dict) -> None:
    source = case["source"]
    spans = [
        {"symbol": s.symbol, "start_line": s.start_line, "end_line": s.end_line}
        for s in resolve_symbols(source)
    ]
    assert spans == [{k: e[k] for k in ("symbol", "start_line", "end_line")} for e in case["expect"]]
    assert [symbol_sha(source, e["symbol"]) for e in case["expect"]] == [
        e["sha"] for e in case["expect"]
    ]
    assert symbol_sha(source, None) == case["file_sha"]
    assert [enclosing_symbol(source, e["line"]) for e in case["enclosing"]] == [
        e["symbol"] for e in case["enclosing"]
    ]


def test_crlf_does_not_change_sha() -> None:
    assert BY_NAME["crlf_line_endings"]["expect"] == BY_NAME["class_with_methods"]["expect"]
    assert code_sha("a\r\nb\r\n") == code_sha("a\nb\n") == code_sha("a\rb\r")


def test_trailing_whitespace_does_not_change_sha() -> None:
    assert code_sha("x = 1   \n\ty\t\n") == code_sha("x = 1\n\ty\n")
    assert BY_NAME["trailing_whitespace_only"]["expect"] == BY_NAME["class_with_methods"]["expect"]


def test_body_change_changes_only_the_enclosing_shas() -> None:
    before = {e["symbol"]: e["sha"] for e in BY_NAME["class_with_methods"]["expect"]}
    after = {e["symbol"]: e["sha"] for e in BY_NAME["body_change"]["expect"]}
    assert before["Pool.lease"] != after["Pool.lease"]
    assert before["Pool"] != after["Pool"]
    assert before["Pool.release"] == after["Pool.release"]


def test_property_setter_gets_a_suffix() -> None:
    names = [s.symbol for s in resolve_symbols(BY_NAME["property_with_setter"]["source"])]
    assert names == ["Box", "Box.value", "Box.value#2"]


def test_unparseable_source_raises_rather_than_reporting_no_symbols() -> None:
    with pytest.raises(SyntaxError):
        resolve_symbols("def broken(:\n    pass\n")


def test_unknown_symbol_raises() -> None:
    with pytest.raises(KeyError):
        symbol_sha("def f():\n    return 1\n", "g")
