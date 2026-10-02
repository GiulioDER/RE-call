"""`supersedes:` holds several references (Validity Frontmatter 1.0, section 5, optional support).

The invariant: a memo that replaces two memos declares both, every reader sees both, and a memo
that declares one keeps exactly the value it had. The failure each test guards is the one the
pre-change code had, where a reader that took only a str dropped a list in silence.

Properties, one test each:

1. Every spelling the spec allows reads as the same references: a flow sequence, a block
   sequence (also nested under `metadata:`), and the key repeated. One reference stays a str,
   and the wikilink spellings `[name]` and `[[name]]` stay one reference.
2. The writer adds a reference to each of those shapes in place, keeps CRLF and indentation,
   reports an edge already declared, and refuses an empty `supersedes:`.
3. `lint` checks every reference: a dangling second one is reported.
4. `rewrite verify` counts an unresolved second reference.
5. `lint --fix` proposes a second edge for a memo that already declares one, keeps the first,
   and still refuses an empty `supersedes:`.
6. The reasoning graph's fallback yields one authored edge per reference.
7. A compiled AML record's list is never read as file references by the Python readers.
8. The arbiter treats a pair as declared whichever reference in the list names it.
9. (Postgres) The store's scan yields one row per listed reference, keeps a file with no or an
   empty list, and gives a compiled AML record one NULL row, whatever its list holds.
10. (Postgres) `PgVectorStore.supersession_all` resolves both edges of a memo declaring two.
11. (Postgres) A compiled AML record, built by `recall_aml.service.build_chunks` itself, adds
    nothing to `PgVectorStore.supersession_all`: no dangling claim on ``[]`` or ``["mem_…"]``,
    and no live edge from a record id resolving against the ``{chunk_id}.md`` file it names. The
    store's scan and the reasoning graph's Python fallback agree on that table.

Red proof, 2026-10-02, each a deliberate mutation of the production line named, with this file
unchanged, each failing in the named test's assertion, then restored (all green):

- P1 `parse_frontmatter` stops reading block items (the pre-change parser):
  `test_every_spelling_the_spec_allows_reads_as_the_same_references`.
- P2 `_supersedes_value` keeps only the first reference:
  `test_every_spelling_the_spec_allows_reads_as_the_same_references`.
- P3 `add_supersedes_target` rewrites a scalar without its existing reference:
  `test_the_writer_adds_to_every_shape_in_place`.
- P4 `lint_corpus` checks only the first reference: `test_lint_checks_every_reference`.
- P5 `rewrite verify` checks only the first reference: `test_verify_counts_an_unresolved_second_reference`.
- P6 `propose_fixes` treats any declared key as closed (the pre-change refusal):
  `test_fix_adds_a_second_edge_and_keeps_the_first`.
- P7 `chunk_supersedes_targets` keeps only the first reference:
  `test_the_graph_fallback_yields_one_edge_per_reference`.
- P8 `chunk_supersedes_targets` reads compiled AML records too:
  `test_a_compiled_aml_record_is_not_read_as_file_references`.
- P9 `declared_pair` checks only the first reference: `test_the_arbiter_sees_every_declared_reference`.
- P10 `SUPERSEDES_TARGET_ROWS_SQL` without the compiled-record exclusion:
  `test_the_scan_expands_a_list_and_gives_a_compiled_record_no_target`.
- P11 `PgVectorStore.supersession_all` back on ``metadata->>'supersedes'`` (the pre-change
  query): `test_the_store_resolves_both_edges_of_a_memo_declaring_two`.

Red proof for the compiled-record change, 2026-10-02, same method:

- C1 `SUPERSEDES_TARGET_ROWS_SQL` back to the #864 text, where a compiled row kept
  ``metadata->>'supersedes'``: `test_the_scan_expands_a_list_and_gives_a_compiled_record_no_target`
  (the compiled rows read ``'["x"]'``, ``'[]'`` and ``'x.md'``) and
  `test_a_compiled_aml_record_adds_nothing_to_the_store_scan` (the edges gain the keys ``[]``
  and ``["mem_…"]``).
- C2 the same fragment with the compiled exclusion removed, so compiled arrays EXPAND:
  `test_a_compiled_aml_record_adds_nothing_to_the_store_scan` (the edges gain
  ``mem_….md -> mem_….md``, a live edge between two compiled records).
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.cli import main
from recall.fix import apply_proposal, propose_fixes
from recall.frontmatter import (
    SupersedesDeclaresNothing,
    add_supersedes_target,
    parse_frontmatter,
    supersedes_targets,
)
from recall.lint import lint_corpus
from recall.reasoning_graph import build_reasoning_graph
from recall.store import PgVectorStore
from recall.supersession import SUPERSEDES_TARGET_ROWS_SQL, chunk_supersedes_targets
from recall.supersession_arbiter import declared_pair, read_note
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db


def _targets(text: str) -> tuple[str, ...]:
    return supersedes_targets(parse_frontmatter(text)[0].get("supersedes"))


def test_every_spelling_the_spec_allows_reads_as_the_same_references() -> None:
    both = ("a.md", "b.md")
    assert _targets("---\nsupersedes: [a.md, \"b.md\"]\n---\nx") == both
    assert _targets("---\nsupersedes:\n  - a.md\n  - b.md\nvalid_from: 2026-01-01\n---\nx") == both
    nested = "---\nmetadata:\n  supersedes:\n    - a.md\n    - b.md\n  type: project\n---\nx"
    assert _targets(nested) == both
    assert parse_frontmatter(nested)[0]["type"] == "project"
    assert _targets("---\nsupersedes: a.md\nsupersedes: b.md\n---\nx") == both
    # The same document spelled twice is one reference.
    assert _targets("---\nsupersedes: [a.md, a, b.md]\n---\nx") == ("a.md", "b.md")
    # One reference keeps the str it always had, so single-edge chunk metadata does not move.
    assert parse_frontmatter("---\nsupersedes: a.md\n---\nx")[0]["supersedes"] == "a.md"
    assert parse_frontmatter("---\nsupersedes: [[a]]\n---\nx")[0]["supersedes"] == "[[a]]"
    assert parse_frontmatter("---\nsupersedes: [a]\n---\nx")[0]["supersedes"] == "[a]"
    # Present and empty still means "supersedes nothing", in every spelling.
    assert parse_frontmatter("---\nsupersedes:\n---\nx")[0]["supersedes"] == ""
    assert parse_frontmatter("---\nsupersedes: []\n---\nx")[0]["supersedes"] == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"# x\n", b"---\nsupersedes: c.md\n---\n# x\n"),
        (b"---\nsupersedes: a.md\n---\nb\n", b"---\nsupersedes:\n  - a.md\n  - c.md\n---\nb\n"),
        (
            b"---\r\nsupersedes: a.md\r\ntype: x\r\n---\r\nb\r\n",
            b"---\r\nsupersedes:\r\n  - a.md\r\n  - c.md\r\ntype: x\r\n---\r\nb\r\n",
        ),
        (
            b"---\nmetadata:\n  supersedes: a.md\n  type: x\n---\nb\n",
            b"---\nmetadata:\n  supersedes:\n    - a.md\n    - c.md\n  type: x\n---\nb\n",
        ),
        (
            b"---\nsupersedes: [a.md, b.md]\n---\nb\n",
            b"---\nsupersedes:\n  - a.md\n  - b.md\n  - c.md\n---\nb\n",
        ),
        (
            b"---\nsupersedes:\n    - a.md\nvalid_from: 2026-01-01\n---\nb\n",
            b"---\nsupersedes:\n    - a.md\n    - c.md\nvalid_from: 2026-01-01\n---\nb\n",
        ),
        (
            b"---\nsupersedes: a.md\nsupersedes: b.md\n---\nb\n",
            b"---\nsupersedes: a.md\nsupersedes: b.md\nsupersedes: c.md\n---\nb\n",
        ),
        (b"---\nsupersedes: [[c]]\n---\nb\n", None),
    ],
    ids=["absent", "scalar", "crlf", "nested", "flow", "block", "repeated", "already"],
)
def test_the_writer_adds_to_every_shape_in_place(raw: bytes, expected: bytes | None) -> None:
    assert add_supersedes_target(raw, "c.md") == expected


def test_the_writer_refuses_an_empty_declaration() -> None:
    with pytest.raises(SupersedesDeclaresNothing):
        add_supersedes_target(b"---\nsupersedes:\n---\nb\n", "c.md")


def _write(directory, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8", newline="\n")


def test_lint_checks_every_reference(tmp_path) -> None:
    _write(tmp_path, "a.md", "# a\n")
    _write(tmp_path, "new.md", "---\nsupersedes: [a.md, missing.md]\n---\n# new\n")
    codes = {(issue.file, issue.code) for issue in lint_corpus(tmp_path)}
    assert ("new.md", "dangling-supersedes") in codes


def test_verify_counts_an_unresolved_second_reference(tmp_path, capsys) -> None:
    _write(tmp_path, "a.md", "# a\n")
    _write(tmp_path, "new.md", "---\nsupersedes:\n  - a.md\n  - missing.md\n---\n# new\n")
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "verify", str(tmp_path)])
    assert exc.value.code == 1
    assert "UNRESOLVED new.md: supersedes 'missing.md'" in capsys.readouterr().out


def test_fix_adds_a_second_edge_and_keeps_the_first(tmp_path) -> None:
    _write(tmp_path, "old_thing_2026.md", "# old\n\nbody")
    _write(tmp_path, "other_thing_2026.md", "# other\n\nbody")
    _write(
        tmp_path,
        "new.md",
        "---\nsupersedes: other_thing_2026.md\n---\n# new\n\nThis supersedes [[old_thing_2026]].",
    )
    proposals, unfixable = propose_fixes(tmp_path)
    assert [p.edit_file for p in proposals] == ["new.md"] and unfixable == []
    apply_proposal(tmp_path, proposals[0])
    text = (tmp_path / "new.md").read_text(encoding="utf-8")
    assert {t.removesuffix(".md") for t in _targets(text)} == {"other_thing_2026", "old_thing_2026"}
    # An empty declaration is still the human's to change.
    _write(tmp_path, "new.md", "---\nsupersedes:\n---\n# new\n\nThis supersedes [[old_thing_2026]].")
    proposals, unfixable = propose_fixes(tmp_path)
    assert proposals == [] and "no reference" in unfixable[0].reason


def _chunk(name: str, metadata: dict[str, object]) -> Chunk:
    return Chunk(name, name, "text", {"file": name, "ord": 0, **metadata})


def test_the_graph_fallback_yields_one_edge_per_reference() -> None:
    chunks = [
        _chunk("a.md", {}),
        _chunk("b.md", {}),
        _chunk("new.md", {"supersedes": ["a.md", "b.md"]}),
    ]
    graph = build_reasoning_graph(
        chunks, tenant_id="t", generation_id="g", pipeline_fingerprint="p"
    )
    assert graph.authored_supersession_map() == {"a.md": "new.md", "b.md": "new.md"}


def test_a_compiled_aml_record_is_not_read_as_file_references() -> None:
    compiled = {"record_type": "compiled", "supersedes": ["a"], "file": "x.md"}
    assert chunk_supersedes_targets(compiled) == ()
    assert chunk_supersedes_targets({"supersedes": ["a.md", "b.md"]}) == ("a.md", "b.md")


def test_the_arbiter_sees_every_declared_reference() -> None:
    newer = read_note("2026-02-01-new.md", "---\nsupersedes: [x.md, 2026-01-01-old.md]\n---\nbody")
    older = read_note("2026-01-01-old.md", "body")
    assert declared_pair(older, newer) and declared_pair(newer, older)


@requires_db
def test_the_scan_expands_a_list_and_gives_a_compiled_record_no_target() -> None:
    rows = [
        '{"file": "a.md"}',
        '{"file": "b.md"}',
        '{"file": "new.md", "supersedes": ["a.md", "b.md"]}',
        '{"file": "one.md", "supersedes": "a.md"}',
        '{"file": "empty.md", "supersedes": []}',
        '{"file": "rec.md", "record_type": "compiled", "supersedes": ["x"]}',
        '{"file": "rec_empty.md", "record_type": "compiled", "supersedes": []}',
        '{"file": "rec_str.md", "record_type": "compiled", "supersedes": "x.md"}',
    ]
    values = ", ".join(f"('{row}'::jsonb)" for row in rows)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        found = conn.execute(
            f"WITH c(metadata) AS (VALUES {values}) "
            f"SELECT c.metadata->>'file', t.supersedes FROM c "
            f"CROSS JOIN LATERAL ({SUPERSEDES_TARGET_ROWS_SQL}) AS t"
        ).fetchall()
    # Sorted here, not by ORDER BY: the database's collation ignores `_` and `.` when ordering.
    assert sorted(found, key=lambda row: (row[0], row[1] or "")) == [
        ("a.md", None),
        ("b.md", None),
        ("empty.md", None),
        ("new.md", "a.md"),
        ("new.md", "b.md"),
        ("one.md", "a.md"),
        ("rec.md", None),
        ("rec_empty.md", None),
        ("rec_str.md", None),
    ]


@requires_db
def test_the_store_resolves_both_edges_of_a_memo_declaring_two() -> None:
    from recall.embeddings import HashingEmbedder

    table = "ms_" + uuid.uuid4().hex[:8]
    emb = HashingEmbedder(dim=64)
    store = PgVectorStore(TEST_DSN, dim=64, table=table)
    try:
        store.ensure_schema()
        chunks = [
            _chunk("a.md", {}),
            _chunk("b.md", {}),
            _chunk("new.md", {"supersedes": ["a.md", "b.md"]}),
        ]
        store.upsert(chunks, emb.embed([c.text for c in chunks]))
        edges, unresolved, _ = store.supersession_all()
        assert edges == {"a.md": "new.md", "b.md": "new.md"}
        assert not unresolved
    finally:
        store.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")


@requires_db
def test_a_compiled_aml_record_adds_nothing_to_the_store_scan() -> None:
    from recall.embeddings import HashingEmbedder
    from recall.reasoning_graph import _supersession_rows
    from recall.supersession import resolve_supersession_candidates
    from recall_aml.models import AddRequest, CodingMemoryRecord, Message
    from recall_aml.service import build_chunks

    request = AddRequest(
        request_id="compiled-supersedes",
        user_id="u",
        session_id="s",
        messages=[Message(role="user", content="the build failed, then the pin fixed it")],
    )

    def record(action: str, supersedes: list[str]) -> CodingMemoryRecord:
        return CodingMemoryRecord(
            kind="successful repair",
            action=action,
            source_session_id="s",
            supersedes=supersedes,
        )

    older = record("pin the dependency", [])
    (older_chunk,) = build_chunks(request, [older], include_raw=False)
    newer = record("pin the dependency to 2.1", [older_chunk.id])
    aml_chunks = build_chunks(request, [older, newer], include_raw=True)
    compiled = [c for c in aml_chunks if c.metadata["record_type"] == "compiled"]
    # The fixture must hold what the service writes, or the assertions below observe nothing.
    assert [c.metadata["supersedes"] for c in compiled] == [[], [older_chunk.id]]
    assert {c.metadata["file"] for c in compiled} == {f"{c.id}.md" for c in compiled}
    assert any(c.metadata["record_type"] == "raw" for c in aml_chunks)

    chunks = [
        *aml_chunks,
        _chunk("a.md", {}),
        _chunk("new.md", {"supersedes": "a.md"}),
    ]
    table = "ms_" + uuid.uuid4().hex[:8]
    emb = HashingEmbedder(dim=64)
    store = PgVectorStore(TEST_DSN, dim=64, table=table)
    try:
        store.ensure_schema()
        store.upsert(chunks, emb.embed([c.text for c in chunks]))
        edges, unresolved, candidates = store.supersession_all()
    finally:
        store.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

    assert edges == {"a.md": "new.md"}
    assert not unresolved
    assert set(candidates) == {"a.md"}
    fallback, _fallback_unresolved, _ = resolve_supersession_candidates(_supersession_rows(chunks))
    assert fallback == edges
