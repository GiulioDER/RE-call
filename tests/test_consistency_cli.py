"""End to end: a repo goes in, a report naming both versions of a claim comes out."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall_consistency.__main__ import _load_questions, _run_probe, main
from tests.consistency_helpers import two_commit_repo


class _FakeEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim


class _FakeStore:
    """Duck-typed like `PgVectorStore` for exactly what `_run_probe` touches: `check_schema()`
    and the context-manager protocol. `check_schema` either passes (the table matches the
    embedder that opened it) or raises `SchemaIncompatible` naming the mismatch, the same choice
    a real store makes after actually reading the table.
    """

    def __init__(self, *, incompatible_as: str = "") -> None:
        self._incompatible_as = incompatible_as

    def check_schema(self) -> None:
        if self._incompatible_as:
            from recall.schema import SchemaIncompatible

            raise SchemaIncompatible(self._incompatible_as)

    def __enter__(self) -> "_FakeStore":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _repo(tmp_path: Path) -> Path:
    """A repo whose document restates one claim with a different number."""
    return two_commit_repo(
        tmp_path,
        "recall at 5 scored 0.92 on known-item queries\n",
        "recall at 5 scored 0.945 on known-item queries\n",
    )


def test_the_cli_writes_a_report_quoting_both_versions(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "report.md"

    rc = main([str(repo), "--out", str(out), "--name", "acme-memory"])

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "acme-memory" in text
    assert "0.92" in text
    assert "0.945" in text
    assert "1 restated claim" in text


def test_the_cli_can_also_emit_the_history_corpus(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    corpus = tmp_path / "corpus"

    rc = main([str(repo), "--out", str(tmp_path / "r.md"), "--corpus-out", str(corpus)])

    assert rc == 0
    memos = sorted(corpus.glob("*.md"))
    assert len(memos) == 2
    assert any("supersedes:" in m.read_text(encoding="utf-8") for m in memos)


def test_the_report_is_written_with_lf_endings(tmp_path: Path) -> None:
    out = tmp_path / "report.md"

    main([str(_repo(tmp_path)), "--out", str(out)])

    assert b"\r\n" not in out.read_bytes()


def test_a_questions_file_that_is_a_json_object_fails_without_naming_its_keys(
    tmp_path: Path,
) -> None:
    """A dict, not a list, must not be silently walked as if its keys were questions."""
    repo = _repo(tmp_path)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"what is the rate limit?": 1}), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                str(repo),
                "--out",
                str(tmp_path / "report.md"),
                "--questions",
                str(questions),
                "--serving-dsn",
                "postgresql://unused/dummy",
            ]
        )

    message = str(exc_info.value)
    assert "what is the rate limit?" not in message


def test_a_questions_entry_that_is_neither_a_string_nor_a_query_object_names_its_position(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(["a fine question", {"not_query": "secret internal detail"}]),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                str(repo),
                "--out",
                str(tmp_path / "report.md"),
                "--questions",
                str(questions),
                "--serving-dsn",
                "postgresql://unused/dummy",
            ]
        )

    message = str(exc_info.value)
    assert "entry 2" in message
    assert "secret internal detail" not in message


def test_run_probe_reaches_the_real_probe_with_injected_fakes() -> None:
    """No mock of `probe` itself and no monkeypatched import: a bare fake embedder and store are
    injected, and the real `probe` -> `trusted_search` call chain runs against them. The strict
    trust policy refuses an uncalibrated store before touching retrieval at all, which is only
    reachable if the injected objects made it all the way through `_run_probe`'s wiring,
    including a `check_schema()` call that passes.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _run_probe(
            "unused-dsn",
            "chunks",
            ["a question"],
            make_embedder=lambda: _FakeEmbedder(dim=8),
            make_store=lambda dsn, dim, table: _FakeStore(),
        )

    assert "question 1 of 1" in str(excinfo.value)


def test_run_probe_exits_naming_both_dimensions_on_a_mismatch() -> None:
    """`check_schema()` is what actually reads the indexed table; its `SchemaIncompatible`
    message already names both dimensions, and `_run_probe` must surface that, not swallow it.
    """
    with pytest.raises(SystemExit) as excinfo:
        _run_probe(
            "unused-dsn",
            "chunks",
            ["a question"],
            make_embedder=lambda: _FakeEmbedder(dim=8),
            make_store=lambda dsn, dim, table: _FakeStore(
                incompatible_as="table 'chunks' uses vector(16), requested dimension is 8"
            ),
        )

    message = str(excinfo.value)
    assert "8" in message
    assert "16" in message


def test_the_cli_creates_the_out_directory_if_missing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "nested" / "dir" / "report.md"

    rc = main([str(repo), "--out", str(out)])

    assert rc == 0
    assert out.exists()


def test_a_questions_file_carrying_contact_fields_is_refused(tmp_path: Path) -> None:
    """The audit needs the question text and nothing else, so contact data is refused on sight.

    Reading past the field would still mean the file was accepted and now sits on disk under
    whatever was promised about retention. Delete the `_contact_keys` branch in `_load_questions`
    and this goes red, because the entry has a valid `query` and would otherwise load cleanly.
    """
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps([{"query": "what is the rate limit?", "email": "someone@example.com"}]),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main([str(_repo(tmp_path)), "--out", str(tmp_path / "r.md"),
              "--questions", str(questions), "--serving-dsn", "postgresql://unused"])

    message = str(excinfo.value)
    assert "entry 1" in message
    assert "email" in message
    assert "someone@example.com" not in message
    assert "field names matching" in message


def test_a_suffixed_contact_field_is_refused_too(tmp_path: Path) -> None:
    """`author_email` is not in the token list; the `_email` suffix rule is what catches it.

    Delete the `endswith("_email")` clause and this goes red while the test above stays green.
    """
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps([{"query": "what is the rate limit?", "author_email": "a@example.com"}]),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main([str(_repo(tmp_path)), "--out", str(tmp_path / "r.md"),
              "--questions", str(questions), "--serving-dsn", "postgresql://unused"])

    message = str(excinfo.value)
    assert "email" in message
    assert "a@example.com" not in message
    # The stem is reported, never the field name: a name can itself be contact data.
    assert "author_email" not in message


@pytest.mark.parametrize(
    ("entry", "expected_stem"),
    [
        ({"query": "q", "contactEmail": "a@example.com"}, "email"),
        ({"query": "q", "phoneNumber": "555"}, "phone"),
        ({"query": "q", "emails": ["a@example.com"]}, "email"),
        ({"query": "q", "meta": {"email": "a@example.com"}}, "email"),
        ({"query": "q", "authors": [{"email": "a@example.com"}]}, "email"),
    ],
    ids=["camel", "camel-suffix", "plural", "nested-dict", "nested-list"],
)
def test_contact_fields_are_caught_however_they_are_spelled_or_buried(
    tmp_path: Path, entry: dict[str, object], expected_stem: str
) -> None:
    """An exact-name list against top-level keys let all five of these through.

    Remove the recursion in `_contact_keys` and the two nested cases go red. Match exact
    tokens instead of stems and the camel and plural cases go red. The camel cases are carried by
    stem matching rather than by any case handling: `contactEmail` lowered contains `email`.
    """
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main([str(_repo(tmp_path)), "--out", str(tmp_path / "r.md"),
              "--questions", str(questions), "--serving-dsn", "postgresql://unused"])

    message = str(excinfo.value)
    assert expected_stem in message
    assert "a@example.com" not in message
    assert "555" not in message


def test_an_ordinary_questions_file_still_loads(tmp_path: Path) -> None:
    """The guard must not refuse a file that carries no contact data at all."""
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps(["plain string question", {"query": "object question", "notes": "fine"}]),
        encoding="utf-8",
    )

    assert _load_questions(questions) == ["plain string question", "object question"]


def test_a_key_that_is_itself_a_contact_value_is_never_echoed(tmp_path: Path) -> None:
    """A JSON object can be keyed BY an address, and `gmail.com` matches the `mail` stem.

    Reporting the field name would then print the address itself, inverting the guard for the
    most common personal email domains. Report the stem instead and the address never appears.
    Change `_contact_stems` to collect key names and this goes red.
    """
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps([{"query": "what is the SLA?", "jane.doe@gmail.com": "asked this"}]),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main([str(_repo(tmp_path)), "--out", str(tmp_path / "r.md"),
              "--questions", str(questions), "--serving-dsn", "postgresql://unused"])

    message = str(excinfo.value)
    assert "mail" in message
    assert "jane.doe" not in message
    assert "@" not in message


def test_an_absurdly_nested_entry_is_refused_rather_than_crashing(tmp_path: Path) -> None:
    """The bound is 32, so 100 levels must be refused rather than walked.

    Depth is deliberately just past the bound rather than past CPython's stack. An earlier version
    of this test nested 3000 levels to provoke a real RecursionError, which passed locally on 3.14
    and failed on the 3.11 floor job inside `json.dumps` itself, before the guard was ever
    reached. That version tested the interpreter's stack limit; this one tests
    `MAX_CONTACT_DEPTH`.

    Remove the `MAX_CONTACT_DEPTH` check and no SystemExit is raised at all, because the entry
    carries a valid `query` and would load cleanly, so this goes red.
    """
    nested: dict[str, object] = {"query": "q"}
    cursor = nested
    for _ in range(100):
        child: dict[str, object] = {}
        cursor["deeper"] = child
        cursor = child
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([nested]), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_questions(questions)

    assert "nests deeper" in str(excinfo.value)
