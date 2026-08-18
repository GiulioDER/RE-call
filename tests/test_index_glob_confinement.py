"""A single-file index request must obey the same glob as a directory walk.

`candidate_files` filtered a DIRECTORY walk to the glob (`**/*.md`) and returned a SINGLE FILE
unconditionally. Confinement to `RECALL_INDEX_ROOT` was treated as the whole control, so the
file-type filter silently did not apply to the branch a client is most likely to call.

That is a credential-disclosure path, not a tidiness issue. `RECALL_INDEX_ROOT` defaults to the
server's working directory; `recall/_env.py` loads `.env` from that same directory; and
docs/AUTH.md's quickstart writes a RELATIVE `tokens.json` there, whose first principal holds a
PLAINTEXT bearer token. So a principal with `recall:write` + `recall:read` on one tenant could
call `recall_index("tokens.json")` and then read every other principal's token — including
principals on OTHER tenants — straight back out of `recall_search`. `chmod 600` does not help:
the read is performed by the server's own user.

The refusal is loud rather than silent (an empty candidate list would report "indexed 0 files",
exit 0 — the same silence the prune guard exists to break) and it names the escape hatch, so the
legitimate `recall index --glob '**/*.py'` case still works.
"""
from __future__ import annotations

import pytest

from recall.index import DEFAULT_INDEX_GLOB, candidate_files
from recall_mcp.service import index_memory


def test_single_file_outside_the_glob_is_refused(tmp_path):
    """The exact disclosure path: a secrets file named directly, not reached by a walk."""
    secret = tmp_path / ".env"
    secret.write_text("VOYAGE_API_KEY=sk-live-EXAMPLE\n")

    with pytest.raises(ValueError) as exc:
        candidate_files(secret)

    message = str(exc.value)
    assert DEFAULT_INDEX_GLOB in message, "the refusal must name the glob that excluded the file"
    assert "sk-live-EXAMPLE" not in message, "an error must never echo the file's contents"


def test_single_file_matching_the_glob_is_still_indexed(tmp_path):
    """The refusal must not break the ordinary single-memo index."""
    memo = tmp_path / "notes.md"
    memo.write_text("# a memo\n")

    assert candidate_files(memo) == [memo]


def test_a_custom_glob_still_admits_its_own_file_type(tmp_path):
    """`recall index --glob '**/*.py'` is a supported workflow — the guard must not block it."""
    module = tmp_path / "mod.py"
    module.write_text("x = 1\n")

    assert candidate_files(module, glob="**/*.py") == [module]


def test_directory_walk_is_unchanged(tmp_path):
    """The walk already filtered correctly; this pins that it still does."""
    (tmp_path / "notes.md").write_text("# a memo\n")
    (tmp_path / ".env").write_text("VOYAGE_API_KEY=sk-live-EXAMPLE\n")

    assert [p.name for p in candidate_files(tmp_path)] == ["notes.md"]


def test_mcp_index_refuses_a_non_corpus_file_before_spending_anything(tmp_path, monkeypatch):
    """At the client-facing boundary: refused, and nothing embedded or billed.

    `store` and `embedder` are None deliberately — reaching either would raise AttributeError
    instead of ValueError, so this also proves the refusal happens BEFORE any work.
    """
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    (tmp_path / "tokens.json").write_text('{"principals": [{"token": "PLAINTEXT-SECRET"}]}')

    debited: list[tuple[int, int]] = []

    with pytest.raises(ValueError):
        index_memory(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            str(tmp_path / "tokens.json"),
            on_measured=lambda files, total: debited.append((files, total)),
        )

    assert debited == [], "a refused request must not debit the tenant's byte quota"
