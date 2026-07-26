"""A rejection must not tell the client where the server keeps its files.

`index_memory`'s confinement check is the one error an unauthorised path probe reliably triggers,
and it echoed the RESOLVED absolute index root back to the caller. Walked over a handful of
guesses that is a free map of the server's filesystem — the deployment directory, the account
name in a home path, the layout of the container — handed to whoever is probing for the thing
`RECALL_INDEX_ROOT` exists to keep them away from.

The caller's OWN path is still echoed: they sent it, so it discloses nothing they did not
already know, and without it the message cannot say which request was refused.
"""
from __future__ import annotations

import pytest

from recall_mcp.service import index_memory


class _ExplodingStore:
    def __getattr__(self, name):
        raise AssertionError(f"store.{name} was reached; the request should have been refused")


def test_a_confinement_refusal_does_not_echo_the_server_root(tmp_path, monkeypatch):
    root = tmp_path / "server-side-corpus-directory"
    root.mkdir()
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    outside = tmp_path / "elsewhere.md"
    outside.write_text("# not yours\n")

    with pytest.raises(ValueError) as exc:
        index_memory(_ExplodingStore(), None, str(outside))  # type: ignore[arg-type]

    message = str(exc.value)
    assert "server-side-corpus-directory" not in message, "the server's root leaked to the client"
    assert str(root) not in message
    assert "RECALL_INDEX_ROOT" in message, "the operator still needs to know which knob to turn"


def test_the_callers_own_path_is_still_named(tmp_path, monkeypatch):
    """Withholding the root must not make the refusal unactionable."""
    root = tmp_path / "corpus"
    root.mkdir()
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    outside = tmp_path / "elsewhere.md"
    outside.write_text("# not yours\n")

    with pytest.raises(ValueError, match="elsewhere.md"):
        index_memory(_ExplodingStore(), None, str(outside))  # type: ignore[arg-type]
