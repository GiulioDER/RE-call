"""`recall calibrate` is the install-time calibration a new user is handed. It must run.

Both tests here pin a defect that shipped, and the pair is deliberate: one made the command
impossible, the other made the failure impossible to diagnose. Either alone is survivable. Together
they meant a tester ran the command printed on the website and got exit code 2 with **nothing on
stdout and nothing on stderr**, which is indistinguishable from a hung terminal.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.cli import main
from recall.embeddings import resolve_embedder

# A DSN nothing listens on. Every test here monkeypatches `calibrate_from_files`, so no connection
# is ever made; the password is present only because the CLI refuses a passwordless DSN before it
# reaches the handler, and that refusal is not what these tests are about.
DSN = "postgresql://u:pw@127.0.0.1:1/recall"


def _queries(tmp_path: Path) -> Path:
    p = tmp_path / "queries.json"
    p.write_text(
        '[{"query": "a", "answerable": true}, {"query": "b", "answerable": false}]',
        encoding="utf-8",
    )
    return p


class TestTheEmbedderArgumentIsASpecNotAName:
    """What reaches `calibrate_from_files` must be something `resolve_embedder` accepts.

    The defect: `_cmd_calibrate` built the embedder and passed its `.name` onward. That round trip
    does not close. `fastembed` resolves to an embedder whose `.name` is
    `BAAI/bge-small-en-v1.5`, and a bare model name is not one of the spellings the resolver
    accepts, so the DEFAULT embedder raised `unknown embedder: 'BAAI/bge-small-en-v1.5'`.

    A name is what an embedder calls itself. A spec is how you ask for one. Only one of them is an
    input, and the test asserts the round trip rather than a literal string so it keeps holding if
    the accepted spellings change.
    """

    @pytest.mark.parametrize("spec", ["hashing", "fastembed"])
    def test_what_is_passed_can_be_resolved_back(self, tmp_path, monkeypatch, spec):
        seen: dict[str, str] = {}

        def fake(**kw):
            seen["embedder_name"] = kw["embedder_name"]
            raise ValueError("stop here: the argument is already captured")

        monkeypatch.setattr("recall.setup.calibrate_from_files", fake)
        with pytest.raises(SystemExit):
            main(["--dsn", DSN, "--embedder", spec, "calibrate", str(_queries(tmp_path))])

        # The assertion that matters: not "it equals the spec" but "the resolver accepts it".
        # `resolve_embedder` is the exact function `calibrate_from_files` calls on this value.
        resolve_embedder(seen["embedder_name"])

    def test_the_default_embedder_is_not_special_cased_into_failing(self, tmp_path, monkeypatch):
        """The default is the one a new user hits, and it was the one that was broken."""
        seen: dict[str, str] = {}

        def fake(**kw):
            seen["embedder_name"] = kw["embedder_name"]
            raise ValueError("stop")

        monkeypatch.setattr("recall.setup.calibrate_from_files", fake)
        monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
        with pytest.raises(SystemExit):
            main(["--dsn", DSN, "calibrate", str(_queries(tmp_path))])

        assert seen["embedder_name"] == "fastembed"
        resolve_embedder(seen["embedder_name"])


class TestAFailureSaysWhy:
    """`raise SystemExit(2) from exc` exits 2 and prints NOTHING.

    `from exc` sets `__cause__` for a traceback that is never rendered, because SystemExit does not
    print one. Every ValueError on this path is an operator-fixable sentence -- a query file with
    one class, an embedder spelling that does not exist -- so the branch that most needed the text
    was the branch that discarded it.
    """

    def test_the_message_reaches_the_operator(self, tmp_path, monkeypatch):
        def fake(**kw):
            raise ValueError("queries file needs at least one answerable AND one unanswerable")

        monkeypatch.setattr("recall.setup.calibrate_from_files", fake)
        with pytest.raises(SystemExit) as excinfo:
            main(["--dsn", DSN, "calibrate", str(_queries(tmp_path))])

        # `SystemExit.code` is the message when constructed with a string. Asserted as a substring
        # so the wording can improve without the test having an opinion about the prefix.
        assert "at least one answerable" in str(excinfo.value.code)

    def test_the_exit_is_not_a_bare_status_code(self, tmp_path, monkeypatch):
        """A regression here would look like a passing test above unless the shape is pinned too."""
        def fake(**kw):
            raise ValueError("boom")

        monkeypatch.setattr("recall.setup.calibrate_from_files", fake)
        with pytest.raises(SystemExit) as excinfo:
            main(["--dsn", DSN, "calibrate", str(_queries(tmp_path))])

        assert excinfo.value.code != 2, "exit 2 with no message is the defect this pins"
        assert isinstance(excinfo.value.code, str)
