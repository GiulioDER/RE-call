import os

import pytest

from recall._env import load_dotenv


def test_load_dotenv_parses_and_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('FOO=bar\n# a comment\nBAZ="q u x"\nEXISTING=fromfile\n', encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    monkeypatch.setenv("EXISTING", "keep")

    load_dotenv(env)

    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "q u x"
    assert os.environ["EXISTING"] == "keep"  # already-set var is not overridden


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "does_not_exist.env")  # must not raise


def test_load_dotenv_repeated_key_resolves_to_first_occurrence(tmp_path, monkeypatch):
    # A rewrite once collected lines into a dict and silently flipped this to last-wins,
    # with nothing here to catch it. `recall/setup.py` appends a `# recall setup` block to
    # the END of an existing .env, so a hand-written line above the block must keep winning.
    env = tmp_path / ".env"
    env.write_text("K=first\nK=second\n", encoding="utf-8")
    monkeypatch.delenv("K", raising=False)

    load_dotenv(env)

    assert os.environ["K"] == "first"


def test_load_dotenv_bad_byte_on_an_already_set_key_does_not_poison_the_rest(tmp_path, monkeypatch):
    # A key that is already exported was never going to be applied, so a NUL on ITS line must
    # not fail keys that would have been applied. Checking validity before checking whether a
    # line matters previously discarded an otherwise-clean file over a byte nothing would read.
    env = tmp_path / ".env"
    env.write_text("ALREADY=x\x00y\nLATER=must_arrive\n", encoding="utf-8")
    monkeypatch.setenv("ALREADY", "pre-set-by-shell")
    monkeypatch.delenv("LATER", raising=False)

    load_dotenv(env)

    assert os.environ["ALREADY"] == "pre-set-by-shell"
    assert os.environ["LATER"] == "must_arrive"


def test_load_dotenv_bad_byte_on_a_key_that_would_apply_discards_the_whole_file(
    tmp_path, monkeypatch
):
    # All-or-nothing for the keys that actually matter: a NUL on a line that WOULD have been
    # applied must raise, and nothing from the file — including earlier valid lines — applies.
    env = tmp_path / ".env"
    env.write_text("EARLY=would_apply\nBROKEN=a\x00b\n", encoding="utf-8")
    monkeypatch.delenv("EARLY", raising=False)
    monkeypatch.delenv("BROKEN", raising=False)

    with pytest.raises(ValueError, match="embedded null character"):
        load_dotenv(env)

    assert "EARLY" not in os.environ  # partial application would be worse than none


def test_the_loader_undoes_exactly_what_the_writer_escaped(tmp_path, monkeypatch):
    """A value written by `_quote_env` must come back identical, escapes and all.

    The loader used to strip the surrounding quotes and stop. That leaves the backslashes in
    place, and `.strip('"')` then removes every trailing quote it finds, including the one an
    escape belonged to, so a value containing a quote came back both mangled and a character
    short. The reasoning arm added the first free text prompts feeding this path, a base URL
    and a model id, so it is now reachable by typing rather than only in theory.
    """
    written = {
        "Q_QUOTE": 'my model "quoted"',
        "Q_BACKSLASH": "C:\\models\\bge",
        "Q_NEWLINE": "first\nsecond",
        "Q_HASH": "has # hash",
        "Q_SPACE": "with space",
        "Q_PLAIN": "plain-value",
        "Q_EMPTY": "",
    }
    from recall.setup import _quote_env

    env = tmp_path / ".env"
    env.write_text(
        "\n".join(f"{k}={_quote_env(v)}" for k, v in written.items()) + "\n",
        encoding="utf-8",
    )
    for key in written:
        monkeypatch.delenv(key, raising=False)

    load_dotenv(env)

    for key, value in written.items():
        assert os.environ[key] == value, f"{key} did not survive the round trip"


def test_a_hand_written_unbalanced_quote_reads_as_it_always_did(tmp_path, monkeypatch):
    """Fixing the round trip must not change how an existing hand-written file is read.

    The writer only ever emits a matched pair, so unbalanced quoting can only come from somebody
    editing `.env` themselves. Those files worked before, and an upgrade that silently reinterprets
    them would break a working install for a reason the reader cannot see. Only a properly quoted
    value changed behaviour.
    """
    env = tmp_path / ".env"
    env.write_text(
        'OPEN_ONLY="postgresql://example/db\n'
        'CLOSE_ONLY=postgresql://example/db"\n'
        "BARE=postgresql://example/db\n",
        encoding="utf-8",
    )
    for key in ("OPEN_ONLY", "CLOSE_ONLY", "BARE"):
        monkeypatch.delenv(key, raising=False)

    load_dotenv(env)

    assert os.environ["OPEN_ONLY"] == "postgresql://example/db"
    assert os.environ["CLOSE_ONLY"] == "postgresql://example/db"
    assert os.environ["BARE"] == "postgresql://example/db"
