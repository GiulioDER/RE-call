import os

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
