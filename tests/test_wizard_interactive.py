"""The questions `recall wizard` asks, driven through the injected prompter rather than a TTY.

The point of the seam is that this file can exist. A prompt flow that reads the real stdin is a
prompt flow nobody tests, and an installer whose seventh question is wrong is wrong forever.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.wizard.database import DatabaseReport, Finding
from recall.wizard.interactive import (
    InteractiveRefusal,
    Prompter,
    ask_config,
    write_config,
)


def _scripted(answers: list[str]) -> tuple[Prompter, list[str]]:
    """A prompter that reads from a list and records everything shown."""
    transcript: list[str] = []
    remaining = list(answers)

    def read(prompt: str) -> str:
        transcript.append(prompt)
        if not remaining:
            raise AssertionError(f"the flow asked more than it was given: {prompt!r}")
        return remaining.pop(0)

    return Prompter(read=read, write=transcript.append), transcript


def _usable(dsn: str, *, expected_dimension: int | None = None) -> DatabaseReport:
    return DatabaseReport(
        dsn=dsn, findings=(Finding(name="reachable", ok=True, detail="connected"),)
    )


def _unusable(dsn: str, *, expected_dimension: int | None = None) -> DatabaseReport:
    return DatabaseReport(
        dsn=dsn,
        findings=(
            Finding(
                name="pgvector",
                ok=False,
                detail="not available",
                blocking=True,
                advice="CREATE EXTENSION vector",
            ),
        ),
    )


# ----------------------------------------------------------------------------------------------
# The two shapes of install, which are the whole reason this surface exists
# ----------------------------------------------------------------------------------------------


def test_the_docker_path_asks_for_a_data_root_and_never_for_a_dsn(tmp_path: Path) -> None:
    prompter, transcript = _scripted(
        ["myproject", "fastembed", "docker", str(tmp_path / "store"), "", "", "", ""]
    )

    config = ask_config(prompter, default_root=tmp_path / "default")

    assert config["data_root"] == str(tmp_path / "store")
    assert "dsn" not in config, "the Docker path must not also pin a connection string"
    assert config["project"] == "myproject"
    assert config["embedder"] == "fastembed"
    assert not any("onnection string" in line for line in transcript), (
        "asking for a DSN on the Docker path is the question this whole choice removes"
    )


def test_the_existing_database_path_asks_for_a_dsn_and_never_for_a_data_root(
    tmp_path: Path,
) -> None:
    """⛔ Case A: a user who already has PostgreSQL must not be made to install Docker.

    The engine has always supported this — `provision_stack` returns immediately when `data_root`
    is absent — but nothing asked, so the only way to express it was to hand-write JSON.
    """
    dsn = "postgresql://me:pw@127.0.0.1:5432/mine"
    prompter, _ = _scripted(["myproject", "hashing", "existing", dsn, "", "", "", "", ""])

    config = ask_config(prompter, default_root=tmp_path, probe=_usable)

    assert config["dsn"] == dsn
    assert "data_root" not in config, (
        "setting both is refused by the engine, and offering both here is how that happens"
    )


def test_the_existing_database_path_can_configure_an_isolated_fact_controller(
    tmp_path: Path,
) -> None:
    serving = "postgresql://me:pw@127.0.0.1:5432/mine"
    controller = "postgresql://fact_writer:factpw@127.0.0.1:5432/mine"
    prompter, transcript = _scripted(
        ["myproject", "hashing", "existing", serving, controller, "", "", "", ""]
    )

    config = ask_config(prompter, default_root=tmp_path, probe=_usable)

    assert config["fact_write_dsn"] == controller
    assert any("isolated fact-controller" in line for line in transcript)


def test_a_remote_dsn_is_accepted_when_it_probes_clean(tmp_path: Path) -> None:
    """Case B, the half that is code rather than documentation: a reachable remote database."""
    dsn = "postgresql://me:realpassword@db.example.invalid:5432/recall"
    prompter, _ = _scripted(["myproject", "hashing", "existing", dsn, "", "", "", "", ""])

    config = ask_config(prompter, default_root=tmp_path, probe=_usable)

    assert config["dsn"] == dsn


# ----------------------------------------------------------------------------------------------
# Refusing to accept a database on faith
# ----------------------------------------------------------------------------------------------


def test_an_unusable_database_is_re_asked_rather_than_accepted(tmp_path: Path) -> None:
    """⚠️ Taking a DSN on faith is what makes the existing-database path a trap.

    Every way it can be wrong fails minutes later, during a build, naming something other than the
    connection string. So the probe runs BEFORE the answer is accepted.
    """
    good = "postgresql://me:pw@127.0.0.1:5432/good"
    seen: list[str] = []

    def probe(dsn: str, *, expected_dimension: int | None = None) -> DatabaseReport:
        seen.append(dsn)
        return _usable(dsn) if dsn == good else _unusable(dsn)

    prompter, transcript = _scripted(
        [
            "p",
            "hashing",
            "existing",
            "postgresql://me:pw@127.0.0.1:5432/bad",
            good,
            "",
            "",
            "",
            "",
            "",
        ]
    )

    config = ask_config(prompter, default_root=tmp_path, probe=probe)

    assert seen == ["postgresql://me:pw@127.0.0.1:5432/bad", good], "both must be probed"
    assert config["dsn"] == good
    assert any("CREATE EXTENSION vector" in line for line in transcript), (
        "the advice must reach the user, or they cannot fix what was rejected"
    )


def test_giving_up_on_the_database_refuses_rather_than_installing_something_else(
    tmp_path: Path,
) -> None:
    """A bounded loop, because an unbounded re-ask against an EOF stdin is a hang.

    And it must NOT silently fall back to Docker: quietly installing a different thing than the
    user asked for is worse than stopping.
    """
    prompter, _ = _scripted(["p", "hashing", "existing", *["postgresql://x/y"] * 5])

    with pytest.raises(InteractiveRefusal) as caught:
        ask_config(prompter, default_root=tmp_path, probe=_unusable)

    assert "nothing was installed" in str(caught.value)


def test_an_unrecognised_choice_is_re_asked_not_defaulted(tmp_path: Path) -> None:
    """⛔ Silently taking the default when somebody typed something is the worst branch here.

    They asked for one thing, got another, and were never told. The default applies to an EMPTY
    answer only.
    """
    prompter, transcript = _scripted(["p", "fastembed", "postgres", "docker", str(tmp_path), "", "", "", ""])

    config = ask_config(prompter, default_root=tmp_path)

    assert config["data_root"] == str(tmp_path)
    assert any("not one of" in line for line in transcript), (
        "the user must be told their answer was not understood"
    )


def test_an_empty_answer_takes_the_default_and_the_default_is_always_shown(tmp_path: Path) -> None:
    prompter, transcript = _scripted(["", "", "", "", "", "", "", ""])

    config = ask_config(prompter, default_root=tmp_path / "shown")

    assert config["data_root"] == str(tmp_path / "shown")
    assert config["embedder"] == "fastembed"
    assert any(str(tmp_path / "shown") in line for line in transcript), (
        "a default that is applied but not shown is a choice the user did not know they made"
    )


# ----------------------------------------------------------------------------------------------
# The corpora and the wiring, which are optional
# ----------------------------------------------------------------------------------------------


def test_every_root_is_written_because_the_engine_requires_all_three(tmp_path: Path) -> None:
    """⛔ The bug a green question-flow suite could not see.

    An earlier version offered "leave blank to skip", which produced a config `load_config`
    REFUSES: `docs_root`, `code_root` and `memory_root` are mandatory. So an interactive install
    where the user skipped one failed at the next step with a message about missing JSON keys —
    an installer handing the user a validation error about its own artefact.

    Found by round-tripping the written file, not by the ten tests of the questions, every one of
    which passed. The questions were right and the artefact was not.
    """
    docs = tmp_path / "mydocs"
    docs.mkdir()
    prompter, _ = _scripted(["p", "hashing", "docker", str(tmp_path), str(docs), "", "", ""])

    config = ask_config(prompter, default_root=tmp_path)

    assert config["docs_root"] == str(docs.resolve()), "an explicit answer is honoured"
    # The blanks take a suggested directory under the install rather than vanishing. An empty or
    # absent directory is a state the pipeline already reports as normal on a first install, so
    # "I have nothing for this yet" stays expressible without an invalid config.
    assert config["code_root"] == str((tmp_path / "code").resolve())
    assert config["memory_root"] == str((tmp_path / "memory").resolve())
    assert "project_root" not in config, "blank wiring must skip wiring, not wire the cwd"


@pytest.mark.parametrize(
    "answers, label",
    [
        (["p", "hashing", "docker", "{tmp}", "", "", "", ""], "docker, every root defaulted"),
        (
            ["p", "hashing", "docker", "{tmp}", "{tmp}/d", "{tmp}/c", "{tmp}/m", ""],
            "docker, every root given",
        ),
    ],
)
def test_what_the_flow_writes_is_what_the_headless_installer_accepts(
    tmp_path: Path, answers: list[str], label: str
) -> None:
    """⚠️ The assertion that the artefact is real, rather than that the questions were asked.

    The whole design is that the interactive path writes a file and the headless path runs it. If
    `load_config` rejects what `ask_config` produces, the flow is decorative — and that is exactly
    what happened until this test existed.
    """
    from recall.wizard.headless import load_config

    filled = [a.replace("{tmp}", str(tmp_path)) for a in answers]
    prompter, _ = _scripted(filled)

    document = ask_config(prompter, default_root=tmp_path)
    written = write_config(document, tmp_path / "wizard.json")
    config = load_config(written)

    assert config.project == "p", label
    assert config.embedder == "hashing"
    assert config.data_root is not None
    for root in (config.docs_root, config.code_root, config.memory_root):
        assert root.is_absolute(), f"{root} is relative, and would resolve against the cwd"


# ----------------------------------------------------------------------------------------------
# The artefact
# ----------------------------------------------------------------------------------------------


def test_the_answers_are_written_as_a_config_the_headless_path_can_run(tmp_path: Path) -> None:
    """The interactive flow produces the same artefact the unattended one consumes.

    That is what keeps this from being a second installer: one engine, and a file the user can
    re-run, hand to somebody, or commit.
    """
    target = tmp_path / "nested" / "wizard.json"

    written = write_config({"project": "p", "embedder": "hashing"}, target)

    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "project": "p",
        "embedder": "hashing",
    }
    assert b"\r\n" not in target.read_bytes(), "CRLF rewrites every line on every platform"


def test_a_non_interactive_session_is_refused_before_the_first_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⛔ Without this, `recall wizard` in a script hangs or takes every default silently.

    Both look like a successful install from the outside, which is the failure this project keeps
    paying for. A person is either present to answer, or they should be using `--headless`.
    """
    import sys

    from recall.wizard.interactive import stdin_prompter

    class _NotATty:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", _NotATty())

    with pytest.raises(InteractiveRefusal) as caught:
        stdin_prompter()

    message = str(caught.value)
    assert "--headless" in message, "the refusal must name the way forward"
    assert "terminal" in message
