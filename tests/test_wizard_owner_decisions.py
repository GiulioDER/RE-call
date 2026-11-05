"""Regression tests for the owner's 2026-11-05 decisions on the first-run CCA hunt's residue.

CODE-001, DEPLOY-002, DEPLOY-004, ENV-002, PERF-001 and PERF-003, each decided A. Red proof for
every test was taken on the remote test host against the pre-fix snapshot of the production files
(base 7227f8405) by `python -m cca_checks tautology verify`; the verdicts are recorded in the pull
request. A test needing a symbol the fix added imports it after its first assertion, so a pre-fix
run fails on behaviour rather than on the import.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


# ----------------------------------------------------------------------------------------------
# CODE-001: one list of OS variables for every installer
# ----------------------------------------------------------------------------------------------


def test_lite_setup_registers_the_wizard_list_of_os_variables_and_no_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CODE-001. Invariant: `recall setup --lite` registers exactly the OS variables
    `recall.wizard.wiring.SERVER_OS_ENV` names. Failure caught: lite kept its own copy of the list
    and added it to the block too, so a variable dropped from (or added to) the wizard's list stayed
    (or was missing) in every lite registration. Target: `recall.lite.setup.run_lite_setup`."""
    from recall.lite.setup import SERVER_NAME, run_lite_setup
    from recall.wizard import wiring
    from tests.test_lite_calibration import ContentWordEmbedder, _write_memos

    monkeypatch.setattr(wiring, "SERVER_OS_ENV", ("APPDATA",))
    monkeypatch.setenv("APPDATA", "C:/fake/appdata")
    monkeypatch.setenv("TEMP", "C:/fake/temp")
    project = tmp_path / "project"
    project.mkdir()
    _write_memos(project / "notes", 48)
    config = tmp_path / ".claude.json"
    config.write_text(json.dumps({"projects": {}}), encoding="utf-8")

    run_lite_setup(
        project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(),  # type: ignore[arg-type]
        embedder_name="content-words", config_path=config,
    )

    servers = {
        name: server
        for entry in json.loads(config.read_text(encoding="utf-8"))["projects"].values()
        for name, server in entry.get("mcpServers", {}).items()
    }
    env = servers[SERVER_NAME]["env"]
    assert env["APPDATA"] == "C:/fake/appdata"
    assert "TEMP" not in env, "lite wrote a variable the wizard's list no longer names"


# ----------------------------------------------------------------------------------------------
# DEPLOY-002, DEPLOY-004: the install preflight
# ----------------------------------------------------------------------------------------------


class _FakeRun:
    """Stands in for `recall.wizard.probe._run`, answering by the command it is asked."""

    def __init__(self, answers: dict[str, tuple[int, str]]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, command: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
        key = " ".join(command[:3])
        self.asked.append(key)
        for prefix, (code, out) in self.answers.items():
            if key.startswith(prefix):
                return subprocess.CompletedProcess(command, code, out, "")
        return None


def _wizard_config(tmp_path: Path, **overrides: Any) -> Path:
    for leaf in ("docs", "code", "memory"):
        (tmp_path / leaf).mkdir(exist_ok=True)
    payload: dict[str, Any] = {
        "embedder": "hashing",
        "corpus_version": "2026-11-05",
        "data_root": str(tmp_path / "data"),
        "docs_root": str(tmp_path / "docs"),
        "code_root": str(tmp_path / "code"),
        "memory_root": str(tmp_path / "memory"),
    }
    payload.update(overrides)
    path = tmp_path / "wizard.json"
    path.write_text(json.dumps({k: v for k, v in payload.items() if v is not None}), encoding="utf-8")
    return path


def _run_wizard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: dict[str, tuple[int, str]],
    *, store_python: bool = False, docker_on_path: bool = True, **config: Any,
) -> tuple[list[str], str | None]:
    """`recall wizard --headless` with the probes faked; returns (installs started, exit message)."""
    from recall import cli
    from recall.wizard import headless, probe

    monkeypatch.setattr(probe, "_run", _FakeRun(answers))
    monkeypatch.setattr(probe.shutil, "which", lambda name: "/usr/bin/docker" if docker_on_path else None)
    monkeypatch.setattr(probe, "probe_store_python", lambda: store_python)
    started: list[str] = []
    monkeypatch.setattr(headless, "run_headless", lambda *a, **k: started.append("run") or _Ok())
    path = _wizard_config(tmp_path, **config)
    try:
        cli.main(["wizard", "--headless", "--config", str(path), "--no-state"])
    except SystemExit as exc:
        return started, None if exc.code in (0, None) else str(exc.code)
    return started, None


class _Ok:
    ok = True

    def render(self) -> str:
        return "install complete"


def test_the_wizard_refuses_a_missing_compose_plugin_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-002. Invariant: with a Docker data folder, a missing Docker Compose v2 plugin stops
    `recall wizard` before the install starts, naming the plugin. Failure caught: the stack file was
    written and `docker compose up` then failed under advice to start Docker Desktop.
    Target: the preflight in `setup_wizard._cmd_wizard`."""
    started, message = _run_wizard(
        tmp_path, monkeypatch,
        {"docker version --format": (0, "27.3.1"), "docker compose version": (1, "")},
    )
    assert started == [], "the install started although Compose is missing"
    assert message is not None and "Compose v2 plugin" in message, message


def test_the_wizard_refuses_the_microsoft_store_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-004. Invariant: the Microsoft Store Python is refused before the install starts.
    Failure caught: `blockers()` had no caller, so the Store build indexed nothing and reported
    success. Target: the preflight in `setup_wizard._cmd_wizard`."""
    started, message = _run_wizard(
        tmp_path, monkeypatch,
        {"docker version --format": (0, "27.3.1"), "docker compose version": (0, "2.29.7")},
        store_python=True,
    )
    assert started == [], "the install started on a Store Python"
    assert message is not None and "Microsoft Store" in message, message


def test_an_existing_database_install_needs_no_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-004 control. Invariant: with `dsn` instead of `data_root`, no Docker condition is
    checked, so a machine without Docker still installs against its own database."""
    started, message = _run_wizard(
        tmp_path, monkeypatch, {}, docker_on_path=False,
        data_root=None, dsn="postgresql://recall:recall@127.0.0.1:5432/recall",
    )
    assert message is None, message
    assert started == ["run"]


def test_a_busy_docker_daemon_is_asked_with_docker_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEPLOY-004. Invariant: the daemon probe uses `docker version`, which answers in about half a
    second, not `docker info` (measured median 14s, up to 56s, beyond the probe's bound). Failure
    caught: a healthy busy daemon read as not running. Target: `probe.probe_docker`."""
    from recall.wizard import probe

    fake = _FakeRun({"docker version --format": (0, "27.3.1")})
    monkeypatch.setattr(probe, "_run", fake)
    monkeypatch.setattr(probe.shutil, "which", lambda name: "/usr/bin/docker")
    assert probe.probe_docker() == (True, True), fake.asked


def test_a_machine_sold_as_4_gb_is_not_refused_for_reporting_less(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEPLOY-004. Invariant: the 4 GB floor allows the same reporting slack as the 8 GB tier.
    Failure caught: a 4 GB machine reporting 3.9 GiB was refused. Target: `probe.blockers`."""
    from recall.setup import HardwareProbe
    from recall.wizard import probe

    reported = int(3.9 * 1024**3)
    system = probe.SystemProbe(
        hardware=HardwareProbe(
            cpu_count=4, gpu=None, cuda_available=False, free_bytes=0, internet=False,
            fastembed_available=False, sentence_transformers_available=False,
            total_ram_bytes=reported,
        ),
        total_ram_bytes=reported, available_ram_bytes=reported, docker_installed=True,
        docker_running=False, wsl2_available=None, virtualization_enabled=None,
        cuda_vram_bytes=None, python_executable="python", python_is_store_build=False,
    )
    assert not [b for b in probe.blockers(system) if "memory" in b]


# ----------------------------------------------------------------------------------------------
# ENV-002: the embedding profile that built the corpora reaches the servers
# ----------------------------------------------------------------------------------------------


def test_a_docker_install_refuses_an_embedding_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENV-002. Invariant: with `data_root`, a set RECALL_EMBED_PROFILE is refused before the stack
    is written or started. Failure caught: the corpora were built under the profile while the
    Docker services resolved the plain embedder. Target: `headless._provision`."""
    from recall.wizard import headless
    from tests.test_wizard_state import _CountingSpy, _config

    monkeypatch.setenv("RECALL_EMBED_PROFILE", "bge-small-context-section-v1")
    started: list[object] = []
    monkeypatch.setattr(headless, "bring_up", lambda *a, **k: started.append(a))
    monkeypatch.setattr(headless, "wait_for_database", lambda *a, **k: None)
    config = _config(tmp_path, dsn=None, migration_dsn=None, data_root=str(tmp_path / "data"))

    with pytest.raises(headless.PipelineRefusal, match="RECALL_EMBED_PROFILE"):
        headless.run_headless(config, services=_CountingSpy(), profile_path=tmp_path / "rt.json")
    assert started == [], "the stack was started under a profile it cannot reproduce"


def test_registered_servers_carry_the_embedding_profile_settings(tmp_path: Path) -> None:
    """ENV-002. Invariant: every registered server block carries what the services say reproduces
    the build's embedder. Failure caught: blocks carried only RECALL_EMBEDDER, so a client without
    the profile in its environment resolved another model. Target: `run_headless`, `server_blocks`."""
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    class _Profiled(_CountingSpy):
        def server_embedding_env(self) -> dict[str, str]:
            return {"RECALL_EMBED_PROFILE": "a-profile", "RECALL_MODEL_SHA256": "abc"}

    root = tmp_path / "project"
    root.mkdir()
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    report = run_headless(
        _config(tmp_path, project_root=str(root)), services=_Profiled(), claude_config_path=client
    )
    assert report.servers, "nothing was registered"
    for block in report.servers:
        assert block.env.get("RECALL_EMBED_PROFILE") == "a-profile", block.name
        assert block.env.get("RECALL_MODEL_SHA256") == "abc", block.name


def test_a_profile_is_part_of_what_a_finished_corpus_was_built_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENV-002. Invariant: the resume digest changes with RECALL_EMBED_PROFILE and is unchanged
    without it. Failure caught: a rerun under another profile reused corpora built under the old
    one. Target: `state.config_digest`."""
    from recall.wizard.state import config_digest
    from tests.test_wizard_state import _config

    config = _config(tmp_path)
    monkeypatch.delenv("RECALL_EMBED_PROFILE", raising=False)
    plain = config_digest(config)
    monkeypatch.setenv("RECALL_EMBED_PROFILE", "another-profile")
    assert config_digest(config) != plain
    monkeypatch.setenv("RECALL_EMBED_PROFILE", "")
    assert config_digest(config) == plain


# ----------------------------------------------------------------------------------------------
# PERF-001: explicit-glob walks prune the tool directories
# ----------------------------------------------------------------------------------------------


def test_an_explicit_glob_walk_skips_tool_directories_but_not_hidden_ones(tmp_path: Path) -> None:
    """PERF-001. Invariant: `candidate_files(root, "**/*.py")` (the wizard's code corpus glob) skips
    `.venv`, `node_modules`, `.git` and `__pycache__`, and still includes a file in a plain hidden
    directory, as before. Failure caught: every vendored file under a repository root was read,
    hashed, chunked and embedded. Target: `recall.index._directory_candidates`."""
    from recall.index import candidate_files

    for relative in (
        "app.py", "pkg/mod.py", ".venv/lib/site.py", "node_modules/x/y.py", ".git/hooks/h.py",
        "pkg/__pycache__/c.py", ".config/kept.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")

    found = {p.relative_to(tmp_path.resolve()).as_posix() for p in candidate_files(tmp_path, "**/*.py")}
    assert found == {"app.py", "pkg/mod.py", ".config/kept.py"}, found


# ----------------------------------------------------------------------------------------------
# PERF-003: setup --lite does not default to the whole home folder
# ----------------------------------------------------------------------------------------------


def _lite_setup_in(cwd: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[object], str | None]:
    from recall import cli
    import recall.lite.setup as lite_setup

    monkeypatch.chdir(cwd)
    # `tests/conftest.py` patches `Path.home` itself, so the home folder is set there too.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("RECALL_INDEX_ROOT", raising=False)
    calls: list[object] = []
    monkeypatch.setattr(lite_setup, "run_lite_setup", lambda **kw: calls.append(kw) or _LiteReport())
    try:
        cli.main(["--embedder", "hashing", "setup", "--lite", "--no-register"])
    except SystemExit as exc:
        return calls, None if exc.code in (0, None) else str(exc.code)
    return calls, None


class _LiteReport:
    registration = None
    followup_env: dict[str, str] = {}

    def render(self) -> str:
        return "set up"


def test_lite_setup_refuses_to_default_to_the_home_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-003. Invariant: with neither --memory nor RECALL_INDEX_ROOT, `recall setup --lite` run
    in the home folder refuses before indexing. Failure caught: a fresh Windows terminal opens in the
    home folder, and setup embedded the whole profile with no budget. Target: `_setup_lite`."""
    home = tmp_path / "home"
    home.mkdir()
    calls, message = _lite_setup_in(home, home, monkeypatch)
    assert calls == [], "setup indexed the home folder"
    assert message is not None and "--memory" in message, message


def test_lite_setup_warns_when_the_defaulted_folder_is_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PERF-003. Invariant: a defaulted folder over the index budget is still set up (a person asked)
    but with a warning naming the limit. Failure caught: setup gave no sign it was indexing more
    than the server itself would ever refresh. Target: `_setup_lite`."""
    project = tmp_path / "project"
    project.mkdir()
    for i in range(3):
        (project / f"memo{i}.md").write_text(f"# memo {i}\n\nbody\n", encoding="utf-8")
    monkeypatch.setenv("RECALL_INDEX_MAX_FILES", "1")
    calls, message = _lite_setup_in(project, tmp_path / "elsewhere", monkeypatch)
    assert message is None, message
    assert len(calls) == 1
    assert "warning:" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------
# Guards on the new code itself (found by the regression review, 2026-11-05). These are not
# pre-fix regressions: their red proof is a mutation of the fix, recorded in the pull request.
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "version"),
    [
        ("2.24.6+ds1-0ubuntu1~22.04.1", (2, 24, 6)),
        ("v2.29.7", (2, 29, 7)),
        ("2.29.7-desktop.1", (2, 29, 7)),
        ("2.1.0", (2, 1, 0)),
    ],
)
def test_distro_and_desktop_compose_versions_are_read(
    monkeypatch: pytest.MonkeyPatch, answer: str, version: tuple[int, int, int]
) -> None:
    """Invariant: the Compose version is read however it is spelled, including Ubuntu's
    `docker-compose-v2` package, so the 2.3.3 floor compares a real version. Mutation proof: the
    first parser (split on `-`, digit-only parts) read the Ubuntu string as unknown."""
    from recall.wizard import probe

    monkeypatch.setattr(probe, "_run", _FakeRun({"docker compose version": (0, answer)}))
    assert probe.probe_compose() == (False, version)


def test_an_unreadable_compose_answer_is_not_a_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: a Compose probe that times out is unknown, and unknown never refuses an install,
    as `blockers` documents. Mutation proof: treating an unknown version as missing refused it."""
    started, message = _run_wizard(
        tmp_path, monkeypatch, {"docker version --format": (0, "27.3.1")},  # compose: no answer
    )
    assert message is None, message
    assert started == ["run"]


def test_the_install_report_never_prints_a_profile_api_key(tmp_path: Path) -> None:
    """Invariant: a hosted profile's API key rides in the server blocks but the printed report shows
    it only as set. Mutation proof: rendering every block variable by value printed the key."""
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    class _Hosted(_CountingSpy):
        def server_embedding_env(self) -> dict[str, str]:
            return {"RECALL_EMBED_PROFILE": "hosted", "VOYAGE_API_KEY": "fake-key-not-to-print"}

    root = tmp_path / "project"
    root.mkdir()
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    report = run_headless(
        _config(tmp_path, project_root=str(root)), services=_Hosted(), claude_config_path=client
    )
    assert any(b.env.get("VOYAGE_API_KEY") for b in report.servers), "the key must still be registered"
    rendered = report.render()
    assert "fake-key-not-to-print" not in rendered
    assert "VOYAGE_API_KEY=<set>" in rendered
