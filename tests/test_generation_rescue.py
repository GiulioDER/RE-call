"""Atomic rescue on PostgreSQL: on at first install, and built before every promotion.

Owner decision 2026-11-15: the rescue is on when RE-call is first installed. Under an active rescue
a generation without its artifact refuses every unscoped search (`recall.trust`), so every path
that makes a generation serve must build first (`recall.generation_rescue`).

Red proofs (RP-n below), each a deliberate mutation of the production line named, run against the
test named, then reverted. The DB tests ran red in CI on DO NOT MERGE commits: RP-11, RP-13 and
RP-14 on 629586d87, RP-12 alone on da0392051 (with RP-11 beside it, RP-12 failed for RP-11's reason).
"""

from __future__ import annotations

import argparse
import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psycopg
import pytest

from recall.atomic_rescue import AtomicRescueArtifactError, write_atomic_rescue_artifact
from recall.lite.atomic import ROOT_ENV
from tests.conftest import TEST_DSN, requires_db
from tests.test_wizard_pipeline import (
    _corpus,
    _FakeCalibrations,
    _FakeEmbedder,
    _FakeManager,
    _Recorder,
)


# ------------------------------------------------------------------------------------- the wizard


def test_the_rescue_is_built_after_publish_and_before_promote(tmp_path: Path) -> None:
    """RP-1: `run_corpus` calling `rescue` after `manager.promote` made this fail on the order."""
    from recall.wizard.corpora import docs_corpus
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()

    def rescue(generation_id: str) -> str:
        recorder.note("rescue")
        return f"built for {generation_id}"

    outcome = run_corpus(
        docs_corpus(_corpus(tmp_path, files=45)),
        manager=_FakeManager(recorder),
        calibrations=_FakeCalibrations(recorder),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
        rescue=rescue,
    )

    assert recorder.calls == ["create", "build", "validate", "calibrate", "publish", "rescue", "promote"]
    assert outcome.atomic == "built for gen_test" and "rescue" in outcome.steps


def test_a_failed_rescue_build_refuses_the_promotion_and_reclaims(tmp_path: Path) -> None:
    """RP-2: the `except` around `rescue` swallowing the error (no re-raise) made this fail: the
    generation was promoted. Promoted without its artifact it refuses every unscoped search."""
    from recall.wizard.corpora import docs_corpus
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()
    manager = _FakeManager(recorder)

    def rescue(generation_id: str) -> str:
        raise AtomicRescueArtifactError("no room")

    with pytest.raises(AtomicRescueArtifactError):
        run_corpus(
            docs_corpus(_corpus(tmp_path, files=45)),
            manager=manager,
            calibrations=_FakeCalibrations(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
            rescue=rescue,
        )

    assert "promote" not in recorder.calls
    assert manager.abandoned and "rescue" in manager.abandoned[0]


def test_an_uncertified_generation_builds_no_rescue(tmp_path: Path) -> None:
    """Nothing serves it, and the builder refuses a generation without a certified calibration."""
    from recall.wizard.corpora import docs_corpus
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()

    def rescue(generation_id: str) -> str:
        recorder.note("rescue")
        return "built"

    outcome = run_corpus(
        docs_corpus(_corpus(tmp_path, files=45)),
        manager=_FakeManager(recorder),
        calibrations=_FakeCalibrations(recorder, certified=False),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
        rescue=rescue,
    )

    assert "rescue" not in recorder.calls and outcome.atomic == ""


def _config(tmp_path: Path, **overrides: object):
    from recall.wizard.headless import HeadlessConfig

    values: dict[str, object] = {
        "embedder": "hashing",
        "corpus_version": "2026-01-01",
        "docs_root": tmp_path / "docs",
        "code_root": tmp_path / "code",
        "memory_root": tmp_path / "memory",
        "dsn": "postgresql://recall@localhost/recall",
        "project_root": tmp_path / "project",
    }
    values.update(overrides)
    return HeadlessConfig(**values)  # type: ignore[arg-type]


def test_a_fresh_install_turns_the_rescue_on_with_an_absolute_root(tmp_path: Path, monkeypatch) -> None:
    """RP-3: `mode = "active"` replaced by `mode = "off"` in `_atomic_rescue_env` made this fail."""
    from recall.wizard.headless import _atomic_rescue_env

    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.delenv(ROOT_ENV, raising=False)

    env, note = _atomic_rescue_env(_config(tmp_path))
    assert note == ""
    assert env["RECALL_ATOMIC_RESCUE_MODE"] == "active"
    assert Path(env[ROOT_ENV]).is_absolute()
    assert Path(env[ROOT_ENV]) == (tmp_path / "project" / ".recall" / "atomic").resolve()



def test_a_docker_stack_install_turns_the_rescue_on_with_skip(tmp_path: Path, monkeypatch) -> None:
    """The desktop app's uploads run in containers that cannot reach a host root, so their promotions
    carry no artifact; the host servers answer without it and build it (`skip`,
    docs/DESKTOP_ATOMIC_RESCUE_DESIGN.md). This replaced the test that pinned the rescue OFF beside
    the Docker stack (owner decision 2026-11-17).

    Red proof: against the parent commit `4a073734e`, where the `data_root` branch kept it off:
    `assert {} == {...}`, nothing was registered.
    """
    from recall.wizard.headless import _atomic_rescue_env

    for name in ("RECALL_ATOMIC_RESCUE_MODE", ROOT_ENV, "RECALL_ATOMIC_RESCUE_WHEN_MISSING"):
        monkeypatch.delenv(name, raising=False)

    env, note = _atomic_rescue_env(_config(tmp_path, data_root=tmp_path / "data"))
    assert env == {
        "RECALL_ATOMIC_RESCUE_MODE": "active",
        ROOT_ENV: str((tmp_path / "project" / ".recall" / "atomic").resolve()),
        "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip",
    }
    assert note == ""


def test_a_host_install_keeps_refuse(tmp_path: Path, monkeypatch) -> None:
    """A control: only beside the Docker stack may a search answer without its artifact.

    Red proof: with `skip` added on every path, this fails: the setting is registered."""
    from recall.wizard.headless import _atomic_rescue_env

    for name in ("RECALL_ATOMIC_RESCUE_MODE", ROOT_ENV, "RECALL_ATOMIC_RESCUE_WHEN_MISSING"):
        monkeypatch.delenv(name, raising=False)

    env, _note = _atomic_rescue_env(_config(tmp_path))
    assert "RECALL_ATOMIC_RESCUE_WHEN_MISSING" not in env


def test_host_registration_overrides_an_inherited_rescue_skip(tmp_path: Path, monkeypatch) -> None:
    """ENV-001. With the setting omitted from the block, the launcher inherited skip."""
    from recall.wizard.headless import _atomic_rescue_env, _cli_rescue_env, _with_rescue
    from recall.wizard.pipeline import CorpusOutcome
    from recall.wizard.wiring import ServerBlock
    from recall_mcp.settings import bootstrap_settings

    policy = "RECALL_ATOMIC_RESCUE_WHEN_MISSING"
    monkeypatch.setenv(policy, "skip")
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "active")
    monkeypatch.delenv(ROOT_ENV, raising=False)
    env, _ = _atomic_rescue_env(_config(tmp_path))
    block = ServerBlock(name="p-docs", tenant="p-docs", env={}, rationale="r")
    outcomes = [CorpusOutcome(
        tenant="p-docs", generation_id="gen_docs", certified=True, promoted=True, atomic="built for gen_docs",
    )]
    _with_rescue((block,), outcomes, env)

    runtime = bootstrap_settings({policy: "skip", **block.env})
    assert runtime.env[policy] == "refuse", "the registered host inherited skip"
    assert runtime.env["RECALL_ATOMIC_RESCUE_MODE"] == "active"
    assert _cli_rescue_env((block,))[policy] == "refuse"


def test_a_shell_mode_off_still_wins_beside_the_docker_stack(tmp_path: Path, monkeypatch) -> None:
    """A control: a mode the shell sets is registered as it is, and `off` adds no `skip`."""
    from recall.wizard.headless import _atomic_rescue_env

    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_WHEN_MISSING", raising=False)
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "off")

    env, _note = _atomic_rescue_env(_config(tmp_path, data_root=tmp_path / "data"))
    assert env == {"RECALL_ATOMIC_RESCUE_MODE": "off"}


def test_under_skip_every_tenant_with_a_generation_is_active(tmp_path: Path) -> None:
    """With `skip`, a tenant whose artifact this run did not build registers active and is built on
    its first search; the memory tenant, which has no generation, stays off, since active with no
    generation identity raises before any artifact is looked up.

    Red proof: against the parent commit `4a073734e`, where `_with_rescue` pinned every tenant this
    run did not build for: `['active', 'off', 'off'] == ['active', 'active', 'off']`.
    """
    from recall.wizard.headless import _cli_rescue_env, _with_rescue
    from recall.wizard.pipeline import CorpusOutcome
    from recall.wizard.wiring import ServerBlock

    blocks = tuple(ServerBlock(name=t, tenant=t, env={}, rationale="r") for t in ("p-docs", "p-code", "p-memory"))
    outcomes = [
        CorpusOutcome(tenant="p-docs", generation_id="g1", certified=True, promoted=True, atomic="built for g1"),
        CorpusOutcome(tenant="p-code", generation_id="g2", certified=True, promoted=True),
    ]
    root = str(tmp_path / "atomic")
    skip = {"RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: root, "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip"}

    _with_rescue(blocks, outcomes, skip)

    assert [b.env["RECALL_ATOMIC_RESCUE_MODE"] for b in blocks] == ["active", "active", "off"]
    assert [b.env.get("RECALL_ATOMIC_RESCUE_WHEN_MISSING") for b in blocks] == ["skip", "skip", "skip"]
    assert _cli_rescue_env(blocks)["RECALL_ATOMIC_RESCUE_WHEN_MISSING"] == "skip"


def test_a_shell_mode_is_kept_and_a_mistyped_one_is_refused(tmp_path: Path, monkeypatch) -> None:
    from recall.wizard.headless import ConfigRefusal, _atomic_rescue_env

    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "off")
    assert _atomic_rescue_env(_config(tmp_path)) == ({"RECALL_ATOMIC_RESCUE_MODE": "off"}, "")

    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "activ")
    with pytest.raises(ConfigRefusal) as refused:
        _atomic_rescue_env(_config(tmp_path))
    assert refused.value.keys == ("RECALL_ATOMIC_RESCUE_MODE",)

    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "active")
    monkeypatch.setenv(ROOT_ENV, "relative/atomic")
    with pytest.raises(ConfigRefusal):
        _atomic_rescue_env(_config(tmp_path))


def test_without_the_atomic_dependencies_the_rescue_stays_off(tmp_path: Path, monkeypatch) -> None:
    """RP-4: dropping the `find_spec` check made this fail: active was registered without them."""
    from recall.wizard.headless import _atomic_rescue_env

    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *rest: None if name == "threadpoolctl" else real(name, *rest)
    )

    env, note = _atomic_rescue_env(_config(tmp_path))
    assert env == {} and "recall-rag[atomic]" in note


def test_only_a_tenant_this_run_built_for_is_served_active(tmp_path: Path) -> None:
    """RP-5: `_with_rescue` registering the mode on every block made this fail. The memory tenant
    has no generation and an earlier generation may have no artifact: active refuses both."""
    from recall.wizard.headless import _cli_rescue_env, _with_rescue
    from recall.wizard.pipeline import CorpusOutcome
    from recall.wizard.wiring import ServerBlock

    blocks = tuple(ServerBlock(name=t, tenant=t, env={}, rationale="r") for t in ("p-docs", "p-code", "p-memory"))
    outcomes = [
        CorpusOutcome(tenant="p-docs", generation_id="g1", certified=True, promoted=True, atomic="built for g1"),
        CorpusOutcome(tenant="p-code", generation_id="g2", certified=False, degraded_reason="short"),
    ]
    root = str(tmp_path / "atomic")

    _with_rescue(blocks, outcomes, {"RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: root})

    assert [b.env["RECALL_ATOMIC_RESCUE_MODE"] for b in blocks] == ["active", "off", "off"]
    assert all(b.env[ROOT_ENV] == root for b in blocks)
    assert _cli_rescue_env(blocks) == {
        # Pinned-off tenants are written too, so a shell mode cannot reach them (CCA ENV-003).
        "RECALL_ATOMIC_RESCUE_MODE_BY_TENANT": "p-docs:active,p-code:off,p-memory:off", ROOT_ENV: root,
        "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "refuse",
    }


def test_a_shell_shadow_mode_is_registered_as_it_is(tmp_path: Path) -> None:
    """RP-16: pinning every unbuilt tenant off whatever the mode made this fail: shadow became off."""
    from recall.wizard.headless import _with_rescue
    from recall.wizard.wiring import ServerBlock

    blocks = (ServerBlock(name="p-docs", tenant="p-docs", env={}, rationale="r"),)
    _with_rescue(blocks, [], {"RECALL_ATOMIC_RESCUE_MODE": "shadow"})
    assert blocks[0].env["RECALL_ATOMIC_RESCUE_MODE"] == "shadow"


def test_the_project_env_carries_the_rescue_and_a_rerun_replaces_it(tmp_path: Path) -> None:
    """RP-6: not popping the two keys from the managed block made this fail: a tenant turned off by
    a re-run stayed active in `.env`, so a promote there built while its server refused."""
    from recall.wizard.wiring import write_project_files

    project, memory = tmp_path / "project", tmp_path / "memory"
    memory.mkdir()
    rescue = {"RECALL_ATOMIC_RESCUE_MODE_BY_TENANT": "p-docs:active", ROOT_ENV: str(tmp_path / "atomic")}

    write_project_files(project_root=project, dsn="postgresql://x", embedder="hashing", memory_dir=memory, rescue_env=rescue)
    assert "RECALL_ATOMIC_RESCUE_MODE_BY_TENANT=p-docs:active" in (project / ".env").read_text(encoding="utf-8")

    write_project_files(project_root=project, dsn="postgresql://x", embedder="hashing", memory_dir=memory, rescue_env={})
    assert "RECALL_ATOMIC_RESCUE" not in (project / ".env").read_text(encoding="utf-8")


def test_a_docker_service_serves_with_the_rescue_off() -> None:
    """RP-7: removing the `"off"` pin made this fail: the container inherited active mode and a root
    it does not mount, so every search there would refuse."""
    from recall.wizard.stack import tenant_service

    environment = tenant_service(
        {"RECALL_DSN": "postgresql://x", "RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: "/host/atomic"},
        image="recall:test",
    )["environment"]
    assert environment["RECALL_ATOMIC_RESCUE_MODE"] == "off"  # type: ignore[index]
    assert ROOT_ENV not in environment  # type: ignore[operator]


# ------------------------------------------------------------------------------------------ the CLI


class _CliManager:
    calls: list[str] = []

    def __init__(self, dsn: str, tenant: str, **_: object) -> None:
        self.last_stated_facts = 0

    def promote(self, generation_id: str, *, unsafe_development: bool = False) -> None:
        _CliManager.calls.append(f"promote {generation_id}")

    def previous_generation_id(self) -> str:
        return "gen_previous"

    def rollback(self, **_: object) -> str:
        _CliManager.calls.append("rollback")
        return "gen_previous"

    def gc(self, **_: object) -> tuple[str, ...]:
        return ("gen_old",)

    def retract_collected_facts(self) -> int:
        return 0

    def collected_generation_ids(self) -> set[str]:
        return set()


def _args(command: str, **values: object) -> argparse.Namespace:
    values.setdefault("generation_cmd", command)
    return argparse.Namespace(dsn="postgresql://x", tenant="p-docs", embedder="hashing", **values)


@pytest.mark.parametrize(
    ("command", "values", "target"),
    [
        ("promote", {"generation_id": "gen_new", "unsafe_development_promotion": False}, "gen_new"),
        ("rollback", {}, "gen_previous"),
    ],
)
def test_the_cli_builds_before_it_serves_and_refuses_when_it_cannot(
    monkeypatch, command: str, values: dict[str, object], target: str
) -> None:
    """RP-8: deleting the `_rescue_before_serving` call in the promote branch (and, separately, the
    rollback branch) made the matching case fail: the generation went live without its artifact."""
    import recall.generation_rescue as rescue
    import recall.generations as generations
    from recall.cli_commands.generation_cmd import _cmd_generation

    monkeypatch.setattr(generations, "GenerationManager", _CliManager)
    _CliManager.calls = []
    built: list[str] = []

    def ensure(dsn, tenant, generation_id, embedder, **_):  # noqa: ANN001, ANN202
        built.append(generation_id)
        raise AtomicRescueArtifactError("the calibration is missing")

    monkeypatch.setattr(rescue, "ensure_for_promotion", ensure)
    with pytest.raises(SystemExit) as refused:
        _cmd_generation(_args(command, **values))

    assert built == [target]
    assert _CliManager.calls == []
    # Turned off in the CLI alone, the server would still refuse, so the advice names both.
    assert "RECALL_ATOMIC_RESCUE_MODE=off" in str(refused.value) and "MCP server" in str(refused.value)


def test_publishing_a_calibration_rebuilds_the_rescue(monkeypatch) -> None:
    """An artifact names its calibration, so a new one leaves every unscoped search refusing.

    RP-17: deleting the `_rescue_after_publish` call in the `auto` branch of
    `recall.cli_commands.calibration_cmd._cmd_calibration` made this fail.
    """
    import recall.calibration_v2 as calibration
    import recall.drift as drift
    import recall.generation_rescue as rescue
    from recall.cli_commands.calibration_cmd import _cmd_calibration

    monkeypatch.setattr(calibration, "CalibrationRepository", lambda dsn, tenant: object())
    monkeypatch.setattr(
        drift, "auto_recalibrate",
        lambda *args, **kwargs: SimpleNamespace(
            action="recalibrated", calibration_id="cal_new", published=True, reason="drift"
        ),
    )
    rebuilt: list[str] = []
    monkeypatch.setattr(rescue, "after_publish", lambda dsn, tenant, gid, embedder, **_: rebuilt.append(gid) or "")

    _cmd_calibration(_args("auto", calibration_cmd="auto", generation="gen_live", publish=True, json=False))

    assert rebuilt == ["gen_live"]


def test_a_rebuild_after_publish_that_fails_is_reported_not_raised(monkeypatch) -> None:
    """The publication has happened; the operator is told how to retry rather than handed a traceback
    that reads as a failed publish. RP-18: narrowing the `except` in `after_publish` to `KeyError` made this fail: the error escaped."""
    import recall.generation_rescue as rescue

    def ensure(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AtomicRescueArtifactError("the root is full")

    monkeypatch.setattr(rescue, "ensure_for_promotion", ensure)
    line = rescue.after_publish("postgresql://x", "p-docs", "gen_live", lambda: _FakeEmbedder())
    assert line.startswith("WARNING") and "recall --tenant p-docs generation rescue gen_live" in line


@pytest.mark.parametrize("policy, consequence", [
    ("skip", "continue without the rescue"),
    ("refuse", "searches refuse"),
])
def test_calibration_rebuild_warning_matches_the_missing_rescue_policy(monkeypatch, policy: str, consequence: str) -> None:
    """DOC-001. A desktop host previously printed the refusal warning under skip."""
    from recall import generation_rescue

    def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AtomicRescueArtifactError("the root is full")

    monkeypatch.setattr(generation_rescue, "ensure_for_promotion", fail)
    env = {"RECALL_ATOMIC_RESCUE_WHEN_MISSING": policy}
    warning = generation_rescue.after_publish("postgresql://x", "p-docs", "gen_live", _FakeEmbedder, env=env)
    assert consequence in warning, "the warning described the wrong missing-artifact policy"
    assert "recall --tenant p-docs generation rescue gen_live" in warning


def test_gc_removes_the_artifacts_of_the_collected_generations(tmp_path: Path, monkeypatch, capsys) -> None:
    """RP-9: removing the `remove_artifacts(collected)` call made this fail: the matrix stayed."""
    import recall.generations as generations
    from recall.cli_commands.generation_cmd import _cmd_generation

    monkeypatch.setattr(generations, "GenerationManager", _CliManager)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    collected, kept = _artifact(tmp_path, "gen_old", "a.md"), _artifact(tmp_path, "gen_live", "a.md")

    _cmd_generation(_args("gc", retention_days=7, retain_previous=2))

    assert not collected.exists() and kept.exists()
    assert "removed 1 atomic rescue artifact" in capsys.readouterr().out


# ---------------------------------------------------------------------------- files, no database


def _artifact(root: Path, generation_id: str, source: str) -> Path:
    write_atomic_rescue_artifact(
        root / generation_id,
        matrix=np.eye(1, 4, dtype=np.float32),
        views=[{"chunk_id": "c1", "source": source, "parent_ordinal": 0, "view_ordinal": 0}],
        generation_id=generation_id,
        calibration_id="cal_1",
        pipeline_fingerprint="p",
        corpus_fingerprint="c",
        embedding_profile="hashing",
        embedding_fingerprint="f",
        ordinary_chunk_count=1,
        source_commit="test",
    )
    return root / generation_id


def test_a_forget_erases_the_views_and_the_artifacts_naming_the_source(tmp_path: Path) -> None:
    """RP-10: `erase_forgotten` returning before the artifact removal made this fail."""
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest
    from recall.generation_rescue import erase_forgotten

    text = "the kingfisher protocol reconciles ledger drift at dawn"
    with AtomicViewStore(tmp_path / "micro-views.sqlite") as views:
        views.put("profile", chunk_view_digest(text), [[1.0, 0.0, 0.0, 0.0]])
        views.commit()
    naming, other, foreign = (
        _artifact(tmp_path, "gen_a", "forgotten.md"),
        _artifact(tmp_path, "gen_b", "kept.md"),
        _artifact(tmp_path, "gen_x", "forgotten.md"),
    )

    erased, removed = erase_forgotten(["gen_a", "gen_b"], [text], ["forgotten.md"], env={ROOT_ENV: str(tmp_path)})

    assert (erased, removed) == (1, 1)
    assert not naming.exists() and other.exists()
    assert foreign.exists(), "another tenant's generation is not this forget's to remove"
    with AtomicViewStore(tmp_path / "micro-views.sqlite") as views:
        assert views.count() == 0


def test_with_the_rescue_off_a_promotion_builds_nothing_and_loads_no_model(tmp_path: Path) -> None:
    from recall.generation_rescue import ensure_for_promotion

    def embedder():  # noqa: ANN202
        raise AssertionError("no model is loaded when the rescue is off")

    env = {"RECALL_ATOMIC_RESCUE_MODE": "off", ROOT_ENV: str(tmp_path)}
    assert ensure_for_promotion("postgresql://x", "t", "gen", embedder, env=env) is None
    by_tenant = {"RECALL_ATOMIC_RESCUE_MODE": "active", "RECALL_ATOMIC_RESCUE_MODE_BY_TENANT": "t:off", ROOT_ENV: str(tmp_path)}
    assert ensure_for_promotion("postgresql://x", "t", "gen", embedder, env=by_tenant) is None
    with pytest.raises(AtomicRescueArtifactError, match="absolute"):
        ensure_for_promotion("postgresql://x", "t", "gen", embedder, env={"RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: "rel"})


# -------------------------------------------------------------------------------- the database

#: Long enough to yield micro views (`recall.atomizer.chunk_micro_views` needs four content words).
_BODY = "document body {index} records how the archive keeps its ledgers balanced every evening"


def _bodies() -> dict[str, bytes]:
    from tests.test_calibration_carry_forward import CORPUS_SIZE

    return {f"memo-{index}.md": _BODY.format(index=index).encode() for index in range(CORPUS_SIZE)}


@pytest.fixture
def rescue_tenant():
    from recall.generations import GenerationManager

    tenant = "rescue-test-" + uuid.uuid4().hex[:10]
    yield tenant, GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for table in (
            "recall_calibrations", "recall_calibration_query_sets", "recall_source_tombstones",
            "recall_audit_events", "recall_ingest_jobs", "recall_tenant_state", "recall_generations",
        ):
            conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))


def _certified_ready(manager, embedder, version: str = "v1") -> str:
    """A built, validated generation with a published certified calibration, not yet promoted."""
    from recall.calibration_v2 import CalibrationRepository
    from tests.test_calibration_carry_forward import _labels, _manifest, _reader
    from tests.test_generations import _pipeline

    bodies = _bodies()
    manifest = _manifest(manager.tenant_id, bodies, version)
    generation = manager.create(manifest, _pipeline(embedder.name))
    manager.build(generation.generation_id, _reader(manifest, bodies), embedder, lambda t: [t])
    manager.validate(generation.generation_id)
    repo = CalibrationRepository(TEST_DSN, manager.tenant_id, actor="pytest")
    repo.publish(repo.calibrate(generation.generation_id, _labels(), embedder).calibration_id)
    return generation.generation_id


def _search(tenant: str, embedder, root: Path):
    from recall.generation_store import GenerationStore
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    env = {"RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: str(root)}
    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        return trusted_search(store, embedder, "answer-1", k=5, policy=TrustPolicy.from_env({}), env=env)


@requires_db
def test_a_generation_built_before_promotion_serves_under_an_active_rescue(rescue_tenant, tmp_path: Path) -> None:
    """The end-to-end claim: build, promote, and the active search is served rather than refused.

    RP-11 (CI, commit 629586d87): `ensure_for_promotion` returning None before `build_artifact`
    made this fail at `report is not None`: nothing was built for the generation it then promoted.
    """
    from recall.generation_rescue import ensure_for_promotion, generation_artifact_dirs
    from tests.test_calibration_carry_forward import _CarryEmbedder

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    generation_id = _certified_ready(manager, embedder)
    env = {"RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: str(tmp_path)}

    report = ensure_for_promotion(TEST_DSN, tenant, generation_id, lambda: embedder, env=env)
    manager.promote(generation_id, unsafe_development=True)

    assert report is not None and report.startswith(f"built for {generation_id}")
    assert [path.name.split(".")[0] for path in generation_artifact_dirs(tmp_path, generation_id)] == [generation_id]
    result = _search(tenant, embedder, tmp_path)
    assert result.trust_state == "trusted" and result.generation_id == generation_id
    again = ensure_for_promotion(TEST_DSN, tenant, generation_id, lambda: embedder, env=env)
    assert again == f"in force for {generation_id}"


@requires_db
def test_a_recalibrated_generation_gets_a_new_artifact(rescue_tenant, tmp_path: Path) -> None:
    """An artifact names its calibration, so a new one must replace it or every search refuses.

    RP-12 (CI, commit da0392051): `build_artifact` returning "in force" whenever the manifest file
    exists, without the lineage check, made this fail at the report: the artifact of the replaced
    calibration was kept as "in force", and the search behind it would refuse on calibration_id.
    """
    from recall.calibration_v2 import CalibrationRepository
    from recall.generation_rescue import after_publish, build_artifact
    from tests.test_calibration_carry_forward import _CarryEmbedder, _labels

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    generation_id = _certified_ready(manager, embedder)
    build_artifact(TEST_DSN, tenant, generation_id, embedder, tmp_path)
    manager.promote(generation_id, unsafe_development=True)
    repo = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    repo.publish(repo.calibrate(generation_id, _labels(), embedder).calibration_id)

    env = {"RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: str(tmp_path)}
    report = after_publish(TEST_DSN, tenant, generation_id, lambda: embedder, env=env)

    assert report.startswith(f"atomic rescue: built for {generation_id}"), report
    assert _search(tenant, embedder, tmp_path).trust_state == "trusted"


@requires_db
def test_a_postgres_forget_erases_what_the_rescue_keeps(rescue_tenant, tmp_path: Path, monkeypatch) -> None:
    """RP-13 (CI): deleting the `GenerationStore` branch of the rescue erasure in
    `recall_mcp.lifecycle._forget_memory_unlocked` made this fail: the artifact naming the
    forgotten memo, and its view vectors, stayed under the root."""
    from recall.atomic_view_store import AtomicViewStore
    from recall.generation_rescue import build_artifact
    from recall.generation_store import GenerationStore
    from recall_mcp.lifecycle import forget_memory
    from tests.test_calibration_carry_forward import _CarryEmbedder

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    generation_id = _certified_ready(manager, embedder)
    build_artifact(TEST_DSN, tenant, generation_id, embedder, tmp_path)
    manager.promote(generation_id, unsafe_development=True)
    with AtomicViewStore(tmp_path / "micro-views.sqlite") as views:
        before = views.count()
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    source = f"s3://approved/corpora/{tenant}/memo-0.md"

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        forget_memory(store, [source])

    assert not (tmp_path / generation_id).exists()
    with AtomicViewStore(tmp_path / "micro-views.sqlite") as views:
        assert views.count() == before - 1


@requires_db
def test_a_desktop_upload_whose_rescue_cannot_be_built_is_not_activated(tmp_path: Path, monkeypatch) -> None:
    """RP-14 (CI): deleting the `ensure_for_promotion` call in `generation_ingest` made this fail:
    the upload returned normally, its generation built without an artifact."""
    import recall.generation_rescue as rescue
    import recall_mcp.desktop_ingest as desktop
    from recall.embeddings import HOSTED_UNVERIFIED_DIGEST, EmbeddingProfile
    from recall.store import PgVectorStore

    class _HostedEmbedder:
        """Hosted identity, so production admits the build, and deterministic vectors."""

        name = "voyage:voyage-4"
        dim = 64
        profile = EmbeddingProfile(
            profile_id="voyage-4-v1",
            model_name="voyage-4",
            artifact_digest=HOSTED_UNVERIFIED_DIGEST,
            dimension=64,
            query_mode="query",
            passage_mode="document",
        )

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(t) % 7) + 1.0 for _ in range(64)] for t in texts]

    monkeypatch.setenv("RECALL_ENV", "production")
    monkeypatch.setattr(desktop, "_certify_upload", lambda *args: None)
    built: list[str] = []

    def ensure(dsn, tenant, generation_id, embedder, **_):  # noqa: ANN001, ANN202
        built.append(generation_id)
        raise AtomicRescueArtifactError("no room")

    monkeypatch.setattr(rescue, "ensure_for_promotion", ensure)
    tenant = "rescue-upload-" + uuid.uuid4().hex[:10]
    staged = tmp_path / tenant / "job1"
    staged.mkdir(parents=True)
    (staged / "memo.md").write_text(_BODY.format(index=0), encoding="utf-8")
    store = PgVectorStore(TEST_DSN, dim=64, tenant=tenant)
    try:
        with pytest.raises(AtomicRescueArtifactError):
            desktop.generation_ingest(store, _HostedEmbedder(), str(staged), "text")
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            active = conn.execute(
                "SELECT active_generation_id FROM recall_tenant_state WHERE tenant_id = %s", (tenant,)
            ).fetchone()
    finally:
        store.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            for table in ("recall_chunks_v1", "recall_audit_events", "recall_tenant_state", "recall_generations"):
                conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))

    assert len(built) == 1
    assert active is None or active[0] is None


@requires_db
def test_a_build_that_loses_the_race_keeps_the_winners_artifact(rescue_tenant, tmp_path: Path, monkeypatch) -> None:
    """Two builds of one generation (a promote beside `recall generation rescue`): the one that
    finishes second must keep the artifact the first published, not delete it and write its own,
    which leaves a window with no artifact and, on Windows with the matrix mapped, a half-removed one.

    Red against 6fa3c8306 (the pre-fix `build_artifact`): the second build removed the first's
    artifact and returned "built for", so the published `constructed_at` changed.

    Its path is the keyed `<generation_id>.<calibration_id>` one since 2026-11-17. Red in CI
    (run 44125888491, DO NOT MERGE commit 5f6ab43cc, reverted by 6afc6e98b) with `build_artifact`
    writing the old `<generation_id>` key again: FileNotFoundError on the winner's keyed manifest,
    since the build wrote it elsewhere.
    """
    import json

    import recall.atomic_view_store as view_store
    from recall.generation_rescue import _published_calibration, build_artifact
    from tests.test_calibration_carry_forward import _CarryEmbedder

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    generation_id = _certified_ready(manager, embedder)
    calibration_id = _published_calibration(TEST_DSN, tenant, generation_id)
    manifest = tmp_path / f"{generation_id}.{calibration_id}" / "manifest.json"
    real = view_store.assemble
    winner: list[str] = []

    def overtaken(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        built = real(*args, **kwargs)
        if not winner:
            winner.append("racing")
            build_artifact(TEST_DSN, tenant, generation_id, embedder, tmp_path)  # publishes first
            winner.append(json.loads(manifest.read_text(encoding="utf-8"))["constructed_at"])
        return built

    monkeypatch.setattr(view_store, "assemble", overtaken)

    report = build_artifact(TEST_DSN, tenant, generation_id, embedder, tmp_path)

    assert report == f"in force for {generation_id}", report
    assert json.loads(manifest.read_text(encoding="utf-8"))["constructed_at"] == winner[1]


@requires_db
def test_a_rollback_whose_target_moved_during_its_build_is_refused(rescue_tenant, monkeypatch) -> None:
    """`rollback` builds the target's artifact before it takes the tenant-state lock, so a promote
    landing meanwhile moves the target to a generation nothing built for. That rollback must be
    refused, not activate the unbuilt generation.

    Red against 6fa3c8306 (the pre-fix rollback): the rollback activated the second generation,
    whose artifact was never built.
    """
    import recall.generation_rescue as rescue
    from recall.cli_commands.generation_cmd import _cmd_generation
    from tests.test_calibration_carry_forward import _CarryEmbedder

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    first, second, third = (_certified_ready(manager, embedder, version) for version in ("v1", "v2", "v3"))
    manager.promote(first, unsafe_development=True)
    manager.promote(second, unsafe_development=True)
    built: list[str] = []

    def ensure(dsn, tenant_id, generation_id, embedder_factory, **_):  # noqa: ANN001, ANN202
        built.append(generation_id)
        manager.promote(third, unsafe_development=True)  # lands while the build runs
        return f"built for {generation_id}"

    monkeypatch.setattr(rescue, "ensure_for_promotion", ensure)
    args = argparse.Namespace(dsn=TEST_DSN, tenant=tenant, embedder="hashing", generation_cmd="rollback")

    with pytest.raises((SystemExit, ValueError, RuntimeError)) as refused:
        _cmd_generation(args)

    assert built == [first]
    assert manager.active_generation_id() == third, "the unbuilt generation must not be activated"
    assert first in str(refused.value)


@requires_db
def test_prune_preserves_the_active_rescue_when_the_previous_pointer_returns(
    rescue_tenant, tmp_path: Path, monkeypatch
) -> None:
    """STAKES-002. Two equal pointer reads can straddle changes that make the inventory stale."""
    from recall.generation_rescue import prune_superseded
    from recall.generations import _GENERATION_COLUMNS
    from tests.test_calibration_carry_forward import _CarryEmbedder

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    retired, active, previous, next_ready = (
        _certified_ready(manager, embedder, version) for version in ("v1", "v2", "v3", "v4")
    )
    for generation_id in (retired, active, previous):
        manager.promote(generation_id, unsafe_development=True)
    assert manager.rollback() == active
    assert manager.previous_generation_id() == previous
    artifact = _artifact(tmp_path, active, "kept.md")

    real_execute = psycopg.Connection.execute
    interleaved = False

    def execute_with_promotions(conn, query, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        nonlocal interleaved
        inventory = f"SELECT {_GENERATION_COLUMNS} FROM recall_generations "
        if not interleaved and str(query).startswith(inventory) and "ORDER BY created_at" in str(query):
            interleaved = True
            manager.promote(next_ready, unsafe_development=True)
            with manager._connect() as writer:
                writer.execute(
                    "UPDATE recall_generations SET retired_at = '2000-01-01' "
                    "WHERE tenant_id = %s AND generation_id = %s",
                    (tenant, active),
                )
            rows = real_execute(conn, query, *args, **kwargs)
            assert manager.rollback() == active
            manager.promote(previous, unsafe_development=True)
            assert manager.rollback() == active
            return rows
        return real_execute(conn, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.Connection, "execute", execute_with_promotions)
    prune_superseded(TEST_DSN, tenant, tmp_path)

    assert interleaved, "the overlapping publications did not run"
    assert manager.active_generation_id() == active
    assert manager.previous_generation_id() == previous
    assert artifact.is_dir(), "prune deleted the rescue of the active generation"


@requires_db
@pytest.mark.parametrize("assembly_fails", [False, True], ids=["assembly-completes", "assembly-errors"])
def test_collection_after_forget_does_not_leave_reinserted_rescue_views(
    rescue_tenant, tmp_path: Path, monkeypatch, assembly_fails: bool
) -> None:
    """SEC-003. The builder used to leave cached views if its generation had been collected."""
    from datetime import UTC, datetime, timedelta

    from recall import atomic_view_store
    from recall.atomizer import chunk_micro_views
    from recall.embedding_core import embedding_profile
    from recall.generation_rescue import build_artifact
    from recall.generation_store import GenerationStore
    from recall.generation_types import NoActiveGeneration
    from recall_mcp.lifecycle import forget_memory
    from tests.test_calibration_carry_forward import _CarryEmbedder

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    first, second, third = (_certified_ready(manager, embedder, ver) for ver in ("v1", "v2", "v3"))
    manager.promote(first, unsafe_development=True)
    source = f"s3://approved/corpora/{tenant}/memo-0.md"
    assemble = atomic_view_store.assemble
    forgotten: list[str] = []
    profile = embedding_profile(embedder).fingerprint()

    def cached(views: atomic_view_store.AtomicViewStore):  # noqa: ANN202
        text = forgotten[0]
        return views.get(profile, atomic_view_store.chunk_view_digest(text),
                         view_count=len(chunk_micro_views(text)), dimension=embedder.dim)

    def assemble_after_collection(parents, views, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        forgotten.extend(text for _, uri, _, text in parents if uri == source)
        assert len(forgotten) == 1
        with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
            result = forget_memory(store, [source], erase_memo_requests=False)
        assert result.chunks_removed > 0
        for generation_id in (second, third):
            manager.promote(generation_id, unsafe_development=True)
        assert first in manager.gc(
            now=datetime.now(UTC) + timedelta(days=1), retention_days=0, retain_previous=0,
        )
        assembled = assemble(parents, views, *args, **kwargs)
        assert cached(views) is not None, "assembly must commit the forgotten vectors for this regression"
        if assembly_fails:
            raise RuntimeError("assembly failed after committing a cache batch")
        return assembled

    monkeypatch.setattr(atomic_view_store, "assemble", assemble_after_collection)
    with pytest.raises(NoActiveGeneration):
        build_artifact(TEST_DSN, tenant, first, embedder, tmp_path)

    with atomic_view_store.AtomicViewStore(tmp_path / "micro-views.sqlite") as views:
        assert cached(views) is None, "collection left the forgotten rescue vectors"
        assert views.count() == len(_bodies()) - 1, "collection removed unrelated cached views"
    assert not list(tmp_path.glob(f"{first}.*"))
