"""Resumable state: what it must reuse, what it must NOT reuse, and when it must forget.

The module exists because a corpus takes minutes to build and the wizard drives three of them, so a
run that dies on the second and rebuilds the first is the difference between an install a user
retries and one they abandon. Each property below is a way that could go wrong while still looking
like it works:

- reusing a DEGRADED corpus would make the re-run a no-op for the one corpus being re-run to fix;
- writing the state only at the end would lose exactly the run that never reaches the end;
- trusting a state file whose config has changed would skip work that is no longer valid;
- refusing to install because a state file is corrupt would be worse than ignoring it.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from recall.wizard.headless import HeadlessConfig, LegacyIndex, load_config, run_headless
from recall.wizard.pipeline import CorpusOutcome
from recall.wizard.state import (
    DIGEST_FIELDS,
    IGNORED_FIELDS,
    WizardState,
    config_digest,
    load_state,
    save_state,
)


def _config(tmp_path: Path, **overrides: Any) -> HeadlessConfig:
    payload: dict[str, Any] = {
        "dsn": "postgresql://recall:recall@127.0.0.1:1/recall",
        "migration_dsn": "postgresql://recall:recall@127.0.0.1:1/recall",
        "embedder": "hashing",
        "corpus_version": "2026-08-18",
        "docs_root": str(tmp_path / "docs"),
        "code_root": str(tmp_path / "repo"),
        "memory_root": str(tmp_path / "memory"),
    }
    payload.update(overrides)
    path = tmp_path / "wizard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_config(path)


def _promoted(tenant: str) -> CorpusOutcome:
    return CorpusOutcome(
        tenant=tenant,
        generation_id=f"gen_{tenant}",
        calibration_id=f"cal_{tenant}",
        certified=True,
        promoted=True,
        answerable=40,
        unanswerable=40,
        steps=("build", "validate", "calibrate", "publish", "promote"),
    )


# ----------------------------------------------------------------------------------------------
# The digest: what invalidates finished work
# ----------------------------------------------------------------------------------------------


def test_every_config_field_is_classified_as_digested_or_ignored() -> None:
    """A field added to `HeadlessConfig` without a decision here would silently not invalidate.

    That is the dangerous direction: a new field that changes what gets built, left out of the
    digest, means a stale corpus is reused forever and the operator cannot tell why re-running
    changes nothing.
    """
    declared = {f.name for f in fields(HeadlessConfig)}
    classified = set(DIGEST_FIELDS) | set(IGNORED_FIELDS)

    assert declared == classified, (
        f"unclassified config fields: {sorted(declared - classified)}; "
        f"classified but absent: {sorted(classified - declared)}"
    )
    assert not (set(DIGEST_FIELDS) & set(IGNORED_FIELDS)), "a field cannot be both"


@pytest.mark.parametrize("field_name", DIGEST_FIELDS)
def test_changing_a_digested_field_changes_the_digest(tmp_path: Path, field_name: str) -> None:
    """Parametrised over the real tuple, so a field added to it is covered without being retyped."""
    if field_name == "data_root":
        # `data_root` is the ALTERNATIVE to `dsn`, so a config carrying both is refused. Compare
        # two provisioning configs instead of bolting a location onto a DSN one.
        base = _config(
            tmp_path, dsn=None, migration_dsn=None, data_root=str(tmp_path / "here")
        )
        changed = _config(
            tmp_path, dsn=None, migration_dsn=None, data_root=str(tmp_path / "elsewhere")
        )
    else:
        base = _config(tmp_path)
        changed = _config(
            tmp_path,
            **{field_name: str(tmp_path / "other") if field_name.endswith("_root") else "changed"},
        )
    assert config_digest(base) != config_digest(changed), f"{field_name} must invalidate"


@pytest.mark.parametrize("field_name", IGNORED_FIELDS)
def test_changing_an_ignored_field_leaves_the_digest_alone(tmp_path: Path, field_name: str) -> None:
    """These decide who connects or where config is written, not what is built.

    So they must not throw away finished work. The value is chosen per field rather than shared,
    because `project_root` is validated as an absolute path and a single placeholder would fail
    validation rather than test the digest.
    """
    replacements = {
        "migration_dsn": "postgresql://recall_migrator:pw@127.0.0.1:1/recall",
        "serving_role": "recall_server",
        "fact_write_dsn": "postgresql://recall_fact_writer:pw@127.0.0.1:1/recall",
        "project_root": str(tmp_path / "elsewhere"),
    }
    assert field_name in replacements, f"no replacement value chosen for {field_name}"

    base = _config(tmp_path)
    changed = _config(tmp_path, **{field_name: replacements[field_name]})
    assert config_digest(base) == config_digest(changed), f"{field_name} must not invalidate"


def test_the_digest_does_not_depend_on_the_path_separator(tmp_path: Path) -> None:
    """`str(Path)` differs by platform, so a state file would be discarded when moved.

    Rendering with `as_posix()` means the same logical root digests the same on Windows and Linux,
    which matters because the config itself is written once and used on whichever machine installs.
    """
    config = _config(tmp_path)
    # Only the roots that are SET: `data_root` is absent on a DSN-driven config.
    assert "\\" not in json.dumps(
        {
            n: getattr(config, n).as_posix()
            for n in DIGEST_FIELDS
            if n.endswith("_root") and getattr(config, n) is not None
        }
    )


# ----------------------------------------------------------------------------------------------
# Reading and writing
# ----------------------------------------------------------------------------------------------


def test_state_round_trips_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    state = WizardState(digest="d" * 8).with_outcome(_promoted("default-docs")).with_indexed(
        LegacyIndex(tenant="default-memory", files=2, chunks=7)
    )
    save_state(path, state)

    restored = load_state(path, digest="d" * 8)
    assert restored == state, "a state file read back must compare equal to the run that wrote it"
    assert restored.outcome_for("default-docs") is not None
    assert restored.indexed_for("default-memory") == ("default-memory", 2, 7)


def test_a_digest_mismatch_forgets_everything(tmp_path: Path) -> None:
    """The operator changed the embedder or a root, so nothing recorded is valid any more."""
    path = tmp_path / "s.json"
    save_state(path, WizardState(digest="old").with_outcome(_promoted("default-docs")))

    fresh = load_state(path, digest="new")
    assert fresh.outcomes == (), "work recorded under another configuration must not be reused"
    assert fresh.digest == "new"


@pytest.mark.parametrize(
    "content",
    ["", "not json at all", "[]", '{"schema_version": 2, "digest": "d"}', '{"digest": ""}'],
)
def test_an_unreadable_state_file_is_ignored_rather_than_fatal(tmp_path: Path, content: str) -> None:
    """A corrupt state file must not stop an install.

    The worst case of ignoring it is repeating work that was already done. The worst case of
    refusing is a user who cannot install until they delete a file nobody told them about.
    """
    path = tmp_path / "s.json"
    path.write_text(content, encoding="utf-8")

    state = load_state(path, digest="d")
    assert state == WizardState(digest="d")


def test_an_absent_state_file_is_ignored(tmp_path: Path) -> None:
    assert load_state(tmp_path / "nope.json", digest="d") == WizardState(digest="d")


def test_the_write_is_atomic_and_leaves_no_temporary(tmp_path: Path) -> None:
    """The file is rewritten after every corpus, so a partial write is a real risk.

    A truncated file would be rejected by `from_dict` and the whole run's progress discarded, which
    is the failure this module exists to prevent.
    """
    path = tmp_path / "nested" / "s.json"
    save_state(path, WizardState(digest="d").with_outcome(_promoted("default-docs")))

    assert path.exists()
    assert not list(path.parent.glob("*.tmp")), "no temporary file may survive the write"
    assert load_state(path, digest="d").outcome_for("default-docs") is not None


# ----------------------------------------------------------------------------------------------
# What the driver does with it
# ----------------------------------------------------------------------------------------------


class _CountingSpy:
    """Counts how many times each corpus was actually built."""

    def __init__(self, *, crash: set[str] | None = None) -> None:
        self.built: list[str] = []
        self.legacy: list[str] = []
        self._crash = crash or set()

    def dim(self) -> int:
        from recall.embeddings import resolve_embedder

        return resolve_embedder("hashing").dim

    def apply_schema(self, dsn: str, *, dim: int) -> None:
        pass

    def grant(self, dsn: str, *, role: str) -> None:
        pass

    def index_legacy(self, spec: Any) -> Any:
        self.legacy.append(spec.tenant)
        return LegacyIndex(tenant=spec.tenant, files=1, chunks=4)

    def run(self, spec: Any, *, progress: Any = None) -> Any:
        self.built.append(spec.tenant)
        if spec.tenant in self._crash:
            raise RuntimeError(f"psycopg: connection lost building {spec.tenant}")
        return _promoted(spec.tenant)


def test_a_crash_part_way_keeps_what_finished(tmp_path: Path) -> None:
    """The whole point. `docs` is promoted, `code` dies; the re-run must not rebuild `docs`.

    Rebuilding it is not merely slow: every re-run creates a NEW generation and copies every chunk
    row and embedding into it, so a crash loop grows the database without bound.
    """
    config = _config(tmp_path)
    state_path = tmp_path / "s.json"

    first = _CountingSpy(crash={"default-code"})
    report = run_headless(config, services=first, state_path=state_path)
    assert first.built == ["default-docs", "default-code"]
    assert [f.tenant for f in report.failures] == ["default-code"]
    assert report.ok is False

    second = _CountingSpy()
    again = run_headless(config, services=second, state_path=state_path)

    assert second.built == ["default-code"], "docs was already promoted and must not be rebuilt"
    assert "default-docs" in again.reused
    assert {o.tenant for o in again.outcomes} == {"default-docs", "default-code"}
    assert again.ok is True
    assert "reused from a previous run" in again.render()


def test_a_degraded_corpus_is_retried_rather_than_reused(tmp_path: Path) -> None:
    """A degraded corpus was left unpromoted SO THAT re-running retries it.

    Reusing it would make the re-run a no-op for the one corpus the operator is re-running to fix,
    which is the most confusing possible outcome: nothing changes and nothing says why.
    """

    class _Degrading(_CountingSpy):
        def run(self, spec: Any, *, progress: Any = None) -> Any:
            self.built.append(spec.tenant)
            return CorpusOutcome(
                tenant=spec.tenant,
                generation_id=f"gen_{spec.tenant}",
                certified=False,
                promoted=False,
                degraded_reason="separability below the bar",
                previously_serving="gen_old",
            )

    config = _config(tmp_path)
    state_path = tmp_path / "s.json"

    first = _Degrading()
    run_headless(config, services=first, state_path=state_path)
    assert first.built == ["default-docs", "default-code"]

    second = _Degrading()
    run_headless(config, services=second, state_path=state_path)
    assert second.built == ["default-docs", "default-code"], "an unpromoted corpus must be retried"


def test_fresh_ignores_recorded_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state_path = tmp_path / "s.json"
    run_headless(config, services=_CountingSpy(), state_path=state_path)

    again = _CountingSpy()
    report = run_headless(config, services=again, state_path=state_path, fresh=True)

    assert again.built == ["default-docs", "default-code"], "--fresh must rebuild everything"
    assert report.reused == ()


def test_no_state_path_means_no_state_file_and_no_reuse(tmp_path: Path) -> None:
    """The default must not start writing files beside a caller's config unasked."""
    config = _config(tmp_path)
    run_headless(config, services=_CountingSpy())
    second = _CountingSpy()
    run_headless(config, services=second)

    assert second.built == ["default-docs", "default-code"]
    assert not list(tmp_path.glob("*.state.json"))


def test_a_state_file_that_cannot_be_written_does_not_fail_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State is an optimisation. Losing it costs a rebuild; raising costs the completed corpus.

    The default state path sits beside the operator's config, which they may not own, so a
    read-only directory is an ordinary condition rather than an exotic one. Found by reviewing my
    own diff: `save_state` raises `OSError` and nothing caught it, so the wizard would have crashed
    AFTER finishing the expensive work.
    """
    import recall.wizard.headless as H

    def _boom(path: Path, state: Any) -> None:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(H, "save_state", _boom)

    notes: list[str] = []
    spy = _CountingSpy()
    report = run_headless(
        _config(tmp_path), services=spy, state_path=tmp_path / "ro" / "s.json", progress=notes.append
    )

    assert report.ok is True, "the install must still succeed"
    assert spy.built == ["default-docs", "default-code"]
    assert any("could not write" in n for n in notes), "and it must SAY the run is not resumable"


def test_an_indexed_corpus_is_reused_too(tmp_path: Path) -> None:
    """`memory` is indexed rather than built, and re-indexing it on every retry is the same waste."""
    config = _config(tmp_path)
    state_path = tmp_path / "s.json"
    run_headless(config, services=_CountingSpy(), state_path=state_path)

    second = _CountingSpy()
    report = run_headless(config, services=second, state_path=state_path)

    assert second.legacy == [], "memory was already indexed"
    assert [i.tenant for i in report.indexed] == ["default-memory"]
    assert "default-memory" in report.reused
