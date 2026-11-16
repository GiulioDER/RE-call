"""CCA deferred pass 20261116T132021Z over run 20261116T104135Z (the atomic rescue on at first install).

Each test names the finding it pins. No database: the PostgreSQL store and calibration repository
are the fakes of `tests.test_atomic_rescue_default_on_cca`, and the forget tests drive
`recall_mcp.lifecycle._forget_memory_unlocked` with a `GenerationStore` double. Red proof, against
the pre-fix files at 307938837 (`cca_checks tautology verify`), is in the pull request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
)
from recall.lite.atomic import MODE_BY_TENANT_ENV, MODE_ENV, ROOT_ENV
from tests.test_atomic_rescue_default_on_cca import _Corpus, _publish, corpus  # noqa: F401  # a fixture
from tests.test_calibration_carry_forward import _CarryEmbedder

def test_a_cli_publish_whose_rebuild_cannot_build_the_embedder_warns(tmp_path: Path, monkeypatch, capsys) -> None:
    """BUG-005: the CLI embedder factory exits on a failed construction, and `after_publish`
    catches Exception only, so a publish that had committed exited non-zero without the retry
    hint. Invariant: the publish command returns, printing the WARNING with the retry command.
    """
    import recall.cli_commands.calibration_cmd as calibration_cmd

    def unbuildable(spec: str):  # noqa: ANN202
        raise SystemExit("the embedding provider is unreachable")

    monkeypatch.setattr(calibration_cmd, "_make_embedder", unbuildable)
    monkeypatch.setenv(MODE_ENV, "active")
    monkeypatch.delenv(MODE_BY_TENANT_ENV, raising=False)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))

    calibration_cmd._rescue_after_publish(
        argparse.Namespace(dsn="postgresql://x", tenant="t", embedder="hashing"), "gen_1"
    )

    err = capsys.readouterr().err
    assert err.startswith("WARNING"), err
    assert "generation rescue gen_1" in err and "unreachable" in err


def test_an_older_calibrations_build_does_not_replace_the_newer_one(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """STAKES-004: a build for calibration 1, overtaken by a publish of calibration 2 and its build,
    deleted calibration 2's artifact and wrote its own, so the generation served calibration 2
    beside a calibration 1 artifact and every unscoped search refused.

    Invariant: the older build refuses, and calibration 2's artifact stays.
    """
    import recall.atomic_view_store as view_store
    from recall.generation_rescue import build_artifact

    embedder = _CarryEmbedder()
    real = view_store.assemble
    overtaken: list[bool] = []

    def published_meanwhile(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        built = real(*args, **kwargs)
        if not overtaken:
            overtaken.append(True)
            corpus.repository.calibration_id = "cal_2"  # type: ignore[attr-defined]
            assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")
        return built

    monkeypatch.setattr(view_store, "assemble", published_meanwhile)

    with pytest.raises(AtomicRescueArtifactError, match="published calibration .* changed"):
        build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)

    clear_atomic_rescue_artifact_cache()
    assert load_atomic_rescue_artifact(tmp_path / "gen_1" / "manifest.json").calibration_id == "cal_2"


class _NoScan:
    def __iter__(self):  # noqa: ANN204
        raise AssertionError("the chunks of a generation whose artifact is in force were read")


def test_an_artifact_in_force_is_kept_without_reading_the_chunks(corpus: _Corpus, tmp_path: Path) -> None:  # noqa: F811
    """PERF-002: `build_artifact` read every chunk's text before it asked whether the artifact
    in place already serves this lineage (a rollback, a repeated `recall generation rescue`).
    Invariant: the second build answers "in force" without a chunk scan.
    """
    from recall.generation_rescue import build_artifact

    embedder = _CarryEmbedder()
    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")
    corpus.chunks = _NoScan()  # type: ignore[assignment]

    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path) == "in force for gen_1"


def test_a_forget_removes_an_artifact_whose_views_cannot_be_read(tmp_path: Path) -> None:
    """STAKES-005: an artifact whose `views.json` could not be read (a removal Windows left half
    done keeps the mapped matrix) was treated as not naming the forgotten source, so the forget
    kept its vectors and reported success. Invariant: it is removed; a readable artifact of other
    sources stays, and a generation with no artifact counts for nothing.
    """
    from recall.generation_rescue import erase_forgotten

    damaged, other = tmp_path / "gen_1", tmp_path / "gen_2"
    _publish(damaged, "cal_1")
    _publish(other, "cal_1")
    (damaged / "views.json").write_text("{", encoding="utf-8")
    assert "a.md" in (other / "views.json").read_text(encoding="utf-8"), "negative control"

    erased, removed = erase_forgotten(["gen_1", "gen_2", "gen_3"], [], ["forgotten.md"], env={ROOT_ENV: str(tmp_path)})

    assert not damaged.exists(), "an unreadable artifact, which may hold the forgotten vectors, was kept"
    assert other.exists() and (erased, removed) == (0, 1)


def _generation_store(read: list[list[str]]):  # noqa: ANN202
    """A `GenerationStore` double holding one source, for `_forget_memory_unlocked`."""
    from recall.generation_store import GenerationStore

    class _Store(GenerationStore):
        def __init__(self) -> None:  # no connection
            self._tenant, self._dsn = "t", "postgresql://x"

        def sources_for_identifiers(self, requested: list[str]) -> dict[str, list[str]]:
            return {source: [source] for source in requested}

        def texts_for_sources(self, sources: list[str]) -> list[tuple[str, dict[str, object]]]:
            read.append(list(sources))
            return [("the forgotten memo's text", {})]

        def delete_sources(self, sources: list[str], *, withdraw: bool = False) -> int:
            return len(sources)

    return _Store()


@pytest.fixture
def forget_without_a_root(monkeypatch):  # noqa: ANN001, ANN201
    """`_forget_memory_unlocked` on a `GenerationStore`, in a shell with no rescue settings."""
    import recall_mcp.lifecycle as lifecycle

    for name in (MODE_ENV, MODE_BY_TENANT_ENV, ROOT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(lifecycle, "_fact_ledger_for", lambda store: None)
    monkeypatch.setattr(lifecycle, "delete_staged_sources", lambda tenant, sources: 0)
    monkeypatch.setattr(lifecycle, "erase_cached_digests", lambda digests: (0, 0))
    return lifecycle


def test_a_postgres_forget_without_the_root_says_it_erased_nothing_of_the_rescue(forget_without_a_root, monkeypatch) -> None:
    """SEC-003: a CLI `recall forget` run where RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is not set (the
    wizard writes it to the project `.env` and the server blocks only) skipped the rescue erasure
    and reported success with no word of it. Invariant: the message names the variable.
    """
    import recall.generations as generations

    monkeypatch.setattr(generations, "GenerationManager", lambda *args, **kwargs: SimpleNamespace(
        list_generations=lambda: [], collected_generation_ids=lambda: set()
    ))

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["memo.md"], erase_memo_requests=False
    )

    assert result.chunks_removed == 1
    assert "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is not set" in result.message, result.message


def test_a_postgres_forget_with_a_relative_root_says_it_is_not_absolute(forget_without_a_root, monkeypatch) -> None:
    """SEC-003, follow-up: a relative root, which no builder writes under, was reported as not set.
    Invariant: the note says the root is not an absolute path.

    Red proof (recorded in the pull request): with `state` in `_forget_memory_unlocked` mutated to
    always read "is not set", this fails at the assertion below.
    """
    monkeypatch.setenv(ROOT_ENV, "relative/atomic")

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["memo.md"], erase_memo_requests=False
    )

    assert "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is not an absolute path" in result.message, result.message


def test_a_postgres_forget_without_the_root_reads_nothing_for_the_rescue(forget_without_a_root, monkeypatch) -> None:
    """PERF-005: with no root the rescue keeps nothing to erase, yet every PostgreSQL forget read
    the forgotten texts and listed the generation history for it. Invariant: only the embedding
    cache's read of the texts happens, and no generation is listed.
    """
    import recall.generations as generations

    listed: list[str] = []

    def manager(*args: object, **kwargs: object) -> SimpleNamespace:
        listed.append("listed")
        return SimpleNamespace(list_generations=lambda: [], collected_generation_ids=lambda: set())

    monkeypatch.setattr(generations, "GenerationManager", manager)
    read: list[list[str]] = []

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store(read), ["memo.md"], erase_memo_requests=False
    )

    assert result.chunks_removed == 1
    assert listed == [], "the generation history was listed for a rescue with no root"
    assert read == [["memo.md"]], f"the texts were read again for a rescue with no root: {read}"


def test_a_wizard_rerun_builds_under_the_mode_it_registers(tmp_path: Path, monkeypatch) -> None:
    """ENV-003: the wizard decides each tenant's mode from RECALL_ATOMIC_RESCUE_MODE, but its
    builder resolved through the process environment, where a re-run from the project directory
    holds the previous run's RECALL_ATOMIC_RESCUE_MODE_BY_TENANT from `.env`; `t:shadow` there made
    an active run build nothing. Invariant: the builder resolves the mode this run registers.
    """
    import recall.generation_rescue as rescue
    from recall.lite.atomic import rescue_active
    from recall.wizard.headless import _RealServices

    monkeypatch.setenv(MODE_BY_TENANT_ENV, "t:shadow")
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(rescue, "ensure_for_promotion", lambda *args, env, **_: seen.append(dict(env)) or "built")
    services = _RealServices(config=SimpleNamespace(resolved_dsn="postgresql://x"))  # type: ignore[arg-type]
    services.atomic_env = {MODE_ENV: "active", ROOT_ENV: str(tmp_path)}

    builder = services._rescue_builder("t")
    assert builder is not None
    builder("gen_1")

    assert rescue_active("t", seen[0]), "the previous run's per-tenant mode overrode this run's"


def test_a_lite_write_that_built_its_artifact_starts_no_refresh(tmp_path: Path, monkeypatch) -> None:
    """PERF-003: `recall_index` and `recall index` build the artifact for the memory they wrote but
    write no refresh state, so the server's next check took the store for unrescued and ran a
    full refresh, the search waiting on it. Invariant: the artifact in force answers the check.
    """
    from recall.index import Indexer
    from recall.lite import LiteStore
    from recall.lite.atomic import ensure_rescue_artifact
    from recall.lite.calibration import wait_for_refit
    from recall_mcp.lite_refresh import LiteAutoRefresher
    from tests.test_lite_atomic_rescue import _blank_rescue_environment, _corpus, _env, _in_force
    from tests.test_lite_calibration import DIM, ContentWordEmbedder

    _blank_rescue_environment(monkeypatch)
    root = tmp_path / "atomic"
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    now = [1000.0]
    refresher = LiteAutoRefresher(
        store, embedder, [_corpus(tmp_path)], interval_s=30, wait_s=120, clock=lambda: now[0], env=_env(root)
    )
    try:
        refresher.before_retrieval()
        wait_for_refit(store)
        other = tmp_path / "other"
        other.mkdir()
        (other / "elsewhere.md").write_text(" ".join(f"else{k:02d}" for k in range(40)) + "\n", encoding="utf-8")
        Indexer(store, embedder, env={}).index_path(other)
        wait_for_refit(store)
        assert refresher._rescue_stale(), "negative control: a write that built nothing needs a refresh"

        ensure_rescue_artifact(store, embedder, env=_env(root))  # what `recall_index` does after its write

        assert _in_force(store, root).is_file()
        assert not refresher._rescue_stale(), "a refresh was asked for although the artifact is in force"
    finally:
        refresher.stop()


def test_a_damaged_lite_flat_artifact_is_removed_by_a_forget(tmp_path: Path) -> None:
    """STAKES-005, the Lite twin: a flat `lite-*` artifact at the top of the root whose `views.json`
    cannot be read is removed by a forget, as one naming the source is; it can never serve."""
    from recall.calibration_v2 import CalibrationStatus
    from recall.lite.atomic import erase_forgotten

    flat = tmp_path / ("lite-" + "0" * 16 + ".cal_1")
    flat.mkdir()
    (flat / "matrix.npy").write_bytes(b"the forgotten chunks' view vectors")
    (flat / "views.json").write_text(json.dumps({"no": "views"}), encoding="utf-8")
    store = SimpleNamespace(
        path=tmp_path / "memory.lite",
        tenant="t",
        resolve_calibration=lambda: SimpleNamespace(status=CalibrationStatus.MISSING, artifact=None),
    )

    erase_forgotten(store, [], ["forgotten.md"], env={ROOT_ENV: str(tmp_path)})  # type: ignore[arg-type]

    assert not flat.exists()
