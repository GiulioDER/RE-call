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
from tests.conftest import TEST_DSN, requires_db
from tests.test_atomic_rescue_default_on_cca import _Corpus, _publish, corpus  # noqa: F401  # a fixture
from tests.test_generation_rescue import rescue_tenant  # noqa: F401  # a fixture
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
    # The rescue's factory, shared with promote and rollback since CCA CODE-012, reads it here.
    monkeypatch.setattr("recall.cli_commands._shared._make_embedder", unbuildable)
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
    assert load_atomic_rescue_artifact(tmp_path / "gen_1.cal_2" / "manifest.json").calibration_id == "cal_2"


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

        def delete_sources(
            self, sources: list[str], *, withdraw: bool = False,
            erased_texts: list[tuple[str, dict[str, object]]] | None = None,
        ) -> int:
            # Protocol-only fixture update for committed DELETE payloads; root assertions stay unchanged.
            if erased_texts is not None:
                erased_texts.extend(("the forgotten memo's text", {}) for _ in sources)
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


def _manager(recorded: set[str], listed: list[str]):  # noqa: ANN202
    """A `GenerationManager` double: the roots builds recorded, and one generation, `gen_1`."""

    def manager(*args: object, **kwargs: object) -> SimpleNamespace:
        def generations() -> list[SimpleNamespace]:
            listed.append("listed")
            return [SimpleNamespace(generation_id="gen_1")]

        return SimpleNamespace(
            rescue_roots=lambda: set(recorded),
            list_generations=generations,
            collected_generation_ids=lambda: set(),
        )

    return manager


def test_a_build_records_the_root_it_builds_under(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """SEC-003, the record: a forget run where RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is not set could
    not know that the tenant had a rescue, or where. Invariant: a build records its root before it
    writes anything under it (the view store first, review BUG-001), and a build that finds its
    artifact in force records it too, since one built before the record existed has none (BUG-004).
    """
    import recall.generations as generations
    from recall.generation_rescue import build_artifact

    real = generations.GenerationManager
    stores_at_record: list[bool] = []

    class _Spy(real):  # type: ignore[misc, valid-type]
        def record_rescue_root(self, generation_id: str, root: str) -> None:
            stores_at_record.append((Path(root) / "micro-views.sqlite").exists())
            super().record_rescue_root(generation_id, root)

    monkeypatch.setattr(generations, "GenerationManager", _Spy)
    embedder = _CarryEmbedder()

    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")
    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path) == "in force for gen_1"

    assert corpus.recorded == [("gen_1", str(tmp_path))] * 2
    assert stores_at_record[0] is False, "the root was recorded after views were stored under it"


def test_a_postgres_forget_of_a_tenant_with_no_rescue_says_and_reads_nothing_of_it(
    forget_without_a_root, monkeypatch
) -> None:
    """SEC-003 and PERF-005: with no root configured and none recorded, the tenant never built a
    rescue, yet every forget appended a note about one. Invariant: no note, no read of the
    forgotten texts beyond the embedding cache's, and no listing of the generations.
    """
    import recall.generations as generations

    listed: list[str] = []
    monkeypatch.setattr(generations, "GenerationManager", _manager(set(), listed))
    read: list[list[str]] = []

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store(read), ["memo.md"], erase_memo_requests=False
    )

    assert result.chunks_removed == 1
    assert "atomic rescue" not in result.message.lower(), result.message
    assert listed == [], "the generation history was listed for a tenant with no rescue"
    assert read == [["memo.md"]], f"the texts were read again for a tenant with no rescue: {read}"


def test_a_postgres_forget_without_the_root_erases_under_the_recorded_one(
    forget_without_a_root, monkeypatch, tmp_path: Path
) -> None:
    """SEC-003: a CLI forget run where RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is not set (the wizard
    writes it to the project `.env` and the server blocks only) left the forgotten text's views
    and the artifact naming its source. Invariant: both are erased under the root a build recorded.
    """
    import recall.generations as generations
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest

    root = tmp_path / "atomic"
    _publish(root / "gen_1", "cal_1")  # names a.md
    with AtomicViewStore(root / "micro-views.sqlite") as views:
        views.put("profile", chunk_view_digest("the forgotten memo's text"), [[1.0, 0.0, 0.0, 0.0]])
        views.commit()
    monkeypatch.setattr(generations, "GenerationManager", _manager({str(root)}, []))

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["a.md"], erase_memo_requests=False
    )

    assert not (root / "gen_1").exists(), result.message
    with AtomicViewStore(root / "micro-views.sqlite") as views:
        assert views.count() == 0, "the forgotten text's views were kept"
    assert "Erased 1 atomic rescue view set(s) and removed 1" in result.message, result.message


def test_a_postgres_forget_names_a_recorded_root_it_cannot_reach(forget_without_a_root, monkeypatch, tmp_path: Path) -> None:
    """SEC-003: when the rescue was built under a root this process cannot reach (another host, a
    container), nothing there is erased and a re-run finds no rows. Invariant: the note names it."""
    import recall.generations as generations

    elsewhere = str(tmp_path / "on-another-host")
    monkeypatch.setattr(generations, "GenerationManager", _manager({elsewhere}, []))

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["memo.md"], erase_memo_requests=False
    )

    assert f"built under {elsewhere}, which this process cannot reach" in result.message, result.message


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


def test_a_damaged_lite_flat_artifact_is_reported_by_a_forget(tmp_path: Path) -> None:
    """An unreadable legacy Lite artifact cannot establish store ownership.

    On a7024d569, erase_forgotten returned success and deleted the flat folder.
    """
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
        rescue_roots=lambda: [],  # the store records no build root
    )

    warning = ""
    try:
        erase_forgotten(store, [], ["forgotten.md"], env={ROOT_ENV: str(tmp_path)})  # type: ignore[arg-type]
    except AtomicRescueArtifactError as exc:
        warning = str(exc)

    assert "owner is unknown" in warning and flat.exists(), "unowned cleanup was not reported"


def test_forget_roots_never_probes_a_network_root_from_the_record(tmp_path: Path, monkeypatch) -> None:
    """Review SEC-001: a root read from the database was probed before its kind was checked, and on
    Windows a UNC root opens a connection to the host it names. Invariant: UNC and device roots
    are reported unreachable without any filesystem call on them.
    """
    import recall.generations as generations
    from recall.generation_rescue import forget_roots

    forged = {"//attacker/share", "\\\\attacker\\share", "\\\\?\\C:\\atomic"}
    probed: list[str] = []
    for name in ("is_dir", "is_file", "exists"):
        real = getattr(Path, name)

        def spy(self: Path, *args: object, _real=real, **kwargs: object) -> bool:  # noqa: ANN001
            probed.append(str(self))
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(Path, name, spy)
    monkeypatch.setattr(generations, "GenerationManager", _manager(forged, []))

    usable, unreachable, unread = forget_roots("postgresql://x", "t", env={})

    assert (usable, sorted(unreachable), unread) == ([], sorted(forged), False)
    assert not any("attacker" in path or "?" in path for path in probed), probed


def test_a_forget_keeps_a_damaged_folder_that_is_not_an_artifact(tmp_path: Path) -> None:
    """Review SEC-002: an unreadable `views.json` made any folder removable, so a forged root or
    generation id could reach an unrelated directory. Invariant: a folder holding anything but
    artifact files is kept, while a damaged artifact (STAKES-005) is still removed.
    """
    from recall.generation_rescue import erase_forgotten

    unrelated, damaged = tmp_path / "gen_1", tmp_path / "gen_2"
    unrelated.mkdir()
    (unrelated / "views.json").write_text("{", encoding="utf-8")
    (unrelated / "thesis.docx").write_bytes(b"not an artifact")
    _publish(damaged, "cal_1")
    (damaged / "views.json").write_text("{", encoding="utf-8")

    _erased, removed = erase_forgotten(["gen_1", "gen_2"], [], ["forgotten.md"], env={ROOT_ENV: str(tmp_path)})

    assert (unrelated / "thesis.docx").exists(), "a folder that is not an artifact was removed"
    assert not damaged.exists() and removed == 1


def test_an_unreadable_record_still_erases_under_the_configured_root(tmp_path: Path, monkeypatch) -> None:
    """Review BUG-002: when the record could not be read, the configured root was dropped with it,
    so a forget that always erased under RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT erased nothing.
    Invariant: the configured root is returned, with the record flagged unread.
    """
    import recall.generations as generations
    from recall.generation_rescue import forget_roots

    def unreadable(*args: object, **kwargs: object) -> SimpleNamespace:
        def fail() -> set[str]:
            raise OSError("the database is unreachable")

        return SimpleNamespace(rescue_roots=fail)

    monkeypatch.setattr(generations, "GenerationManager", unreadable)

    assert forget_roots("postgresql://x", "t", env={ROOT_ENV: str(tmp_path)}) == ([tmp_path], [], True)


def test_a_recorded_root_left_empty_here_is_not_reported(forget_without_a_root, monkeypatch, tmp_path: Path) -> None:
    """Review BUG-004: a recorded root whose view store was deleted, as the note advises, was
    reported unreachable on every later forget, and an audit row cannot be retired. Invariant: a
    recorded root that is a directory here without the view store holds nothing and is not named.
    """
    import recall.generations as generations

    emptied = tmp_path / "atomic"
    emptied.mkdir()
    monkeypatch.setattr(generations, "GenerationManager", _manager({str(emptied)}, []))

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["memo.md"], erase_memo_requests=False
    )

    assert "atomic rescue" not in result.message.lower(), result.message


def test_a_root_whose_erasure_fails_does_not_stop_the_others(forget_without_a_root, monkeypatch, tmp_path: Path) -> None:
    """Review BUG-003: one root's failure ended the erasure under every later root and replaced
    the counts with a warning that named only the variable. Invariant: the other roots are erased
    and the failing one is named.
    """
    import recall.generation_rescue as rescue
    import recall.generations as generations
    from recall.atomic_view_store import AtomicViewStore

    roots = [tmp_path / "a-fails", tmp_path / "b-erased"]
    for root in roots:
        _publish(root / "gen_1", "cal_1")  # names a.md
        with AtomicViewStore(root / "micro-views.sqlite") as views:
            views.commit()
    real = rescue.erase_forgotten

    def erase(generation_ids, texts, sources, *, env):  # noqa: ANN001, ANN202
        if env[ROOT_ENV] == str(roots[0]):
            raise OSError("the matrix is held open")
        return real(generation_ids, texts, sources, env=env)

    monkeypatch.setattr(rescue, "erase_forgotten", erase)
    monkeypatch.setattr(generations, "GenerationManager", _manager({str(root) for root in roots}, []))

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["a.md"], erase_memo_requests=False
    )

    assert not (roots[1] / "gen_1").exists(), "the root after the failing one was not erased"
    assert f"under {roots[0]} failed" in result.message, result.message


def test_dat_001_forget_removes_only_its_abandoned_writer_stages(tmp_path: Path) -> None:
    """Red on 31e6841d plus prior local fixes: erase_forgotten left both abandoned stages."""
    from recall.generation_rescue import erase_forgotten

    matching = tmp_path / ".gen_1.cal_1.0123456789abcdef"
    legacy = tmp_path / ".gen_1.0123456789abcdef"
    partial = tmp_path / ".gen_1.cal_3.fedcba9876543210"
    unrelated_source = tmp_path / ".gen_1.cal_2.0123456789abcdef"
    unrelated_generation = tmp_path / ".gen_10.cal_1.0123456789abcdef"
    unrelated_directory = tmp_path / ".gen_1.notes.0123456789abcdef"
    for directory in (matching, legacy, unrelated_source, unrelated_generation, unrelated_directory):
        _publish(directory, "cal_1")
    (unrelated_source / "views.json").write_text(
        json.dumps({"views": [{"source": "other.md"}]}), encoding="utf-8"
    )
    partial.mkdir()
    (partial / "matrix.npy").write_bytes(b"synthetic incomplete matrix")

    result = erase_forgotten(["gen_1"], [], ["a.md"], env={ROOT_ENV: str(tmp_path)})

    assert not any(directory.exists() for directory in (matching, legacy, partial)), "forget left an abandoned writer stage"
    assert result == (0, 3)
    assert all(directory.is_dir() for directory in (unrelated_source, unrelated_generation, unrelated_directory))


def test_dat_001_forget_reports_failed_stage_removal_and_continues(
    forget_without_a_root, monkeypatch, tmp_path: Path,
) -> None:
    """Red on 31e6841d plus prior local fixes: public forget omitted the failed-root warning."""
    import recall.atomic_rescue as atomic
    import recall.generations as generations

    failing_root, other_root = tmp_path / "a-fails", tmp_path / "b-erased"
    stage_name = ".gen_1.cal_1.0123456789abcdef"
    blocked_stage = failing_root / stage_name
    for root in (failing_root, other_root):
        _publish(root / stage_name, "cal_1")
    remove = atomic.shutil.rmtree

    def remove_unless_blocked(path: Path, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if path != blocked_stage:
            remove(path, *args, **kwargs)

    monkeypatch.setattr(atomic.shutil, "rmtree", remove_unless_blocked)
    monkeypatch.setattr(generations, "GenerationManager", _manager({str(failing_root), str(other_root)}, []))
    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["a.md"], erase_memo_requests=False,
    )

    assert f"under {failing_root} failed" in result.message, result.message
    assert blocked_stage.is_dir()
    assert not (other_root / stage_name).exists(), "forget stopped before cleaning the remaining root"


@pytest.mark.parametrize("partial", [False, True], ids=["committed", "partial_commit"])
def test_dat_001_forget_refreshes_roots_after_committed_deletion(
    forget_without_a_root, monkeypatch, tmp_path: Path, partial: bool,
) -> None:
    """Red on 31e6841d: only the root list read before deletion was used for stage cleanup."""
    import recall.generations as generations

    recorded: set[str] = set()
    stage = tmp_path / "writer-root" / ".gen_1.cal_1.0123456789abcdef"
    store = _generation_store([])
    delete = store.delete_sources

    def delete_with_new_root(sources: list[str], **kwargs: object) -> int:
        _publish(stage, "cal_1")
        recorded.add(str(stage.parent))
        count = delete(sources, **kwargs)
        if partial:
            raise RuntimeError("synthetic failure after committed deletion")
        return count

    monkeypatch.setattr(generations, "GenerationManager", _manager(recorded, []))
    monkeypatch.setattr(store, "delete_sources", delete_with_new_root)
    if partial:
        with pytest.raises(RuntimeError, match="after committed deletion"):
            forget_without_a_root._forget_memory_unlocked(store, ["a.md"], erase_memo_requests=False)
    else:
        result = forget_without_a_root._forget_memory_unlocked(store, ["a.md"], erase_memo_requests=False)
        assert result.chunks_removed == 1
    assert not stage.exists(), "forget missed the stage under the newly recorded root"


@requires_db
def test_a_postgres_forget_without_the_root_finds_it_in_the_audit_trail(rescue_tenant, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """Review TEST-001: the record's INSERT and SELECT against a real database. Build with the
    root, forget with it unset; the artifact naming the source is removed.

    Red proof: not established. It needs PostgreSQL, which runs in CI only while the test host is
    frozen; the doubles above carry the proof for the forget's use of the record.
    """
    from recall.generation_rescue import build_artifact
    from recall.generation_store import GenerationStore
    from recall_mcp.lifecycle import forget_memory
    from tests.test_generation_rescue import _certified_ready

    tenant, manager = rescue_tenant
    embedder = _CarryEmbedder()
    generation_id = _certified_ready(manager, embedder)
    build_artifact(TEST_DSN, tenant, generation_id, embedder, tmp_path)
    manager.promote(generation_id, unsafe_development=True)
    monkeypatch.delenv(ROOT_ENV, raising=False)

    with GenerationStore(TEST_DSN, embedder.dim, tenant=tenant) as store:
        forget_memory(store, [f"s3://approved/corpora/{tenant}/memo-0.md"])

    assert not (tmp_path / generation_id).exists()


def test_a_root_the_forget_may_not_read_is_reported_not_raised(tmp_path: Path, monkeypatch) -> None:
    """Review NEW-001: Python before 3.14 raises PermissionError from `Path.is_dir`, and the forget
    reads the roots before it deletes anything, so one unreadable root refused every forget of the
    tenant. Invariant: such a root, recorded or configured, is reported unreachable.
    """
    import recall.generations as generations
    from recall.generation_rescue import forget_roots

    locked_recorded, locked_configured = tmp_path / "recorded", tmp_path / "configured"
    real = Path.is_dir

    def is_dir(self: Path) -> bool:
        if self in (locked_recorded, locked_configured):
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    monkeypatch.setattr(generations, "GenerationManager", _manager({str(locked_recorded)}, []))

    usable, unreachable, unread = forget_roots("postgresql://x", "t", env={ROOT_ENV: str(locked_configured)})

    assert (usable, sorted(unreachable), unread) == ([], sorted([str(locked_recorded), str(locked_configured)]), False)


def test_a_recorded_root_without_its_view_store_still_has_its_artifacts_erased(
    forget_without_a_root, monkeypatch, tmp_path: Path
) -> None:
    """Review NEW-002: a recorded root whose view store was deleted was skipped although its
    generation folders still held the forgotten chunks' vectors. Invariant: the artifact naming the
    forgotten source is removed there.
    """
    import recall.generations as generations

    root = tmp_path / "atomic"
    _publish(root / "gen_1", "cal_1")  # names a.md; no micro-views.sqlite
    monkeypatch.setattr(generations, "GenerationManager", _manager({str(root)}, []))

    result = forget_without_a_root._forget_memory_unlocked(
        _generation_store([]), ["a.md"], erase_memo_requests=False
    )

    assert not (root / "gen_1").exists(), result.message
