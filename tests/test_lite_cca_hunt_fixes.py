"""Fixes from the DEEP CCA hunt of `recall/lite` (run 20261103T223409Z).

Invariants and the failure each one catches:
- H1 (NUM-001) a stop word joined to other words by an underscore is dropped from a keyword query,
  as PostgreSQL's parser drops it, so `is_lite_dsn` ranks like `lite dsn` and a chunk of stop
  words alone matches nothing.
- H2 (ENV-002) a lite DSN naming a relative path is refused before any file is created.
- H3 (STAKES-001) deleting a source deletes the calibration evidence of every fit measured on it
  (whose test questions quote its text); evidence of a fit that never saw the source stays.
- H4 (DAT-001) a newest calibration row that does not verify does not stop the next index from
  calibrating: `ensure_calibrated` refits and the store serves trusted answers again.
- H5 (BUG-001) a staging file that cannot be removed after its memo was published does not turn
  the write into an error, and the memo keeps its author.
- H6 (DEPLOY-001) `recall setup --lite` refuses to register the server when the MCP SDK is not
  installed, instead of registering one that cannot start.
- H7 (ENV-003) for a store outside `.recall/`, setup names the refresh state file to ignore.
- H8 to H11 (PERF-001, residue) a store whose fit did not certify is refitted only once it has
  grown by a quarter of the chunks that fit measured (inline, and in a background job's re-check),
  a refit it does get runs off the write path when the write asks for the background, and a fit
  for another model is refitted at once whatever the growth.

Red proof: each test's docstring names the mutation (the pre-fix line of the production code) it
was run against and the assertion that failed.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.index import Indexer
from recall.lite import LiteStore, LiteStoreError
from recall.lite import memo_requests
from recall.lite.calibration import auto_calibrate, ensure_calibrated, schedule_refit, wait_for_refit
from recall.lite.setup import run_lite_setup
from recall.types import Chunk
from tests.test_lite_calibration import DIM as WIDE
from tests.test_lite_calibration import ContentWordEmbedder, _store, _strict, _write_memos

DIM = 8


def _vec(*head: float) -> list[float]:
    return [*head, *([0.0] * (DIM - len(head)))]


def _chunk(cid: str, text: str, source: str) -> Chunk:
    return Chunk(id=cid, source=source, text=text, metadata={"file": source.rsplit("/", 1)[-1]})


def test_h1_an_underscore_joined_stop_word_is_dropped(tmp_path: Path) -> None:
    """Red against `_TERM = re.compile(r"\\w+")`: the chunk of stop words alone matched."""
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    store.replace_sources(["/m/a.md"], [_chunk("a", "this is what it is and is", "/m/a.md")], [_vec(1, 0)])
    store.replace_sources(["/m/b.md"], [_chunk("b", "lite dsn parsing", "/m/b.md")], [_vec(0, 1)])
    glued = store.query_sparse("is_lite_dsn", k=5)
    spaced = store.query_sparse("lite dsn", k=5)
    assert [h.chunk.id for h in glued] == ["b"], "a chunk of stop words matched an underscore-joined stop word"
    assert [h.score for h in glued] == pytest.approx([h.score for h in spaced]), "the stop word changed the rank"


def test_h2_a_relative_lite_dsn_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `lite_dsn_path` returning the path unchecked: DID NOT RAISE, and the file appeared."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LiteStoreError, match="absolute path"):
        LiteStore.from_dsn("sqlite:///home/me/memory.db", dim=DIM)
    assert not (tmp_path / "home").exists(), "a relative DSN created a store under the working directory"
    absolute = LiteStore.from_dsn("sqlite:///" + (tmp_path / "ok.db").as_posix(), dim=DIM)
    assert absolute.path.name == "ok.db"


def _evidence_rows(store: LiteStore) -> int:
    return int(store._metadata_values("SELECT count(*) FROM calibration_evidence")[0][0])  # noqa: SLF001


def test_h3_deleting_a_source_deletes_the_evidence_quoting_it(tmp_path: Path) -> None:
    """Red against `delete_sources` without the evidence DELETE: "a deleted source's questions stayed"."""
    store = _store(tmp_path, 48)
    assert auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    assert _evidence_rows(store) == 1
    later = tmp_path / "m" / "later.md"
    later.write_text("# A later memo\n\nWritten after the fit.\n", encoding="utf-8")
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    sources = sorted({row.source for row in store.iter_chunks()})
    later_source = next(s for s in sources if s.endswith("later.md"))
    store.delete_sources([later_source])
    assert _evidence_rows(store) == 1, "a source the fit never saw took the fit's evidence with it"
    fitted = next(s for s in sources if s != later_source)
    store.delete_sources([fitted])
    assert _evidence_rows(store) == 0, "a deleted source's questions stayed in the calibration evidence"
    assert store.resolve_calibration().status is CalibrationStatus.STALE, "a fit without evidence still carried"


def test_h4_an_unreadable_newest_calibration_is_refitted(tmp_path: Path) -> None:
    """Red against `fit = artifact_from_json(raw)` unguarded in `ensure_calibrated`: CalibrationError."""
    store = _store(tmp_path, 48)
    auto_calibrate(store, ContentWordEmbedder())
    with store._lock:  # noqa: SLF001  # the test plays the part of someone editing the file
        raw = store._conn.execute("SELECT payload FROM calibrations").fetchone()[0]  # noqa: SLF001
        tampered = re.sub(r'"threshold":-?[0-9.eE+-]+', '"threshold":-0.99', raw, count=1)
        assert tampered != raw
        store._conn.execute("UPDATE calibrations SET payload = ?", (tampered,))  # noqa: SLF001
    outcome = ensure_calibrated(store, ContentWordEmbedder())
    assert outcome.status is CalibrationStatus.CERTIFIED, outcome.reason
    assert _strict(store).trust_state == "trusted", "the refit did not replace the unreadable calibration"


def test_h5_an_unremovable_staging_file_keeps_the_published_memo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `staged.unlink(missing_ok=True)` in `_create`'s finally: PermissionError."""
    folder = tmp_path / "memos"
    folder.mkdir()
    ledger = folder / ".recall-authors.json"
    real_unlink = Path.unlink

    def held_open(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith(".recall-memo-"):
            raise PermissionError(32, "being used by another process", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held_open)
    reply = memo_requests.write_memo(
        folder, ledger, "A note.\n", kind="note", author="assistant", supersedes=None,
        request_id=None, expected_revision=None, include_receipt=False,
    )
    memo_id = reply["memo_id"]
    assert (folder / f"{memo_id}.md").exists()
    assert memo_requests.load_ledger(ledger)["authors"].get(memo_id) == "assistant", "the published memo lost its author"


def test_h6_setup_refuses_to_register_without_the_mcp_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `run_lite_setup` without the `find_spec("mcp")` check: DID NOT RAISE (it registered)."""
    project = tmp_path / "project"
    project.mkdir()
    _write_memos(project / "notes", 30)
    config = tmp_path / "claude.json"
    config.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "mcp", None)
    with pytest.raises(ValueError, match="MCP extra"):
        run_lite_setup(
            project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(),
            embedder_name="content-words", config_path=config,
        )
    assert json.loads(config.read_text(encoding="utf-8")) == {"projects": {}}, "a server was registered anyway"


def test_h7_a_custom_store_names_its_refresh_state_to_ignore(tmp_path: Path) -> None:
    """Red against setup's note naming only `<store>*`: "the refresh state file was not named"."""
    project = tmp_path / "project"
    project.mkdir()
    _write_memos(project / "notes", 30)
    report = run_lite_setup(
        project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(),
        embedder_name="content-words", store_path=project / "data" / "memory.db", register=False,
    )
    assert any("data/memory.refresh.json" in note for note in report.notes), "the refresh state file was not named"


class _FlatEmbedder:
    """Every text on one vector: no test question separates, so no fit ever certifies."""

    dim = WIDE

    def __init__(self, name: str = "test-flat") -> None:
        self.name = name
        self.calls: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(threading.get_ident())
        return [[1.0] + [0.0] * (WIDE - 1) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _uncertified(tmp_path: Path) -> tuple[LiteStore, Path, int]:
    """A 30-memo store whose fit did not certify, its memo folder, and its calibration row count."""
    store = LiteStore(tmp_path / "memory.db", dim=WIDE)
    root = tmp_path / "m"
    _write_memos(root, 30)
    Indexer(store, _FlatEmbedder(), env={}).index_path(root)
    assert auto_calibrate(store, _FlatEmbedder()).status is CalibrationStatus.UNCERTIFIED
    return store, root, len(store.calibration_rows())


def _grow(store: LiteStore, root: Path, count: int, start: int) -> None:
    _write_memos(root, count, start=start)
    Indexer(store, _FlatEmbedder(), env={}).index_path(root)


def test_h8_an_uncertified_store_is_refitted_only_after_a_quarter_more_memory(tmp_path: Path) -> None:
    """Red against the pre-fix `ensure_calibrated` (master bb28fcced, no `_uncertified_refit_waits`
    call): "a write under a quarter refitted an uncertified store", 30 to 32 chunks ran a fit."""
    store, root, fits = _uncertified(tmp_path)
    try:
        _grow(store, root, 2, 30)
        outcome = ensure_calibrated(store, _FlatEmbedder())
        assert len(store.calibration_rows()) == fits, "a write under a quarter refitted an uncertified store"
        assert outcome.status is CalibrationStatus.STALE
        _grow(store, root, 8, 32)
        ensure_calibrated(store, _FlatEmbedder())
        assert len(store.calibration_rows()) == fits + 1, "growing past a quarter did not refit"
    finally:
        store.close()


def test_h9_a_due_refit_of_an_uncertified_store_runs_in_the_background(tmp_path: Path) -> None:
    """Red against the pre-fix `if background and fit.certified:` in `ensure_calibrated`: "the refit
    of an uncertified store ran on the write's own thread"."""
    store, root, fits = _uncertified(tmp_path)
    try:
        _grow(store, root, 10, 30)
        embedder = _FlatEmbedder()
        outcome = ensure_calibrated(store, embedder, background=True)
        assert threading.get_ident() not in embedder.calls, "the refit of an uncertified store ran on the write's own thread"
        assert outcome.status is CalibrationStatus.STALE
        wait_for_refit(store, timeout=120)
        assert embedder.calls, "no background refit ran"
        assert len(store.calibration_rows()) == fits + 1, "the background refit stored no fit"
    finally:
        store.close()


def test_h10_a_background_refit_waits_for_the_same_quarter(tmp_path: Path) -> None:
    """Red against `_refit_wanted` without its `_uncertified_refit_waits` check: "a background
    refit fitted an uncertified store that grew under a quarter"."""
    store, root, fits = _uncertified(tmp_path)
    try:
        _grow(store, root, 2, 30)
        schedule_refit(store, _FlatEmbedder())
        wait_for_refit(store, timeout=120)
        assert len(store.calibration_rows()) == fits, "a background refit fitted an uncertified store that grew under a quarter"
    finally:
        store.close()


def test_h11_a_fit_for_another_model_is_refitted_whatever_the_growth(tmp_path: Path) -> None:
    """Red against `_uncertified_refit_waits` without `or _fitted_for_another_model(store, fit)`:
    "the store kept another model's uncertified fit"."""
    store, root, fits = _uncertified(tmp_path)
    try:
        _grow(store, root, 2, 30)
        # What a model switch leaves: the store bound to the model that re-embedded it.
        with sqlite3.connect(store.path) as conn:
            conn.execute("UPDATE meta SET value = 'test-flat-2' WHERE key = 'embedder_model'")
        conn.close()
        ensure_calibrated(store, _FlatEmbedder("test-flat-2"))
        assert len(store.calibration_rows()) == fits + 1, "the store kept another model's uncertified fit"
    finally:
        store.close()
