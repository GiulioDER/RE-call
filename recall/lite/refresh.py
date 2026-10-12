"""Keep a lite store current with its memo folders, the way the production pipeline keeps its corpus.

The production memory pipeline runs manifest, build, validate, calibrate, promote and gc on a
schedule, short-circuits when the corpus is unchanged, reports without changing anything under
`--check`, and leaves its outcome in a state file. A lite store has one corpus and no generations, so
the same ideas map onto less machinery:

| production | lite |
|---|---|
| manifest of the memo folders | `snapshot`: (relative path, size, mtime_ns) of every file `index_path` would read |
| short-circuit when unchanged | the snapshot and the store's corpus equal those the last successful refresh left |
| `embed.lock` queue | the store's write lane, `lite_lane(..., purpose="refresh", coalesce=True)` |
| build | `Indexer.index_path`, which already skips unchanged content and prunes vanished files |
| calibrate | `ensure_calibrated` (carry forward, refit when needed) |
| promote, gc | nothing: one corpus, no generations |
| `last-memory-refresh.json` | `<db stem>.refresh.json` beside the store |

The trigger lives in the MCP server (`recall_mcp.lite_refresh`), because it is the only reader of
the store and it runs exactly when somebody may search; `recall lite refresh` is the same call for a
person or a scheduler. Design: recall-lab `research/designs/2026-10-12-lite-auto-refresh.md`.

A memo folder may contain the store itself. Its files never enter a snapshot because the indexer's
default walk never reads them: the state file is `.json`, which `recall.index` excludes by default,
and the database, its `-wal`/`-shm` and its lane files are not document types. Were that to change,
each refresh would move the next snapshot and the server would refresh forever; R12 pins it.
"""

from __future__ import annotations

import enum
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from recall.atomic_write import atomic_write_bytes
from recall.lite.writer_lock import LiteWriterBusy, LiteWriterCoalesced, lite_lane, lite_lane_status

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite.store import LiteStore
    from recall.store import PgVectorStore

__all__ = [
    "CheckReport",
    "Outcome",
    "PURPOSE",
    "RefreshReport",
    "Snapshot",
    "check",
    "read_state",
    "refresh",
    "snapshot",
    "state_path",
]

_log = logging.getLogger(__name__)

#: The write lane purpose; `coalesce=True` matches waiters on it.
PURPOSE = "refresh"
STATE_SCHEMA = 1


class Outcome(str, enum.Enum):
    REFRESHED = "refreshed"
    UNCHANGED = "unchanged"
    COALESCED = "coalesced"
    BUSY = "busy"
    FAILED = "failed"


@dataclass(frozen=True)
class Snapshot:
    """What the memo folders hold now, as cheaply as it can be asked: stat only, nothing read."""

    fingerprint: str
    files: dict[Path, list[Path]]
    missing: list[Path]

    @property
    def count(self) -> int:
        return sum(len(found) for found in self.files.values())


def state_path(db: str | Path) -> Path:
    path = Path(db).resolve()
    return path.with_name(path.stem + ".refresh.json")


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def snapshot(roots: Sequence[str | Path]) -> Snapshot:
    """Fingerprint the files `Indexer.index_path` would read under each root.

    The walk is `recall.index.candidate_files`, the indexer's own, so the snapshot can never name a
    different set from the one indexed. Size and `st_mtime_ns` stand in for content: a false change
    costs one pass in which the indexer skips each unchanged file by its content hash.
    """
    from recall.index import candidate_files

    digest = hashlib.sha256()
    files: dict[Path, list[Path]] = {}
    missing: list[Path] = []
    for root in sorted({Path(r).resolve() for r in roots}):
        digest.update(b"root\x00" + str(root).encode("utf-8") + b"\x00")
        if not root.is_dir():
            missing.append(root)
            files[root] = []
            continue
        kept: list[Path] = []
        for path in candidate_files(root):
            resolved = path.resolve()
            try:
                stat = resolved.stat()
            except OSError:
                continue  # vanished between the walk and the stat; the next snapshot sees it gone
            kept.append(resolved)
            rel = resolved.relative_to(root).as_posix()
            digest.update(f"{rel}\x00{stat.st_size}:{stat.st_mtime_ns}\x00".encode())
        files[root] = kept
    return Snapshot(digest.hexdigest(), files, missing)


def read_state(db: str | Path) -> dict[str, Any]:
    """The state file, or `{}` when it is absent, unreadable or of another schema."""
    try:
        document = json.loads(state_path(db).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict) or document.get("schema") != STATE_SCHEMA:
        return {}
    return document


def _write_state(db: Path, document: dict[str, Any]) -> None:
    atomic_write_bytes(state_path(db), json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))


@dataclass
class RefreshReport:
    outcome: Outcome
    reason: str
    fingerprint: str = ""
    files_seen: int = 0
    indexed_files: int = 0
    chunks_written: int = 0
    skipped: int = 0
    deleted: int = 0
    chunks_total: int = 0
    calibration: str = ""
    calibration_reason: str = ""
    holder: dict[str, object] | None = None
    started_utc: str = ""
    finished_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["outcome"] = self.outcome.value
        return values

    def render(self) -> str:
        lines = [f"refresh      {self.outcome.value}: {self.reason}"]
        if self.outcome is Outcome.REFRESHED:
            lines.append(
                f"indexed      {self.chunks_written} chunk(s) from {self.indexed_files} file(s); "
                f"{self.skipped} unchanged; {self.deleted} pruned; {self.chunks_total} chunk(s) in memory"
            )
        if self.calibration:
            lines.append(f"calibration  {self.calibration}: {self.calibration_reason}")
        return "\n".join(lines)


def _calibrate(store: LiteStore, embedder: Embedder, background: bool) -> Any:
    """`ensure_calibrated`, in the background when asked and when this version can.

    The `background` keyword arrives with the carry-forward change (PR 911). Asked for where it
    does not exist yet, the fit runs inline, which is what every caller did before it.
    """
    from recall.lite.calibration import ensure_calibrated

    if background and "background" in inspect.signature(ensure_calibrated).parameters:
        return cast(Any, ensure_calibrated)(store, embedder, background=True)
    return ensure_calibrated(store, embedder)


def _finish(db: Path, roots: list[Path], report: RefreshReport, *, success_generation: str | None = None) -> RefreshReport:
    """Write the state file for this run. `success_generation` marks it the new last success."""
    report.finished_utc = _utc()
    document: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "store": str(db),
        "roots": [str(r) for r in roots],
        "last_check_utc": report.finished_utc,
        "last": report.to_dict(),
        "last_success": read_state(db).get("last_success"),
    }
    if success_generation is not None:
        document["last_success"] = {
            "fingerprint": report.fingerprint,
            "generation_id": success_generation,
            "finished_utc": report.finished_utc,
        }
    _write_state(db, document)
    return report


def refresh(
    store: LiteStore,
    roots: Sequence[str | Path],
    embedder: Embedder,
    *,
    force: bool = False,
    wait_s: float | None = None,
    background_calibration: bool = False,
) -> RefreshReport:
    """Bring `store` up to date with `roots`. See the module docstring; never raises on a busy lane."""
    db = Path(store.path).resolve()
    resolved = [Path(r).resolve() for r in roots]
    report = RefreshReport(Outcome.UNCHANGED, "", started_utc=_utc())
    before = snapshot(resolved)
    report.fingerprint, report.files_seen = before.fingerprint, before.count
    if before.missing:
        report.outcome = Outcome.FAILED
        report.reason = "memory folder not found: " + ", ".join(str(m) for m in before.missing)
        return _finish(db, resolved, report)
    last = read_state(db).get("last_success") or {}
    if (
        not force
        and last.get("fingerprint") == before.fingerprint
        and last.get("generation_id") == store.lite_generation_id()
    ):
        # Nothing is written: this path holds no lane, and rewriting the file here could put back
        # a `last_success` that a concurrent refresh has just replaced.
        report.reason = f"no memo changed since {last.get('finished_utc')}"
        report.chunks_total = store.count()
        report.finished_utc = _utc()
        return report
    try:
        with lite_lane(db, lane="write", purpose=PURPOSE, wait_s=wait_s, coalesce=True):
            return _refresh_locked(store, db, resolved, embedder, report, background_calibration)
    except LiteWriterCoalesced:
        report.outcome = Outcome.COALESCED
        report.reason = "a refresh is already queued for this store and will see these changes"
        return _finish(db, resolved, report)
    except LiteWriterBusy as exc:
        report.outcome, report.reason, report.holder = Outcome.BUSY, str(exc), exc.holder
        return _finish(db, resolved, report)


def _refresh_locked(
    store: LiteStore,
    db: Path,
    roots: list[Path],
    embedder: Embedder,
    report: RefreshReport,
    background_calibration: bool,
) -> RefreshReport:
    from recall.cache import default_cache
    from recall.context import context_policy_for_profile
    from recall.embeddings import embedding_profile_id
    from recall.index import Indexer, chunk_text

    # Taken again inside the lane: whatever changed while this refresh queued is indexed now, and
    # whatever changes while it indexes moves the next snapshot, so it is refreshed next time.
    current = snapshot(roots)
    report.fingerprint, report.files_seen = current.fingerprint, current.count
    if current.missing:
        # Gone while this refresh queued. Indexing an empty list would read as "every memo deleted".
        report.outcome = Outcome.FAILED
        report.reason = "memory folder not found: " + ", ".join(str(m) for m in current.missing)
        return _finish(db, roots, report)
    started = time.monotonic()
    try:
        with default_cache() as cache:
            # Typed as the Postgres store, as everywhere the shared Indexer takes a lite one.
            indexer = Indexer(
                cast("PgVectorStore", store),
                embedder,
                chunker=chunk_text,
                cache=cache,
                context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
            )
            for root in roots:
                stats = indexer.index_path(root, files=current.files[root])
                report.indexed_files += stats.files
                report.chunks_written += stats.chunks
                report.skipped += stats.skipped
                report.deleted += stats.deleted
        outcome = _calibrate(store, embedder, background_calibration)
    except Exception as exc:  # BROAD-CATCH: fail-open  # recorded in the state file and returned
        _log.warning("lite refresh of %s failed: %s", db, exc)
        report.outcome, report.reason = Outcome.FAILED, f"{type(exc).__name__}: {exc}"
        report.chunks_total = store.count()
        return _finish(db, roots, report)
    report.outcome = Outcome.REFRESHED
    report.reason = f"{current.count} file(s) checked in {time.monotonic() - started:.1f}s"
    report.calibration, report.calibration_reason = outcome.status.value, outcome.reason
    report.chunks_total = store.count()
    return _finish(db, roots, report, success_generation=store.lite_generation_id())


@dataclass
class CheckReport:
    """`recall lite refresh --check`: what a refresh would find, with nothing opened for writing."""

    store: Path
    roots: list[Path]
    current: bool
    files_seen: int
    missing: list[Path] = field(default_factory=list)
    last: dict[str, Any] | None = None
    last_success_utc: str | None = None
    holder: dict[str, object] | None = None
    waiters: list[dict[str, object]] = field(default_factory=list)
    chunks: int = 0
    calibration: str = ""
    calibration_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["store"] = str(self.store)
        values["roots"] = [str(r) for r in self.roots]
        values["missing"] = [str(m) for m in self.missing]
        return values

    def render(self) -> str:
        if self.missing:
            state = "memory folder not found: " + ", ".join(str(m) for m in self.missing)
        elif self.current:
            state = f"current: no memo changed since the last refresh ({self.last_success_utc})"
        elif self.last_success_utc is None:
            state = "never refreshed"
        else:
            state = f"changed since the last refresh ({self.last_success_utc}); a refresh would index it"
        lines = [
            f"store        {self.store}",
            f"memory       {', '.join(str(r) for r in self.roots)}",
            f"files        {self.files_seen}",
            f"memos        {state}",
        ]
        if self.last:
            lines.append(f"last run     {self.last.get('outcome')} at {self.last.get('finished_utc')}: {self.last.get('reason')}")
        if self.holder:
            lines.append(f"writer       held by {self.holder.get('purpose')} (pid {self.holder.get('pid')}); {len(self.waiters)} waiting")
        else:
            lines.append(f"writer       free; {len(self.waiters)} waiting")
        lines.append(f"chunks       {self.chunks}")
        reason = f": {self.calibration_reason}" if self.calibration_reason else ""
        lines.append(f"calibration  {self.calibration}{reason}")
        return "\n".join(lines)


def check(db: str | Path, roots: Sequence[str | Path]) -> CheckReport:
    """Report what `refresh` would find. Opens the store read-only and writes nothing anywhere."""
    from recall.lite.inspect import inspect_lite_file

    target = Path(db).resolve()
    resolved = [Path(r).resolve() for r in roots]
    snap = snapshot(resolved)
    state = read_state(target)
    last_success = state.get("last_success") or {}
    info = inspect_lite_file(target)
    current = (
        not snap.missing
        and info.exists
        and last_success.get("fingerprint") == snap.fingerprint
        and last_success.get("generation_id") == info.generation_id
    )
    lane = lite_lane_status(target, "write")
    last = state.get("last")
    calibration_reason = ""
    if isinstance(last, dict) and last.get("calibration") == info.calibration.value:
        # The reason a refresh was given (the cold start text among them) still describes the store.
        calibration_reason = str(last.get("calibration_reason") or "")
    if info.error:
        calibration_reason = info.error
    return CheckReport(
        store=target,
        roots=resolved,
        current=bool(current),
        files_seen=snap.count,
        missing=snap.missing,
        last=last if isinstance(last, dict) else None,
        last_success_utc=last_success.get("finished_utc"),
        holder=lane.holder,
        waiters=lane.waiters,
        chunks=info.chunks,
        calibration=info.calibration.value if info.exists else "no store",
        calibration_reason=calibration_reason,
    )
