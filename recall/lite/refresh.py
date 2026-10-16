"""Keep a lite store current with its memo folders, the way the production pipeline keeps its corpus.

The production memory pipeline runs manifest, build, validate, calibrate, promote and gc on a
schedule, short-circuits when the corpus is unchanged, reports without changing anything under
`--check`, and leaves its outcome in a state file (here: each run that held the write lane). A lite store has one corpus and no generations, so
the same ideas map onto less machinery:

| production | lite |
|---|---|
| manifest of the memo folders | `snapshot`: (relative path, size, mtime_ns) of every file `index_path` would read |
| short-circuit when unchanged | the snapshot and the store's corpus equal those the last successful refresh left |
| `embed.lock` queue | the store's write lane, `lite_lane(..., purpose=refresh_purpose(roots), coalesce=True)` |
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
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from recall.atomic_write import atomic_write_bytes
from recall.index_budget import IndexBudget, budget_refusal
from recall.lite.writer_lock import LiteWriterBusy, LiteWriterCoalesced, lite_lane, lite_lane_status

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite.calibration import CalibrationOutcome
    from recall.lite.store import LiteStore
    from recall.security_policy import AccessContext, SourceSecurityPolicy
    from recall.store import PgVectorStore

__all__ = [
    "CheckReport",
    "Outcome",
    "PURPOSE",
    "RefreshReport",
    "Snapshot",
    "check",
    "over_budget",
    "policy_key",
    "read_state",
    "refresh",
    "refresh_access_context",
    "refresh_purpose",
    "snapshot",
    "state_path",
]

_log = logging.getLogger(__name__)

#: The write lane purpose's prefix; `refresh_purpose` adds the folders, so only a refresh of the
#: same folders coalesces.
PURPOSE = "refresh"
STATE_SCHEMA = 1
#: A whole-second mtime this close to "now" may be followed, within the same tick, by an edit the
#: stat cannot see (FAT, exFAT, HFS+, some network shares). Such a snapshot is never recorded as
#: clean (CCA BUG-007).
_RACY_WINDOW_NS = 2_000_000_000


def refresh_purpose(roots: Sequence[str | Path]) -> str:
    """The write lane purpose for refreshing exactly `roots` (CCA STAKES-003).

    Coalescing matches a waiter by purpose, so a refresh of other folders must never count as one
    that "will see these changes".
    """
    folders = "\x00".join(sorted(str(Path(r).resolve()) for r in roots))
    return f"{PURPOSE}:{hashlib.sha256(folders.encode('utf-8')).hexdigest()[:12]}"


def _missing_reason(missing: Sequence[Path]) -> str:
    return "memory folder not found: " + ", ".join(str(m) for m in missing)


class Outcome(str, enum.Enum):
    """What a refresh did.

    - `refreshed`: indexed what changed and recalibrated. A cold start below the calibration
      floor is still `refreshed`; its calibration reason says why trusted answers wait.
    - `unchanged`: no memo changed since the last success; nothing was taken or written.
    - `coalesced`: a refresh of the same folders is already queued and will see these changes.
    - `busy`: the write lane stayed held past the wait.
    - `failed`: nothing was recorded as current (missing folder, over the budget, denied by the
      policy, an indexing or calibration error); the reason says which.
    """

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
    total_bytes: int = 0
    racy: bool = False  # a whole-second mtime within `_RACY_WINDOW_NS` of the walk (CCA BUG-007)

    @property
    def count(self) -> int:
        return sum(len(found) for found in self.files.values())


def state_path(db: str | Path) -> Path:
    path = Path(db).resolve()
    return path.with_name(path.stem + ".refresh.json")


def refresh_access_context(tenant: str) -> AccessContext:
    """The one context a refresh applies the source policy with, whoever runs it.

    It is the context the MCP server's own `recall_index` gets on a stdio server (no token: principal
    `local`, purpose `retrieval`), because the server is what serves this store. The CLI and the
    server share one state file, so a refresh under any other context would record a "current"
    state the server then trusts (CCA FIX-001, architect REVISE).
    """
    from recall.security_policy import AccessContext

    return AccessContext(principal="local", tenant=tenant)


def policy_key(policy: SourceSecurityPolicy | None, context: AccessContext | None) -> str | None:
    """What a refresh's policy decisions depend on: the policy digest and the deciding context."""
    if policy is None:
        return None
    assert context is not None
    material = [policy.digest, context.principal, context.tenant, context.purpose, context.clearance,
                str(context.egress_allowed)]
    return hashlib.sha256("\x00".join(material).encode("utf-8")).hexdigest()


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def snapshot(roots: Sequence[str | Path]) -> Snapshot:
    """Fingerprint the files `Indexer.index_path` would read under each root.

    The walk is `recall.index.candidate_files_with_stats`, the indexer's own walk with the stat it
    already takes, so the snapshot can never name a different set from the one indexed. Size and `st_mtime_ns` stand in for content: a false change
    costs one pass in which the indexer skips each unchanged file by its content hash.
    """
    from recall.index import candidate_files_with_stats

    digest = hashlib.sha256()
    files: dict[Path, list[Path]] = {}
    missing: list[Path] = []
    total_bytes = 0
    racy = False
    now_ns = time.time_ns()
    for root in sorted({Path(r).resolve() for r in roots}):
        digest.update(b"root\x00" + str(root).encode("utf-8") + b"\x00")
        if not root.is_dir():
            missing.append(root)
            files[root] = []
            continue
        kept: list[Path] = []
        for path, stat in candidate_files_with_stats(root):
            # Kept exactly as the walk names it: the indexer keys rows on this spelling, so a
            # resolved path would file a symlinked memo under its target, beside the rows every
            # other indexer wrote under the link (CCA BUG-001). The stat is the one the walk took
            # to check the file is regular (it follows the link), so nothing is stat'ed twice
            # (CCA PERF-003).
            kept.append(path)
            total_bytes += stat.st_size
            # Both sides: FAT rounds mtimes up, and a future-dated file must not mark every
            # snapshot racy for as long as its date stays ahead (deferred pass, L5.5).
            if stat.st_mtime_ns % 1_000_000_000 == 0 and abs(now_ns - stat.st_mtime_ns) < _RACY_WINDOW_NS:
                racy = True
            rel = path.relative_to(root).as_posix()
            digest.update(f"{rel}\x00{stat.st_size}:{stat.st_mtime_ns}\x00".encode())
        files[root] = kept
    return Snapshot(digest.hexdigest(), files, missing, total_bytes, racy)


def over_budget(snap: Snapshot, budget: IndexBudget | None) -> str | None:
    """Why `snap` is more than `budget` allows an unattended refresh to index, or None."""
    if budget is None:
        return None
    folders = ", ".join(str(root) for root in snap.files)
    refusal = budget_refusal(f"the memory folder {folders}", snap.count, snap.total_bytes, budget)
    return None if refusal is None else refusal + " Nothing was indexed."


def read_state(db: str | Path) -> dict[str, Any]:
    """The state file, or `{}` when it is absent, unreadable or of another schema."""
    try:
        document = json.loads(state_path(db).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict) or document.get("schema") != STATE_SCHEMA:
        return {}
    return document


#: Attempts at replacing the state file. On Windows a reader without FILE_SHARE_DELETE (Python's
#: `open`) makes the replace fail with PermissionError for as long as it holds the file open.
_REPLACE_ATTEMPTS = 5


def _write_state(db: Path, document: dict[str, Any]) -> None:
    """Replace the state file; a sharing violation that persists is logged, never raised.

    The state file reports a refresh; it is not the refresh. Raising on a reader holding the file
    would turn an index that succeeded into a traceback and a lost outcome (CCA BUG-003). Any other
    OSError (a full disk) still raises.
    """
    data = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            atomic_write_bytes(state_path(db), data)
            return
        except PermissionError as exc:
            if attempt == _REPLACE_ATTEMPTS:
                _log.warning("lite refresh state %s not written: %s", state_path(db), exc)
                return
            time.sleep(0.05 * attempt)


@dataclass
class RefreshReport:
    """One refresh: its outcome and reason, what it indexed, and the calibration it left.

    `fingerprint` is the snapshot the outcome is about; `to_dict` is what the state file's `last`
    and `recall_stats.refresh` carry.
    """

    outcome: Outcome
    reason: str
    fingerprint: str = ""
    files_seen: int = 0
    indexed_files: int = 0
    chunks_written: int = 0
    skipped: int = 0
    deleted: int = 0
    denied: int = 0
    chunks_total: int = 0
    calibration: str = ""
    calibration_reason: str = ""
    facts_stated: int = 0
    facts_malformed: int = 0
    stated: str = ""
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
            if self.denied:
                lines.append(f"denied       {self.denied} file(s) refused by the source policy, not indexed")
        if self.calibration:
            lines.append(f"calibration  {self.calibration}: {self.calibration_reason}")
        if self.stated:
            lines.append(self.stated)
        return "\n".join(lines)


def _calibrate(store: LiteStore, embedder: Embedder, background: bool) -> CalibrationOutcome:
    """`ensure_calibrated`, in the background when asked and when this version can.

    The `background` keyword arrives with the carry-forward change (PR 911). Asked for where it
    does not exist yet, the fit runs inline, which is what every caller did before it.
    """
    from recall.lite.calibration import ensure_calibrated

    if background and "background" in inspect.signature(ensure_calibrated).parameters:
        return cast("CalibrationOutcome", cast(Any, ensure_calibrated)(store, embedder, background=True))
    return ensure_calibrated(store, embedder)


def _finish(
    db: Path,
    roots: list[Path],
    report: RefreshReport,
    *,
    success_generation: str | None = None,
    corpus_version: int | None = None,
    policy: str | None = None,
    key_root: Path | None = None,
) -> RefreshReport:
    """Write the state file for this run. `success_generation` marks it the new last success."""
    report.finished_utc = _utc()
    document: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "store": str(db),
        "roots": [str(r) for r in roots],
        "key_root": str(key_root) if key_root is not None else read_state(db).get("key_root"),
        "last_check_utc": report.finished_utc,
        "last": report.to_dict(),
        "last_success": read_state(db).get("last_success"),
    }
    if success_generation is not None:
        document["last_success"] = {
            "fingerprint": report.fingerprint,
            "generation_id": success_generation,
            "corpus_version": corpus_version,
            "policy": policy,
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
    allow_prune: bool = False,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
    budget: IndexBudget | None = None,
    before: Snapshot | None = None,
    key_root: str | Path | None = None,
) -> RefreshReport:
    """Bring `store` up to date with `roots`. See the module docstring; never raises on a busy lane.

    `before` is a snapshot of `roots` the caller has just taken (the server's check), so the
    folder is not walked twice for one decision (CCA PERF-004).

    `key_root` is the folder stored file names are relative to (RECALL_INDEX_ROOT on a lite
    server, owner decision 2026-10-13). It applies to each root it contains; it is recorded in
    the state file, where `recall lite refresh` reads it back, so a person refreshing a subfolder
    files names exactly as the server does.

    `budget` bounds what an unattended refresh may index (`recall.index_budget`, the rule
    `recall_index` applies): a folder over it is FAILED, nothing indexed or pruned. The server
    passes it; a person running `recall lite refresh` is not bound by it (CCA SEC-002).

    `security_policy` and `security_context` are the operator's source policy, applied exactly as
    `recall_index` applies it: a file the policy denies is not indexed (and counted in `denied`),
    and the indexer redacts the rest before storing. Without them a refresh would store what the
    policy refuses, and rewrite rows indexed under it without their policy digest, so search would
    stop returning them (CCA SEC-001). `allow_prune` is for a person who has confirmed a large
    deletion (`recall lite refresh --allow-prune`); the server never passes it.
    """
    if security_policy is not None and security_context is None:
        raise ValueError("security_context is required when security_policy is configured")
    key = policy_key(security_policy, security_context)
    db = Path(store.path).resolve()
    resolved = [Path(r).resolve() for r in roots]
    report = RefreshReport(Outcome.UNCHANGED, "", started_utc=_utc())
    if before is None:
        before = snapshot(resolved)
    report.fingerprint, report.files_seen = before.fingerprint, before.count
    # A missing folder is refused under the lane (`_refresh_locked`), where the refusal is written
    # safely; nothing outside the lane writes the state file (CCA BUG-004).
    last = read_state(db).get("last_success") or {}
    if (
        not force
        and not before.missing
        and last.get("fingerprint") == before.fingerprint
        # The store's write counter, not `lite_generation_id()`: that loads every row and
        # embedding to hash the corpus, which a scheduled check of an unchanged folder would pay
        # on every run (CCA PERF-005).
        and last.get("corpus_version") == store._version()
        and last.get("policy") == key
    ):
        # Nothing is written: this path holds no lane, and rewriting the file here could put back
        # a `last_success` that a concurrent refresh has just replaced.
        report.reason = f"no memo changed since {last.get('finished_utc')}"
        report.chunks_total = store.count()
        report.finished_utc = _utc()
        return report
    try:
        with lite_lane(db, lane="write", purpose=refresh_purpose(resolved), wait_s=wait_s, coalesce=True):
            return _refresh_locked(
                store,
                db,
                resolved,
                embedder,
                report,
                background_calibration=background_calibration,
                allow_prune=allow_prune,
                security_policy=security_policy,
                security_context=security_context,
                budget=budget,
                key_root=None if key_root is None else Path(key_root).resolve(),
            )
    except LiteWriterCoalesced:
        report.outcome = Outcome.COALESCED
        report.reason = "a refresh of these folders is already queued and will see these changes"
    except LiteWriterBusy as exc:
        report.outcome, report.reason, report.holder = Outcome.BUSY, str(exc), exc.holder
    # Not written: this path holds no lane (CCA BUG-004). Both are transient, and the caller
    # reports them; the next refresh records what happened.
    report.finished_utc = _utc()
    return report


def _refresh_locked(
    store: LiteStore,
    db: Path,
    roots: list[Path],
    embedder: Embedder,
    report: RefreshReport,
    *,
    background_calibration: bool,
    allow_prune: bool,
    security_policy: SourceSecurityPolicy | None,
    security_context: AccessContext | None,
    budget: IndexBudget | None,
    key_root: Path | None = None,
) -> RefreshReport:
    from recall.cache import default_cache
    from recall.context import context_policy_for_profile
    from recall.embeddings import embedding_profile_id
    from recall.index import Indexer, PruneGuardTripped, chunk_text

    # Taken again inside the lane: whatever changed while this refresh queued is indexed now, and
    # whatever changes while it indexes moves the next snapshot, so it is refreshed next time.
    current = snapshot(roots)
    report.fingerprint, report.files_seen = current.fingerprint, current.count
    if current.missing:
        # Gone while this refresh queued. Indexing an empty list would read as "every memo deleted".
        report.outcome = Outcome.FAILED
        report.reason = _missing_reason(current.missing)
        return _finish(db, roots, report)
    if (refusal := over_budget(current, budget)) is not None:
        # Refused under the lane, never before it: the refusal is written to the state file, and
        # a write outside the lane could put back a `last_success` a concurrent refresh replaced.
        report.outcome, report.reason = Outcome.FAILED, refusal
        return _finish(db, roots, report)
    allowed = current.files
    if security_policy is not None:
        assert security_context is not None  # checked by `refresh`
        allowed = {
            root: [
                f for f in found
                if security_policy.decide(f.relative_to(root).as_posix(), security_context).allowed
            ]
            for root, found in current.files.items()
        }
        report.denied = current.count - sum(len(found) for found in allowed.values())
        if current.count and report.denied == current.count:
            # A refresh that indexes nothing must not mark the folder's changes as consumed.
            report.outcome = Outcome.FAILED
            report.reason = (
                f"the source policy denies all {current.count} memo file(s) to principal "
                f"{security_context.principal!r}, purpose {security_context.purpose!r}; nothing was indexed"
            )
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
                allow_prune=allow_prune,
                security_policy=security_policy,
                security_context=security_context,
            )
            for root in roots:
                keys = key_root if key_root is not None and root.is_relative_to(key_root) else None
                stats = indexer.index_path(root, files=allowed[root], key_root=keys)
                report.indexed_files += stats.files
                report.chunks_written += stats.chunks
                report.skipped += stats.skipped
                report.deleted += stats.deleted
        calibration = _calibrate(store, embedder, background_calibration)
    except Exception as exc:  # BROAD-CATCH: fail-open  # recorded in the state file and returned
        _log.warning("lite refresh of %s failed: %s", db, exc)
        report.outcome, report.reason = Outcome.FAILED, f"{type(exc).__name__}: {exc}"
        if isinstance(exc, PruneGuardTripped):
            # The guard's own advice names `recall index`; a lite user runs this command.
            report.reason += " To accept the deletions, run `recall lite refresh --allow-prune`."
        # The failure may be the store's own (closed, corrupt, locked): counting it must not stop
        # the failure from being recorded (CCA BUG-002).
        with suppress(Exception):
            report.chunks_total = store.count()
        return _finish(db, roots, report, key_root=key_root)
    # The facts the refreshed memos state about themselves, recorded once each; never raises
    # (`recall.stated_facts`). Ledger writes do not move `_version()`, so this cannot re-trigger a
    # refresh.
    from recall.stated_facts import reconcile_stated_facts

    stated = reconcile_stated_facts(store, roots=roots)
    report.facts_stated, report.facts_malformed = stated.facts_stated, stated.facts_malformed
    report.stated = stated.render()
    report.outcome = Outcome.REFRESHED
    report.reason = f"{current.count} file(s) checked in {time.monotonic() - started:.1f}s"
    report.calibration, report.calibration_reason = calibration.status.value, calibration.reason
    report.chunks_total = store.count()
    if current.racy:
        # A memo may have been edited after it was read, inside the same coarse mtime tick. Record
        # the success under a fingerprint no snapshot produces, so the next check refreshes again;
        # the indexer then skips unchanged files by their content hash (CCA BUG-007).
        report.fingerprint = current.fingerprint + ":racy"
    return _finish(
        db,
        roots,
        report,
        success_generation=store.lite_generation_id(),
        corpus_version=store._version(),
        policy=policy_key(security_policy, security_context),
        key_root=key_root,
    )


@dataclass
class CheckReport:
    """`recall lite refresh --check`: what a refresh would find, with nothing opened for writing."""

    store: Path
    roots: list[Path]
    current: bool
    files_seen: int
    bytes_seen: int = 0
    budget: str | None = None
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
            state = _missing_reason(self.missing)
        elif self.current:
            state = f"current: no memo changed since the last refresh ({self.last_success_utc})"
        elif self.last_success_utc is None:
            state = "never refreshed"
        else:
            state = f"changed since the last refresh ({self.last_success_utc}); a refresh would index it"
        lines = [
            f"store        {self.store}",
            f"memory       {', '.join(str(r) for r in self.roots)}",
            f"files        {self.files_seen} ({self.bytes_seen} bytes)",
            f"memos        {state}",
        ]
        if self.last:
            lines.append(f"last run     {self.last.get('outcome')} at {self.last.get('finished_utc')}: {self.last.get('reason')}")
        if self.holder:
            lines.append(f"writer       held by {self.holder.get('purpose')} (pid {self.holder.get('pid')}); {len(self.waiters)} waiting")
        else:
            lines.append(f"writer       free; {len(self.waiters)} waiting")
        if self.budget:
            lines.append(f"budget       {self.budget}")
        lines.append(f"chunks       {self.chunks}")
        reason = f": {self.calibration_reason}" if self.calibration_reason else ""
        lines.append(f"calibration  {self.calibration}{reason}")
        return "\n".join(lines)


def check(
    db: str | Path,
    roots: Sequence[str | Path],
    *,
    policy: str | None = None,
    budget: IndexBudget | None = None,
) -> CheckReport:
    """Report what `refresh` would find. Opens the store read-only and writes nothing anywhere.

    `policy` is the `policy_key` a refresh would apply (None for no policy): a refresh under a
    different policy or context is not current.
    """
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
        and last_success.get("policy") == policy
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
        bytes_seen=snap.total_bytes,
        budget=None if budget is None else (
            over_budget(snap, budget)
            or f"within it: {snap.count} of {budget.max_files} file(s), {snap.total_bytes} of "
            f"{budget.max_bytes} byte(s) (RECALL_INDEX_MAX_FILES, RECALL_INDEX_MAX_BYTES)"
        ),
        missing=snap.missing,
        last=last if isinstance(last, dict) else None,
        last_success_utc=last_success.get("finished_utc"),
        holder=lane.holder,
        waiters=lane.waiters,
        chunks=info.chunks,
        calibration=info.calibration.value if info.exists else "no store",
        calibration_reason=calibration_reason,
    )
