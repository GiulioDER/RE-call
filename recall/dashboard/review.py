"""The review queue behind `recall dashboard`: what is pending, and the two decisions a person makes.

Three sources feed it. The first two are filesystem only, so the page works on a corpus with no
database behind it:

* agent reports from `recall_report_stale`, read from `<root>/.recall/reports.sqlite3`;
* supersession proposals from the model arbiter, read CACHE-ONLY (`corpus_proposals(cache_only=True)`),
  so opening the page never spends anything;
* when a read-only corpus database is connected, the same agent reports as `stale_report` rows in
  `recall_audit_events`. A server on another host writes its sidecar where this machine cannot
  read it, so these rows are how its agents' reports reach the page.

A database row names memos as search served them (`recall/x.md`), so each name is mapped to the
one memo in this folder whose path it ends with. The row carries each quote's fingerprint, not its
text, and the quote is recovered from THIS folder's file (`stale_reports.find_quote`), which is the
verbatim check: the person decides on the text in front of them, and a row whose memo is not here,
or whose quote is not in it, is counted in the notes and not shown. Only the tenants this folder is
bound to are read (`recall dashboard --reports-tenant`, default `memory`). The rows are
append-only and the connection is read-only, so a database report is closed the way an arbiter
proposal is: a rejection goes to the rejection ledger, and an accept writes the declaration, after
which `already_declared` hides it.

All are keyed by `rewrite.claim_key` over the names in this folder, so one claim reported by an
agent, recorded in the database and proposed by the arbiter is one row. A claim a person rejected,
or one the newer memo already declares, is not shown.

Accepting runs the same chain as `recall rewrite apply`: `review_proposal`,
`accept_reviewed_proposal`, `promote_accepted_proposal`, then `apply_rewrite`, which writes
`supersedes:` into the newer memo. The page shows the planned edit first and sends back a hash of
the memo it showed; if the file changed in between, the accept is refused and the reviewer sees the
new plan. Rejecting records the claim in the rejection ledger, which both sources honour.
"""

from __future__ import annotations

import functools
import hashlib
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from recall.dashboard.db import MAX_STALE_REPORTS, DatabaseUnavailable
from recall.document import parse_document
from recall.errors import RecallError
from recall.frontmatter import supersedes_key, supersedes_targets
from recall.promotion import (
    PromotedFact,
    accept_reviewed_proposal,
    promote_accepted_proposal,
    review_proposal,
)
from recall.reasoning_proposals.types import InferenceProposal
from recall.rewrite import (
    RejectionLedger,
    RewritePlan,
    RewriteRefused,
    apply_rewrite,
    claim_key,
    corpus_proposals,
    default_ledger_path,
    plan_rewrite,
)
from recall.stale_reports import StaleReport, StaleReportQueue, default_queue_path, find_quote

AGENT_RULE_ID = "agent_stale_report"
AGENT_PROVIDER_ID = "recall_report_stale"
AGENT_PROVIDER_REVISION = "stale_reports.v1"

#: One decision at a time per process. `apply_rewrite` takes no lock, and two writes to one memo are
#: last writer wins; the memo hash below catches a writer outside this process.
_DECISION_LOCK = threading.Lock()


class ReviewRefused(ValueError, RecallError):
    """A decision that was not carried out, with the reason to show the reviewer."""


@dataclass(frozen=True)
class QueueItem:
    """One pending claim that `replacing` supersedes `stale`."""

    claim: str
    stale: str
    replacing: str
    stale_quote: str
    current_quote: str
    proposal: InferenceProposal
    origins: tuple[str, ...]
    details: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Queue:
    items: tuple[QueueItem, ...]
    notes: tuple[str, ...]


def agent_proposal(report: StaleReport) -> InferenceProposal:
    """The proposal an agent report stands for, so it goes through the same review chain."""
    return InferenceProposal(
        id=f"agent:{report.claim_key}",
        source_evidence_ids=(report.stale_source, report.replacing_source),
        proposed_relation="supersedes",
        subject_id=report.stale_source,
        object_id=report.replacing_source,
        explanation=f"reported by {report.client} ({report.report_count} report(s)) while using both memos",
        model_id=report.client,
        pipeline_id=AGENT_PROVIDER_ID,
        provider_id=AGENT_PROVIDER_ID,
        provider_revision=AGENT_PROVIDER_REVISION,
        confidence=None,
        uncertainty=("an agent's reading of two memos for one task; a person decides",),
        generation_id="filesystem",
        status="requires_review",
        rule_id=AGENT_RULE_ID,
        metadata={"quote_older": report.stale_quote, "quote_newer": report.current_quote},
    )


def rejected_claims(root: Path) -> frozenset[str]:
    """Claims in the ledger, read without creating it."""
    path = default_ledger_path(root)
    if not path.exists():
        return frozenset()
    with RejectionLedger(path) as ledger:
        return ledger.claims()


def already_declared(root: Path, stale: str, replacing: str) -> bool:
    """Whether the newer memo already names the stale one in `supersedes:`."""
    path = root / replacing
    try:
        meta = parse_document(path.read_text(encoding="utf-8-sig")).meta
    except (OSError, UnicodeDecodeError):
        return False
    wanted = supersedes_key(stale)
    return any(supersedes_key(target) == wanted for target in supersedes_targets(meta.get("supersedes")))


def memo_files(root: Path) -> list[Path]:
    """Every memo under the root, skipping dot directories such as the `.recall` sidecar."""
    return sorted(
        path
        for path in root.rglob("*.md")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def local_memo(source: str, names: Sequence[str]) -> str | None:
    """The memo in this folder that a served source names, by its trailing path components.

    `recall/x.md` and `file:///srv/memory/recall/x.md` both name `x.md`; with `x.md` and
    `recall/x.md` both here, the longer match wins, and two different names cannot match at the
    same length, so the longest match is the one memo. No match is None.

    What this cannot tell apart is the other direction: a served corpus that holds two stores'
    memos under one file name (`recall/x.md`, `other/x.md`) maps both to `x.md` here. The verbatim
    quote check against this folder's file is what keeps a report about the other one off the page,
    and the review page names the served source beside the local one.
    """
    parts = [part for part in source.replace("\\", "/").split("/") if part]
    best: str | None = None
    best_length = 0
    for name in names:
        tail = name.split("/")
        if len(tail) <= len(parts) and parts[-len(tail):] == tail and len(tail) > best_length:
            best, best_length = name, len(tail)
    return best


def _stamp(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value or "")


def _instant(value: object) -> datetime:
    """A row's time as an aware instant, so rows written under different offsets order correctly."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@functools.lru_cache(maxsize=4 * MAX_STALE_REPORTS)  # two quotes a row, with room for a second folder version
def _quote_in(path: str, mtime_ns: int, size: int, sha256: str, chars: int) -> str | None:
    """`find_quote` over one version of one memo, remembered across requests.

    Keyed on the file's modification time and size, so an edited memo is searched again; the
    search hashes every window of the memo, which is too slow to repeat on every page. A failed
    read raises rather than returning None, because `lru_cache` keeps a result but never an
    exception: a file locked for a moment must not read as "quote not found" until it changes.
    """
    return find_quote(Path(path).read_text(encoding="utf-8", errors="replace"), sha256, chars)


def _quote(path: Path, sha256: object, chars: object) -> str | None:
    if not isinstance(sha256, str) or not isinstance(chars, int) or isinstance(chars, bool):
        return None
    try:
        stat = path.stat()
        return _quote_in(str(path), stat.st_mtime_ns, stat.st_size, sha256, chars)
    except OSError:
        return None  # gone or unreadable right now: nothing to show this time, asked again next time


def database_items(
    root: Path,
    rows: Sequence[Mapping[str, object]],
    closed: Callable[[str, str, str], bool] | None = None,
) -> tuple[dict[str, QueueItem], list[str]]:
    """Queue items for the `stale_report` rows that name memos here and whose quotes are found in them.

    `closed(claim, stale, replacing)` names the claims already decided (rejected, or declared in
    the newer memo); their rows are counted and skipped before any quote is searched for.
    """
    names = [path.relative_to(root).as_posix() for path in memo_files(root)]

    # Each kept row with the local pair and the two quotes found for IT, so an item never shows
    # one row's quote under another row's memo.
    groups: dict[str, list[tuple[Mapping[str, object], str, str, str, str]]] = {}
    elsewhere = unquoted = decided = 0
    for row in rows:
        stale = local_memo(str(row.get("stale_source") or ""), names)
        replacing = local_memo(str(row.get("replacing_source") or ""), names)
        if stale is None or replacing is None or stale == replacing:
            elsewhere += 1
            continue
        key = claim_key("supersedes", stale, replacing)
        if closed is not None and closed(key, stale, replacing):
            decided += 1
            continue
        stale_quote = _quote(root / stale, row.get("stale_quote_sha256"), row.get("stale_quote_chars"))
        current_quote = _quote(root / replacing, row.get("current_quote_sha256"), row.get("current_quote_chars"))
        if stale_quote is None or current_quote is None:
            unquoted += 1
            continue
        groups.setdefault(key, []).append((row, stale, replacing, stale_quote, current_quote))

    items: dict[str, QueueItem] = {}
    for key, kept in groups.items():
        newest, stale, replacing, stale_quote, current_quote = max(kept, key=lambda k: _instant(k[0].get("created_at")))
        reports = [k[0] for k in kept]
        stamps = sorted((r.get("created_at") for r in reports), key=_instant)
        clients = ", ".join(sorted({str(r.get("client") or "unknown") for r in reports}))
        tenants = ", ".join(sorted({str(r.get("tenant") or "?") for r in reports}))
        task = next((str(r["task"]) for r in reports if r.get("task")), None)
        report = StaleReport(
            claim_key=key, stale_source=stale, replacing_source=replacing,
            stale_quote=stale_quote, current_quote=current_quote,
            client=clients, task=task, first_reported_at=_stamp(stamps[0]), last_reported_at=_stamp(stamps[-1]),
            report_count=len(reports), status="pending", reviewer_id=None, review_note=None, reviewed_at=None,
        )
        items[key] = QueueItem(
            claim=key, stale=stale, replacing=replacing,
            stale_quote=stale_quote, current_quote=current_quote,
            proposal=agent_proposal(report), origins=("agent",),
            details=(
                f"reported {len(reports)} time(s) by {clients}, recorded in the corpus database (tenant {tenants})",
                f"served as {newest.get('stale_source')} and {newest.get('replacing_source')}",
            )
            + ((f"task: {task}",) if task else ()),
        )
    notes = [f"corpus database: {len(rows)} agent report(s) read, {sum(len(k) for k in groups.values())} about memos here"]
    if len(rows) >= MAX_STALE_REPORTS:
        notes.append(f"corpus database: only the newest {MAX_STALE_REPORTS} reports were read; older ones are not shown")
    if decided:
        notes.append(f"corpus database: {decided} report(s) about claims already decided")
    if elsewhere:
        notes.append(f"corpus database: {elsewhere} report(s) name a memo this folder does not hold")
    if unquoted:
        notes.append(f"corpus database: {unquoted} report(s) quote text that is not verbatim in this folder's memos")
    return items, notes


def pending_reports(root: Path) -> list[StaleReport]:
    path = default_queue_path(root)
    if not path.exists():
        return []
    with StaleReportQueue(path) as queue:
        return queue.list("pending")


def build_queue(
    root: Path,
    *,
    arbiter_proposals: Callable[[Path], tuple[InferenceProposal, ...]] | None = None,
    database_reports: Callable[[], Sequence[Mapping[str, object]]] | None = None,
) -> Queue:
    """Every pending claim, from every source, minus rejected and already declared ones.

    `database_reports` returns the `stale_report` rows of a connected corpus database
    (`recall.dashboard.db.stale_reports`); without it the queue is this folder's alone.
    """
    root = root.resolve()
    notes: list[str] = []
    by_claim: dict[str, QueueItem] = {}

    for report in pending_reports(root):
        by_claim[report.claim_key] = QueueItem(
            claim=report.claim_key,
            stale=report.stale_source,
            replacing=report.replacing_source,
            stale_quote=report.stale_quote,
            current_quote=report.current_quote,
            proposal=agent_proposal(report),
            origins=("agent",),
            details=(f"reported {report.report_count} time(s) by {report.client}",)
            + ((f"task: {report.task}",) if report.task else ()),
        )

    if database_reports is not None:
        try:
            rows = database_reports()
        except DatabaseUnavailable as exc:
            notes.append(f"corpus database: unreachable, so only this folder's reports are shown ({exc})")
        else:
            rejected_now = rejected_claims(root)
            found, found_notes = database_items(
                root, rows, closed=lambda key, stale, replacing: key in rejected_now or already_declared(root, stale, replacing)
            )
            notes.extend(found_notes)
            for key, item in found.items():
                current = by_claim.get(key)
                if current is None:
                    by_claim[key] = item
                    continue
                # The same reports may be in both, when the server that took them could also read
                # this folder; the item keeps the sidecar's identity so a decision closes that row.
                by_claim[key] = QueueItem(
                    claim=current.claim, stale=current.stale, replacing=current.replacing,
                    stale_quote=current.stale_quote, current_quote=current.current_quote,
                    proposal=current.proposal, origins=current.origins,
                    details=current.details + item.details[:1],
                )

    summaries: list[str] = []
    fetch = arbiter_proposals or (
        lambda r: corpus_proposals(r, on_arbiter_run=lambda run: summaries.append(run.summary()), cache_only=True)
    )
    try:
        proposals = fetch(root)
    except RewriteRefused as exc:
        notes.append(f"arbiter: {exc}")
        proposals = ()
    notes.extend(summaries)
    for proposal in proposals:
        if proposal.proposed_relation != "supersedes":
            continue
        key = claim_key("supersedes", proposal.subject_id, proposal.object_id)
        detail = proposal.explanation
        if key in by_claim:
            current = by_claim[key]
            by_claim[key] = QueueItem(
                claim=current.claim,
                stale=current.stale,
                replacing=current.replacing,
                stale_quote=current.stale_quote,
                current_quote=current.current_quote,
                proposal=current.proposal,
                origins=current.origins + ("arbiter",),
                details=current.details + (detail,),
            )
            continue
        by_claim[key] = QueueItem(
            claim=key,
            stale=proposal.subject_id,
            replacing=proposal.object_id,
            stale_quote=str(proposal.metadata.get("quote_older", "")),
            current_quote=str(proposal.metadata.get("quote_newer", "")),
            proposal=proposal,
            origins=("arbiter",),
            details=(detail,),
        )

    rejected = rejected_claims(root)
    items = tuple(
        item
        for key, item in sorted(by_claim.items())
        if key not in rejected and not already_declared(root, item.stale, item.replacing)
    )
    return Queue(items=items, notes=tuple(notes))


def memo_digest(root: Path, name: str) -> str:
    """The hash of the memo a decision would edit, as the page showed it."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ReviewRefused(f"{name!r} is outside the corpus")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReviewRefused(f"{name!r} cannot be read: {exc}") from exc


def _promoted(item: QueueItem, reviewer: str, note: str, now: datetime) -> PromotedFact:
    if not reviewer.strip() or not note.strip():
        raise ReviewRefused("a decision needs your name and a note")
    try:
        reviewed = review_proposal(item.proposal, reviewer_id=reviewer, reviewed_at=now, audit_note=note)
        return promote_accepted_proposal(accept_reviewed_proposal(reviewed), promoted_at=now)
    except ValueError as exc:
        raise ReviewRefused(str(exc)) from exc


def preview(root: Path, item: QueueItem, now: datetime) -> RewritePlan:
    """The edit an accept would make. Writes nothing; the identity used here is never recorded."""
    fact = _promoted(item, "preview", "preview only, never written", now)
    try:
        return plan_rewrite(root, fact)
    except RewriteRefused as exc:
        raise ReviewRefused(str(exc)) from exc


def accept(root: Path, item: QueueItem, *, reviewer: str, note: str, shown_digest: str, now: datetime) -> RewritePlan:
    """Write the declaration, if the memo is still the one the reviewer saw."""
    root = root.resolve()
    fact = _promoted(item, reviewer, note, now)
    with _DECISION_LOCK:
        plan = preview(root, item, now)
        if memo_digest(root, plan.edit_file) != shown_digest:
            raise ReviewRefused(f"{plan.edit_file} changed since you reviewed it; review the new version")
        try:
            with RejectionLedger(default_ledger_path(root)) as ledger:
                result = apply_rewrite(root, fact, ledger=ledger, apply=True)
        except RewriteRefused as exc:
            raise ReviewRefused(str(exc)) from exc
        if not result.written:
            raise ReviewRefused(f"not written: {result.refusal}")
        _close_report(root, item.claim, "accepted", reviewer, note, now)
        return plan


def reject(root: Path, item: QueueItem, *, reviewer: str, note: str, now: datetime) -> None:
    """Record the claim as declined, so neither source shows it again."""
    root = root.resolve()
    if not reviewer.strip() or not note.strip():
        raise ReviewRefused("a decision needs your name and a note")
    with _DECISION_LOCK:
        try:
            with RejectionLedger(default_ledger_path(root)) as ledger:
                ledger.reject(item.claim, reviewer_id=reviewer, reason=note, rejected_at=now)
        except RewriteRefused as exc:
            raise ReviewRefused(str(exc)) from exc
        _close_report(root, item.claim, "rejected", reviewer, note, now)


def _close_report(root: Path, claim: str, status: str, reviewer: str, note: str, now: datetime) -> None:
    path = default_queue_path(root)
    if not path.exists():
        return
    try:
        with StaleReportQueue(path) as queue:
            report = queue.get(claim)
            if report is not None and report.status == "pending":
                queue.mark_reviewed(claim, status=status, reviewer_id=reviewer, note=note, reviewed_at=now)
    except sqlite3.Error as exc:
        raise ReviewRefused(f"the decision was recorded but the report queue could not be updated: {exc}") from exc
