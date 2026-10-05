"""The review queue behind `recall dashboard`: what is pending, and the two decisions a person makes.

Two sources feed it, both filesystem only, so the page works on a corpus with no database behind it:

* agent reports from `recall_report_stale`, read from `<root>/.recall/reports.sqlite3`;
* supersession proposals from the model arbiter, read CACHE-ONLY (`corpus_proposals(cache_only=True)`),
  so opening the page never spends anything.

Both are keyed by `rewrite.claim_key`, so one claim reported by an agent and proposed by the arbiter
is one row. A claim a person rejected, or one the newer memo already declares, is not shown.

Accepting runs the same chain as `recall rewrite apply`: `review_proposal`,
`accept_reviewed_proposal`, `promote_accepted_proposal`, then `apply_rewrite`, which writes
`supersedes:` into the newer memo. The page shows the planned edit first and sends back a hash of
the memo it showed; if the file changed in between, the accept is refused and the reviewer sees the
new plan. Rejecting records the claim in the rejection ledger, which both sources honour.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from recall.document import parse_document
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
from recall.stale_reports import StaleReport, StaleReportQueue, default_queue_path

AGENT_RULE_ID = "agent_stale_report"
AGENT_PROVIDER_ID = "recall_report_stale"
AGENT_PROVIDER_REVISION = "stale_reports.v1"

#: One decision at a time per process. `apply_rewrite` takes no lock, and two writes to one memo are
#: last writer wins; the memo hash below catches a writer outside this process.
_DECISION_LOCK = threading.Lock()


class ReviewRefused(ValueError):
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
) -> Queue:
    """Every pending claim, from both sources, minus rejected and already declared ones."""
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
