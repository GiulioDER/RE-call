"""Write a REVIEWED promotion back into the corpus file it is about.

`promotion.py` has held a complete `proposed → reviewed → accepted → promoted` machine since
session 3, with no callers outside its own tests. This module is its first one. That ordering is
the point: the review machine was built before anything could act on its output, so nothing could
ever reach a corpus file by skipping it. Keeping that true is this module's only real job.

**The type is the guard.** `apply_rewrite` accepts a `PromotedFact` and nothing else. An
`InferenceProposal` — the raw, unreviewed thing a model emits — is not merely discouraged here,
it is unrepresentable: there is no code path that converts one without going through
`review_proposal` → `accept_reviewed_proposal` → `promote_accepted_proposal`, each of which
demands a named reviewer, a timestamp and an audit note. A convention would have been enough
right up until the first person in a hurry; a signature is enforced by the type checker on every
future call site, including the ones nobody reviews.

**Two destinations, and the split is not stylistic.** `parse_frontmatter` recognises exactly
three keys (`frontmatter.py:12`) and silently discards every other line in the block. So a
`contradicts:` line written into the frontmatter would be invisible to retrieval and perfectly
legible to a human reading the file — which is the worst of both, a relation that looks declared
and acts undeclared. Relations the schema cannot act on go into a delimited derived block in the
body instead, where they are honestly marked as machine-written and are excluded from
`human_body`, the extractor's input.

**Nothing here decides whether a relation is true.** By the time a `PromotedFact` exists a human
has already said so, on the record. This module decides only *where the bytes go*, and refuses
whenever the answer is not unambiguous.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from recall.atomic_write import atomic_write_bytes
from recall.frontmatter import parse_frontmatter, supersedes_key
from recall.lint import DEFAULT_GLOB
from recall.observability import get_logger
from recall.promotion import PromotedFact
from recall.trust import metadata_is_trusted

_log = get_logger("rewrite")

Destination = Literal["frontmatter", "derived"]

#: Relations the trust layer can act on, and which therefore belong in the frontmatter block.
#: This set is not extensible from here: it is `frontmatter.VALIDITY_KEYS`, and adding to it
#: without teaching the parser would write a key that reads as declared and behaves as absent.
FRONTMATTER_RELATIONS: dict[str, str] = {
    "supersedes": "supersedes",
    "valid_from": "valid_from",
    "valid_until": "valid_until",
}

#: Relations that are real, reviewed, and unrepresentable in a three-key schema. They are
#: recorded where a human can see them and a parser will not mistake them for validity metadata.
DERIVED_RELATIONS: dict[str, str] = {
    "contradicts": "contradicts",
    "same_entity": "same_entity",
    "status": "status",
}

DERIVED_BEGIN = "<!-- recall:derived -->"
DERIVED_END = "<!-- /recall:derived -->"

_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class RewritePlan:
    """Where one promoted fact would land. Pure: computing it touches no file."""

    edit_file: str
    key: str
    value: str
    destination: Destination
    fact_id: str
    proposal_id: str


@dataclass(frozen=True)
class RewriteResult:
    """What `apply_rewrite` did, and when it declined, why.

    `planned` and `written` are separate because a dry run must still be able to report a real
    proposal, and because "refused" and "would have, but you did not pass --apply" are different
    answers that a single boolean would merge into one.
    """

    plan: RewritePlan | None
    planned: bool
    written: bool
    skipped_reason: str | None = None


# --- validation --------------------------------------------------------------------------------


def _untrusted_reason(fact: object) -> str:
    """Name the specific field that makes `fact` untrusted.

    `metadata_is_trusted` returns a bare bool, which is the right shape for a guard and the wrong
    shape for an error message: "refused" without "because your audit_note is blank" sends the
    caller to read this module's source. The checks below mirror
    `reviewed_promotion_is_trusted_metadata` (`promotion.py:184`) field for field, and the final
    fallback exists because that function is the authority — if it grows a condition this list
    has not learned yet, the refusal must still hold rather than fall through to a write.
    """
    if not isinstance(fact, PromotedFact):
        return f"apply_rewrite requires a reviewed PromotedFact, got {type(fact).__name__}"
    if fact.state != "promoted":
        return f"refusing a fact whose state is {fact.state!r}, not 'promoted'"
    if not fact.reviewer_id.strip():
        return "refusing a fact with no reviewer_id: no proposal reaches a corpus file unattributed"
    if not fact.audit_note.strip():
        return "refusing a fact with a blank audit_note: the reason for the edit is part of it"
    if not fact.proposal_evidence_ids:
        return "refusing a fact with empty proposal_evidence_ids: nothing supports it"
    for field_name in ("source_generation_id", "source_provider_id", "source_model_id",
                       "source_model_revision"):
        if not str(getattr(fact, field_name)).strip():
            return f"refusing a fact with an empty {field_name}: its origin is unauditable"
    return "refusing a fact that metadata_is_trusted rejected"


def _require_trusted(fact: PromotedFact) -> None:
    """The write site's gate, and `metadata_is_trusted`'s first non-test caller."""
    if not metadata_is_trusted(fact):
        raise ValueError(_untrusted_reason(fact))


# --- resolution --------------------------------------------------------------------------------


def _resolve(root: Path, name: str, glob: str) -> str:
    """The single corpus file `name` refers to, or a refusal.

    Ambiguity is refused rather than resolved by any tie-break. `fix.py` reached the same
    conclusion the expensive way: every looser rule it tried against the real corpus produced
    matches that could not be acted on, and a proposal a human must re-check is not a saving.
    """
    files = sorted(root.glob(glob)) if root.is_dir() else [root]
    key = supersedes_key(name)
    matches = [
        f.relative_to(root).as_posix() if root.is_dir() else f.name
        for f in files
        if supersedes_key(f.name) == key
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"{name!r} matches {len(matches)} files in the corpus; refusing to guess which"
        if matches
        else f"{name!r} is not a file in the corpus"
    )


def plan_rewrite(
    root: str | Path, fact: PromotedFact, *, glob: str = DEFAULT_GLOB
) -> RewritePlan:
    """Where this fact would be written. Reads the corpus listing; writes nothing."""
    root = Path(root)
    _require_trusted(fact)
    relation = fact.relation
    if relation in FRONTMATTER_RELATIONS:
        destination: Destination = "frontmatter"
        key = FRONTMATTER_RELATIONS[relation]
    elif relation in DERIVED_RELATIONS:
        destination = "derived"
        key = DERIVED_RELATIONS[relation]
    else:
        raise ValueError(
            f"no destination is defined for relation {relation!r}; known relations are "
            f"{sorted(FRONTMATTER_RELATIONS) + sorted(DERIVED_RELATIONS)}"
        )
    return RewritePlan(
        edit_file=_resolve(root, fact.subject_id, glob),
        key=key,
        # Resolved for BOTH destinations. A derived-block relation is not load-bearing for
        # retrieval, which is exactly why it would be tempting to skip the check — and why the
        # block would then fill up with names that resolve to nothing, the state `fix.py:62`
        # describes as output a human must themselves filter.
        value=_resolve(root, fact.object_id, glob),
        destination=destination,
        fact_id=fact.fact_id,
        proposal_id=fact.proposal_id,
    )


# --- byte-preserving editing ---------------------------------------------------------------------


def _newline(body: bytes) -> bytes:
    """The line ending this file already uses.

    A file is CRLF if it contains one. The alternative — following the platform, which is what
    text-mode ``open`` does — rewrites every line of a CRLF memo when inserting one line into it,
    and shows up in review as a whole-file diff with one real change hidden inside.
    """
    return b"\r\n" if b"\r\n" in body else b"\n"


def _insert_frontmatter_key(raw: bytes, key: str, value: str) -> bytes:
    """Add ``key: value`` to the frontmatter block, leaving every other byte alone."""
    bom = _BOM if raw.startswith(_BOM) else b""
    body = raw[len(bom):]
    nl = _newline(body)
    line = f"{key}: {value}".encode("utf-8") + nl
    lines = body.splitlines(keepends=True)

    if lines and lines[0].strip() == b"---":
        for i, ln in enumerate(lines[1:], start=1):
            if ln.strip() == b"---":
                return bom + b"".join(lines[:i]) + line + b"".join(lines[i:])
        # An unclosed block is not repaired here: treating the whole file as body and prepending
        # a fresh block is the same choice `fix.py` makes, and it is the conservative one — the
        # alternative is inventing a closing fence somewhere in the user's prose.
    return bom + b"---" + nl + line + b"---" + nl + body


def _insert_derived_line(raw: bytes, key: str, value: str) -> bytes:
    """Add ``key: value`` to the derived block, creating the block if it is absent."""
    bom = _BOM if raw.startswith(_BOM) else b""
    body = raw[len(bom):]
    nl = _newline(body)
    line = f"{key}: {value}".encode("utf-8") + nl
    begin, end = DERIVED_BEGIN.encode("utf-8"), DERIVED_END.encode("utf-8")

    start, idx = body.find(begin), body.find(end)
    # BEGIN must actually precede END. Requiring only that both exist meant a memo whose prose
    # merely QUOTES the closing marker got machine-written text spliced into the middle of the
    # author's sentence. That is not just cosmetic: it changes `human_body`, which changes the
    # claim cache key, which re-invokes the model on prose no human touched.
    if start != -1 and idx != -1 and start < idx:
        return bom + body[:idx] + line + body[idx:]
    prefix = body if body.endswith(nl) or not body else body + nl
    return bom + prefix + nl + begin + nl + line + end + nl


def _derived_records(text: str, line: str) -> bool:
    """Whether the derived BLOCK already carries exactly this line.

    Scoped to the block and matched whole-line, because the first version tested the raw
    substring against the whole file: an author quoting ``status: old.md`` anywhere in their
    prose silently suppressed a reviewed write and told the operator it was already recorded. A
    refusal that reports something untrue is worse than no refusal, because it ends the enquiry.
    """
    start, end = text.find(DERIVED_BEGIN), text.find(DERIVED_END)
    if start == -1 or end == -1 or end < start:
        return False
    block = text[start + len(DERIVED_BEGIN):end]
    return any(existing.strip() == line for existing in block.splitlines())


def _has_unclosed_frontmatter(text: str) -> bool:
    """True when the file opens a ``---`` block and never closes it.

    `parse_frontmatter` returns ``({}, text)`` for this case, which is indistinguishable from
    "no frontmatter at all" — and the two need opposite treatment: an absent block should be
    created, a broken one must not be papered over with a second block.
    """
    lines = text.split("\n")
    if not lines or lines[0].lstrip("﻿").strip() != "---":
        return False
    return not any(line.strip() == "---" for line in lines[1:])


def human_body(text: str) -> str:
    """The author's own prose: no frontmatter, no derived block, right-stripped.

    This is the extractor's prompt input and part of its cache key, which is why the derived
    block has to come out. If it did not, this system's own writes would change the text the
    model sees, so applying an accepted proposal would invalidate the cache entry for every memo
    it touched and re-invoke the model on prose no human had altered.
    """
    _meta, body = parse_frontmatter(text)
    begin, end = body.find(DERIVED_BEGIN), body.find(DERIVED_END)
    if begin != -1 and end != -1 and end > begin:
        body = body[:begin] + body[end + len(DERIVED_END):]
    return body.rstrip()


# --- the rejection ledger ------------------------------------------------------------------------


class RejectionLedger:
    """Durable record of proposals a human has already said no to.

    `fix.py:62` states the failure this prevents: "a proposal tool whose output must itself be
    filtered has not saved anyone any work." A reviewer who rejects a proposal and then sees it
    again next run has been given a worse tool than no tool, because now the rejection costs them
    something every time.

    **Why a SQLite sidecar and not the memo.** Rejections accumulate without bound and say
    nothing about what the memo means; a `rejected:` line per declined proposal would turn the
    document into a log of decisions about the document. The corpus holds what is true, the
    sidecar holds what has been asked and answered.

    ⚠️ **Deferred cost, stated rather than hidden: a rejection does not travel.** It is local to
    the machine that holds the sidecar file, so the same proposal can be declined once per
    reviewer per checkout. The alternative is a Postgres table, and that is refused for now on
    three counts: claims are content-keyed and generation-independent while the schema has been
    generation-scoped by design since 0008; extraction runs on the ingest side, and `recall index`
    is already refused under `RECALL_ENV=production` (`cli.py:1209`), so the one deployment where
    Postgres is the system of record is the one this never runs in; and the *applied* edit is
    already durable in the file itself, so what is at stake here is a re-ask, not data loss.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS rejections ("
            "  proposal_id TEXT PRIMARY KEY,"
            "  fact_id     TEXT NOT NULL,"
            "  reviewer_id TEXT NOT NULL,"
            "  note        TEXT NOT NULL,"
            "  rejected_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def reject(
        self, fact: PromotedFact, *, reviewer_id: str, note: str, at: datetime | None = None
    ) -> None:
        """Record that `reviewer_id` declined this proposal, and why.

        Keyed on `proposal_id`, which is a content hash: the same relation proposed again from
        the same evidence is the same proposal and stays rejected, while a genuinely different
        proposal about the same pair of files is a new question and gets asked.

        A rejection carries a reviewer and a note for the same reason a promotion does. An
        anonymous, unexplained "no" is not reviewable either, and it is the record that will be
        consulted when someone asks why an obvious-looking edge was never declared.
        """
        if not reviewer_id.strip():
            raise ValueError("a rejection requires a reviewer_id")
        if not note.strip():
            raise ValueError("a rejection requires a note")
        stamp = (at or datetime.now(timezone.utc)).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO rejections"
            " (proposal_id, fact_id, reviewer_id, note, rejected_at) VALUES (?, ?, ?, ?, ?)",
            (fact.proposal_id, fact.fact_id, reviewer_id, note, stamp),
        )
        self._conn.commit()

    def is_rejected(self, fact: PromotedFact) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM rejections WHERE proposal_id = ?", (fact.proposal_id,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RejectionLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- bridging the rule-based extractor into the review machine ------------------------------------

#: Generation identity for edges sourced from prose markers rather than a model. Spelled out as
#: constants for the same reason `entailment.py` pins its model revision: an artifact that cannot
#: say which version of which extractor produced it cannot be re-derived or blamed.
PROSE_PROVIDER_ID = "recall.fix"
PROSE_MODEL_ID = "prose-markers"
PROSE_PROVIDER_REVISION = "fix-v1"
PROSE_PIPELINE_ID = "corpus-rewrite"
PROSE_GENERATION_ID = "corpus"


def prose_edge_proposal_id(edit_file: str, target: str, evidence_file: str, evidence: str) -> str:
    """A content-addressed id for one prose-stated edge.

    Content-addressed rather than sequential because the rejection ledger is keyed on it and has
    to survive across separate processes: the *same* edge re-derived tomorrow must hash to the
    same id, or a reviewer's "no" silently expires and the proposal returns. Sequence numbers,
    timestamps and object identity all fail that test.

    This mirrors the discipline `_providers.py:86` enforces at the model boundary, where the
    library recomputes the id and raises if the provider's disagrees. The rule is the same here
    and the reason is the same: the extractor supplies semantics, the library computes identity.
    """
    digest = hashlib.sha256()
    for part in (PROSE_PROVIDER_ID, PROSE_MODEL_ID, PROSE_PROVIDER_REVISION, PROSE_PIPELINE_ID,
                 "supersedes", edit_file, target, evidence_file, evidence):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return f"ip_{digest.hexdigest()[:32]}"


def promoted_prose_edge(
    *,
    edit_file: str,
    target: str,
    evidence_file: str,
    evidence: str,
    reviewer_id: str,
    audit_note: str,
    at: datetime,
) -> PromotedFact:
    """Run one prose-stated edge through the real review machine and return the promoted fact.

    Deliberately *not* a shortcut around `promotion.py`. It builds the raw proposal and then
    calls `review_proposal` → `accept_reviewed_proposal` → `promote_accepted_proposal`, so every
    field those functions demand is demanded here too. Hand-assembling a `PromotedFact` would
    have been three lines shorter and would have created the one thing this design forbids: a
    promoted fact that never passed through review.

    The reviewer is a parameter with no default, and the caller above it makes `--reviewer` a
    required flag, so the "named human" is named at the only point where a human is present.
    """
    from recall.promotion import (
        accept_reviewed_proposal,
        promote_accepted_proposal,
        review_proposal,
    )
    from recall.reasoning_proposals import InferenceProposal

    proposal = InferenceProposal(
        id=prose_edge_proposal_id(edit_file, target, evidence_file, evidence),
        source_evidence_ids=(evidence_file,),
        proposed_relation="supersedes",
        subject_id=edit_file,
        object_id=target,
        explanation=f"{evidence_file} states {evidence!r}",
        model_id=PROSE_MODEL_ID,
        pipeline_id=PROSE_PIPELINE_ID,
        provider_id=PROSE_PROVIDER_ID,
        provider_revision=PROSE_PROVIDER_REVISION,
        confidence=None,
        uncertainty=(),
        generation_id=PROSE_GENERATION_ID,
        rule_id="prose-closure-marker",
    )
    review = review_proposal(
        proposal, reviewer_id=reviewer_id, reviewed_at=at, audit_note=audit_note
    )
    return promote_accepted_proposal(accept_reviewed_proposal(review), promoted_at=at)


# --- the write path ------------------------------------------------------------------------------


def apply_rewrite(
    root: str | Path,
    fact: PromotedFact,
    *,
    apply: bool = False,
    ledger: RejectionLedger | None = None,
    glob: str = DEFAULT_GLOB,
) -> RewriteResult:
    """Declare `fact` in the corpus under `root`. Dry run unless `apply=True`.

    The default is a dry run for the reason `cli.py:954` gives about the other writer — "a tool
    that rewrites your memory the first time you try it has earned distrust" — which applies with
    more force here, because this path writes relations a *model* proposed. The reviewer's yes is
    already on the record by now; the dry run is about the operator seeing the diff before the
    corpus changes under them.
    """
    root = Path(root)
    plan = plan_rewrite(root, fact, glob=glob)

    if ledger is not None and ledger.is_rejected(fact):
        return RewriteResult(plan, planned=False, written=False,
                             skipped_reason=f"proposal {plan.proposal_id} was rejected by a "
                                            "reviewer and will not be offered again")

    target = root / plan.edit_file if root.is_dir() else root
    raw = target.read_bytes()

    if plan.destination == "frontmatter":
        decoded = raw.decode("utf-8-sig", errors="replace")
        if _has_unclosed_frontmatter(decoded):
            # `parse_frontmatter` reports NO frontmatter for an unclosed block, so the
            # "already declares" guard below could not fire and the writer prepended a second
            # block. The file then stated two different predecessors, retrieval acted on the
            # newer one, and the result claimed success. Refusing is the only safe answer: the
            # alternative is inventing a closing fence somewhere in the author's prose.
            return RewriteResult(plan, planned=False, written=False,
                                 skipped_reason=f"{plan.edit_file} has an unclosed frontmatter "
                                                "block; refusing to write into a file whose "
                                                "metadata cannot be parsed")
        meta, _body = parse_frontmatter(decoded)
        if meta.get(plan.key):
            # `fix.py:264` refuses this too. A second writer that did not would make that refusal
            # decorative, since either tool can be the one that runs first.
            return RewriteResult(plan, planned=False, written=False,
                                 skipped_reason=f"{plan.edit_file} already declares "
                                                f"{plan.key}: {meta[plan.key]!r}")
        updated = _insert_frontmatter_key(raw, plan.key, plan.value)
    else:
        line = f"{plan.key}: {plan.value}"
        if _derived_records(raw.decode("utf-8-sig", errors="replace"), line):
            return RewriteResult(plan, planned=False, written=False,
                                 skipped_reason=f"{plan.edit_file} already records {line!r}")
        updated = _insert_derived_line(raw, plan.key, plan.value)

    if not apply:
        return RewriteResult(plan, planned=True, written=False)

    atomic_write_bytes(target, updated)
    _log.info(
        "declared %s: %s in %s (fact %s, reviewer %s)",
        plan.key, plan.value, plan.edit_file, plan.fact_id, fact.reviewer_id,
    )
    return RewriteResult(plan, planned=True, written=True)


__all__ = [
    "DERIVED_BEGIN",
    "DERIVED_END",
    "DERIVED_RELATIONS",
    "FRONTMATTER_RELATIONS",
    "RejectionLedger",
    "RewritePlan",
    "RewriteResult",
    "apply_rewrite",
    "human_body",
    "plan_rewrite",
]
