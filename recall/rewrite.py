"""Write a reviewed promotion back into the corpus file it came from.

`recall/promotion.py` has held a complete `proposed → reviewed → accepted → promoted` machine for
some time with no caller outside its own tests. A gate nothing passes through protects nothing;
this module is its first caller, and the reason the gate exists.

**What may be written, and where.** `recall/frontmatter.py` recognises exactly three keys, and the
trust layer acts on those and nothing else. This module writes into that block and adds no key to
it — `FRONTMATTER_KEYS` *is* `VALIDITY_KEYS`, imported rather than restated, so the set cannot
drift by someone editing this file. Relations the frontmatter has no vocabulary for
(`contradicts`, `same_entity`, and a `status` no relation emits yet) go into the derived block: a
delimited, machine-owned region appended to the body, which `parse_frontmatter` sees as ordinary
text and the trust layer therefore never mistakes for authored metadata. Inventing
`contradicts:` as a fourth frontmatter key would have made the trust layer's input surface bigger
without making the trust layer able to act on it.

**Direction.** For `supersedes`, `subject_id` is the SUPERSEDED document and `object_id` the
superseding one — the orientation `recall/reasoning_graph.py` builds `authored_supersedes` with
(`from_file=target`, `to_file=superseding`) and `recall/reasoning_proposals/_deterministic.py`
mirrors. The schema has no `superseded_by`, so the key lands on `object_id` and names
`subject_id`. Inverting that declares the live memo stale and demotes it beneath the memo it
replaced, which is the exact failure the trust layer exists to prevent.

**Why `PromotedFact` and not `InferenceProposal`.** `apply_rewrite` is typed to accept only the
former. A reviewer reading a call site cannot tell whether an `InferenceProposal` argument passed
review; a `PromotedFact` argument they can, because only `promote_accepted_proposal` produces one
and it refuses without a named reviewer, a timestamp, an audit note, evidence ids and provider
identity. `metadata_is_trusted` re-checks all of it here, before any byte is written, for callers
that are not type-checked.

**Dry run by default.** `apply=False` unless asked, for the reason `recall lint --fix` gives: a
tool that rewrites your memory the first time you try it has earned distrust.

**Rejections live in a SQLite sidecar, not in the memo.** A proposal a human has declined must
stay declined, or the tool's output has to be filtered by hand every run and has saved nobody any
work. It is not written into the corpus file because rejections grow without bound and are noise
in a document a person reads. The key is the CLAIM — relation plus the two normalised document
names — and deliberately not `proposal.id`, which hashes in `generation_id` and `pipeline_id` and
would therefore forget every rejection at the next re-index.

**No Postgres migration.** The head is 0013 and this adds nothing to it, on three grounds that
have to hold together:

* claims are content keyed and generation independent, while every table since migration `0008`
  is generation scoped by design — a claim stored there would be re-proposed under the next
  generation id, which is precisely the resurfacing this ledger exists to stop;
* extraction runs on the ingest side, and `recall index` refuses to run under
  `RECALL_ENV=production`, so the one deployment where Postgres is the system of record is the one
  this code never executes in;
* the outcome that matters — the applied edit — is already durable in the file itself. Only the
  *refusals* need somewhere to live.

⚠️ **The deferred cost, stated rather than discovered:** a rejection recorded on one machine does
not travel. Two people reviewing the same corpus each decline the same bad proposal separately,
and a fresh clone starts with an empty ledger. That is a real limitation of the sidecar and the
thing to revisit if this ever runs somewhere with more than one reviewer.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from recall.atomic_write import atomic_write_bytes
from recall.frontmatter import VALIDITY_KEYS, parse_frontmatter, supersedes_key
from recall.lineage import canonical_sha256
from recall.observability import get_logger
from recall.promotion import PromotedFact
from recall.trust import metadata_is_trusted

_log = get_logger("rewrite")

#: Imported, never restated. "No new frontmatter keys" is only a property if this module cannot
#: name one that `recall/frontmatter.py` does not already recognise.
FRONTMATTER_KEYS: tuple[str, ...] = VALIDITY_KEYS

#: Everything the frontmatter has no vocabulary for. `status` is routable but no relation emits it
#: yet: the routing table is the design, and leaving a hole in it invites a fourth frontmatter key.
DERIVED_KEYS: tuple[str, ...] = ("contradicts", "same_entity", "status")

DERIVED_OPEN = "<!-- recall:derived -->"
DERIVED_CLOSE = "<!-- /recall:derived -->"

Destination = Literal["frontmatter", "derived"]

_BOM = b"\xef\xbb\xbf"
_LEDGER_DIR = ".recall"
_LEDGER_NAME = "rejections.sqlite3"


class RewriteRefused(ValueError):
    """A rewrite was asked for that this module will not perform."""


def destination(key: str) -> Destination:
    """Where `key` is written, or a refusal if it is written nowhere.

    The refusal is the interesting half. `superseded_by` is the key a reader keeps reaching for
    and the schema has never had; routing it anywhere would be inventing corpus vocabulary.
    """
    if key in FRONTMATTER_KEYS:
        return "frontmatter"
    if key in DERIVED_KEYS:
        return "derived"
    raise RewriteRefused(
        f"unknown_key: {key!r} is neither a recognised frontmatter key {FRONTMATTER_KEYS} "
        f"nor a derived-block key {DERIVED_KEYS}"
    )


def claim_key(relation: str, subject_id: str, object_id: str) -> str:
    """The generation-independent identity of a claim, used to key rejections.

    `supersedes_key` normalises the two document references so `old.md`, `old` and `[[old]]` are
    one claim rather than three — a rejection a human made against one spelling has been made
    against all of them.
    """
    return "claim_" + canonical_sha256(
        {
            "relation": relation,
            "subject": supersedes_key(subject_id),
            "object": supersedes_key(object_id),
        }
    )[:32]


def default_ledger_path(root: Path) -> Path:
    """Where the sidecar lives for a corpus: `<root>/.recall/rejections.sqlite3`.

    Beside the corpus rather than in it — a dotted directory the corpus glob does not match, so a
    growing binary never becomes a memo, and so a corpus moved to another machine leaves its
    rejections behind visibly rather than silently.
    """
    return root / _LEDGER_DIR / _LEDGER_NAME


class RejectionLedger:
    """Durable record of the claims a human has declined, so they stay declined."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # Opened into a local and only adopted once the schema is there. A constructor that
        # raises after connecting leaves a handle no `__exit__` can ever close, because
        # `with RejectionLedger(p) as l:` never reaches `__enter__` when `__init__` raises.
        try:
            connection = sqlite3.connect(str(self._path), timeout=30.0)
        except sqlite3.Error as exc:
            raise RewriteRefused(
                f"rejection ledger at {self._path} could not be opened: {exc}"
            ) from exc
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rejected_claims ("
                "claim_key TEXT PRIMARY KEY, reviewer_id TEXT NOT NULL, "
                "reason TEXT NOT NULL, rejected_at TEXT NOT NULL)"
            )
            connection.commit()
        except BaseException as exc:
            connection.close()
            if isinstance(exc, sqlite3.Error):
                raise RewriteRefused(
                    f"rejection ledger at {self._path} is not usable: {exc}"
                ) from exc
            raise
        self._conn = connection

    def reject(
        self, claim: str, *, reviewer_id: str, reason: str, rejected_at: datetime
    ) -> None:
        """Record a declined claim. A rejection without a named human and a reason is refused.

        The same rule as promotion, for the same reason: an unattributed entry here silences a
        proposal forever and nobody can later find out who decided that, or why.
        """
        if not reviewer_id.strip():
            raise RewriteRefused("reviewer_id is required to reject a claim")
        if not reason.strip():
            raise RewriteRefused("reason is required to reject a claim")
        self._conn.execute(
            "INSERT OR REPLACE INTO rejected_claims "
            "(claim_key, reviewer_id, reason, rejected_at) VALUES (?, ?, ?, ?)",
            (claim, reviewer_id, reason, rejected_at.isoformat()),
        )
        self._conn.commit()

    def is_rejected(self, claim: str) -> bool:
        try:
            row = self._conn.execute(
                "SELECT 1 FROM rejected_claims WHERE claim_key = ?", (claim,)
            ).fetchone()
        except sqlite3.Error as exc:
            # A locked, closed or externally-truncated ledger must not surface as a sqlite type
            # `apply_rewrite`'s docstring never promised. Refusing is the safe reading anyway:
            # a ledger that cannot be consulted might hold a rejection for this very claim.
            raise RewriteRefused(f"rejection ledger at {self._path} could not be read: {exc}") from exc
        return row is not None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RejectionLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class RewritePlan:
    """Exactly what would be written, in enough detail to review without reading the file."""

    edit_file: str          # root-relative path of the memo that gains the key
    key: str                # `supersedes` | `contradicts` | `same_entity` | ...
    value: str              # the other document, written verbatim as the fact named it
    block: Destination
    claim: str              # the ledger key for this claim
    fact_id: str
    proposal_id: str
    reviewer_id: str


@dataclass(frozen=True)
class RewriteResult:
    """What happened. `plan` is populated even on a dry run, and even on a reported refusal."""

    plan: RewritePlan | None
    written: bool
    refusal: str | None = None


def _refuse_untrusted(fact: object) -> PromotedFact:
    """The validator, run before the corpus is even looked at.

    `metadata_is_trusted` is the single definition of "reviewed enough to touch the corpus" and
    this is its first caller outside tests. It answers a bool, which is the right shape for a
    guard and the wrong shape for a message, so each field is named here first — a refusal that
    says only "not trusted" sends a reviewer to read the dataclass.
    """
    if not isinstance(fact, PromotedFact):
        raise RewriteRefused(
            f"refusing to write: expected a reviewed PromotedFact, got "
            f"{type(fact).__name__} — an unreviewed proposal never reaches corpus metadata"
        )
    if fact.state != "promoted":
        raise RewriteRefused(f"refusing to write: state is {fact.state!r}, not 'promoted'")
    if not fact.reviewer_id.strip():
        raise RewriteRefused("refusing to write: reviewer_id is empty — no named human reviewed this")
    if not fact.audit_note.strip():
        raise RewriteRefused("refusing to write: audit_note is empty — the review left no evidence")
    if not fact.proposal_evidence_ids:
        raise RewriteRefused("refusing to write: proposal_evidence_ids is empty")
    for field in ("source_generation_id", "source_provider_id", "source_model_id",
                  "source_model_revision"):
        if not str(getattr(fact, field)).strip():
            raise RewriteRefused(f"refusing to write: {field} is empty")
    if not metadata_is_trusted(fact):
        # Unreachable through the checks above today, and deliberately kept: `metadata_is_trusted`
        # is the authority, and if it grows a condition this module has not enumerated, the write
        # must stop rather than proceed on a stale list of fields.
        raise RewriteRefused("refusing to write: metadata_is_trusted() rejected this fact")
    return fact


def _refuse_multiline(value: str) -> None:
    """A written value must occupy exactly one line, because a line is all either block can hold.

    `key: value` is formatted and inserted verbatim, so a newline inside `value` does not corrupt
    the file — it writes a SECOND key into the block, which is strictly worse. `valid_until:
    1999-01-01` is the version that matters: nothing looks broken, the memo is silently expired,
    and the trust layer then does exactly what it was told. Neither `subject_id` nor `object_id`
    originates in this package (`_providers.py` builds them with a bare `str()` over a provider's
    raw JSON), so this is the boundary where that has to be checked. `key` needs no such check —
    it comes from `destination`, which is a closed set.
    """
    if any(ch in value for ch in ("\n", "\r", "\x00")):
        raise RewriteRefused(
            f"refusing to write: {value!r} is not a single line — a newline in a written value "
            f"declares an additional frontmatter key, which is how a live memo gets expired"
        )


def _readable_text(raw: bytes, rel: str) -> str:
    """The memo decoded, or a refusal naming it.

    Both blocks are written as UTF-8, so a file that is not UTF-8 text cannot be edited: appending
    UTF-8 bytes to a UTF-16 memo (Notepad's "Unicode") turns its tail into mojibake, and letting
    the frontmatter path raise a bare `UnicodeDecodeError` hands a batch caller an exception type
    this module's docstring never promised. NUL is checked separately because UTF-16 ASCII decodes
    as valid UTF-8 — every other byte is a NUL — and would otherwise pass.
    """
    if b"\x00" in raw:
        raise RewriteRefused(
            f"refusing to write: {rel} contains NUL bytes, so it is not a UTF-8 text memo"
        )
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RewriteRefused(f"refusing to write: {rel} is not valid UTF-8 ({exc.reason})") from exc


def _resolve(root: Path, ref: str) -> str:
    """The one corpus file `ref` names, root-relative. Zero or several matches is a refusal."""
    key = supersedes_key(ref)
    matches = sorted(
        f.relative_to(root).as_posix() for f in root.rglob("*.md") if supersedes_key(f.name) == key
    )
    if len(matches) != 1:
        raise RewriteRefused(
            f"refusing to write: {ref!r} matches {len(matches)} files in the corpus, not one"
        )
    return matches[0]


def plan_rewrite(root: Path, fact: PromotedFact) -> RewritePlan:
    """What `apply_rewrite` would do, with no side effect of any kind."""
    checked = _refuse_untrusted(fact)
    block = destination(checked.relation)
    # `supersedes` is the asymmetric one: the key goes on the SUPERSEDING memo (`object_id`) and
    # names the superseded one. The derived relations are stated by their subject.
    if checked.relation == "supersedes":
        edit_ref, value = checked.object_id, checked.subject_id
    else:
        edit_ref, value = checked.subject_id, checked.object_id
    _refuse_multiline(value)
    return RewritePlan(
        edit_file=_resolve(root, edit_ref),
        key=checked.relation,
        value=value,
        block=block,
        claim=claim_key(checked.relation, checked.subject_id, checked.object_id),
        fact_id=checked.fact_id,
        proposal_id=checked.proposal_id,
        reviewer_id=checked.reviewer_id,
    )


def apply_rewrite(
    root: Path,
    fact: PromotedFact,
    *,
    ledger: RejectionLedger | None = None,
    apply: bool = False,
) -> RewriteResult:
    """Declare `fact` in the corpus file it belongs to, or report why it was not.

    Two kinds of "no" are distinguished on purpose. A `RewriteRefused` is a caller error — an
    unreviewed fact, an unwritable key, an unresolvable reference — and raises. A memo that
    already declares the key, or a claim a human has already rejected, is a normal outcome of a
    batch run and comes back as a populated `RewriteResult` with `written=False` and a reason.
    Neither is silent: a refusal a caller cannot see is indistinguishable from a write.

    ⚠️ **Single writer.** The read, the already-declares check and the swap are three steps with
    no lock between them. `atomic_write_bytes` guarantees the file is never half written, but two
    concurrent calls against the same memo are last-writer-wins: both read a memo with no key,
    both compute an insertion, and the second swap discards the first — while both return
    `written=True`. `root` must also be a directory; a single-file root resolves to nothing and is
    refused, which differs from `fix.apply_proposal`, and is worth knowing if the two are ever
    driven from one call site.
    """
    plan = plan_rewrite(root, fact)
    if ledger is not None and ledger.is_rejected(plan.claim):
        return RewriteResult(plan, False, f"claim {plan.claim} was rejected by a reviewer")

    path = root / plan.edit_file
    raw = path.read_bytes()
    text = _readable_text(raw, plan.edit_file)
    if plan.block == "frontmatter":
        meta, _ = parse_frontmatter(text)
        if plan.key in meta:
            # PRESENCE, not truthiness. `supersedes:` with nothing after it is a human writing
            # "this supersedes nothing"; reading that as absence appends a second line and leaves
            # the block holding two copies of a single-valued key.
            return RewriteResult(
                plan, False, f"{plan.edit_file} already declares {plan.key}: {meta[plan.key]!r}"
            )
        updated = _insert_frontmatter_line(raw, plan.key, plan.value)
    else:
        derived = _upsert_derived_entry(raw, plan.key, plan.value)
        if derived is None:
            return RewriteResult(
                plan, False, f"{plan.edit_file} already declares {plan.key}: {plan.value!r}"
            )
        updated = derived

    if not apply:
        # Dry run by DEFAULT: this edits the user's own documents, and a tool that rewrites your
        # memory the first time you try it has earned distrust.
        return RewriteResult(plan, False, None)

    atomic_write_bytes(path, updated)
    _log.info(
        "declared %s: %s in %s (fact %s, reviewer %s)",
        plan.key, plan.value, plan.edit_file, plan.fact_id, plan.reviewer_id,
    )
    return RewriteResult(plan, True, None)


# --- byte-level editing ---------------------------------------------------------------------------
#
# Everything below works on bytes and rejoins the file's own line terminators. Decoding to `str`,
# splitting on "\n" and re-encoding is how `recall/fix.py` eats a Windows memo's BOM and leaves one
# LF-only line inside a CRLF file: both are invisible in an editor and present in every diff.


def _newline(raw: bytes) -> bytes:
    """The file's line terminator, taken from its first line rather than from the platform."""
    index = raw.find(b"\n")
    if index == -1:
        return b"\n"
    return b"\r\n" if raw[index - 1 : index] == b"\r" else b"\n"


def _lines(data: bytes) -> list[bytes]:
    """Split on LF only, keeping terminators — the same line boundary `parse_frontmatter` uses.

    Deliberately NOT `bytes.splitlines`, which also splits on a lone CR. `parse_frontmatter`
    splits on ``"\\n"`` alone, so a CR-terminated file has no frontmatter as far as recall is
    concerned. A writer that disagreed would find an opening ``---``, splice the key into a block
    the parser cannot see, and — because the parser still reports nothing declared — do it again
    on every subsequent run. Two parsers with two ideas of a line is the defect; sharing one is
    the fix.

    `keepends` is the other half: rejoining with ``b"".join`` reproduces the input exactly.
    """
    parts = data.split(b"\n")
    lines = [part + b"\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])  # a final line with no trailing newline
    return lines


def _terminator(line: bytes, default: bytes) -> bytes:
    """A line's own ending, so an insertion beside it matches it exactly."""
    if line.endswith(b"\r\n"):
        return b"\r\n"
    if line.endswith(b"\n"):
        return b"\n"
    return default  # the final line of a file with no trailing newline


def _insert_frontmatter_line(raw: bytes, key: str, value: str) -> bytes:
    """`key: value` into the frontmatter block, adding one if the file has none.

    The BOM is carried across untouched rather than decoded away, and the inserted line borrows
    the closing fence's terminator, so a CRLF memo gains a CRLF line.

    The search for the closing fence is unbounded, and deliberately so. A memo whose body happens
    to contain a `---` thematic break within what looks like a block gets the key written next to
    prose, which is startling — but `parse_frontmatter` makes exactly the same reading, and a
    writer that disagreed with the parser would put the key somewhere the trust layer cannot see
    it and rewrite it again on every run. `parse_frontmatter` defines what the block is; this
    follows it. Tighten both together or neither.
    """
    bom = _BOM if raw.startswith(_BOM) else b""
    body = raw[len(bom) :]
    newline = _newline(body)
    entry = f"{key}: {value}".encode("utf-8")
    lines = _lines(body)
    if lines and lines[0].strip() == b"---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == b"---":
                lines.insert(index, entry + _terminator(line, newline))
                return bom + b"".join(lines)
        # unclosed block — treat as no frontmatter rather than corrupt it further
    return bom + b"---" + newline + entry + newline + b"---" + newline + body


def _upsert_derived_entry(raw: bytes, key: str, value: str) -> bytes | None:
    """`key: value` into the derived block, or ``None`` when it is already there.

    One block per file. A second block would make the file's derived state ambiguous and leave no
    way to strip the machine's annotations back off. `key: value` is the identity, not `key` alone
    — a memo may legitimately contradict more than one other memo.
    """
    bom = _BOM if raw.startswith(_BOM) else b""
    body = raw[len(bom) :]
    newline = _newline(body)
    entry = f"{key}: {value}".encode("utf-8")
    open_marker, close_marker = DERIVED_OPEN.encode("utf-8"), DERIVED_CLOSE.encode("utf-8")
    lines = _lines(body)

    opened: int | None = None
    closed: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == open_marker and opened is None:
            opened = index
        elif stripped == close_marker and opened is not None and closed is None:
            closed = index
    if opened is not None and closed is not None:
        if any(line.strip() == entry for line in lines[opened + 1 : closed]):
            return None
        lines.insert(closed, entry + _terminator(lines[closed], newline))
        return bom + b"".join(lines)
    if opened is not None:
        # Falling through to the append branch would write a SECOND opening marker past the first
        # and duplicate the entry the dedup above exists to catch — producing exactly the
        # ambiguity the single-block rule prevents. A half-open machine-owned region is something
        # a human has to look at, not something to append past.
        raise RewriteRefused(
            "refusing to write: the derived block is unclosed — "
            f"{DERIVED_OPEN} has no matching {DERIVED_CLOSE}"
        )

    tail = body if (not body or body.endswith(b"\n")) else body + newline
    return (
        bom + tail
        + open_marker + newline
        + entry + newline
        + close_marker + newline
    )


__all__ = [
    "DERIVED_CLOSE",
    "DERIVED_KEYS",
    "DERIVED_OPEN",
    "FRONTMATTER_KEYS",
    "RejectionLedger",
    "RewritePlan",
    "RewriteRefused",
    "RewriteResult",
    "apply_rewrite",
    "claim_key",
    "default_ledger_path",
    "destination",
    "plan_rewrite",
]
