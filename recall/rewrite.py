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
from recall.frontmatter import (
    VALIDITY_KEYS,
    dominant_newline,
    insert_frontmatter_line,
    line_terminator,
    parse_frontmatter,
    split_bom,
    split_lines,
    supersedes_key,
)
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
        try:
            if self._path.parent and not self._path.parent.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Hardening the connect and leaving the mkdir above it raw meant a `.recall` name
            # already taken by a file — which `default_ledger_path` walks straight into — escaped
            # as a bare OSError instead of this module's refusal type.
            raise RewriteRefused(
                f"rejection ledger directory for {self._path} could not be created: {exc}"
            ) from exc
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
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO rejected_claims "
                "(claim_key, reviewer_id, reason, rejected_at) VALUES (?, ?, ?, ?)",
                (claim, reviewer_id, reason, rejected_at.isoformat()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            # Both halves of this class have to agree on their error contract. Translating only
            # `is_rejected` meant the same ledger in the same state raised two different exception
            # types depending on which way you touched it.
            try:
                # Translating the failure is not the same as undoing it. A COMMIT that fails
                # leaves the INSERT pending, and the next successful `reject` commits both — so a
                # rejection the reviewer was told was refused becomes durable anyway, or not,
                # depending on whether the process exits first.
                self._conn.rollback()
            except sqlite3.Error:
                pass
            raise RewriteRefused(
                f"rejection ledger at {self._path} could not be written: {exc}"
            ) from exc

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


def _refuse_unwritable_value(value: str) -> None:
    """A written value must be one clean line, because that is all either block can hold.

    `key: value` is formatted and inserted verbatim, so a break inside `value` does not corrupt
    the file — it writes a SECOND key into the block, which is strictly worse. `valid_until:
    1999-01-01` is the version that matters: nothing looks broken, the memo is silently expired,
    and the trust layer then does exactly what it was told. Neither `subject_id` nor `object_id`
    originates in this package (`_providers.py` builds them with a bare `str()` over a provider's
    raw JSON), so this is the boundary where that has to be checked. `key` needs no such check —
    it comes from `destination`, which is a closed set.

    **A break is whatever Python calls a break, not a list of three characters.** Refusing only
    ``\\n`` / ``\\r`` satisfies `parse_frontmatter`, which splits on ``"\\n"`` alone — and misses
    `context.document_title`, which reads the SAME block with `str.splitlines()` and therefore
    breaks on U+2028, U+2029, U+0085, VT and FF. A value carrying U+2028 declares a `title:` line
    that one reader honours and the other cannot see, and that title is fed into the embedding
    text of every chunk of the document. Deferring to `str.splitlines` is what keeps this guard
    correct when a third reader appears.

    Surrounding whitespace is refused rather than trimmed. The derived block is idempotent only
    because its dedup recognises the line it wrote, and it compares on `strip()`; a trailing space
    makes an entry that never matches itself and is appended again on every run. Trimming here
    would fix that and introduce a subtler disagreement instead — `_resolve` and `claim_key`
    normalise through `supersedes_key` while the written value stays raw, so a value that needs
    cleaning is one the ledger and the file would remember differently.
    """
    if "\x00" in value:
        raise RewriteRefused(f"refusing to write: {value!r} contains a NUL byte")
    # `len(splitlines()) > 1` is the whole test. A value ENDING in a separator leaves one line, but
    # every separator `str.splitlines` recognises is also `str.isspace`, so the surrounding-
    # whitespace refusal below catches those — verified across all ten. An extra
    # `value != "".join(value.splitlines())` clause was here and no input could reach it.
    if len(value.splitlines()) > 1:
        raise RewriteRefused(
            f"refusing to write: {value!r} is not a single line — a line break in a written value "
            f"declares an additional key, which is how a live memo gets expired"
        )
    if not value.strip():
        raise RewriteRefused("refusing to write: the value is empty")
    if value != value.strip():
        raise RewriteRefused(
            f"refusing to write: {value!r} carries surrounding whitespace, so the line it writes "
            f"would not match the line its own dedup looks for, and would be re-appended forever"
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
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RewriteRefused(f"refusing to write: {rel} is not valid UTF-8 ({exc.reason})") from exc
    # Universal newlines, applied by hand, because every other reader of a memo gets it for free
    # from `Path.read_text(newline=None)` and this path decodes bytes it already holds. Skipping
    # it made `parse_frontmatter` report no block for a CR-terminated memo here while reporting
    # one everywhere else, and the two answers drive the same write.
    return text.replace("\r\n", "\n").replace("\r", "\n")


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
    _refuse_unwritable_value(value)
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
        updated = insert_frontmatter_line(raw, plan.key, plan.value)
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





def _fenced_flags(lines: list[bytes]) -> tuple[list[bool], bool]:
    """`(is this line inside a code fence, is a fence still open at the end)`.

    Matched the way CommonMark matches them, not toggled. A parity toggle gets both directions
    wrong and one of them is severe:

    * an inner ``` closes an outer ````, so wrapping an example in a longer fence — the only way
      to document this format at all — inverts the parity and exposes the markers inside it;
    * a fence left open swallows the rest of the file, so the module cannot see the block it
      wrote itself and appends a fresh one on every run.

    So a closing fence must use its opener's character with a run at least as long and carry
    nothing else, an opener may carry an info string, and neither may be indented four or more
    columns, which is an indented code block rather than a fence.

    One more rule, and it is the one that bites: **a backtick fence's info string may not contain
    a backtick.** ``` ```yaml``` is the fence ``` is therefore a paragraph with an inline code
    span in it, not an opener. Reading it as one starts a fence that never closes, which both
    hides the memo's real derived block (so the file becomes unwritable) and, when a genuine fence
    follows, inverts every later flag so the entry is written INSIDE the user's code. A tilde
    fence has no such restriction: `~~~` info strings may contain backticks and tildes alike.
    """
    flags: list[bool] = []
    fence: tuple[bytes, int] | None = None
    for line in lines:
        bare = line.rstrip(b"\r\n")
        stripped = bare.lstrip(b" ")
        indent = len(bare) - len(stripped)
        marker = stripped[:1]
        rest = stripped.lstrip(marker) if marker in (b"`", b"~") else b""
        run = len(stripped) - len(rest) if marker in (b"`", b"~") else 0
        if indent <= 3 and run >= 3:
            if fence is None:
                if marker != b"`" or b"`" not in rest:
                    fence = (marker, run)
                    flags.append(True)  # the opening fence line is part of the block
                    continue
            elif marker == fence[0] and run >= fence[1] and not rest.strip():
                fence = None
                flags.append(True)  # so is the closing one
                continue
        flags.append(fence is not None)
    return flags, fence is not None


def _derived_span(lines: list[bytes]) -> tuple[int, int, list[bool]] | None:
    """The one derived block's `(open, close, fenced-flags)`, `None` if there is none.

    Fence aware, because a memo DOCUMENTING this format contains these markers and a scan without
    that writes the new entry into the user's example — editing human prose, the one thing this
    module promises never to do. Markers must also start at column 0: accepting an indented one
    lets an indented code block supply one.

    The flags are returned rather than recomputed so the caller's dedup reads the same view. When
    only the marker scan skipped fenced lines, an entry shown as an EXAMPLE inside the block
    suppressed the real annotation, and said so with a refusal that was untrue.

    Anything other than exactly one open followed by one close is refused rather than guessed at.
    A half-open region cannot be appended past (that writes a second opening marker), a stray
    close cannot either (that is how a file grows an unpaired marker in the first place), and two
    complete blocks would have their entries deduped against only the first — writing a duplicate
    of something already recorded a few lines below.
    """
    flags, _ = _fenced_flags(lines)
    opens: list[int] = []
    closes: list[int] = []
    for index, line in enumerate(lines):
        if flags[index]:
            continue
        bare = line.rstrip(b"\r\n")
        if bare == DERIVED_OPEN.encode("utf-8"):
            opens.append(index)
        elif bare == DERIVED_CLOSE.encode("utf-8"):
            closes.append(index)
    if not opens and not closes:
        return None
    if len(opens) != 1 or len(closes) != 1 or opens[0] > closes[0]:
        raise RewriteRefused(
            f"refusing to write: the derived block is malformed — found {len(opens)} "
            f"{DERIVED_OPEN} and {len(closes)} {DERIVED_CLOSE}, expected exactly one of each "
            f"in that order (an unclosed or duplicated block needs a human)"
        )
    return opens[0], closes[0], flags




def _upsert_derived_entry(raw: bytes, key: str, value: str) -> bytes | None:
    """`key: value` into the derived block, or ``None`` when it is already there.

    One block per file. A second block would make the file's derived state ambiguous and leave no
    way to strip the machine's annotations back off. `key: value` is the identity, not `key` alone
    — a memo may legitimately contradict more than one other memo.
    """
    bom, body = split_bom(raw)
    newline = dominant_newline(body)
    entry = f"{key}: {value}".encode("utf-8")
    open_marker, close_marker = DERIVED_OPEN.encode("utf-8"), DERIVED_CLOSE.encode("utf-8")
    lines = split_lines(body)

    span = _derived_span(lines)
    if span is not None:
        opened, closed, fenced = span
        # `strip`, not `rstrip`: a human who hand-indented an entry still wrote that entry, and
        # matching it is what keeps a second run from adding a near-duplicate beside it. Values
        # carrying their own surrounding whitespace are refused upstream, so the two readings
        # differ only for lines this module did not write. Fenced lines are skipped for the same
        # reason the marker scan skips them: an example is not a declaration.
        if any(
            line.strip() == entry
            for index, line in enumerate(lines[opened + 1 : closed], start=opened + 1)
            if not fenced[index]
        ):
            return None
        lines.insert(closed, entry + line_terminator(lines[closed], newline))
        return bom + b"".join(lines)

    if _fenced_flags(lines)[1]:
        # CommonMark reads an unclosed fence as running to the end of the document, so appending
        # here really would put the machine's annotations inside the user's code block — where
        # the next run cannot see them, and appends another. Three runs, three blocks. The file's
        # structure is ambiguous and a human has to close the fence.
        # A CR-only special case used to live here, explaining that such a memo reads as a single
        # line so a leading fence marker spans the file. That was true only while `split_lines`
        # split on LF alone; it now splits on a lone CR too, matching the universal-newline
        # decoding every reader of a memo already gets. A CR-only file with a CLOSED fence is now
        # simply fine, and one with an open fence has an open fence — so the line endings are no
        # longer the cause and naming them would send a reader after the wrong thing.
        raise RewriteRefused(
            "refusing to write: a code fence is still open at the end of the file, so an "
            "appended derived block would land inside it and be invisible to the next run"
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
