"""Turn prose closure markers into declared `supersedes:` edges.

`recall lint` finds memos whose body says *"superseded by X"* / *"replaces X"* while the
frontmatter declares nothing — 60 of them in a real 792-memo corpus, against 2 declared edges.
The relation is being written; it is just written where retrieval cannot act on it.

Detection already worked. This adds the write-back, under a rule that refuses far more often
than it acts.

⚠️ **Measured on that corpus, it proposes ZERO edges.** Four survived the mechanical rules and
all four were wrong on review: one was reported speech, two superseded a *claim* or *scope*
inside their target rather than the target, and the last was hedged (`"Supersedes/augments"`) —
its author, asked directly, said *augments*. Each became a refusal. So this is a **reviewing
aid, not an automation**: it narrows 60 prose markers to the handful worth a human's attention
and declines to guess at the rest. Treat a non-empty proposal list as a question, not an answer.

The refusal rules:

**A fix is proposed only when the target is PROVABLE.** The body must name a document — as a
`[[wikilink]]`, a bare `name.md`, or a bare stem — in the same sentence as the marker, and that
name must resolve to exactly one file in the corpus. A bare `DEPRECATED` with no target is
reported as needing a human, never guessed at.

**Direction follows the marker's voice**, and it decides WHICH FILE is edited:

- *"this supersedes X"* → `supersedes: X` goes on **this** memo.
- *"this is superseded by X"* → the edge belongs on **X**, because the schema has no
  `superseded_by`. Getting this backwards would declare the live memo stale and demote it
  beneath the one it replaced — the exact failure the trust layer exists to prevent, caused by
  the tool meant to fix it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from recall.atomic_write import atomic_write_bytes
from recall.document import parse_document
from recall.frontmatter import (
    NAME_STAND_IN_MARK,
    encodable_name,
    has_line_break,
    insert_frontmatter_line,
    supersedes_key,
    writable_reference,
)
from recall.lint import DEFAULT_GLOB
from recall.observability import get_logger

_log = get_logger("fix")

#: Markers where the SUBJECT of the sentence supersedes the named target.
_ACTIVE = r"(?:supersedes|replaces|supercedes)"
#: Markers where the subject IS superseded by the named target — the edge goes on the target.
_PASSIVE = r"(?:superseded\s+by|superceded\s+by|replaced\s+by)"

#: A document reference. Three forms only, and each is deliberately hard to match by accident:
#:
#:   [[wikilink]]            the corpus convention
#:   some_memo_name.md       an explicit filename
#:   some_memo_2026-07-14    a bare stem, but ONLY when it carries a 20xx year
#:
#: Everything looser was tried against the real corpus and produced garbage. Single brackets
#: match markdown checkboxes (`[x]`, `[ ]`) and ordinary prose asides; backticks match inline
#: code, which these memos are full of — one match captured a 600-character paragraph, another
#: `curate_wallets.wallet_weight = clamp(...)`. None of it could ever resolve to a file, so
#: nothing unsafe was written, but the SKIP list became noise no human could act on. A proposal
#: tool whose output must itself be filtered has not saved anyone any work.
_REF = (
    r"\[\[([^\]\n]{1,120})\]\]"
    r"|([A-Za-z0-9][\w\-]{5,}\.md)"
    r"|([a-z][a-z0-9]*(?:[_\-][a-z0-9]+)*[_\-]20\d\d(?:[_\-]\d\d){0,2})"
)

_PASSIVE_RE = re.compile(
    rf"(?P<marker>{_PASSIVE})[^\n.;]{{0,40}}?(?:{_REF})", re.IGNORECASE
)
_ACTIVE_RE = re.compile(
    rf"\b(?P<marker>{_ACTIVE})[^\n.;]{{0,40}}?(?:{_REF})", re.IGNORECASE
)


class UnreadableMemo(ValueError):
    """`apply_proposal` was pointed at a memo it will not rewrite, and said which and why."""


@dataclass(frozen=True)
class Proposal:
    """One edge to declare. `edit_file` is the memo that gains `supersedes: <target>`."""

    edit_file: str      # root-relative path of the file to modify
    target: str         # value to write for `supersedes:`
    evidence_file: str  # the memo whose prose stated the relation
    evidence: str       # the matched phrase, so a human can judge it


@dataclass(frozen=True)
class Unfixable:
    """A closure marker whose target could not be proved. Reported, never guessed."""

    file: str
    reason: str


#: Filenames that catalogue other memos rather than making a claim of their own.
_INDEX_NAMES = ("index", "memory", "readme", "gates_table", "toc")


def _is_index(rel: str) -> bool:
    """True for a catalogue file — one that lists memos instead of superseding one."""
    stem = supersedes_key(rel).lower()
    return any(stem == n or stem.endswith(f"_{n}") or stem.startswith(f"{n}_")
               for n in _INDEX_NAMES)


def _first_ref(match: re.Match) -> str | None:
    """The document reference in a marker match.

    Skips group 1, which is the named `marker` group — the verb phrase itself is never the
    reference, and returning it would propose an edge onto a file called "Supersedes".
    """
    for group in match.groups()[1:]:
        if group:
            return str(group).strip()
    return None


#: Words that mean the SUBJECT of the marker is some other document, so the sentence is
#: reporting a relation rather than declaring this memo's own.
_OTHER_DOC = re.compile(
    r"\[\[[^\]\n]+\]\]|\b[\w\-]{6,}\.md\b|\b(?:memo|doc|document|note|entry|file|index)s?\b",
    re.IGNORECASE,
)
#: "supersedes the <noun> in X" — a part of X, not X.
_PARTIAL_SCOPE = re.compile(r"\bthe\b.+\bin\b", re.IGNORECASE | re.DOTALL)
#: Clause boundaries; the subject of a marker lives after the nearest one.
_CLAUSE_END = (".", ";", ":", "\n", "—", "-")


def _is_reported_speech(body: str, marker_start: int) -> bool:
    """True when the marker's subject is ANOTHER document, not the memo being read.

    Real corpus, `project-docs-rag-trust-layer-deployed-2026-07-17.md`:

        First annotations: LRP closure memo supersedes `project_lrp_maker_2026-06-24`

    The subject of "supersedes" is *the LRP closure memo*. Attributing the claim to the document
    that merely NARRATES it invented a second, false claimant for an edge another memo already
    declares correctly — the worst kind of false positive, because it looks authoritative.
    """
    head = body[:marker_start]
    cut = max((head.rfind(c) for c in _CLAUSE_END), default=-1)
    return bool(_OTHER_DOC.search(head[cut + 1:]))


#: Qualifiers that weaken the claim from "replaces" to "relates to".
_HEDGE_BEFORE = re.compile(
    r"\b(?:partially|partly|largely|mostly|arguably|effectively|broadly|possibly)\s+$",
    re.IGNORECASE,
)
#: "supersedes/augments X", "supersedes or augments X" — the author declined to commit.
_HEDGE_AFTER = re.compile(r"^\s*(?:/|\bor\b)\s*\w+", re.IGNORECASE)


def _is_hedged(body: str, marker_start: int, after: str) -> bool:
    """True when the author qualified the claim rather than making it.

    From the real corpus: `"Supersedes/augments [[feedback_ci_green_constraints_2026-06-22]]"`.
    Asked directly, the author's answer was **augments** — the slash was doing real work. An
    augmenting memo does not replace its predecessor, and declaring the edge would demote a memo
    that is still current.

    A hedge is the author saying they are not sure. Resolving it for them is exactly the kind of
    confident wrong answer this project exists to avoid.
    """
    return bool(_HEDGE_BEFORE.search(body[:marker_start]) or _HEDGE_AFTER.match(after))


def _is_partial_scope(between: str) -> bool:
    """True for "supersedes the <noun> in X" — X's *claim* or *scope*, not X itself.

    Real corpus: "Supersedes the *inferred* "maker" claim in [[...]]" and "Supersedes the scope
    in [[...]]". Declaring `supersedes:` there would demote the WHOLE predecessor and lose
    everything else it holds, when only one part of it was replaced.
    """
    return bool(_PARTIAL_SCOPE.search(between))


def _accept(body: str, m: re.Match) -> str | None:
    """The reference this match declares, or None when the sentence does not declare one."""
    ref = _first_ref(m)
    if not ref:
        return None
    if _is_reported_speech(body, m.start()):
        return None
    marker_end = m.end("marker")
    ref_start = min(m.start(g) for g in range(2, (m.lastindex or 1) + 1) if m.group(g))
    between = body[marker_end:ref_start]
    if _is_hedged(body, m.start(), between) or _is_partial_scope(between):
        return None
    return ref


def extract_edges(body: str) -> tuple[list[str], list[str]]:
    """``(actively_supersedes, superseded_by)`` document references named in `body`.

    Pure and file-free so the direction rule — the part that would silently invert the
    supersession graph if wrong — is testable on strings alone.
    """
    active = [r for m in _ACTIVE_RE.finditer(body) if (r := _accept(body, m))]
    passive = [r for m in _PASSIVE_RE.finditer(body) if (r := _accept(body, m))]
    # "superseded by X" also matches the active pattern's bare "supersede" stem in some
    # phrasings; passive wins, since its voice is the more specific reading.
    passive_keys = {supersedes_key(p) for p in passive}
    active = [a for a in active if supersedes_key(a) not in passive_keys]
    return active, passive


def propose_fixes(
    path: str | Path, glob: str = DEFAULT_GLOB
) -> tuple[list[Proposal], list[Unfixable]]:
    """Scan a corpus and return the edges that can be declared, plus what needs a human."""
    root = Path(path)
    files = sorted(root.glob(glob)) if root.is_dir() else [root]
    rel = {f: (f.relative_to(root).as_posix() if root.is_dir() else f.name) for f in files}

    by_key: dict[str, list[str]] = {}
    for f in files:
        by_key.setdefault(supersedes_key(f.name), []).append(rel[f])

    existing: dict[str, str] = {}
    bodies: dict[str, str] = {}
    for f in files:
        try:
            document = parse_document(f.read_text(encoding="utf-8-sig"))
            meta, body = document.meta, document.human_body
        except (UnicodeDecodeError, OSError):
            continue
        bodies[rel[f]] = body
        if meta.get("supersedes"):
            existing[rel[f]] = meta["supersedes"]

    proposals: list[Proposal] = []
    unfixable: list[Unfixable] = []
    seen: set[tuple[str, str]] = set()
    #: Edges parked because their value would split the written line. Keyed by the dedup pair AND
    #: the source memo: the pair is what decides suppression below, and the source is what keeps
    #: two DIFFERENT malformed files from collapsing into one report. Keying on the pair alone
    #: named only the first of them, so the operator renamed it, re-ran, and only then learned of
    #: the second — N bad files costing N runs to surface. `setdefault` still collapses repeats
    #: of one source naming the same edge twice, which is the case that is genuinely one report.
    refused: dict[tuple[tuple[str, str], str], tuple[str, str, str]] = {}

    for name, body in bodies.items():
        if _is_index(name):
            # An index ENUMERATES closed decisions; it does not supersede them. On the real
            # corpus `closed_hypotheses_index.md` listing an archived memo was read as
            # "the archive supersedes the index" — syntactically valid, semantically backwards.
            continue
        active, passive = extract_edges(body)
        if not active and not passive:
            continue
        for ref, edit_file, target_name in (
            [(r, name, r) for r in active] + [(r, None, r) for r in passive]
        ):
            key = supersedes_key(ref)
            candidates = by_key.get(key, [])
            if len(candidates) != 1:
                unfixable.append(Unfixable(
                    name,
                    f"names {ref!r}, which matches {len(candidates)} files in the corpus"
                    if candidates else f"names {ref!r}, which is not a file in the corpus",
                ))
                continue
            resolved = candidates[0]
            if resolved == name:
                continue  # self-reference: lint reports it separately
            if edit_file is not None:
                writer, value = edit_file, target_name
            else:
                # Passive voice: the OTHER file is the one that supersedes this memo, so the
                # value is THIS memo's own corpus path. That makes this the only branch whose
                # value is a corpus NAME rather than a reference read out of a body this
                # package has already decoded as UTF-8, and so the only one that needs the
                # boundary the sibling commands share. A POSIX filename is bytes: one that is
                # not valid UTF-8 arrives as a lone surrogate through `Path.glob`, and
                # `insert_frontmatter_line` encodes the value as UTF-8 — which raised out of
                # the apply loop, uncaught, after earlier memos in the same run had already
                # been rewritten. `writer` is a path this package OPENS, so it stays exactly as
                # the filesystem handed it over.
                writer = resolved
                value = writable_reference(encodable_name(name))
                if NAME_STAND_IN_MARK in value:
                    # `writable_reference` has already dropped the directories no reader
                    # compares, so a marker still here is on the FILE's own name and no
                    # spelling is left: the raw name cannot enter a UTF-8 memo, and `lint`,
                    # `check`, `fix`, `store` and the reasoning graph all resolve a declared
                    # edge by comparing raw filenames, so the stand-in would read as an edge to
                    # a human and as an unresolved one to every reader here. Refused at PROPOSE
                    # time, so the dry run says so before `--apply` has written anything.
                    unfixable.append(Unfixable(
                        name,
                        f"has a name that is not valid UTF-8, so the edge {writer} would "
                        f"declare names a file no reader of the corpus resolves. Rename it "
                        f"first; the edge can be declared once it has a name a memo can hold",
                    ))
                    continue
            if writer in existing:
                unfixable.append(Unfixable(
                    writer,
                    f"already declares supersedes: {existing[writer]!r} — refusing to overwrite",
                ))
                continue
            pair = (writer, supersedes_key(value))
            if pair in seen:
                continue
            if has_line_break(value):
                # DEFERRED, not decided here. `insert_frontmatter_line` refuses this too, but a
                # raise at WRITE time is met halfway through the apply loop, once earlier memos
                # have already been rewritten, so the refusal has to be reportable by the dry run
                # like every other one here.
                #
                # It cannot be decided inline, in either position. Reported before the dedup it
                # fires for a second spelling of an edge this same run declares correctly, since
                # `supersedes_key` compares on the stem. Reported after it, without joining
                # `seen`, the same edge is reported once per spelling AND can be reported
                # alongside the very proposal that declares it correctly — which prints SKIP for
                # a memo the next line then writes to. Adding it to `seen` is not the fix either:
                # that suppresses the correct proposal whenever the bad spelling is iterated
                # first. `bodies` follows `sorted(root.glob(...))`, so which spelling comes first
                # is stable but arbitrary — inline would be right for one of the two orders and
                # wrong for the other, which is not a property worth resting on.
                #
                # So the pair is parked and judged once the loop knows every edge it could
                # declare. The predicate is imported rather than restated, so the reporter and
                # the writer cannot drift apart.
                #
                # `name`, not `writer`, is the file reported: in the passive direction the value
                # IS the source memo's own path and `writer` is the innocent memo it would be
                # written into, so blaming `writer` sends the operator to the one file they
                # cannot fix. Same subject as the refusal above at `names {ref!r}`.
                refused.setdefault((pair, name), (name, value, writer))
                continue
            seen.add(pair)
            proposals.append(Proposal(writer, value, name, ref))
    for (pair, _source_key), (source, value, writer) in refused.items():
        if pair in seen:
            continue  # another spelling of this same edge was declared correctly
        unfixable.append(Unfixable(
            source,
            f"names {value!r}, which contains a line break; it would be written into {writer}",
        ))
    return proposals, unfixable


def apply_proposal(root: Path, p: Proposal) -> None:
    """Insert `supersedes: <target>` into `p.edit_file`'s frontmatter, preserving everything else.

    Rewrites only the frontmatter block: a file without one gains a minimal block above its
    existing content, and a file with one keeps its other keys, order and body BYTE FOR BYTE.

    That last part was a false claim for a long time. This decoded `utf-8-sig` and wrote `utf-8`,
    which drops a Windows-authored memo's BOM — one `parse_frontmatter` tolerates precisely
    because editors add it — and split on ``"\\n"``, which normalised every line ending in the
    file. Both are invisible to a test that reads back through `read_text`, which is why they
    survived a 28-test suite. The insertion now goes through `recall/frontmatter.py`, so the two
    writers of the user's own memos and the parser that reads them share one definition of a line
    and one of a leading BOM.

    It now also shares the parser's definition of WHETHER a block exists at all. A leading
    horizontal rule is not frontmatter, and the old write path scanned forward for the next
    ``---`` anyway, inserted `supersedes:` into the middle of the author's prose, and then
    caused that inserted key to parse as frontmatter on the next read. `insert_frontmatter_line`
    now asks `frontmatter_span`, so a thematic-break opening gets a real block prepended above it
    rather than being edited in place.

    The target is checked before anything is opened. `propose_fixes` already refuses a value no
    reader could resolve, so these fire only for a `Proposal` built elsewhere — but relying on
    that is exactly what the apply loop was doing when `UnicodeEncodeError` walked past its
    `except` and left a partial run behind. `UnreadableMemo` is the vocabulary that loop already
    reports as SKIP, so containment does not depend on who built the proposal.

    Two checks, because the two spellings of one unnamable file need different sentences. A raw
    surrogate is what `insert_frontmatter_line` cannot encode. A stand-in encodes perfectly and
    resolves for nobody, and it is the one a reviewer can hand back after reading it in a
    report, so "is not valid UTF-8" would be a false thing to say about the value in front of
    them and would name nothing they could act on.
    """
    if NAME_STAND_IN_MARK in p.target:
        # The marker is stripped from the message: it carries a NUL, which is there to keep two
        # names apart and not for a reviewer to read.
        raise UnreadableMemo(
            f"{p.edit_file} cannot declare supersedes: "
            f"{p.target.replace(NAME_STAND_IN_MARK, '')!r}, this corpus's stand-in for a file "
            f"whose name is not valid UTF-8, which no reader of the corpus resolves"
        )
    try:
        p.target.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnreadableMemo(
            f"{p.edit_file} cannot declare supersedes: {p.target!r}, which is not valid UTF-8 "
            f"({exc.reason})"
        ) from exc
    f = root / p.edit_file if root.is_dir() else root
    raw = f.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        # `propose_fixes` skips a file it cannot decode, so such a file never appears in
        # `existing` and its authored `supersedes:` is invisible to the overwrite refusal — but it
        # can still be the TARGET of a passive-voice marker in some other memo, and so still be
        # chosen as `edit_file`. Writing then replaces a declared edge (`parse_frontmatter` is
        # last-wins) and, because the file still will not decode next run, appends another line
        # every time. The old `read_text` raised here; keep refusing, with a message that says
        # which file and why.
        raise UnreadableMemo(f"{p.edit_file} is not valid UTF-8 ({exc.reason})") from exc
    if parse_document(text).meta.get("supersedes"):
        # Re-checked against THIS file rather than the corpus-wide scan, for the same reason:
        # the scan's view can be missing a file it could not read at the time.
        raise UnreadableMemo(f"{p.edit_file} already declares supersedes — refusing to overwrite")
    line = f"supersedes: {p.target}"
    atomic_write_bytes(f, insert_frontmatter_line(raw, "supersedes", p.target))
    _log.info("declared %s in %s", line, p.edit_file)
