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

from recall.frontmatter import parse_frontmatter, supersedes_key
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
            meta, body = parse_frontmatter(f.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, OSError):
            continue
        bodies[rel[f]] = body
        if meta.get("supersedes"):
            existing[rel[f]] = meta["supersedes"]

    proposals: list[Proposal] = []
    unfixable: list[Unfixable] = []
    seen: set[tuple[str, str]] = set()

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
            # passive voice: the OTHER file is the one that supersedes this memo
            writer = edit_file if edit_file is not None else resolved
            value = target_name if edit_file is not None else name
            if writer in existing:
                unfixable.append(Unfixable(
                    writer,
                    f"already declares supersedes: {existing[writer]!r} — refusing to overwrite",
                ))
                continue
            pair = (writer, supersedes_key(value))
            if pair in seen:
                continue
            seen.add(pair)
            proposals.append(Proposal(writer, value, name, ref))
    return proposals, unfixable


def apply_proposal(root: Path, p: Proposal) -> None:
    """Insert `supersedes: <target>` into `p.edit_file`'s frontmatter, preserving everything else.

    Rewrites only the frontmatter block: a file without one gains a minimal block above its
    existing content, and a file with one keeps its other keys, order and body byte-for-byte.
    """
    f = root / p.edit_file if root.is_dir() else root
    text = f.read_text(encoding="utf-8-sig")
    line = f"supersedes: {p.target}"
    lines = text.split("\n")
    if lines and lines[0].lstrip("﻿").strip() == "---":
        for i, ln in enumerate(lines[1:], start=1):
            if ln.strip() == "---":
                lines.insert(i, line)
                break
        else:  # unclosed block — treat as no frontmatter rather than corrupt it further
            lines = ["---", line, "---", *lines]
    else:
        lines = ["---", line, "---", *lines]
    f.write_text("\n".join(lines), encoding="utf-8")
    _log.info("declared %s in %s", line, p.edit_file)
