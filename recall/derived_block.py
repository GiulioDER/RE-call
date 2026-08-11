"""A machine-owned, regenerable block at the end of a memo body.

Frontmatter recognises three keys — `supersedes`, `valid_from`, `valid_until`
(`recall/frontmatter.py:12`) — and the trust layer acts on those and nothing else. The relations
an extractor proposes, `contradicts` and `same_entity`, have nowhere to land. Putting them in
frontmatter would put a machine's inference into the namespace a human authors, where the trust
layer reads it as authored truth.

So they land here instead: a fenced block, last in the body, owned by the machine and stripped
from every path that reads a document as evidence.

**The hazard this module exists to prevent.** `content_hash` is over raw file bytes
(`recall/index.py:518`), so writing a block re-indexes the file. If the block were chunked, the
next extraction pass would read its own prior output back as evidence and amplify: a proposal
becomes a citation for the next proposal, and the corpus grows a self-referential belief no human
ever stated. `split_derived_block` is the only thing standing between the two, which is why it is
total — it never raises, on any input, including input no writer would ever produce.

**Placement is not a preference.** `structure_chunks` computes offsets with
`body.find(text, ...)` (`recall/context.py:197`). While `human` is a strict prefix of `body`,
every offset is identical with or without a block, so `text_start` / `text_end`
(`recall/index.py:600`) are invariant. Prepending would shift every offset in every chunk of every
file that gains a block. End placement also keeps the block out of `document_title`
(`recall/context.py:159`), which reads only frontmatter and the first H1.

**HTML comment fences**, so every markdown renderer hides the block and it can never be mistaken
for frontmatter.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from recall.frontmatter import supersedes_key
from recall.lineage import canonical_sha256

#: Alone on a line. `.strip()` is used for the comparison, so an indented fence still counts —
#: the read path errs toward stripping MORE, never less.
OPEN_FENCE = "<!-- recall:derived v1 -->"
CLOSE_FENCE = "<!-- /recall:derived -->"


@dataclass(frozen=True)
class DerivedSplit:
    """A body cut into the part humans wrote and the part the machine owns."""

    human: str
    block_text: str
    fence_start: int | None


def _first_fence_offset(body: str) -> int | None:
    cursor = 0
    for line in body.splitlines(keepends=True):
        if line.strip() == OPEN_FENCE:
            return cursor
        cursor += len(line)
    return None


def split_derived_block(body: str) -> DerivedSplit:
    """Cut a body at its first derived-block fence. Total: never raises, on any input.

    `.human` is ALWAYS a prefix of `body` — block or not, well-formed or not — because the rule
    is one rule: strip from the first open fence to EOF. Two blocks, an unclosed fence, a human's
    prose appended after the close fence: all the same. A rejoin around each block would preserve
    that appended prose but would stop `.human` being a prefix, and then the offset invariance
    holds only on well-formed files. A qualified guarantee is one that stops being checked.

    The cost is real and is reported, not hidden: prose after a close fence is excluded from
    retrieval, and `derived-block-not-last` names the byte count it costs.

    **The `rstrip()` is on BOTH branches, and the no-fence branch is the load-bearing one.**
    Pre-write a body ends ``"...adopted.\\n"``. Post-write the block sits after one blank line, so
    ``body[:fence_start]`` ends ``"...adopted.\\n\\n"``. Both rstrip to ``"...adopted."``. Strip
    only when a fence is present and the two differ, the extraction cache key changes on the
    first write, and the fixed point fails on iteration one — indistinguishable, from the outside,
    from model nondeterminism. ``s[:n].rstrip()`` is still a prefix of ``s``, so this costs
    nothing.
    """
    start = _first_fence_offset(body)
    if start is None:
        return DerivedSplit(body.rstrip(), "", None)
    return DerivedSplit(body[:start].rstrip(), body[start:], start)


#: Bumped only by a grammar change, and hashed INTO the digest so a v2 block cannot collide with
#: a v1 hash of the same entries.
DERIVED_BLOCK_VERSION = 1

#: The only heads a derived block may carry.
DERIVED_HEADS = ("contradicts", "same_entity", "status")

#: Refused BY NAME rather than falling through to "unknown head". These three have frontmatter
#: keys the trust layer reads (`recall/frontmatter.py:12`); a second copy in the body is a second
#: source of truth that can disagree with the first, and the error should say so.
FORBIDDEN_HEADS = ("supersedes", "valid_from", "valid_until")

#: Closed vocabulary. It deliberately EXCLUDES `deprecated` and `obsolete`, which are in
#: CLOSURE_MARKERS (`recall/lint.py:36`): written literally, the machine's own block would trip
#: the linter built to find prose closure.
STATUS_VALUES = ("open", "adopted", "closed", "superseded", "rejected", "abandoned")
STATUS_ALIASES = {"deprecated": "superseded", "obsolete": "superseded"}

REQUIRED_SUBKEYS = ("proposal", "provider", "reviewer", "at")
OPTIONAL_SUBKEYS = ("note",)
INDENT = "  "

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class DerivedBlockError(ValueError):
    """A derived block is malformed, forbidden, or disagrees with its digest."""


class DerivedDigestMismatch(DerivedBlockError):
    """The block parses, and its structure does not hash to the digest it carries.

    A distinct type rather than a message the caller sniffs for. `diagnose_derived_block` has to
    tell `derived-block-tampered` from `derived-block-malformed`, and a branch keyed off another
    function's error STRING is an interface nobody knows they are maintaining.
    """


@dataclass(frozen=True)
class DerivedEntry:
    """One machine-proposed relation, with the review that let it into the file."""

    head: str
    value: str
    proposal: str
    provider: str
    reviewer: str
    at: str
    note: str = ""


def _is_utc_instant(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _validate_entry(entry: DerivedEntry) -> None:
    """Every rule here is a refusal. Nothing in this function repairs anything."""
    if entry.head in FORBIDDEN_HEADS:
        raise DerivedBlockError(
            f"head {entry.head!r} is frontmatter the trust layer reads; a copy in the body "
            f"would be a second source of truth that can disagree with the first"
        )
    if entry.head not in DERIVED_HEADS:
        raise DerivedBlockError(f"unknown head {entry.head!r}")
    if entry.head == "status":
        if entry.value in STATUS_ALIASES:
            raise DerivedBlockError(
                f"status {entry.value!r} is a closure marker and would trip recall lint; "
                f"write {STATUS_ALIASES[entry.value]!r}"
            )
        if entry.value not in STATUS_VALUES:
            raise DerivedBlockError(
                f"status {entry.value!r} is outside the closed vocabulary "
                f"{' | '.join(STATUS_VALUES)}"
            )
    elif not entry.value or supersedes_key(entry.value) != entry.value:
        # Refuse the wikilink, do NOT unwrap it. `supersedes_key` exists to normalise what a
        # HUMAN wrote (`recall/frontmatter.py:63`); a machine writing its own file has no excuse.
        raise DerivedBlockError(f"value {entry.value!r} is not a bare document stem")
    if not _HEX64.match(entry.proposal):
        raise DerivedBlockError("proposal must be 64 lowercase hex characters")
    if not _is_utc_instant(entry.at):
        raise DerivedBlockError("at must be a Z-suffixed UTC ISO-8601 instant")
    if not entry.provider.strip():
        raise DerivedBlockError("provider must be a non-empty single line")
    if not entry.reviewer.strip():
        raise DerivedBlockError("reviewer must be a non-empty single line")
    for name, text in (
        ("value", entry.value), ("provider", entry.provider),
        ("reviewer", entry.reviewer), ("note", entry.note),
    ):
        if "\n" in text:
            raise DerivedBlockError(f"{name} must be a single line")


def _validate_set(entries: Sequence[DerivedEntry]) -> None:
    if not entries:
        raise DerivedBlockError("a derived block with no entries is churn; remove the block")
    keys = [(entry.head, entry.value) for entry in entries]
    if len(set(keys)) != len(keys):
        # Not fussiness: a duplicate (head, value) makes the sort non-total, so the caller's
        # input order would leak into the rendered bytes and the re-render would not be stable.
        raise DerivedBlockError("duplicate entry: (head, value) must be unique within a block")
    if sum(1 for entry in entries if entry.head == "status") > 1:
        raise DerivedBlockError("at most one status entry per block")


def derived_digest(entries: Sequence[DerivedEntry]) -> str:
    """`canonical_sha256` over the parsed structure — deliberately NOT over the raw bytes.

    Hashing bytes would report every CRLF checkout as tampered, and this repo reads `utf-8-sig`
    and tolerates a BOM precisely because it lives on both Windows and Linux.
    """
    return canonical_sha256(
        {
            "v": DERIVED_BLOCK_VERSION,
            "entries": [
                {
                    "head": entry.head, "value": entry.value, "proposal": entry.proposal,
                    "provider": entry.provider, "reviewer": entry.reviewer, "at": entry.at,
                    "note": entry.note,
                }
                for entry in entries
            ],
        }
    )


def _normalise(entry: DerivedEntry) -> DerivedEntry:
    if entry.head == "status" and entry.value in STATUS_ALIASES:
        return replace(entry, value=STATUS_ALIASES[entry.value])
    return entry


def render_derived_block(entries: Sequence[DerivedEntry]) -> str:
    """Render a block, sorted and digested. The output always ends in the close fence and one \\n.

    This is the ONE place a repair happens, and it is a repair at the boundary rather than to a
    file: `deprecated` / `obsolete` arriving from a proposal are normalised to `superseded`.
    `parse_derived_block` refuses those same values, because a file containing them is claiming
    something the grammar does not permit.
    """
    ordered = tuple(sorted((_normalise(e) for e in entries), key=lambda e: (e.head, e.value)))
    _validate_set(ordered)
    for entry in ordered:
        _validate_entry(entry)
    lines = [OPEN_FENCE]
    for entry in ordered:
        lines.append(f"{entry.head}: {entry.value}")
        lines.append(f"{INDENT}proposal: {entry.proposal}")
        lines.append(f"{INDENT}provider: {entry.provider}")
        lines.append(f"{INDENT}reviewer: {entry.reviewer}")
        lines.append(f"{INDENT}at: {entry.at}")
        if entry.note:
            lines.append(f"{INDENT}note: {entry.note}")
    lines.append(f"digest: {derived_digest(ordered)}")
    lines.append(CLOSE_FENCE)
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class DerivedBlock:
    """A parsed block. `digest` is what the FILE claims; `verify_derived_block` checks it."""

    entries: tuple[DerivedEntry, ...]
    digest: str


def _entry_from(raw: dict[str, str]) -> DerivedEntry:
    missing = [key for key in REQUIRED_SUBKEYS if key not in raw]
    if missing:
        raise DerivedBlockError(
            f"entry {raw['head']}: {raw['value']!r} is missing {', '.join(missing)}"
        )
    entry = DerivedEntry(
        head=raw["head"],
        value=raw["value"],
        proposal=raw["proposal"],
        provider=raw["provider"],
        reviewer=raw["reviewer"],
        at=raw["at"],
        note=raw.get("note", ""),
    )
    _validate_entry(entry)
    return entry


def parse_derived_block(text: str) -> DerivedBlock:
    """Parse a block. Refusal only, never repair. Does NOT check the digest.

    Parsing and verifying are separate so `diagnose_derived_block` can tell a half-written file
    (`derived-block-malformed`) from an integrity breach (`derived-block-tampered`). Reporting the
    first as the second sends whoever reads the lint output looking for an attacker.

    CRLF is normalised and a BOM stripped before anything else, for the same reason the digest is
    over structure: this repo reads `utf-8-sig` and lives on both Windows and Linux, so neither
    is evidence of anything.
    """
    lines = text.replace("\r\n", "\n").lstrip("﻿").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0].strip() != OPEN_FENCE:
        raise DerivedBlockError("a derived block must start with the open fence")
    closes = [i for i, line in enumerate(lines) if line.strip() == CLOSE_FENCE]
    if not closes:
        raise DerivedBlockError("unclosed block: no close fence line")
    close_at = closes[0]
    if any(line.strip() == OPEN_FENCE for line in lines[1:close_at]):
        raise DerivedBlockError("a second open fence inside a block")
    if any(line.strip() for line in lines[close_at + 1 :]):
        if any(line.strip() == OPEN_FENCE for line in lines[close_at + 1 :]):
            raise DerivedBlockError("a second open fence follows the close fence")
        raise DerivedBlockError("content after the close fence")

    inner = lines[1:close_at]
    if not inner or not inner[-1].startswith("digest: "):
        raise DerivedBlockError("digest must be the last line before the close fence")
    digest = inner[-1][len("digest: ") :].strip()
    if not _HEX64.match(digest):
        raise DerivedBlockError("digest must be 64 lowercase hex characters")

    entries: list[DerivedEntry] = []
    current: dict[str, str] | None = None
    for line in inner[:-1]:
        if not line.strip():
            raise DerivedBlockError("a derived block contains no blank lines")
        if line.startswith(INDENT):
            if current is None:
                raise DerivedBlockError("a sub-key appears before any head")
            rest = line[len(INDENT) :]
            if rest.startswith(" "):
                raise DerivedBlockError("a sub-key must be indented exactly two spaces")
            key, separator, value = rest.partition(": ")
            if not separator:
                raise DerivedBlockError(f"malformed sub-key line {rest!r}")
            if key not in REQUIRED_SUBKEYS + OPTIONAL_SUBKEYS:
                raise DerivedBlockError(f"unknown sub-key {key!r}")
            if key in current:
                raise DerivedBlockError(f"duplicate sub-key {key!r}")
            current[key] = value
        else:
            if current is not None:
                entries.append(_entry_from(current))
            head, separator, value = line.partition(": ")
            if not separator:
                raise DerivedBlockError(f"malformed head line {line!r}")
            current = {"head": head, "value": value}
    if current is not None:
        entries.append(_entry_from(current))

    _validate_set(entries)
    keys = [(entry.head, entry.value) for entry in entries]
    if keys != sorted(keys):
        raise DerivedBlockError("entries must be sorted by (head, value)")
    return DerivedBlock(tuple(entries), digest)


def verify_derived_block(text: str) -> DerivedBlock:
    """Parse, then check the digest. The function a write path calls before touching a file.

    Same posture as `recall/fix.py:264` refusing to overwrite what a human wrote: a file whose
    block disagrees with its own structure is not a file to repair.
    """
    block = parse_derived_block(text)
    expected = derived_digest(block.entries)
    if expected != block.digest:
        raise DerivedDigestMismatch(
            f"digest mismatch: the block claims {block.digest}, "
            f"its structure hashes to {expected}"
        )
    return block
