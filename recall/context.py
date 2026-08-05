"""Deterministic contextual passage representations for embedding only."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from recall.embedding_registry import (
    CONTEXT_POLICY_VERSION,
    ContextMode,
    find_registered_profile,
)

__all__ = [
    "NEIGHBOR_MAX_CHARS",
    "SECTION_DEGRADED_MAX_CHARS",
    "SECTION_MAX_CHARS",
    "SOURCE_MAX_CHARS",
    "TITLE_MAX_CHARS",
    "ContextMode",
    "ContextPolicy",
    "StructuredChunk",
    "Tokenizer",
    "context_policy_for_profile",
    "contextual_passages",
    "document_title",
    "root_relative_source",
    "structure_chunks",
]

#: Caps on the STRUCTURAL fields. They bound what a hostile or merely long document can spend of
#: the embedder's window before the chunk itself is reached, which is why they are named here
#: rather than written as literals at each call site: a test can assert the boundary the code
#: actually applies, and a change to one is a change to one place.
TITLE_MAX_CHARS = 256
SOURCE_MAX_CHARS = 256
SECTION_MAX_CHARS = 512
#: The reduced section cap used by the second degradation rung, before the section is dropped
#: entirely. Section detail is shortened before it is discarded.
SECTION_DEGRADED_MAX_CHARS = 256
#: Neighbour mode adds at most this many characters from EACH adjacent chunk: the tail of the
#: preceding one and the head of the following one.
NEIGHBOR_MAX_CHARS = 200


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class ContextPolicy:
    mode: ContextMode = "none"
    version: str = CONTEXT_POLICY_VERSION
    max_tokens: int | None = None
    tokenizer: Tokenizer | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.tokenizer is None:
            raise ValueError("context max_tokens requires an exact tokenizer")


@dataclass(frozen=True)
class StructuredChunk:
    text: str
    start: int
    end: int
    headings: tuple[str, ...]


def context_policy_for_profile(profile_id: str) -> ContextPolicy:
    """The context policy a registered profile indexes under.

    Reads `recall.embedding_registry`, which is the only place a profile's context mode is
    written down. This used to be a second dict literal over the same vocabulary as the one in
    `make_embedder`, listing three of the six profiles and silently defaulting the rest.

    An UNREGISTERED id still gets `mode="none"`, deliberately: `hashing-64`, a fine-tuned `st:`
    model and every evaluation embedder reach this function through
    `embedding_profile_id(embedder)` and index raw chunk text. Refusing here would break them
    without protecting anything, because the hazard the default used to hide (a registered
    profile present in one map and absent from the other) no longer has a second map to hide in.
    """
    entry = find_registered_profile(profile_id)
    return ContextPolicy(mode=entry.context_mode if entry is not None else "none")


def _clean(value: str, limit: int | None = None) -> str:
    """Strip control characters, trim, and optionally cap.

    `limit=None` means "normalise, do not cap". The two are separate operations because doing
    them together is what let a cap silently invalidate a check that had already passed: see
    `root_relative_source`, which validates a path it must not then truncate.
    """
    clean = "".join(ch for ch in value if not unicodedata.category(ch).startswith("C"))
    clean = clean.strip()
    return clean if limit is None else clean[:limit]


def _fold(value: str) -> str:
    """One line, runs of control characters and whitespace folded to a single space.

    For NEIGHBOUR excerpts. `_clean` would delete a newline outright and weld two words together
    (``"foo\\nbar"`` becomes ``"foobar"``), which is wrong for text going to an embedder, so a run
    of control characters collapses to one space instead.

    It does not cap: the caller slices, and it slices the FOLDED string. Folding before the slice
    is what puts the neighbour budget in the same unit as every other cap here — without it the
    200 counts raw code points, so newlines are spent against a budget every other field measures
    after normalisation. Folding AFTER the slice would be worse than either: it would return
    fewer than 200 characters while claiming 200.
    """
    folded = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in value)
    return re.sub(r"\s+", " ", folded).strip()


def root_relative_source(source: str) -> str:
    """Validate a source path as root-relative POSIX and return it normalised, UNCAPPED.

    Every caller in this package already passes a root-relative path (`Indexer.index_path`
    computes `relative_to(root)`), so this refuses rather than sanitises: a path that reaches
    here absolute is a caller that lost its root, and quietly trimming it would embed the host's
    filesystem layout into stored vectors under a field the operator reads as "root-relative".
    An absolute path is also not stable across machines, so two deployments indexing the same
    corpus would produce different embedding text for identical content.

    Three orderings are load-bearing here, and two of them were wrong when this was first written:

    * Control characters are stripped BEFORE the traversal check. Stripping can CREATE a
      traversal (``.\\x00.`` becomes ``..``), so a check that ran first would bless a string the
      rendered field does not contain.
    * **It does not truncate.** It used to return ``normalised[:SOURCE_MAX_CHARS]``, applied
      after the checks, which can MANUFACTURE what they just refused: ``"a" * 253 + "/..x"``
      passed the traversal check and came back as a 256-character path whose last segment is
      ``..``. A guard that mutates its value after validating it does not hold on its own output.
      The cap belongs to the rendered FIELD, and is applied where that field is built.
    * `document_title`'s basename fallback reads this return value, so truncating here also
      changed a document's title: at 264 characters the cut landed on a ``/``, the basename was
      empty, and the ``title:`` field vanished from the passage entirely.

    The drive-letter test requires a separator or end of string after the colon. A bare
    ``^[A-Za-z]:`` also refuses ``a:b/notes.md``, a legal relative path on Linux and macOS, and
    ``relative_to(root).as_posix()`` produces exactly that shape.
    """
    normalised = _clean(source.replace("\\", "/"))
    if not normalised:
        raise ValueError("source must be a non-empty root-relative path")
    if normalised.startswith("/") or re.match(r"^[A-Za-z]:(/|$)", normalised):
        # The offending value is deliberately NOT interpolated. This fires on a caller that lost
        # its root, so the value IS an absolute host path, and echoing it puts the filesystem
        # layout (and any username in it) into the logs — the disclosure the guard exists to
        # prevent. The caller names the file it was reading.
        raise ValueError("source must be root-relative, not absolute")
    if any(segment == ".." for segment in normalised.split("/")):
        raise ValueError("source must be root-relative, without traversal")
    return normalised


def document_title(raw: str, body: str, source: str) -> str:
    """Choose frontmatter title, first H1, then the root relative basename.

    The frontmatter key must be TOP LEVEL. `key.strip()` alone matched an indented `title:`
    nested under any other mapping, and because the scan returns on its first hit a nested title
    appearing above the real one won \u2014 silently embedding a sub-object's label as the document's.
    Indentation is the only thing that distinguishes the two, so it cannot be stripped before the
    comparison.
    """
    lines = raw.splitlines()
    if lines and lines[0].lstrip("\ufeff").strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line[:1].isspace():
                continue
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == "title":
                return _clean(value.strip().strip("'\""), TITLE_MAX_CHARS)
    for line in body.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return _clean(match.group(1), TITLE_MAX_CHARS)
    return _clean(root_relative_source(source).rsplit("/", 1)[-1], TITLE_MAX_CHARS)


def structure_chunks(body: str, chunks: list[str]) -> list[StructuredChunk]:
    """Attach stable source offsets and heading hierarchy to existing public chunks."""
    heading_events: list[tuple[int, int, str]] = []
    cursor = 0
    for line in body.splitlines(keepends=True):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line.rstrip("\r\n"))
        if match:
            heading_events.append((cursor, len(match.group(1)), _clean(match.group(2), 256)))
        cursor += len(line)
    output: list[StructuredChunk] = []
    previous_start = -1
    for text in chunks:
        start = body.find(text, previous_start + 1)
        if start < 0:
            start = body.find(text)
        if start < 0:
            # Some public chunkers normalise blank-line runs while packing. Match the same
            # visible tokens across arbitrary source whitespace so offsets remain useful without
            # changing the chunker's long-standing return value.
            tokens = text.split()
            flexible = re.compile(r"\s+".join(re.escape(token) for token in tokens))
            match = flexible.search(body, max(previous_start + 1, 0))
            if match is None:
                match = flexible.search(body)
            if match is None:
                raise ValueError("chunk text could not be mapped back to its source document")
            start = match.start()
            source_end = match.end()
        else:
            source_end = start + len(text)
        previous_start = start
        hierarchy: list[str] = []
        levels: list[int] = []
        for position, level, heading in heading_events:
            if position > start:
                break
            while levels and levels[-1] >= level:
                levels.pop()
                hierarchy.pop()
            levels.append(level)
            hierarchy.append(heading)
        output.append(
            StructuredChunk(text=text, start=start, end=source_end, headings=tuple(hierarchy))
        )
    return output


def _render(
    content: str,
    *,
    title: str = "",
    section: str = "",
    source: str = "",
    previous: str = "",
    following: str = "",
) -> str:
    fields = []
    if title:
        fields.append(f"title: {title}")
    if section:
        fields.append(f"section: {section}")
    if source:
        fields.append(f"source: {source}")
    if previous:
        fields.append(f"previous: {previous}")
    fields.append(f"content: {content}")
    if following:
        fields.append(f"following: {following}")
    return "\n".join(fields)


#: The degradation ladder's rungs, in the order they are tried. The names are the ORDER the
#: policy states: neighbour context goes first, section detail second, title detail last. Every
#: rung carries the complete current chunk; the chunk is never shortened to make room.
DEGRADATION_ORDER = (
    "full",
    "drop-neighbor",
    "shorten-section",
    "drop-section",
    "drop-title",
    "chunk-only",
)

#: `document` mode carries no neighbour and no section, so the first three rungs would render
#: byte-identically to `drop-section`. Derived as a SUFFIX of the full order rather than written
#: out again, so document mode cannot acquire an order of its own.
DOCUMENT_DEGRADATION_ORDER = DEGRADATION_ORDER[3:]


def _degradation_ladder(
    chunk_text: str,
    *,
    mode: ContextMode,
    title: str,
    section: str,
    source: str,
    previous: str,
    following: str,
) -> list[tuple[str, str]]:
    """`(rung, embedding text)` from richest to poorest, emitted in `DEGRADATION_ORDER`.

    The rungs are rendered into a mapping and then emitted in the constant's order, so the
    constant IS the order rather than a second literal that happens to agree with one. It was the
    latter first: renaming the labels built here left every test green, which made the constant
    decorative and this docstring's earlier claim — that a test could assert the order from it —
    false.

    A consequence worth stating, because it moves where the guarantee lives: a test comparing what
    this emits against `DEGRADATION_ORDER` now compares the code with itself. The order is pinned
    by two things that do not derive from it, a monotonic token-budget sweep over the surviving
    fields, and a written-out expected sequence.

    `document` mode carries no neighbour and no section, so it emits a derived SUFFIX of the same
    order rather than an order of its own.
    """
    rendered = {
        "full": _render(chunk_text, title=title, section=section, source=source,
                        previous=previous, following=following),
        "drop-neighbor": _render(chunk_text, title=title, section=section, source=source),
        "shorten-section": _render(chunk_text, title=title,
                                   section=_clean(section, SECTION_DEGRADED_MAX_CHARS),
                                   source=source),
        "drop-section": _render(chunk_text, title=title, source=source),
        "drop-title": _render(chunk_text, source=source),
        "chunk-only": chunk_text,
    }
    # Emitted in DEGRADATION_ORDER, not in the order written above. The constant read as the
    # ladder's specification while being an independent second literal: renaming these labels
    # left every test green, so the declared order was decorative and the docstring's claim that
    # a test can assert it was false. Now there is one order and this is it.
    order = DOCUMENT_DEGRADATION_ORDER if mode == "document" else DEGRADATION_ORDER
    return [(rung, rendered[rung]) for rung in order]


def contextual_passages(
    raw: str,
    body: str,
    chunks: list[str],
    source: str,
    policy: ContextPolicy,
) -> tuple[list[StructuredChunk], list[str]]:
    """Return structured chunks and embedding text, never altering public chunk text.

    The source is validated for EVERY mode, including `none`, which returns before it would use
    one. A guard reachable only on the expensive path is a guard the cheapest caller skips, and
    the mode is chosen by the profile rather than by this call site.
    """
    # Validated here, capped here. `root_relative_source` deliberately does not truncate, because
    # a cap applied inside it runs after its own checks and can reintroduce what they refused.
    safe_source = _clean(root_relative_source(source), SOURCE_MAX_CHARS)
    structured = structure_chunks(body, chunks)
    if policy.mode == "none":
        return structured, list(chunks)
    title = document_title(raw, body, source)
    passages: list[str] = []
    for index, chunk in enumerate(structured):
        contextual = policy.mode in {"section", "neighbor"}
        section = _clean(" > ".join(chunk.headings), SECTION_MAX_CHARS) if contextual else ""
        neighbors = policy.mode == "neighbor"
        # Folded to one line BEFORE the 200 is counted, so the neighbour budget is in the same
        # unit as the other caps. The tail of the one before, the head of the one after.
        previous = (
            _fold(chunks[index - 1])[-NEIGHBOR_MAX_CHARS:] if neighbors and index else ""
        )
        following = (
            _fold(chunks[index + 1])[:NEIGHBOR_MAX_CHARS]
            if neighbors and index + 1 < len(chunks)
            else ""
        )
        ladder = _degradation_ladder(
            chunk.text, mode=policy.mode, title=title, section=section,
            source=safe_source, previous=previous, following=following,
        )
        chosen = ladder[0][1]
        if policy.max_tokens is not None:
            assert policy.tokenizer is not None
            chosen = next(
                (text for _, text in ladder
                 if policy.tokenizer.count_tokens(text) <= policy.max_tokens),
                chunk.text,
            )
        passages.append(chosen)
    return structured, passages
