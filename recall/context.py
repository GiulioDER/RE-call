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


def _clean(value: str, limit: int) -> str:
    clean = "".join(ch for ch in value if not unicodedata.category(ch).startswith("C"))
    return clean.strip()[:limit]


def root_relative_source(source: str) -> str:
    """Normalise a source path to a root-relative POSIX one, refusing anything else.

    Every caller in this package already passes a root-relative path (`Indexer.index_path`
    computes `relative_to(root)`), so this refuses rather than sanitises: a path that reaches
    here absolute is a caller that lost its root, and quietly trimming it would embed the host's
    filesystem layout into stored vectors under a field the operator reads as "root-relative".
    An absolute path is also not stable across machines, so two deployments indexing the same
    corpus would produce different embedding text for identical content.

    Control characters are stripped BEFORE the traversal check, not after. Stripping can CREATE a
    traversal (``.\\x00.`` becomes ``..``), so a check that ran first would pass a string that the
    rendered field does not contain.
    """
    normalised = _clean(source.replace("\\", "/"), len(source) + 1)
    if not normalised:
        raise ValueError("source must be a non-empty root-relative path")
    if normalised.startswith("/") or re.match(r"^[A-Za-z]:", normalised):
        raise ValueError(f"source must be root-relative, not absolute: {source!r}")
    if any(segment == ".." for segment in normalised.split("/")):
        raise ValueError(f"source must be root-relative, without traversal: {source!r}")
    return normalised[:SOURCE_MAX_CHARS]


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
    """`(rung, embedding text)` from richest to poorest, in `DEGRADATION_ORDER`.

    Built as labelled pairs rather than a bare list so a test can assert the ORDER the policy
    promises, instead of inferring it from which fields happen to survive a sampled token budget.
    A sweep can only show that some reordering was not exercised; the labels are the claim.

    `document` mode has no neighbour or section rungs to drop, so it yields the subsequence of
    the same ladder rather than a second, independently ordered one.
    """
    rungs = [
        ("full", _render(chunk_text, title=title, section=section, source=source,
                         previous=previous, following=following)),
        ("drop-neighbor", _render(chunk_text, title=title, section=section, source=source)),
        ("shorten-section", _render(chunk_text, title=title,
                                    section=_clean(section, SECTION_DEGRADED_MAX_CHARS),
                                    source=source)),
        ("drop-section", _render(chunk_text, title=title, source=source)),
        ("drop-title", _render(chunk_text, source=source)),
        ("chunk-only", chunk_text),
    ]
    if mode == "document":
        rungs = [rung for rung in rungs if rung[0] in {"drop-section", "drop-title", "chunk-only"}]
    return rungs


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
    safe_source = root_relative_source(source)
    structured = structure_chunks(body, chunks)
    if policy.mode == "none":
        return structured, list(chunks)
    title = document_title(raw, body, source)
    passages: list[str] = []
    for index, chunk in enumerate(structured):
        contextual = policy.mode in {"section", "neighbor"}
        section = _clean(" > ".join(chunk.headings), SECTION_MAX_CHARS) if contextual else ""
        neighbors = policy.mode == "neighbor"
        previous = chunks[index - 1][-NEIGHBOR_MAX_CHARS:] if neighbors and index else ""
        following = (
            chunks[index + 1][:NEIGHBOR_MAX_CHARS]
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
