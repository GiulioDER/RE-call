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
    "ContextMode",
    "ContextPolicy",
    "StructuredChunk",
    "Tokenizer",
    "context_policy_for_profile",
    "contextual_passages",
    "document_title",
    "structure_chunks",
]


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


def document_title(raw: str, body: str, source: str) -> str:
    """Choose frontmatter title, first H1, then the root relative basename."""
    lines = raw.splitlines()
    if lines and lines[0].lstrip("\ufeff").strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == "title":
                return _clean(value.strip().strip("'\""), 256)
    for line in body.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return _clean(match.group(1), 256)
    return _clean(source.replace("\\", "/").rsplit("/", 1)[-1], 256)


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


def contextual_passages(
    raw: str,
    body: str,
    chunks: list[str],
    source: str,
    policy: ContextPolicy,
) -> tuple[list[StructuredChunk], list[str]]:
    """Return structured chunks and embedding text, never altering public chunk text."""
    structured = structure_chunks(body, chunks)
    if policy.mode == "none":
        return structured, list(chunks)
    title = document_title(raw, body, source)
    safe_source = _clean(source.replace("\\", "/"), 256)
    passages: list[str] = []
    for index, chunk in enumerate(structured):
        section = _clean(" > ".join(chunk.headings), 512) if policy.mode in {"section", "neighbor"} else ""
        previous = chunks[index - 1][-200:] if policy.mode == "neighbor" and index else ""
        following = chunks[index + 1][:200] if policy.mode == "neighbor" and index + 1 < len(chunks) else ""
        candidates = [
            _render(chunk.text, title=title, section=section, source=safe_source,
                    previous=previous, following=following),
            _render(chunk.text, title=title, section=section, source=safe_source),
            _render(chunk.text, title=title, section=_clean(section, 256), source=safe_source),
            _render(chunk.text, title=title, source=safe_source),
            _render(chunk.text, source=safe_source),
            chunk.text,
        ]
        if policy.mode == "document":
            candidates = [
                _render(chunk.text, title=title, source=safe_source),
                _render(chunk.text, source=safe_source),
                chunk.text,
            ]
        chosen = candidates[0]
        if policy.max_tokens is not None:
            assert policy.tokenizer is not None
            chosen = next(
                (candidate for candidate in candidates
                 if policy.tokenizer.count_tokens(candidate) <= policy.max_tokens),
                chunk.text,
            )
        passages.append(chosen)
    return structured, passages
