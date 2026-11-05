"""One identity for an indexed document, whatever spelling a row carries it under.

A memo reaches the store under several spellings of one name: an absolute path or a `file://` URI
with percent-escapes, a backslash or a slash, NFC or the NFD a macOS checkout writes. A check that
compares one spelling with another misses the record or matches the wrong one, and CCA found that
shape in 42 confirmed RE-call findings, 11 of them P1 (class `identity-mismatch`, measured
2026-11-04). The comparison sites each derived the root their own way, so this module is the one
derivation they call: `source_identity(source, file)` gives the root the file was indexed under
and the root-relative file, both NFC, so two rows of one document compare equal and two roots
holding the same relative path do not.

Case is kept: on a case-sensitive file system `Notes.md` beside `notes.md` is another memo.
Reference matching (a `supersedes:` target naming a file loosely) is a different question, folded
by `recall.supersession.reference_path_key`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from recall.uploads import source_location


def nfc(value: str) -> str:
    """One spelling of a stored name for comparing names of the same file: NFC, case kept."""
    return unicodedata.normalize("NFC", value)


def root_of(source: object, file: str) -> str | None:
    """The root a chunk was indexed under: its absolute `source` less the root-relative `file`.

    None when the row carries no source (no root is known, and every such file shares one).
    A source that does not end in its file (a URI with percent-escapes, a hand-built fixture)
    stands for its own root: two such files never share a root, which only matters when they
    also share a root-relative path, and then keeping them apart is the point.
    """
    if not isinstance(source, str) or not source:
        return None
    posix_source = source.replace("\\", "/")
    posix_file = file.replace("\\", "/")
    if posix_source == posix_file:
        return ""
    if posix_source.endswith("/" + posix_file):
        return posix_source[: -len(posix_file)]
    return posix_source


@dataclass(frozen=True)
class SourceIdentity:
    """The root a document was indexed under (None when unknown) and its root-relative file."""

    root: str | None
    file: str


def source_identity(source: str | None, file: str) -> SourceIdentity:
    """The identity of the document a row with this absolute `source` and `file` belongs to.

    The source is read through `recall.uploads.source_location`, so a `file://` URI and the plain
    path it names have one root. Generation files keep a URI's escapes while indexed files may
    already be decoded, so the file is decoded only when it matches the URI's raw path, never a
    plain path's literal percent name.
    """
    if source and "://" in source:
        path = urlsplit(source).path
        posix_file = file.replace("\\", "/")
        if path == posix_file or path.endswith("/" + posix_file):
            file = unquote(file)
    root = root_of(source_location(source), file) if source else None
    return SourceIdentity(None if root is None else nfc(root), nfc(file.replace("\\", "/")))
