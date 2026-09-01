"""Deciding what a hosted sync should upload, without uploading anything.

This module is the diff. It answers "which of my files does the server not already have, and in
what batches may I send them", and it does so as pure functions over dictionaries so the decision
can be tested without a network, a database, or a server.

**Nothing here imports `recall`.** That is the same rule the rest of this package follows and for
the same measured reason: `recall/__init__.py` costs about a second, and a `SessionStart` hook runs
before the user's first turn of every session.

⛔ **The digest is the whole contract, and getting it wrong fails silently.** The server stores a
content hash computed one specific way, and `recall_inventory` reports it. If this module hashed
differently — raw bytes for markdown, say — no local file would ever match its server entry, every
sync would upload the entire corpus, and it would all look like it was working. `digest_of` mirrors
`recall/index.py` deliberately and duplicates it rather than importing it, so the duplication is
visible and this docstring is where the next reader is told to check both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

#: Suffixes the server treats as markdown, and therefore hashes as decoded text rather than as
#: bytes. Kept in sync with `recall/index.py`; see this module's docstring.
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})


@dataclass(frozen=True)
class Limits:
    """Client-side bounds, deliberately under the server's own.

    `recall_ingest` refuses more than 500 files or 50 MiB per request, and its byte check runs
    after decoding. Sitting under both means a batch this module builds is never refused for a
    reason the client could have seen coming, which matters because a refusal costs the whole
    batch rather than one file.
    """

    max_files: int = 400
    max_bytes: int = 40 * 1024 * 1024
    max_file_bytes: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class Change:
    """One local file the server does not have, or does not have at this content."""

    name: str
    """Root-prefixed POSIX relative path, e.g. `project/agents/notes.md`.

    The prefix namespaces the two memory roots a project can have. Without it, two roots with the
    same internal layout collide into one name and each sync overwrites the other's file.
    """

    path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class SyncPlan:
    """What to do, decided before anything is sent."""

    upload: list[list[Change]] = field(default_factory=list)
    """Batches, each within `Limits`. Empty when nothing changed."""

    forget: list[str] = field(default_factory=list)
    """Server `source` strings for entries the client no longer has.

    Populated only when the caller asks for deletion, because "absent locally" and "should be
    erased" are different claims: a client that has never seen a directory must not conclude the
    server should forget it.
    """

    oversize: list[str] = field(default_factory=list)
    """Files skipped for exceeding `max_file_bytes`, by name.

    Named rather than silently dropped. An oversized file will never fix itself, and a sync that
    quietly omits one leaves the user believing content is stored that is not.
    """

    unchanged: int = 0
    skipped_unreadable: list[str] = field(default_factory=list)
    """Files that could not be read. Not an error, and not silence either."""


def digest_of(path: Path) -> tuple[str, int] | None:
    """`(sha256, size)` as the SERVER would compute it, or `None` if the file cannot be read.

    Mirrors `recall/index.py`:

    * markdown is read as text with `utf-8-sig` (which strips a BOM), has NUL characters removed,
      and is hashed as UTF-8 — so a CRLF file and an LF file of the same content agree, because
      text mode normalises newlines;
    * anything else is hashed as its raw bytes.

    A file that vanished or cannot be decoded returns `None` rather than raising. The hooks must
    never take a session down, and a corpus routinely contains one unreadable thing.
    """
    try:
        if path.suffix.lower() in MARKDOWN_SUFFIXES:
            text = path.read_text(encoding="utf-8-sig").replace("\x00", "")
            data = text.encode("utf-8")
        else:
            data = path.read_bytes()
    except (OSError, UnicodeDecodeError):
        return None
    return hashlib.sha256(data).hexdigest(), len(data)


def scan(roots: list[tuple[str, Path]], glob: str = "**/*.md") -> dict[str, Change]:
    """Every file under each root, keyed by its root-prefixed relative name.

    `roots` is `(root_id, path)` pairs. Both of a project's memory directories are passed, not one:
    they are both real on machines today, and picking one silently drops whatever the other holds.

    A name that appears under two roots keeps the FIRST, because `roots` is ordered nearest-first
    and the nearer store is the one the user is working in.
    """
    found: dict[str, Change] = {}
    for root_id, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob(glob)):
            if not path.is_file():
                continue
            name = f"{root_id}/{path.relative_to(root).as_posix()}"
            if name in found:
                continue
            got = digest_of(path)
            if got is None:
                continue
            found[name] = Change(name=name, path=path, sha256=got[0], size=got[1])
    return found


def plan(
    local: dict[str, Change],
    remote: dict[str, str],
    limits: Limits | None = None,
    *,
    forget_missing: bool = False,
) -> SyncPlan:
    """Decide what to upload. Pure: no filesystem, no network, no clock.

    `remote` maps the server's `source` string to its stored digest, which is what
    `recall_inventory` returns. A remote entry with an empty digest is treated as UNKNOWN and
    therefore re-uploaded: an empty hash means the server cannot tell us what it holds, and
    assuming "unchanged" there would make a stale copy permanent.

    Matching is by SUFFIX, not equality. The server's `source` is a URI whose prefix is a staging
    path the client neither knows nor should care about; what both sides agree on is the relative
    name at the end of it.
    """
    limits = limits or Limits()
    remote_by_name = {source: digest for source, digest in remote.items()}

    def stored_digest(name: str) -> str | None:
        exact = remote_by_name.get(name)
        if exact is not None:
            return exact
        for source, digest in remote_by_name.items():
            if source.endswith("/" + name) or source.endswith(name):
                return digest
        return None

    changed: list[Change] = []
    oversize: list[str] = []
    unchanged = 0
    for name in sorted(local):
        change = local[name]
        if change.size > limits.max_file_bytes:
            oversize.append(name)
            continue
        stored = stored_digest(name)
        if stored and stored == change.sha256:
            unchanged += 1
            continue
        changed.append(change)

    batches: list[list[Change]] = []
    current: list[Change] = []
    current_bytes = 0
    for change in changed:
        too_many = len(current) + 1 > limits.max_files
        too_big = current_bytes + change.size > limits.max_bytes
        if current and (too_many or too_big):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(change)
        current_bytes += change.size
    if current:
        batches.append(current)

    forget: list[str] = []
    if forget_missing:
        local_names = set(local)
        for source in sorted(remote_by_name):
            tail = source.rsplit("/", 1)[-1]
            if not any(n == source or n.endswith("/" + tail) or n == tail for n in local_names):
                forget.append(source)

    return SyncPlan(
        upload=batches,
        forget=forget,
        oversize=oversize,
        unchanged=unchanged,
    )
