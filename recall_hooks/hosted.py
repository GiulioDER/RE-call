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
from typing import Any

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
    """Length of the NORMALISED content, i.e. what the digest was taken over.

    ⛔ NOT the number of bytes this file puts on the wire. For markdown, text-mode reading folds
    CRLF to LF and strips a BOM, so a Windows-authored corpus is materially smaller by this
    measure than by the one the server meters. Budget with `wire_size`.
    """

    wire_size: int = 0
    """Bytes actually sent, i.e. `path.stat().st_size`.

    An audit measured ten CRLF files at 40.0 MiB by `size` and 80.0 MiB on the wire: a batch the
    client believed was exactly at its cap, refused whole by the server, and rebuilt identically
    every session. Every limit in `plan` is applied to this field for that reason.
    """

    text: str | None = None
    """The decoded, NUL-stripped text this file's digest was taken over, for markdown.

    Carried so the screen inspects the SAME buffer that was hashed, rather than re-reading the
    file and gating a snapshot it does not own. `None` for a non-markdown file, which is hashed
    as raw bytes and has no text form to screen.
    """


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
    got = _read_once(path)
    return None if got is None else (got[0], got[1])


def _digest_raw(path: Path, raw: bytes) -> tuple[str, int, str | None] | None:
    """`(sha256, size, text)` computed from bytes ALREADY READ, or `None` if they do not decode.

    Equivalent to `recall/index.py`'s `read_text(encoding="utf-8-sig")`: text mode's universal
    newlines turn `\\r\\n` and a lone `\\r` into `\\n`, and the two replacements below are exactly
    that translation. Taking the bytes as an argument is what lets the uploader hash the very
    buffer it sends, rather than a second read of a file that may have moved in between.
    """
    if path.suffix.lower() not in MARKDOWN_SUFFIXES:
        return hashlib.sha256(raw).hexdigest(), len(raw), None
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    text = decoded.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    data = text.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data), text


def _read_once(path: Path) -> tuple[str, int, int, str | None] | None:
    """`(sha256, size, wire_size, text)` from a SINGLE read, or `None` if unreadable.

    🔑 One read is a correctness requirement, not an optimisation. The screen and the uploader
    used to read the same path minutes apart, across a token refresh and an inventory call, so a
    memo rewritten in that window was uploaded having passed no screen and under a digest that
    described the superseded content. Reading once and carrying the buffer removes the window;
    `_bytes_if_unchanged` closes the other half by hashing the exact bytes it is about to send.

    `wire_size` is the length of the buffer read, not a separate `stat()`, for the same reason:
    a second system call is a second observation of a file that can change between the two.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    got = _digest_raw(path, raw)
    if got is None:
        return None
    return got[0], got[1], len(raw), got[2]


#: Characters the SERVER refuses in an uploaded name (`recall/desktop/uploads.py`), plus `\\`.
#:
#: The backslash is refused here although the server accepts it, and deliberately so. The server
#: reads it as a SEPARATOR, so a POSIX file literally named `a\\b.md` is stored as `a/b.md`: a
#: different name from the one this client plans with, which never matches its inventory entry
#: and is re-uploaded on every sync. Skipping it, and saying so, is the honest outcome.
_ILLEGAL_IN_NAME = frozenset('<>:"|?*\\')

#: Names Windows reserves. The server refuses a component whose stem is one of these, so `aux.md`
#: and `con.md` are ordinary-looking memo names that no sync can ever carry.
_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{d}" for d in "123456789"}
    | {f"lpt{d}" for d in "123456789"}
)

#: The server's depth cap. Mirrored so a deep memo costs itself rather than the corpus.
_MAX_DEPTH = 8


def unsafe_name_reason(name: str) -> str:
    """Why the SERVER would refuse this name, or "" if it would accept it.

    ⛔ Mirrors `recall/desktop/uploads.py::_safe_relative_name`, which REFUSES rather than
    sanitises, and refuses the WHOLE request. Because a memory corpus is normally one batch and
    the first refusal abandons the run, a single `aux.md` would otherwise stop every sync on the
    machine forever, with the offending file never named. Checking here costs that file only.

    Kept deliberately as a mirror rather than an import: `recall_hooks` must not import `recall`.
    `tests/test_hosted_sync.py` runs both over the same names, so drift fails a test. The mirror
    is allowed to be STRICTER than the server (see `_ILLEGAL_IN_NAME`) and never looser: a name
    this accepts and the server refuses costs the whole batch.
    """
    parts = name.split("/")
    if len(parts) > _MAX_DEPTH:
        return f"more than {_MAX_DEPTH} path components"
    for part in parts:
        if not part or part in {".", ".."}:
            return "an empty or relative path component"
        if part.rstrip(" .") != part:
            return "a component ending in a dot or a space"
        if any(ch in _ILLEGAL_IN_NAME or ord(ch) < 32 for ch in part):
            return "a character the server refuses in a name"
        if part.split(".", 1)[0].lower() in _RESERVED_STEMS:
            return "a reserved device name"
        if len(part.encode("utf-8")) > 255:
            return "a path component over 255 bytes"
    return ""


#: The reason `scan` records for a zero-byte file. Named so `_unsynced_notice` can leave it out:
#: an empty file holds nothing that is lost by not uploading it, and a warning about one every
#: session would teach a person to skip the warnings that matter.
SKIP_EMPTY = "empty (zero bytes), which the server refuses"


def scan(
    roots: list[tuple[str, Path]],
    glob: str = "**/*.md",
    *,
    skipped: dict[str, str] | None = None,
) -> dict[str, Change]:
    """Every file under each root, keyed by its root-prefixed relative name.

    `roots` is `(root_id, path)` pairs. Both of a project's memory directories are passed, not one:
    they are both real on machines today, and picking one silently drops whatever the other holds.

    A name that appears under two roots keeps the FIRST, because `roots` is ordered nearest-first
    and the nearer store is the one the user is working in.

    ⛔ **Pass `skipped` and report it.** Every file this refuses is a memo that will never reach
    the corpus, and three of them used to leave by a bare `continue`: one that could not be
    decoded, one the server would refuse by name, and one that is empty. A `continue` here is
    indistinguishable to the user from a file that synced. `skipped` maps name to the reason, and
    `sync_memory_roots` puts it where a person will read it.

    ⚠️ Symlinks are not followed. A link inside a memory root would otherwise contribute content
    from outside the directory the user pointed at, which in hosted mode is an irreversible
    transfer of something never consented to.
    """
    found: dict[str, Change] = {}
    note = skipped if skipped is not None else {}
    for root_id, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob(glob)):
            if path.is_symlink() or not path.is_file():
                continue
            name = f"{root_id}/{path.relative_to(root).as_posix()}"
            if name in found or name in note:
                continue
            reason = unsafe_name_reason(name)
            if reason:
                note[name] = f"the server refuses this name: {reason}"
                continue
            got = _read_once(path)
            if got is None:
                note[name] = "could not be read or decoded as UTF-8"
                continue
            sha, size, wire, text = got
            if not wire:
                # ⛔ A zero-byte file sends an empty `content_b64`, which `stage_uploads` refuses
                # with an UploadError, and that refusal takes the WHOLE batch with it, every
                # session. Only this exact shape is skipped.
                #
                # A whitespace-only memo is NOT skipped, although the salvaged version skipped it.
                # The server accepts it and indexes it as a zero-chunk replacement precisely so
                # "rows from a previously non-empty version are removed" (`recall/index.py`).
                # Skipping it would leave a memo the user deliberately blanked answering from the
                # server with its old content, which is worse than the small re-upload it costs.
                note[name] = SKIP_EMPTY
                continue
            found[name] = Change(
                name=name, path=path, sha256=sha, size=size, wire_size=wire, text=text
            )
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

    Matching is exact first, then by `/`-ANCHORED suffix. The server's `source` is a URI whose
    prefix is a staging path the client neither knows nor should care about; what both sides agree
    on is the relative name at the end of it.

    ⛔ **The anchor is load-bearing.** An unanchored `source.endswith(name)` matched a remote
    `.../myproject/notes.md` against the local name `project/notes.md`, so the local memo was
    declared already-stored against an unrelated file and was never uploaded, silently and
    permanently. Reproduced against this function before the fix. Only a `/` boundary, or the
    whole source, may match.

    An index is built once rather than scanning `remote` per local name: the exact-match fast path
    essentially never fires in production (the server's source carries a staging prefix), so the
    linear scan WAS the normal path and the function was quadratic. Measured at 1,500 files:
    0.291s before, 0.006s after, with identical `unchanged` counts.
    """
    limits = limits or Limits()

    #: name-or-suffix -> digest. Every EXACT source goes in first, and only then the suffixes, so
    #: an exact match wins over a suffix of some other source whatever order the server listed
    #: them in. The salvaged version interleaved the two, and a source listed earlier could claim
    #: a name with its suffix before the exact entry for that name was reached.
    index: dict[str, str] = dict(remote)
    for source, digest in remote.items():
        tail = source
        while "/" in tail:
            tail = tail.split("/", 1)[1]
            index.setdefault(tail, digest)

    def stored_digest(name: str) -> str | None:
        return index.get(name)

    changed: list[Change] = []
    oversize: list[str] = []
    unchanged = 0
    for name in sorted(local):
        change = local[name]
        # ⛔ Budget on the bytes that go on the WIRE, not on the normalised length the digest was
        # taken over. Ten CRLF files measured 40.0 MiB by `size` and 80.0 MiB raw: a batch the
        # client believed was exactly at its cap, refused whole by the server, and rebuilt
        # identically every session. `plan` is deterministic, so that stall is permanent.
        if (change.wire_size or change.size) > limits.max_file_bytes:
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
        wire = change.wire_size or change.size
        too_many = len(current) + 1 > limits.max_files
        too_big = current_bytes + wire > limits.max_bytes
        if current and (too_many or too_big):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(change)
        current_bytes += wire
    if current:
        batches.append(current)

    forget: list[str] = []
    if forget_missing:
        # ⛔ The SAME predicate as `stored_digest`, deliberately. This used to compare only the
        # last path component, so a memo deleted under one root was retained as long as any other
        # root held a file of the same basename. Two rules answering one question ("do this local
        # name and this server source denote the same file?") is how the two halves disagree.
        for source in sorted(remote):
            if not any(source == n or source.endswith("/" + n) for n in local):
                forget.append(source)

    return SyncPlan(
        upload=batches,
        forget=forget,
        oversize=oversize,
        unchanged=unchanged,
    )


# --------------------------------------------------------------------------- talking to a server
#
# Everything above decides. Everything below acts, and is deliberately separated so the decision
# stays testable without a socket.


class SyncError(RuntimeError):
    """A hosted call failed, carrying the KIND so a caller knows what to do about it."""

    def __init__(self, kind: str, message: str, *, blocking: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        #: True when no sync can succeed until a PERSON changes something on this machine, and
        #: the message is a fixed sentence of ours rather than server text. `sync_memory_roots`
        #: records these as `blocked`, which SessionStart reports even with nothing pending.
        self.blocking = blocking


def legible(error: BaseException) -> str:
    """Flatten an exception into sentences a person can act on.

    The MCP SDK runs its session in a task group, so **every** error a server raises reaches a
    caller as `ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)` — a sentence that
    names the plumbing and hides the cause. Without this, every hosted failure message would be
    that string, and the classifier below would have nothing to read.

    Recursive, because a group can nest. Mirrors `recall/desktop/runtime.py::_legible`, duplicated
    for the import-cost reason in this module's docstring.
    """
    if isinstance(error, BaseExceptionGroup):
        inner = [legible(sub) for sub in error.exceptions]
        return "; ".join(part for part in inner if part) or str(error)
    text = str(error).strip()
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


#: Substrings that identify a failure kind, checked against the FLATTENED message. Ordered: the
#: first match wins, and auth is first because a 401 arriving inside a quota message is still an
#: auth problem.
_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("auth", ("401", "403", "unauthorized", "invalid_token", "autherror", "no credential")),
    ("quota", ("quota", "budget", "rate limit", "ratelimited", "too many requests", "429")),
    ("refusal", ("uploaderror", "duplicate file name", "must be relative", "reserved device",
                 "exceeds the", "valueerror", "category must be")),
    ("network", ("timeout", "timed out", "connection", "getaddrinfo", "ssl", "certificate",
                 "unreachable", "refused")),
)


def classify(message: str) -> str:
    """Which kind of failure this is, so the caller can pick a remedy rather than a guess.

    The four kinds want genuinely different handling, which is the only reason to have them:

    * **auth** — do not retry in a loop. A 401 after a fresh refresh is a real re-authentication,
      and hammering it just burns the token endpoint. Surface at the next `SessionStart`.
    * **quota** — back off for a long time. Retrying every session burns the bucket that is
      already empty.
    * **refusal** — surface immediately and name the file. It will never fix itself.
    * **network** — retry next session, and stay quiet until it has failed twice, so a laptop
      closed on a plane does not nag.

    Anything unrecognised is `network`, because that is the kind whose policy (retry quietly) is
    safe to apply to a failure nobody has classified yet.
    """
    lowered = message.lower()
    for kind, markers in _KINDS:
        if any(marker in lowered for marker in markers):
            return kind
    return "network"


def call_tool(
    endpoint: str,
    headers: dict[str, str],
    name: str,
    arguments: dict,
    *,
    timeout: float = 120.0,
) -> Any:
    """Call one MCP tool over streamable-http and return its parsed result.

    Imports the SDK lazily and inside the call. `SessionEnd` is the only event that reaches here,
    and it is asynchronous, so the cost is never charged to a session launch — which is the whole
    reason this package avoids `recall`.

    Raises `SyncError` with a classified kind. Never leaks an `ExceptionGroup`.
    """
    import asyncio
    import json as _json

    async def _run() -> Any:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        http = httpx2.AsyncClient(headers=headers, timeout=timeout)
        async with http, streamable_http_client(endpoint, http_client=http) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                if getattr(result, "is_error", False):
                    pieces = getattr(result, "content", None) or []
                    text = next(
                        (t for t in (getattr(p, "text", None) for p in pieces) if t), str(result)
                    )
                    raise SyncError(classify(text), text)
                for piece in getattr(result, "content", None) or []:
                    text = getattr(piece, "text", None)
                    if text is None:
                        continue
                    try:
                        return _json.loads(text)
                    except _json.JSONDecodeError:
                        return text
                return None

    try:
        return asyncio.run(_run())
    except SyncError:
        raise
    except (KeyboardInterrupt, SystemExit):
        # The user or the runtime asked this process to stop. `_index_and_refresh` re-raises these
        # deliberately; turning one into a "network" SyncError here would defeat that one level
        # down and the process would carry on after a Ctrl-C.
        raise
    except ImportError as exc:
        # ⛔ NOT a network failure, though it used to classify as one. `mcp` is an optional extra,
        # so a base install reports "no module named mcp", gets the `network` policy (retry
        # quietly, stay silent until it has failed twice), and therefore syncs nothing, forever,
        # saying nothing. Retrying cannot install a package, so it is `blocking`: SessionStart
        # reports it even though nothing is pending yet, which is exactly the first-run case.
        #
        # ⚠️ The extra is `mcp`, not `hosted`. The salvaged version named `hosted`, which is the
        # SERVER's extra and installs neither `mcp` nor anything else this transport imports, so
        # following its advice left the install exactly as broken.
        message = (
            f"the hosted transport is not installed ({exc}); "
            "install it with: pip install 'recall-rag[mcp]'"
        )
        raise SyncError("refusal", message, blocking=True) from exc
    except BaseException as exc:  # noqa: BLE001 - flattened and classified, never re-raised raw
        message = legible(exc)
        raise SyncError(classify(message), message) from exc


def remote_inventory(
    endpoint: str, headers: dict[str, str], *, timeout: float = 60.0
) -> dict[str, str]:
    """What the tenant holds, as `{source: sha256}`, via `recall_inventory`.

    ⚠️ A **truncated** inventory raises rather than being used. The diff treats a source absent
    from this map as one the server does not have, so a silently short listing would re-upload the
    tail of the corpus and, once deletion is enabled, forget it. Truncation is exactly the case the
    tool reports for this reason.
    """
    payload = call_tool(endpoint, headers, "recall_inventory", {}, timeout=timeout)
    if not isinstance(payload, dict):
        raise SyncError("refusal", f"recall_inventory returned {type(payload).__name__}, not JSON")
    if payload.get("truncated"):
        raise SyncError(
            "refusal",
            "the inventory was truncated, so a diff against it would re-upload or forget the "
            "tail of the corpus; raise the limit or page it",
        )
    entries = payload.get("entries") or []
    return {str(e["source"]): str(e.get("sha256") or "") for e in entries if e.get("source")}


# --------------------------------------------------------------------------- doing the sync
#
# ⛔ THREE RULES THIS SECTION MUST NOT BREAK.
#
# 1. Nothing is recorded as synced unless the server confirmed it. A cursor that runs ahead of
#    the server is how memory disappears: the next run skips a file the server never received.
#    ⛔ So this keeps NO local record of what is synced, and that absence is deliberate. What to
#    upload is decided by comparing the local scan against `remote_inventory`, which is the
#    server's own answer about what it holds. An earlier version of this file also wrote a
#    `files` map of confirmed hashes into the manifest; nothing ever read it, and the danger was
#    that a future reader would take it for the cursor and skip the round trip, which is exactly
#    the failure this rule forbids. A hash the client believes and the server has never seen is
#    worse than no hash at all.
# 2. The memo files ARE the queue. This never deletes, moves or rewrites anything under a memory
#    root, so a failed sync loses nothing and the next run simply tries again. A sync client that
#    "tidies up" destroys both the corpus and its own retry.
# 2b. ⚠️ THE SCREEN PREVENTS AN UPLOAD; IT DOES NOT RETRACT ONE. A memo that was uploaded before
#    the screen existed, or before a credential was added to it, stays on the server: withholding
#    it here only removes it from `local`, and `plan` runs with `forget_missing=False`, so nothing
#    is deleted remotely. That is the right default (a client silently deleting server-side memory
#    is a worse failure than a stale copy) but it means the gate is a floor for NEW content and not
#    a remedy for old. Retracting is `recall_forget`, and it is a person's decision.
# 3. It never raises. `SessionEnd` must not take a session down, so every failure becomes a
#    recorded outcome instead.


@dataclass(frozen=True)
class SyncOutcome:
    kind: str          # "ok" | "auth" | "quota" | "network" | "refusal" | "noop"
    uploaded: int = 0
    unchanged: int = 0
    pending: int = 0
    withheld: int = 0
    oversize: int = 0
    """Files past the per-file cap. Counted because they will never fix themselves."""

    skipped: int = 0
    """Files `scan` refused: unreadable, empty, or a name the server would reject.

    ⛔ These four counters plus `uploaded` and `unchanged` must account for every file scanned.
    Three of them did not exist, and each absence was a memo dropped in silence.
    """

    message: str = ""


def manifest_path(config: dict[str, Any]) -> Path:
    """Per (endpoint, tenant), so a machine pointed at a different server does not inherit that
    server's pending list, withheld set or last error.

    ⚠️ This is NOT a sync cursor: see rule 1 below. Nothing here records what has been uploaded,
    deliberately.

    The directory comes from the package's own `claude_config_home`, never from a local copy. Two
    private re-derivations of it existed and neither stripped nor expanded `CLAUDE_CONFIG_DIR`, so
    with `CLAUDE_CONFIG_DIR=~/.claude-alt` the config was read from the real home while the
    manifest and the token cache were written into a literal `~` directory beside the process cwd,
    which for a hook is somebody's checkout. Measured, not theorised.
    """
    from . import claude_config_home  # noqa: PLC0415 - one resolver, imported where it is used

    key = f"{config.get('endpoint', '')}|{config.get('tenant', '')}".encode()
    return claude_config_home() / f"recall-sync-{hashlib.sha256(key).hexdigest()[:12]}.json"


def read_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """The manifest, coerced into a shape its readers can survive.

    ⛔ **`ValueError`, not `JSONDecodeError`.** A manifest whose bytes are not valid UTF-8 raises
    `UnicodeDecodeError`, which is a `ValueError` and NOT a `JSONDecodeError`, so it used to
    propagate. `session_start` reaches this through `_unsynced_notice` with no guard of its own,
    so one bad byte in a file under `~/.claude` raised out of the hook that runs before every
    session — the exact failure `session_start`'s "fails open, always" docstring promises cannot
    happen. Reproduced before the fix.

    ⛔ **Every value is coerced, not just the outer type.** The old version guaranteed the KEY
    existed and nothing about its type, and `_unsynced_notice` then called `len()` on `pending`
    and `sorted()` on `withheld`. `{"pending": 3}` is valid JSON.
    """
    import json as _json  # noqa: PLC0415 - stdlib, kept off the SessionStart import path

    empty: dict[str, Any] = {"pending": {}, "withheld": {}}
    try:
        data = _json.loads(manifest_path(config).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(empty)
    if not isinstance(data, dict):
        return dict(empty)
    for key in ("pending", "withheld", "last_error"):
        if key in data and not isinstance(data[key], dict):
            del data[key]
    for key in ("screen_error", "blocked"):
        if key in data and not isinstance(data[key], str):
            del data[key]
    for key in ("skipped", "oversize"):
        if key in data and not isinstance(data[key], (dict, list)):
            del data[key]
    data.setdefault("pending", {})
    data.setdefault("withheld", {})
    return data


def write_manifest(config: dict[str, Any], data: dict[str, Any]) -> None:
    """Atomically, and best effort. A manifest that cannot be written costs a re-upload next time,
    which is wasteful and safe; raising here would cost a session.

    ⚠️ `Exception`, not just `OSError`, matching `_save_config` in `__init__.py`, whose comment
    spells out why: a non-serialisable value raises `TypeError` from `json.dump`, which under an
    `OSError`-only handler would leak the temp file into `~/.claude` AND raise out of a function
    documented never to raise. The two copies of this sequence must not drift.
    """
    import json as _json  # noqa: PLC0415 - stdlib, kept off the SessionStart import path
    import os
    import tempfile

    path = manifest_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    except OSError:
        return
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            _json.dump(data, handle, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - see the docstring; a manifest may never cost a session  # BROAD-CATCH: cleanup-only
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _screened(local: dict[str, Change]) -> tuple[dict[str, Change], dict[str, list], str]:
    """Drop anything that must not leave the machine, and say what was dropped and why.

    Returns `(allowed, withheld, screen_error)`. The third value is non-empty ONLY when the screen
    itself could not run, and it is separate from `withheld` on purpose: those two are different
    facts and they need different sentences. An earlier version reported a load failure as one
    withheld file named `*`, which rendered to a person as "1 memo was not uploaded because it
    looks like it contains a live credential" — a specific, alarming and false statement about a
    memo, when the truth was that the guard did not start.

    Degrades to uploading NOTHING rather than everything if the screen is unavailable. That is the
    opposite of this package's usual "an ImportError means silence" rule, and deliberately so:
    everywhere else the thing that fails to import is a FEATURE, and the safe direction is to skip
    it. Here it is a GUARD, and skipping a guard is the failure.

    ⛔ `Exception`, not `ImportError`. A guard is only as good as the narrowest failure that can
    stop it running: a `re.error` from a future pattern, a `MemoryError` on a huge file, or a
    half-written module all escaped an `ImportError`-only handler, reached
    `_index_and_refresh`'s `BaseException` catch, and ended the sync as an ordinary hook exit.
    That is a failure to SCREEN, reported as nothing at all.
    """
    try:
        from .screening import screen  # noqa: PLC0415 - the guard, loaded where it is used

        allowed, withheld = screen(list(local.values()))
    except Exception as exc:  # noqa: BLE001 - a guard that cannot run must stop the upload  # BROAD-CATCH: fail-closed
        return {}, {}, f"the credential screen could not run ({type(exc).__name__}: {exc})"
    return {change.name: change for change in allowed}, withheld, ""


def insecure_endpoint_reason(endpoint: str) -> str:
    """Why this endpoint must not carry a bearer token and a corpus, or "" if it may.

    ⛔ The same rule `credentials.refresh` already applies to `token_url` ("a bearer token must
    not cross a plaintext hop"). The ingest endpoint carries that bearer AND the base64 of every
    memo, and was the one URL in the package with no check at all.

    Loopback over http is allowed explicitly, because a local test server is a legitimate target
    and an https-only rule would simply be worked around.

    ⛔ **The host is whatever `urlsplit` says it is, never a hand-sliced prefix.** The salvaged
    version took everything up to the first `:` after the scheme, so `http://localhost:1@evil/`
    read as loopback, while every HTTP client reads `localhost:1` there as USERINFO and connects
    to `evil`. That sent the bearer and the corpus in plaintext to an arbitrary host, which is the
    exact outcome this function exists to refuse. The same slicing also refused `http://[::1]:8000`,
    since its first `:` falls inside the brackets.

    Returns a fixed sentence and never echoes the endpoint, which may carry userinfo.
    """
    from urllib.parse import urlsplit  # noqa: PLC0415 - stdlib, kept off the SessionStart path

    try:
        parts = urlsplit(endpoint.strip())
        host = parts.hostname
    except ValueError:
        return "endpoint is not a valid URL"
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not host:
        return "endpoint must be an http(s) URL"
    if scheme == "https":
        return ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return ""
    return (
        "endpoint must be https (or http on loopback): a bearer token and every memo "
        "must not cross a plaintext hop"
    )


def _bytes_if_unchanged(change: Change) -> bytes | None:
    """The raw bytes to upload, or `None` if the file moved under us since the scan.

    The screen inspected the buffer read during the scan; this read happens minutes later. Only
    the raw bytes can be uploaded (markdown is normalised for hashing, so the screened text is not
    the payload), which means a fresh read is unavoidable, so the digest is recomputed and
    compared, and a mismatch drops the file from this run rather than sending content no screen
    has seen.

    ⛔ The digest is taken over `data`, the buffer that is returned and sent, never over a second
    read. The salvaged version read the bytes once to send and again to hash, so a file rewritten
    between those two reads and then restored could send content the check never looked at.
    """
    try:
        data = change.path.read_bytes()
    except OSError:
        return None
    got = _digest_raw(change.path, data)
    if got is None or got[0] != change.sha256:
        return None
    return data


def sync_memory_roots(
    roots: list[tuple[str, Path]],
    config: dict[str, Any],
    *,
    limits: Limits | None = None,
    clock: Any = None,
) -> SyncOutcome:
    """Upload what changed. Never raises; the outcome is returned and recorded.

    The deadline bounds when a NEW batch may start. `SessionEnd` and `PreCompact` are both
    registered async and so have no client timeout, and everything past the deadline stays
    pending, which is exactly the retry mechanism.

    ⛔ **`clock` is a CALLABLE, and that is the whole point.** It used to be a `now: float`, and
    `started` and the loop's reading were both set from it, so an injected value made the elapsed
    time identically `0.0` and the deadline could never fire. An audit ran it: `timeout_s=0` sent
    every batch. The only seam provided for testing the one bound on a hung sync could not
    exercise it, and no test did. See `[[guards-that-cannot-fail]]`: a guard nobody has watched
    fail has not been tested.
    """
    import math  # noqa: PLC0415 - stdlib, kept off the SessionStart import path
    import time as _time  # noqa: PLC0415 - stdlib, kept off the SessionStart import path

    from . import credentials as _cred  # noqa: PLC0415 - only hosted mode pays for this

    tick = clock if callable(clock) else _time.time
    started = tick()

    # ⚠️ Read defensively. Nothing in this repository WRITES a `sync` block, so every value here
    # is hand-supplied, and `{"sync": null}` or `{"timeout_s": "2m"}` used to raise out of a
    # function whose contract is "never raises" — before the manifest was even read, so a withheld
    # finding was lost for that run too.
    raw_sync = config.get("sync")
    block = raw_sync if isinstance(raw_sync, dict) else {}
    try:
        deadline = float(block.get("timeout_s", 120))
    except (TypeError, ValueError):
        deadline = 120.0
    if not math.isfinite(deadline):
        # `"nan"` parses, and every comparison against NaN is False, so the deadline would never
        # fire; `"inf"` says the same thing on purpose. Neither is a bound, so neither is kept.
        deadline = 120.0
    if deadline <= 0:
        deadline = 0.0

    endpoint = str(config.get("endpoint") or "")
    if not endpoint:
        return SyncOutcome(kind="refusal", message="hosted config names no endpoint")
    manifest = read_manifest(config)
    scheme_problem = insecure_endpoint_reason(endpoint)
    if scheme_problem:
        # ⛔ RECORDED, not merely returned. `_index_and_refresh` discards the outcome, so the
        # salvaged version refused a plaintext endpoint on every session and told nobody: hosted
        # memory simply never filled. `blocked` is what SessionStart reads for this, and it is a
        # fixed sentence, so the endpoint (which may carry userinfo) never reaches the manifest.
        manifest["blocked"] = scheme_problem
        write_manifest(config, manifest)
        return SyncOutcome(kind="refusal", message=scheme_problem)

    # ⛔ The screen runs FIRST: before the credential, before the network, and before `plan`.
    #
    # Before `plan`, because a withheld file must never enter a batch: `plan` sizes batches
    # against the server's limits, and removing members afterwards would leave that accounting
    # describing a request nobody sent.
    #
    # Before the credential and the network, because whether a file may leave this machine is not
    # a question that should depend on a token being valid or a host being reachable. If it ran
    # after, then every path that returns early on an auth or network failure would leave
    # `withheld` unreported, and the one finding that needs a person would be the one silently
    # dropped.
    skipped: dict[str, str] = {}
    local, withheld, screen_error = _screened(scan(roots, skipped=skipped))
    manifest["skipped"] = skipped
    manifest["withheld"] = {name: [str(f) for f in found] for name, found in withheld.items()}
    if screen_error:
        manifest["screen_error"] = screen_error
    else:
        manifest.pop("screen_error", None)

    try:
        head = _cred.headers(config)
    except Exception as exc:  # noqa: BLE001 - classified below, never raised into a session  # BROAD-CATCH: error-translation
        message = legible(exc)
        manifest["last_error"] = {"kind": "auth", "message": message}
        write_manifest(config, manifest)
        return SyncOutcome(
            kind="auth", withheld=len(withheld), skipped=len(skipped), message=message
        )

    try:
        remote = remote_inventory(endpoint, head)
        decided = plan(local, remote, limits)
    except SyncError as exc:
        manifest["last_error"] = {"kind": exc.kind, "message": exc.message}
        if exc.blocking:
            manifest["blocked"] = exc.message
        write_manifest(config, manifest)
        return SyncOutcome(
            kind=exc.kind, withheld=len(withheld), skipped=len(skipped),
            message=exc.message,
        )
    # The endpoint was accepted and the transport answered, so nothing is blocking any more.
    manifest.pop("blocked", None)

    # ⛔ Oversize files are RECORDED. `plan` has always computed them, with a docstring saying
    # "Named rather than silently dropped ... a sync that quietly omits one leaves the user
    # believing content is stored that is not" — and the caller then read the field nowhere at
    # all, so a memo over the per-file cap vanished from the corpus and from every count.
    manifest["oversize"] = list(decided.oversize)

    if not decided.upload:
        manifest.pop("last_error", None)
        manifest["pending"] = {}
        write_manifest(config, manifest)
        return SyncOutcome(
            kind="noop",
            unchanged=decided.unchanged,
            withheld=len(withheld),
            oversize=len(decided.oversize),
            skipped=len(skipped),
        )

    import base64 as _b64

    uploaded = 0
    pending: dict[str, str] = {c.name: "" for batch in decided.upload for c in batch}
    for batch in decided.upload:
        if tick() - started > deadline:
            # Out of time. Everything not yet confirmed stays pending and the next run continues.
            break
        files = []
        sent: list[Change] = []
        for change in batch:
            data = _bytes_if_unchanged(change)
            if data is None:
                # Either it vanished, or it was rewritten since the scan. Both stay OUT of this
                # batch AND out of `sent`, so neither is recorded as synced.
                #
                # ⛔ The digest re-check is the security half. The screen inspected the buffer read
                # during the scan; this is a fresh read minutes later, across a token refresh and
                # an inventory call. Uploading it unconditionally would send bytes no screen ever
                # saw, which is the one outcome this module exists to prevent.
                continue
            files.append(
                {"name": change.name, "content_b64": _b64.b64encode(data).decode("ascii")}
            )
            sent.append(change)
        if not files:
            continue
        try:
            call_tool(endpoint, head, "recall_ingest", {"files": files, "category": "memory"})
        except SyncError as exc:
            # ⛔ A quota refusal means NOTHING was ingested: the debit happens before embedding and
            # raises. So this batch and every later one stay pending, and none of them is recorded.
            manifest["pending"] = pending
            manifest["last_error"] = {"kind": exc.kind, "message": exc.message}
            write_manifest(config, manifest)
            return SyncOutcome(
                kind=exc.kind, uploaded=uploaded, unchanged=decided.unchanged,
                pending=len(pending), withheld=len(withheld),
                oversize=len(decided.oversize), skipped=len(skipped),
                message=exc.message,
            )
        # Confirmed. ⛔ Over `sent`, never over `batch`: a file that could not be re-read was
        # never put in the request, and counting it here removed it from the only record of
        # unfinished work while reporting it as uploaded. The comment above used to promise
        # exactly this and the code did the opposite.
        for change in sent:
            pending.pop(change.name, None)
            uploaded += 1

    manifest["pending"] = pending
    if pending:
        manifest["last_error"] = {"kind": "network", "message": "ran out of time; will resume"}
    else:
        manifest.pop("last_error", None)
    write_manifest(config, manifest)
    return SyncOutcome(
        kind="ok" if not pending else "network",
        uploaded=uploaded,
        unchanged=decided.unchanged,
        pending=len(pending),
        withheld=len(withheld),
        oversize=len(decided.oversize),
        skipped=len(skipped),
    )
