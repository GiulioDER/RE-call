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


# --------------------------------------------------------------------------- talking to a server
#
# Everything above decides. Everything below acts, and is deliberately separated so the decision
# stays testable without a socket.


class SyncError(RuntimeError):
    """A hosted call failed, carrying the KIND so a caller knows what to do about it."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


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
    message: str = ""


def manifest_path(config: dict) -> Path:
    """Per (endpoint, tenant), so pointing a machine at a different server starts a fresh cursor
    rather than inheriting beliefs about a corpus that server has never seen."""
    import hashlib as _h

    key = f"{config.get('endpoint', '')}|{config.get('tenant', '')}".encode()
    return _config_home() / f"recall-sync-{_h.sha256(key).hexdigest()[:12]}.json"


def _config_home() -> Path:
    import os

    override = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".claude"


def read_manifest(config: dict) -> dict:
    import json as _json

    try:
        data = _json.loads(manifest_path(config).read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return {"pending": {}}
    if not isinstance(data, dict):
        return {"pending": {}}
    data.setdefault("pending", {})
    return data


def write_manifest(config: dict, data: dict) -> None:
    """Atomically, and best effort. A manifest that cannot be written costs a re-upload next time,
    which is wasteful and safe; raising here would cost a session."""
    import json as _json
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
    except OSError:
        tmp.unlink(missing_ok=True)


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
    """
    try:
        from .screening import screen
    except ImportError as exc:
        return {}, {}, f"the credential screen could not be loaded ({exc})"
    allowed, withheld = screen(list(local.values()))
    return {change.name: change for change in allowed}, withheld, ""


def sync_memory_roots(
    roots: list[tuple[str, Path]],
    config: dict,
    *,
    limits: Limits | None = None,
    now: float | None = None,
) -> SyncOutcome:
    """Upload what changed. Never raises; the outcome is returned and recorded.

    The deadline is real rather than decorative. `SessionEnd` is registered async and so has NO
    client timeout, which means a hung TLS connection leaves an orphan process alive indefinitely
    rather than being cleaned up. Everything past the deadline stays pending, which is exactly the
    retry mechanism.
    """
    import time as _time

    from . import credentials as _cred

    started = _time.time() if now is None else now
    deadline = float(config.get("sync", {}).get("timeout_s", 120))
    endpoint = str(config.get("endpoint") or "")
    if not endpoint:
        return SyncOutcome(kind="refusal", message="hosted config names no endpoint")

    manifest = read_manifest(config)

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
    local, withheld, screen_error = _screened(scan(roots))
    manifest["withheld"] = {name: [str(f) for f in found] for name, found in withheld.items()}
    if screen_error:
        manifest["screen_error"] = screen_error
    else:
        manifest.pop("screen_error", None)

    try:
        head = _cred.headers(config)
    except Exception as exc:  # noqa: BLE001 - classified below, never raised into a session
        message = legible(exc)
        manifest["last_error"] = {"kind": "auth", "message": message}
        write_manifest(config, manifest)
        return SyncOutcome(kind="auth", withheld=len(withheld), message=message)

    try:
        remote = remote_inventory(endpoint, head)
        decided = plan(local, remote, limits)
    except SyncError as exc:
        manifest["last_error"] = {"kind": exc.kind, "message": exc.message}
        write_manifest(config, manifest)
        return SyncOutcome(kind=exc.kind, withheld=len(withheld), message=exc.message)

    if not decided.upload:
        manifest.pop("last_error", None)
        manifest["pending"] = {}
        write_manifest(config, manifest)
        return SyncOutcome(
            kind="noop", unchanged=decided.unchanged, withheld=len(withheld)
        )

    import base64 as _b64

    uploaded = 0
    pending: dict[str, str] = {name: "" for batch in decided.upload for name in
                              (c.name for c in batch)}
    for batch in decided.upload:
        if (_time.time() if now is None else now) - started > deadline:
            # Out of time. Everything not yet confirmed stays pending and the next run continues.
            break
        files = []
        for change in batch:
            try:
                files.append(
                    {
                        "name": change.name,
                        "content_b64": _b64.b64encode(change.path.read_bytes()).decode("ascii"),
                    }
                )
            except OSError:
                # Vanished between the scan and the read. Not an error, and not silence: it stays
                # out of this batch and out of `files`, so it is never recorded as synced.
                continue
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
                pending=len(pending), withheld=len(withheld), message=exc.message,
            )
        # Confirmed.
        for change in batch:
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
    )

