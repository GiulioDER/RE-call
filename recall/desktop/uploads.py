"""Bounded upload staging used by the desktop to hand files to an MCP runtime."""

from __future__ import annotations

import base64
import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path

from recall.errors import RecallError
from recall.observability import get_logger

_LOG = get_logger("desktop.uploads")

_MAX_TOTAL_BYTES = 50 * 1024 * 1024
_MAX_MIB = _MAX_TOTAL_BYTES // (1024 * 1024)
_OVERSIZE_MSG = f"upload exceeds the {_MAX_MIB} MiB request limit"


class UploadError(ValueError, RecallError):
    """Raised when a desktop upload is malformed or exceeds its safety limits."""


#: Deepest relative path a staged upload may carry. A memory directory is organised by hand, so
#: eight levels is generous; the cap exists so a hostile name cannot force an unbounded mkdir
#: chain, and so a path stays inside the filesystem's own limits on every platform.
_MAX_UPLOAD_DEPTH = 8

#: Windows refuses these as filenames whatever the extension, and a server staging a hostile name
#: on Windows would fail deep inside `write_bytes` rather than at the boundary. Checked on every
#: platform so a Linux server and a Windows one accept exactly the same uploads.
_RESERVED_WINDOWS_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{d}" for d in "123456789"}
    | {f"lpt{d}" for d in "123456789"}
)


def _safe_relative_name(raw: str) -> str:
    """A caller-supplied upload name as a safe relative POSIX path, or raise `UploadError`.

    This replaced `Path(raw).name`, which was safe and lossy: it made traversal impossible by
    throwing the directories away, and in doing so made `notes/a.md` and `archive/a.md` collide
    into one `a.md` — so any upload from a tree with subdirectories tripped the duplicate-name
    refusal and failed as a whole, permanently.

    Keeping the path means every hazard the basename collapse used to absorb now has to be refused
    deliberately. It **refuses rather than sanitises**, because a silently rewritten name stages
    content under a path the caller never asked for, and a sync client's manifest would then
    disagree with the server about what is stored, which is a divergence nothing would report.
    """
    if not isinstance(raw, str) or not raw:
        raise UploadError("every uploaded file needs name and content_b64")
    if "\x00" in raw:
        raise UploadError("upload name contains a NUL byte")
    # A Windows client sends `notes\a.md`. On POSIX that is one component, so normalise before
    # anything else looks at the parts, or the checks below would inspect the wrong shape and the
    # server would create a file whose name contains a separator.
    candidate = raw.replace("\\", "/")
    if candidate.startswith("/"):
        raise UploadError(f"upload name must be relative, got {raw!r}")
    # `C:/x` is absolute on Windows and a plain relative path on POSIX, so it cannot be left to
    # `PurePath` to classify: the same manifest must be accepted or refused identically on both.
    head = candidate.split("/", 1)[0]
    if len(head) == 2 and head[1] == ":" and head[0].isalpha():
        raise UploadError(f"upload name must not carry a drive letter, got {raw!r}")
    parts = [part for part in candidate.split("/") if part not in ("", ".")]
    if not parts:
        raise UploadError(f"upload name names no file, got {raw!r}")
    if any(part == ".." for part in parts):
        raise UploadError(f"upload name must not traverse upwards, got {raw!r}")
    if candidate.endswith("/"):
        raise UploadError(f"upload name must name a file, not a directory, got {raw!r}")
    if len(parts) > _MAX_UPLOAD_DEPTH:
        raise UploadError(
            f"upload name is deeper than {_MAX_UPLOAD_DEPTH} levels, got {raw!r}"
        )
    for part in parts:
        if len(part.encode("utf-8")) > 255:
            raise UploadError(f"upload name component is longer than 255 bytes in {raw!r}")
        # ⛔ Windows silently strips trailing dots and spaces from a path component, so `.. `
        # becomes `..` AFTER the traversal check above has already accepted it, and `x.` becomes
        # a different file than the caller named. Found by trying it: `notes/.. /x.md` passed
        # every check here and then died inside `write_bytes` with an uncaught FileNotFoundError,
        # which is not an UploadError and so escapes every caller that handles refusals.
        # Refuse the shape rather than trusting the resolve() check to catch its consequences.
        if part.rstrip(" .") != part:
            raise UploadError(
                f"upload name component ends with a dot or space, which Windows strips, "
                f"in {raw!r}"
            )
        if part.split(".", 1)[0].lower() in _RESERVED_WINDOWS_NAMES:
            raise UploadError(f"upload name uses a reserved device name in {raw!r}")
    return "/".join(parts)


def _tenant_uploads_root(tenant: str) -> Path:
    """Where one tenant's staged uploads live. The single source of this path.

    Staging (`stage_uploads`) and erasure (`delete_staged_sources`) both derive it: if they
    ever disagreed, `delete_staged_sources`'s confinement check would silently skip every
    file and erasure would stop happening while still reporting success. One helper makes
    that divergence impossible.
    """
    return Path(os.environ.get("RECALL_INDEX_ROOT", ".")).resolve() / "uploads" / tenant


def delete_staged_sources(tenant: str, sources: Iterable[str]) -> int:
    """Unlink staged upload files whose DB rows were just erased. Returns files removed.

    Without this, "permanently delete" left the original text on the server filesystem,
    inside the index root, where the next index run over `uploads/` would re-ingest exactly
    the content the caller was told is gone. Best effort by design (the DB delete is already
    committed), and hard-confined to `RECALL_INDEX_ROOT/uploads/<tenant>/`: a forgotten
    source indexed from the user's own directory must NEVER delete the user's file.

    Accepts both source spellings: `file://` URIs (generation-mode manifests) and plain
    absolute paths (the legacy `source` column). Lives HERE beside `stage_uploads` because
    recall_mcp is documented, and AST-checked, as making zero direct file-write calls.
    """
    from urllib.parse import urlsplit
    from urllib.request import url2pathname

    uploads_root = _tenant_uploads_root(tenant)
    removed = 0
    touched_dirs: set[Path] = set()
    for source in sources:
        raw = str(source)
        if raw.startswith("file://"):
            raw = url2pathname(urlsplit(raw).path)
        try:
            path = Path(raw).resolve()
        except (OSError, ValueError):
            continue
        if not path.is_relative_to(uploads_root):
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            _LOG.warning("could not remove staged file for a forgotten source: %s", path)
            continue
        removed += 1
        touched_dirs.add(path.parent)
    for directory in touched_dirs:
        try:
            directory.rmdir()  # only succeeds when empty, which is the point
        except OSError:
            pass
    return removed


def discard_staging(root: Path) -> None:
    """Remove one job's staging tree, best effort. The failure-path counterpart of
    `stage_uploads`: a refused or failed ingest must not leave content inside the index
    root for a later index run to resurrect."""
    shutil.rmtree(root, ignore_errors=True)


def stage_uploads(tenant: str, files: list[dict[str, str]]) -> tuple[str, Path, int]:
    """Decode a bounded upload into the configured server-side staging root.

    Returns (job_id, staging_root, total_bytes). The byte total is what the caller
    debits against the tenant's index-bytes quota; measuring it here keeps the
    figure identical to what was actually written.

    On any refusal the job's staging directory is removed, so a rejected upload
    never leaves partial files behind for a later index run to pick up.
    """
    if not files or len(files) > 500:
        raise UploadError("files must contain between 1 and 500 entries")
    job_id = uuid.uuid4().hex
    root = _tenant_uploads_root(tenant) / job_id
    total = 0
    seen: set[str] = set()
    root.mkdir(parents=True, exist_ok=True)
    try:
        for item in files:
            encoded = item.get("content_b64", "")
            if not encoded:
                raise UploadError("every uploaded file needs name and content_b64")
            name = _safe_relative_name(item.get("name", ""))
            if name in seen:
                # Two entries with one name would silently drop the first from the
                # indexed set; refuse rather than guess which one was meant. Keyed on the
                # relative path, so two files that merely share a basename no longer collide.
                raise UploadError(f"duplicate file name {name!r} in one upload")
            seen.add(name)
            # Refuse an oversized entry before materialising it: base64 encodes 3
            # bytes into 4 characters, so the encoded length bounds the decoded
            # size without decoding. The post-decode check below stays as the
            # exact enforcement.
            if (len(encoded) // 4) * 3 > _MAX_TOTAL_BYTES - total:
                raise UploadError(_OVERSIZE_MSG)
            try:
                data = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise UploadError(f"invalid base64 content for {name!r}") from exc
            total += len(data)
            if total > _MAX_TOTAL_BYTES:
                raise UploadError(_OVERSIZE_MSG)
            target = root / name
            # Belt and braces after normalisation: `_safe_relative_name` has already refused
            # every traversal shape, so this can only fire if that function is weakened. It is
            # cheap, and it is the check that would notice.
            if not target.resolve().is_relative_to(root.resolve()):
                raise UploadError(f"upload name escapes the staging root: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    except BaseException:
        discard_staging(root)
        raise
    return job_id, root, total
