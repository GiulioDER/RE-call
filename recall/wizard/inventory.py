"""Turn a local directory into the object inventory `recall manifest create --objects` consumes.

Calibration binds to a generation, a generation is built from a manifest, and a manifest is built
from an inventory. `file://` manifests exist (see `recall/lineage.py:251`) precisely so a corpus in
a directory can become a generation without object storage — but nothing produced that inventory,
so the capability was documented and unreachable. This is the producer.

Two properties are load-bearing, and both are inherited rather than reimplemented:

* **The walk is `recall.index.candidate_files`.** It is the same walk `index_path` performs, so what
  an inventory claims and what indexing would read cannot drift apart, and its refusal of a single
  file outside the glob is a security boundary this must not widen.
* **`version_id` is the content digest.** For a `file://` object that is not a convention, it is
  enforced by `ManifestObjectV1.__post_init__`: a local file has no version other than its bytes,
  and any other value would name an immutability guarantee the filesystem cannot provide.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from recall.index import candidate_files
from recall.lint import DEFAULT_GLOB

#: Last resort when nothing else claims the extension. `ManifestObjectV1` refuses an empty
#: media_type, and `mimetypes.guess_type` returns None for plenty of real corpus files (`.mdx`,
#: extensionless), so a fallback is required rather than merely tidy.
FALLBACK_MEDIA_TYPE = "application/octet-stream"

#: Consulted BEFORE `mimetypes`, not after. The stdlib table is built for serving files to
#: browsers and gets source extensions wrong in ways that matter here: `.ts` resolves to
#: `video/vnd.dlna.mpeg-tts`, which would label every TypeScript file in a code corpus as video.
#: These entries are the formats recall actually chunks.
MEDIA_TYPES: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mdx": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
    ".py": "text/x-python",
    ".pyi": "text/x-python",
    ".ts": "text/x-typescript",
    ".tsx": "text/x-typescript",
    ".js": "text/javascript",
    ".jsx": "text/javascript",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".java": "text/x-java",
    ".rb": "text/x-ruby",
    ".c": "text/x-c",
    ".h": "text/x-c",
    ".cpp": "text/x-c++",
    ".hpp": "text/x-c++",
    ".cs": "text/x-csharp",
    ".sh": "text/x-shellscript",
    ".sql": "text/x-sql",
    ".toml": "text/x-toml",
    ".yaml": "text/x-yaml",
    ".yml": "text/x-yaml",
    ".json": "application/json",
}


def media_type_for(path: Path) -> str:
    """The media type recorded for `path`, never empty."""
    suffix = path.suffix.lower()
    if suffix in MEDIA_TYPES:
        return MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or FALLBACK_MEDIA_TYPE


#: Read size for the streaming digest. Bounds peak memory at one buffer rather than one file, so
#: a stray multi-gigabyte file matched by a wide glob cannot end the run with a `MemoryError`.
_READ_CHUNK_BYTES = 1024 * 1024


def _entry(path: Path) -> dict[str, Any]:
    """One inventory entry for `path`.

    Size and digest come from ONE open handle, streamed. Taking the size from `stat()` and the
    digest from a separate read would describe two different states of a file being written
    concurrently, and `LocalObjectReader.fetch` checks length before digest, so the mismatch would
    surface as a size error naming neither cause.

    The URI is the WALKED path, not `path.resolve()`. Resolving looked more careful and was wrong:
    `_confined_to` keeps the path it walked while filtering on the resolved one, so a symlink and
    its target both inside the corpus are two walked paths, and resolving collapsed them to one
    URI. `IndexManifestV1` then refused the whole manifest for a duplicate URI, naming no file.
    Using the walked path also keeps this module's claim honest: `index_path` indexes both as two
    distinct sources, so an inventory that merged them would describe a different corpus than the
    one indexing would read. The reader resolves before its containment check, so a symlink still
    reads its target's bytes, which is what the digest here is taken over.
    """
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    hexdigest = digest.hexdigest()
    return {
        # `as_uri()` percent-encodes, which is what `url2pathname` reverses on the reader side.
        # It requires an absolute path; `candidate_files` resolves the root before walking, so
        # every path it yields is already absolute.
        "uri": path.as_uri(),
        "version_id": hexdigest,
        "media_type": media_type_for(path),
        "size": size,
        "sha256": hexdigest,
    }


@dataclass(frozen=True)
class InventoryReport:
    """What an inventory run produced, including what it could not read.

    `vanished` is carried out to the caller rather than logged or dropped. A file that disappeared
    between the walk and the read is absorbed so one deletion cannot end the run, but absorbing it
    silently would mean an inventory could describe fewer files than the corpus holds with nothing
    saying so — and the corpus fingerprint that gets calibrated is computed from exactly this list.
    """

    entries: list[dict[str, Any]]
    vanished: int

    @property
    def written(self) -> int:
        return len(self.entries)


def build_inventory_report(root: str | Path, glob: str = DEFAULT_GLOB) -> InventoryReport:
    """Inventory entries for every file under `root` matching `glob`, sorted by URI.

    Refuses an empty result. `IndexManifestV1` accepts an empty object tuple, so an empty corpus
    would build a generation holding nothing and calibration would go on to measure an empty index
    — a green run that certifies nothing. `candidate_files` refuses a glob mismatch for the same
    reason, and this extends that refusal to the case where the glob matched but the tree was bare.
    """
    if PurePath(glob).is_absolute() or glob.startswith(("/", "\\")):
        # `Path.glob` raises NotImplementedError("Non-relative patterns are unsupported"), which
        # is neither ValueError nor OSError, so it escaped the CLI's catch as a bare traceback on
        # what is a natural first guess for somebody who has just been shown an absolute path.
        raise ValueError(
            f"the glob {glob!r} is absolute. It is applied relative to the path argument, so pass "
            f"a pattern like '**/*.md' and give the directory as the path."
        )
    files = candidate_files(root, glob)
    entries: list[dict[str, Any]] = []
    vanished = 0
    for file in files:
        try:
            entries.append(_entry(file))
        except (FileNotFoundError, NotADirectoryError):
            # A file listed by the walk and gone by the read. `index_path` absorbs exactly this
            # (recall/index.py:575), and an inventory that aborted the whole corpus over one
            # deleted file would be strictly worse than the walk it claims to mirror. Counted, not
            # silent: the caller reports it.
            vanished += 1
        except OSError as exc:
            # Everything else is named. On Windows the common one is a file held open by another
            # process, and `[Errno 13]` alone tells the reader nothing about which file or how far
            # the run got.
            raise OSError(
                f"cannot read {str(file)!r} while building the inventory "
                f"({type(exc).__name__}: {exc}). {len(entries)} of {len(files)} file(s) had been "
                f"read. Close anything holding the file open, or narrow the glob."
            ) from exc
    if not entries:
        raise ValueError(
            f"no files under {str(Path(root).resolve())!r} match the glob {glob!r}, so there is "
            f"nothing to index. An empty inventory builds an empty generation, which calibrates "
            f"against nothing and reports success. Check the path and the glob."
        )
    return InventoryReport(
        entries=sorted(entries, key=lambda e: e["uri"]),
        vanished=vanished,
    )


def build_inventory(root: str | Path, glob: str = DEFAULT_GLOB) -> list[dict[str, Any]]:
    """Just the entries, for callers that do not need the skip count."""
    return build_inventory_report(root, glob).entries


def write_inventory(
    root: str | Path,
    output: str | Path,
    glob: str = DEFAULT_GLOB,
) -> InventoryReport:
    """Write the inventory for `root` to `output` as JSON, and report what it contains.

    `newline="\\n"` because a manifest's identity is the digest of its bytes: letting Python
    translate to CRLF on Windows would give the same corpus two different corpus fingerprints
    depending on which machine built it, and a fingerprint change is what
    `CALIBRATION_STALE` reports.
    """
    report = build_inventory_report(root, glob)
    path = Path(output)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report.entries, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report
