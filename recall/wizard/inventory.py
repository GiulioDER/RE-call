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
from pathlib import Path
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


def _entry(path: Path) -> dict[str, Any]:
    """One inventory entry for `path`.

    The bytes are read ONCE and both `size` and `sha256` derive from that single read. Taking the
    size from `stat()` and the digest from a separate read would describe two different states of a
    file being written concurrently, and `LocalObjectReader.fetch` checks length before digest, so
    the mismatch would surface as a size error naming neither cause.
    """
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return {
        # `as_uri()` percent-encodes, which is what `url2pathname` reverses on the reader side.
        # `resolve()` first, because a manifest entry has to name one absolute path and the reader
        # resolves before its containment check.
        "uri": path.resolve().as_uri(),
        "version_id": digest,
        "media_type": media_type_for(path),
        "size": len(data),
        "sha256": digest,
    }


def build_inventory(root: str | Path, glob: str = DEFAULT_GLOB) -> list[dict[str, Any]]:
    """Inventory entries for every file under `root` matching `glob`, sorted by URI.

    Refuses an empty result. `IndexManifestV1` accepts an empty object tuple, so an empty corpus
    would build a generation holding nothing and calibration would go on to measure an empty index
    — a green run that certifies nothing. `candidate_files` refuses a glob mismatch for the same
    reason, and this extends that refusal to the case where the glob matched but the tree was bare.
    """
    files = candidate_files(root, glob)
    if not files:
        raise ValueError(
            f"no files under {str(Path(root).resolve())!r} match the glob {glob!r}, so there is "
            f"nothing to index. An empty inventory builds an empty generation, which calibrates "
            f"against nothing and reports success. Check the path and the glob."
        )
    return sorted((_entry(f) for f in files), key=lambda e: e["uri"])


def write_inventory(
    root: str | Path,
    output: str | Path,
    glob: str = DEFAULT_GLOB,
) -> int:
    """Write the inventory for `root` to `output` as JSON. Returns the entry count.

    `newline="\\n"` because a manifest's identity is the digest of its bytes: letting Python
    translate to CRLF on Windows would give the same corpus two different corpus fingerprints
    depending on which machine built it, and a fingerprint change is what
    `CALIBRATION_STALE` reports.
    """
    entries = build_inventory(root, glob)
    path = Path(output)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return len(entries)
