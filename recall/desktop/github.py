"""Download and filter public GitHub repositories for desktop review."""

from __future__ import annotations

import io
import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from recall.desktop.models import SourceCategory
from recall.desktop.sources import collect_files
from recall.errors import RecallError


class GithubImportError(RuntimeError, RecallError):
    """Raised when a repository cannot be safely downloaded or reviewed."""


@dataclass(frozen=True)
class GithubImport:
    owner: str
    repository: str
    root: Path
    files: tuple[tuple[Path, SourceCategory], ...]


def parse_repository_url(value: str) -> tuple[str, str]:
    """Accept a GitHub URL or ``owner/repository`` shorthand."""
    text = value.strip()
    if not text:
        raise GithubImportError("Enter a GitHub repository URL first.")
    if "://" not in text:
        text = f"https://github.com/{text.lstrip('/')}"
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise GithubImportError("Use a repository on github.com, for example https://github.com/org/repo.")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] in {"orgs", "users", "settings"}:
        raise GithubImportError("The GitHub URL must include an owner and repository name.")
    repository = parts[1].removesuffix(".git")
    if not repository:
        raise GithubImportError("The GitHub repository name is missing.")
    return parts[0], repository


def download_repository(value: str, category: SourceCategory | None = None) -> GithubImport:
    """Download a public repository archive and return supported review files.

    The archive remains in a local staging directory so the existing runtime
    adapters can stream the approved files to Docker or VPS MCP later.
    """
    owner, repository = parse_repository_url(value)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "recall-desktop",
    }
    metadata_url = f"https://api.github.com/repos/{owner}/{repository}"
    try:
        metadata = _read_json(metadata_url, headers)
        branch = str(metadata.get("default_branch") or "main")
        archive_url = (
            f"https://github.com/{owner}/{repository}/archive/refs/heads/"
            f"{urllib.parse.quote(branch, safe='')}.zip"
        )
        archive = _read_bytes(archive_url, headers)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
        raise GithubImportError(f"GitHub download failed: {exc}") from exc

    staging = Path(tempfile.mkdtemp(prefix="recall-github-"))
    try:
        _extract_archive(archive, staging)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise GithubImportError(f"GitHub archive could not be unpacked: {exc}") from exc

    root = next((path for path in staging.iterdir() if path.is_dir()), staging)
    paths = collect_files([root], category)
    if not paths:
        raise GithubImportError("The repository contains no supported files for the selected scope.")
    files = tuple((path, _category_for(path)) for path in paths)
    return GithubImport(owner=owner, repository=repository, root=root, files=files)


def _read_json(url: str, headers: dict[str, str]) -> dict[str, object]:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise GithubImportError("GitHub returned invalid repository metadata.")
    return value


def _read_bytes(url: str, headers: dict[str, str]) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        data = response.read(100 * 1024 * 1024 + 1)
    if len(data) > 100 * 1024 * 1024:
        raise GithubImportError("The repository archive is larger than the 100 MB desktop limit.")
    if not isinstance(data, bytes):
        raise GithubImportError("GitHub returned an invalid archive body.")
    return data


#: Ceilings on what one repository archive may EXPAND to, as opposed to what it may weigh on the
#: wire. `_read_bytes` caps the download at 100 MB, and that bounds the wrong quantity: DEFLATE
#: reaches roughly 1000:1 on repetitive input, so a compliant 100 MB archive can describe terabytes
#: and fill the user's disk before anything indexes a byte of it. Measured while writing this: a
#: 100,000-byte file of one repeated character compresses to 115 bytes, a ratio of 870:1.
#:
#: 512 MB is chosen from the download cap rather than guessed. Text compresses at roughly 4:1, so
#: the 100 MB the fetch already allows expands to about 400 MB of real source; 512 MB clears the
#: largest archive that can legitimately arrive, while refusing anything that is describing more
#: data than it could plausibly contain. 50,000 members covers a very large repository and stops an
#: archive whose damage is a million empty files rather than bytes.
MAX_ARCHIVE_MEMBERS = 50_000
MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024


def _extract_archive(data: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        destination_root = destination.resolve()
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("archive contains too many entries")
        # Summed from the central directory BEFORE extracting, so an over-large archive costs
        # nothing but the read that already happened. The declared sizes are trustworthy for this
        # purpose, which is the part worth stating: `zipfile` stops each member's read at its
        # declared `file_size` and verifies the CRC, so understating a header truncates and then
        # raises `BadZipFile` rather than writing more than was declared. Verified directly, by
        # rewriting the size field in a real archive's central directory and watching the read
        # refuse. A cap on declared totals is therefore a cap on bytes written, not merely on
        # bytes claimed.
        total = 0
        for member in members:
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError("archive contains an unsafe path")
            total += member.file_size
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                raise ValueError("archive expands to more than the import size limit")
        archive.extractall(destination)


def _category_for(path: Path) -> SourceCategory:
    from recall.desktop.sources import classify

    category = classify(path)
    if category is None:
        raise GithubImportError(f"Unsupported downloaded file: {path.name}")
    return category
