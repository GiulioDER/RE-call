"""Release discovery and safe staging for the Windows client."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from recall.desktop.models import ReleaseInfo
from recall.errors import RecallError


class UpdateError(RuntimeError, RecallError):
    pass


def latest_release(api_url: str = "https://api.github.com/repos/GiulioDER/RE-call/releases/latest") -> ReleaseInfo:
    request = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json", "User-Agent": "recall-desktop"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise UpdateError(f"release check failed: {type(exc).__name__}") from exc
    assets = payload.get("assets", [])
    asset = next((item for item in assets if str(item.get("name", "")).lower().endswith(".exe")), None)
    if asset is None:
        raise UpdateError("the latest release has no Windows installer")
    digest = str(asset.get("digest") or "")
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    # .get, not subscripts: a malformed asset entry must surface as this module's own
    # UpdateError, not a raw KeyError escaping the error contract every caller catches.
    url = str(asset.get("browser_download_url") or "")
    name = str(asset.get("name") or "")
    if not url or not name:
        raise UpdateError("the release asset is missing its download URL or name")
    return ReleaseInfo(
        version=str(payload.get("tag_name", "")).lstrip("v"),
        url=url,
        sha256=digest or None,
        asset_name=name,
    )


def _version_key(value: str) -> tuple[int, ...]:
    parts = []
    for token in value.split("."):
        digits = "".join(char for char in token if char.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def is_newer(current: str, candidate: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def download_and_verify(release: ReleaseInfo, target_dir: Path, expected_sha256: str | None = None) -> Path:
    """Download the installer and verify its SHA-256, refusing when there is nothing to verify.

    A release without a digest used to skip verification silently and stage the .exe as if it
    had been checked — indistinguishable from a verified one to the caller and the log. An
    unverifiable installer is now a refusal, which is the only honest answer for a file the
    next step executes. The digest arrives in the same unauthenticated API response as the
    URL, so this protects download integrity, not release authenticity.
    """
    expected = expected_sha256 or release.sha256
    if not expected:
        raise UpdateError(
            "the release metadata carries no sha256 digest, so the installer cannot be "
            "verified; refusing to stage an unverified executable"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = release.asset_name or "recall-desktop-update.exe"
    destination = target_dir / filename
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(release.url, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest.lower() != expected.lower():
            raise UpdateError("download checksum does not match the release metadata")
        temporary.replace(destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
