"""Release discovery and safe staging for the Windows client."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from recall.desktop.models import ReleaseInfo


class UpdateError(RuntimeError):
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
    return ReleaseInfo(
        version=str(payload.get("tag_name", "")).lstrip("v"),
        url=str(asset["browser_download_url"]),
        sha256=digest or None,
        asset_name=str(asset["name"]),
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
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = release.asset_name or "recall-desktop-update.exe"
    destination = target_dir / filename
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(release.url, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        expected = expected_sha256 or release.sha256
        if expected and digest.lower() != expected.lower():
            raise UpdateError("download checksum does not match the release metadata")
        temporary.replace(destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
