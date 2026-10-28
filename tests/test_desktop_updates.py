"""`download_and_verify` refuses a URL or asset name it did not choose, before any I/O.

These live outside `tests/test_desktop.py` on purpose: that module opens with
`pytest.importorskip("PySide6")`, and `recall.desktop.updates` needs no Qt, so tests placed there
would skip on every run that lacks the desktop extra.

Red proof, 2026-09-30, recorded per test below. Each refusal test was run against a
mutation of `recall.desktop.updates.download_and_verify` with the matching check deleted. The fake
transport serves a payload whose digest matches, so where the unguarded function could stage the
file it did, and the test failed with `DID NOT RAISE UpdateError`.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from recall.desktop import updates
from recall.desktop.models import ReleaseInfo
from recall.desktop.updates import UpdateError, download_and_verify

PAYLOAD = b"installer bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the transport with one that serves PAYLOAD and records every URL opened."""
    opened: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        opened.append(url)
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr(updates.urllib.request, "urlopen", fake_urlopen)
    return opened


def _release(url: str = "https://github.com/o/r/releases/download/v1/Setup.exe",
             name: str | None = "RE-call-Setup-1.0.0.exe") -> ReleaseInfo:
    return ReleaseInfo(version="1.0.0", url=url, sha256=DIGEST, asset_name=name)


def test_a_verified_https_release_is_staged_inside_the_target_directory(
    tmp_path: Path, fetched: list[str]
) -> None:
    """The control: without it, the refusals below would also pass against a function that
    refused everything."""
    staged = download_and_verify(_release(), tmp_path / "stage")

    assert staged == tmp_path / "stage" / "RE-call-Setup-1.0.0.exe"
    assert staged.read_bytes() == PAYLOAD
    assert fetched == ["https://github.com/o/r/releases/download/v1/Setup.exe"]


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "http://github.com/o/r/Setup.exe", "ftp://example.com/Setup.exe"],
)
def test_a_non_https_url_is_refused_before_it_is_opened(
    tmp_path: Path, fetched: list[str], url: str
) -> None:
    """Red proof: mutation deleting the `https://` check; all three cases failed with DID NOT
    RAISE, the fake transport having opened each URL."""
    with pytest.raises(UpdateError, match="non-https"):
        download_and_verify(_release(url=url), tmp_path / "stage")

    assert fetched == []
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize(
    "name",
    ["../escaped.exe", "sub/Setup.exe", "sub\\Setup.exe", "C:Setup.exe", "..", ".hidden.exe"],
)
def test_an_asset_name_that_is_not_a_plain_file_name_is_refused(
    tmp_path: Path, fetched: list[str], name: str
) -> None:
    """Red proof: mutation deleting the `_SAFE_ASSET_NAME` check; all six cases failed.

    "../escaped.exe", "sub\\Setup.exe", "C:Setup.exe" and ".hidden.exe" failed with DID NOT RAISE:
    the unguarded function staged them, the first one outside `target_dir`. "sub/Setup.exe"
    (FileNotFoundError) and ".." (OSError, EBUSY on the rename) failed because a raw OS error
    escaped instead of `UpdateError`, the one exception this module's callers catch. On Windows
    "C:Setup.exe" is the escape case too, since joining it discards `target_dir`."""
    with pytest.raises(UpdateError, match="not a plain file name"):
        download_and_verify(_release(name=name), tmp_path / "stage")

    assert fetched == []
    assert list(tmp_path.iterdir()) == []


def _release_api(monkeypatch: pytest.MonkeyPatch, assets: list[dict[str, str]]) -> None:
    payload = {"tag_name": "v0.15.0", "assets": assets}

    def fake_urlopen(request: object, timeout: float) -> io.BytesIO:
        del request, timeout
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(updates.urllib.request, "urlopen", fake_urlopen)


def test_the_update_check_finds_the_installer_the_release_pipeline_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: `latest_release` selects the asset `windows-installer.yml` uploads.

    That workflow attaches `recall-install-<tag>-windows-x64.zip` and nothing else for Windows;
    the check looked for a `.exe`, so it could not succeed against any release. A stray `.exe`
    asset of another name is not the installer. Red proof (selection by `.exe` suffix): fails
    with UpdateError "the latest release has no Windows installer" on the published zip.
    """
    zip_name = "recall-install-v0.15.0-windows-x64.zip"
    _release_api(monkeypatch, [
        {"name": "recall_rag-0.15.0-py3-none-any.whl", "browser_download_url": "https://x/a.whl"},
        {"name": zip_name, "browser_download_url": "https://x/" + zip_name, "digest": "sha256:" + "a" * 64},
    ])

    release = updates.latest_release()

    assert (release.version, release.asset_name, release.sha256) == ("0.15.0", zip_name, "a" * 64)

    _release_api(monkeypatch, [{"name": "something-else.exe", "browser_download_url": "https://x/s.exe"}])
    with pytest.raises(UpdateError, match="no Windows installer"):
        updates.latest_release()
