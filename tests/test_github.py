from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from recall.desktop import github
from recall.desktop.github import GithubImportError, download_repository, parse_repository_url
from recall.desktop.models import SourceCategory


def test_parse_github_repository_url() -> None:
    assert parse_repository_url("https://github.com/acme/project.git") == ("acme", "project")
    assert parse_repository_url("acme/project") == ("acme", "project")


def test_parse_github_rejects_non_github_host() -> None:
    with pytest.raises(GithubImportError, match="github.com"):
        parse_repository_url("https://example.com/acme/project")


def test_download_repository_filters_code(monkeypatch: pytest.MonkeyPatch) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("project-main/app.py", "print('ok')")
        value.writestr("project-main/readme.md", "read me")
        value.writestr("project-main/image.png", b"not indexed")

    monkeypatch.setattr(
        "recall.desktop.github._read_json",
        lambda url, headers: {"default_branch": "main"},
    )
    monkeypatch.setattr("recall.desktop.github._read_bytes", lambda url, headers: archive.getvalue())

    result = download_repository("https://github.com/acme/project", SourceCategory.CODE)

    assert result.owner == "acme"
    assert result.repository == "project"
    assert [(path.name, category) for path, category in result.files] == [("app.py", SourceCategory.CODE)]
    assert Path(result.root, "app.py").read_text(encoding="utf-8") == "print('ok')"


# ---------------------------------------------------------------------------------------------
# Decompression bounds. `_read_bytes` caps the DOWNLOAD at 100 MB, which bounds the wire and not
# the disk: DEFLATE reaches ~1000:1 on repetitive input, so a compliant archive can describe far
# more than the machine has room for.
# ---------------------------------------------------------------------------------------------


def _zip_of(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buf.getvalue()


def test_an_archive_that_expands_past_the_limit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bomb is small on the wire and huge on disk, which is the whole point.

    The cap is lowered rather than a half-gigabyte archive being built, because the assertion is
    about the BOUND being enforced, not about its value. Building the real thing would make the
    test slow and would still not prove anything the lowered bound does not.
    """
    monkeypatch.setattr(github, "MAX_ARCHIVE_TOTAL_BYTES", 50_000)
    data = _zip_of({"repo/big.txt": b"A" * 200_000})

    assert len(data) < 5_000, "the fixture must be small compressed, or it is not a bomb"
    with pytest.raises(ValueError, match="import size limit"):
        github._extract_archive(data, tmp_path)


def test_an_archive_with_too_many_entries_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded separately from bytes: a million empty files costs inodes, not disk."""
    monkeypatch.setattr(github, "MAX_ARCHIVE_MEMBERS", 3)
    data = _zip_of({f"repo/f{i}.txt": b"x" for i in range(10)})

    with pytest.raises(ValueError, match="too many entries"):
        github._extract_archive(data, tmp_path)


def test_an_ordinary_archive_still_extracts(tmp_path: Path) -> None:
    """The bound must not refuse the normal case, which is what makes it a backstop."""
    github._extract_archive(_zip_of({"repo/README.md": b"# hello"}), tmp_path)
    assert (tmp_path / "repo" / "README.md").read_bytes() == b"# hello"


def test_the_byte_cap_is_enforced_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing mid-extraction is not a cap, it just makes the damage partial.

    The over-large member is second, so a check that ran per-file DURING extraction would already
    have written the first one. Nothing may land on disk.
    """
    monkeypatch.setattr(github, "MAX_ARCHIVE_TOTAL_BYTES", 50_000)
    data = _zip_of({"repo/small.txt": b"ok", "repo/big.txt": b"A" * 200_000})

    with pytest.raises(ValueError, match="import size limit"):
        github._extract_archive(data, tmp_path)
    assert list(tmp_path.iterdir()) == [], "a refused archive must leave nothing behind"


def test_the_traversal_guard_still_refuses_an_escaping_member(tmp_path: Path) -> None:
    """Kept alongside the new bounds: adding a size loop must not displace the path check."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("../escaped.txt", b"nope")

    with pytest.raises(ValueError, match="unsafe path"):
        github._extract_archive(buf.getvalue(), tmp_path)
