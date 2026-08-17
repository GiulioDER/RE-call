from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

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
