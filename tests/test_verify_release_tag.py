"""The release workflow must not publish a tag whose version differs from package metadata."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import verify_release_tag  # noqa: E402


def _write_project(root: pathlib.Path, version: str) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "recall-rag"\nversion = "{version}"\n',
        encoding="utf-8",
    )


def test_matching_tag_is_accepted(tmp_path: pathlib.Path) -> None:
    _write_project(tmp_path, "0.14.0")

    assert verify_release_tag.verify("v0.14.0", tmp_path) == "0.14.0"


def test_mismatched_tag_is_refused(tmp_path: pathlib.Path) -> None:
    _write_project(tmp_path, "0.14.0")

    with pytest.raises(verify_release_tag.Refusal, match="pyproject.toml declares 0.14.0"):
        verify_release_tag.verify("v0.14.1", tmp_path)


@pytest.mark.parametrize("tag", ["0.14.0", "v0.14", "release-0.14.0"])
def test_non_release_tag_is_refused(tmp_path: pathlib.Path, tag: str) -> None:
    _write_project(tmp_path, "0.14.0")

    with pytest.raises(verify_release_tag.Refusal, match="vX.Y.Z"):
        verify_release_tag.verify(tag, tmp_path)
