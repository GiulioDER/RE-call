from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from recall_aml.compiler import prompt_digest
from scripts.aml_release_manifest import (
    BOUND_REPOSITORY_ARTIFACTS,
    build_manifest,
    sha256_file,
    verify_repository,
    write_manifest,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_verify_repository_rejects_a_dirty_tracked_checkout(tmp_path: Path) -> None:
    """RED: bypassing the status check allowed mutable source into a frozen release."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "release-test@example.invalid")
    _git(tmp_path, "config", "user.name", "Release Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "fixture")
    head = _git(tmp_path, "rev-parse", "HEAD")

    assert verify_repository(tmp_path, head) == head
    tracked.write_text("mutable\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkout changes"):
        verify_repository(tmp_path, head)


def test_verify_repository_rejects_an_untracked_file(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "release-test@example.invalid")
    _git(tmp_path, "config", "user.name", "Release Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "fixture")
    head = _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "untracked.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkout changes"):
        verify_repository(tmp_path, head)


def test_manifest_binds_artifact_bytes_and_excludes_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED: hashing a path label instead of bytes left content changes undetected."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in BOUND_REPOSITORY_ARTIFACTS.values():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"artifact:{relative.as_posix()}".encode())
    wheel = repo / "dist" / "recall_rag.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"wheel one")
    secret = "do-not-serialize-this-secret-value"
    monkeypatch.setenv("RECALL_AML_API_KEY", secret)

    first = build_manifest(
        repo_root=repo,
        wheel_path=wheel,
        commit="a" * 40,
        variant_name="A4_pack_7000",
    )
    wheel.write_bytes(b"wheel two")
    second = build_manifest(
        repo_root=repo,
        wheel_path=wheel,
        commit="a" * 40,
        variant_name="A4_pack_7000",
    )

    assert first["artifacts"]["wheel"]["sha256"] == hashlib.sha256(b"wheel one").hexdigest()
    assert first["artifacts"]["wheel"]["sha256"] != second["artifacts"]["wheel"]["sha256"]
    assert first["compiler_prompt_sha256"] == prompt_digest()
    assert secret not in json.dumps(first)
    assert first["secret_policy"]["values_included"] is False


def test_manifest_rejects_unknown_variant_and_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown AML Hosted variant"):
        build_manifest(
            repo_root=tmp_path,
            wheel_path=tmp_path / "missing.whl",
            commit="a" * 40,
            variant_name="not-registered",
        )
    with pytest.raises(FileNotFoundError, match="required release artifact"):
        build_manifest(
            repo_root=tmp_path,
            wheel_path=tmp_path / "missing.whl",
            commit="a" * 40,
            variant_name="A4_pack_7000",
        )


def test_write_manifest_refuses_overwrite(tmp_path: Path) -> None:
    """RED: replacing the existence guard silently rewrote the immutable receipt."""
    output = tmp_path / "release.json"
    write_manifest(output, {"sequence": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_manifest(output, {"sequence": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"sequence": 1}


def test_sha256_file_streams_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    payload = b"a" * (1024 * 1024 + 17)
    artifact.write_bytes(payload)
    assert sha256_file(artifact) == hashlib.sha256(payload).hexdigest()
