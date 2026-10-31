"""SEC-001: parse the exact bytes whose checksum the CLI verified.

Red proof: on the pre-fix generation_cmd.py, the unchanged test consumes
corpus_version='replacement' and fails the final approved-version assertion.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from recall.cli import main as cli_main
from recall.generations import BuildStats, GenerationManager
from recall.lineage import IndexManifestV1, ManifestObjectV1


def test_cli_consumes_the_verified_manifest_buffer(tmp_path, monkeypatch) -> None:
    source = tmp_path / "memo.md"
    body = b"approved local source"
    source.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    entry = ManifestObjectV1(source.as_uri(), digest, "text/markdown", len(body), digest)
    approved = IndexManifestV1("manifest-bytes", "approved", (entry,))
    replacement = IndexManifestV1("manifest-bytes", "replacement", (entry,))
    path = tmp_path / "manifest.json"
    raw = approved.to_json().encode()
    path.write_bytes(raw)
    captured = {}
    original_read = Path.read_bytes
    reads = []

    def replace_after_read(current: Path) -> bytes:
        data = original_read(current)
        if current == path:
            reads.append(data)
            path.write_bytes(replacement.to_json().encode())
        return data

    def create(self, manifest, pipeline, **kwargs):
        captured["manifest"] = manifest
        return SimpleNamespace(generation_id="gen_manifest_bytes")

    monkeypatch.setenv("RECALL_ENV", "production")
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(tmp_path))
    monkeypatch.setenv("RECALL_AUTO_CALIBRATE", "off")
    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    monkeypatch.setattr(GenerationManager, "create", create)
    monkeypatch.setattr(
        GenerationManager, "build",
        lambda *args, **kwargs: BuildStats("gen_manifest_bytes", 1, 1, 0, 0, 0, 0),
    )
    cli_main([
        "--serving-dsn", "postgresql://recall:recall@127.0.0.1:1/recall",
        "--tenant", "manifest-bytes", "--embedder", "hashing",
        "generation", "build", str(path), "--manifest-sha256",
        hashlib.sha256(raw).hexdigest(), "--manifest-size", str(len(raw)),
        "--no-commit-stamp",
    ])
    assert captured["manifest"].corpus_version == "approved", "parsed unverified replacement bytes"
    assert len(reads) == 1
