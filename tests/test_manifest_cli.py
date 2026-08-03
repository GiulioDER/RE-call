from __future__ import annotations

import hashlib
import json
from io import BytesIO

from recall.cli import main
from recall.lineage import IndexManifestV1
from recall.manifest import S3Allowlist, S3ObjectReader


class _S3:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def get_object(self, **kwargs):
        return {
            "Body": BytesIO(self.data),
            "ContentLength": len(self.data),
            "VersionId": kwargs["VersionId"],
        }


def test_manifest_create_and_verify_cli(tmp_path, monkeypatch, capsys) -> None:
    data = b"immutable memo"
    inventory = tmp_path / "objects.json"
    inventory.write_text(
        json.dumps(
            [
                {
                    "uri": "s3://approved/corpora/tenant-a/memo.md",
                    "version_id": "object-v1",
                    "media_type": "text/markdown",
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"

    main(
        [
            "--tenant",
            "tenant-a",
            "manifest",
            "create",
            "--corpus-version",
            "corpus-v1",
            "--objects",
            str(inventory),
            "--output",
            str(manifest_path),
        ]
    )
    created = IndexManifestV1.from_json(manifest_path.read_bytes())
    assert created.tenant_id == "tenant-a"
    assert "sha256=" in capsys.readouterr().out

    reader = S3ObjectReader(_S3(data), S3Allowlist.parse("approved/corpora/"))
    monkeypatch.setattr(
        S3ObjectReader,
        "from_environment",
        classmethod(lambda cls: reader),
    )
    main(["--tenant", "tenant-a", "manifest", "verify", str(manifest_path)])
    assert "verified sha256=" in capsys.readouterr().out

