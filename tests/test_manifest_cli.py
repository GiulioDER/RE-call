from __future__ import annotations

import hashlib
import json
from io import BytesIO

import pytest

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


# ---------------------------------------------------------------------------------------
# `manifest inventory` — the producer `manifest create --objects` never had
# ---------------------------------------------------------------------------------------


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "one.md").write_bytes(b"# One\n\nFirst.\n")
    (root / "sub" / "two.md").write_bytes(b"# Two\n\nSecond.\n")
    (root / "code.py").write_bytes(b"def f():\n    return 1\n")
    return root


def test_manifest_inventory_feeds_create_and_verify(tmp_path, monkeypatch, capsys) -> None:
    """The end-to-end reason this command exists: a directory becomes a verified manifest.

    Everything between `inventory` and `verify` is the shipped code path, unmocked. If the URI
    form, the digest, the byte count or the `version_id` invariant were wrong, `verify` is what
    would notice, because it re-reads every file and re-hashes it.
    """
    root = _corpus(tmp_path)
    inventory = tmp_path / "inventory.json"
    manifest_path = tmp_path / "manifest.json"

    main(["manifest", "inventory", str(root), "--output", str(inventory)])
    out = capsys.readouterr().out
    assert "objects=2" in out  # two .md files; code.py is outside the default glob

    main(
        [
            "--tenant", "tenant-a",
            "manifest", "create",
            "--corpus-version", "corpus-v1",
            "--objects", str(inventory),
            "--output", str(manifest_path),
        ]
    )
    capsys.readouterr()

    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(root))
    main(["--tenant", "tenant-a", "manifest", "verify", str(manifest_path)])
    assert "verified sha256=" in capsys.readouterr().out


def test_manifest_inventory_honours_the_glob(tmp_path, capsys) -> None:
    root = _corpus(tmp_path)
    inventory = tmp_path / "code.json"
    main(["manifest", "inventory", str(root), "--glob", "**/*.py", "--output", str(inventory)])
    assert "objects=1" in capsys.readouterr().out
    entries = json.loads(inventory.read_text(encoding="utf-8"))
    assert entries[0]["uri"].endswith("/code.py")


def test_manifest_inventory_refuses_an_empty_result(tmp_path, capsys) -> None:
    """An empty inventory builds an empty generation that calibrates against nothing."""
    import pytest

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["manifest", "inventory", str(empty), "--output", str(tmp_path / "x.json")])
    assert "nothing to index" in str(exc.value)


def test_manifest_inventory_reports_files_that_vanished(tmp_path, monkeypatch, capsys) -> None:
    """A shorter inventory than the corpus must say so: the fingerprint is computed from it."""
    import recall.wizard.inventory as module

    root = _corpus(tmp_path)
    real_entry = module._entry
    seen = {"n": 0}

    def racing(path):
        seen["n"] += 1
        if seen["n"] == 1:
            (root / "sub" / "two.md").unlink()
        return real_entry(path)

    monkeypatch.setattr(module, "_entry", racing)
    main(["manifest", "inventory", str(root), "--output", str(tmp_path / "inv.json")])
    out = capsys.readouterr().out
    assert "objects=1" in out
    assert "1 skipped" in out


def test_an_exception_with_no_message_still_says_something(tmp_path, monkeypatch) -> None:
    """`str(MemoryError())` is empty, so re-raising the message alone printed a blank line.

    It is in the caught set precisely so a wide glob meeting a huge file is diagnosable, and the
    catch's stated purpose is that the message survives. For this member it did not.
    """
    import pytest

    import recall.wizard.inventory as module

    root = _corpus(tmp_path)

    def out_of_memory(path):
        raise MemoryError()

    monkeypatch.setattr(module, "_entry", out_of_memory)
    with pytest.raises(SystemExit) as exc:
        main(["manifest", "inventory", str(root), "--output", str(tmp_path / "inv.json")])
    assert "MemoryError" in str(exc.value)
    assert str(exc.value).strip(), "exiting with a blank message diagnoses nothing"


def test_manifest_inventory_needs_no_database(tmp_path, monkeypatch, capsys) -> None:
    """It touches only the filesystem, so it must not trip the insecure-DSN refusal.

    Commands marked `_opens_db` fail closed on the default local DSN. Building an inventory opens
    no connection, and requiring a configured database to describe a folder would put a wizard's
    very first step behind the thing that step exists to help configure.
    """
    monkeypatch.delenv("RECALL_DSN", raising=False)
    monkeypatch.delenv("RECALL_SERVING_DSN", raising=False)
    root = _corpus(tmp_path)
    main(["manifest", "inventory", str(root), "--output", str(tmp_path / "inv.json")])
    assert "objects=2" in capsys.readouterr().out



# ---------------------------------------------------------------------------------------
# `manifest verify` on a local manifest: the pins it is given are compared (#1436)
# ---------------------------------------------------------------------------------------


def _local_manifest(tmp_path, monkeypatch):
    """A verifiable local manifest, with the sha256 and size that honestly describe its file."""
    root = _corpus(tmp_path)
    inventory = tmp_path / "inventory.json"
    manifest_path = tmp_path / "manifest.json"
    main(["manifest", "inventory", str(root), "--output", str(inventory)])
    main(
        [
            "--tenant", "tenant-a",
            "manifest", "create",
            "--corpus-version", "corpus-v1",
            "--objects", str(inventory),
            "--output", str(manifest_path),
        ]
    )
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(root))
    raw = manifest_path.read_bytes()
    return manifest_path, hashlib.sha256(raw).hexdigest(), len(raw)


def test_local_verify_compares_the_sha256_it_is_given(tmp_path, monkeypatch, capsys) -> None:
    """Invariant: a `--sha256` that does not describe the local manifest fails the verify.

    Failure mode: the pin was read only on the `s3://` branch, so a wrong digest printed
    `verified` and the caller believed the manifest had been checked against it.

    Red proof: `tests/test_manifest_cli.py::test_local_verify_compares_the_sha256_it_is_given`
    against master 2e546bec1 (`recall/cli_commands/manifest_cmd.py::_cmd_manifest`, local branch
    `load_manifest(args.manifest)`): `Failed: DID NOT RAISE <class 'SystemExit'>`.
    """
    path, _digest, size = _local_manifest(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        main(
            ["--tenant", "tenant-a", "manifest", "verify", str(path),
             "--sha256", "b" * 64, "--size", str(size)]
        )


def test_local_verify_compares_the_size_it_is_given(tmp_path, monkeypatch, capsys) -> None:
    """Invariant: a `--size` that does not describe the local manifest fails the verify.

    Red proof: `tests/test_manifest_cli.py::test_local_verify_compares_the_size_it_is_given`
    against master 2e546bec1, same line as above: `Failed: DID NOT RAISE <class 'SystemExit'>`.
    """
    path, digest, _size = _local_manifest(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="size mismatch"):
        main(["--tenant", "tenant-a", "manifest", "verify", str(path), "--size", "999999"])
    with pytest.raises(SystemExit, match="size mismatch"):
        main(
            ["--tenant", "tenant-a", "manifest", "verify", str(path),
             "--sha256", digest, "--size", "999999"]
        )


def test_local_verify_passes_with_honest_pins(tmp_path, monkeypatch, capsys) -> None:
    """Control for the two tests above: the right pins still verify, so they refuse only a lie."""
    path, digest, size = _local_manifest(tmp_path, monkeypatch)
    capsys.readouterr()
    main(
        ["--tenant", "tenant-a", "manifest", "verify", str(path),
         "--sha256", digest, "--size", str(size)]
    )
    assert "verified sha256=" in capsys.readouterr().out


def test_local_verify_refuses_a_version_id_before_reading(tmp_path, monkeypatch) -> None:
    """Invariant: `--version-id` on a local manifest is refused, before the file is even read.

    A local file has no object version, so the pin cannot be checked; accepting it and printing
    `verified` claims a check that never happened. The second call names no file, so a refusal
    that came after reading would say "cannot read manifest" instead.

    Red proof: `tests/test_manifest_cli.py::test_local_verify_refuses_a_version_id_before_reading`
    against master 2e546bec1 (local branch `load_manifest(args.manifest)`, no refusal): the first
    call printed `verified`, `Failed: DID NOT RAISE <class 'SystemExit'>`.
    """
    path, digest, size = _local_manifest(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="--version-id"):
        main(
            ["--tenant", "tenant-a", "manifest", "verify", str(path),
             "--version-id", "v1", "--sha256", digest, "--size", str(size)]
        )
    with pytest.raises(SystemExit, match="--version-id"):
        main(
            ["--tenant", "tenant-a", "manifest", "verify", str(tmp_path / "absent.json"),
             "--version-id", "v1"]
        )


def test_the_pins_the_cli_prints_are_the_pins_it_accepts(tmp_path, monkeypatch, capsys) -> None:
    """Invariant: the pin values `manifest create` and `manifest verify` print for a local
    manifest are the ones `--sha256` and `--size` then accept.

    Failure mode: both commands printed only `sha256=<canonical digest>`, while the local pin
    compares the SHA-256 of the file's bytes (the file adds a newline, CRLF on Windows), so an
    operator who pinned the printed value was always refused with a message saying the manifest
    had changed (CCA run 20261119T212834Z, NUM-001).

    Red proof: this test against master f25e5ab1e (`recall/cli_commands/manifest_cmd.py`,
    `_cmd_manifest`, which printed no file pins) failed on the first `file_sha256=` assertion;
    recorded in the PR.
    """
    import re

    path, digest, size = _local_manifest(tmp_path, monkeypatch)
    created = re.search(r"file_sha256=([0-9a-f]{64}) size=(\d+)", capsys.readouterr().out)
    assert created, "manifest create printed no file pins"
    main(
        ["--tenant", "tenant-a", "manifest", "verify", str(path),
         "--sha256", created.group(1), "--size", created.group(2)]
    )
    verified = re.search(r"file_sha256=([0-9a-f]{64}) size=(\d+)", capsys.readouterr().out)
    assert verified and verified.groups() == created.groups() == (digest, str(size))
