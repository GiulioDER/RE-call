"""A local manifest is admissible in production when it is allowlisted and verified.

⛔ **What this file is worth is decided by the mutation run, not by these tests passing.** The
failure mode it guards is not "production accepts a local manifest" but the quieter one it was
written for: `--manifest-sha256` and `--manifest-size` were accepted on the local path and used for
nothing, so a caller supplying them got no verification and no warning either.
"""

from __future__ import annotations

import hashlib
import re

import psycopg
import pytest

from recall.cli import main as cli_main
from recall.cli_commands.generation_cmd import _verify_local_manifest
from recall.lineage import IndexManifestV1, ManifestObjectV1
from tests.conftest import TEST_DSN, requires_db


@pytest.fixture
def manifest(tmp_path):
    """A file plus the digest and size that honestly describe it."""
    path = tmp_path / "memory-manifest.json"
    body = b'{"schema_version": 1, "tenant_id": "memory", "objects": []}'
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest(), len(body)


# ------------------------------------------------------------------ verification, every environment


def test_a_supplied_checksum_is_actually_checked(manifest) -> None:
    """The defect this file exists for: a digest accepted and then ignored.

    Silently discarding a checksum is worse than never accepting one, because the caller believes
    the check happened. `bin/build_generation_voyage.sh` supplies both on every run.
    """
    path, _digest, size = manifest
    with pytest.raises(SystemExit, match="checksum mismatch"):
        _verify_local_manifest(str(path), sha256="b" * 64, size=size, environment="development")


def test_a_supplied_size_is_actually_checked(manifest) -> None:
    path, digest, _size = manifest
    with pytest.raises(SystemExit, match="size mismatch"):
        _verify_local_manifest(str(path), sha256=digest, size=999999, environment="development")


def test_an_honest_pair_passes(manifest) -> None:
    path, digest, size = manifest
    _verify_local_manifest(str(path), sha256=digest, size=size, environment="development")


def test_development_still_needs_no_digest_at_all(manifest) -> None:
    """Verification is opt-in outside production; requiring it would break every local workflow."""
    path, _digest, _size = manifest
    _verify_local_manifest(str(path), sha256=None, size=None, environment="development")


def test_a_missing_manifest_is_reported_as_a_manifest_problem(tmp_path) -> None:
    with pytest.raises(SystemExit, match="cannot read manifest"):
        _verify_local_manifest(
            str(tmp_path / "absent.json"), sha256="a" * 64, size=1, environment="development"
        )


# ------------------------------------------------------------------------------- production policy


def test_production_refuses_a_local_manifest_with_no_allowlist(manifest, monkeypatch) -> None:
    """Without the allowlist a manifest can name any file on the machine, so this must refuse."""
    path, digest, size = manifest
    monkeypatch.delenv("RECALL_LOCAL_ALLOWLIST", raising=False)
    with pytest.raises(SystemExit, match="RECALL_LOCAL_ALLOWLIST"):
        _verify_local_manifest(str(path), sha256=digest, size=size, environment="production")


def test_production_refuses_an_allowlist_that_is_only_whitespace(manifest, monkeypatch) -> None:
    """A set-but-empty variable is the shape that slips past a bare truthiness check."""
    path, digest, size = manifest
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", "   ")
    with pytest.raises(SystemExit, match="RECALL_LOCAL_ALLOWLIST"):
        _verify_local_manifest(str(path), sha256=digest, size=size, environment="production")


def test_production_refuses_an_unverified_local_manifest(manifest, monkeypatch, tmp_path) -> None:
    """Allowlisted is not enough: in production the manifest must be verified, not merely parsed."""
    path, digest, size = manifest
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(tmp_path))
    with pytest.raises(SystemExit, match="requires --manifest-sha256 and --manifest-size"):
        _verify_local_manifest(str(path), sha256=None, size=None, environment="production")
    with pytest.raises(SystemExit, match="requires --manifest-sha256 and --manifest-size"):
        _verify_local_manifest(str(path), sha256=digest, size=None, environment="production")
    with pytest.raises(SystemExit, match="requires --manifest-sha256 and --manifest-size"):
        _verify_local_manifest(str(path), sha256=None, size=size, environment="production")


def test_production_accepts_an_allowlisted_verified_local_manifest(
    manifest, monkeypatch, tmp_path
) -> None:
    """🔑 The behaviour the whole change exists for.

    Before this, production refused every local manifest outright, so the only route was
    `RECALL_ENV=development` — which also redirects reads to the legacy table.
    """
    path, digest, size = manifest
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(tmp_path))
    _verify_local_manifest(str(path), sha256=digest, size=size, environment="production")


def test_production_still_verifies_after_the_policy_checks_pass(
    manifest, monkeypatch, tmp_path
) -> None:
    """The allowlist and the digest requirement must not become an early return past the check."""
    path, _digest, size = manifest
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(tmp_path))
    with pytest.raises(SystemExit, match="checksum mismatch"):
        _verify_local_manifest(str(path), sha256="c" * 64, size=size, environment="production")


# ------------------------------------------------------------- the call site, which nothing above
#
# ⛔ **Added because a mutation proved everything above is inert-safe.** Reverting the call site in
# `_cmd_generation` to the original `raise SystemExit("production generation builds require a
# versioned S3 manifest")` left all ten tests above GREEN, because every one of them calls
# `_verify_local_manifest` directly and none goes through the CLI. The helper would have been
# correct, tested, and never reached — the same defect class that appeared twice already in the
# sibling hosted-embedder change.

def _local_manifest(tenant: str, tmp_path):
    """A one-object manifest naming a real file inside `tmp_path`, with honest digests."""
    body = b"a local corpus source"
    source = tmp_path / "memo.md"
    source.write_bytes(body)
    # ⚠️ `version_id` must BE the content digest for a file:// object. `lineage.py` refuses any
    # other value, on the grounds that a local file has no version other than its contents and
    # anything else would name an immutability guarantee the filesystem does not provide. The
    # first version of this fixture passed "v1" and was refused, correctly.
    body_digest = hashlib.sha256(body).hexdigest()
    manifest = IndexManifestV1(
        tenant,
        "corpus-local-v1",
        (
            ManifestObjectV1(
                source.resolve().as_uri(),
                body_digest,
                "text/markdown",
                len(body),
                body_digest,
            ),
        ),
    )
    path = tmp_path / "manifest.json"
    raw = manifest.to_json().encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), len(raw)


@requires_db
def test_the_cli_builds_a_local_manifest_in_production(tmp_path, monkeypatch, capsys) -> None:
    """🔑 End to end through `recall generation build`, with production genuinely set.

    This is the command `bin/build_generation_voyage.sh` runs. Before this change it died with
    "production generation builds require a versioned S3 manifest" and the only way through was
    `RECALL_ENV=development`, which also redirects reads to the legacy table.
    """
    tenant = "localgate-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:10]
    path, digest, size = _local_manifest(tenant, tmp_path)
    monkeypatch.setenv("RECALL_ENV", "production")
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(tmp_path))
    argv = [
        "--serving-dsn", TEST_DSN, "--tenant", tenant, "--embedder", "hashing",
        "generation", "build", str(path),
        "--manifest-sha256", digest, "--manifest-size", str(size),
        # ⚠️ No `--unverified-development`: `HashingEmbedder` carries a real revision, so its
        # identity is VERIFIED and production has nothing to waive. Passing the flag made this test
        # fail on the sibling hosted-embedder gate instead of exercising the manifest gate, which
        # would have measured the wrong thing while looking like a failure of this change.
        "--no-commit-stamp",
    ]
    try:
        cli_main(argv)
        out = capsys.readouterr().out
        assert re.search(r"built gen_[0-9a-f]+:", out), out
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            for table in ("recall_chunks_v1", "recall_generations"):
                conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))


@requires_db
def test_the_cli_refuses_a_local_manifest_in_production_without_an_allowlist(
    tmp_path, monkeypatch
) -> None:
    """The exemption is a policy, not a hole: the CLI must still refuse the unguarded case."""
    tenant = "localgate-deny-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]
    path, digest, size = _local_manifest(tenant, tmp_path)
    monkeypatch.setenv("RECALL_ENV", "production")
    monkeypatch.delenv("RECALL_LOCAL_ALLOWLIST", raising=False)
    with pytest.raises(SystemExit, match="RECALL_LOCAL_ALLOWLIST"):
        cli_main([
            "--serving-dsn", TEST_DSN, "--tenant", tenant, "--embedder", "hashing",
            "generation", "build", str(path),
            "--manifest-sha256", digest, "--manifest-size", str(size),
        ])
