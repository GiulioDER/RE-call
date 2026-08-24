"""`recall manifest`: create or verify immutable corpus manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from recall.lint import DEFAULT_GLOB


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_manifest = sub.add_parser("manifest", help="create or verify immutable corpus manifests")
    p_manifest.set_defaults(func=_cmd_manifest)
    manifest_sub = p_manifest.add_subparsers(dest="manifest_cmd", required=True)
    p_manifest_create = manifest_sub.add_parser(
        "create", help="canonicalise an S3 object inventory"
    )
    p_manifest_create.add_argument("--corpus-version", required=True)
    p_manifest_create.add_argument("--objects", required=True, help="JSON array of object entries")
    p_manifest_create.add_argument("--output", required=True)
    # The producer for `create --objects`. Without it, `file://` manifests were reachable only by
    # hand-writing a JSON array with a correct sha256 per file, which is why a corpus in a
    # directory could not realistically become a generation, and therefore could not be calibrated.
    p_manifest_inventory = manifest_sub.add_parser(
        "inventory", help="build a file:// object inventory from a local directory"
    )
    p_manifest_inventory.add_argument("path")
    p_manifest_inventory.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="file glob to inventory — e.g. '**/*.py' for code. Default: markdown.",
    )
    p_manifest_inventory.add_argument("--output", required=True)
    p_manifest_verify = manifest_sub.add_parser("verify", help="verify every immutable S3 object")
    p_manifest_verify.add_argument("manifest")
    p_manifest_verify.add_argument("--version-id")
    p_manifest_verify.add_argument("--sha256")
    p_manifest_verify.add_argument("--size", type=int)


def _cmd_manifest(args: argparse.Namespace) -> None:
    from recall.lineage import IndexManifestV1, ManifestObjectV1
    from recall.manifest import (
        ExtractingS3ObjectReader,
        ObjectReader,
        S3ObjectReader,
        load_inventory,
        load_manifest,
        reader_for_manifest,
    )

    if args.manifest_cmd == "inventory":
        # Handled before anything tenant- or reader-shaped is built. An inventory describes a
        # directory and belongs to no tenant yet; `create` is where a tenant is attached.
        from recall.wizard.inventory import write_inventory

        try:
            report = write_inventory(args.path, args.output, args.glob)
        except (ValueError, OSError, NotImplementedError, MemoryError) as exc:
            # `candidate_files` and `build_inventory_report` both refuse loudly and their
            # messages name the way forward (the glob, the path). Re-raising as SystemExit
            # keeps that message and drops a traceback nobody running an install wizard can
            # act on. `NotImplementedError` is in the set because `Path.glob` raises it for a
            # non-relative pattern, and `MemoryError` because a wide glob can meet a file
            # larger than RAM; neither is a ValueError or an OSError, so both used to escape.
            # `str(MemoryError())` is the empty string, so re-raising the message alone would
            # exit 1 printing a blank line: the one member of this tuple for which "keeps that
            # message" was false. The class name is the diagnosis when there is no message.
            raise SystemExit(
                str(exc)
                or f"{type(exc).__name__} while building the inventory from {args.path!r}. "
                "Narrow --glob, or free memory."
            ) from exc
        skipped = (
            f", {report.vanished} skipped (disappeared while reading)" if report.vanished else ""
        )
        print(f"wrote {args.output} objects={report.written}{skipped}")
        return

    if args.manifest_cmd == "create":
        manifest = IndexManifestV1(
            args.tenant,
            args.corpus_version,
            load_inventory(args.objects),
        )
        Path(args.output).write_text(manifest.to_json(), encoding="utf-8")
        print(f"wrote {args.output} sha256={manifest.digest} objects={len(manifest.objects)}")
        return
    # Chosen from the manifest's own objects rather than assumed. `manifest verify` on a
    # file:// manifest previously failed with an S3 allowlist error before reading anything.
    reader: ObjectReader | None = None
    if args.manifest.startswith("s3://"):
        if args.version_id is None or args.sha256 is None or args.size is None:
            raise SystemExit("an S3 manifest requires --version-id, --sha256 and --size")
        reference = ManifestObjectV1(
            args.manifest,
            args.version_id,
            "application/json",
            args.size,
            args.sha256,
        )
        base_reader = S3ObjectReader.from_environment()
        manifest = IndexManifestV1.from_json(base_reader.fetch(reference).data)
        reader = ExtractingS3ObjectReader(base_reader)
    else:
        manifest = load_manifest(args.manifest)
    if manifest.tenant_id != args.tenant:
        raise SystemExit(
            f"manifest tenant {manifest.tenant_id!r} does not match --tenant {args.tenant!r}"
        )
    if reader is None:
        reader = reader_for_manifest(manifest)
    reader.verify(manifest)
    print(f"verified sha256={manifest.digest} objects={len(manifest.objects)}")
