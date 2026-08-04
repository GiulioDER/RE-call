from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

from recall.calibration import Calibration
from recall.embeddings import Embedder, HashingEmbedder
from recall.index import Indexer, PruneGuardTripped, chunk_code, chunk_text
from recall.lint import DEFAULT_GLOB
from recall.observability import configure_logging
from recall.store import DEFAULT_TENANT, PgVectorStore, warn_if_insecure_dsn
from recall.trust import terminal_safe, trusted_search
from recall.types import TrustedResult

DEFAULT_DSN = os.environ.get(
    "RECALL_SERVING_DSN",
    os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"),
)
DEFAULT_MIGRATION_DSN = os.environ.get("RECALL_MIGRATION_DSN")


def _make_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise SystemExit(f"unknown embedder: {name}")


def _print_result(result: TrustedResult) -> None:
    flags = []
    if result.abstained:
        flags.append("ABSTAIN")
    if result.gap_warning:
        flags.append("GAP")
    if result.staleness.stale:
        flags.append("STALE")
    print(f"[{' '.join(flags) if flags else 'ok'}] query={result.query!r}")
    if result.reason:
        print(f"  reason: {result.reason}")
    for h in result.hits:
        # All three are corpus-controlled and all three are printed to a terminal, which
        # INTERPRETS ANSI escapes rather than showing them — a file name carrying `\x1b[2K\r`
        # erases the line it was printed on. Same class as the `advice` injection, different
        # interpreter. `terminal_safe` filters, so ordinary names render exactly as authored.
        preview = terminal_safe(h.chunk.text).replace("\n", " ")[:52]
        name = terminal_safe(h.provenance.file or h.chunk.source)
        redirect = (
            f" -> use {terminal_safe(h.validity.superseded_by)}" if h.validity.superseded_by else ""
        )
        print(
            f"  {h.verdict:<14} conf={h.confidence:.2f} cos={h.cosine:.3f}  "
            f"{name}{redirect}  {preview!r}"
        )


def _run_queries(
    store: PgVectorStore,
    embedder: Embedder,
    queries: list[str],
    calibration: Calibration | None,
) -> None:
    for q in queries:
        _print_result(trusted_search(store, embedder, q, calibration=calibration))
        print()


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # clean UTF-8 output on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    # Without this the library's loggers have no handler, so every _log.info is discarded — which
    # is how `index` came to prune rows while printing nothing about it.
    configure_logging()
    parser = argparse.ArgumentParser(prog="recall")
    parser.add_argument(
        "--serving-dsn",
        "--dsn",
        dest="dsn",
        default=DEFAULT_DSN,
        help="unprivileged application DSN (env: RECALL_SERVING_DSN; --dsn is deprecated)",
    )
    parser.add_argument(
        "--migration-dsn",
        default=DEFAULT_MIGRATION_DSN,
        help="DDL-owner DSN used only by `schema apply` (env: RECALL_MIGRATION_DSN)",
    )
    parser.add_argument("--embedder", default="fastembed", choices=["fastembed", "hashing"])
    parser.add_argument(
        "--table",
        default="chunks",
        help="table to read/write (default: chunks). Use a throwaway name to keep an "
        "experiment out of your real memory index.",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help=f"tenant namespace to operate on (default: {DEFAULT_TENANT}). Every command is "
        f"scoped to one tenant; `forget` in particular deletes nothing outside it, so an "
        f"erasure request against another tenant needs this flag.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_schema = sub.add_parser("schema", help="inspect or apply versioned database migrations")
    p_schema.add_argument(
        "--dim",
        type=int,
        default=None,
        help="embedding dimension (default: infer from --embedder)",
    )
    schema_sub = p_schema.add_subparsers(dest="schema_cmd", required=True)
    schema_sub.add_parser("status", help="show installed and required schema versions")
    schema_sub.add_parser("plan", help="show pending migrations without changing the database")
    schema_sub.add_parser("apply", help="apply pending migrations with the migration role")
    p_schema_grants = schema_sub.add_parser(
        "grants",
        help="print the GRANT statements a serving role needs (prints SQL, runs none)",
    )
    p_schema_grants.add_argument("--role", required=True, help="the serving role name")
    p_schema_grants.add_argument(
        "--enterprise",
        action="store_true",
        help="also grant the enterprise control-plane tables and their sequence",
    )

    p_manifest = sub.add_parser("manifest", help="create or verify immutable corpus manifests")
    manifest_sub = p_manifest.add_subparsers(dest="manifest_cmd", required=True)
    p_manifest_create = manifest_sub.add_parser(
        "create", help="canonicalise an S3 object inventory"
    )
    p_manifest_create.add_argument("--corpus-version", required=True)
    p_manifest_create.add_argument("--objects", required=True, help="JSON array of object entries")
    p_manifest_create.add_argument("--output", required=True)
    p_manifest_verify = manifest_sub.add_parser("verify", help="verify every immutable S3 object")
    p_manifest_verify.add_argument("manifest")
    p_manifest_verify.add_argument("--version-id")
    p_manifest_verify.add_argument("--sha256")
    p_manifest_verify.add_argument("--size", type=int)

    p_generation = sub.add_parser("generation", help="manage immutable blue green generations")
    generation_sub = p_generation.add_subparsers(dest="generation_cmd", required=True)
    p_build = generation_sub.add_parser("build", help="create and build a generation")
    p_build.add_argument("manifest")
    p_build.add_argument("--manifest-version-id")
    p_build.add_argument("--manifest-sha256")
    p_build.add_argument("--manifest-size", type=int)
    p_build.add_argument("--embedder-provider", default=None)
    p_build.add_argument("--embedder-revision", default=None)
    p_build.add_argument("--embedder-artifact-digest", default=None)
    p_build.add_argument("--unverified-development", action="store_true")
    p_build.add_argument("--chunker", choices=["text", "code"], default="text")
    p_build.add_argument("--max-chars", type=int, default=800)
    p_build.add_argument("--overlap", type=int, default=80)
    p_validate = generation_sub.add_parser("validate", help="validate a built generation")
    p_validate.add_argument("generation_id")
    p_promote = generation_sub.add_parser("promote", help="promote a ready generation")
    p_promote.add_argument("generation_id")
    p_promote.add_argument("--unsafe-development-promotion", action="store_true")
    generation_sub.add_parser("rollback", help="atomically restore the previous generation")
    generation_sub.add_parser("list", help="list immutable generation history")
    p_gc = generation_sub.add_parser("gc", help="collect expired retired generations")
    p_gc.add_argument("--retention-days", type=int, default=7)
    p_gc.add_argument("--retain-previous", type=int, default=2)

    p_index = sub.add_parser("index", help="index a folder of markdown or code")
    p_index.add_argument("path")
    p_index.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="file glob to index — e.g. '**/*.py' for code (auto-uses code chunking). Default: markdown.",
    )
    p_index.add_argument(
        "--allow-prune",
        action="store_true",
        help="permit this run to drop most of the indexed corpus. Re-indexing removes files that "
        "are gone from disk; when most of them vanish at once that is refused, because it "
        "usually means the corpus is missing rather than deleted. Pass this once you have "
        "confirmed the files really are gone.",
    )

    p_forget = sub.add_parser(
        "forget",
        help="permanently delete indexed memory for the given source(s) — irreversible",
    )
    p_forget.add_argument(
        "sources",
        nargs="+",
        help="source value(s) to forget, exactly as stored (see the `source` field in "
        "`recall search` output)",
    )
    p_forget.add_argument(
        "--yes",
        action="store_true",
        help="actually delete. Without this flag, forget only PREVIEWS what would be removed "
        "and changes nothing — this command is the right-to-erasure path, it is "
        "irreversible, and it is also invoked from scripts, so a typo or an unattended "
        "run must not silently wipe a corpus. Re-run with --yes once the preview looks right.",
    )

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument(
        "--entail",
        action="store_true",
        help="opt-in entailment stage: demote hits that don't answer the query "
        "(requires recall[entail]; downloads the QNLI judge on first use)",
    )

    sub.add_parser("demo", help="index corpus/ and run sample memory queries")
    sub.add_parser("code", help="index recall's own source and run sample code queries")

    p_lint = sub.add_parser(
        "lint",
        help="check a corpus's supersession graph for broken/missing edges (no DB needed)",
    )
    p_lint.add_argument("path")
    p_lint.add_argument("--glob", default=DEFAULT_GLOB)
    p_lint.add_argument(
        "--semantic",
        action="store_true",
        help="also run the retrieval-based MISSING-edge check: flag memos highly similar to a "
        "prior closed decision they don't reference (needs the DB + embedder; opt-in)",
    )
    p_lint.add_argument(
        "--fix",
        action="store_true",
        help="propose the frontmatter `supersedes:` edge for each closure marker whose target "
        "is provable. DRY RUN by default — prints the plan and changes nothing.",
    )
    p_lint.add_argument(
        "--apply",
        action="store_true",
        help="with --fix, actually write the proposed edges to the memo files",
    )
    p_lint.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="cosine threshold for --semantic (default: the calibrated abstention threshold "
        "for this embedder; must be calibrated per embedder — see FINDINGS section 2)",
    )

    p_check = sub.add_parser(
        "check",
        help="write-time gate: for the memo(s) you are committing, ask for the supersession "
        "edge while you still know the answer (no DB needed)",
    )
    p_check.add_argument("paths", nargs="+", help="the memo file(s) being written")
    p_check.add_argument(
        "--corpus",
        default=None,
        help="corpus dir used to filter candidates to real documents (default: each file's own "
        "directory)",
    )
    p_check.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when a memo needs an edge — use this in a pre-commit hook",
    )

    p_cal = sub.add_parser(
        "calibrate",
        help="measure a calibration bound to one immutable generation",
    )
    p_cal.add_argument("--generation", required=True)
    p_cal.add_argument("--queries", dest="query_file", required=True)
    p_cal.add_argument("--publish", action="store_true")

    p_calibration = sub.add_parser("calibration", help="inspect or transfer calibration artifacts")
    calibration_sub = p_calibration.add_subparsers(dest="calibration_cmd", required=True)
    calibration_sub.add_parser("list")
    p_cal_show = calibration_sub.add_parser("show")
    p_cal_show.add_argument("calibration_id")
    p_cal_export = calibration_sub.add_parser("export")
    p_cal_export.add_argument("calibration_id")
    p_cal_export.add_argument("--output", required=True)
    p_cal_import = calibration_sub.add_parser("import")
    p_cal_import.add_argument("path")

    args = parser.parse_args(argv)
    warn_if_insecure_dsn(args.dsn)  # loud stderr note if default creds target a remote host

    if args.cmd == "schema":
        from recall.schema import apply_migrations, schema_plan, schema_status

        if args.schema_cmd == "grants":
            # Prints SQL for an operator to run as the object owner; touches no database, so
            # it needs neither a DSN nor an embedder.
            from recall.schema import serving_grants

            for statement in serving_grants(
                args.role, table=args.table, enterprise=args.enterprise
            ):
                print(statement)
            return
        dim = args.dim if args.dim is not None else _make_embedder(args.embedder).dim
        inspect_dsn = args.migration_dsn or args.dsn
        if args.schema_cmd == "status":
            status = schema_status(inspect_dsn, table=args.table, dim=dim)
            print(f"table: {status.table}")
            print(f"current: {status.current_version or 'none'}")
            print(f"required: {status.required_version}")
            print(f"compatible: {'yes' if status.compatible else 'no'}")
            for migration in status.migrations:
                print(f"{migration.version} {migration.state:<7} {migration.filename}")
            if not status.compatible:
                raise SystemExit(1)
            return
        if args.schema_cmd == "plan":
            pending = schema_plan(inspect_dsn, table=args.table, dim=dim)
            if not pending:
                print("schema is current; no changes planned")
            else:
                for migration in pending:
                    print(f"would apply {migration.version} {migration.filename}")
            return
        if not args.migration_dsn:
            raise SystemExit(
                "schema apply requires --migration-dsn or RECALL_MIGRATION_DSN; "
                "the serving DSN is never used for DDL"
            )
        applied = apply_migrations(args.migration_dsn, table=args.table, dim=dim)
        if not applied:
            print("schema is current; nothing applied")
        else:
            for migration in applied:
                print(f"applied {migration.version} {migration.filename}")
        return

    if args.cmd == "manifest":
        from recall.lineage import IndexManifestV1, ManifestObjectV1
        from recall.manifest import S3ObjectReader, load_inventory, load_manifest

        if args.manifest_cmd == "create":
            manifest = IndexManifestV1(
                args.tenant,
                args.corpus_version,
                load_inventory(args.objects),
            )
            Path(args.output).write_text(manifest.to_json(), encoding="utf-8")
            print(f"wrote {args.output} sha256={manifest.digest} objects={len(manifest.objects)}")
            return
        reader = S3ObjectReader.from_environment()
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
            manifest = IndexManifestV1.from_json(reader.fetch(reference).data)
        else:
            manifest = load_manifest(args.manifest)
        if manifest.tenant_id != args.tenant:
            raise SystemExit(
                f"manifest tenant {manifest.tenant_id!r} does not match --tenant {args.tenant!r}"
            )
        reader.verify(manifest)
        print(f"verified sha256={manifest.digest} objects={len(manifest.objects)}")
        return

    if args.cmd == "generation":
        from recall.generations import GenerationManager

        manager = GenerationManager(args.dsn, args.tenant)
        if args.generation_cmd == "list":
            for generation in manager.list_generations():
                print(
                    f"{generation.generation_id} {generation.state.value:<18} "
                    f"pipeline={generation.pipeline_fingerprint} "
                    f"corpus={generation.corpus_fingerprint}"
                )
            return
        if args.generation_cmd == "rollback":
            print(f"active generation: {manager.rollback()}")
            return
        if args.generation_cmd == "gc":
            collected = manager.gc(
                retention_days=args.retention_days,
                retain_previous=args.retain_previous,
            )
            print(f"collected {len(collected)} generation(s): {', '.join(collected) or '(none)'}")
            return
        if args.generation_cmd == "validate":
            generation_validation = manager.validate(args.generation_id)
            print(
                f"ready {generation_validation.generation_id}: "
                f"{generation_validation.sources} sources, "
                f"{generation_validation.chunks} chunks"
            )
            return
        if args.generation_cmd == "promote":
            manager.promote(
                args.generation_id,
                unsafe_development=args.unsafe_development_promotion,
            )
            print(f"active generation: {args.generation_id}")
            return

        from recall.lineage import (
            ChunkerIdentity,
            EmbedderIdentity,
            IndexManifestV1,
            ManifestObjectV1,
            PipelineIdentity,
        )
        from recall.manifest import S3ObjectReader, load_manifest

        environment = manager.environment
        reader = S3ObjectReader.from_environment()
        if args.manifest.startswith("s3://"):
            if (
                args.manifest_version_id is None
                or args.manifest_sha256 is None
                or args.manifest_size is None
            ):
                raise SystemExit(
                    "an S3 manifest requires --manifest-version-id, --manifest-sha256 and "
                    "--manifest-size"
                )
            reference = ManifestObjectV1(
                args.manifest,
                args.manifest_version_id,
                "application/json",
                args.manifest_size,
                args.manifest_sha256,
            )
            manifest = IndexManifestV1.from_json(reader.fetch(reference).data)
        else:
            if environment == "production":
                raise SystemExit("production generation builds require a versioned S3 manifest")
            manifest = load_manifest(args.manifest)
        embedder = _make_embedder(args.embedder)
        revision = args.embedder_revision
        provider = args.embedder_provider
        if isinstance(embedder, HashingEmbedder):
            provider = provider or "recall"
            revision = revision or "hashing-md5-bow-v1"
        else:
            provider = provider or "fastembed"
        identity = EmbedderIdentity(
            provider=provider,
            model=embedder.name,
            dimension=embedder.dim,
            revision=revision,
            artifact_digest=args.embedder_artifact_digest,
            unverified_reason=(
                "explicit development build"
                if args.unverified_development
                and not revision
                and not args.embedder_artifact_digest
                else None
            ),
        )
        if args.chunker == "code":
            generation_chunker = functools.partial(chunk_code, max_chars=args.max_chars)
            chunker_identity = ChunkerIdentity(
                "recall.chunk_code", 1, {"max_chars": args.max_chars}
            )
        else:
            generation_chunker = functools.partial(
                chunk_text, max_chars=args.max_chars, overlap=args.overlap
            )
            chunker_identity = ChunkerIdentity(
                "recall.chunk_text",
                1,
                {"max_chars": args.max_chars, "overlap": args.overlap},
            )
        pipeline = PipelineIdentity(identity, chunker_identity)
        generation = manager.create(
            manifest,
            pipeline,
            allow_unverified=args.unverified_development,
        )
        generation_stats = manager.build(
            generation.generation_id,
            reader,
            embedder,
            generation_chunker,
        )
        print(
            f"built {generation_stats.generation_id}: {generation_stats.objects} objects, "
            f"{generation_stats.chunks} chunks, {generation_stats.reused_objects} objects "
            f"reused; run `recall generation validate {generation_stats.generation_id}`"
        )
        return

    if args.cmd == "lint":  # pure filesystem check — no embedder, no DB
        from recall.lint import lint_corpus

        try:
            issues = lint_corpus(args.path, glob=args.glob)
        except FileNotFoundError as exc:
            print(f"recall lint: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        for i in issues:
            print(f"{i.level:<8} {i.code:<26} {i.file}: {i.message}")
        errors = sum(1 for i in issues if i.level == "error")
        warnings = len(issues) - errors

        if args.fix:
            from recall.fix import apply_proposal, propose_fixes

            proposals, unfixable = propose_fixes(args.path, glob=args.glob)
            print()
            for p in proposals:
                print(f"  {p.edit_file}: + supersedes: {p.target}")
                print(f"      because {p.evidence_file} says {p.evidence!r}")
            for u in unfixable:
                print(f"  SKIP {u.file}: {u.reason}")
            print(f"\n{len(proposals)} edge(s) proposable, {len(unfixable)} need a human")
            if not args.apply:
                # Dry run by DEFAULT: this edits the user's own documents, and a tool that
                # rewrites your memory the first time you try it has earned distrust.
                print("dry run — nothing written. Re-run with --apply to write these edges.")
            else:
                root = Path(args.path)
                for p in proposals:
                    apply_proposal(root, p)
                print(f"wrote {len(proposals)} edge(s).")

        chains = []
        if args.semantic:  # opt-in retrieval-based missing-edge check (needs DB + embedder)
            from recall.semantic_lint import semantic_lint

            emb = _make_embedder(args.embedder)
            thr = args.threshold if args.threshold is not None else 0.70
            chains = semantic_lint(args.dsn, emb, args.path, threshold=thr, glob=args.glob)
            for c in chains:
                print(
                    f"warning  unlinked-chain             {c.new_memo}: highly similar "
                    f"(cos={c.cosine:.2f}) to closed decision {c.prior!r} it does not "
                    f"reference — add `supersedes: {c.prior}`?"
                )
            warnings += len(chains)

        print(f"{errors} errors, {warnings} warnings")
        if errors:
            raise SystemExit(1)
        return

    if args.cmd == "check":  # pure filesystem check — no embedder, no DB
        from recall.check import check_file, corpus_names, format_prompt

        needs = 0
        for raw in args.paths:
            f = Path(raw)
            if not f.exists():
                print(f"recall check: no such file: {raw}", file=sys.stderr)
                raise SystemExit(2)
            names = corpus_names(args.corpus or f.parent)
            result = check_file(f, names)
            if result.needs_attention:
                needs += 1
                print(format_prompt(result))
        if needs:
            print(f"\n{needs} memo(s) state a closure in prose only.")
            if args.strict:
                raise SystemExit(1)
        return

    if args.cmd == "calibration":
        import json

        from recall.calibration_v2 import CalibrationError, CalibrationRepository

        repository = CalibrationRepository(args.dsn, args.tenant)
        try:
            if args.calibration_cmd == "list":
                print(json.dumps(repository.list_records(), indent=2, default=str))
            elif args.calibration_cmd == "show":
                print(
                    json.dumps(repository.show_record(args.calibration_id), indent=2, default=str)
                )
            elif args.calibration_cmd == "export":
                print(repository.export_bundle(args.calibration_id, args.output))
            else:
                print(repository.import_bundle(args.path))
        except CalibrationError as exc:
            raise SystemExit(str(exc)) from exc
        return

    embedder = _make_embedder(args.embedder)
    if args.cmd == "calibrate":
        from recall.calibration_v2 import CalibrationError, CalibrationRepository, load_query_set

        repository = CalibrationRepository(args.dsn, args.tenant)
        try:
            labels, _digest = load_query_set(args.query_file)
            artifact = repository.calibrate(args.generation, labels, embedder)
            if args.publish:
                artifact = repository.publish(artifact.calibration_id)
        except CalibrationError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"calibration: {artifact.calibration_id}")
        print(f"status: {artifact.status.value}")
        print(f"generation: {artifact.generation_id}")
        print(f"pipeline: {artifact.pipeline_fingerprint}")
        print(f"corpus: {artifact.corpus_fingerprint}")
        print(f"queries: {artifact.query_set_digest}")
        return
    calibration = None

    if args.cmd == "index":
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            raise SystemExit(
                "local filesystem indexing is development-only; build from an immutable S3 "
                "manifest in production"
            )
        chunker = chunk_code if args.glob.endswith(".py") else chunk_text
        with PgVectorStore(
            args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
        ) as store:
            store.check_schema()
            indexer = Indexer(store, embedder, chunker=chunker, allow_prune=args.allow_prune)
            try:
                stats = indexer.index_path(args.path, glob=args.glob)
            except PruneGuardTripped as exc:
                # The message carries the recovery instructions; a traceback would bury them.
                raise SystemExit(str(exc)) from exc
            # `files` counts what was RE-indexed, not what is in the index, so an unchanged
            # re-run reports 0/0 — which reads as "the index is empty" unless `skipped` is shown
            # beside it. `deleted` matters more: pruning is the destructive half of `index`, and
            # reporting it only through a log record meant a deletion could happen in silence.
            summary = f"indexed {stats.chunks} chunks from {stats.files} files"
            if stats.skipped:
                summary += f", {stats.skipped} unchanged"
            if stats.deleted:
                summary += f", pruned {stats.deleted} source(s) no longer on disk"
            print(summary)
    elif args.cmd == "forget":
        from recall.generation_store import GenerationStore
        from recall.generations import NoActiveGeneration

        generation_mode = os.environ.get("RECALL_ENV", "development").lower() == "production"
        # Keep a GenerationStore-typed handle alongside the widened one: the corpus probe below
        # exists only on the subclass, and narrowing here is what lets the type checker see it.
        gen_store: GenerationStore | None = (
            GenerationStore(args.dsn, embedder.dim, tenant=args.tenant)
            if generation_mode
            else None
        )
        forget_store: PgVectorStore = (
            gen_store
            if gen_store is not None
            else PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant)
        )
        with forget_store as store:
            store.check_schema()
            requested = list(dict.fromkeys(args.sources))
            # Reject a blank argument before anything commits. `recall forget "$A" "$B" --yes`
            # with one variable unset otherwise erased the first source, raised out of the
            # per-source loop, and printed nothing at all.
            if any(not source.strip() for source in requested):
                raise SystemExit(
                    "forget: empty source argument (an unset shell variable?); nothing deleted"
                )
            if gen_store is not None:
                # Widen the existence check, do not drop it, and ask the right question.
                # `source_content_hashes()` is scoped to ONE generation, so FILTERING on it
                # called a source that had left the active generation "not found" and left it
                # with its rows and no tombstone. But no check at all is not the answer either:
                # forgetting a never-indexed URI writes a permanent tombstone (nothing deletes
                # one, and `build()` skips every manifest entry it matches), so a typo would
                # irreversibly bar that URI. The question is "does the corpus contain this",
                # which the MANIFEST answers and chunk rows do not: an object that chunks to
                # nothing is built as `empty_objects` and writes no row, yet is unquestionably
                # part of the corpus and must be erasable.
                known = (
                    gen_store.sources_in_any_generation()
                    | gen_store.sources_in_any_manifest()
                    | gen_store.sources_in_legacy_table()
                )
                targets = [s for s in requested if s in known]
                unseen = [s for s in requested if s not in known]
                unseen_note = (
                    "not present in any generation, manifest, or the adopted v0.8 table, so "
                    f"NOT erased and NOT tombstoned (check for typos): {', '.join(unseen)}"
                )
            else:
                # The v0.8 table has no generations, so the probe covers everything the
                # tenant owns and an absent source really is a typo. Computed here rather than
                # above because the generation branch never reads it, and on a GenerationStore
                # it costs an active-generation lookup plus a DISTINCT scan.
                try:
                    visible_now = set(store.source_content_hashes())
                except NoActiveGeneration:
                    visible_now = set()
                targets = [s for s in requested if s in visible_now]
                unseen = [s for s in requested if s not in visible_now]
                unseen_note = f"not found (check for typos): {', '.join(unseen)}"
            if not args.yes:
                print(
                    f"DRY RUN: would forget {len(targets)} source(s): "
                    f"{', '.join(targets) if targets else '(none)'}"
                )
                if unseen:
                    print(unseen_note)
                print("nothing deleted — re-run with --yes to actually delete.")
            else:
                # One source per call: `delete_sources` commits a separate transaction each,
                # so a failure part way through leaves the earlier ones erased. Reporting from
                # a finally means a partial erasure is never silent.
                removed = 0
                erased: list[str] = []
                try:
                    for source in targets:
                        removed += store.delete_sources([source])
                        erased.append(source)
                finally:
                    if len(erased) == len(targets):
                        print(f"forgot {removed} chunk(s) from {len(erased)} source(s)")
                    else:
                        # Name them. On an irreversible path, "the rest" is not an answer the
                        # operator can act on, and the survivors are otherwise recoverable only
                        # by re-deriving the argument order by hand.
                        missed = [s for s in targets if s not in set(erased)]
                        print(
                            f"forgot {removed} chunk(s) from {len(erased)} of {len(targets)} "
                            f"source(s); NOT reached: {', '.join(missed)}"
                        )
                    if unseen:
                        print(unseen_note)
    elif args.cmd == "search":
        entail_judge = None
        if args.entail:
            from recall.entailment import QnliEntailmentJudge

            entail_judge = QnliEntailmentJudge()
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            from recall.generation_store import GenerationStore

            store_context: PgVectorStore = GenerationStore(
                args.dsn, embedder.dim, tenant=args.tenant
            )
        else:
            store_context = PgVectorStore(
                args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
            )
        with store_context as store:
            store.check_schema()
            _print_result(
                trusted_search(
                    store,
                    embedder,
                    args.query,
                    k=args.k,
                    calibration=calibration,
                    entailment=entail_judge,
                )
            )
    elif args.cmd == "demo":
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            raise SystemExit("the filesystem demo is unavailable in production")
        with PgVectorStore(
            args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
        ) as store:
            store.check_schema()
            stats = Indexer(store, embedder).index_path("corpus")
            print(f"indexed {stats.chunks} chunks from {stats.files} files\n")
            _run_queries(
                store,
                embedder,
                [
                    "what did we decide about caching?",
                    "do we inject retrieved context into the prompt?",
                    "how many requests per second can a client make?",
                    "how do we handle penguins on mars?",
                ],
                calibration,
            )
    elif args.cmd == "code":
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            raise SystemExit("local source indexing is unavailable in production")
        # index recall's own package source (content-agnostic engine, code-aware chunking)
        src = Path(__file__).resolve().parent
        with PgVectorStore(
            args.dsn, dim=embedder.dim, table="recall_code", tenant=args.tenant
        ) as store:
            store.check_schema()
            stats = Indexer(store, embedder, chunker=chunk_code).index_path(src, glob="**/*.py")
            print(f"indexed {stats.chunks} code chunks from {stats.files} files\n")
            _run_queries(
                store,
                embedder,
                [
                    "where is reciprocal rank fusion implemented?",
                    "how are embeddings stored in postgres?",
                    "how does cross-encoder reranking reorder hits?",
                ],
                calibration,
            )
if __name__ == "__main__":
    main()
