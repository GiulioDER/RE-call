from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from recall.calibration import Calibration, load_for
from recall.calibration_v2 import CalibrationBindingError, CalibrationRepository
from recall._env import load_dotenv
from recall.entailment import resolve_entailment_judge
from recall.embeddings import Embedder, resolve_embedder
from recall.index import Indexer, PruneGuardTripped, chunk_code, chunk_text
from recall.lint import DEFAULT_GLOB
from recall.observability import configure_logging
from recall.store import DEFAULT_TENANT, PgVectorStore, require_secure_dsn, warn_if_insecure_dsn
from recall.trust import terminal_safe, trusted_search
from recall.types import TrustedResult

load_dotenv()
DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


def _make_embedder(name: str) -> Embedder:
    try:
        return resolve_embedder(name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
            f" -> use {terminal_safe(h.validity.superseded_by)}"
            if h.validity.superseded_by
            else ""
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
    entailment=None,
) -> None:
    policy, calibration = _cli_trust(embedder, calibration)
    for q in queries:
        _print_result(
            trusted_search(
                store,
                embedder,
                q,
                calibration=calibration,
                entailment=entailment,
                policy=policy,
            )
        )
        print()


def _cli_policy():
    from recall.trust_policy import TrustPolicy

    return TrustPolicy.from_env()


def _cli_trust(
    embedder: Embedder, calibration: Calibration | None
) -> tuple[object, Calibration | None]:
    policy = _cli_policy()
    if calibration is None and not policy.strict:
        from recall.calibration import Calibration as _Calibration
        from recall.guards import DEFAULT_GAP_THRESHOLD

        calibration = _Calibration(embedder=embedder.name, threshold=DEFAULT_GAP_THRESHOLD)
        print(
            f"[development] using an UNCERTIFIED demonstration threshold of "
            f"{DEFAULT_GAP_THRESHOLD}. This is not a calibration: it is bound to no tenant, "
            f"generation or corpus, and production refuses rather than assuming it."
        )
    return policy, calibration


def _cmd_schema(args: argparse.Namespace) -> None:
    from recall.schema import apply_migrations, schema_plan, schema_status, serving_grants

    if args.schema_cmd == "plan":
        pending = schema_plan(args.dsn, table=args.table, dim=args.dim)
        for migration in pending:
            print(f"would apply {migration.version} {migration.filename}")
        if not pending:
            print("schema is current")
        return
    if args.schema_cmd == "apply":
        applied = apply_migrations(args.dsn, table=args.table, dim=args.dim)
        for migration in applied:
            print(f"applied {migration.version} {migration.filename}")
        if not applied:
            print("schema is current")
        return
    if args.schema_cmd == "status":
        status = schema_status(args.dsn, table=args.table, dim=args.dim)
        print(f"current: {status.current_version or 'none'}")
        print(f"required: {status.required_version}")
        print(f"compatible: {'yes' if status.compatible else 'no'}")
        return
    if args.schema_cmd == "grants":
        for statement in serving_grants(args.role, table=args.table, enterprise=args.enterprise):
            print(statement)
        return
    raise SystemExit(f"unknown schema subcommand: {args.schema_cmd}")


def _cmd_calibrate(args: argparse.Namespace) -> None:
    embedder = _make_embedder(args.embedder)
    repo = CalibrationRepository(args.dsn, args.tenant)
    try:
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        artifact = repo.calibrate(args.generation, queries, embedder)
        if args.publish:
            artifact = repo.publish(artifact.calibration_id)
    except (OSError, json.JSONDecodeError, CalibrationBindingError) as exc:
        raise SystemExit(2) from exc
    print(f"calibration: {artifact.calibration_id}")
    print(f"status: {artifact.status.value}")
    print(f"generation: {artifact.generation_id}")
    print(f"tenant: {artifact.tenant_id}")


def _cmd_calibration(args: argparse.Namespace) -> None:
    repo = CalibrationRepository(args.dsn, args.tenant)
    if args.calibration_cmd == "list":
        for row in repo.list_records():
            print(
                f"{row['calibration_id']}  {row['generation_id']}  "
                f"{row['lifecycle_state']}  certified={row['certified']}  "
                f"{row['created_at']}"
            )
        return
    if args.calibration_cmd == "show":
        print(json.dumps(repo.show_record(args.calibration_id), indent=2, sort_keys=True))
        return
    if args.calibration_cmd == "export":
        path = repo.export_bundle(args.calibration_id, args.output)
        print(path)
        return
    if args.calibration_cmd == "import":
        calibration_id = repo.import_bundle(args.path)
        print(calibration_id)
        return
    raise SystemExit(f"unknown calibration subcommand: {args.calibration_cmd}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # clean UTF-8 output on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    # Without this the library's loggers have no handler, so every _log.info is discarded — which
    # is how `index` came to prune rows while printing nothing about it.
    configure_logging()
    parser = argparse.ArgumentParser(prog="recall")
    parser.add_argument(
        "--dsn",
        "--serving-dsn",
        "--migration-dsn",
        dest="dsn",
        default=DEFAULT_DSN,
        help="database DSN used by the selected command",
    )
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument(
        "--table", default="chunks",
        help="table to read/write (default: chunks). Use a throwaway name to keep an "
             "experiment out of your real memory index.",
    )
    parser.add_argument(
        "--tenant", default=DEFAULT_TENANT,
        help=f"tenant namespace to operate on (default: {DEFAULT_TENANT}). Every command is "
             f"scoped to one tenant; `forget` in particular deletes nothing outside it, so an "
             f"erasure request against another tenant needs this flag.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_schema = sub.add_parser("schema", help="inspect and apply versioned database migrations")
    p_schema.add_argument("--dim", type=int, required=True)
    schema_sub = p_schema.add_subparsers(dest="schema_cmd", required=True)
    schema_sub.add_parser("plan", help="show pending migrations")
    schema_sub.add_parser("apply", help="apply pending migrations")
    schema_sub.add_parser("status", help="show migration status")
    p_grants = schema_sub.add_parser("grants", help="print grant statements")
    p_grants.add_argument("--role", required=True)
    p_grants.add_argument("--enterprise", action="store_true")

    p_index = sub.add_parser("index", help="index a folder of markdown or code")
    p_index.add_argument("path")
    p_index.add_argument(
        "--glob", default=DEFAULT_GLOB,
        help="file glob to index — e.g. '**/*.py' for code (auto-uses code chunking). Default: markdown.",
    )
    p_index.add_argument(
        "--allow-prune", action="store_true",
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
        "sources", nargs="+",
        help="source value(s) to forget, exactly as stored (see the `source` field in "
             "`recall search` output)",
    )
    p_forget.add_argument(
        "--yes", action="store_true",
        help="actually delete. Without this flag, forget only PREVIEWS what would be removed "
             "and changes nothing — this command is the right-to-erasure path, it is "
             "irreversible, and it is also invoked from scripts, so a typo or an unattended "
             "run must not silently wipe a corpus. Re-run with --yes once the preview looks right.",
    )

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument(
        "--entail", action="store_true",
        help="opt-in entailment stage: demote hits that don't answer the query "
             "(requires recall[entail]; downloads the QNLI judge on first use)",
    )

    sub.add_parser("demo", help="index corpus/ and run sample memory queries")
    sub.add_parser("code", help="index recall's own source and run sample code queries")
    sub.add_parser("setup", help="run the first install wizard and write a local .env file")

    p_lint = sub.add_parser(
        "lint",
        help="check a corpus's supersession graph for broken/missing edges (no DB needed)",
    )
    p_lint.add_argument("path")
    p_lint.add_argument("--glob", default=DEFAULT_GLOB)
    p_lint.add_argument(
        "--semantic", action="store_true",
        help="also run the retrieval-based MISSING-edge check: flag memos highly similar to a "
             "prior closed decision they don't reference (needs the DB + embedder; opt-in)",
    )
    p_lint.add_argument(
        "--fix", action="store_true",
        help="propose the frontmatter `supersedes:` edge for each closure marker whose target "
             "is provable. DRY RUN by default — prints the plan and changes nothing.",
    )
    p_lint.add_argument(
        "--apply", action="store_true",
        help="with --fix, actually write the proposed edges to the memo files",
    )
    p_lint.add_argument(
        "--threshold", type=float, default=None,
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
        "--corpus", default=None,
        help="corpus dir used to filter candidates to real documents (default: each file's own "
             "directory)",
    )
    p_check.add_argument(
        "--strict", action="store_true",
        help="exit 1 when a memo needs an edge — use this in a pre-commit hook",
    )

    p_calibrate = sub.add_parser(
        "calibrate",
        help="create and optionally publish a calibration",
    )
    p_calibrate.add_argument("--generation", required=True)
    p_calibrate.add_argument("--queries", required=True)
    p_calibrate.add_argument("--publish", action="store_true")

    p_calibration = sub.add_parser(
        "calibration",
        help="inspect and exchange calibration artifacts",
    )
    cal_sub = p_calibration.add_subparsers(dest="calibration_cmd", required=True)
    cal_sub.add_parser("list", help="list calibration records")
    p_show = cal_sub.add_parser("show", help="show a calibration record")
    p_show.add_argument("calibration_id")
    p_export = cal_sub.add_parser("export", help="export a calibration bundle")
    p_export.add_argument("calibration_id")
    p_export.add_argument("--output", required=True)
    p_import = cal_sub.add_parser("import", help="import a calibration bundle")
    p_import.add_argument("path")

    args = parser.parse_args(argv)
    db_backed_cmds = {
        "schema",
        "index",
        "forget",
        "search",
        "demo",
        "code",
        "calibrate",
        "calibration",
    }
    if args.cmd in db_backed_cmds:
        require_secure_dsn(args.dsn)
    else:
        warn_if_insecure_dsn(args.dsn)  # loud stderr note if default creds target a remote host

    if args.cmd == "schema":
        _cmd_schema(args)
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
            cal = load_for(emb.name)
            thr = args.threshold if args.threshold is not None else (
                cal.threshold if cal else 0.70
            )
            chains = semantic_lint(args.dsn, emb, args.path, threshold=thr, glob=args.glob)
            for c in chains:
                print(f"warning  unlinked-chain             {c.new_memo}: highly similar "
                      f"(cos={c.cosine:.2f}) to closed decision {c.prior!r} it does not "
                      f"reference — add `supersedes: {c.prior}`?")
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

    if args.cmd == "index":
        embedder = _make_embedder(args.embedder)
        chunker = chunk_code if args.glob.endswith(".py") else chunk_text
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant) as store:
            store.ensure_schema()
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
        embedder = _make_embedder(args.embedder)
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant) as store:
            store.ensure_schema()
            requested = list(dict.fromkeys(args.sources))
            existing = store.source_content_hashes()
            found = [s for s in requested if s in existing]
            not_found = [s for s in requested if s not in existing]
            if not args.yes:
                print(f"DRY RUN: would forget {len(found)} source(s): "
                      f"{', '.join(found) if found else '(none)'}")
                if not_found:
                    print(f"not found (check for typos): {', '.join(not_found)}")
                print("nothing deleted — re-run with --yes to actually delete.")
            else:
                removed = store.delete_sources(found)
                print(f"forgot {removed} chunk(s) from {len(found)} source(s)")
                if not_found:
                    print(f"not found (check for typos): {', '.join(not_found)}")
    elif args.cmd == "search":
        embedder = _make_embedder(args.embedder)
        calibration = load_for(embedder.name)
        if args.entail:
            from recall.entailment import QnliEntailmentJudge

            entail_judge = QnliEntailmentJudge()
        else:
            entail_judge = resolve_entailment_judge()
        policy, calibration = _cli_trust(embedder, calibration)
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant) as store:
            store.ensure_schema()
            _print_result(
                trusted_search(store, embedder, args.query, k=args.k, calibration=calibration,
                               entailment=entail_judge, policy=policy)
            )
    elif args.cmd == "demo":
        embedder = _make_embedder(args.embedder)
        calibration = load_for(embedder.name)
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant) as store:
            store.ensure_schema()
            stats = Indexer(store, embedder).index_path("corpus")
            print(f"indexed {stats.chunks} chunks from {stats.files} files\n")
            _run_queries(store, embedder, [
                "what did we decide about caching?",
                "do we inject retrieved context into the prompt?",
                "how many requests per second can a client make?",
                "how do we handle penguins on mars?",
            ], calibration, entailment=resolve_entailment_judge())
    elif args.cmd == "code":
        embedder = _make_embedder(args.embedder)
        calibration = load_for(embedder.name)
        # index recall's own package source (content-agnostic engine, code-aware chunking)
        src = Path(__file__).resolve().parent
        with PgVectorStore(args.dsn, dim=embedder.dim, table="recall_code", tenant=args.tenant) as store:
            store.ensure_schema()
            stats = Indexer(store, embedder, chunker=chunk_code).index_path(src, glob="**/*.py")
            print(f"indexed {stats.chunks} code chunks from {stats.files} files\n")
            _run_queries(store, embedder, [
                "where is reciprocal rank fusion implemented?",
                "how are embeddings stored in postgres?",
                "how does cross-encoder reranking reorder hits?",
            ], calibration, entailment=resolve_entailment_judge())
    elif args.cmd == "setup":
        from recall.setup import run_setup_wizard

        run_setup_wizard(dsn=args.dsn)
    elif args.cmd == "calibrate":
        _cmd_calibrate(args)
    elif args.cmd == "calibration":
        _cmd_calibration(args)


if __name__ == "__main__":
    main()
