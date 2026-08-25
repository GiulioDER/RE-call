"""`recall index`, `forget`, `search`, and the `demo`/`code` sample commands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from recall.context import context_policy_for_profile
from recall.embeddings import embedding_profile_id
from recall.index import (
    DEFAULT_INDEX_GLOB,
    Indexer,
    PruneGuardTripped,
    chunk_code,
    chunk_text,
    head_commit,
)
from recall.index_lock import ConcurrentIndex
from recall.retriever import DocumentExpansionPolicy
from recall.store import PgVectorStore
from recall.trust import terminal_safe, trusted_search
from recall.trust_policy import TrustRefusal
from recall.types import TrustedResult
from recall_mcp.translation import provider_from_env, translate_for_display

from recall.cli_commands._shared import (
    _cli_trust,
    _entailment_judge,
    _make_embedder,
    _print_result,
    _run_queries,
)
from recall._env import env_is_production


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_index = sub.add_parser("index", help="index a folder of supported documents or code")

    p_index.set_defaults(_opens_db=True, func=_cmd_index)
    p_index.add_argument("path")
    p_index.add_argument(
        "--glob",
        default=DEFAULT_INDEX_GLOB,
        help="file glob to index, for example '**/*.py' for code. Default: supported documents and code.",
    )
    p_index.add_argument(
        "--project",
        default=None,
        help="stamp every chunk with the project that produced it. Not inferred from the path: a "
             "directory name is not a project, and a guessed value reads as authoritative while "
             "being wrong in every worktree.",
    )
    p_index.add_argument(
        "--no-commit-stamp",
        action="store_true",
        help="do not record the repository's HEAD on each chunk. The commit is what makes a stale "
             "chunk DETECTABLE rather than merely suspected, and it cannot be reconstructed later.",
    )
    p_index.add_argument(
        "--batch-chunks",
        type=int,
        default=None,
        help="chunks to embed per batch. Bounds the embedder's peak allocation: fastembed pads a "
             "batch to its longest member, so a large batch of long chunks asks onnxruntime for "
             "gigabytes and fails PARTWAY THROUGH a run. Defaults to RECALL_INDEX_BATCH_CHUNKS if "
             "the host sets one, else 512.",
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

    p_forget.set_defaults(_opens_db=True, func=_cmd_forget)
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

    p_search.set_defaults(_opens_db=True, func=_cmd_search)
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument(
        "--entail",
        action="store_true",
        # No install command in this string, deliberately. argparse wraps help through
        # `textwrap.wrap`, which defaults to `break_on_hyphens=True`, so `recall-rag[entail]`
        # renders as `recall-` / `rag[entail]` at COLUMNS 63-69 and 111-123 — including 120,
        # which is a very common terminal width. A command that arrives pre-broken is worse
        # than no command. The exact line lives in the ImportError that fires when the extra is
        # actually missing (`recall/entailment.py`), which argparse never touches.
        help="opt-in entailment stage: demote hits that don't answer the query "
        "(requires the entail extra; downloads the QNLI judge on first use)",
    )
    p_search.add_argument(
        "--evidence",
        action="store_true",
        help="also print the generator-neutral evidence bundle and the exact prompt it renders "
        "to, as JSON. Only verdict-ok hits enter the bundle; document expansion groups them "
        "in document order. An abstention produces an empty bundle. Additive: the normal "
        "listing is printed either way.",
    )
    p_search.add_argument(
        "--expand-documents",
        action="store_true",
        help="for relational queries, rerun calibrated retrieval inside the top source documents "
        "and assemble evidence in document order",
    )
    p_search.add_argument(
        "--locale",
        help="optional presentation language for an additive localized display; canonical text "
        "and evidence remain unchanged",
    )


def register_demo_code(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sub.add_parser("demo", help="index corpus/ and run sample memory queries").set_defaults(
        _opens_db=True, func=_cmd_demo
    )
    sub.add_parser(
        "code", help="index recall's own source and run sample code queries"
    ).set_defaults(_opens_db=True, func=_cmd_code)


def _print_localized_result(result: TrustedResult, locale: str) -> None:
    """Print optional display translations without changing the canonical CLI result."""

    from recall_mcp.translation import normalize_locale

    try:
        normalized = normalize_locale(locale)
        if normalized is None:
            return
        provider = provider_from_env()
        values, translated, warning = translate_for_display(
            [hit.chunk.text for hit in result.hits], normalized, provider
        )
    except ValueError as exc:
        raise SystemExit(f"translation: {exc}") from exc
    print(f"[localized:{normalized} provider={provider.name} translated={translated}]")
    if warning:
        print(f"  warning: {warning}")
    for hit, value in zip(result.hits, values, strict=True):
        print(f"  chunk_id={terminal_safe(hit.chunk.id)!r}  {terminal_safe(value)!r}")


def _print_evidence(
    result: TrustedResult, max_items: int, *, document_mode: bool = False
) -> None:
    """Print the generator-neutral evidence bundle and the exact prompt it renders to.

    JSON, not prose, and that is a safety property rather than a formatting preference. Every
    string here is corpus-controlled, and `json.dumps` escapes control characters — so the ANSI
    payload `terminal_safe` strips from the human-readable listing above arrives as a literal
    `\\u001b` here instead of driving the terminal. The operator sees the byte that is actually in
    their corpus, which is what a debugging surface owes them.

    This is the CLI's whole exposure of `recall.evidence`: the bundle a generator would be given,
    plus `system` and `user` exactly as `render_evidence_prompt` produces them, so an operator can
    inspect the boundary without writing a program against the library.
    """
    from recall.evidence import EvidencePolicy, build_evidence_bundle, render_evidence_prompt

    # `max(1, ...)`: `-k` has no lower bound, and `EvidencePolicy` refuses `max_items < 1`, so
    # `recall search q -k 0 --evidence` raised an uncaught ValueError out of a dataclass
    # constructor. A CLI flag combination must not produce a traceback.
    bundle = build_evidence_bundle(
        result,
        EvidencePolicy(
            max_items=max(1, max_items),
            bundle_mode="document" if document_mode else "retrieval",
        ),
    )
    system, user = render_evidence_prompt(bundle)
    payload = {
        "bundle": asdict(bundle),
        "prompt": {"system": system, "user": user},
    }
    print(json.dumps(payload, indent=2, default=str))


def _cmd_index(args: argparse.Namespace) -> None:
    embedder = _make_embedder(args.embedder)
    if env_is_production():
        raise SystemExit(
            "local filesystem indexing is development-only; build from an immutable S3 "
            "manifest in production"
        )
    chunker = chunk_code if args.glob.endswith(".py") else chunk_text
    with PgVectorStore(
        args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
    ) as store:
        store.check_schema()
        # Stamped by default, opt OUT rather than opt in. A corpus indexed without a commit
        # cannot have one added afterwards, and the run that skips it is always the run nobody
        # was watching.
        commit = None if args.no_commit_stamp else head_commit(args.path)
        indexer = Indexer(
            store,
            embedder,
            chunker=chunker,
            context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
            allow_prune=args.allow_prune,
            project=args.project,
            indexed_commit=commit,
            batch_chunks=args.batch_chunks,
        )
        try:
            stats = indexer.index_path(args.path, glob=args.glob)
        except PruneGuardTripped as exc:
            # The message carries the recovery instructions; a traceback would bury them.
            raise SystemExit(str(exc)) from exc
        except ConcurrentIndex as exc:
            # Same reasoning: this is an expected outcome of two sessions closing at once,
            # not a defect, and its message names the holder and what to do about it.
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


def _cmd_forget(args: argparse.Namespace) -> None:
    from recall.generation_store import GenerationStore
    from recall.generations import NoActiveGeneration

    embedder = _make_embedder(args.embedder)
    generation_mode = env_is_production()
    # Keep a GenerationStore-typed handle alongside the widened one: the corpus probe below
    # exists only on the subclass, and narrowing here is what lets the type checker see it.
    gen_store: GenerationStore | None = (
        GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) if generation_mode else None
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
            # Scoped to `requested`, and the SAME call the MCP surface makes. Asking the
            # wholesale question here instead put the two surfaces on separate copies of
            # the live-state list, and only this one was pinned by a test.
            known = (
                gen_store.sources_in_any_generation()
                | gen_store.manifest_uris_matching(list(requested))
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


def refusal_message(exc: TrustRefusal) -> str:
    """Render a strict-mode refusal as the CLI error it is, rather than a traceback.

    A refusal is the gate working. Before this, `recall search` let `TrustRefusal` escape to the
    top of the interpreter, so the single most common outcome of a fresh install -- documented as
    exactly that on the troubleshooting page -- arrived as an eleven-frame Python traceback ending
    in `recall.trust_policy.TrustRefusal: INDEX_NOT_READY: refused in strict trust mode`. That
    reads as a crash in the tool, which is the opposite of what happened, and it buries the one
    line that says what to do next.

    Three things go in, and the choice of each is the point:

    - **The code, first and alone on its line.** `TrustFailureCode` is a published interface that
      callers automate on, and the spelling is pinned by test. Someone reading this error is one
      grep away from the table in `docs/CALIBRATION.md` that explains it.
    - **`exc.advice`**, which already exists for precisely this and states, per code, that no
      trustworthy decision was possible. That distinction -- the gate could not run, as against
      the gate ran and found nothing -- is the whole reason the codes exist, so it must survive
      into the operator-facing text rather than being paraphrased here.
    - **The identity**, because "which tenant, which generation" is the first question anyone
      asks and it is already on the exception.

    ⛔ **No corpus bytes, and that needs no filter here.** `TrustRefusal` is built only from
    fields the system controls: no chunk text, no source names, no preview, not even the query.
    This function cannot leak what it was never given, which is the property the exception was
    designed around and the reason it is safe to print in full.

    `terminal_safe` is still applied to the identifiers. They are system-controlled in the sense
    that matters for leakage, but `--tenant` is operator-supplied and a tenant name carrying
    `\\x1b[2K\\r` could erase the line it was printed on -- the same trick `terminal_safe` exists
    to stop for corpus-controlled strings elsewhere in this module.
    """
    lines = [
        f"{exc.code.value}: refused in strict trust mode.",
        "",
        exc.advice,
        "",
        f"  tenant             {terminal_safe(exc.tenant_id) or '-'}",
        f"  generation         {terminal_safe(exc.generation_id) or '-'}",
        f"  calibration status {terminal_safe(exc.calibration_status) or '-'}",
    ]
    if exc.calibration_id:
        lines.append(f"  calibration        {terminal_safe(exc.calibration_id)}")
    lines += [
        "",
        # Named as an INSPECTION mode with its cost stated, never as the fix. A relaxed gate has
        # no failure mode once it is unnecessary -- it stops erroring and quietly stamps
        # `degraded` on answers that had earned `trusted` -- so an error message that recommends
        # it without saying what it costs is how a workaround outlives the problem it solved.
        "To look at results while you finish setting up, RECALL_TRUST_MODE=development retrieves",
        "and marks every result degraded with no certified threshold behind it. It is for",
        "inspection, not for serving. docs/CALIBRATION.md explains every code above.",
    ]
    return "\n".join(lines)


def _cmd_search(args: argparse.Namespace) -> None:
    embedder = _make_embedder(args.embedder)
    # ⚠️ Deliberately NOT `load_for(embedder.name)`, and a bug audit talked me into that once.
    #
    # `trusted_search` only consults the generation-bound resolver when `calibration is None`
    # (recall/trust.py). Passing a legacy artifact sets calibration_status="legacy_unbound",
    # which the strict policy maps to CALIBRATION_UNCERTIFIED, so pre-loading it REFUSES
    # searches on a deployment that has a properly certified, generation-bound calibration.
    # recall/trust.py states the rule directly: legacy JSON "is deliberately never auto-loaded:
    # it has no tenant, generation, pipeline, corpus, or labelled query-set binding".
    #
    # 🔑 The open consequence, which is a DESIGN question and not an oversight: the artifact
    # `recall calibrate` writes is therefore not read back by this path. Resolve that by
    # deciding where install-time calibration binds, not by reinstating the line below.
    calibration = None
    # `resolve_entailment_judge` reads RECALL_ENTAILMENT (the opt-in the setup wizard
    # writes) plus RECALL_ENTAILMENT_MODEL / _REVISION. Constructing QnliEntailmentJudge()
    # directly ignored all three, so a pinned model was silently replaced by the default
    # download. The explicit --entail flag still forces it on when the env says nothing.
    # `--entail` resolves with the opt-in FORCED and never consults the env's own value,
    # so a malformed RECALL_ENTAILMENT cannot defeat an explicit flag. Checking the plain
    # resolver first would refuse before the flag was ever considered. Forcing goes THROUGH
    # the resolver rather than constructing the judge bare, because the bare form ignores
    # RECALL_ENTAILMENT_MODEL/_REVISION — the defect this block exists to fix. `recall
    # setup` writes RECALL_ENTAILMENT="0", so the forcing path is the common one.
    entail_judge = _entailment_judge(force=True) if args.entail else _entailment_judge()
    if env_is_production():
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
        _search_policy, _search_calibration = _cli_trust(embedder, calibration)
        # RECALL_DECISION_LEDGER=1 appends this decision (or its refusal) to the tenant's
        # audit table. Off unless the operator asked; a malformed value warns and stays off.
        from recall.decision_ledger import DecisionLedger

        # Caught at the call, not around the whole command: a refusal from `trusted_search` is a
        # policy decision with a rendered form, while an exception from the printing below would
        # be a real bug and must keep its traceback.
        try:
            _search_result = trusted_search(
                store,
                embedder,
                args.query,
                # `-k` has no lower bound and `trusted_search`
                # refuses k < 1 as its FIRST statement, so `-k 0` tracebacked out of the
                # library before any of this command's own guards were reached. Clamped at
                # the source; the clamp in `_print_evidence` stays as defence in depth.
                k=max(1, args.k),
                calibration=_search_calibration,
                entailment=entail_judge,
                policy=_search_policy,
                document_expansion=(
                    DocumentExpansionPolicy(enabled=True) if args.expand_documents else None
                ),
                ledger=DecisionLedger.from_env(store, actor="cli"),
            )
        except TrustRefusal as exc:
            raise SystemExit(refusal_message(exc)) from exc
        _print_result(_search_result)
        if args.locale:
            _print_localized_result(_search_result, args.locale)
        if args.evidence:
            _print_evidence(
                _search_result,
                max_items=args.k,
                document_mode=args.expand_documents,
            )


def _cmd_demo(args: argparse.Namespace) -> None:
    embedder = _make_embedder(args.embedder)
    # Never auto-loaded; see the note beside `calibration = None` in `_cmd_search` above.
    calibration = None
    if env_is_production():
        raise SystemExit("the filesystem demo is unavailable in production")
    # Resolved BEFORE the store opens and the corpus is indexed: a bad
    # RECALL_ENTAILMENT value raises, and failing after the expensive work is the
    # shape `search` already avoids.
    _demo_judge = _entailment_judge()
    with PgVectorStore(
        args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
    ) as store:
        store.check_schema()
        stats = Indexer(
            store,
            embedder,
            context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
        ).index_path("corpus")
        print(f"indexed {stats.chunks} chunks from {stats.files} files\n")
        _run_queries(
            store,
            embedder,
            [
                "what did we decide about caching?",
                "do we inject retrieved context into the prompt?",
                "how many requests per second can a client make?",
                # A deliberately unanswerable demo query. Its subject is chosen NOT to appear
                # in `recall/eval/offtopic_subjects.json`: a distinctive word from that pool,
                # written anywhere under `recall/`, disqualifies its subject for every code
                # corpus rooted at this repository. `tests/test_eval_synthetic.py` asserts it.
                "how do we handle llamas on mars?",
            ],
            calibration,
            _demo_judge,
        )


def _cmd_code(args: argparse.Namespace) -> None:
    embedder = _make_embedder(args.embedder)
    # Never auto-loaded; see the note beside `calibration = None` in `_cmd_search` above.
    calibration = None
    if env_is_production():
        raise SystemExit("local source indexing is unavailable in production")
    # index recall's own package source (content-agnostic engine, code-aware chunking)
    src = Path(__file__).resolve().parents[1]
    # Resolved BEFORE the store opens and the corpus is indexed: a bad
    # RECALL_ENTAILMENT value raises, and failing after the expensive work is the
    # shape `search` already avoids.
    _demo_judge = _entailment_judge()
    with PgVectorStore(
        args.dsn, dim=embedder.dim, table="recall_code", tenant=args.tenant
    ) as store:
        store.check_schema()
        stats = Indexer(
            store,
            embedder,
            chunker=chunk_code,
            context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
        ).index_path(src, glob="**/*.py")
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
            _demo_judge,
        )
