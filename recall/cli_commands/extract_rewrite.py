"""`recall extract` and `recall rewrite`: filesystem-only claim extraction and review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recall.lint import DEFAULT_GLOB

from recall.cli_commands._shared import _positive_int


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    # No `_opens_db`: extraction is an ingest-side filesystem concern and never connects. The
    # set of DB-opening commands is derived from these declarations, so leaving it off IS the
    # answer to the question that guard asks, not an omission.
    p_extract = sub.add_parser(
        "extract",
        help="extract structured truth claims from memo prose (no DB needed; writes nothing)",
    )
    p_extract.set_defaults(func=_run_extract)
    extract_sub = p_extract.add_subparsers(dest="extract_cmd", required=True)
    # `description` as well as `help`: `help` shows in the PARENT's listing, `description` in
    # this subparser's own `--help`, which is where someone checks what it will do to their
    # files. Stating "writes nothing" only in the parent listing leaves that question
    # unanswered exactly where it gets asked.
    _extract_run_blurb = (
        "Extract claims from a corpus. This writes nothing: it has no --apply, because "
        "declaring a claim needs a named human at `recall rewrite apply`. Review with "
        "`recall rewrite plan`."
    )
    p_extract_run = extract_sub.add_parser(
        "run", help=_extract_run_blurb, description=_extract_run_blurb
    )
    p_extract_run.add_argument("path")
    p_extract_run.add_argument("--glob", default=DEFAULT_GLOB)
    p_extract_run.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="read at most this many files. Targets still resolve against the WHOLE corpus, "
        "so sampling does not change the answers.",
    )
    p_extract_run.add_argument(
        "--recheck",
        action="store_true",
        help="re-call the engine on cached keys and report the mismatch rate, to MEASURE "
        "whether determinism holds rather than assume it (needs --cache)",
    )
    # A PATH again, now that there is a store behind it. It was briefly a boolean, because an
    # earlier version accepted a path, ignored it entirely and built a process-local cache, so
    # nothing was written to PATH and a second run hit nothing. Advertising a persistence that
    # does not exist is worse than not offering it; the flag came back when the persistence did.
    p_extract_run.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help="persist extraction results at PATH, so re-ingesting an unchanged memo does not "
        "re-pay for it. Also what makes --recheck possible.",
    )
    _status_vocabulary_help = (
        "comma-separated lifecycle words this corpus uses, e.g. Final,Rejected,Deferred. "
        "Defaults to the memo set. Matching is case-insensitive and the spelling given here "
        "is what is stored. This does NOT widen what `recall rewrite` may write."
    )
    p_extract_run.add_argument(
        "--status-vocabulary", default=None, metavar="W,X,Y", help=_status_vocabulary_help
    )
    _extract_show_blurb = (
        "Show the claims and refusals for a single file. Targets are resolved against the "
        "file's own directory, not against the file alone. This writes nothing."
    )
    p_extract_show = extract_sub.add_parser(
        "show", help=_extract_show_blurb, description=_extract_show_blurb
    )
    p_extract_show.add_argument("file")
    p_extract_show.add_argument("--glob", default=DEFAULT_GLOB)
    p_extract_show.add_argument(
        "--status-vocabulary", default=None, metavar="W,X,Y", help=_status_vocabulary_help
    )

    # No `_opens_db` here either: review and declaration are filesystem work.
    p_rewrite = sub.add_parser(
        "rewrite",
        help="review extracted claims and declare accepted ones in corpus frontmatter",
    )
    p_rewrite.set_defaults(func=_cmd_rewrite)
    rewrite_sub = p_rewrite.add_subparsers(dest="rewrite_cmd", required=True)

    _plan_blurb = "Show every proposal the corpus states, and change nothing."
    p_rw_plan = rewrite_sub.add_parser("plan", help=_plan_blurb, description=_plan_blurb)
    p_rw_plan.add_argument("path")
    p_rw_plan.add_argument("--glob", default=DEFAULT_GLOB)

    _apply_blurb = (
        "Declare ONE reviewed proposal in its memo. DRY RUN by default: it prints the plan and "
        "changes nothing unless --apply is given. --reviewer and --note are required because "
        "nothing reaches corpus metadata without a named human."
    )
    p_rw_apply = rewrite_sub.add_parser("apply", help=_apply_blurb, description=_apply_blurb)
    p_rw_apply.add_argument("path")
    p_rw_apply.add_argument("--glob", default=DEFAULT_GLOB)
    # Either identity, but one of them. `--claim` exists because `recall_rewrite_plan` over MCP
    # derives proposals from the STORE graph while this command derives them from the
    # filesystem extractor, and an id hashes in provider, tenant, generation and pipeline: the
    # two id spaces are disjoint, so every id that surface emits is one `--proposal` refuses.
    # Claim keys are generation independent, which is also why the rejection ledger uses them.
    _apply_id = p_rw_apply.add_mutually_exclusive_group(required=True)
    _apply_id.add_argument("--proposal", help="the proposal id to declare")
    _apply_id.add_argument(
        "--claim", help="the claim key to declare, as reported by recall_rewrite_plan over MCP"
    )
    p_rw_apply.add_argument(
        "--reviewer", required=True, help="identity of the human accepting this proposal"
    )
    p_rw_apply.add_argument(
        "--note", required=True, help="why it was accepted; kept in the audit record"
    )
    p_rw_apply.add_argument(
        "--apply", action="store_true", help="actually write the edit to the memo file"
    )

    # `reject` takes a PATH, which the original design sketch did not. The ledger is keyed by
    # CLAIM (relation plus the two normalised document names), deliberately not by proposal id,
    # because ids hash in the generation and would forget every rejection at the next re-index.
    # Resolving an id to a claim key therefore needs the corpus.
    _reject_blurb = (
        "Record a human's refusal so the proposal does not resurface. Needs the corpus, "
        "because the ledger is keyed by claim rather than by proposal id."
    )
    p_rw_reject = rewrite_sub.add_parser("reject", help=_reject_blurb, description=_reject_blurb)
    p_rw_reject.add_argument("path")
    p_rw_reject.add_argument("--glob", default=DEFAULT_GLOB)
    _reject_id = p_rw_reject.add_mutually_exclusive_group(required=True)
    _reject_id.add_argument("--proposal")
    _reject_id.add_argument("--claim", help="the claim key, as reported by recall_rewrite_plan")
    p_rw_reject.add_argument("--reviewer", required=True)
    p_rw_reject.add_argument("--note", required=True)

    _verify_blurb = "Check that every declared supersedes edge still resolves to one file."
    p_rw_verify = rewrite_sub.add_parser(
        "verify", help=_verify_blurb, description=_verify_blurb
    )
    p_rw_verify.add_argument("path")
    p_rw_verify.add_argument("--glob", default=DEFAULT_GLOB)


def _parse_status_vocabulary(raw: str | None) -> tuple[str, ...] | None:
    """Parse `--status-vocabulary`. `None` means the shipped memo set.

    The split is the flag's SHAPE; every judgement about the result belongs to
    `coerce_status_vocabulary`, which already refuses an empty list, a bare string, blank and
    non-str entries and casefold collisions, and which strips. Deliberately NOT the labelling
    runner's `tuple(v.strip() for v in raw.split(",") if v.strip())`: that comprehension swallows
    exactly the blanks the coercion exists to refuse, so `Final,,Rejected` would pass quietly and
    `,` would collapse to an empty vocabulary — which refuses every status claim at a BATCH rung,
    the original defect re-entered through the flag added to remove it.
    """
    if raw is None:
        return None
    from recall.truth_extraction.types import coerce_status_vocabulary

    try:
        return coerce_status_vocabulary(raw.split(","))
    except ValueError as exc:
        print(f"recall extract: --status-vocabulary: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _rejected_claims(ledger_path: Path) -> frozenset[str]:
    """Every claim key a human has rejected, read without creating the ledger.

    Returns an empty set when the sidecar does not exist: nothing has been rejected yet. A
    ledger that exists but cannot be read is a different matter and propagates, because it
    might hold a rejection for the very claim being asked about.
    """
    import sqlite3

    if not ledger_path.exists():
        return frozenset()
    try:
        with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
            return frozenset(
                row[0] for row in conn.execute("SELECT claim_key FROM rejected_claims")
            )
    except sqlite3.Error as exc:
        from recall.rewrite import RewriteRefused

        raise RewriteRefused(
            f"rejection ledger at {ledger_path} could not be read: {exc}"
        ) from exc


def _already_declared(root: Path, proposal: object) -> bool:
    """True when the memo already states this proposal's key, so it needs no second review."""
    from recall.document import parse_document
    from recall.rewrite import RewriteRefused, _resolve, destination, route_relation

    try:
        routed = route_relation(
            proposal.proposed_relation,  # type: ignore[attr-defined]
            proposal.subject_id,  # type: ignore[attr-defined]
            proposal.object_id,  # type: ignore[attr-defined]
        )
        if destination(routed.key) != "frontmatter":
            # A derived-block key is multi valued for `contradicts` and `same_entity`, so
            # "already there" is not a property of the key alone. Left to the write path.
            return False
        corpus_root = root if root.is_dir() else root.parent
        path = corpus_root / routed.edit_file
        if not path.is_file():
            # Falls back to the write path's OWN resolution, for the corpus names that are not
            # usable relative paths. A memo whose filename is not valid UTF-8 is named by the
            # stand-in `encodable_name` returns, so the join above opens nothing and every run
            # re-offered an edge that memo already declares as unreviewed work. A queue that
            # never converges is the defect DECLARED exists to prevent. The join stays first
            # because this runs once per proposal and `_resolve` walks the whole corpus.
            path = corpus_root / _resolve(corpus_root, routed.edit_file)
        meta = parse_document(path.read_text(encoding="utf-8-sig")).meta
    except (RewriteRefused, UnicodeDecodeError, OSError):
        return False
    return routed.key in meta


def _cmd_rewrite(args: argparse.Namespace) -> None:
    from recall.rewrite import RewriteRefused

    # `required=True` is satisfied by `--reviewer ""`. The gate at the parser is the first
    # half; this is the second. A gate a caller passes by typing nothing is a field, not a
    # person, and this is the one command in the library that edits a user's own memos.
    for field in ("reviewer", "note"):
        value = getattr(args, field, None)
        if value is not None and not value.strip():
            print(f"recall rewrite: --{field} must not be empty", file=sys.stderr)
            raise SystemExit(2)
    try:
        _run_rewrite(args)
    except RewriteRefused as exc:
        print(f"recall rewrite: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _run_rewrite(args: argparse.Namespace) -> None:
    """`recall rewrite plan|apply|reject|verify`. Filesystem only; never opens the database.

    Proposals are re-derived from the corpus on every verb rather than stored, because a
    proposal is a reading of the corpus as it stands now. See `rewrite.corpus_proposals`.
    """
    from datetime import datetime, timezone

    from recall.document import parse_document
    from recall.frontmatter import supersedes_key
    from recall.promotion import (
        accept_reviewed_proposal,
        promote_accepted_proposal,
        review_proposal,
    )
    from recall.rewrite import (
        RejectionLedger,
        apply_rewrite,
        claim_key,
        corpus_proposals,
        default_ledger_path,
        plan_rewrite,
    )

    root = Path(args.path)
    if not root.exists():
        print(f"recall rewrite: no such path: {root}", file=sys.stderr)
        raise SystemExit(2)

    if args.rewrite_cmd == "verify":
        # The check `recall lint` makes, scoped to what this command writes: a declared edge
        # whose target does not resolve is the defect `supersedes_key` exists for, and it is
        # exactly what a bad rewrite would leave behind.
        corpus_root = root if root.is_dir() else root.parent
        try:
            corpus_paths = sorted(corpus_root.glob(args.glob))
        except (ValueError, NotImplementedError, OSError) as exc:
            print(f"recall rewrite: --glob {args.glob!r}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        # Checking ONE memo still resolves against the corpus it lives in; scoping the corpus
        # to the single file reported every edge that does resolve as unresolved.
        paths = corpus_paths if root.is_dir() else [root]

        def _rel(path: Path) -> str:
            try:
                return path.relative_to(corpus_root).as_posix()
            except ValueError:
                return path.name

        # A LIST per key, not a single name. Overwriting collapsed `legal/old.md` and
        # `eng/old.md` onto one entry, so an edge naming `old.md` was reported as resolved
        # when it resolves to two files and therefore to none: `lint` calls that
        # `ambiguous-supersedes-target` and `_resolve` refuses to write it.
        by_key: dict[str, list[str]] = {}
        for path in corpus_paths:
            if path.is_file():
                by_key.setdefault(supersedes_key(path.name), []).append(_rel(path))

        unresolved = 0
        for path in paths:
            if not path.is_file():
                continue
            try:
                meta = parse_document(path.read_text(encoding="utf-8-sig")).meta
            except (UnicodeDecodeError, OSError) as exc:
                print(f"  UNREADABLE {_rel(path)}: {exc}")
                continue
            target = meta.get("supersedes")
            if not isinstance(target, str) or not target:
                continue
            matches = by_key.get(supersedes_key(target), [])
            if not matches:
                print(
                    f"  UNRESOLVED {_rel(path)}: supersedes {target!r}, "
                    f"which is not in the corpus"
                )
                unresolved += 1
            elif len(matches) > 1:
                print(
                    f"  AMBIGUOUS {_rel(path)}: supersedes {target!r}, which matches "
                    f"{len(matches)} files: {', '.join(matches)}"
                )
                unresolved += 1
        print(f"\n{unresolved} unresolved edge(s)")
        if unresolved:
            raise SystemExit(1)
        return

    # `RewriteRefused` is caught by the `_cmd_rewrite` wrapper above, which turns it into the
    # same `recall rewrite: <reason>` and exit 2. Catching it again here only risked the two
    # disagreeing, and the local handler referenced a name this function never imported.
    proposals = corpus_proposals(root, args.glob)
    ledger_path = default_ledger_path(root if root.is_dir() else root.parent)

    if args.rewrite_cmd == "plan":
        # Read WITHOUT creating the ledger. Opening it for write made `plan` create
        # `<root>/.recall/rejections.sqlite3` while printing "nothing written", and made the
        # command fail outright on a corpus the user cannot write to. A plan is a listing, not
        # a decision, so a missing ledger simply means nothing has been rejected yet.
        rejected = _rejected_claims(ledger_path)
        for proposal in proposals:
            claim = claim_key(
                proposal.proposed_relation, proposal.subject_id, proposal.object_id
            )
            if claim in rejected:
                mark = "REJECTED"
            elif _already_declared(root, proposal):
                # Otherwise an accepted proposal reappears every run, indistinguishable from
                # unreviewed work. It needs no storage: the corpus file already states it.
                mark = "DECLARED"
            else:
                mark = "review  "
            print(f"  {mark} {proposal.id}  {proposal.proposed_relation}")
            print(f"      {proposal.subject_id} -> {proposal.object_id}")
            print(f"      {proposal.explanation}")
            # Printed so a plan from THIS command and a plan from `recall_rewrite_plan` over
            # MCP name the same thing. Their proposal ids never match; their claim keys do.
            print(f"      claim {claim}")
        print(f"\n{len(proposals)} proposal(s)")
        print("dry run — nothing written. Declare one with `recall rewrite apply`.")
        return

    wanted = getattr(args, "claim", None)
    if wanted:
        chosen = next(
            (
                p
                for p in proposals
                if claim_key(p.proposed_relation, p.subject_id, p.object_id) == wanted
            ),
            None,
        )
    else:
        chosen = next((p for p in proposals if p.id == args.proposal), None)
    if chosen is None:
        given = f"claim {wanted!r}" if wanted else f"proposal {args.proposal!r}"
        print(
            f"recall rewrite: no {given} in {root}. "
            f"Run `recall rewrite plan {root}` to list them.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    now = datetime.now(timezone.utc)
    claim = claim_key(chosen.proposed_relation, chosen.subject_id, chosen.object_id)

    if args.rewrite_cmd == "reject":
        with RejectionLedger(ledger_path) as ledger:
            ledger.reject(
                claim, reviewer_id=args.reviewer, reason=args.note, rejected_at=now
            )
        print(f"recorded {args.reviewer}'s rejection of {chosen.id} as claim {claim}")
        return

    reviewed = review_proposal(
        chosen, reviewer_id=args.reviewer, reviewed_at=now, audit_note=args.note
    )
    fact = promote_accepted_proposal(accept_reviewed_proposal(reviewed), promoted_at=now)
    plan = plan_rewrite(root, fact)
    print(f"  {plan.edit_file}: + {plan.key}: {plan.value}  (in the {plan.block} block)")
    if not args.apply:
        # Dry run by DEFAULT: this edits the user's own documents, and a tool that rewrites
        # your memory the first time you try it has earned distrust.
        #
        # The ledger is consulted HERE, not only under --apply. Skipping it meant the preview
        # of an already rejected claim printed "Re-run with --apply to write this edge", and
        # --apply then refused it: a preview that promises a write the real run declines.
        if claim in _rejected_claims(ledger_path):
            print(f"not written: claim {claim} was already rejected by a reviewer")
            raise SystemExit(1)
        print("dry run — nothing written. Re-run with --apply to write this edge.")
        return
    with RejectionLedger(ledger_path) as ledger:
        result = apply_rewrite(root, fact, ledger=ledger, apply=True)
    if result.written:
        print("written")
        return
    # Non-zero when nothing was written. Exiting 0 on a refusal left a script unable to tell a
    # completed declaration from a declined one. 2 stays reserved for caller error.
    print(f"not written: {result.refusal}")
    raise SystemExit(1)


def _run_extract(args: argparse.Namespace) -> None:
    """`recall extract run|show`. Reads the corpus, writes nothing, never opens the database.

    Extraction is OFF unless `RECALL_TRUTH_EXTRACTION` is set, mirroring `entailment.py`, and an
    unknown engine name is refused rather than downgraded to the deterministic one: silently
    running a different engine than the one named would make the audit record wrong about how a
    claim was produced.
    """
    from recall.frontmatter import encodable_name
    from recall.truth_extraction import resolve_extraction_engine
    from recall.truth_extraction.extract import extract_corpus_claims_for_report

    try:
        engine = resolve_extraction_engine()
    except (ValueError, ImportError) as exc:
        print(f"recall extract: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if engine is None:
        print(
            "recall extract: extraction is off. Set RECALL_TRUTH_EXTRACTION=1 to enable it. "
            "See docs/TRUTH_EXTRACTION_DESIGN.md for what it does and what it refuses.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Parsed BEFORE the path check and before the glob, because it depends on nothing about the
    # corpus. The sibling `--recheck` check below carries the same reasoning from the other
    # direction: a flag error knowable at parse time must not arrive after a corpus of model
    # calls has been paid for, under a complete-looking report.
    status_vocabulary = _parse_status_vocabulary(args.status_vocabulary)

    root = Path(args.path if args.extract_cmd == "run" else args.file)
    if not root.exists():
        # Refused, not reported as an empty corpus. `0 claim(s) for review` on a typo reads as
        # "this corpus states nothing", which is the opposite of "I never looked". `lint` and
        # `check` both exit 2 here; this matches them.
        print(f"recall extract: no such path: {root}", file=sys.stderr)
        raise SystemExit(2)
    if args.extract_cmd == "show" and not root.is_file():
        print(f"recall extract show: not a file: {root}", file=sys.stderr)
        raise SystemExit(2)

    def _glob(directory: Path) -> list[Path]:
        # `Path.glob` raises ValueError on an empty pattern and NotImplementedError on an
        # absolute one. Neither is an OSError, so both escaped as a traceback.
        try:
            return sorted(directory.glob(args.glob))
        except (ValueError, NotImplementedError, OSError) as exc:
            print(f"recall extract: --glob {args.glob!r}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    if args.extract_cmd == "run":
        # The CORPUS is always the full glob. Slicing it with `--limit` shrinks the set a
        # supersession target resolves against, so a memo naming a real neighbour is refused
        # with "which is not a file in the corpus" about a file sitting right beside it. That
        # fabricated refusal reads exactly like a real one, and `--limit` is a sampling flag
        # whose entire purpose is to look at part of a corpus without changing the answers.
        corpus_paths = _glob(root) if root.is_dir() else _glob(root.parent)
        paths = corpus_paths if root.is_dir() else [root]
        if args.limit is not None:
            paths = paths[: args.limit]
    else:
        # `show` REPORTS one file but resolves targets against the corpus that file lives in,
        # for the same reason.
        paths = [root]
        corpus_paths = _glob(root.parent)

    corpus_root = root if root.is_dir() else root.parent

    def _key(path: Path) -> str:
        # Keyed by path relative to the corpus root, not by bare basename. The default glob is
        # recursive, so `legal/policy.md` and `eng/policy.md` collapsed onto one dict key: one
        # file was silently dropped, and the ladder's "matches N files in the corpus" refusal
        # could never fire, because the index had deduplicated the ambiguity away.
        #
        # `encodable_name` for the same reason it is in `rewrite._key`: a filename that is not
        # valid UTF-8 is a lone surrogate here, and it flows into the prompt, the cache key and
        # the report. Each of those was hardened separately; the boundary is one place.
        try:
            return encodable_name(path.relative_to(corpus_root).as_posix())
        except ValueError:
            return encodable_name(path.name)

    documents: dict[str, str] = {}
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            documents[_key(path)] = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError) as exc:
            # Per file, and note `UnicodeDecodeError` is a ValueError, NOT an OSError. One memo
            # that is not UTF-8 aborted the whole run and discarded every file already decoded.
            # `lint.py` and `fix.py` both catch the pair and keep going.
            unreadable.append((_key(path), str(exc)))
    corpus_names = tuple(sorted(_key(p) for p in corpus_paths if p.is_file()))

    # Validated BEFORE any extraction runs. Deferring it meant a user paid for a whole corpus of
    # model calls and then got exit 2 on a flag combination knowable at parse time, with a
    # complete looking report already printed above the error.
    if getattr(args, "recheck", False) and getattr(args, "cache", None) is None:
        print(
            "recall extract: --recheck needs --cache PATH; there is nothing to re-check "
            "against an empty cache",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cache = None
    if getattr(args, "cache", None) is not None:
        from recall.truth_extraction._sqlite_cache import (
            ExtractionCacheRefused,
            SqliteExtractionCache,
        )

        try:
            cache = SqliteExtractionCache(args.cache)
        except ExtractionCacheRefused as exc:
            # Refused at OPEN, before a single memo is read, because the alternative is
            # discovering halfway through a corpus that the path was somebody else's database.
            print(f"recall extract: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    try:
        extractions = extract_corpus_claims_for_report(
            documents,
            engine=engine,
            corpus_names=corpus_names,
            cache=cache,
            status_vocabulary=status_vocabulary,
        )
        for name, reason in unreadable:
            print(f"  UNREADABLE {name}: {reason}")
        for item in extractions:
            for claim in item.claims:
                print(f"  {item.file}: {claim.kind}")
                print(f"      quote {claim.quote!r}")
            for refusal in item.rejections:
                print(f"  SKIP {item.file}: {refusal.rung}: {refusal.reason}")
            if item.batch_rejection is not None:
                print(
                    f"  REFUSED {item.file}: {item.batch_rejection.rung}: "
                    f"{item.batch_rejection.reason}"
                )
        total = sum(len(item.claims) for item in extractions)
        print(f"\n{len(documents)} file(s) read, {total} claim(s) for review")
        if status_vocabulary is not None:
            # Named only when it is not the default: two runs whose outputs differ have nothing
            # on screen saying why otherwise. The parenthetical is why: this report is the ONLY
            # place a custom vocabulary is honoured. `recall rewrite` extracts under the shipped
            # set regardless, so a status claim printed above can vanish from `rewrite plan`
            # with nothing else on screen to explain it, right after this command's own closing
            # line says "review with `recall rewrite plan`".
            print(
                f"status vocabulary: {', '.join(status_vocabulary)} "
                "(measurement only; recall rewrite still writes the shipped set)"
            )

        if cache is not None and getattr(args, "recheck", False):
            from recall.truth_extraction._cache import recheck_cached_extractions

            # The SAME `corpus_names` the extraction ran with. `extraction_cache_key` hashes them,
            # so passing the document keys instead produced a different key for every file, every
            # lookup missed, and the report read "0 checked, rate not measured" without saying why.
            report = recheck_cached_extractions(
                documents,
                engine=engine,
                corpus_names=corpus_names,
                cache=cache,
                # The SAME vocabulary the extraction ran with, for the same reason as
                # `corpus_names`: it is in the cache key AND the prompt. Omitted, every lookup
                # misses and the report reads `checked=0` — a determinism measurement that
                # silently became a non-measurement.
                status_vocabulary=status_vocabulary,
            )
            rate = "not measured" if report.mismatch_rate is None else f"{report.mismatch_rate:.3f}"
            print(
                f"recheck: {report.checked} checked, {report.mismatched} mismatched, "
                f"{report.errored} errored, rate {rate}"
            )

        # Dry run is the ONLY run. This command has no --apply, because declaring a claim needs a
        # named human at `recall rewrite apply`, not a flag here.
        print("nothing written — review with `recall rewrite plan`")
    finally:
        if cache is not None:
            # try/finally, so the counters are REPORTED and the sqlite connection closed on
            # every exit path. Doing it only after the last print meant any exception in
            # extraction, printing or recheck leaked the connection and, worse, skipped the
            # one line that makes a silently degraded cache visible.
            if cache.write_failures or cache.corrupt or cache.stale:
                # Read directly, not through getattr with a default: a renamed counter must
                # break a test, not quietly report a degraded run as clean.
                print(
                    f"cache: {cache.write_failures} write failure(s), "
                    f"{cache.corrupt} unusable, {cache.stale} from an older cache version"
                )
            cache.close()
