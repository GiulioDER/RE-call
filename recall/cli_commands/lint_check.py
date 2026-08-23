"""`recall lint` and `recall check`: corpus supersession-graph checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recall.calibration import load_for
from recall.embeddings import Embedder
from recall.lint import DEFAULT_GLOB

from recall.cli_commands._shared import _make_embedder


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_lint = sub.add_parser(
        "lint",
        help="check a corpus's supersession graph for broken/missing edges (no DB needed)",
    )

    p_lint.set_defaults(_opens_db=True, func=_cmd_lint)
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
    p_check.set_defaults(func=_cmd_check)
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


def _cmd_lint(args: argparse.Namespace) -> None:
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

    # Bound HERE, not inside `if args.fix:`, because `--semantic` is reachable without
    # `--fix` and consumes it below. Initialising it in the fix block made plain
    # `recall lint <path> --semantic` die with UnboundLocalError before doing any work.
    _validated_emb: Embedder | None = None

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
        if args.apply and args.semantic:
            # Resolve the embedder BEFORE writing. `--semantic` needs one, and dropping
            # argparse's `choices=` moved an unknown spelling's failure from "exit 2 before
            # anything happened" to "after apply_proposal has already rewritten the memos".
            # This is the only destructive path that resolved it late.
            _validated_emb = _make_embedder(args.embedder)
        if not args.apply:
            # Dry run by DEFAULT: this edits the user's own documents, and a tool that
            # rewrites your memory the first time you try it has earned distrust.
            print("dry run — nothing written. Re-run with --apply to write these edges.")
        else:
            root = Path(args.path)
            written = 0
            for p in proposals:
                try:
                    apply_proposal(root, p)
                except ValueError as exc:
                    # One memo the writer refuses must not discard the rest of the run. The
                    # loop previously had no guard, so a single undecodable file aborted with
                    # a traceback AFTER the earlier proposals had already been written — the
                    # worst of both, a partial apply the user has to reconstruct by hand.
                    #
                    # `ValueError`, not `UnreadableMemo`, because the writer has a second
                    # refusal now: `insert_frontmatter_line` rejects a value carrying a line
                    # break. `UnreadableMemo` subclasses `ValueError`, so this still catches
                    # everything it did, and the widening is what keeps that second refusal
                    # from reintroducing the exact partial apply this handler exists to stop.
                    # `propose_fixes` filters those values first, so reaching here means a
                    # caller built the `Proposal` itself.
                    #
                    # The memo is named explicitly rather than left to the exception's own
                    # text. `UnreadableMemo` opens with the path, but the widening admits
                    # exceptions raised further down that do not: a filename that is not
                    # valid UTF-8 arrives surrogate-escaped from `Path.glob`, passes the
                    # line-break check, and makes the writer's `.encode("utf-8")` raise
                    # `UnicodeEncodeError` — also a `ValueError`, and its message names a
                    # code point, not a file. A nameless SKIP in a list of many, followed by
                    # "skipped 1", leaves the operator of a destructive command unable to
                    # tell WHICH memo was passed over.
                    print(f"  SKIP {p.edit_file}: {exc}")
                    continue
                written += 1
            print(f"wrote {written} edge(s), skipped {len(proposals) - written}.")

    chains = []
    if args.semantic:  # opt-in retrieval-based missing-edge check (needs DB + embedder)
        from recall.semantic_lint import semantic_lint

        # Reuse the instance built for pre-write validation. Constructing twice is not
        # free: the cloud embedders probe the API inside __init__, so a second build is a
        # second billable request, and the local ones reload the model.
        emb = _validated_emb or _make_embedder(args.embedder)
        # --threshold's help promises "the calibrated abstention threshold for this
        # embedder" as the default; hardcoding 0.70 made that untrue on every corpus
        # whose calibration says otherwise.
        _cal = load_for(emb.name)
        if args.threshold is not None:
            thr, _src = args.threshold, "--threshold"
        elif _cal is not None:
            thr, _src = _cal.threshold, f"calibrated for {emb.name}"
        else:
            # `load_for` returns None WITHOUT raising when the artifact is keyed to a
            # different embedder, so this fallback is reachable even when a calibration
            # file exists — notably because the setup wizard keys it by the embedder
            # SPELLING while `recall calibrate` keys it by `embedder.name`. Saying which
            # threshold was used is the difference between a silent wrong answer and a
            # visible one; the help text promises the calibrated value.
            thr, _src = 0.70, f"UNCALIBRATED default, no calibration matched {emb.name}"
        print(f"semantic threshold: {thr:.2f} ({_src})")
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


def _cmd_check(args: argparse.Namespace) -> None:
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
