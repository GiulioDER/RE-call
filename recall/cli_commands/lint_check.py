"""`recall lint` and `recall check`: corpus supersession-graph checks."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from recall.calibration import load_for, resolved_path
from recall.embeddings import Embedder
from recall.lint import DEFAULT_GLOB

from recall.cli_commands._shared import _make_embedder

#: What a bare `--apply` parses to. It used to mean "write every proposal", which put each edge the
#: dry run listed into the corpus with nobody choosing it; it is refused now, with the new form.
_NO_SELECTION = ""


def _shown(text: str) -> str:
    """`text` for the terminal, with every non-printable character escaped.

    A proposal's target is copied from memo prose, and the reviewer picks ids from these lines:
    an escape sequence in a wikilink could rewrite the line beside an id, so the line shown would
    not be the edge the id writes.
    """
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in text)


def _selected_ids(raw: str) -> list[str]:
    """The proposal ids `--apply` names, in order, each once.

    An id may keep the brackets the dry run prints it in, so ``[3f2a91c0d4]`` copied from that
    line selects the same edge.
    """
    ids: list[str] = []
    for item in raw.split(","):
        item = item.strip().strip("[]").strip().lower()
        if item and item not in ids:
            ids.append(item)
    return ids


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
    # `append`, so a repeated `--apply` arrives as two values and is refused, rather than the last
    # one silently replacing the first.
    p_lint.add_argument(
        "--apply",
        nargs="?",
        action="append",
        const=_NO_SELECTION,
        default=None,
        metavar="IDS",
        help="with --fix, write the edges you reviewed: the comma-separated ids the dry run "
        "prints, e.g. --apply 3f2a91c0d4,77b1e0a9c2, given once. Every other proposal is left "
        "unwritten.",
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

    if args.apply is not None and not args.fix:
        print("recall lint: --apply writes the edges --fix proposes; add --fix", file=sys.stderr)
        raise SystemExit(2)
    if args.apply is not None and len(args.apply) > 1:
        # A repeated `--apply` kept only its last value, so `--apply a --apply b` wrote b and
        # said nothing about a.
        print(
            "recall lint: give --apply once, with the ids comma-separated. Nothing was written.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        issues = lint_corpus(args.path, glob=args.glob)
    except FileNotFoundError as exc:
        print(f"recall lint: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    for i in issues:
        print(f"{i.level:<8} {i.code:<26} {i.file}: {i.message}")
    errors = sum(1 for i in issues if i.level == "error")
    warnings = len(issues) - errors

    # Bound on both branches, because `--semantic` is reachable without `--fix` and consumes
    # it below. Bound only inside the fix block, plain `recall lint <path> --semantic` died
    # with UnboundLocalError before doing any work.
    validated_emb = _run_fix(args) if args.fix else None
    if args.semantic:  # opt-in retrieval-based missing-edge check (needs DB + embedder)
        warnings += _run_semantic(args, validated_emb)

    print(f"{errors} errors, {warnings} warnings")
    if errors:
        raise SystemExit(1)


def _run_fix(args: argparse.Namespace) -> Embedder | None:
    """`--fix`: print each proposal and write the ones `--apply` names.

    Returns the embedder `--semantic` was resolved with before anything was written, or None.
    """
    from recall.fix import Proposal, apply_proposal, propose_fixes

    validated_emb: Embedder | None = None
    proposals, unfixable = propose_fixes(args.path, glob=args.glob)
    print()
    for p in proposals:
        print(f"  [{p.id}] {_shown(p.edit_file)}: + supersedes: {_shown(p.target)}")
        print(f"      because {_shown(p.evidence_file)} says {p.evidence!r}")
    for u in unfixable:
        print(f"  SKIP {_shown(u.file)}: {_shown(u.reason)}")
    print(f"\n{len(proposals)} edge(s) proposable, {len(unfixable)} need a human")
    selected: list[Proposal] = []
    if args.apply is not None:
        # Every id is checked before anything is written, so a typo or an id from an older
        # dry run (the corpus moved, the sentence changed) writes nothing rather than part
        # of a selection. An id hashes the edge and the sentence the reviewer read, so it
        # cannot come to mean a different edge.
        wanted = _selected_ids(args.apply[0])
        if not wanted:
            print(
                "recall lint: --apply needs the ids of the edges you reviewed, e.g. "
                "--apply <id>,<id> with ids from the list above. Nothing was written.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        by_id = {p.id: p for p in proposals}
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            print(
                f"recall lint: no proposal {', '.join(unknown)} in {args.path} now. Re-run "
                f"without --apply and choose from the list it prints. Nothing was written.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        # The id is a 40-bit prefix of a hash, so two proposals can share one, and whoever
        # writes memo prose can choose a sentence that does. Such an id names no one edge.
        shared = Counter(p.id for p in proposals)
        ambiguous = [i for i in wanted if shared[i] > 1]
        if ambiguous:
            print(
                f"recall lint: more than one proposal has the id {', '.join(ambiguous)}, so it "
                f"names no one edge. Nothing was written.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        selected = [by_id[i] for i in wanted]
    if selected and args.semantic:
        # Resolve the embedder BEFORE writing. `--semantic` needs one, and dropping
        # argparse's `choices=` moved an unknown spelling's failure from "exit 2 before
        # anything happened" to "after apply_proposal has already rewritten the memos".
        # This is the only destructive path that resolved it late.
        validated_emb = _make_embedder(args.embedder)
    if args.apply is None:
        # Dry run by DEFAULT: this edits the user's own documents, and a tool that
        # rewrites your memory the first time you try it has earned distrust.
        print(
            "dry run — nothing written. Review each edge, then write the ones you accept "
            "with --apply <id>[,<id>...]."
        )
    else:
        root = Path(args.path)
        written = 0
        for p in selected:
            try:
                apply_proposal(root, p, glob=args.glob)
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
                print(f"  SKIP {_shown(p.edit_file)}: {_shown(str(exc))}")
                continue
            written += 1
        print(
            f"wrote {written} edge(s), skipped {len(selected) - written}; "
            f"{len(proposals) - len(selected)} not selected."
        )
    return validated_emb


def _run_semantic(args: argparse.Namespace, validated_emb: Embedder | None) -> int:
    """`--semantic`: print each unlinked chain and return how many there are."""
    from recall.semantic_lint import semantic_lint

    # Reuse the instance built for pre-write validation. Constructing twice is not
    # free: the cloud embedders probe the API inside __init__, so a second build is a
    # second billable request, and the local ones reload the model.
    emb = validated_emb or _make_embedder(args.embedder)
    # --threshold's help promises "the calibrated abstention threshold for this
    # embedder" as the default; hardcoding 0.70 made that untrue on every corpus
    # whose calibration says otherwise.
    _cal = load_for(emb.name)
    # Named because, with RECALL_CALIBRATION unset, the file is relative to the working directory.
    _where = resolved_path()
    if args.threshold is not None:
        thr, _src = args.threshold, "--threshold"
    elif _cal is not None:
        thr, _src = _cal.threshold, f"calibrated for {emb.name} in {_where}"
    else:
        # `load_for` returns None WITHOUT raising when the artifact is keyed to a
        # different embedder, so this fallback is reachable even when a calibration
        # file exists — notably because the setup wizard keys it by the embedder
        # SPELLING while `recall calibrate` keys it by `embedder.name`. Saying which
        # threshold was used is the difference between a silent wrong answer and a
        # visible one; the help text promises the calibrated value.
        thr, _src = 0.70, f"UNCALIBRATED default, no calibration matched {emb.name} in {_where}"
    print(f"semantic threshold: {thr:.2f} ({_src})")
    chains = semantic_lint(args.dsn, emb, args.path, threshold=thr, glob=args.glob)
    for c in chains:
        print(
            f"warning  unlinked-chain             {c.new_memo}: highly similar "
            f"(cos={c.cosine:.2f}) to closed decision {c.prior!r} it does not "
            f"reference — add `supersedes: {c.prior}`?"
        )
    return len(chains)


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
