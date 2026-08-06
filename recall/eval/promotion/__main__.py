"""`python -m recall.eval.promotion` — freeze, run an arm, decide.

Three subcommands, in the order they must happen:

``freeze``  write the frozen input manifest. Refuses to overwrite one, because the whole value of
            a frozen manifest is that it was fixed before any candidate existed and an
            overwrite-in-place erases the evidence that it was.
``run``     score one arm into a resumable ledger.
``decide``  build the gate input from two arms and write the promotion decision.

Latency: ``decide`` emits PENDING unless ``--certified-latency-p95-ms`` is passed, and PENDING
fails the gate. The flag exists so that supplying a latency is a deliberate act by someone who has
a reference host, and its help text says what certifying it means. This program has no such host
(see the standing blockers in `docs/ENTERPRISE_PROGRAM_STATUS.md`), so every decision it produces
today is PENDING on latency, by design and not by omission.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

from recall.eval.promotion.adapters import ADAPTERS, CorpusAdapter, CorpusUnavailable, label_kind
from recall.eval.promotion.aggregate import decide as run_gate
from recall.eval.promotion.aggregate import write_decision
from recall.eval.promotion.manifest import FrozenQuestion, read_manifest, write_manifest
from recall.eval.promotion.records import record_from_dict
from recall.eval.promotion.run import LEDGER_ID_FIELDS, ArmConfig, score_arm
from recall.eval.resume import read_ledger
from recall.lint import DEFAULT_GLOB


def _adapters(names: list[str]) -> list[CorpusAdapter]:
    unknown = [name for name in names if name not in ADAPTERS]
    if unknown:
        raise SystemExit(f"unknown corpus/corpora {unknown}; known: {sorted(ADAPTERS)}")
    return [ADAPTERS[name]() for name in names]


def scope_questions(
    frozen: "list[FrozenQuestion]", corpus: str
) -> "list[FrozenQuestion]":
    """The frozen questions one adapter owns.

    A named function rather than an inline comprehension so a test can exercise THIS code instead
    of restating it. The first test written for this defect re-implemented the expression against
    the adapter, which meant it passed with the fix reverted — a guard that could not fire, in the
    fix for a bug about a filter that matched nothing.

    `startswith` as well as equality: `MtragAdapter` stamps `mtrag:<domain>` so the gate
    stratifies per domain, and an equality-only filter made `--corpus mtrag` match nothing at all.
    The colon matters — without it, a future `labelled_v2` corpus would be swept into `labelled`.
    """
    prefix = f"{corpus}:"
    return [
        question
        for question in frozen
        if question.corpus == corpus or question.corpus.startswith(prefix)
    ]


def cmd_freeze(args: argparse.Namespace) -> int:
    if args.out.exists() and not args.force:
        raise SystemExit(
            f"{args.out} already exists. A frozen manifest is evidence that the question set was "
            f"fixed before any candidate result existed; overwriting it in place destroys exactly "
            f"that. Write a new path, or pass --force if you are re-freezing deliberately."
        )
    questions: list[FrozenQuestion] = []
    corpus_hashes: dict[str, str] = {}
    for adapter in _adapters(args.corpus):
        reason = adapter.unavailable_reason()
        if reason is not None:
            raise SystemExit(
                f"{reason}\nRefusing to freeze a manifest missing a corpus that was asked for. "
                f"Drop it from --corpus if it is genuinely not part of this experiment."
            )
        questions.extend(adapter.freeze())
        corpus_hashes.update(adapter.corpus_hashes())
    digest = write_manifest(args.out, questions, corpus_hashes=corpus_hashes)
    answerable = len([question for question in questions if question.answerable])
    print(
        f"froze {len(questions)} questions ({answerable} answerable, "
        f"{len(questions) - answerable} unanswerable) into {args.out}\ndigest {digest}"
    )
    return 0


#: A literal file extension, ANCHORED at the end of the glob's final component.
#: The first version of this guard tested for a "." anywhere in that component plus a five-member
#: denylist. The CCA bug auditor measured both halves and found the shape open and the denylist
#: dead. `**/[.]*` (dotfiles EXCLUSIVELY), `**/*.?*` (the semantic twin of `**/*.*`, which the
#: denylist blocked by name), `**/*.[a-z]*` and `**/*.[!x]*` all passed and all pulled `.env`,
#: `.npmrc` and `tokens.json` out of a seeded tree. An exhaustive fuzz over 8.1M strings then
#: showed the denylist changed the verdict on ZERO of them: every member was already caught by the
#: shape rules, so the part carrying the provenance comment was the part deciding nothing.
#: Naming five strings does not close a shape.
_EXTENSION = re.compile(r"\.[A-Za-z0-9_+-]+$")


def _refuse_overbroad_glob(glob: str) -> None:
    """Refuse a pattern that would index whatever happens to be in the directory.

    The empty-index refusal in `_indexed_store` is ONE-SIDED: it fires when the glob is too NARROW
    (zero chunks) and cannot fire when it is too BROAD, because that yields chunks > 0. A corpus
    directory that also holds a `.env` would then be embedded into the index, returned verbatim by
    search, and written into the evidence artifact.

    The final component must END in a literal extension, and may not carry a character class, a
    `?`, or a leading dot: each of those reconstructs a wildcard the anchor is there to forbid.
    Validated as a strict tightening rather than asserted — every pattern this repository uses
    (`**/*.md`, `**/*.rst`, `**/*.py`, `*.txt`, `corpus/**/*.rst`, `**/*.tar.gz`) is still
    accepted, and every measured bypass is now refused.

    ⚠️ Scope, stated because the first version's message claimed more than it checked: an
    extension-named glob still admits a file WITH that extension. `**/*.json` matches
    `tokens.json`. This bounds the pattern's SHAPE; it is not a secrets filter.
    """
    final = glob.rsplit("/", 1)[-1]
    # The anchor alone. A first draft also tested `startswith(".")`, `"[" in final` and
    # `"?" in final`; a mutation sweep showed BOTH survive deletion, because a final
    # component carrying a class or a `?` cannot end in `\.[A-Za-z0-9_+-]+` anyway. Keeping
    # them would have repeated the defect the auditor had just found in `_OVERBROAD_GLOBS`:
    # a clause that reads as the boundary and decides nothing.
    if not _EXTENSION.search(final):
        raise SystemExit(
            f"--glob {glob!r} does not end in a literal file extension, so it would index "
            f"whatever the directory happens to contain rather than a named corpus. Name an "
            f"extension, e.g. '**/*.rst'. (This bounds the pattern's shape; a glob naming an "
            f"extension still matches every file carrying it.)"
        )


@contextlib.contextmanager
def _indexed_store(args: argparse.Namespace, adapter: CorpusAdapter) -> Iterator[tuple]:
    """A throwaway uuid-named store holding the corpus, dropped on the way out.

    Setup runs INSIDE the guard, matching `recall/eval/harness.py::_throwaway_store`: a failure
    while creating the schema or indexing must still drop the table and close the connection.
    """
    from recall.context import context_policy_for_profile
    from recall.embeddings import embedding_profile_id
    from recall.index import Indexer
    from recall.store import PgVectorStore
    from recall_mcp.service import make_embedder

    # `recall_mcp.service.make_embedder` rather than a local construction: it is the one site that
    # resolves an embedder through `recall.embedding_registry`, so an arm's profile identity comes
    # from the registry instead of from a second vocabulary maintained here.
    # Before the model load and before any database work: this is a pure predicate over argv,
    # and the DSN refusal twenty lines away already states that convention. Measured order was
    # make_embedder -> connect -> CREATE TABLE -> refuse, which touched the operator's database
    # for a request that was always going to be refused.
    _refuse_overbroad_glob(args.glob)
    embedder = make_embedder(args.embedder)
    # The context policy has to come from the arm's own profile. `Indexer` defaults to
    # `ContextPolicy()`, which is mode "none" / `raw-v1`, and it REFUSES an embedder whose profile
    # declares anything else. So without this every context profile was unindexable through this
    # harness: measured, all three failed with "embedding profile context 'context-document-v1'
    # does not match index context 'raw-v1'" before a single question was scored. A campaign
    # comparing context modes could not run at all, and the failure was at index time rather than
    # in the numbers, which is the only reason it was not a silent wrong result.
    context_policy = context_policy_for_profile(embedding_profile_id(embedder))
    store = PgVectorStore(args.dsn, dim=embedder.dim, table=f"promo_{uuid.uuid4().hex[:8]}")
    try:
        store.ensure_schema()
        corpus_dir = args.corpus_dir or getattr(adapter, "corpus_dir", None)
        if corpus_dir is None:
            raise SystemExit(
                f"{adapter.name}: no corpus directory to index. Pass --corpus-dir pointing at the "
                f"documents this corpus's labels refer to."
            )
        # `index_path` defaults to `**/*.md`. The PEPs corpus is `.rst`, so the default indexed
        # ZERO files and every one of 110 questions came back abstained with an empty pool. The
        # gate's `VacuousArm` guard did catch it, but only after four arms had each paid for a
        # full embedding pass, and its message blames the label space, which was not the fault.
        stats = Indexer(store, embedder, context_policy=context_policy).index_path(
            Path(corpus_dir), args.glob
        )
        # An empty index cannot produce a measurement, and 110 abstentions is not a result. Refuse
        # where the cause is still legible rather than three steps downstream.
        if not stats.chunks:
            raise SystemExit(
                f"{adapter.name}: indexing {corpus_dir} with glob {args.glob!r} produced no "
                f"chunks. Every question would abstain against an empty index and the gate would "
                f"refuse the arm as vacuous, blaming the label space. Check --glob against the "
                f"corpus file extension."
            )
        yield store, embedder
    finally:
        try:
            store.drop_table()
        except Exception:
            pass
        finally:
            store.close()


def cmd_run(args: argparse.Namespace) -> int:
    from recall.embeddings import embedding_profile
    from recall.eval.promotion.search import StoreSearch

    # Refused HERE, before the embedder is built. Moving the DSN out of argv removed the only
    # thing asserting one exists, and `PgVectorStore` does not validate it: `None` reaches psycopg
    # as an AttributeError after a model load, and an EMPTY string — the ordinary shape of an
    # unset systemd or CI variable — is a valid libpq conninfo that resolves to the LOCAL
    # DEFAULTS. On a host with peer or trust auth that connects to the operator's own database,
    # where this command then creates, indexes and drops a table.
    if not args.dsn:
        raise SystemExit(
            "no DSN. Pass --dsn, or set $RECALL_SERVING_DSN or $RECALL_DSN. An empty value is "
            "refused rather than passed through: libpq reads an empty conninfo as 'use the local "
            "defaults', which would index this corpus into whatever database happens to be there."
        )
    frozen, header = read_manifest(args.manifest)
    if args.expected_digest and header.get("digest") != args.expected_digest:
        raise SystemExit(
            f"{args.manifest} has digest {header.get('digest')}, not the expected "
            f"{args.expected_digest}."
        )
    (adapter,) = _adapters([args.corpus])
    reason = adapter.unavailable_reason()
    if reason is not None:
        raise SystemExit(reason)
    scoped = scope_questions(frozen, adapter.name)
    if not scoped:
        raise SystemExit(
            f"the manifest holds no question for corpus {adapter.name!r}; it covers "
            f"{sorted({question.corpus for question in frozen})}."
        )

    with _indexed_store(args, adapter) as (store, embedder):
        profile = embedding_profile(embedder)
        arm = ArmConfig(
            label=args.arm,
            embedding_profile_id=profile.profile_id,
            retrieval_profile=args.retrieval_profile,
            generation=args.generation,
            candidate_pool=args.candidate_k,
            embedding_fingerprint=profile.fingerprint(),
        )
        from recall.trust_policy import TrustPolicy

        policy = (
            TrustPolicy.development()
            if args.trust_policy == "development"
            else TrustPolicy.strict_policy()
        )
        if args.trust_policy == "development":
            print(
                "trust policy: DEVELOPMENT — the trust gate does not run, every hit carries the "
                "'unverified' verdict, and abstention is degraded. This arm's safety metrics "
                "describe a degraded system; they are not a measurement of the trust layer."
            )
        search = StoreSearch(
            store=store,
            embedder=embedder,
            k=args.k,
            candidate_k=args.candidate_k,
            retrieval_profile=args.retrieval_profile,
            generation=args.generation,
            label_kind=label_kind(adapter),
            policy=policy,
        )
        out = args.out or args.out_dir / f"{arm.artifact_stem()}.jsonl"
        result = score_arm(
            scoped, adapter.queries(), search, arm, out, resume=not args.no_resume
        )
    if result.repaired_truncated_tail:
        print(f"note: {out} had a truncated trailing line from an earlier run; it was repaired.")
    print(
        f"arm {args.arm}: {result.scored_now} scored now, {result.resumed} resumed, "
        f"{len(result.records)} total -> {out}"
    )
    return 0


def _load_records(path: Path) -> list:
    read = read_ledger(path, id_field=LEDGER_ID_FIELDS)
    if not read.rows:
        raise SystemExit(f"{path} holds no records")
    if read.truncated:
        raise SystemExit(
            f"{path} ends in a truncated line, so the arm is incomplete. Re-run it to completion "
            f"before deciding; a gate over a partial arm is a gate over a different experiment."
        )
    return [record_from_dict(row) for row in read.rows]


def cmd_decide(args: argparse.Namespace) -> int:
    frozen, header = read_manifest(args.manifest)
    baseline = _load_records(args.baseline)
    candidate = _load_records(args.candidate)
    # The whole frozen manifest, never a subset derived from the evidence files. An earlier
    # version narrowed `frozen` to the corpora present in the BASELINE ledger, which defeated the
    # first refusal this harness declares: a corpus missing from both arms was silently dropped,
    # `UnpairedArms` never saw it, and the decision was stamped with the FULL manifest digest —
    # certifying a question set that was never gated. Measured on a two-corpus manifest: with both
    # corpora present the gate returned `promoted=false` citing a regression on the second; with
    # that corpus absent from both ledgers the same manifest returned `promoted=true`.
    #
    # Restricting a campaign to fewer corpora is a legitimate thing to want. It is done by
    # freezing a manifest for those corpora, which produces its own digest, not by handing the
    # gate a manifest whose questions were quietly discarded.
    decision, document = run_gate(
        baseline,
        candidate,
        frozen,
        manifest_digest=header["digest"],
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        security_green=args.security_green,
        latency_budget_ms=args.latency_budget_ms,
        certified_latency_p95_ms=args.certified_latency_p95_ms,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_decision(args.out, document)
    print(json.dumps({"promoted": decision.promoted, "failures": list(decision.failures)}, indent=2))
    print(f"decision written to {args.out}")
    # Exit 0 whether or not it promoted: the gate ran and answered. A non-zero exit for "did not
    # promote" would make a refusal indistinguishable from a harness that crashed, and this
    # program's expected answer today is a refusal.
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recall.eval.promotion", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="write a frozen input manifest")
    freeze.add_argument("--corpus", action="append", required=True, choices=sorted(ADAPTERS))
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--force", action="store_true", help="re-freeze over an existing manifest")
    freeze.set_defaults(func=cmd_freeze)

    run = sub.add_parser("run", help="score one arm into a resumable ledger")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--expected-digest", default=None)
    run.add_argument("--corpus", required=True, choices=sorted(ADAPTERS))
    run.add_argument("--arm", required=True, help="this arm's label, e.g. baseline or candidate")
    run.add_argument(
        "--dsn",
        default=os.environ.get("RECALL_SERVING_DSN") or os.environ.get("RECALL_DSN"),
        help=(
            "defaults to $RECALL_SERVING_DSN or $RECALL_DSN. Prefer the environment: a DSN on "
            "the command line puts the database password into argv, where any local user can "
            "read it for the whole run, and into shell history and CI logs."
        ),
    )
    run.add_argument("--corpus-dir", type=Path, default=None)
    run.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help=(
            "file pattern to index, passed to Indexer.index_path. Defaults to markdown; "
            "a corpus of another extension (the PEPs are .rst) indexes ZERO files under "
            "the default and every question then abstains against an empty index."
        ),
    )
    run.add_argument(
        "--embedder",
        default="fastembed",
        choices=("fastembed", "hashing"),
        help=(
            "resolved through recall_mcp.service.make_embedder, so RECALL_EMBED_PROFILE selects "
            "a registered profile and the arm's identity comes from the registry."
        ),
    )
    run.add_argument("--out", type=Path, default=None)
    run.add_argument("--out-dir", type=Path, default=Path("results/promotion"))
    run.add_argument("--k", type=int, default=20)
    run.add_argument("--candidate-k", type=int, default=20)
    run.add_argument("--retrieval-profile", default="legacy")
    run.add_argument("--generation", default="legacy")
    run.add_argument(
        "--trust-policy",
        default="strict",
        choices=("strict", "development"),
        help=(
            "STRICT by default, so a harness that forgets to choose cannot score a degraded "
            "system and publish the numbers as if the trust gate had run. A plain store with no "
            "generation-bound certified calibration REFUSES under strict, which is the correct "
            "answer; 'development' records 'unverified' verdicts and degraded abstention."
        ),
    )
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=cmd_run)

    decide = sub.add_parser("decide", help="run the gate over two arms")
    decide.add_argument("--manifest", type=Path, required=True)
    decide.add_argument("--baseline", type=Path, required=True)
    decide.add_argument("--candidate", type=Path, required=True)
    decide.add_argument("--baseline-label", default="baseline")
    decide.add_argument("--candidate-label", default="candidate")
    decide.add_argument("--out", type=Path, required=True)
    decide.add_argument("--latency-budget-ms", type=float, default=250.0)
    decide.add_argument(
        "--certified-latency-p95-ms",
        type=float,
        default=None,
        help=(
            "p95 measured on an IDLE REFERENCE HOST. Omitted by default, which records latency as "
            "PENDING and FAILS the gate on latency grounds. Pass this only when the number came "
            "from a host that can carry the claim; a p95 from a loaded machine measures that load."
        ),
    )
    decide.add_argument(
        "--security-green",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether security verification passed. Defaults to False: unverified is not green.",
    )
    decide.add_argument("--bootstrap-samples", type=int, default=10_000)
    decide.set_defaults(func=cmd_decide)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CorpusUnavailable as exc:
        print(f"corpus unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
