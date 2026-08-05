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


def _adapters(names: list[str]) -> list[CorpusAdapter]:
    unknown = [name for name in names if name not in ADAPTERS]
    if unknown:
        raise SystemExit(f"unknown corpus/corpora {unknown}; known: {sorted(ADAPTERS)}")
    return [ADAPTERS[name]() for name in names]


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


@contextlib.contextmanager
def _indexed_store(args: argparse.Namespace, adapter: CorpusAdapter) -> Iterator[tuple]:
    """A throwaway uuid-named store holding the corpus, dropped on the way out.

    Setup runs INSIDE the guard, matching `recall/eval/harness.py::_throwaway_store`: a failure
    while creating the schema or indexing must still drop the table and close the connection.
    """
    from recall.index import Indexer
    from recall.store import PgVectorStore
    from recall_mcp.service import make_embedder

    # `recall_mcp.service.make_embedder` rather than a local construction: it is the one site that
    # resolves an embedder through `recall.embedding_registry`, so an arm's profile identity comes
    # from the registry instead of from a second vocabulary maintained here.
    embedder = make_embedder(args.embedder)
    store = PgVectorStore(args.dsn, dim=embedder.dim, table=f"promo_{uuid.uuid4().hex[:8]}")
    try:
        store.ensure_schema()
        corpus_dir = args.corpus_dir or getattr(adapter, "corpus_dir", None)
        if corpus_dir is None:
            raise SystemExit(
                f"{adapter.name}: no corpus directory to index. Pass --corpus-dir pointing at the "
                f"documents this corpus's labels refer to."
            )
        Indexer(store, embedder).index_path(Path(corpus_dir))
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
    scoped = [question for question in frozen if question.corpus == adapter.name]
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
    scoped = [
        question
        for question in frozen
        if question.corpus in {record.corpus for record in baseline}
    ]
    decision, document = run_gate(
        baseline,
        candidate,
        scoped or frozen,
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
    run.add_argument("--dsn", required=True)
    run.add_argument("--corpus-dir", type=Path, default=None)
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
