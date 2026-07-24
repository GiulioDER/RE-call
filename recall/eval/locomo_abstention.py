"""Does calibration or the entailment judge let RE-call abstain on LOCOMO's adversarials?

The main runner (`recall.eval.locomo`) measured the *default* configuration: 0.00 abstention on
446 adversarial questions. This asks whether the two mechanisms the library ships to raise that —
threshold calibration and the entailment judge — actually do, and at what cost to answerable
questions. A mode that abstains on everything scores 1.00 on adversarials and is useless, so both
sides are measured together:

- **adversarial abstention rate** (category 5, unanswerable) — want HIGH
- **answerable false-abstain rate** (categories 1-4) — want LOW

Four modes:
  default      no calibration, no judge — reproduces the main runner's 0.00
  calibrated   per-conversation threshold, fit IN-SAMPLE on that conversation's own answerable and
               adversarial top-cosines. This is deliberately the most generous possible calibration:
               it sees the exact distributions it is later scored on. If it cannot separate them
               even so, no honestly-fit calibration can — the point being tested is that the two
               distributions overlap because adversarials are on-topic, not that a threshold was
               chosen badly.
  entail       default threshold + QNLI cross-encoder ("does this passage answer the question?").
               The lever built for wrong-attribution near-misses.
  both         calibrated + entail.

Reuses the tables `recall.eval.locomo` already indexed (table `locomo_chunks`, one tenant per
conversation) — no re-index. Run the main runner first if the tables are absent.

::

    python -m recall.eval.locomo_abstention --data locomo10.json \\
        --answerable-sample 40 --out results/locomo_abstention.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from recall.calibration import Calibration, from_samples
from recall.embeddings import Embedder
from recall.eval.locomo import (
    ADVERSARIAL_CATEGORY,
    ANSWERABLE_CATEGORIES,
    _make_embedder,
    _rate,
)
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.trust import trusted_search

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")

MODES = ("default", "calibrated", "entail", "both")


def _top_cosine(retriever: HybridRetriever, query: str, k: int) -> float | None:
    result = retriever.search(query, k=k)
    return result.hits[0].score if result.hits else None


def _partition_questions(
    qa: list[dict[str, Any]], answerable_sample: int, rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one conversation's questions into (answerable-with-evidence, adversarial).

    Answerable questions are sampled to bound the cost of the entailment pass; adversarials are
    taken whole, because they are the smaller class and the one the whole experiment is about.
    """
    answerable = [
        q
        for q in qa
        if q.get("category") in ANSWERABLE_CATEGORIES
        and q.get("question")
        and (q.get("evidence") or [])
    ]
    adversarial = [
        q for q in qa if q.get("category") == ADVERSARIAL_CATEGORY and q.get("question")
    ]
    if 0 < answerable_sample < len(answerable):
        answerable = rng.sample(answerable, answerable_sample)
    return answerable, adversarial


def _fit_calibration(
    retriever: HybridRetriever,
    embedder_name: str,
    answerable: list[dict[str, Any]],
    adversarial: list[dict[str, Any]],
    k: int,
) -> Calibration:
    """In-sample calibration: fit the threshold on the very questions it will be scored on.

    This is the calibration UPPER BOUND, not a realistic operating point — a deployed calibration
    is fit on a held-out labelled set. Fitting in-sample here means any failure to separate is a
    property of the data (overlapping distributions), not of the fit.
    """
    ans_cos = [c for q in answerable if (c := _top_cosine(retriever, q["question"], k)) is not None]
    adv_cos = [
        c for q in adversarial if (c := _top_cosine(retriever, q["question"], k)) is not None
    ]
    return from_samples(embedder_name, ans_cos, adv_cos)


def run(
    data_path: Path,
    *,
    dsn: str,
    embedder_name: str,
    k: int,
    answerable_sample: int,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    conversations = json.loads(data_path.read_text(encoding="utf-8"))
    if limit is not None:
        conversations = conversations[:limit]
    rng = random.Random(seed)

    embedder: Embedder = _make_embedder(embedder_name)

    # The judge loads a cross-encoder once and is reused across every question and conversation.
    from recall.entailment import QnliEntailmentJudge

    print("loading entailment judge (first run downloads the cross-encoder)...", flush=True)
    judge = QnliEntailmentJudge()

    # abstained flags, pooled across conversations: [mode] -> list[bool]
    adv_abstain: dict[str, list[bool]] = {m: [] for m in MODES}
    ans_abstain: dict[str, list[bool]] = {m: [] for m in MODES}
    thresholds: list[float] = []
    started = time.time()

    for i, conv in enumerate(conversations):
        sample_id = conv.get("sample_id") or f"conv{i}"
        tenant = f"locomo-{sample_id}"
        qa = conv.get("qa") or []
        answerable, adversarial = _partition_questions(qa, answerable_sample, rng)

        with PgVectorStore(dsn, dim=embedder.dim, tenant=tenant, table="locomo_chunks") as store:
            retriever = HybridRetriever(store, embedder)
            cal = _fit_calibration(retriever, embedder_name, answerable, adversarial, k)
            thresholds.append(cal.threshold)

            def _score(q: dict[str, Any], bucket: dict[str, list[bool]]) -> None:
                question = q["question"]
                # default / calibrated share the no-judge path; entail / both add the judge.
                bucket["default"].append(
                    trusted_search(store, embedder, question, k=k).abstained
                )
                bucket["calibrated"].append(
                    trusted_search(store, embedder, question, k=k, calibration=cal).abstained
                )
                bucket["entail"].append(
                    trusted_search(store, embedder, question, k=k, entailment=judge).abstained
                )
                bucket["both"].append(
                    trusted_search(
                        store, embedder, question, k=k, calibration=cal, entailment=judge
                    ).abstained
                )

            for q in adversarial:
                _score(q, adv_abstain)
            for q in answerable:
                _score(q, ans_abstain)

        print(
            f"  [{i + 1}/{len(conversations)}] {sample_id}: "
            f"threshold={cal.threshold:.3f}  adv={len(adversarial)}  ans={len(answerable)}",
            flush=True,
        )

    return {
        "benchmark": "LOCOMO — abstention ablation",
        "embedder": embedder_name,
        "k": k,
        "answerable_sample_per_conv": answerable_sample,
        "seed": seed,
        "conversations": len(conversations),
        "elapsed_s": round(time.time() - started, 1),
        "calibrated_threshold": {
            "min": round(min(thresholds), 3),
            "max": round(max(thresholds), 3),
            "mean": round(sum(thresholds) / len(thresholds), 3),
        },
        "modes": {
            m: {
                "adversarial_abstention": _rate(adv_abstain[m]),
                "answerable_false_abstain": _rate(ans_abstain[m]),
            }
            for m in MODES
        },
    }


def _print_report(r: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"LOCOMO abstention ablation · {r['conversations']} conversations · {r['embedder']}")
    print(f"calibrated threshold across conversations: {r['calibrated_threshold']}")
    print("=" * 78)
    print()
    print(f"{'mode':<12} {'adversarial abstain':<26} {'answerable false-abstain':<26}")
    print(f"{'':12} {'(want HIGH)':<26} {'(want LOW)':<26}")
    print("-" * 78)
    for m in MODES:
        adv = r["modes"][m]["adversarial_abstention"]
        ans = r["modes"][m]["answerable_false_abstain"]
        adv_s = f"{adv['rate']:.3f} [{adv['ci95'][0]:.2f},{adv['ci95'][1]:.2f}] n={adv['n']}"
        ans_s = f"{ans['rate']:.3f} [{ans['ci95'][0]:.2f},{ans['ci95'][1]:.2f}] n={ans['n']}"
        print(f"{m:<12} {adv_s:<26} {ans_s:<26}")
    print()
    print("A mode is only useful if adversarial abstention rises WITHOUT answerable false-abstain")
    print("rising to match. Calibration fit in-sample is calibration's best case, not a real one.")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m recall.eval.locomo_abstention")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--embedder", default="fastembed")
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--answerable-sample",
        type=int,
        default=40,
        help="answerable questions per conversation to score for false-abstain (0 = all)",
    )
    p.add_argument("--conversations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.data.exists():
        p.error(f"{args.data} not found — fetch locomo10.json (see recall.eval.locomo)")

    report = run(
        args.data,
        dsn=args.dsn,
        embedder_name=args.embedder,
        k=args.k,
        answerable_sample=args.answerable_sample,
        limit=args.conversations,
        seed=args.seed,
    )
    _print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"full report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
