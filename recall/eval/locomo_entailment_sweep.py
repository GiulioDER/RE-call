"""Does a STRONGER entailment judge shift the LOCOMO abstention tradeoff, or just move along it?

FINDINGS §9 measured the default judge (`cross-encoder/qnli-distilroberta-base`, threshold 0.5):
adversarial abstention 0.374 for answerable false-abstain 0.263. This asks whether a better judge
does better — and separates the two ways it could:

- **Moving along the same curve** — a threshold change trades adversarial abstention for
  false-abstain at a fixed judge quality. Cheap, and no judge is "stronger" for it.
- **Shifting the whole curve up** — a better judge catches more adversarials at the SAME
  false-abstain. That is what "a stronger judge helps" has to mean to be worth the cost.

The experiment separates them by measuring the full ROC. Each judge scores every candidate ONCE;
the threshold is then swept analytically (abstain iff every ok hit scores below it), so a whole
curve costs one model pass, not one pass per threshold. Three judges on one axis:

  qnli-distilroberta-base   the shipped default (baseline)
  qnli-electra-base         same QNLI task, stronger base
  nli-deberta-v3-large      the strongest available entailment model — but 3-way NLI, not QNLI, so
                            it is applied as (premise=passage, hypothesis=question) and read on the
                            entailment class. A different framing, flagged as such: if it wins it
                            wins on a task it was not built for, and that is worth knowing.

Only the `entail`-alone arm is studied (default threshold path + judge) — calibration is a separate
lever already mapped in §9b. Reuses the tables `recall.eval.locomo` indexed (`locomo_chunks`,
one tenant per conversation); run that first.

::

    python -m recall.eval.locomo_entailment_sweep --data locomo10.json \\
        --answerable-sample 40 --out results/locomo_entailment_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from recall.embeddings import Embedder
from recall.eval.locomo import _make_embedder
from recall.eval.locomo_abstention import _partition_questions
from recall.eval.metrics import wilson_ci
from recall.store import PgVectorStore
from recall.trust import trusted_search

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")

#: (label, model_id). Order is baseline first.
#
# `cross-encoder/nli-deberta-v3-large` — the strongest available entailment model — was tried and
# excluded. It is a 3-way NLI model, so it has to be applied as (premise=passage, hypothesis=query);
# but the query is a QUESTION, and an NLI model trained on declarative hypotheses scores the
# entailment class ≈0 for a question-hypothesis whether the passage answers it or not (measured:
# 0.000 for both the right and the wrong-speaker passage on the toy adversarial). Making it usable
# would need a question→statement rewrite, which is a generation step this library does not have.
# Recorded rather than silently dropped: the strongest model is not usable here without changing
# the task.
DEFAULT_JUDGES = (
    ("qnli-distilroberta (default)", "cross-encoder/qnli-distilroberta-base"),
    ("qnli-electra-base", "cross-encoder/qnli-electra-base"),
)

#: Thresholds swept per judge. All judges here emit a probability in [0, 1] (sigmoid for the
#: 1-label QNLI models, softmax-entailment for the 3-label NLI model), so one grid fits all.
THRESHOLD_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def _entailment_index(id2label: dict[int, str]) -> int:
    """Which output column is the entailment class, read from the model's own config.

    Never hardcode it: cross-encoder NLI checkpoints disagree on column order (some
    contradiction/entailment/neutral, some entailment/neutral/contradiction), and guessing wrong
    silently inverts the judge — it would look like it is scoring answers when it is scoring
    contradictions.
    """
    for idx, label in id2label.items():
        if "entail" in label.lower():
            return int(idx)
    raise ValueError(f"no entailment label found in {id2label!r}")


def build_scorer(model_id: str) -> Callable[[str, list[str]], list[float]]:
    """Return score(query, passages) -> probability-in-[0,1] per passage, higher = more entailing.

    Handles both judge shapes from the model config, so a QNLI and an NLI checkpoint are called
    through one interface and land on the same [0, 1] threshold grid.
    """
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_id)
    num_labels = ce.model.config.num_labels

    if num_labels == 1:
        # QNLI: predict applies sigmoid; the pair is (question, candidate sentence).
        def score_qnli(query: str, passages: list[str]) -> list[float]:
            if not passages:
                return []
            return [float(s) for s in ce.predict([(query, p) for p in passages])]

        return score_qnli

    ent_idx = _entailment_index(dict(ce.model.config.id2label))

    # NLI: (premise=passage, hypothesis=question); read the entailment-class probability.
    def score_nli(query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        probs = ce.predict([(p, query) for p in passages], apply_softmax=True)
        return [float(row[ent_idx]) for row in probs]

    return score_nli


@dataclass
class QuestionScores:
    """One question's judge evidence, gathered once and reused across every threshold.

    `max_entail` is None when the question had no ok hit under the default path — it already
    abstains regardless of the judge, so it abstains at every threshold. Kept distinct from a
    real score of 0.0, which would only abstain above threshold 0.0.
    """

    is_adversarial: bool
    max_entail: float | None


def _abstains_at(q: QuestionScores, threshold: float) -> bool:
    # apply_entailment demotes an ok hit whose score < threshold; the result abstains when no ok
    # hit survives — i.e. when the BEST ok hit is still below threshold.
    if q.max_entail is None:
        return True
    return q.max_entail < threshold


def _rate(flags: list[bool]) -> dict[str, Any]:
    lo, hi = wilson_ci(flags)
    return {
        "n": len(flags),
        "rate": round(sum(flags) / len(flags), 4) if flags else float("nan"),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def gather_scores(
    judge_model: str,
    conversations: list[dict[str, Any]],
    *,
    dsn: str,
    embedder: Embedder,
    k: int,
    answerable_sample: int,
    seed: int,
) -> list[QuestionScores]:
    """Score every adversarial + sampled-answerable question once with one judge."""
    scorer = build_scorer(judge_model)
    rng = random.Random(seed)
    out: list[QuestionScores] = []

    for i, conv in enumerate(conversations):
        sample_id = conv.get("sample_id") or f"conv{i}"
        tenant = f"locomo-{sample_id}"
        answerable, adversarial = _partition_questions(conv.get("qa") or [], answerable_sample, rng)
        with PgVectorStore(dsn, dim=embedder.dim, tenant=tenant, table="locomo_chunks") as store:
            for q, is_adv in [(q, True) for q in adversarial] + [(q, False) for q in answerable]:
                # Default path (no calibration, no judge) to get the ok hits the judge would see.
                res = trusted_search(store, embedder, q["question"], k=k)
                ok_texts = [h.chunk.text for h in res.hits if h.verdict == "ok"]
                scores = scorer(q["question"], ok_texts)
                out.append(
                    QuestionScores(
                        is_adversarial=is_adv,
                        max_entail=max(scores) if scores else None,
                    )
                )
        print(f"    {sample_id}: adv={len(adversarial)} ans={len(answerable)}", flush=True)
    return out


def sweep(scores: list[QuestionScores]) -> list[dict[str, Any]]:
    """ROC points: for each threshold, adversarial abstention and answerable false-abstain."""
    adv = [s for s in scores if s.is_adversarial]
    ans = [s for s in scores if not s.is_adversarial]
    points = []
    for t in THRESHOLD_GRID:
        adv_flags = [_abstains_at(s, t) for s in adv]
        ans_flags = [_abstains_at(s, t) for s in ans]
        points.append(
            {
                "threshold": t,
                "adversarial_abstention": _rate(adv_flags),
                "answerable_false_abstain": _rate(ans_flags),
                # discrimination: how much MORE it abstains on adversarial than answerable.
                "separation": round(
                    _rate(adv_flags)["rate"] - _rate(ans_flags)["rate"], 4
                ),
            }
        )
    return points


def run(
    data_path: Path,
    *,
    dsn: str,
    embedder_name: str,
    k: int,
    answerable_sample: int,
    limit: int | None,
    seed: int,
    judges: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    conversations = json.loads(data_path.read_text(encoding="utf-8"))
    if limit is not None:
        conversations = conversations[:limit]
    embedder = _make_embedder(embedder_name)
    started = time.time()

    per_judge: dict[str, Any] = {}
    for label, model_id in judges:
        print(f"[judge] {label} ({model_id})", flush=True)
        scores = gather_scores(
            model_id,
            conversations,
            dsn=dsn,
            embedder=embedder,
            k=k,
            answerable_sample=answerable_sample,
            seed=seed,
        )
        points = sweep(scores)
        best = max(points, key=lambda p: p["separation"])
        per_judge[label] = {
            "model": model_id,
            "roc": points,
            "best_separation_point": best,
        }

    return {
        "benchmark": "LOCOMO — entailment judge sweep",
        "embedder": embedder_name,
        "k": k,
        "answerable_sample_per_conv": answerable_sample,
        "seed": seed,
        "conversations": len(conversations),
        "elapsed_s": round(time.time() - started, 1),
        "judges": per_judge,
    }


def _print_report(r: dict[str, Any]) -> None:
    print()
    print("=" * 84)
    print(f"LOCOMO entailment sweep · {r['conversations']} conversations · {r['embedder']}")
    print("=" * 84)
    for label, j in r["judges"].items():
        print(f"\n{label}  ({j['model']})")
        print(f"  {'thresh':>7} {'adv abstain':>14} {'ans false-abstain':>18} {'separation':>11}")
        for p in j["roc"]:
            adv = p["adversarial_abstention"]["rate"]
            ans = p["answerable_false_abstain"]["rate"]
            star = "  <- best sep" if p is j["best_separation_point"] else ""
            print(f"  {p['threshold']:>7.2f} {adv:>14.3f} {ans:>18.3f} {p['separation']:>11.3f}{star}")
    print()
    print("A stronger judge shows as a higher BEST separation (more adversarial abstention at the")
    print("same answerable false-abstain). A better threshold on the SAME judge only moves the point.")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m recall.eval.locomo_entailment_sweep")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--embedder", default="fastembed")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--answerable-sample", type=int, default=40)
    p.add_argument("--conversations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--only-default-judge",
        action="store_true",
        help="sweep just the shipped distilroberta judge (skips the two model downloads)",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.data.exists():
        p.error(f"{args.data} not found — fetch locomo10.json (see recall.eval.locomo)")

    judges = DEFAULT_JUDGES[:1] if args.only_default_judge else DEFAULT_JUDGES
    report = run(
        args.data,
        dsn=args.dsn,
        embedder_name=args.embedder,
        k=args.k,
        answerable_sample=args.answerable_sample,
        limit=args.conversations,
        seed=args.seed,
        judges=judges,
    )
    _print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"full report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
