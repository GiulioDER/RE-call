"""Where does ATM-Bench's remaining loss actually sit, given the evidence was already retrieved?

`docs/ATM_BENCH.md` §4 establishes the shape of the problem: retrieval finds the evidence for
roughly nine questions in ten while the score lands at roughly seven in ten. That gap is the
answer side. This script decomposes it, and it does so with **zero provider calls**: the run's
per-question judgements already exist in the official evaluator's output, and every figure below
is an aggregation over them.

That property is the reason this is a script and not a run. A judge pass over the full split costs
real money and cannot be repeated for free, so a decomposition that needs one would be measured
once and never re-checked. This one is re-runnable against the archived package at any time, which
is what lets its artifact be regenerated rather than trusted.

**The abstention definition is the evaluator's own.** `is_abstention` is imported from the
ATM-Bench checkout rather than reimplemented here, because a local reimplementation would be
measuring this file instead of the thing the score was computed from. That is also why
`--atm-bench-root` is required rather than optional: there is no correct fallback, and a silently
degraded definition would move every number in the artifact without failing.

**What this script deliberately does NOT write.** The artifact holds aggregates only: counts,
means and ratios. No question text, no gold answer, no model answer, and no per-question row. The
ATM-Bench corpus is third-party licensed data and the run package is archived outside this tree;
copying either into `results/` to back a number would be trading one problem for a worse one.

Prior work: searched `docs_search(source_type='memory', "ATM Bench answer selection abstention
over-abstention diagnosis replay official scorer")` before writing this. The binding hit is
[[atm-over-abstention-is-the-largest-mechanism]], which records this decomposition being run from
throwaway scripts in a scratch directory and says in as many words that they belong at
`benchmarks/atm_answer_diagnosis.py`. This file is that, with the pure aggregation split out so it
can be tested without the dataset present.

Run:

    python benchmarks/atm_answer_diagnosis.py \\
        --package /path/to/atm-benchmark-20260821 \\
        --ground-truth /path/to/atm-bench.json \\
        --atm-bench-root /path/to/atm-bench-official \\
        --out results/atm/atm_answer_diagnosis_20260822.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.artifact_contract import load_published_artifact

#: The three ATM-Bench question types, in the order the official summary reports them.
QTYPES: tuple[str, ...] = ("number", "list_recall", "open_end")

#: A question counts as "the answer was on screen" when at least this share of the gold answer's
#: content tokens appear in the evidence text the model actually received. 0.8 rather than 1.0
#: because gold answers carry formatting the packed evidence need not reproduce verbatim (currency
#: symbols, thousands separators, an article), and rather than something lower because at 0.5 the
#: measure starts counting questions whose gold answer merely shares vocabulary with the evidence.
#: ⚠️ It is a LEXICAL proxy for "answerable from what was shown", not a judgement, so it over-counts
#: wherever the tokens co-occur without the relationship the question asks about.
ON_SCREEN_THRESHOLD = 0.8

#: Below this share, the gold item was retrieved but its text does not carry the answer at all:
#: the modality ceiling of answering from a generated description rather than from the image.
MODALITY_FLOOR_THRESHOLD = 0.5


@dataclass(frozen=True)
class QuestionRecord:
    """One question, reduced to the facts this decomposition needs.

    Deliberately not a view over the source rows: everything downstream operates on this, so the
    aggregation can be tested on hand-built records with no dataset, no evaluator checkout and no
    archived package present.
    """

    qid: str
    qtype: str
    score: float
    model_abstained: bool
    gold_is_abstention: bool
    evidence_complete: bool
    evidence_hit: bool
    #: Share of the gold answer's content tokens present in the packed evidence, or `None` when the
    #: gold answer has no content tokens at all and the measure is undefined.
    coverage: float | None
    #: Score of the RANK-1 returned hit, which is the signal a trust-layer gate would threshold.
    #:
    #: ⚠️ NOT `max(score)` over the returned hits, and the two genuinely differ here: a reranker
    #: reorders without rescoring, so the highest-scoring chunk need not be the one it put first.
    #: `results/ARTIFACTS.md` records the same distinction biting the enterprise dense-floor probe.
    #: Written as `max()` first, which moved the published separability from 0.493 to 0.491 and both
    #: medians by about 0.02 -- a wrong number that reads exactly as plausible as the right one.
    top1_score: float | None


def _mean(values: Iterable[float]) -> float:
    seq = list(values)
    return statistics.fmean(seq) if seq else 0.0


def token_coverage(
    gold_answer: str,
    evidence_text: str,
    tokenize: Callable[[str], Sequence[str]],
    stopwords: frozenset[str] | set[str],
) -> float | None:
    """Share of the gold answer's content tokens that appear in the evidence.

    Returns `None`, never 0.0, when the gold answer reduces to no content tokens. The distinction
    matters: 0.0 means "the answer was not on screen" and would be counted as a recoverable miss,
    while `None` means the measure could not be taken. Collapsing the two is the `.get(k, default)`
    failure mode -- an absent input becoming a clean, plausible, wrong number.
    """
    tokens = [t for t in tokenize(gold_answer) if t not in stopwords]
    if not tokens:
        return None
    haystack = evidence_text.lower()
    return sum(1 for t in tokens if t in haystack) / len(tokens)


def recoverable_with_complete_evidence(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    """Points lost on questions whose COMPLETE gold evidence was in the retrieved top k.

    This is the honest denominator for any answer-side proposal: a question whose evidence was
    never retrieved cannot be won by a better reader, so counting it would inflate the prize.
    """
    total = len(records)
    complete = [r for r in records if r.evidence_complete]
    lost = sum(1.0 - r.score for r in complete)
    per_type: dict[str, Any] = {}
    for qtype in QTYPES:
        subset = [r for r in records if r.qtype == qtype]
        if not subset:
            continue
        subset_complete = [r for r in subset if r.evidence_complete]
        subset_lost = sum(1.0 - r.score for r in subset_complete)
        per_type[qtype] = {
            "count": len(subset),
            "qs": _mean(r.score for r in subset),
            "complete_evidence_rate": len(subset_complete) / len(subset),
            "qs_given_complete": _mean(r.score for r in subset_complete),
            "qs_given_incomplete": _mean(r.score for r in subset if not r.evidence_complete),
            "points_lost": subset_lost,
            "qs_points_lost": 100 * subset_lost / total,
        }
    return {
        "points_lost": lost,
        "qs_points_lost": 100 * lost / total,
        "by_qtype": per_type,
    }


def abstention_breakdown(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    """Refusals, split into the ones the benchmark wanted and the ones that are dead loss.

    The asymmetry is the useful part and it is what makes a rescue safe or unsafe: if every
    gold-abstention question is one type, then a rescue restricted to the other types cannot cost
    a point it would otherwise have earned.
    """
    total = len(records)
    abstained = [r for r in records if r.model_abstained]
    wrong = [r for r in abstained if not r.gold_is_abstention and r.score < 0.5]
    per_type: dict[str, Any] = {}
    for qtype in QTYPES:
        subset = [r for r in records if r.qtype == qtype]
        if not subset:
            continue
        subset_abstained = [r for r in subset if r.model_abstained]
        per_type[qtype] = {
            "count": len(subset),
            "abstentions": len(subset_abstained),
            "abstention_rate": len(subset_abstained) / len(subset),
            "mean_score_when_abstaining": _mean(r.score for r in subset_abstained),
            "gold_abstention_questions": sum(1 for r in subset if r.gold_is_abstention),
            "wrong_abstentions": sum(1 for r in wrong if r.qtype == qtype),
        }
    lost = sum(1.0 - r.score for r in wrong)
    on_screen = [
        r for r in wrong if r.coverage is not None and r.coverage >= ON_SCREEN_THRESHOLD
    ]
    return {
        "abstentions": len(abstained),
        "abstention_rate": len(abstained) / total,
        "abstention_rate_percent": 100 * len(abstained) / total,
        # "Correct" here means NOT DEAD LOSS, which is the quantity a rescue proposal is sized
        # against. It is `abstained - wrong`, so it also admits a refusal the judge gave credit to
        # on an answerable question. On the published run the two agree exactly (17 either way),
        # and the strict count is emitted beside it so a future run where they diverge says so
        # instead of quietly widening what "correct" means.
        "correct_abstentions": len(abstained) - len(wrong),
        "abstentions_on_gold_abstention_questions": sum(
            1 for r in abstained if r.gold_is_abstention
        ),
        "wrong_abstentions": len(wrong),
        "wrong_abstention_points_lost": lost,
        "wrong_abstention_qs_points_lost": 100 * lost / total,
        "wrong_abstentions_with_complete_evidence": sum(1 for r in wrong if r.evidence_complete),
        "wrong_abstentions_with_answer_on_screen": len(on_screen),
        "gold_abstention_questions": sum(1 for r in records if r.gold_is_abstention),
        "by_qtype": per_type,
    }


def abstention_separability(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    """Does the retrieval score tell a correct refusal from a wrong one?

    Reported as the probability that a randomly drawn correct-abstention score is BELOW a randomly
    drawn wrong-abstention one, ties counted as half. 0.5 is a coin. This exists to kill an idea
    rather than to support one: "gate abstention on the calibrated trust layer" is the obvious
    proposal, and a threshold on a signal that does not separate is a rigged coin, not a judgement.
    """
    correct = [r.top1_score for r in records
               if r.model_abstained and r.gold_is_abstention and r.top1_score is not None]
    wrong = [r.top1_score for r in records
             if r.model_abstained and not r.gold_is_abstention and r.score < 0.5
             and r.top1_score is not None]
    if not correct or not wrong:
        return {"measurable": False, "correct_n": len(correct), "wrong_n": len(wrong)}
    wins = sum(
        1.0 if c < w else 0.5 if c == w else 0.0
        for c in correct
        for w in wrong
    )
    return {
        "measurable": True,
        "correct_n": len(correct),
        "wrong_n": len(wrong),
        "p_correct_below_wrong": wins / (len(correct) * len(wrong)),
        "correct_median": statistics.median(correct),
        "wrong_median": statistics.median(wrong),
    }


def answered_wrong_with_evidence(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    """Questions answered wrongly with the answer's own tokens in the evidence on screen.

    The selection failure, as distinct from the refusal failure: the model produced an answer, a
    real one, and picked the wrong item out of what it had been shown.
    """
    total = len(records)
    hits = [
        r for r in records
        if not r.model_abstained
        and r.score < 0.5
        and r.coverage is not None
        and r.coverage >= ON_SCREEN_THRESHOLD
    ]
    lost = sum(1.0 - r.score for r in hits)
    return {
        "count": len(hits),
        "points_lost": lost,
        "qs_points_lost": 100 * lost / total,
        "by_qtype": {q: sum(1 for r in hits if r.qtype == q) for q in QTYPES},
        "on_screen_threshold": ON_SCREEN_THRESHOLD,
        "on_screen_threshold_percent": 100 * ON_SCREEN_THRESHOLD,
    }


def modality_floor(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    """Questions where the gold item WAS retrieved but its text does not carry the answer.

    Not winnable by any prompt or selector, because the answer is in the image and the pipeline
    sees a generated description of it. Published so that nobody plans against the recoverable
    total without subtracting this.

    🔁 The figure this replaces was 20 questions / 1.97 QS, and it was wrong for a reason worth
    keeping: the throwaway script that first produced it tested `(coverage or 0) < 0.5`, so the two
    questions whose gold answer has no content tokens -- where coverage is UNMEASURABLE -- became a
    coverage of 0 and were counted as "the answer was not on screen". `token_coverage` returns
    `None` rather than 0.0 precisely to make that impossible, and this filter honours it.
    """
    total = len(records)
    # `score < 0.5` is load-bearing, not a tidy-up. The floor is a statement about LOSS, so a
    # question the model got right anyway is not stuck on it. Omitting the filter counted 33
    # questions worth 1.78 QS instead of 20 worth 1.97 -- more questions and fewer points, which is
    # the signature of sweeping in ones that scored well.
    stuck = [
        r for r in records
        if r.evidence_hit
        and r.score < 0.5
        and r.coverage is not None
        and r.coverage < MODALITY_FLOOR_THRESHOLD
    ]
    lost = sum(1.0 - r.score for r in stuck)
    return {
        "count": len(stuck),
        "points_lost": lost,
        "qs_points_lost": 100 * lost / total,
        "coverage_threshold": MODALITY_FLOOR_THRESHOLD,
    }


def summarise(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    """The whole decomposition, from records alone. No I/O, so it is testable without the dataset."""
    if not records:
        raise ValueError("no records: refusing to publish a decomposition of nothing")
    return {
        "question_count": len(records),
        "qs": _mean(r.score for r in records),
        "qs_percent": 100 * _mean(r.score for r in records),
        "qs_by_qtype": {
            q: _mean(r.score for r in records if r.qtype == q)
            for q in QTYPES
            if any(r.qtype == q for r in records)
        },
        "recoverable_with_complete_evidence": recoverable_with_complete_evidence(records),
        "abstention": abstention_breakdown(records),
        "abstention_separability": abstention_separability(records),
        "answered_wrong_with_evidence_on_screen": answered_wrong_with_evidence(records),
        "modality_floor": modality_floor(records),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_records(
    package: Path,
    ground_truth: Path,
    is_abstention: Callable[[str], bool],
    tokenize: Callable[[str], Sequence[str]],
    stopwords: frozenset[str] | set[str],
) -> tuple[QuestionRecord, ...]:
    """Build records from the archived run package and the released ground truth.

    Every lookup below is `[...]`, never `.get(key, default)`. A missing prediction or a missing
    retrieval row must raise here rather than become a clean zero that scores as an ordinary
    negative result and silently drags every aggregate down.
    """
    gold_rows: Any = json.loads(ground_truth.read_text(encoding="utf-8"))
    if isinstance(gold_rows, Mapping):
        gold_rows = gold_rows["qas"]
    gold = {str(row["id"]): row for row in gold_rows}

    answers = {str(row["id"]): row["answer"] for row in _read_jsonl(package / "answers.jsonl")}
    retrieval = {str(row["id"]): row for row in _read_jsonl(package / "retrieval.jsonl")}
    judged = {
        str(row["id"]): row
        for row in json.loads(
            (package / "official_eval/atm_openai_gpt-5-mini.json").read_text(encoding="utf-8")
        )
    }

    missing = sorted(set(gold) - (set(answers) & set(retrieval) & set(judged)))
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(gold)} questions are missing an answer, a retrieval row or a "
            f"judgement; refusing to publish a partial decomposition. First: {missing[:3]}"
        )

    records: list[QuestionRecord] = []
    for qid, row in gold.items():
        retrieved = retrieval[qid]
        hits = retrieved["hits"]
        gold_evidence = {str(x) for x in row["evidence_ids"]}
        returned = {str(x) for x in retrieved["retrieval_ids"]}
        records.append(
            QuestionRecord(
                qid=qid,
                qtype=row["qtype"],
                score=float(judged[qid]["accuracy"]),
                model_abstained=is_abstention(answers[qid]),
                gold_is_abstention=is_abstention(row["answer"]),
                evidence_complete=bool(gold_evidence) and gold_evidence <= returned,
                evidence_hit=bool(gold_evidence & returned),
                coverage=token_coverage(
                    row["answer"], " ".join(h["text"] for h in hits), tokenize, stopwords
                ),
                top1_score=float(hits[0]["score"]) if hits else None,
            )
        )
    return tuple(records)


def _import_official_normalizer(root: Path) -> tuple[Any, Any, Any]:
    """Import the evaluator's own abstention and tokenisation rules, or stop.

    No fallback on purpose. A vendored copy would drift from the definition the published score was
    computed with, and the drift would be invisible: every number here would still be produced, and
    still be wrong.
    """
    if not (root / "memqa").is_dir():
        raise SystemExit(
            f"--atm-bench-root does not look like an ATM-Bench checkout (no memqa/ under {root}). "
            f"This script needs the evaluator's own is_abstention, and will not substitute its own."
        )
    sys.path.insert(0, str(root))
    try:
        from memqa.utils.evaluator.normalizer import STOPWORDS, is_abstention, tokenize
    except ImportError as exc:  # pragma: no cover - depends on an external checkout
        raise SystemExit(f"could not import the official normalizer from {root}: {exc}") from exc
    return is_abstention, tokenize, STOPWORDS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True, help="the archived run package directory")
    parser.add_argument("--ground-truth", type=Path, required=True, help="atm-bench.json")
    parser.add_argument("--atm-bench-root", type=Path, required=True, help="the ATM-Bench checkout")
    parser.add_argument("--out", type=Path, required=True, help="artifact to write")
    parser.add_argument("--run-artifact", type=Path, default=None,
                        help="the run's own artifact, whose QS this decomposition must reproduce")
    args = parser.parse_args(argv)

    is_abstention, tokenize, stopwords = _import_official_normalizer(args.atm_bench_root)
    records = load_records(args.package, args.ground_truth, is_abstention, tokenize, stopwords)
    summary = summarise(records)

    # The apparatus check, and it is not decoration: if the replay does not reproduce the published
    # QS then every decomposition below it is describing a different run, and the right outcome is
    # to refuse rather than to publish a plausible table.
    if args.run_artifact is not None:
        # `load_published_artifact`, never a bare `json.loads`. `benchmarks.run` marks an artifact
        # it REFUSED to publish in band, and a refused file is byte-identical to a real measurement
        # to anything that does not honour the mark -- so reading it directly would let a
        # quarantined run silently become the baseline this decomposition validates itself against.
        published = load_published_artifact(args.run_artifact)["official_score"]["qs"]
        if abs(published - summary["qs"]) > 5e-5:
            raise SystemExit(
                f"replay does not reproduce the published score: {summary['qs']:.6f} against "
                f"{published:.6f}. Refusing to write an artifact for a run this is not."
            )
        summary["reproduces_published_qs"] = True
        summary["published_qs"] = published

    summary["cost_claims"] = []
    summary["_provenance"] = {
        "generation": "post-#81/#84",
        "status": "current",
        "note": (
            "Decomposition of the answer-side loss on the 2026-08-21 ATM-Bench full run, computed "
            "with zero provider calls by aggregating the official evaluator's own per-question "
            "judgements. Abstention and tokenisation come from the evaluator's normalizer, not "
            "from a local reimplementation. Aggregates only: no question text, no gold answer and "
            "no model answer is copied into this file, because the corpus is third-party data."
        ),
        "backs": ["docs/ATM_BENCH.md section 5"],
        "superseded_by": None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {args.out}")
    print(f"  QS {summary['qs_percent']:.4f} over {summary['question_count']} questions")
    rec = summary["recoverable_with_complete_evidence"]
    print(f"  recoverable with complete evidence: {rec['qs_points_lost']:.2f} QS")
    ab = summary["abstention"]
    print(f"  abstentions {ab['abstentions']} ({ab['correct_abstentions']} correct), "
          f"wrong ones cost {ab['wrong_abstention_qs_points_lost']:.2f} QS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
