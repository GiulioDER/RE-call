"""Token-F1 re-scoring of head-to-head artifacts that were graded by an LLM judge.

::

    python -m benchmarks.token_f1 --compare A.json B.json

Reads the JSON artifacts written by ``benchmarks.run``. Like ``benchmarks.analyze`` it is pure
offline arithmetic -- no model calls, no database, no new artifact -- because every artifact
already stores the generated ``answer`` and the ``gold`` string per question. That is the whole
point of this module: a run that cost real money to generate can be re-scored under a *different*
metric for nothing, as often as a reviewer likes.

Why token F1 at all, when this repo's published tables are LLM-as-judge accuracy: because other
people's tables are token F1, and the two are not interchangeable. Judge accuracy is a binary
verdict from a model; token F1 is bag-of-words overlap against the answer key. They disagree on
exactly the answers this project cares about -- an exactly-right date phrased unlike the key
scores 1.0 from a judge and can score well under 1.0 here. Publishing one and comparing against
the other is the error this module exists to prevent.

The implementation is the standard SQuAD normalisation (lowercase, drop punctuation, drop
articles, collapse whitespace) over a multiset intersection. It is deliberately the textbook
version rather than anything tuned: the only defensible use of a foreign metric is the foreign
definition of it.

Prior work: none for this metric. Searched ``docs_search(source_type="memory")`` for the LOCOMO
head-to-head and for BEAM before writing this -- the relevant memos are
[[project-recall-memory-benchmark-and-voyage-2026-07-24]],
[[project-recall-mem0-locomo-article-adversarial-review-2026-07-25]] and
[[project-recall-beam-benchmark-2026-07-28]], and every one of them scores with an LLM judge.
A ``grep`` for ``f1`` across ``benchmarks/``, ``recall/`` and ``scripts/`` returned nothing, so
this is the first token-F1 scorer in the repo rather than a re-measurement of an existing one.

Two properties of this scorer that a reader should not have to discover by reading the code:

- **An abstention on an answerable question is scored, not dropped.** The refusal text is compared
  to the gold like any other answer and lands near 0.0. Dropping those rows would let a system
  raise its mean by refusing the questions it finds hard, which is the same failure mode
  ``analyze``'s discrimination metric exists to catch.
- **The comparison is PAIRED**, for the reason spelled out at length in ``benchmarks.analyze``:
  both arms answer the identical question set, so the evidence about a difference lives in the
  per-question differences, not in two independent means. The reported interval is a bootstrap
  over those per-question differences. Independent per-arm intervals would be the weaker question
  and are not offered here.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.artifact_contract import load_published_artifact

#: Resamples for the paired bootstrap. Large enough that the 2.5/97.5 percentiles are stable to
#: the third decimal across seeds, which is the precision these numbers are quoted at.
BOOTSTRAP_RESAMPLES = 10_000

#: Fixed so the published interval is reproducible. A bootstrap whose seed is the clock cannot be
#: checked by the reader, and "re-run it and see" is not a citation.
BOOTSTRAP_SEED = 0

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD answer normalisation: casefold, strip punctuation and articles, collapse spaces."""
    lowered = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def token_f1(prediction: str, gold: str) -> float:
    """Standard SQuAD token F1 between a prediction and a gold answer.

    Both empty is a match (1.0); exactly one empty is a miss (0.0) -- the degenerate cases the
    harmonic mean cannot express, handled explicitly rather than by a divide-by-zero guard.
    """
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0
    precision = shared / len(pred_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _answerable(artifact: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """`question_id -> (answer, gold)` for the answerable questions of one artifact.

    Adversarial questions are excluded because they have no gold answer to overlap with; their
    axis is abstention, which ``benchmarks.analyze`` reports and this metric cannot express.
    """
    return {
        outcome["question_id"]: (outcome.get("answer") or "", outcome.get("gold") or "")
        for outcome in artifact["outcomes"]
        if not outcome.get("is_adversarial")
    }


def paired_bootstrap(
    differences: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for the mean of per-question differences."""
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def compare(a_path: Path, b_path: Path) -> dict[str, Any]:
    """Paired token-F1 comparison of two artifacts over the questions they share."""
    a_artifact = load_published_artifact(a_path)
    b_artifact = load_published_artifact(b_path)
    a_rows, b_rows = _answerable(a_artifact), _answerable(b_artifact)

    shared = sorted(set(a_rows) & set(b_rows))
    if not shared:
        raise SystemExit(f"no shared answerable questions between {a_path} and {b_path}")

    a_scores = [token_f1(*a_rows[qid]) for qid in shared]
    b_scores = [token_f1(*b_rows[qid]) for qid in shared]
    differences = [x - y for x, y in zip(a_scores, b_scores, strict=True)]
    a_mean = sum(a_scores) / len(a_scores)
    b_mean = sum(b_scores) / len(b_scores)
    low, high = paired_bootstrap(differences)

    return {
        "metric": "token F1 (SQuAD normalisation), answerable questions only",
        "n": len(shared),
        "a": {"path": str(a_path), "arm": a_artifact.get("arm"),
              "model": a_artifact.get("model"), "token_f1": a_mean},
        "b": {"path": str(b_path), "arm": b_artifact.get("arm"),
              "model": b_artifact.get("model"), "token_f1": b_mean},
        "delta": a_mean - b_mean,
        "ci95_paired_bootstrap": [low, high],
        "dropped_unshared": {
            "a_only": len(set(a_rows) - set(b_rows)),
            "b_only": len(set(b_rows) - set(a_rows)),
        },
    }


def _report(result: Mapping[str, Any]) -> list[str]:
    a, b = result["a"], result["b"]
    low, high = result["ci95_paired_bootstrap"]
    dropped = result["dropped_unshared"]
    lines = [
        result["metric"],
        f"  n (paired)        {result['n']}",
        f"  {str(a['arm']):<16}  {a['token_f1']:.4f}   [{a['model']}]",
        f"  {str(b['arm']):<16}  {b['token_f1']:.4f}   [{b['model']}]",
        f"  delta             {result['delta']:+.4f}   95% CI [{low:+.4f}, {high:+.4f}]",
    ]
    if dropped["a_only"] or dropped["b_only"]:
        lines.append(
            f"  unshared dropped  {dropped['a_only']} A-only, {dropped['b_only']} B-only"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.token_f1",
        description="Paired token-F1 re-scoring of two results artifacts, offline.",
    )
    parser.add_argument(
        "--compare", nargs=2, required=True, metavar=("A.json", "B.json"),
        help="two results artifacts over the SAME question set, to score pairwise",
    )
    parser.add_argument("--json", action="store_true", help="emit the raw result object")
    args = parser.parse_args(argv)

    result = compare(Path(args.compare[0]), Path(args.compare[1]))
    print(json.dumps(result, indent=2) if args.json else "\n".join(_report(result)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
