"""Re-score a finished results artifact with a different judge, without re-running generation.

::

    python -m benchmarks.rejudge --input benchmarks/results/recall_...json \\
        --model anthropic/claude-sonnet-4.5 --out benchmarks/results/recall_...rejudged.json

Why this exists
---------------
The head-to-head runs are judged by ``openai/gpt-4o-mini``, and an independent audit of LOCOMO
scoring (github.com/dial481/locomo-audit, ``ap-baseline/``) measured that judge accepting **62.81%
of deliberately wrong-but-on-topic answers**: it rewards vagueness. A published claim that rests
on one lenient judge is one hostile reviewer away from being worthless, and "we re-ran everything
with a better judge" costs the price of the whole benchmark again.

It does not have to. ``benchmarks.run`` deliberately writes ``question``, ``gold`` and ``answer``
into every per-question record precisely so the SCORING step can be replayed independently of the
RETRIEVAL and GENERATION steps. This module is that replay: it reads an artifact, calls a judge of
the reader's choosing on the stored triples, and writes a NEW artifact in the same schema. The
retrieved context and the generated answers are byte-identical to the original run, so the
difference between the two files is attributable to the judge and to nothing else.

The rule that decides the denominator
-------------------------------------
Only the records the ORIGINAL run actually put in front of a judge are re-judged, and the rule is
copied from `benchmarks.pipeline.run_question` rather than re-derived:

* adversarial   -> ``correct`` stays ``None``. There is no gold answer to grade against.
* abstained     -> ``correct`` stays ``False``, judge not called. Refusing an answerable question
                   is wrong by definition, not a grading question.
* everything else -> re-judged.

Getting this wrong is not a cosmetic error. Feeding the abstentions to the judge would let a
strong judge "rescue" a refusal, and grading adversarials would add 446 unscoreable rows to the
accuracy denominator. Either one moves the headline number while looking like a judge effect.
Both are pinned by tests.

The agreement report
--------------------
The output carries an agreement block: how many records were judged, how many verdicts the two
judges agreed on, and — this is the part that matters — the FULL LIST of disagreements with the
question, the gold answer, the model's answer and both verdicts. A reader can read the
disagreements and form their own view rather than take the new number on trust. Flips are counted
in both directions, because a judge that only ever flips one way is a judge with a thumb on the
scale.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.llm import Completer, OpenRouterLLM, is_terminal
from benchmarks.pipeline import Outcome, aggregate, judge_correct
from benchmarks.artifact_contract import load_published_artifact


def to_outcome(record: Mapping[str, Any]) -> Outcome:
    """Rebuild the scored `Outcome` from one artifact record, dropping the joined-in text.

    A record is an `Outcome` plus ``question`` and ``gold`` (see ``benchmarks.run``), and
    `benchmarks.pipeline.aggregate` takes `Outcome` objects — so re-aggregating a file means
    coming back through this. `Outcome.__post_init__` re-checks the ``correct is None iff
    adversarial`` invariant on the way, which turns a hand-edited or truncated artifact into an
    error here instead of into a plausible-looking published number.
    """
    return Outcome(
        question_id=str(record["question_id"]),
        category=str(record["category"]),
        is_adversarial=bool(record["is_adversarial"]),
        context=str(record["context"]),
        answer=str(record["answer"]),
        abstained=bool(record["abstained"]),
        correct=None if record["correct"] is None else bool(record["correct"]),
        query_class=str(record.get("query_class", "unknown")),
        matched_rules=tuple(str(value) for value in record.get("matched_rules", ())),
        routing_profile=str(record.get("routing_profile", "fast")),
        routing_expansion=record.get("routing_expansion"),
        routing_mode=str(record.get("routing_mode", "shadow")),
        evidence_budget=record.get("evidence_budget"),
        evidence_tokens_exact=record.get("evidence_tokens_exact"),
        input_tokens_exact=record.get("input_tokens_exact"),
    )


def _was_judged(record: Mapping[str, Any]) -> bool:
    """True iff the original run called the judge on this record — the ONE rule that must not drift.

    Mirrors the branch structure of `benchmarks.pipeline.run_question`: the judge runs only for a
    question that is answerable AND was actually answered. Everything else has a verdict fixed by
    construction, and re-judging it would change what the accuracy rate is a rate OF.
    """
    return not bool(record["is_adversarial"]) and not bool(record["abstained"])


def _disagreement(record: Mapping[str, Any], old: bool, new: bool) -> dict[str, Any]:
    """One human-readable row of the disagreement list: everything needed to adjudicate by eye."""
    return {
        "question_id": str(record["question_id"]),
        "category": str(record["category"]),
        "question": str(record.get("question", "")),
        "gold": str(record.get("gold", "")),
        "answer": str(record["answer"]),
        "old_verdict": old,
        "new_verdict": new,
    }


def rejudge_outcomes(
    outcomes: Sequence[Mapping[str, Any]], completer: Completer
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-score `outcomes` with `completer` as the judge; return (new records, agreement report).

    The input is never mutated: every record is copied before its ``correct`` field is touched, so
    the caller's list — and, in the CLI's case, the file it was parsed from — is left exactly as
    it was found. The artifact cost real money and may already have been published; re-scoring it
    in place would destroy the baseline the re-score is supposed to be compared against.
    """
    records: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    failed: list[str] = []
    n_judged = n_agree = flipped_to_correct = flipped_to_incorrect = 0

    for record in outcomes:
        # Validate through Outcome even for the rows we will not re-judge: a record whose
        # `correct`/`is_adversarial` pair breaks the invariant is a corrupt artifact, and it is
        # better to say so than to re-aggregate around it.
        to_outcome(record)
        new_record = dict(record)
        if not _was_judged(record):
            records.append(new_record)
            continue

        old = bool(record["correct"])
        try:
            new = judge_correct(
                completer,
                str(record.get("question", "")),
                str(record.get("gold", "")),
                str(record["answer"]),
            )
        except Exception as exc:  # noqa: BLE001 - re-raised if terminal, quarantined otherwise
            if is_terminal(exc):
                # A dead account re-judges nothing. Quarantining it would report an agreement rate
                # computed over the handful of records that ran before the credit ran out.
                raise
            # ⛔ The record keeps its ORIGINAL verdict and is counted as FAILED, not as agreed.
            # `n_judged` is the denominator of `agreement_rate`, the number this tool exists to
            # produce, so leaving the old verdict in place and saying nothing would score a broken
            # judge as one that agrees with the baseline perfectly. The whole loop used to die
            # here instead, discarding every record already re-judged, because the output is
            # written only after it finishes.
            new_record["rejudge_failed"] = True
            failed.append(str(record.get("question_id", "")))
            records.append(new_record)
            continue

        n_judged += 1
        new_record["correct"] = new
        records.append(new_record)

        if new == old:
            n_agree += 1
        else:
            disagreements.append(_disagreement(record, old, new))
            if new:
                flipped_to_correct += 1
            else:
                flipped_to_incorrect += 1

    n_disagree = n_judged - n_agree
    report = {
        "n_judged": n_judged,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        # None, not 1.0, for an artifact with nothing to judge: a vacuous "perfect agreement" is
        # the kind of number that gets quoted without its n.
        "agreement_rate": None if n_judged == 0 else round(n_agree / n_judged, 4),
        "flipped_to_correct": flipped_to_correct,
        "flipped_to_incorrect": flipped_to_incorrect,
        # Reported, never folded into the rate above. A record whose re-judgment failed kept its
        # ORIGINAL verdict, so counting it as agreement would be counting the baseline against
        # itself; it is excluded from `n_judged` and named here instead.
        "n_failed": len(failed),
        "failed_question_ids": failed,
        "disagreements": disagreements,
    }
    return records, report


def rejudge_document(
    doc: Mapping[str, Any], completer: Completer, judge_model: str, source: str
) -> dict[str, Any]:
    """A re-scored copy of a whole results artifact, in the schema `benchmarks.run` writes.

    Same top-level keys, same ``outcomes`` record shape, a freshly computed ``aggregate`` from the
    shipped `benchmarks.pipeline.aggregate` — so the output drops straight into
    ``python -m benchmarks.analyze --compare`` beside an artifact from the original judge, and a
    reader does not have to learn a second file format to check the re-score.

    The provenance block is what stops the two files being confusable. It names the judge model
    that produced these verdicts, points at the artifact it was derived from, and keeps the
    ORIGINAL aggregate alongside the new one, so the judge effect is readable from the re-scored
    file on its own.
    """
    records, report = rejudge_outcomes(doc["outcomes"], completer)
    payload = dict(doc)
    payload["outcomes"] = records
    payload["aggregate"] = aggregate([to_outcome(r) for r in records])
    # ⛔ `aggregate` sits at the top of the payload beside `original_aggregate`, and that pair is
    # exactly what a reader diffs to attribute a change to the judge. A record whose re-judgment
    # FAILED kept its baseline verdict, so it enters that rate unmarked and the headline becomes a
    # silent mixture of two judges. `n_failed` was reasoned about for `agreement_rate` and not for
    # this, which is the number more people will read. Surfaced here so the mixture cannot be read
    # without it; the per-record `rejudge_failed` flag says which ones.
    payload["aggregate_mixed_verdicts"] = {
        "n": report["n_failed"],
        "question_ids": report["failed_question_ids"],
        "note": (
            "records whose re-judgment failed and kept the ORIGINAL judge's verdict. They are "
            "included in `aggregate` above and excluded from `rejudge.agreement`, so a non-zero "
            "n here means `aggregate` is not attributable to the new judge alone."
        ),
    }
    payload["rejudge"] = {
        "judge_model": judge_model,
        "source": source,
        "original_judge_model": (doc.get("config") or {}).get("model"),
        "original_aggregate": doc.get("aggregate"),
        "agreement": report,
    }
    return payload


def _load_document(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = load_published_artifact(path)
    if "outcomes" not in doc:
        raise ValueError(f"{path} has no 'outcomes' array — it is not a results artifact")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.rejudge",
        description="Re-score a results artifact with a stronger judge. No generation, no DB.",
    )
    parser.add_argument("--input", type=Path, required=True, help="results artifact to re-score")
    parser.add_argument(
        "--model", required=True, help="judge model, e.g. anthropic/claude-sonnet-4.5"
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write the new artifact")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"{args.input} not found")
    # Resolved, not compared as spelled: `--out ./x.json` and `--input x.json` are the same file.
    # Overwriting the input would destroy the baseline this whole module exists to compare against.
    if args.out.resolve() == args.input.resolve():
        parser.error("--out must differ from --input: the source artifact is never modified")
    if args.out.exists():
        parser.error(f"{args.out} already exists - refusing to overwrite a results artifact")

    # Imported here rather than at module scope so this module is importable (and testable) in an
    # environment where `benchmarks.run`'s dependencies are not installed.
    from benchmarks.run import validate_openrouter_key

    try:
        key = validate_openrouter_key(os.environ.get("OPENROUTER_API_KEY"))
    except ValueError as exc:
        parser.error(str(exc))

    doc = _load_document(args.input)
    llm = OpenRouterLLM(model=args.model, api_key=key)
    payload = rejudge_document(doc, llm.complete, args.model, str(args.input))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report: dict[str, Any] = payload["rejudge"]["agreement"]
    print(
        f"re-judged {report['n_judged']} records with {args.model}: "
        f"agreement={report['agreement_rate']} "
        f"(+{report['flipped_to_correct']} correct / -{report['flipped_to_incorrect']} correct)"
    )
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"re-scored artifact -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
