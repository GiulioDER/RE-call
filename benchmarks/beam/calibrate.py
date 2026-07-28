"""Fit RE-call's abstention threshold for the BEAM embedder, OUT OF SAMPLE.

::

    # $0 of LLM spend — embeddings and cosines only, no answerer, no judge.
    python -m benchmarks.beam.calibrate --data <shards> --conversations 30-34 \
        --embedder router:openai/text-embedding-3-small

Why this step exists
--------------------
`recall.trust` abstains when no hit clears a cosine threshold, and that threshold is
**per-embedder**: `recall.calibration`'s own words are "each model's cosines live in a different
regime". With no calibration on disk the library falls back to `DEFAULT_GAP_THRESHOLD = 0.50` and
flags every result `calibrated=False`.

The BEAM arm runs on `text-embedding-3-small` (see FINDINGS §9g) — an embedder this repo has never
calibrated. Measured on conversation 0, its top-1 cosines run 0.41-0.76 against a 0.50 floor, so
the default is not absurd, but it is still a CONSTANT rather than a measurement: a cell produced
under it reports the constant as much as the retriever. Since abstention is the axis this whole
arm exists to price, that is the one number that must not be an accident.

Why out of sample
-----------------
Fitting the threshold on the questions it is then scored against is in-sample tuning — the exact
error this project has already paid for once (an ML quality override moved 0.55 -> 0.50 on
in-sample evidence and had to be reverted). So the threshold is fit on conversations the scored
run never touches, and the scored run is told which ones those were.

The labels are BEAM's own, not ours: the `abstention` category is unanswerable BY CONSTRUCTION
(its questions ask about things absent from the conversation), and the other nine are answerable.
No hand-labelling is involved, so there is nothing here to tune toward a preferred answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import iter_conversations
from benchmarks.beam.systems import BEAM_TABLE, BeamRecallSystem
from recall.calibration import from_samples, save
from recall.eval.locomo import DEFAULT_DSN

#: BEAM's unanswerable category. Everything else is answerable.
UNANSWERABLE_TYPE = "abstention"


def _parse_indices(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Shard directory or parquet")
    parser.add_argument("--chat-size", default="1M")
    parser.add_argument(
        "--conversations",
        required=True,
        help="HELD-OUT conversations to fit on, e.g. 30-34. The scored run must exclude these.",
    )
    parser.add_argument("--embedder", default="router:openai/text-embedding-3-small")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--table", default=BEAM_TABLE)
    parser.add_argument("--out", type=Path, default=None, help="Calibration JSON path")
    args = parser.parse_args()

    indices = _parse_indices(args.conversations)
    system = BeamRecallSystem(args.dsn, embedder_name=args.embedder, table=args.table)
    embedder_name = system.describe()["embedder"]["model"]

    answerable: list[float] = []
    unanswerable: list[float] = []
    per_question: list[dict[str, Any]] = []

    for conv in iter_conversations(args.data, args.chat_size, indices):
        system.ingest(conv)
        print(f"  conversation {conv.index}: indexed, {len(conv.questions)} questions", flush=True)
        for q in conv.questions:
            # The RAW top-1 cosine, deliberately NOT `retrieve()`: retrieve applies the very
            # threshold being fitted, so sampling through it would only ever return scores above
            # the current one and the fit would chase its own tail.
            top = system.top_cosine(q.question)
            (unanswerable if q.question_type == UNANSWERABLE_TYPE else answerable).append(top)
            per_question.append(
                {"question_id": q.question_id, "type": q.question_type, "top_cosine": round(top, 4)}
            )

    if len(answerable) < 2 or len(unanswerable) < 2:
        raise SystemExit(
            f"not enough samples to fit: {len(answerable)} answerable, "
            f"{len(unanswerable)} unanswerable — widen --conversations"
        )

    cal = from_samples(embedder_name, answerable, unanswerable)
    path = save(cal, args.out)

    report = {
        "embedder": embedder_name,
        "fit_on_conversations": indices,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "threshold": cal.threshold,
        "scale": cal.scale,
        "separability": cal.separability,
        "certified": cal.certified,
        "certification_reason": cal.certification_reason,
        "calibration_path": str(path),
        # Kept so a reader can see the distributions the threshold came from rather than trust
        # the single number it collapsed to.
        "per_question": per_question,
    }
    if args.out:
        args.out.with_name(args.out.stem + "_report.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8"
        )
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
