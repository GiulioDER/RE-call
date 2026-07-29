"""Fit RE-call's abstention threshold for the BEAM embedder, OUT OF SAMPLE.

::

    # $0 of LLM spend — embeddings and cosines only, no answerer, no judge.
    # --out is REQUIRED: without it this used to write the PROCESS-GLOBAL calibration.json that
    # `trusted_search` autoloads for every later run started from the same directory.
    # The fit this produces on BEAM is 0.617 and UNCERTIFIED (see below), so the run refuses to
    # write the calibration unless you say you want it kept for study. The *_report.json beside
    # it is written either way.
    python -m benchmarks.beam.calibrate --data <shards> --conversations 30-34 \
        --embedder router:openai/text-embedding-3-small --out beam_calibration.json \
        --save-uncertified

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

from benchmarks.beam.dataset import iter_conversations, parse_conversation_indices
from benchmarks.beam.systems import BEAM_TABLE, BeamRecallSystem
from recall.calibration import from_samples, save
from recall.eval.locomo import DEFAULT_DSN

#: BEAM's unanswerable category. Everything else is answerable.
UNANSWERABLE_TYPE = "abstention"


def _parse_indices(spec: str) -> list[int]:
    """Kept as a thin alias; the implementation lives in `dataset` so all seven agree."""
    return parse_conversation_indices(spec)


def _stamp_fit_set(path: Path, indices: list[int]) -> None:
    """Record which conversations a calibration was fitted on, inside the calibration file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fit_on_conversations"] = sorted(indices)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


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
    # REQUIRED. With `default=None`, `recall.calibration.save` resolved to the PROCESS-GLOBAL
    # default (`$RECALL_CALIBRATION` or `./calibration.json`), which `trusted_search`
    # autoloads for every later query started from that directory. So running this probe
    # silently re-tuned the abstention threshold for everything that ran afterwards — and
    # this module's own docstring records that the fit it produces is 0.617, UNCERTIFIED,
    # and abstains MORE than the shipped 0.50.
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the fitted calibration; name it explicitly — "
                             "the default path is autoloaded by every later run")
    parser.add_argument(
        "--save-uncertified", action="store_true",
        help="write the calibration even when certification FAILED (default: refuse)",
    )
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
    certified_ok = bool(cal.certified)  # `None` must never read as True — see calibration.py:200

    # The REPORT is written first, unconditionally. It holds `per_question` — the raw top-cosines
    # for every question across --conversations, i.e. the entire expensive output of the run. A
    # refusal that fired before this point threw that away too, so an uncertified fit produced no
    # calibration, no report and no diagnosis: the run cost the same and left nothing to read.
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
        "calibration_written": certified_ok or args.save_uncertified,
        "calibration_path": str(args.out),
        # Kept so a reader can see the distributions the threshold came from rather than trust
        # the single number it collapsed to.
        "per_question": per_question,
    }
    args.out.with_name(args.out.stem + "_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))

    # Certification is a verdict on whether the fit is USABLE, and it used to be reported after
    # the calibration had already been written — with `--out` defaulting to the process-global
    # path that `trusted_search` autoloads. So the documented outcome of running this probe was
    # that an uncertified threshold silently became the default for every later run. A diagnosis
    # that changes nothing is not a gate.
    if not certified_ok and not args.save_uncertified:
        raise SystemExit(
            f"calibration NOT certified ({cal.certification_reason}); refusing to write "
            f"{args.out}. The report beside it was still written, so nothing measured is lost. "
            f"Pass --save-uncertified if you are deliberately keeping the fit for study."
        )
    path = save(cal, args.out)
    # Persisted INSIDE the calibration, not only in the sidecar report: the out-of-sample
    # contract ("the scored run must exclude these") was help text with nothing able to check it,
    # because the fit set was not recorded anywhere the scoring run could read.
    _stamp_fit_set(path, indices)


if __name__ == "__main__":
    main()
