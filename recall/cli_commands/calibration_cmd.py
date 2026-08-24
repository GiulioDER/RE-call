"""`recall calibrate` and `recall calibration`: fit, inspect and transfer calibrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Eager, NOT to defer a cost: `recall.store` is imported at module level elsewhere in the CLI
# and already pulls in psycopg and pgvector, so nothing would be saved by a function-local
# import. `register` reads these for its help strings and the handlers reuse the same bindings
# rather than importing them again.
from recall.calibration_v2 import (
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    DEFAULT_MAX_CORPUS_DELTA,
)
from recall.drift import DRIFT_SCREEN_DELTA
from recall.setup import CalibrationResult

from recall.cli_commands._shared import _make_embedder


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    # Top-level `calibrate` is the INSTALL-TIME step: it fits the abstention threshold to the
    # operator's own corpus, which is the only thing that makes the shipped default meaningful.
    # The generation-bound measurement is an enterprise operation and lives under `calibration`,
    # beside the artifacts it produces.
    p_cal = sub.add_parser(
        "calibrate",
        help="calibrate the abstention threshold for this embedder against labeled queries",
    )
    p_cal.set_defaults(_opens_db=True, func=_cmd_calibrate)
    p_cal.add_argument("queries", help="JSON list of {query, answerable, relevant_ids} entries")
    p_cal.add_argument(
        "--corpus", default=None, help="corpus dir (default: the built-in eval corpus)"
    )
    p_cal.add_argument("--out", default=None, help="output path (default: calibration.json)")

    p_calibration = sub.add_parser("calibration", help="inspect or transfer calibration artifacts")

    p_calibration.set_defaults(_opens_db=True, func=_cmd_calibration)
    calibration_sub = p_calibration.add_subparsers(dest="calibration_cmd", required=True)
    p_cal_measure = calibration_sub.add_parser(
        "calibrate", help="measure a calibration bound to one immutable generation"
    )
    p_cal_measure.add_argument("--generation", required=True)
    p_cal_measure.add_argument("--queries", dest="query_file", required=True)
    p_cal_measure.add_argument("--publish", action="store_true")

    p_cal_carry = calibration_sub.add_parser(
        "carry-forward",
        help="re-verify a published threshold against a rebuilt generation, without refitting it",
    )
    p_cal_carry.add_argument(
        "--generation", required=True, help="the NEW generation to bind the threshold to"
    )
    p_cal_carry.add_argument(
        "--from",
        dest="parent_calibration_id",
        default=None,
        help="calibration to carry forward (default: this tenant's most recently published one "
        "on another generation)",
    )
    p_cal_carry.add_argument(
        "--max-corpus-delta",
        type=float,
        default=None,
        help=f"refuse if more than this fraction of sources changed (default "
        f"{DEFAULT_MAX_CORPUS_DELTA}). Lower it per tenant; the default is a ceiling on the "
        f"mechanism, not a measured safe distance",
    )
    p_cal_carry.add_argument(
        "--max-error",
        type=float,
        default=None,
        help=f"reject if the inherited threshold misclassifies more than this fraction of either "
        f"labelled class on the new generation (default {DEFAULT_MAX_CARRY_FORWARD_ERROR}). This "
        f"is the check separability cannot make",
    )
    p_cal_carry.add_argument("--publish", action="store_true")

    p_cal_drift = calibration_sub.add_parser(
        "drift",
        help="ask whether the corpus has moved far enough to need recalibrating",
    )
    # Exactly one of the two, enforced by argparse rather than by a runtime check, because the
    # difference is not a preference: a generation can be probed and a directory cannot.
    drift_target = p_cal_drift.add_mutually_exclusive_group(required=True)
    drift_target.add_argument(
        "--generation",
        default=None,
        help="an already-built generation to compare and, unless --no-probe, to re-score against",
    )
    drift_target.add_argument(
        "--path",
        default=None,
        help="a live corpus directory. Compared, never probed: nothing has indexed it yet, so the "
        "strongest verdict this can reach is a recommendation",
    )
    p_cal_drift.add_argument(
        "--glob", default=None, help="glob for --path (default: the indexing default)"
    )
    p_cal_drift.add_argument(
        "--screen-delta",
        type=float,
        default=None,
        help=f"corpus delta below which no probe is spent (default {DRIFT_SCREEN_DELTA}). Low on "
        f"purpose: firing costs one probe, staying quiet costs a threshold that has silently "
        f"stopped deciding, and the smallest measured failure was at a delta of 0.945 "
        f"(docs/preregistrations/2026-08-21-calibration-drift-trigger.md)",
    )
    p_cal_drift.add_argument(
        "--no-probe",
        dest="probe",
        action="store_false",
        help="screen only. Cheap enough for a post-index hook, and it can never reach a REQUIRED "
        "verdict on error, only on delta",
    )
    p_cal_drift.add_argument("--json", action="store_true", help="machine-readable report")
    p_cal_drift.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when recalibration is recommended or required, for use in CI",
    )

    p_cal_auto = calibration_sub.add_parser(
        "auto",
        help="re-establish a certified calibration on a rebuilt generation without asking for "
        "labels: carry the threshold forward, or refit it on the same stored evidence",
    )
    p_cal_auto.add_argument("--generation", required=True)
    p_cal_auto.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help="write the artifact but leave it a draft, so an operator chooses when it serves",
    )
    p_cal_auto.add_argument("--json", action="store_true")

    calibration_sub.add_parser("list")
    p_cal_show = calibration_sub.add_parser("show")
    p_cal_show.add_argument("calibration_id")
    p_cal_export = calibration_sub.add_parser("export")
    p_cal_export.add_argument("calibration_id")
    p_cal_export.add_argument("--output", required=True)
    p_cal_import = calibration_sub.add_parser("import")
    p_cal_import.add_argument("path")


def _cmd_calibration(args: argparse.Namespace) -> None:
    from recall.calibration_v2 import (
        CalibrationError,
        CalibrationRepository,
        load_query_set,
    )

    repository = CalibrationRepository(args.dsn, args.tenant)
    try:
        if args.calibration_cmd == "list":
            print(json.dumps(repository.list_records(), indent=2, default=str))
        elif args.calibration_cmd == "show":
            print(
                json.dumps(repository.show_record(args.calibration_id), indent=2, default=str)
            )
        elif args.calibration_cmd == "export":
            print(repository.export_bundle(args.calibration_id, args.output))
        elif args.calibration_cmd == "import":
            print(repository.import_bundle(args.path))
        elif args.calibration_cmd == "calibrate":
            labels, _digest = load_query_set(args.query_file)
            artifact = repository.calibrate(
                args.generation, labels, _make_embedder(args.embedder)
            )
            if args.publish:
                artifact = repository.publish(artifact.calibration_id)
            print(f"calibration: {artifact.calibration_id}")
            print(f"status: {artifact.status.value}")
            print(f"generation: {artifact.generation_id}")
            print(f"pipeline: {artifact.pipeline_fingerprint}")
            print(f"corpus: {artifact.corpus_fingerprint}")
            print(f"queries: {artifact.query_set_digest}")
        elif args.calibration_cmd == "carry-forward":
            bound = (
                DEFAULT_MAX_CORPUS_DELTA
                if args.max_corpus_delta is None
                else args.max_corpus_delta
            )
            error_bound = (
                DEFAULT_MAX_CARRY_FORWARD_ERROR
                if args.max_error is None
                else args.max_error
            )
            artifact = repository.carry_forward(
                args.generation,
                _make_embedder(args.embedder),
                parent_calibration_id=args.parent_calibration_id,
                max_corpus_delta=bound,
                max_error=error_bound,
            )
            provenance = dict(artifact.carry_forward or {})
            print(f"calibration: {artifact.calibration_id}")
            print(f"status: {artifact.status.value}")
            print(f"generation: {artifact.generation_id}")
            print(f"carried forward from: {provenance.get('parent_calibration_id')}")
            print(
                f"corpus delta: {provenance.get('corpus_delta'):.4f} "
                f"(+{provenance.get('sources_added')} "
                f"-{provenance.get('sources_removed')} "
                f"~{provenance.get('sources_modified')} "
                f"of {provenance.get('sources_union')} sources, bound {bound})"
            )
            print(f"inherited threshold: {artifact.threshold}")
            # Printed next to the inherited number precisely because it is NOT applied. An
            # operator watching these two diverge over a chain of rebuilds has the warning
            # that the next carry-forward is the one that will fail.
            print(f"a refit here would have chosen: {provenance.get('refit_threshold')}")
            print(
                f"separability on this generation: {artifact.separability:.4f} "
                f"CI [{artifact.separability_ci[0]:.4f}, {artifact.separability_ci[1]:.4f}] "
                f"over {artifact.n_answerable} answerable / "
                f"{artifact.n_unanswerable} unanswerable"
            )
            # Printed beside separability because they answer a different question and can
            # disagree with it: an ordering can stay perfect while the fixed cut stops
            # deciding anything. Reading only the AUC is how that gets missed.
            print(
                f"at the inherited threshold: false abstain "
                f"{provenance.get('false_abstain_rate', 0.0):.1%} of {artifact.n_answerable}, "
                f"false confirm {provenance.get('false_confirm_rate', 0.0):.1%} of "
                f"{artifact.n_unanswerable} (bound {error_bound:.1%})"
            )
            if not artifact.certified:
                print(f"NOT certified: {artifact.certification_reason}")
                raise SystemExit(
                    "the inherited threshold no longer certifies on this generation; "
                    "the evidence is retained as a rejected artifact. Recalibrate with "
                    "`recall calibration calibrate`."
                )
            if args.publish:
                artifact = repository.publish(artifact.calibration_id)
                print(f"published: {artifact.calibration_id}")
        elif args.calibration_cmd == "drift":
            from recall.drift import (
                corpus_objects_for_directory,
                evaluate_drift,
            )

            screen = (
                DRIFT_SCREEN_DELTA if args.screen_delta is None else args.screen_delta
            )
            # A FACTORY, not an embedder. `evaluate_drift` calls it only if it reaches the
            # probe, so a delta below the screen (the common case on a live corpus) costs no
            # model load at all: seconds on a warm machine, a download on a cold one.
            probing = args.probe and args.generation is not None
            drift_report = evaluate_drift(
                repository,
                generation_id=args.generation,
                corpus_objects=(
                    corpus_objects_for_directory(args.path, args.glob)
                    if args.path is not None
                    else None
                ),
                candidate_label=args.path or args.generation,
                embedder=(lambda: _make_embedder(args.embedder)) if probing else None,
                screen_delta=screen,
                probe=args.probe,
            )
            print(
                json.dumps(drift_report.to_dict(), indent=2)
                if args.json
                else drift_report.format()
            )
            if args.strict and drift_report.needs_action:
                raise SystemExit(1)
        elif args.calibration_cmd == "auto":
            from recall.drift import auto_recalibrate

            outcome = auto_recalibrate(
                repository,
                args.generation,
                _make_embedder(args.embedder),
                publish=args.publish,
            )
            if args.json:
                print(json.dumps(outcome.to_dict(), indent=2))
            else:
                print(f"action: {outcome.action}")
                print(f"calibration: {outcome.calibration_id or 'none'}")
                print(f"published: {'yes' if outcome.published else 'no'}")
                print(f"reason: {outcome.reason}")
            # `failed` is the only exit-1 case. `skipped` is a correct, expected outcome for a
            # tenant whose first calibration has not been made yet, and exiting non-zero on it
            # would make a post-index hook look broken on every fresh install.
            if outcome.action == "failed":
                raise SystemExit(1)
        else:  # unreachable while argparse constrains the choices, but never guess a default
            raise SystemExit(f"unknown calibration subcommand: {args.calibration_cmd}")
    except CalibrationError as exc:
        raise SystemExit(str(exc)) from exc


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from recall.calibration import ENV_VAR, _resolve_path
    from recall.setup import calibrate_from_files

    embedder = _make_embedder(args.embedder)
    try:
        calibration_result: CalibrationResult = calibrate_from_files(
            dsn=args.dsn,
            embedder_name=embedder.name,
            queries_path=Path(args.queries),
            corpus_dir=Path(args.corpus) if args.corpus else None,
            out=Path(args.out) if args.out else None,
        )
    except ValueError as exc:
        raise SystemExit(2) from exc
    measured = calibration_result.report
    cal = calibration_result.calibration
    path = calibration_result.path
    print(f"embedder:  {embedder.name}")
    print(f"threshold: {cal.threshold} (scale {cal.scale})")
    sep = "n/a" if cal.separability is None else f"{cal.separability:.3f}"
    ci = cal.separability_ci
    # The interval, not just the point, because the bar is applied to its lower bound — a
    # reader who sees only "0.95" cannot reconstruct why a certification failed.
    sep_ci = "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    print(
        f"separability (AUC): {sep}{sep_ci} over {cal.n_answerable} answerable / "
        f"{cal.n_unanswerable} unanswerable"
    )
    print(
        f"FCR at default 0.50: {measured.fcr_at_050:.2f} -> at calibrated: "
        f"{measured.fcr_at_suggested:.2f}"
    )
    print(f"saved: {path}")
    if args.out and Path(args.out).resolve() != _resolve_path(None).resolve():
        print(
            f"note: searches load {_resolve_path(None)} by default — set "
            f"{ENV_VAR}={path} for this file to be used"
        )

    # Exit non-zero on a threshold the data does not support. The file is still written: the
    # artifact records `certified: false` and the reason, and refusing to write would destroy
    # the evidence of WHY. What changes is that a calibration step can now fail — measured on
    # LongMemEval, an uncertified threshold refused 44% of the questions retrieval had just
    # answered correctly, and neither the API nor the file said anything was wrong.
    if cal.certified is False:
        print(f"\nNOT CERTIFIED: {cal.certification_reason}", file=sys.stderr)
        print(
            "Saved anyway — there is no better threshold for this data — but abstention on "
            "this corpus is not trustworthy. Do NOT read an abstention as evidence that the "
            "answer is absent.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if cal.certified is None:
        print(f"\nnot judged: {cal.certification_reason}", file=sys.stderr)
