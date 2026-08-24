"""`recall generation`: manage immutable blue green generations."""

from __future__ import annotations

import argparse
import functools
import sys
from typing import get_args

from recall.embeddings import Embedder
from recall.index import (
    ChunkerKind,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
)

from recall.cli_commands._shared import _make_embedder, _positive_int


def _non_negative_int(value: str) -> int:
    """A count where zero is meaningful but a negative one is not.

    `--overlap` is the case: 0 means "no shared context between adjacent pieces", which is a real
    choice, while a negative value is written verbatim into an immutable lineage record describing
    a pipeline nothing can have run.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"cannot be negative, got {number}")
    return number


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_generation = sub.add_parser("generation", help="manage immutable blue green generations")

    p_generation.set_defaults(_opens_db=True, func=_cmd_generation)
    generation_sub = p_generation.add_subparsers(dest="generation_cmd", required=True)
    p_build = generation_sub.add_parser("build", help="create and build a generation")
    p_build.add_argument("manifest")
    p_build.add_argument("--manifest-version-id")
    p_build.add_argument("--manifest-sha256")
    p_build.add_argument("--manifest-size", type=int)
    p_build.add_argument("--embedder-provider", default=None)
    p_build.add_argument("--embedder-revision", default=None)
    p_build.add_argument("--embedder-artifact-digest", default=None)
    p_build.add_argument("--unverified-development", action="store_true")
    p_build.add_argument(
        "--project",
        default=None,
        help="stamp every chunk with the project that produced it, as `recall index --project` "
             "does. A calibrated generation without it cannot say where a hit came from.",
    )
    p_build.add_argument(
        "--no-commit-stamp",
        action="store_true",
        help="do not record the repository HEAD on each chunk.",
    )
    # `choices` read off the same `Literal` the builder validates against, rather than repeated
    # here. This was the last of four hand-written copies of the vocabulary, and it is the one at
    # the gate users actually reach, so a chunker added to `ChunkerKind` would have been accepted
    # by `BuildRequest` while this still exited 2.
    p_build.add_argument(
        "--chunker", choices=list(get_args(ChunkerKind)), default="text"
    )
    # `_positive_int`, not bare `int`. `--max-chars 0` reached `manager.create`, wrote the
    # generation row, and only then failed inside `build` on the text path — while on the code path
    # it did not fail at all and recorded `{"max_chars": 0}` for one unsplit chunk.
    p_build.add_argument("--max-chars", type=_positive_int, default=DEFAULT_MAX_CHARS)
    p_build.add_argument("--overlap", type=_non_negative_int, default=DEFAULT_OVERLAP_CHARS)
    p_validate = generation_sub.add_parser("validate", help="validate a built generation")
    p_validate.add_argument("generation_id")
    p_promote = generation_sub.add_parser("promote", help="promote a ready generation")
    p_promote.add_argument("generation_id")
    p_promote.add_argument("--unsafe-development-promotion", action="store_true")
    # The reclaim route for a generation that built and validated and was then never promoted.
    # Without it such a generation was unreachable: `fail` refuses READY, and `gc` collects only
    # `retired` and `failed`, so its full copy of the corpus's chunk rows could never be released.
    # Every uncertified wizard build lands there, which is the ordinary first-install outcome.
    p_abandon = generation_sub.add_parser(
        "abandon",
        help="release a ready-but-unpromoted generation so `gc` can reclaim its rows",
    )
    p_abandon.add_argument("generation_id")
    p_abandon.add_argument(
        "--reason", default="abandoned by operator", help="recorded on the generation row"
    )
    generation_sub.add_parser("rollback", help="atomically restore the previous generation")
    generation_sub.add_parser("list", help="list immutable generation history")
    p_gc = generation_sub.add_parser("gc", help="collect expired retired generations")
    p_gc.add_argument("--retention-days", type=int, default=7)
    p_gc.add_argument("--retain-previous", type=int, default=2)


def _report_drift_after_build(args: argparse.Namespace, generation_id: str) -> None:
    """Say what the new generation did to the calibration serving this tenant, per policy.

    Runs after a build reports success and **can never turn that success into a failure**. A build
    that wrote every chunk correctly has succeeded whether or not the advisory afterwards can reach
    a model, a query set or a database, so every failure here is logged and swallowed. The opposite
    choice would make a corpus unbuildable because a monitor was misconfigured.

    `RECALL_AUTO_CALIBRATE` decides how far this goes:

    * `off` does nothing at all, including opening a connection;
    * `warn`, the default, reports and leaves the decision to the operator;
    * `auto` re-establishes the calibration by carrying the threshold forward, or refitting it on
      the same stored labelled evidence when the threshold has to move.

    ⚠️ **The probe runs here, and the screen alone would not be enough.** A build is exactly the
    moment the new generation IS indexed, so the decisive check is available for the price of one
    retrieval per labelled query; declining to spend that would report a screen result at the one
    moment a real answer was cheap.
    """
    from recall.drift import AutoCalibrationMode

    try:
        mode = AutoCalibrationMode.from_env()
    except ValueError as exc:
        print(f"drift: {exc}", file=sys.stderr)
        return
    if mode is AutoCalibrationMode.OFF:
        return

    from recall.calibration_v2 import CalibrationRepository
    from recall.drift import auto_recalibrate, evaluate_drift

    # Memoised, and passed as a FACTORY rather than as an embedder. Two properties at once:
    # `evaluate_drift` calls it only if it reaches the probe, so a build whose delta is under the
    # screen loads no model at all; and when both the probe and the automatic recalibration below
    # need one, they get the same instance rather than two loads of the same weights.
    @functools.cache
    def build_embedder() -> Embedder:
        return _make_embedder(args.embedder)

    repository = CalibrationRepository(args.dsn, args.tenant)
    try:
        report = evaluate_drift(
            repository,
            generation_id=generation_id,
            embedder=build_embedder,
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring: advice never fails a build
        print(f"drift: could not measure drift for {generation_id}: {exc}", file=sys.stderr)
        return
    print()
    print(report.format())
    if mode is AutoCalibrationMode.WARN or not report.needs_action:
        if report.needs_action:
            print()
            print(
                "RECALL_AUTO_CALIBRATE=auto would re-establish this automatically; "
                f"or run `recall --tenant {args.tenant} calibration auto "
                f"--generation {generation_id}`"
            )
        return
    try:
        outcome = auto_recalibrate(repository, generation_id, build_embedder())
    except Exception as exc:  # noqa: BLE001 - as above
        print(f"drift: automatic recalibration failed for {generation_id}: {exc}", file=sys.stderr)
        return
    print()
    print(f"auto-calibrate: {outcome.action} ({outcome.reason})")


def _cmd_generation(args: argparse.Namespace) -> None:
    from recall.generations import GenerationManager

    manager = GenerationManager(args.dsn, args.tenant)
    if args.generation_cmd == "list":
        for generation in manager.list_generations():
            print(
                f"{generation.generation_id} {generation.state.value:<18} "
                f"pipeline={generation.pipeline_fingerprint} "
                f"corpus={generation.corpus_fingerprint}"
            )
        return
    if args.generation_cmd == "rollback":
        print(f"active generation: {manager.rollback()}")
        return
    if args.generation_cmd == "gc":
        collected = manager.gc(
            retention_days=args.retention_days,
            retain_previous=args.retain_previous,
        )
        print(f"collected {len(collected)} generation(s): {', '.join(collected) or '(none)'}")
        return
    if args.generation_cmd == "validate":
        generation_validation = manager.validate(args.generation_id)
        print(
            f"ready {generation_validation.generation_id}: "
            f"{generation_validation.sources} sources, "
            f"{generation_validation.chunks} chunks"
        )
        return
    if args.generation_cmd == "promote":
        manager.promote(
            args.generation_id,
            unsafe_development=args.unsafe_development_promotion,
        )
        print(f"active generation: {args.generation_id}")
        return
    if args.generation_cmd == "abandon":
        manager.abandon(args.generation_id, args.reason)
        print(f"abandoned {args.generation_id}; `recall generation gc` can now reclaim it")
        return

    from recall.generation_build import BuildRequest, build_generation
    from recall.lineage import IndexManifestV1, ManifestObjectV1
    from recall.manifest import (
        ExtractingS3ObjectReader,
        ObjectReader,
        S3ObjectReader,
        load_manifest,
        reader_for_manifest,
    )

    environment = manager.environment
    # The reader is chosen AFTER the manifest is known, not before. Building the S3 reader
    # up front needs boto3 and RECALL_S3_ALLOWLIST, so a local-only user hit an S3
    # configuration error while doing nothing that involved S3.
    reader: ObjectReader | None = None
    if args.manifest.startswith("s3://"):
        if (
            args.manifest_version_id is None
            or args.manifest_sha256 is None
            or args.manifest_size is None
        ):
            raise SystemExit(
                "an S3 manifest requires --manifest-version-id, --manifest-sha256 and "
                "--manifest-size"
            )
        reference = ManifestObjectV1(
            args.manifest,
            args.manifest_version_id,
            "application/json",
            args.manifest_size,
            args.manifest_sha256,
        )
        # An s3:// manifest needs the S3 reader to fetch the manifest itself.
        base_reader = S3ObjectReader.from_environment()
        manifest = IndexManifestV1.from_json(base_reader.fetch(reference).data)
        reader = ExtractingS3ObjectReader(base_reader)
    else:
        if environment == "production":
            raise SystemExit("production generation builds require a versioned S3 manifest")
        manifest = load_manifest(args.manifest)
    if reader is None:
        reader = reader_for_manifest(manifest)
    embedder = _make_embedder(args.embedder)
    # The assembly itself lives in `recall.generation_build`, because the installation wizard
    # builds generations too and a second copy of it would mean two provenance vocabularies
    # drifting apart with nothing failing. The strings it writes are pinned by
    # `tests/test_generation_build_assembly.py`.
    generation_stats = build_generation(
        manager,
        manifest,
        reader,
        embedder,
        BuildRequest(
            chunker=args.chunker,
            max_chars=args.max_chars,
            overlap=args.overlap,
            provider=args.embedder_provider,
            revision=args.embedder_revision,
            artifact_digest=args.embedder_artifact_digest,
            unverified=args.unverified_development,
            # Same provenance the index path stamps. Without this a CALIBRATED generation
            # carries no record of which project produced each chunk, and the generation path
            # is the only one calibration can use.
            project=args.project,
            # `"."` rather than the corpus: a manifest names objects, not a working tree, and
            # an s3:// one has no local root at all. This is the pre-existing behaviour and is
            # NOT the same root `recall index` uses, which stamps the directory being indexed.
            commit_root=None if args.no_commit_stamp else ".",
        ),
    )
    print(
        f"built {generation_stats.generation_id}: {generation_stats.objects} objects, "
        f"{generation_stats.chunks} chunks, {generation_stats.reused_objects} objects "
        f"reused; run `recall generation validate {generation_stats.generation_id}`"
    )
    _report_drift_after_build(args, generation_stats.generation_id)
