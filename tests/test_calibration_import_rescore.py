"""Import authenticates a bundle on the operator's labels, and the lite fit refuses a repeated query.

SEC-002 of the 2026-10-31 hunt, and the audit of its first fix: a bundle's checksums are unkeyed,
its statistics re-derive from scores it supplies, and its labels are whatever its author chose. So
an import now takes the operator's own labelled set (`labels`, digest checked against the
bundle's), scores it on the named generation, requires the recorded scores to reproduce within
`_RESCORE_TOLERANCE`, and requires a certified bundle to certify on those FRESH scores. A carried
bundle is judged at an error bound no looser than the default, whatever it recorded.

The import tests need no PostgreSQL: the generation lookup and the scorer are replaced, and the
fake connection fails the test if anything is INSERTed, so the property they pin is that a forged
bundle is refused before it is stored. The carried-bound tests call
`_require_scores_support_certification` directly, the lite test indexes a temporary lite store,
and the CLI test replaces the repository.

Red proof (the project rule). Two baselines: master 79156bab (before any re-score), and the first
re-score fix that this file's audit reviewed (the CCA snapshot of `recall/calibration_v2.py`,
labelled "first fix" below). Each test's docstring names the baseline or mutation it failed on and
how. Tests named `..._still_...` are controls: they show a fix did not refuse what it must accept.
The carried-bound control passes on master too; the others call code that exists only now.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import recall.calibration_v2 as calibration_v2
from recall.calibration import Calibration, from_samples
from recall.calibration_v2 import (
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    CalibrationArtifactV2,
    CalibrationBindingError,
    CalibrationRepository,
    _require_scores_reproduce,
    _require_scores_support_certification,
    canonical_query_set,
    canonical_sha256,
)

PIPELINE = "a" * 64
CORPUS = "b" * 64
IDENTITY = {"model": "m"}
N = 20
ANSWERABLE = [0.80 + 0.001 * i for i in range(N)]
UNANSWERABLE = [0.20 + 0.001 * i for i in range(N)]


def _raw_labels() -> list[dict[str, Any]]:
    raw = [{"query": f"answer {i}", "answerable": True} for i in range(N)]
    return raw + [{"query": f"missing {i}", "answerable": False} for i in range(N)]


LABELS = _raw_labels()
DIGEST = canonical_query_set(LABELS)[1]


def _sealed(artifact: CalibrationArtifactV2) -> CalibrationArtifactV2:
    return replace(artifact, checksum=canonical_sha256(artifact.immutable_payload()))


def _fitted(answerable: list[float], unanswerable: list[float]) -> CalibrationArtifactV2:
    """An artifact exactly as `calibrate` would write it from these scores."""
    runtime = from_samples("m", answerable, unanswerable)
    assert runtime.separability is not None and runtime.separability_ci is not None
    certified = runtime.certified is True
    return _sealed(
        CalibrationArtifactV2(
            calibration_id="cal_rescore",
            tenant_id="rescore-tenant",
            generation_id="gen_rescore",
            embedder_identity=IDENTITY,
            pipeline_fingerprint=PIPELINE,
            corpus_fingerprint=CORPUS,
            query_set_digest=DIGEST,
            threshold=runtime.threshold,
            scale=runtime.scale,
            separability=runtime.separability,
            separability_ci=runtime.separability_ci,
            n_answerable=len(answerable),
            n_unanswerable=len(unanswerable),
            certified=certified,
            certification_reason=runtime.certification_reason,
            lifecycle_state="draft" if certified else "rejected",
            created_at=datetime.now(UTC).isoformat(),
            created_by="pytest",
            scores={"answerable": answerable, "unanswerable": unanswerable},
            checksum="0" * 64,
        )
    )


def _carried(threshold: float, recorded_bound: Any) -> CalibrationArtifactV2:
    """A certified carried artifact over the honest separable scores, inheriting `threshold`."""
    check = Calibration(
        embedder="m",
        threshold=threshold,
        scale=0.05,
        separability=1.0,
        n_answerable=N,
        n_unanswerable=N,
    )
    assert check.separability_ci is not None
    return _sealed(
        replace(
            _fitted(ANSWERABLE, UNANSWERABLE),
            calibration_id="cal_carried",
            threshold=threshold,
            scale=0.05,
            separability=1.0,
            separability_ci=check.separability_ci,
            certification_reason="certified",
            carry_forward={
                "parent_calibration_id": "cal_parent",
                "parent_generation_id": "gen_parent",
                "corpus_delta": 0.05,
                "inherited_threshold": threshold,
                "inherited_scale": 0.05,
                "max_carry_forward_error": recorded_bound,
            },
        )
    )


def _bundle(
    path: Path, artifact: CalibrationArtifactV2, query_set: list[dict[str, Any]] | None = None
) -> Path:
    payload: dict[str, Any] = {"artifact": artifact.to_dict(), "query_set": query_set or LABELS}
    payload["bundle_checksum"] = canonical_sha256(dict(payload))
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Result:
    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class _NoInsert:
    def execute(self, sql: str, params: Any = None) -> _Result:
        if "INSERT" in sql:
            raise AssertionError("stored a bundle the generation does not support")
        return _Result()

    def transaction(self):  # noqa: ANN201
        return nullcontext()


def _repository(monkeypatch, measured: tuple[list[float], list[float]]) -> CalibrationRepository:
    """A repository whose generation matches the bundle and whose scorer returns `measured`."""
    repository = CalibrationRepository(
        "postgresql://unused.invalid/none", "rescore-tenant", actor="pytest"
    )

    @contextmanager
    def _connect():
        yield _NoInsert()

    repository._connect = _connect  # type: ignore[method-assign]
    lineage = calibration_v2._GenerationLineage({"embedder": IDENTITY}, PIPELINE, CORPUS)
    repository._generation = lambda _conn, _gid, lock=False: lineage  # type: ignore[method-assign]
    repository.score_query_set = lambda _gid, _embedder, _labels: measured  # type: ignore[method-assign]
    monkeypatch.setattr(calibration_v2.PipelineIdentity, "from_dict", staticmethod(lambda raw: raw))
    monkeypatch.setattr(calibration_v2, "_require_runtime_embedder", lambda _p, _e: None)
    return repository


def _import(repository: CalibrationRepository, path: Path, **kwargs: Any) -> str:
    kwargs.setdefault("labels", LABELS)
    return repository.import_bundle(path, SimpleNamespace(), **kwargs)  # type: ignore[arg-type]


# The labels are the operator's ----------------------------------------------------------------


def test_a_v2_bundle_without_the_operators_labels_is_refused(monkeypatch, tmp_path: Path) -> None:
    """SEC-001 of the audit: re-scoring checks scores, but the bundle chose the labels.

    Red: on the first fix, `import_bundle(path, embedder)` re-scored the bundle's own labels and
    reached the INSERT (`AssertionError: stored a bundle ...`). Target: the `labels is None` guard.
    """
    repository = _repository(monkeypatch, (ANSWERABLE, UNANSWERABLE))
    path = _bundle(tmp_path / "b.json", _fitted(ANSWERABLE, UNANSWERABLE))
    with pytest.raises(CalibrationBindingError, match="needs your own labelled query set"):
        repository.import_bundle(path, SimpleNamespace())  # type: ignore[arg-type]


def test_labels_other_than_the_bundles_are_refused(monkeypatch, tmp_path: Path) -> None:
    """A forger relabels real queries; the operator's own set no longer matches the bundle's.

    Red: mutation of `if operator_digest != digest:` to `if False:` reached the INSERT
    (`AssertionError: stored a bundle ...`).
    """
    relabelled = [dict(item, answerable=not item["answerable"]) for item in LABELS]
    repository = _repository(monkeypatch, (ANSWERABLE, UNANSWERABLE))
    forged = _sealed(
        replace(
            _fitted(ANSWERABLE, UNANSWERABLE), query_set_digest=canonical_query_set(relabelled)[1]
        )
    )
    path = _bundle(tmp_path / "relabelled.json", forged, relabelled)
    with pytest.raises(CalibrationBindingError, match="is not the one this bundle was fitted on"):
        _import(repository, path)


def test_operator_labels_that_repeat_a_query_are_refused(monkeypatch, tmp_path: Path) -> None:
    """SEC-004: one query twice re-scores identically, so the reproduce check cannot see it.

    Red: mutation removing `require_distinct_scored_queries(operator_labels)` from
    `import_bundle` reached the INSERT (`AssertionError: stored a bundle ...`).
    """
    repeated = [dict(item, relevant_ids=[str(n)]) for n, item in enumerate(LABELS[:1] * 2)]
    repeated += LABELS[1:]
    digest = canonical_query_set(repeated)[1]
    repository = _repository(monkeypatch, (ANSWERABLE, UNANSWERABLE))
    path = _bundle(
        tmp_path / "repeated.json",
        _sealed(replace(_fitted(ANSWERABLE, UNANSWERABLE), query_set_digest=digest)),
        repeated,
    )
    with pytest.raises(CalibrationBindingError, match="repeats the query 'answer 0'"):
        _import(repository, path, labels=repeated)


# The scores are this generation's ---------------------------------------------------------------


def test_a_bundle_whose_scores_this_generation_does_not_measure_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    """Honest-looking, internally perfect scores that are not this corpus's.

    Red: on master the import had no re-score and reached the INSERT; mutation removing the
    `_require_scores_reproduce(...)` call did the same (`AssertionError: stored a bundle ...`).
    """
    forged = _fitted([0.95] * N, [0.90] * N)
    repository = _repository(monkeypatch, (ANSWERABLE, UNANSWERABLE))
    path = _bundle(tmp_path / "forged.json", forged)
    with pytest.raises(CalibrationBindingError, match="not what this generation measures"):
        _import(repository, path)


def test_certification_is_judged_on_the_fresh_scores(monkeypatch, tmp_path: Path) -> None:
    """NUM-001: within 0.01 per query, overlapping fresh classes can be recorded separated.

    Fresh scores overlap (separability about 0.80, uncertified); shifting each by 0.0099 records
    them perfectly separated, which re-derives a certified fit and passes the per-query check.
    Red: on the first fix this reached the INSERT (`AssertionError: stored a bundle ...`).
    Target: `_require_fresh_certification`.
    """
    fresh_a = [0.600 + 0.001 * i for i in range(40)]
    fresh_u = [0.585 + 0.001 * i for i in range(40)]
    labels = [{"query": f"a{i}", "answerable": True} for i in range(40)]
    labels += [{"query": f"u{i}", "answerable": False} for i in range(40)]
    digest = canonical_query_set(labels)[1]
    recorded = _fitted([s + 0.0099 for s in fresh_a], [s - 0.0099 for s in fresh_u])
    assert recorded.certified, "the forged recording must certify on its own scores"
    assert from_samples("m", fresh_a, fresh_u).certified is not True, "the fresh scores must not"
    repository = _repository(monkeypatch, (fresh_a, fresh_u))
    path = _bundle(tmp_path / "shifted.json", _sealed(replace(recorded, query_set_digest=digest)), labels)
    with pytest.raises(CalibrationBindingError, match="do not earn"):
        _import(repository, path, labels=labels)


def test_a_v2_bundle_without_an_embedder_is_refused(monkeypatch, tmp_path: Path) -> None:
    """No embedder means no re-score, and an unscored v2 bundle is unauthenticated.

    Red: mutation removing the `if embedder is None: raise ...` guard reached the scorer stub and
    the INSERT (`AssertionError: stored a bundle ...`).
    """
    repository = _repository(monkeypatch, (ANSWERABLE, UNANSWERABLE))
    path = _bundle(tmp_path / "b.json", _fitted(ANSWERABLE, UNANSWERABLE))
    with pytest.raises(CalibrationBindingError, match="needs that generation's embedder"):
        repository.import_bundle(path, labels=LABELS)


def test_an_honest_bundle_still_reaches_the_insert(monkeypatch, tmp_path: Path) -> None:
    """Control: the operator's labels, scores re-measured within tolerance, certified on both."""
    jittered = ([s + 0.004 for s in ANSWERABLE], [s - 0.004 for s in UNANSWERABLE])
    repository = _repository(monkeypatch, jittered)
    path = _bundle(tmp_path / "honest.json", _fitted(ANSWERABLE, UNANSWERABLE))
    with pytest.raises(AssertionError, match="stored a bundle"):
        _import(repository, path)


@pytest.mark.parametrize(
    ("measured", "message"),
    [
        (([0.8] * (N - 1), UNANSWERABLE), "re-scores to 19"),
        (([s + 0.02 for s in ANSWERABLE], UNANSWERABLE), "largest difference is 0.0200"),
    ],
    ids=["count", "drift"],
)
def test_require_scores_reproduce_names_how_the_scores_disagree(measured, message) -> None:
    """Red: mutation of the comparison to `worst > 1.0` let the drift case through
    (`DID NOT RAISE`); mutation dropping the length check let the count case fail inside
    `zip(strict=True)` with a raw ValueError instead of the named refusal."""
    with pytest.raises(CalibrationBindingError, match=message):
        _require_scores_reproduce(_fitted(ANSWERABLE, UNANSWERABLE), *measured)


@pytest.mark.parametrize("position", [0, 7], ids=["first", "later"])
def test_a_non_finite_fresh_score_is_refused(position: int) -> None:
    """NUM-003: `max()` over differences and `> tol` both pass a NaN through.

    A NaN in the first position made the whole class's largest difference NaN, and `nan > 0.01`
    is False; a NaN later was skipped by `max()`. Either way that query went unchecked. Fresh
    scores come from this generation (a zero-norm query vector is one documented source), so a
    NaN is refused by name rather than compared.
    Red: on the first fix (PR 1095 at e213d6ec0) both cases passed the check (`DID NOT RAISE`).
    """
    fresh = list(ANSWERABLE)
    fresh[position] = float("nan")
    with pytest.raises(CalibrationBindingError, match="non-finite"):
        _require_scores_reproduce(_fitted(ANSWERABLE, UNANSWERABLE), fresh, list(UNANSWERABLE))


def test_require_scores_reproduce_still_accepts_embedding_noise() -> None:
    """Control: the measured run-to-run movement of a hosted model (up to 0.0051) passes."""
    _require_scores_reproduce(
        _fitted(ANSWERABLE, UNANSWERABLE),
        [s + 0.0051 for s in ANSWERABLE],
        [s - 0.0051 for s in UNANSWERABLE],
    )


# The embedder is built only when it is needed ----------------------------------------------------


def test_a_legacy_import_never_builds_an_embedder(monkeypatch, tmp_path: Path) -> None:
    """BUG-001: the CLI built `--embedder` for every import, so an offline legacy import failed.

    Red: on the first fix `_cmd_calibration` called `_make_embedder(args.embedder)` before
    `import_bundle`, so the stub below raised `AssertionError: built an embedder for a legacy file`.
    Target: the `make_embedder=` factory in the CLI's import branch.
    """
    from recall.cli_commands import calibration_cmd

    def _no_embedder(_spec: str) -> None:
        raise AssertionError("built an embedder for a legacy file")

    seen: dict[str, Any] = {}

    class _Repository:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def import_bundle(self, path: str, *args: Any, **kwargs: Any) -> str:
            seen.update(path=path, args=args, kwargs=kwargs)
            return "legacy_cal_x"

    monkeypatch.setattr(calibration_cmd, "_make_embedder", _no_embedder)
    monkeypatch.setattr(calibration_v2, "CalibrationRepository", _Repository)
    legacy = tmp_path / "calibration.json"
    legacy.write_text(json.dumps({"threshold": 0.5, "scale": 0.05}), encoding="utf-8")
    args = argparse.Namespace(
        dsn="postgresql://unused.invalid/none",
        tenant="t",
        calibration_cmd="import",
        path=str(legacy),
        labels_file=None,
        embedder="fastembed",
    )
    calibration_cmd._cmd_calibration(args)
    assert seen["path"] == str(legacy)
    assert callable(seen["kwargs"]["make_embedder"]), "the CLI must pass a factory, not an embedder"


# A carried bundle is judged at no looser than the default bound ------------------------------


def test_a_carried_bundle_cannot_widen_its_own_error_bound() -> None:
    """A recorded bound of 1.0 would certify any inherited threshold over honest scores.

    0.81 sits inside the answerable class (scores 0.800 to 0.819), so it abstains on about half
    of the answerable queries: far past the 10% default, inside the 100% the bundle recorded.
    Red: on master `_require_scores_support_certification` accepted it (`DID NOT RAISE`).
    """
    with pytest.raises(CalibrationBindingError, match="never looser than the default 10%"):
        _require_scores_support_certification(_carried(0.81, 1.0))


def test_a_carried_bundle_bound_outside_a_fraction_is_refused() -> None:
    """Red: master accepted a recorded bound of 5.0 (`DID NOT RAISE`)."""
    with pytest.raises(CalibrationBindingError, match=r"fraction in \[0, 1\]"):
        _require_scores_support_certification(_carried(0.5, 5.0))


def test_a_carried_bundle_within_the_default_still_passes() -> None:
    """Control: a threshold between the classes decides both within the default bound."""
    _require_scores_support_certification(_carried(0.5, DEFAULT_MAX_CARRY_FORWARD_ERROR))


# Lite parity: the lite fit refuses one query counted twice ----------------------------------


def test_the_lite_fit_refuses_a_query_repeated_with_other_relevant_ids(tmp_path: Path) -> None:
    """The PostgreSQL fit refuses this since #1091; the lite fit did not.

    Red: on master `auto_calibrate` scored the set and returned an outcome (`DID NOT RAISE`).
    Target: `require_distinct_scored_queries` in `recall.lite.calibration.auto_calibrate`.
    """
    from recall.lite.calibration import auto_calibrate
    from tests.test_lite_calibration import ContentWordEmbedder, _store

    store = _store(tmp_path, 3)
    queries = [
        {"query": "same question", "answerable": True, "relevant_ids": ["a"]},
        {"query": "same question", "answerable": True, "relevant_ids": ["b"]},
        {"query": "another question", "answerable": False},
    ]
    with pytest.raises(CalibrationBindingError, match="repeats the query 'same question'"):
        auto_calibrate(store, ContentWordEmbedder(), queries=queries)
