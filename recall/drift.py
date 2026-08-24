"""Watch a live corpus against the calibration that is serving it, and say when to refit.

Everything else in this package answers the drift question **after** a rebuild.
`CalibrationRepository.resolve` compares fingerprints and returns `STALE` on any mismatch, which is
a yes/no about identity rather than a statement about magnitude; `carry_forward` re-verifies a
threshold against a generation that already exists. Neither can be asked "the corpus on disk has
moved on, does the threshold serving it still decide anything?", which is the question an operator
has between rebuilds and the one this module answers.

**Two tiers, because the cheap signal cannot decide on its own.** That is measured, not assumed:
`docs/preregistrations/2026-08-21-calibration-drift-trigger.md` registers the prediction and records
what the delta-to-error curve actually looks like over three corpora.

* **The screen** is `corpus_delta`, a manifest comparison over `(uri, sha256)` pairs. No embedding,
  no retrieval, no database beyond one row. It is an upper bound on how much *could* have moved,
  never an estimate of how much did: a top-1 cosine is a max over the index, so a change only moves
  a query's score when it lands near that query.
* **The probe** replays the calibration's own stored labelled query set against the index and
  measures what the frozen threshold now costs, per class. This is the tier that decides, and it is
  the same evidence and the same rule `carry_forward` applies.

⛔ **The screen firing is not a verdict, and this module never reports it as one.** When the probe
cannot run, the report says `RECALIBRATE_RECOMMENDED` and names the check that was not made. A
screen promoted to a verdict is how a guard starts crying wolf, and a guard nobody believes is worse
than no guard (`memory/a-guard-that-cries-wolf-is-worse-than-none.md`).

⚠️ **Separability is deliberately not the outcome.** It is threshold-free, so a corpus change that
lifts every unanswerable score by the same amount leaves AUC at 1.00 while sliding the whole class
over a threshold that is not allowed to move. The outcome here is the per-class error of the fixed
cut, each rate over its own denominator. See `memory/separability-cannot-see-a-shifted-class.md`.

## Why no delta, however large, is a verdict on its own

An earlier draft of this module reported `RECALIBRATE_REQUIRED` once the delta passed
`DEFAULT_MAX_CORPUS_DELTA`, on the reasoning that past that point the labelled query set describes a
corpus that no longer exists. That reasoning is intuitive and **the measurement contradicts it**.

Measured 2026-08-21 over 57 snapshots of three real corpus histories:

- The frozen threshold first went over `DEFAULT_MAX_CARRY_FORWARD_ERROR` at a delta of **0.945**,
  and never below it.
- A delta-only rule at 0.25 fires on **56 of 57** snapshots and is right about **5**, a precision of
  **0.09**. It would have demanded recalibration on twenty consecutive states of this repository's
  `docs/` where the threshold was measurably fine, several of them with a LOWER error than the day
  it was fitted.
- The labels proved far more durable than the argument assumed. At delta 0.981 only **27.5%** of the
  answerable queries' original evidence chunks still existed, and the false-abstain rate was
  **0.025**.
- What moved was the **false-confirm** rate, and it tracked corpus GROWTH rather than change as
  such: Spearman 0.95 against growth on `docs`, where growth and delta are collinear at 0.98, so
  this measurement cannot separate the two. A top-1 cosine is a max over the index, so added
  documents can only raise an unanswerable query's score. How much of the corpus was *rewritten*
  predicts nothing about that.

So the probe decides, always, wherever a probe can run. Where one cannot, the strongest verdict
reachable is `RECALIBRATE_RECOMMENDED`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict, cast

from recall.calibration import MIN_SEPARABILITY, Calibration, from_samples, separability
from recall.calibration_v2 import (
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    DEFAULT_MAX_CORPUS_DELTA,
    CalibrationArtifactV2,
    CalibrationError,
    CalibrationRepository,
    CalibrationStatus,
    corpus_delta,
    threshold_error_rates,
)
from recall.embeddings import Embedder
from recall.observability import get_logger

_log = get_logger("drift")

__all__ = [
    "DRIFT_SCREEN_DELTA",
    "AutoCalibrationMode",
    "DriftReport",
    "DriftVerdict",
    "auto_recalibrate",
    "corpus_objects_for_directory",
    "evaluate_drift",
]


#: Corpus delta below which no probe is spent and nothing is reported.
#:
#: **A cost decision with a measured margin, not a tuned optimum, and the difference matters.**
#: Measured 2026-08-21 over 57 snapshots of three real corpus histories
#: (`docs/preregistrations/2026-08-21-calibration-drift-trigger.md`,
#: `results/calibration_drift_2026-08-21.json`): the frozen threshold first went over
#: `DEFAULT_MAX_CARRY_FORWARD_ERROR` at a delta of **0.945**, and never below it. So this screen
#: sits roughly nineteen times below the smallest failure anyone has observed.
#:
#: It is low on purpose. Firing costs one probe, which is a retrieval per labelled query and takes
#: seconds. Staying quiet costs an operator a threshold that has silently stopped deciding, and
#: that failure is invisible by construction. When the two errors are that asymmetric, the screen
#: belongs near zero and the probe does the judging.
#:
#: ⚠️ It is a SCREEN and not a verdict. Below it this module is quiet because a probe was not worth
#: spending, never because the threshold has been shown to be fine.
#:
#: Re-measure, from your own worktree:
#:
#:     python -m benchmarks.calibration_drift --out results/calibration_drift.json
#:     python -m benchmarks.calibration_drift --analyze results/calibration_drift.json
DRIFT_SCREEN_DELTA = 0.05


class DriftVerdict(StrEnum):
    """What an operator should do, in a vocabulary a script can branch on."""

    #: The corpus has not moved far enough to be worth a probe, or the probe says the threshold
    #: still decides. Not a promise; see `DRIFT_SCREEN_DELTA`.
    STABLE = "stable"
    #: The screen fired and the decisive check could not be made. Named separately from REQUIRED
    #: because the remedy differs: run the probe, or recalibrate if you cannot.
    RECALIBRATE_RECOMMENDED = "recalibrate_recommended"
    #: **Measured.** The probe ran and the frozen threshold's error is over the bound, or the
    #: classes have stopped separating. Only ever reached through the probe: the module
    #: docstring records the measurement that removed the delta-only route.
    RECALIBRATE_REQUIRED = "recalibrate_required"
    #: There is nothing to compare against. A missing calibration is not low drift.
    UNKNOWN = "unknown"


class AutoCalibrationMode(StrEnum):
    """How far a deployment lets drift monitoring act on its own.

    `WARN` is the default rather than `AUTO`, and rather than `OFF`, for two different reasons.
    Not `AUTO`, because recalibration republishes the artifact every query is judged against and
    that is an operator's decision on any corpus they did not personally curate. Not `OFF`, because
    a threshold that has stopped deciding fails silently, which is the whole failure mode this
    module exists to end.
    """

    OFF = "off"
    WARN = "warn"
    AUTO = "auto"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AutoCalibrationMode":
        raw = (environ if environ is not None else os.environ).get("RECALL_AUTO_CALIBRATE", "")
        value = raw.strip().lower()
        if not value:
            return cls.WARN
        try:
            return cls(value)
        except ValueError:
            # Refused rather than defaulted. Silently reading `RECALL_AUTO_CALIBRATE=true` as OFF
            # would leave an operator who asked for automation with none, and reading it as AUTO
            # would republish artifacts on the strength of a typo.
            raise ValueError(
                f"RECALL_AUTO_CALIBRATE={raw!r} is not one of "
                f"{', '.join(mode.value for mode in cls)}"
            ) from None


class _CommonFields(TypedDict):
    """The fields every verdict below carries, spelled once.

    A `TypedDict` rather than a plain dict because `**common` on a plain one collapses to
    `dict[str, object]` and every field of every `DriftReport` then reads as a type error. Spelled
    here, the splat is checked: adding a field to `DriftReport` without adding it here fails the
    type check rather than failing at the first caller who hits that branch.
    """

    tenant_id: str
    calibration_id: str | None
    baseline_generation_id: str | None
    candidate: str
    delta: Mapping[str, Any]
    screen_delta: float
    max_error: float


@dataclass(frozen=True)
class DriftReport:
    """One drift measurement, with every number it was decided on."""

    tenant_id: str
    verdict: DriftVerdict
    reason: str
    #: What the live corpus was compared AGAINST: the calibration serving this tenant.
    calibration_id: str | None
    baseline_generation_id: str | None
    #: What was compared: a generation id, or a directory path.
    candidate: str
    #: `corpus_delta` output, or empty when there was nothing to compare.
    delta: Mapping[str, Any]
    #: The probe's per-class error and separability, or None when it was not run. `probe is None`
    #: and `probe` reporting no error are different states and are never collapsed.
    probe: Mapping[str, Any] | None = None
    screen_delta: float = DRIFT_SCREEN_DELTA
    max_error: float = DEFAULT_MAX_CARRY_FORWARD_ERROR

    @property
    def needs_action(self) -> bool:
        return self.verdict in {
            DriftVerdict.RECALIBRATE_RECOMMENDED,
            DriftVerdict.RECALIBRATE_REQUIRED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "calibration_id": self.calibration_id,
            "baseline_generation_id": self.baseline_generation_id,
            "candidate": self.candidate,
            "delta": dict(self.delta),
            "probe": dict(self.probe) if self.probe is not None else None,
            "screen_delta": self.screen_delta,
            "max_error": self.max_error,
        }

    def format(self) -> str:
        """One operator-readable block. The numbers first, the advice last."""
        lines = [
            f"tenant:      {self.tenant_id}",
            f"calibration: {self.calibration_id or 'none'}",
            f"baseline:    {self.baseline_generation_id or 'none'}",
            f"candidate:   {self.candidate}",
        ]
        if self.delta:
            lines.append(
                f"delta:       {self.delta.get('corpus_delta', float('nan')):.4f} "
                f"({self.delta.get('sources_added', 0)} added, "
                f"{self.delta.get('sources_removed', 0)} removed, "
                f"{self.delta.get('sources_modified', 0)} modified "
                f"of {self.delta.get('sources_union', 0)} sources)"
            )
        if self.probe is not None:
            lines.append(
                f"probe:       false abstain "
                f"{self.probe['false_abstain_rate']:.1%} of {self.probe['n_answerable']} "
                f"answerable, false confirm {self.probe['false_confirm_rate']:.1%} of "
                f"{self.probe['n_unanswerable']} unanswerable, bound {self.max_error:.1%}"
            )
            # `separability` is guaranteed non-None by `_probe`, which refuses a one-class result
            # outright; formatted defensively anyway, because a `None` reaching a `.4f` here would
            # turn a monitor into a crash at the moment it had something to report.
            auc = self.probe.get("separability")
            rendered = f"{auc:.4f}" if isinstance(auc, (int, float)) else "unknown"
            lines.append(
                f"             separability {rendered}, refit would be "
                f"{self.probe['refit_threshold']:.4f} against the "
                f"{self.probe['threshold']:.4f} in force"
            )
        else:
            lines.append("probe:       not run")
        lines.append(f"verdict:     {self.verdict.value.upper()}")
        lines.append(f"             {self.reason}")
        return "\n".join(lines)


def corpus_objects_for_directory(root: str | Path, glob: str | None = None) -> list[dict[str, Any]]:
    """Inventory a live directory into the `(uri, sha256)` objects `corpus_delta` compares.

    The walk is `recall.wizard.inventory.build_inventory`, which is `recall.index.candidate_files`,
    which is the walk `index_path` performs. Inheriting it rather than writing a third one is what
    keeps a drift measurement about the corpus that would actually be indexed: a private walk here
    could report a delta over files indexing would never read, and the number would look fine.
    """
    from recall.lint import DEFAULT_GLOB
    from recall.wizard.inventory import build_inventory

    return build_inventory(root, glob or DEFAULT_GLOB)


def _published_calibration(repository: CalibrationRepository) -> CalibrationArtifactV2 | None:
    """The artifact currently serving this tenant, or None.

    Newest published wins, which is the same order `carry_forward` picks a parent by, so the
    calibration this module reports drift for is the one that mechanism would inherit from.
    """
    for record in repository.list_records():
        if str(record.get("lifecycle_state")) != "published":
            continue
        try:
            artifact = repository.get(str(record["calibration_id"]))
        except CalibrationError:  # pragma: no cover - a row that cannot be loaded is not a baseline
            continue
        if artifact.status is CalibrationStatus.CERTIFIED:
            return artifact
    return None


def _probe(
    repository: CalibrationRepository,
    artifact: CalibrationArtifactV2,
    generation_id: str,
    embedder: Embedder,
    *,
    max_error: float,
) -> dict[str, Any]:
    """Re-score the calibration's own labelled evidence and judge the FROZEN threshold on it.

    Deliberately the same evidence, the same sampling rule and the same two conditions
    `carry_forward` applies, reached through the same methods. A monitor that judged by a different
    rule than the gate would tell an operator to act and then be refused, or stay quiet about a
    state the gate would reject.
    """
    labels, digest = repository.stored_query_set(artifact.query_set_digest)
    if digest != artifact.query_set_digest:
        raise CalibrationError(
            "stored labelled query set no longer matches its digest, so it cannot be re-scored"
        )
    answerable, unanswerable = repository.score_query_set(generation_id, embedder, labels)
    if not answerable or not unanswerable:
        # Not a drift finding. Both classes are needed for every number below, and a one-class
        # result means the evidence or the index is wrong rather than that the corpus moved.
        # Raised rather than reported as a clean zero, which is what a missing class would become
        # if it were allowed to flow into `threshold_error_rates` (whose rates are 0.0 for an
        # empty class, by its own contract).
        raise CalibrationError(
            f"re-scoring returned {len(answerable)} answerable and {len(unanswerable)} "
            f"unanswerable scores; both classes are required to judge a threshold, so this is a "
            f"broken index or a broken query set rather than a measurement of drift"
        )
    errors = threshold_error_rates(answerable, unanswerable, artifact.threshold)
    frozen = Calibration(
        embedder=artifact.runtime.embedder,
        threshold=artifact.threshold,
        scale=artifact.scale,
        separability=separability(answerable, unanswerable),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
    )
    refit = from_samples(artifact.runtime.embedder, answerable, unanswerable)
    ci = frozen.separability_ci
    return {
        "threshold": artifact.threshold,
        "refit_threshold": refit.threshold,
        "threshold_drift": refit.threshold - artifact.threshold,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "separability": frozen.separability,
        "separability_ci": list(ci) if ci is not None else None,
        "still_certified": frozen.certified is True,
        "max_error": max(errors["false_abstain_rate"], errors["false_confirm_rate"]),
        "within_error": (
            errors["false_abstain_rate"] <= max_error
            and errors["false_confirm_rate"] <= max_error
        ),
        **errors,
    }


def evaluate_drift(
    repository: CalibrationRepository,
    *,
    generation_id: str | None = None,
    corpus_objects: Sequence[Mapping[str, Any]] | None = None,
    candidate_label: str | None = None,
    #: An embedder, or a callable that builds one. The callable form exists so the SCREEN really
    #: is free: loading model weights costs seconds on a warm machine and a download on a cold
    #: one, and the common case by far is a delta below the screen where no probe is run at all.
    #: Passing an already-built embedder is still fine and is what a caller that has one should do.
    embedder: "Embedder | Callable[[], Embedder] | None" = None,
    screen_delta: float = DRIFT_SCREEN_DELTA,
    max_error: float = DEFAULT_MAX_CARRY_FORWARD_ERROR,
    probe: bool = True,
) -> DriftReport:
    """Compare a corpus against the calibration serving this tenant and say what to do.

    Exactly one of `generation_id` and `corpus_objects` names the candidate corpus. A generation can
    be probed, because its chunks are in the index; a directory cannot, because nothing has embedded
    it yet. That asymmetry is the whole reason RECOMMENDED exists as a separate verdict: a corpus
    nobody has scored cannot produce a measurement, and **no delta is large enough to stand in for
    one**; see the module docstring for the numbers behind that.

    `probe=False` forces the screen-only path even where a probe was possible, which is what a
    post-index hook wants on a corpus that is rebuilt continuously: the probe costs one retrieval
    per labelled query and the screen costs a manifest comparison.
    """
    if (generation_id is None) == (corpus_objects is None):
        raise ValueError("name exactly one of generation_id or corpus_objects")

    tenant = repository.tenant_id
    candidate = candidate_label or generation_id or "<directory>"
    artifact = _published_calibration(repository)
    if artifact is None:
        return DriftReport(
            tenant_id=tenant,
            verdict=DriftVerdict.UNKNOWN,
            reason=(
                "no certified published calibration for this tenant, so there is no threshold to "
                "measure drift against. A missing calibration is not low drift: calibrate first."
            ),
            calibration_id=None,
            baseline_generation_id=None,
            candidate=candidate,
            delta={},
            screen_delta=screen_delta,
            max_error=max_error,
        )

    baseline = repository.manifest_objects_for(artifact.generation_id)
    candidate_objects = (
        list(corpus_objects)
        if corpus_objects is not None
        else repository.manifest_objects_for(str(generation_id))
    )
    delta = corpus_delta(baseline, candidate_objects)
    magnitude = float(delta["corpus_delta"])
    common: _CommonFields = {
        "tenant_id": tenant,
        "calibration_id": artifact.calibration_id,
        "baseline_generation_id": artifact.generation_id,
        "candidate": candidate,
        "delta": delta,
        "screen_delta": screen_delta,
        "max_error": max_error,
    }

    if magnitude == 0.0:
        return DriftReport(
            verdict=DriftVerdict.STABLE,
            reason=(
                f"the corpus is byte-identical to the {delta['sources_parent']} sources "
                f"calibration {artifact.calibration_id} was fitted on"
            ),
            **common,
        )

    if magnitude < screen_delta:
        return DriftReport(
            verdict=DriftVerdict.STABLE,
            reason=(
                f"corpus delta {magnitude:.3f} is below the {screen_delta:.3f} screen, so the "
                f"probe is not worth spending. This is a screen, not a clean bill of health: it "
                f"bounds how much COULD have moved, never how much did."
            ),
            **common,
        )

    if not probe or embedder is None or generation_id is None:
        # Ordered by which reason is the ROOT one. A directory reaches here with no embedder built,
        # because building one to probe something unindexed would be wasted work, so an
        # embedder-first test would report the symptom and send the operator to supply a model
        # that could not have helped.
        missing = (
            "a directory has no index to score against, so only a rebuilt generation can be probed"
            if generation_id is None
            else "probing was disabled"
            if not probe
            else "no embedder was supplied"
        )
        return DriftReport(
            verdict=DriftVerdict.RECALIBRATE_RECOMMENDED,
            reason=(
                f"corpus delta {magnitude:.3f} is over the {screen_delta:.3f} screen and the "
                f"decisive check was not made: {missing}. The screen bounds how much could have "
                f"moved, so this is a reason to look, not a measurement that the threshold failed."
            ),
            **common,
        )

    # Resolved HERE and nowhere earlier: every return above this line is a verdict reached without
    # a model, which is the property that makes running this after every rebuild affordable.
    #
    # Discriminated by `embed`, the one method `Embedder` is defined by, rather than by
    # `callable()`. No embedder in the tree defines `__call__` today, so both tests agree; but a
    # `callable()` test says "this is not a factory" by elimination, and the day somebody adds
    # `__call__` to an embedder it would silently start invoking the model as though it built one.
    # Asking for the attribute that makes it an embedder cannot go wrong that way.
    resolved = cast("Embedder", embedder) if hasattr(embedder, "embed") else embedder()
    try:
        measured = _probe(repository, artifact, generation_id, resolved, max_error=max_error)
    except CalibrationError as exc:
        # The decisive check could not be made, which is a RECOMMENDATION and never a verdict of
        # its own. Reported rather than raised because the caller is usually a monitor: a stored
        # query set that has gone missing is exactly the state an operator needs told about, and
        # a traceback out of a post-build advisory would instead read as a broken build.
        return DriftReport(
            verdict=DriftVerdict.RECALIBRATE_RECOMMENDED,
            reason=(
                f"corpus delta {magnitude:.3f} is over the {screen_delta:.3f} screen and the "
                f"probe could not run: {exc}"
            ),
            **common,
        )
    if not measured["within_error"]:
        return DriftReport(
            verdict=DriftVerdict.RECALIBRATE_REQUIRED,
            reason=(
                f"the threshold {artifact.threshold:.4f} in force no longer decides this corpus: "
                f"false abstain {measured['false_abstain_rate']:.1%} of "
                f"{measured['n_answerable']} answerable, false confirm "
                f"{measured['false_confirm_rate']:.1%} of {measured['n_unanswerable']} "
                f"unanswerable, against a bound of {max_error:.1%}. A refit would place it at "
                f"{measured['refit_threshold']:.4f}."
            ),
            probe=measured,
            **common,
        )
    if not measured["still_certified"]:
        return DriftReport(
            verdict=DriftVerdict.RECALIBRATE_REQUIRED,
            reason=(
                f"the threshold still decides within the error bound, but the classes have stopped "
                f"separating: the 95% lower bound on separability is below {MIN_SEPARABILITY}. No "
                f"threshold separates them, so moving this one only trades one error for the other."
            ),
            probe=measured,
            **common,
        )
    return DriftReport(
        verdict=DriftVerdict.STABLE,
        reason=(
            f"corpus delta {magnitude:.3f} cleared the screen and the probe cleared the bound: "
            f"false abstain {measured['false_abstain_rate']:.1%}, false confirm "
            f"{measured['false_confirm_rate']:.1%}, both at or under {max_error:.1%}. A refit "
            f"would move the threshold by {measured['threshold_drift']:+.4f}."
        ),
        probe=measured,
        **common,
    )


@dataclass(frozen=True)
class AutoCalibrationOutcome:
    """What automatic recalibration did, and which of the two paths it took."""

    #: `carried_forward`, `recalibrated`, `skipped`, or `failed`.
    action: str
    calibration_id: str | None
    reason: str
    published: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "calibration_id": self.calibration_id,
            "reason": self.reason,
            "published": self.published,
        }


def auto_recalibrate(
    repository: CalibrationRepository,
    generation_id: str,
    embedder: Embedder,
    *,
    publish: bool = True,
    max_corpus_delta: float = DEFAULT_MAX_CORPUS_DELTA,
) -> AutoCalibrationOutcome:
    """Re-establish a certified calibration on `generation_id` without asking a human for labels.

    Two paths, cheapest first, and **neither loosens certification**. Both end in an artifact that
    had to clear the same bar a hand-driven `recall calibration calibrate` clears, so the automation
    is in what gets run, never in what gets accepted.

    1. **`carry_forward`.** Re-scores the parent's stored evidence and inherits the threshold only
       if the frozen cut still holds on the new scores. Cheapest, and it keeps the operating point
       an operator has already seen.
    2. **A fresh fit against that same stored labelled set.** Reached when carry-forward is refused,
       which is exactly the case where the threshold needs to MOVE rather than be re-verified.

    ⚠️ **`max_corpus_delta` bounds only how far path 1 will try, and path 2 is NOT bounded by it.**
    An earlier draft of this docstring claimed the opposite: that past that delta the stored
    questions describe a corpus that no longer exists, so a refit would be fitting to stale
    evidence. The measurement contradicts that, and the code never implemented it. At a delta of
    0.981 on this repository's `docs/`, only 27.5% of the answerable queries' original evidence
    chunks still existed and the false-abstain rate was 0.025: the questions kept working long
    after the specific text that first answered them was gone. Certification still gates the refit,
    so a set that genuinely has stopped describing the corpus is refused by the usual bar rather
    than by a delta nobody measured.

    ⛔ **What neither path does is invent questions.** Both reuse the stored labelled set. Producing
    a fresh one is `recall.wizard.queryset`, and doing that unattended on a corpus nobody has looked
    at is a different decision from re-running an existing measurement.
    """
    artifact = _published_calibration(repository)
    if artifact is None:
        return AutoCalibrationOutcome(
            action="skipped",
            calibration_id=None,
            reason=(
                "no certified published calibration to carry forward from; the first calibration "
                "on a corpus is a human decision about what the labelled questions should be"
            ),
        )
    if artifact.generation_id == generation_id:
        return AutoCalibrationOutcome(
            action="skipped",
            calibration_id=artifact.calibration_id,
            reason=f"calibration {artifact.calibration_id} is already bound to {generation_id}",
        )

    try:
        carried = repository.carry_forward(
            generation_id,
            embedder,
            parent_calibration_id=artifact.calibration_id,
            max_corpus_delta=max_corpus_delta,
        )
    except CalibrationError as exc:
        # `carried` is deliberately left unbound here rather than set to None. It is only ever read
        # inside the `else` branch, where the call succeeded, so a None binding would widen its
        # type for no reader and make every later attribute access look unsafe.
        carry_refusal = str(exc)
    else:
        carry_refusal = ""
        if carried.certified:
            if publish:
                carried = repository.publish(carried.calibration_id)
            return AutoCalibrationOutcome(
                action="carried_forward",
                calibration_id=carried.calibration_id,
                reason=(
                    f"the threshold {artifact.threshold:.4f} still holds on this generation's "
                    f"fresh scores, so it was inherited rather than refitted"
                ),
                published=publish,
            )
        carry_refusal = carried.certification_reason

    # The threshold has to move. Refit on the SAME labelled evidence, which is what makes this
    # unattended: no new questions are invented, and the fit is judged by the usual certification.
    _log.info("carry-forward did not certify (%s); refitting on the stored query set", carry_refusal)
    try:
        labels, _digest = repository.stored_query_set(artifact.query_set_digest)
        fitted = repository.calibrate(generation_id, labels, embedder)
    except CalibrationError as exc:
        return AutoCalibrationOutcome(
            action="failed",
            calibration_id=None,
            reason=(
                f"carry-forward was refused ({carry_refusal or 'see log'}) and the refit could not "
                f"be made either: {exc}"
            ),
        )
    if not fitted.certified:
        return AutoCalibrationOutcome(
            action="failed",
            calibration_id=fitted.calibration_id,
            reason=(
                f"a fresh fit on the stored labelled set does not certify either: "
                f"{fitted.certification_reason}. The corpus needs labelled questions drawn from "
                f"what it is now, which is not a decision this should make unattended."
            ),
        )
    if publish:
        fitted = repository.publish(fitted.calibration_id)
    return AutoCalibrationOutcome(
        action="recalibrated",
        calibration_id=fitted.calibration_id,
        reason=(
            f"carry-forward was refused ({carry_refusal or 'threshold no longer held'}), so the "
            f"threshold was refitted on the same labelled evidence and moved from "
            f"{artifact.threshold:.4f} to {fitted.threshold:.4f}"
        ),
        published=publish,
    )
