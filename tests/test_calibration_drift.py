"""Drift monitoring says what to do about a corpus that has moved, and never overstates it.

Three distinctions carry the whole feature, and each has a test whose failure would collapse it
into something that looks the same and is not.

* **A missing calibration is not low drift.** `test_no_calibration_is_unknown_not_stable`.
* **The screen is not a verdict.** When the decisive probe cannot run, the strongest thing that may
  be said is RECOMMENDED. `test_a_directory_can_only_ever_be_recommended` and
  `test_probe_disabled_says_so_rather_than_reporting_stable`.
* **Separability cannot see a shifted class.** A corpus change can leave the classes perfectly
  ordered, certification passing, and the fixed cut deciding nothing.
  `test_required_when_the_frozen_threshold_stops_deciding` is that case, and it asserts AUC is
  still 1.00 so a future reader can tell the fixture still reproduces the hole.

The fixtures are the carry-forward ones, imported rather than copied: drift monitoring judges by
the same rule the carry-forward gate enforces, and two fixture sets would let the monitor and the
gate drift apart without anything failing.
"""

from __future__ import annotations

import json

import pytest

from recall.calibration_v2 import (
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    CalibrationRepository,
    CalibrationStatus,
)
from recall.drift import (
    DRIFT_SCREEN_DELTA,
    AutoCalibrationMode,
    DriftReport,
    DriftVerdict,
    auto_recalibrate,
    corpus_objects_for_directory,
    evaluate_drift,
)
from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import (  # noqa: F401 - `carry_tenant` is a fixture
    CORPUS_SIZE,
    _bodies,
    _CarryEmbedder,
    _labels,
    _ready,
    carry_tenant,
)

# --------------------------------------------------------------------------------------------
# The mode, which decides how far anything is allowed to act on its own.
# --------------------------------------------------------------------------------------------


def test_the_default_mode_is_warn_not_off_and_not_auto() -> None:
    """Both other defaults are wrong in a way that is invisible until it matters.

    `OFF` restores the silent failure the module exists to end: a threshold that has stopped
    deciding looks exactly like one that is fine. `AUTO` republishes the artifact every query is
    judged against, on a corpus nobody has looked at, without being asked.
    """
    assert AutoCalibrationMode.from_env({}) is AutoCalibrationMode.WARN
    assert AutoCalibrationMode.from_env({"RECALL_AUTO_CALIBRATE": ""}) is AutoCalibrationMode.WARN
    assert AutoCalibrationMode.from_env({"RECALL_AUTO_CALIBRATE": " AUTO "}) is (
        AutoCalibrationMode.AUTO
    )
    assert AutoCalibrationMode.from_env({"RECALL_AUTO_CALIBRATE": "off"}) is (
        AutoCalibrationMode.OFF
    )


def test_an_unrecognised_mode_is_refused_rather_than_guessed() -> None:
    """`true` is the value somebody who wanted automation would actually write.

    Reading it as OFF leaves them with no automation and no error; reading it as AUTO republishes
    artifacts on the strength of a typo. Neither is recoverable by the operator, because neither
    says anything.
    """
    with pytest.raises(ValueError, match="RECALL_AUTO_CALIBRATE"):
        AutoCalibrationMode.from_env({"RECALL_AUTO_CALIBRATE": "true"})


# --------------------------------------------------------------------------------------------
# The inventory, which is what a live directory is compared through.
# --------------------------------------------------------------------------------------------


def test_a_directory_inventories_into_objects_the_delta_can_compare(tmp_path) -> None:
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    objects = corpus_objects_for_directory(tmp_path)
    assert {entry["uri"].rsplit("/", 1)[-1] for entry in objects} == {"a.md", "b.md"}
    # Every entry carries the digest `corpus_delta` compares on. Without it the delta silently
    # becomes a comparison of filenames, which reports an in-place edit as no change at all.
    assert all(len(entry["sha256"]) == 64 for entry in objects)


def test_editing_a_file_in_place_changes_its_inventory_digest(tmp_path) -> None:
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    before = corpus_objects_for_directory(tmp_path)
    (tmp_path / "a.md").write_text("alpha revised", encoding="utf-8")
    after = corpus_objects_for_directory(tmp_path)
    assert before[0]["uri"] == after[0]["uri"]
    assert before[0]["sha256"] != after[0]["sha256"]


# --------------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (DriftVerdict.STABLE, False),
        (DriftVerdict.UNKNOWN, False),
        (DriftVerdict.RECALIBRATE_RECOMMENDED, True),
        (DriftVerdict.RECALIBRATE_REQUIRED, True),
    ],
)
def test_needs_action_covers_both_recalibrate_verdicts(verdict, expected) -> None:
    """UNKNOWN is deliberately NOT an action.

    It means nothing could be measured, usually because the tenant has never been calibrated, and
    a monitor that shouted "recalibrate" at every uncalibrated tenant would be shouting at the one
    population that already knows.
    """
    report = DriftReport(
        tenant_id="t",
        verdict=verdict,
        reason="",
        calibration_id=None,
        baseline_generation_id=None,
        candidate="c",
        delta={},
    )
    assert report.needs_action is expected


def test_a_report_without_a_probe_says_so_rather_than_printing_zeroes() -> None:
    report = DriftReport(
        tenant_id="t",
        verdict=DriftVerdict.STABLE,
        reason="below the screen",
        calibration_id="cal_1",
        baseline_generation_id="gen_1",
        candidate="gen_2",
        delta={
            "corpus_delta": 0.01,
            "sources_added": 1,
            "sources_removed": 0,
            "sources_modified": 0,
            "sources_union": 100,
        },
    )
    text = report.format()
    assert "probe:       not run" in text
    assert report.to_dict()["probe"] is None


# --------------------------------------------------------------------------------------------
# The decision, against a real generation and a real calibration.
# --------------------------------------------------------------------------------------------


@requires_db
def test_no_calibration_is_unknown_not_stable(carry_tenant) -> None:  # noqa: F811
    """The distinction this whole module rests on.

    An uncalibrated tenant has zero measured drift by every arithmetic definition, and reporting
    STABLE would be a monitor telling an operator that a threshold which does not exist is holding
    up fine.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    generation = _ready(manager, embedder, _bodies(), "v1")
    report = evaluate_drift(
        CalibrationRepository(TEST_DSN, tenant, actor="pytest"),
        generation_id=generation,
        embedder=embedder,
    )
    assert report.verdict is DriftVerdict.UNKNOWN
    assert report.needs_action is False
    assert "not low drift" in report.reason
    assert report.probe is None


@requires_db
def test_an_unchanged_corpus_is_stable_without_spending_a_probe(carry_tenant) -> None:  # noqa: F811
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    generation = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(generation, _labels(), embedder).calibration_id)

    report = evaluate_drift(repository, generation_id=generation, embedder=embedder)
    assert report.verdict is DriftVerdict.STABLE
    assert report.delta["corpus_delta"] == 0.0
    # Not merely "stable": nothing was embedded to establish it. A monitor that probes an identical
    # corpus is spending a retrieval per labelled query on every no-op rebuild.
    assert report.probe is None


@requires_db
def test_a_change_below_the_screen_stays_quiet_and_says_it_is_a_screen(carry_tenant) -> None:  # noqa: F811, E501
    """One file of twenty is 0.048, just under the 0.05 screen.

    The reason string has to disclaim, because "STABLE" on its own would be read as a measurement
    that the threshold was checked and held, and nothing was checked.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    parent = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent, _labels(), embedder).calibration_id)

    child = _ready(manager, embedder, _bodies(added=1), "v2")
    report = evaluate_drift(repository, generation_id=child, embedder=embedder)
    assert report.delta["corpus_delta"] == pytest.approx(1 / (CORPUS_SIZE + 1))
    assert report.delta["corpus_delta"] < DRIFT_SCREEN_DELTA
    assert report.verdict is DriftVerdict.STABLE
    assert report.probe is None
    assert "screen, not a clean bill of health" in report.reason


@requires_db
def test_a_change_over_the_screen_is_probed_and_can_come_back_stable(carry_tenant) -> None:  # noqa: F811, E501
    """The case that makes the screen a screen rather than a trigger.

    Two files changed clears the screen, so the probe runs; the threshold still decides, so the
    verdict is STABLE **with the numbers behind it**. A design that reported RECALIBRATE on delta
    alone would send an operator to refit a threshold that is working.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    parent = _ready(manager, embedder, _bodies(), "v1")
    published = repository.publish(
        repository.calibrate(parent, _labels(), embedder).calibration_id
    )

    child = _ready(manager, embedder, _bodies(changed=2), "v2")
    report = evaluate_drift(repository, generation_id=child, embedder=embedder)
    assert report.delta["corpus_delta"] >= DRIFT_SCREEN_DELTA
    assert report.verdict is DriftVerdict.STABLE
    assert report.probe is not None
    assert report.probe["threshold"] == published.threshold
    assert report.probe["within_error"] is True
    assert report.probe["n_answerable"] == 22 and report.probe["n_unanswerable"] == 22


@requires_db
def test_required_when_the_frozen_threshold_stops_deciding(carry_tenant) -> None:  # noqa: F811
    """Separability perfect, certification passing, and the cut deciding nothing.

    ⚠️ If the separability assertion below ever fails, the fixture has stopped reproducing the
    hole and this test is no longer evidence for the check it guards. That is worth more than the
    verdict assertion, because a broken fixture and a working monitor look identical from the
    verdict alone.
    """
    tenant, manager = carry_tenant
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    clean = _CarryEmbedder()
    parent = _ready(manager, clean, _bodies(), "v1")
    repository.publish(repository.calibrate(parent, _labels(), clean).calibration_id)

    poisoned = _CarryEmbedder(poisoned=True)
    child = _ready(manager, poisoned, _bodies(added=2), "v2")
    report = evaluate_drift(repository, generation_id=child, embedder=poisoned)

    assert report.verdict is DriftVerdict.RECALIBRATE_REQUIRED
    assert report.probe is not None
    assert report.probe["separability"] == pytest.approx(1.0)
    assert report.probe["still_certified"] is True
    assert report.probe["false_confirm_rate"] > DEFAULT_MAX_CARRY_FORWARD_ERROR
    assert report.probe["false_abstain_rate"] == pytest.approx(0.0)
    assert "no longer decides this corpus" in report.reason


@requires_db
def test_a_huge_delta_is_still_probed_rather_than_condemned_on_its_size(carry_tenant) -> None:  # noqa: F811, E501
    """Every source rewritten, and the verdict is still whatever the probe says.

    This test asserts the ABSENCE of a rule an earlier draft had: RECALIBRATE_REQUIRED once the
    delta passed `DEFAULT_MAX_CORPUS_DELTA`. Measured over 38 snapshots of two real corpus
    histories, that rule fires on 37 and is right about 4, and the frozen threshold's error did not
    cross the bound below a delta of 0.945. Condemning a corpus on its delta is a guard crying wolf
    nine times in ten.

    If this ever starts returning REQUIRED with `probe is None`, a delta-only route has come back.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    parent = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent, _labels(), embedder).calibration_id)

    child = _ready(manager, embedder, _bodies(changed=CORPUS_SIZE), "v2")
    report = evaluate_drift(repository, generation_id=child, embedder=embedder)
    assert report.delta["corpus_delta"] == pytest.approx(1.0)
    assert report.probe is not None, "a probeable generation must be probed at any delta"
    assert report.verdict is DriftVerdict.STABLE
    assert report.probe["within_error"] is True


@requires_db
def test_a_directory_can_only_ever_be_recommended(carry_tenant, tmp_path) -> None:  # noqa: F811
    """Nothing has indexed a directory, so the decisive check does not exist for it.

    The temptation is to let a large directory delta reach REQUIRED, since the delta is the same
    number. It must not: the delta is an upper bound on how much could have moved, and promoting it
    to a measurement is exactly how a screen becomes a guard that cries wolf.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    parent = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent, _labels(), embedder).calibration_id)

    for index in range(5):
        (tmp_path / f"unrelated-{index}.md").write_text(f"something else {index}", encoding="utf-8")
    report = evaluate_drift(
        repository,
        corpus_objects=corpus_objects_for_directory(tmp_path),
        candidate_label=str(tmp_path),
        embedder=embedder,
    )
    # Nothing in common with the calibrated corpus, so the delta is TOTAL: the largest number this
    # signal can produce, on a corpus that shares not one source with the calibrated one. Even here
    # the verdict is a recommendation, because nothing has scored it.
    assert report.delta["corpus_delta"] == pytest.approx(1.0)
    assert report.verdict is DriftVerdict.RECALIBRATE_RECOMMENDED
    assert report.probe is None
    assert "no index to score against" in report.reason


@requires_db
def test_probe_disabled_says_so_rather_than_reporting_stable(carry_tenant) -> None:  # noqa: F811
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    parent = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent, _labels(), embedder).calibration_id)

    child = _ready(manager, embedder, _bodies(changed=2), "v2")
    report = evaluate_drift(repository, generation_id=child, embedder=embedder, probe=False)
    assert report.verdict is DriftVerdict.RECALIBRATE_RECOMMENDED
    assert report.probe is None
    assert "probing was disabled" in report.reason
    # And the same corpus WITH the probe comes back stable, which is the pair that shows the
    # recommendation is about the missing check rather than about the corpus.
    assert (
        evaluate_drift(repository, generation_id=child, embedder=embedder).verdict
        is DriftVerdict.STABLE
    )


def test_naming_both_or_neither_candidate_is_refused() -> None:
    """A caller that names both has a bug that would otherwise resolve silently to one of them."""
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_drift(object(), generation_id="g", corpus_objects=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_drift(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Acting on it.
# --------------------------------------------------------------------------------------------


@requires_db
def test_auto_carries_the_threshold_forward_when_it_still_holds(carry_tenant) -> None:  # noqa: F811
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    parent = _ready(manager, embedder, _bodies(), "v1")
    published = repository.publish(
        repository.calibrate(parent, _labels(), embedder).calibration_id
    )

    child = _ready(manager, embedder, _bodies(changed=2), "v2")
    outcome = auto_recalibrate(repository, child, embedder)
    assert outcome.action == "carried_forward"
    assert outcome.published is True
    # The cheap path keeps the operating point an operator has already seen, which is the reason to
    # try it first rather than refitting unconditionally.
    carried = repository.get(str(outcome.calibration_id))
    assert carried.threshold == published.threshold
    assert repository.resolve(child).status is CalibrationStatus.CERTIFIED


@requires_db
def test_auto_refits_on_the_same_evidence_when_carry_forward_is_refused(carry_tenant) -> None:  # noqa: F811, E501
    """The path that makes this automation rather than a retry.

    The poisoned corpus is exactly the case carry-forward must refuse, and it is also the case
    where a refit is the right answer: the labels are still about this corpus, only the boundary
    has moved. So the threshold moves, on the same stored evidence, with no human asked for labels.
    """
    tenant, manager = carry_tenant
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    clean = _CarryEmbedder()
    parent = _ready(manager, clean, _bodies(), "v1")
    published = repository.publish(
        repository.calibrate(parent, _labels(), clean).calibration_id
    )

    poisoned = _CarryEmbedder(poisoned=True)
    child = _ready(manager, poisoned, _bodies(added=2), "v2")
    outcome = auto_recalibrate(repository, child, poisoned)
    assert outcome.action == "recalibrated"
    refitted = repository.get(str(outcome.calibration_id))
    assert refitted.certified is True
    assert refitted.threshold != published.threshold
    assert refitted.query_set_digest == published.query_set_digest
    assert repository.resolve(child).status is CalibrationStatus.CERTIFIED


@requires_db
def test_auto_skips_rather_than_inventing_a_first_calibration(carry_tenant) -> None:  # noqa: F811
    """The first calibration on a corpus is a decision about what the questions should be.

    `skipped` and not `failed`, because a fresh install has no calibration by definition and a
    post-build hook exiting non-zero on every fresh install is a hook that gets removed.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    generation = _ready(manager, embedder, _bodies(), "v1")
    outcome = auto_recalibrate(
        CalibrationRepository(TEST_DSN, tenant, actor="pytest"), generation, embedder
    )
    assert outcome.action == "skipped"
    assert outcome.calibration_id is None
    assert "human decision" in outcome.reason


# --------------------------------------------------------------------------------------------
# The CLI wiring. The tests above exercise the decision; these exercise the dispatch, which is
# where a feature that works is still unreachable.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def _cli_embedder(monkeypatch):
    """Make `--embedder` resolve to the fixture embedder these tests build generations with.

    Only the model is stubbed. Argparse, the dispatch branch, the report rendering and the exit
    codes are the real ones, which is the half that the library tests cannot reach: a correct
    `evaluate_drift` behind a mutually-exclusive group that refuses both arguments is a feature
    nobody can run.
    """
    from recall.cli_commands import calibration_cmd

    monkeypatch.setattr(calibration_cmd, "_make_embedder", lambda _name: _CarryEmbedder())
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")


def test_cli_drift_refuses_both_a_generation_and_a_path(capsys) -> None:
    """Argparse enforces it, so the refusal costs no database and no model.

    Named as a test because the mutual exclusion is a design decision, not a formality: a
    generation can be probed and a directory cannot, so accepting both would mean silently
    choosing which question was asked.
    """
    from recall.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--tenant", "t", "calibration", "drift", "--generation", "g", "--path", "."])
    assert exit_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


@requires_db
def test_cli_drift_reports_a_directory_and_stays_quiet_under_strict(
    carry_tenant, tmp_path, capsys, _cli_embedder  # noqa: F811
) -> None:
    """An uncalibrated tenant: UNKNOWN, and `--strict` must NOT exit 1 on it.

    The exit code is the assertion that matters. `--strict` is for CI, and a fresh tenant has no
    calibration by definition; failing the build there would train everyone to drop the flag.
    """
    from recall.cli import main

    tenant, _manager = carry_tenant
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    main(["--tenant", tenant, "--dsn", TEST_DSN, "calibration", "drift",
          "--path", str(tmp_path), "--strict"])
    out = capsys.readouterr().out
    assert "verdict:     UNKNOWN" in out
    assert "not low drift" in out


@requires_db
def test_cli_drift_probes_a_generation_and_exits_1_under_strict(
    carry_tenant, capsys, _cli_embedder  # noqa: F811
) -> None:
    """The whole path end to end: build, calibrate, publish, poison, ask the CLI."""
    from recall.cli import main

    tenant, manager = carry_tenant
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    clean = _CarryEmbedder()
    parent = _ready(manager, clean, _bodies(), "v1")
    repository.publish(repository.calibrate(parent, _labels(), clean).calibration_id)
    child = _ready(manager, _CarryEmbedder(poisoned=True), _bodies(added=2), "v2")

    with pytest.raises(SystemExit) as exit_info:
        main(["--tenant", tenant, "--dsn", TEST_DSN, "calibration", "drift",
              "--generation", child, "--strict"])
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "verdict:     RECALIBRATE_REQUIRED" in out
    # The numbers behind the verdict are printed, not just the verdict. An operator told to
    # recalibrate and not told what failed cannot tell a real drift from a broken monitor.
    assert "false abstain" in out and "false confirm" in out
    assert "refit would be" in out


@requires_db
def test_cli_drift_json_is_machine_readable(carry_tenant, tmp_path, capsys, _cli_embedder) -> None:  # noqa: F811, E501
    from recall.cli import main

    tenant, _manager = carry_tenant
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    main(["--tenant", tenant, "--dsn", TEST_DSN, "calibration", "drift",
          "--path", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "unknown"
    assert payload["probe"] is None


@requires_db
def test_cli_auto_on_an_uncalibrated_tenant_exits_zero(carry_tenant, capsys, _cli_embedder) -> None:  # noqa: F811, E501
    """`skipped` is a correct outcome, so it must not look like a failure to a post-build hook."""
    from recall.cli import main

    tenant, manager = carry_tenant
    generation = _ready(manager, _CarryEmbedder(), _bodies(), "v1")
    main(["--tenant", tenant, "--dsn", TEST_DSN, "calibration", "auto", "--generation", generation])
    out = capsys.readouterr().out
    assert "action: skipped" in out
    assert "human decision" in out
