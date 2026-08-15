"""The pre-registration is the authority over the artifacts, asserted rather than asked for.

Invariant I5 reads "labels frozen before arms: this file's and `gold.manifest.jsonl`'s git commit
timestamps precede every arm artifact, checked by the runner", and its named vacuous version is
"a comment claiming 'labelled first'". Until this module existed the invariant WAS that comment:
the pre-registration lived on `claude/truth-extraction-prereg` and the results on
`claude/truth-extraction-adjudication`, two branches neither of which contained the other, so no
ordering between them was expressible, let alone checked.

A second failure this module closes is not an ordering one. The four-fixture public bridge is
**P10** in the pre-registration; the runner, the validator, the tests and the published artifact
all called it **P7**, which is a different registered prediction ("targets naming a file outside
the corpus, M1, exactly 0") on a different instrument. Every word of the surrounding prose was
right and the identifier was wrong, which is the worst shape for it to be in: a reader who
cross-checks the published result against the pre-registration lands on a prediction the result
does not score. So the id is now derived from the pre-registration rather than from the previous
line of code.

⚠️ **The git layer cannot be the whole guard.** `master`'s ruleset bans merge commits, so this
work lands as a squash, and `938caad` stops being reachable from `master` the moment it does.
Every check here that must survive that is written against file CONTENT, and the git checks that
cannot are scoped to "when the commit is reachable" with the reason stated. The external anchor
is the pushed branch on the public repository, which is what the pre-registration tells a reader
to check and what no test in this repository can substitute for.

Properties, one test each:
  1. Every arm artifact the experiment published is present, so nothing below runs over an empty
     set. A `parametrize` over an empty glob reports SKIPPED inside a green run.
  2. The pre-registration carries exactly one parseable registration block, with every field.
  3. `registration_commit` is a full 40-hex sha, so it names exactly one commit.
  4. Every registered prediction id parses and none is registered twice.
  5. A prediction row inside a fenced code block is not mistaken for a registration.
  6. The fixtures artifact's prediction id resolves to the row about transplanted fixtures.
  7. The retired `p7_*` keys are absent from the published artifact.
  8. A second yaml block under one `## Registration` heading is refused, not silently first-won.
  9. A registration section with no yaml block does not adopt a fence from a later section.
  10. Quoted scalars and trailing comments are refused, so the error names the registration
      rather than the artifact under test.
  11. I5: every committed arm artifact was generated strictly after the registration.
  12. I5 refuses an artifact generated before the registration, and one generated at the same instant.
  13. I5 refuses a naive timestamp rather than comparing two clocks.
  14. I5 refuses a mapping that is not a registration block, with a message, rather than KeyError.
  15. The frozen gold manifest still has the digest and question count the pre-registration froze,
      and the runner refuses to start once they move.
  15b. The runner's write site enforces I5 against the REAL pre-registration, so the invariant is
      not one that runs only in tests.
  16. I5, git layer: when the registration commit is reachable its author date matches the block
      and precedes the earliest commit touching every tracked arm artifact.
  17. The git layer's comparison actually rejects a violation, so the skip above is not a
      guard that cannot fail.
  18. `census.json` is excluded from I5 deliberately, and would fail it: it is the input the
      predictions were written against, not a result.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from benchmarks.labelling.truth_extraction import run_arms
from benchmarks.labelling.truth_extraction.artifact_contract import (
    REGISTRATION_FIELDS,
    read_prediction_ids,
    read_registration,
    validate_gold_manifest_frozen,
    validate_registration_ordering,
)

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "results" / "truth_extraction" / "PREREGISTRATION-prose-extraction.md"
RESULTS = REPO / "results" / "truth_extraction"
MANIFEST = REPO / "benchmarks" / "labelling" / "truth_extraction" / "gold.manifest.jsonl"
FIXTURES_ARTIFACT = RESULTS / "arm_P10_fixtures.json"

#: The committed arm artifacts. `census.json` is NOT one; see
#: `test_the_census_is_excluded_from_i5_and_would_fail_it`. Named rather than numbered: the
#: property list above is renumbered whenever one is inserted, and an ordinal reference rots.
ARM_ARTIFACTS = sorted(RESULTS.glob("arm_*.json"))

#: Named, not just globbed. A glob that matches nothing raises nothing: `Path.glob` on a renamed
#: or moved directory yields an empty list, a `parametrize` over it reports SKIPPED inside a green
#: run, and every loop below iterates zero times. Three I5 assertions would go quiet together,
#: including the census negative control. So the names the experiment actually published are
#: written down, and a rename has to come here and say so.
EXPECTED_ARMS = {"arm_R1_rules.json", "arm_M1_model.json", "arm_P10_fixtures.json"}


def _text() -> str:
    return PREREG.read_text(encoding="utf-8")


def _registration() -> dict[str, str]:
    return read_registration(_text())


def _git_author_dates(*args: str) -> list[int] | None:
    """Author timestamps from `git log`, newest first. `None` when git cannot answer.

    Author date rather than committer date because a rebase rewrites the second and preserves the
    first, and this branch was rebased onto `master` to join the two histories in the first place.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO), "log", "--format=%at", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    if done.returncode != 0:
        return None
    lines = [line for line in done.stdout.split() if line]
    return [int(line) for line in lines] or None


def _author_ts(rev: str) -> int | None:
    """The author timestamp of one commit, or `None` when it is not reachable."""
    stamps = _git_author_dates("-1", rev)
    return stamps[0] if stamps else None


def _earliest_ts(path: Path) -> int | None:
    """The EARLIEST author date of any commit touching `path`, following renames.

    `min`, not "the last line git printed". Two reasons, and the first is not the obvious one:

    - `git log` orders by topology and commit date, not by author date, so the last line is the
      oldest ANCESTOR rather than the earliest authored commit. Those coincide only while author
      and committer dates run together, and this module's whole premise is that they diverge
      under rebase, which is how the two histories here were joined in the first place.
    - Earliest-of-all is the conservative reading. Newest-touching would pass an artifact created
      before the pre-registration and merely reformatted after it, which is exactly the case I5
      exists to catch.

    ⚠️ `--follow` rests on git's rename similarity heuristic. A rename that also rewrote the body
    enough to fall below the threshold truncates the history here, and the pre-registration-
    predating ancestor becomes invisible. That is a limit of the git layer, not a reason to trust
    it less than the content checks, which do not depend on history at all.
    """
    stamps = _git_author_dates("--follow", "--", str(path))
    return min(stamps) if stamps else None


def _in_head(path: Path) -> bool:
    """Whether `path` exists in the current commit.

    `ls-tree HEAD`, not `ls-files`: the latter reports a merely STAGED file as tracked, and a
    staged file has no commit history to compare, so the git layer would fail on the ordinary act
    of generating an artifact and staging it.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO), "ls-tree", "HEAD", "--", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return False
    return done.returncode == 0 and bool(done.stdout.strip())


def _assert_precedes(registration_ts: int, artifact_ts: int, artifact: str) -> None:
    """The pure half of the git layer, so it can be exercised without a repository."""
    if registration_ts >= artifact_ts:
        raise AssertionError(
            f"I5 FAILED for {artifact}: the pre-registration commit is dated "
            f"{datetime.fromtimestamp(registration_ts, timezone.utc).isoformat()}, at or after "
            f"the commit that introduced the artifact "
            f"({datetime.fromtimestamp(artifact_ts, timezone.utc).isoformat()})"
        )


def test_the_arm_artifacts_the_experiment_published_are_all_present():
    """The non-vacuity guard for every parametrized and looping check below."""
    assert {p.name for p in ARM_ARTIFACTS} == EXPECTED_ARMS


def test_the_registration_block_parses_and_is_complete():
    registration = _registration()
    assert set(REGISTRATION_FIELDS) <= set(registration)
    assert all(registration[field] for field in REGISTRATION_FIELDS)


def test_the_registration_commit_is_a_full_sha():
    """An abbreviated sha is ambiguous across a repository's lifetime; the anchor must not be."""
    sha = _registration()["registration_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", sha), f"{sha!r} is not a full 40-hex sha"


def test_every_prediction_id_is_registered_once():
    ids = read_prediction_ids(_text())
    # 12 registered predictions: P1-P7 on the adjudicated pack, P8-P9 recall, P10 the public
    # bridge, P11-P12 deferred to the private corpus.
    assert len(ids) == 12, f"expected 12 registered predictions, parsed {sorted(ids)}"
    assert "P10" in ids and "P7" in ids


def test_the_fixtures_artifact_publishes_the_id_the_prereg_registered_for_it():
    """The guard the P7/P10 mislabel would have failed.

    Discriminating on the row text rather than on the id, because the id is exactly what was
    wrong. P10's row says "transplanted fixtures refused by m1"; P7's says "targets naming a file
    outside the corpus". Republishing this artifact as `p7_holds` puts the second row under a
    result that measures the first, and this assertion goes red.
    """
    ids = read_prediction_ids(_text())
    payload = json.loads(FIXTURES_ARTIFACT.read_text(encoding="utf-8"))
    published = sorted(k[: -len("_holds")] for k in payload if k.endswith("_holds"))
    assert len(published) == 1, f"the fixtures artifact must publish one prediction, got {published}"

    key = published[0].upper()
    assert key in ids, f"{key} is not a registered prediction id"
    row = ids[key]
    assert "fixtures" in row, (
        f"the fixtures artifact publishes {key}, whose registered row is {row!r}. That row is "
        f"not about the transplanted fixtures, so the artifact is scoring one prediction and "
        f"labelling it with another"
    )
    assert "outside the corpus" not in row


def test_a_prediction_row_inside_a_code_fence_is_not_registered():
    """The pre-registration already carries fences, so an illustrative row is one edit away.

    Unstripped, the duplicate raises and takes every caller of `read_prediction_ids` down at
    once, blaming the pre-registration for a duplicate it does not have.
    """
    doc = "| P1 | the real one |\n\n```\n| P1 | an example, not a registration |\n```\n"
    assert read_prediction_ids(doc) == {"P1": "the real one |"}


def test_a_prediction_cannot_be_introduced_in_the_section_that_scores_it():
    """Scanning stops at `## Result`, which is both a fix and the stronger rule.

    The scoring table restates every id, so reading it as a registration made P1 look registered
    twice. Bounding it also means a prediction registered only in the Result section, which is to
    say after the outcome was known, is not registered at all.
    """
    doc = "| P1 | the registered one |\n\n## Result\n\n| P1 | scored |\n| P99 | invented late |\n"
    ids = read_prediction_ids(doc)
    assert ids == {"P1": "the registered one |"}
    assert "P99" not in ids

    # The heading is anchored, so a document OPENING with it is bounded like any other. Nothing
    # is registered, and an empty registration is refused rather than returned as an empty map:
    # "no predictions were registered" and "no predictions parsed" must not look the same.
    with pytest.raises(ValueError, match="no prediction rows"):
        read_prediction_ids("## Result\n\n| P1 | a |\n\n| P2 | b |\n")

    # ...and a heading that merely starts with "Result" is not the bound. As a prefix match this
    # truncated at `## Results overview` and returned a short dict rather than raising, which is
    # the shape that silently drops registrations.
    assert set(read_prediction_ids("| P1 | a |\n\n## Results overview\n\n| P2 | b |\n")) == {"P1", "P2"}


def test_the_result_bound_does_not_reopen_the_fenced_row_hole():
    """Order of operations: fences are stripped BEFORE the bound is taken.

    Bounded first, a fence spanning the `## Result` line is left unterminated in the prefix, so
    the fence regex cannot match it and every illustrative row inside it registers. That is the
    exact hole `test_a_prediction_row_inside_a_code_fence_is_not_registered` exists to close, and
    taking the bound first quietly reopened it.
    """
    doc = (
        "| P1 | the registered one |\n\n"
        "```\n| P2 | illustrative, inside a fence |\n## Result\n```\n\n"
        "| P3 | registered after the fence |\n"
    )
    ids = read_prediction_ids(doc)
    assert "P2" not in ids, "a row inside a fence was registered"
    assert set(ids) == {"P1", "P3"}, "a real row after the fence was dropped"


def test_an_amended_registration_block_is_refused_rather_than_first_won():
    """Two blocks under ONE heading is the shape an amendment takes, and it must not be silent.

    A single heading-anchored match reports this as "found 1" and returns the STALE first block,
    so I5 would be measured against a superseded timestamp with nothing raised.
    """
    doc = (
        "## Registration\n\n```yaml\nregistration_commit: " + "a" * 40 + "\n```\n\n"
        "An amendment, wrongly placed beside the block it supersedes rather than replacing it.\n\n"
        "```yaml\nregistration_commit: " + "b" * 40 + "\n```\n"
    )
    with pytest.raises(ValueError, match="exactly one yaml block"):
        read_registration(doc)


def test_a_registration_section_does_not_adopt_a_later_sections_fence():
    """Unbounded, the section would reach forward and register whatever fence it found first."""
    doc = (
        "## Registration\n\nThe block was removed in an edit.\n\n"
        "## Provenance\n\n```yaml\nregistration_commit: " + "c" * 40 + "\n```\n"
    )
    with pytest.raises(ValueError, match="exactly one yaml block"):
        read_registration(doc)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gold_manifest_questions", "51  # frozen 2026-08-14"),
        ("registration_authored", '"2026-08-14T19:30:04+00:00"'),
        ("gold_manifest_digest", '"' + "d" * 64 + '"'),
        ("registration_commit", "938caad"),
    ],
    ids=["trailing-comment", "quoted-timestamp", "quoted-digest", "abbreviated-sha"],
)
def test_a_yaml_spelling_this_parser_does_not_implement_is_refused(field: str, value: str):
    """Half-reading a value defers the failure and misattributes it.

    An unquoted-only parser that accepts `"2026-..."` verbatim fails later as "unparseable
    timestamp for <artifact>", naming the artifact for a defect in this block; a quoted digest
    surfaces as "the gold manifest digest has moved", which is the worst available false alarm
    for a check whose whole job is noticing that the labels were regenerated.
    """
    registration = _registration()
    registration[field] = value
    block = "\n".join(f"{k}: {v}" for k, v in registration.items())
    with pytest.raises(ValueError, match="does not match"):
        read_registration(f"## Registration\n\n```yaml\n{block}\n```\n")


def test_i5_refuses_a_mapping_that_is_not_a_registration():
    """Exported and called from the write site, so a missing key must refuse, not KeyError."""
    payload = {"_provenance": {"generated_at": "2026-08-15T14:07:39+00:00"}}
    with pytest.raises(ValueError, match="registration_authored"):
        validate_registration_ordering({}, payload, artifact="fake")


def test_the_retired_p7_keys_are_gone_from_the_published_artifact():
    raw = FIXTURES_ARTIFACT.read_text(encoding="utf-8")
    assert "p7_" not in raw
    assert not (RESULTS / "arm_P7_fixtures.json").exists(), "the old filename is still on disk"


@pytest.mark.parametrize("artifact", ARM_ARTIFACTS, ids=lambda p: p.name)
def test_i5_holds_for_every_committed_arm_artifact(artifact: Path):
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    validate_registration_ordering(_registration(), payload, artifact=artifact.name)


def test_i5_refuses_a_result_that_predates_or_matches_its_own_prediction():
    registration = _registration()
    authored = datetime.fromisoformat(registration["registration_authored"])
    for offset, label in ((timedelta(seconds=-1), "before"), (timedelta(0), "the same instant")):
        payload = {"_provenance": {"generated_at": (authored + offset).isoformat()}}
        with pytest.raises(ValueError, match="I5 FAILED"):
            validate_registration_ordering(registration, payload, artifact=f"fake-{label}")

    # ...and the positive control: one second later is accepted, so the check is not refusing
    # everything handed to it.
    ok = {"_provenance": {"generated_at": (authored + timedelta(seconds=1)).isoformat()}}
    validate_registration_ordering(registration, ok, artifact="fake-after")


def test_i5_refuses_a_naive_timestamp():
    payload = {"_provenance": {"generated_at": "2026-08-15T14:07:39"}}
    with pytest.raises(ValueError, match="offset"):
        validate_registration_ordering(_registration(), payload, artifact="fake-naive")


def test_the_gold_manifest_still_matches_the_frozen_digest():
    """I5's other half: the labels are frozen, so regenerating them after the fact shows here."""
    registration = _registration()
    header = json.loads(MANIFEST.read_text(encoding="utf-8").splitlines()[0])
    assert header["digest"] == registration["gold_manifest_digest"]
    assert header["n_questions"] == int(registration["gold_manifest_questions"])


def test_i5_git_layer_when_the_registration_commit_is_reachable():
    sha = _registration()["registration_commit"]
    registration_ts = _author_ts(sha)
    if registration_ts is None:
        pytest.skip(
            f"{sha[:8]} is not reachable from this checkout. Expected on `master` once this "
            f"branch lands as a squash; the pushed `claude/truth-extraction-prereg` branch is "
            f"the external anchor. The content checks above still ran."
        )

    authored = datetime.fromisoformat(_registration()["registration_authored"])
    assert registration_ts == int(authored.timestamp()), (
        "the registration block's timestamp disagrees with the commit it names, which is how a "
        "backdated block would look"
    )
    # NAMED, not counted. A bare `checked > 0` counter passed at 2 while skipping
    # `arm_P10_fixtures.json`, which is the one artifact this whole change renames: two published
    # artifacts were in HEAD, the third was only staged, and the guard reported green over the
    # exact file it existed to cover. A count cannot tell you WHICH one it skipped.
    #
    # An artifact the glob picks up that the experiment did not publish is a fresh run's output
    # and is legitimately skipped; every artifact in EXPECTED_ARMS must be compared.
    for artifact in ARM_ARTIFACTS:
        if artifact.name not in EXPECTED_ARMS:
            continue
        assert _in_head(artifact), (
            f"{artifact.name} is a published arm artifact but is not in HEAD, so the git layer "
            f"would skip the one file it exists to check. Commit it before trusting this test"
        )
        artifact_ts = _earliest_ts(artifact)
        assert artifact_ts is not None, f"{artifact.name} is in HEAD but has no commit history"
        _assert_precedes(registration_ts, artifact_ts, artifact.name)


def test_the_git_layer_comparison_rejects_a_violation():
    """`test_i5_git_layer_...` skips when the commit is unreachable; its logic is exercised here.

    Without this, a checkout where the skip fires would report a green I5 having compared nothing.
    """
    _assert_precedes(1_000, 2_000, "ok")  # positive control: earlier registration is accepted
    for registration_ts, artifact_ts, label in ((2_000, 1_000, "after"), (1_000, 1_000, "equal")):
        with pytest.raises(AssertionError, match="I5 FAILED"):
            _assert_precedes(registration_ts, artifact_ts, f"violation-{label}")


def test_the_runner_write_site_enforces_i5_against_the_real_preregistration(tmp_path: Path):
    """I5 is registered as "checked by the runner", so the wiring is asserted, not assumed.

    The adjacent I7 row names "a validator that runs only in tests" as the vacuous shape, and an
    I5 that lived only in this file would have been exactly that.
    """
    authored = datetime.fromisoformat(_registration()["registration_authored"])
    # A payload that is otherwise VALID. Built by taking a real published artifact and moving only
    # its timestamp: a hand-rolled partial payload is refused by `validate_arm_result` first, and
    # the `raises` below would then pass for a reason that has nothing to do with I5.
    payload = json.loads((RESULTS / "arm_R1_rules.json").read_text(encoding="utf-8"))
    payload["_provenance"]["generated_at"] = (authored - timedelta(days=1)).isoformat()

    out = tmp_path / "arm.json"
    with pytest.raises(ValueError, match="I5 FAILED"):
        run_arms._emit(payload, out)
    assert not out.exists(), "an I5-violating artifact was written anyway"

    # Positive control: the same payload, generated after the registration, is written. Without
    # it, the check above could be refusing this payload for some reason it never names.
    payload["_provenance"]["generated_at"] = (authored + timedelta(days=1)).isoformat()
    run_arms._emit(payload, out)
    assert out.is_file()


def test_the_runner_refuses_to_start_when_the_gold_labels_have_moved():
    """I5's frozen-labels half, at the runner's pre-flight rather than only here."""
    registration = _registration()
    header = json.loads(MANIFEST.read_text(encoding="utf-8").splitlines()[0])
    validate_gold_manifest_frozen(registration, header)  # must not raise

    for key, value in (("digest", "0" * 64), ("n_questions", 999)):
        with pytest.raises(ValueError, match="I5 FAILED"):
            validate_gold_manifest_frozen(registration, {**header, key: value})


def test_the_census_is_excluded_from_i5_and_would_fail_it():
    """The scoping decision, as a fact rather than a comment in the validator.

    The census was generated 2026-08-11 and the pre-registration was authored on the 14th AGAINST
    it: the commit subject is "pre-register the prose extraction arms against the measured
    census". Holding an input to the ordering an output must meet would invert the experiment, so
    if this ever stops failing, the census has been regenerated after the prediction and the
    ceiling every recall claim rests on is no longer the one that was predicted against.
    """
    census = RESULTS / "census.json"
    assert census not in ARM_ARTIFACTS
    payload = json.loads(census.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="I5 FAILED"):
        validate_registration_ordering(_registration(), payload, artifact="census.json")
