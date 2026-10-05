"""The supersession arbiter proposes undeclared edges, the right way round, and pays once.

Properties, one test each unless noted:

1. The trigger passes a shared file-name date or overlapping slug words, and nothing else.
2. Direction comes from `modified` when both memos carry it, else from the file-name date, and
   is never guessed.
3. A stated probability of 0.5 or more counts only with two grounded, verbatim quotes; a boolean
   is not a probability.
4. A run proposes the pair at or above the threshold with the SUPERSEDED memo as `older`, and
   maps the two quotes to the right memo whichever one was shown as Note A.
5. A pair either memo already declares is never sent to the model.
6. A pair with no readable direction is dropped before the call, not after it.
7. `max_pairs` bounds the calls, and the rest are counted, not dropped in silence.
8. A second run over an unchanged corpus calls nothing; a failed call is not cached.
9. The arbiter is off by default, and a malformed setting refuses.
10. Through `recall rewrite`: `plan` lists the proposal with both quotes and the run summary,
    and `apply --apply` writes `supersedes:` onto the NEWER memo, naming the older.
11. A proposal whose newer memo already supersedes a DIFFERENT memo is ordinary review work, and
    `apply` adds the second reference; only an empty `supersedes:` is BLOCKED.
12. The direction check keeps a gated pair only when the model calls the metadata's newer note
    current: a disagreement, an `unclear`, an ungrounded quote or a failed call drops it, and the
    run counts the drop. A failed direction call is retried on the next run.
13. The direction check is on by default, `RECALL_ARBITER_DIRECTION_CHECK=0` turns it off (no
    direction call at all), and a malformed value refuses.
14. The gate's cache keys are unchanged by the check, and the two questions never share a key.

Red proof, 2026-10-02, each a deliberate mutation of `recall/supersession_arbiter.py` or
`recall/cli_commands/extract_rewrite.py` with this file unchanged, each failing in the named
test's assertion, then restored (all green):

- M1 `orient` returns `(newer, older)`: `test_a_run_proposes_the_older_memo_as_superseded`,
  `test_rewrite_plan_and_apply_write_supersedes_onto_the_newer_memo` and four more.
- M2 `score` skips the grounding check: `test_a_confident_claim_needs_two_grounded_quotes`.
- M3 `run` compares `<=` the threshold: `test_a_run_proposes_the_older_memo_as_superseded`.
- M4 `older_is_a` is always True: `test_quotes_follow_the_memo_whichever_was_shown_first`.
- M5 `declared_pair` always False: `test_a_declared_pair_is_never_sent_to_the_model`.
- M6 the budget slice removed: `test_max_pairs_bounds_the_calls_and_counts_the_rest`.
- M7 `cache.put` removed: `test_a_second_run_calls_nothing_and_a_failure_is_retried`.
- M8 the CLI quote lines removed: `test_rewrite_plan_and_apply_write_supersedes_onto_the_newer_memo`.
- M9 `_declared_state` in `recall/cli_commands/extract_rewrite.py` treats a declared different
  reference as the same edge (returns "declared"):
  `test_a_memo_that_supersedes_another_takes_a_second_edge`. Returning "blocked" for it fails the
  same test, and dropping the membership check fails
  `tests/test_cli_rewrite.py::test_an_already_declared_proposal_is_marked_in_plan`.
- M10 the empty-declaration branch of `_declared_state` returns None:
  `test_an_empty_supersedes_is_blocked_not_filled_in`.

Red proof for the direction check, 2026-10-03, same procedure, each against
`recall/supersession_arbiter.py`:

- X1 a disagreement is kept (`current != verdict.newer` read as `current is None`):
  `test_a_disagreement_drops_the_pair_and_is_counted`.
- X2 an unclear answer counts as agreement:
  `test_an_unclear_or_ungrounded_answer_drops_the_pair`.
- X3 `current_note` skips the grounding check:
  `test_an_unclear_or_ungrounded_answer_drops_the_pair`.
- X4 `direction_check` ignored: `test_the_check_can_be_switched_off_and_is_on_by_default`.
- X5 `_switch` defaults to off: `test_the_check_can_be_switched_off_and_is_on_by_default`.
- X6 `cache_key` ignores the prompt: `test_the_gate_keys_are_unchanged_and_the_questions_never_share_one`.
- X7 a failed direction call is cached as an empty reply instead of skipped:
  `test_a_failed_direction_call_drops_the_pair_and_is_retried`.
15 to 17. A cache-only arbiter answers only from the cache: it never calls the model, counts and
reports what it skipped, needs no API key, and holds a client that refuses if anything calls it.

Red proof for cache-only, 2026-10-05, same procedure (JUnit XML), each against
`recall/supersession_arbiter.py`:
- C1 the cache-only early return removed:
  `test_a_cache_only_run_calls_nothing_and_counts_what_it_skipped`, assert 4 == 0 (model calls).
- C2 skipped pairs counted as 0: the same test, assert 0 == 2.
- C3 the never-called client answers instead of raising:
  `test_a_cache_only_arbiter_needs_no_key_and_cannot_call`, DID NOT RAISE RuntimeError.
- C4 the summary omits the skipped count: the first test, AssertionError on the summary text.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from recall import supersession_arbiter as arb
from recall.cli import main
from recall.frontmatter import parse_frontmatter, supersedes_targets
from recall.rewrite import RewriteRefused, corpus_proposals
from recall.supersession_arbiter import (
    ArbiterCache,
    ArbiterSettings,
    SupersessionArbiter,
    arbiter_settings,
    blinded_order,
    orient,
    read_note,
    score,
    triggered,
)

CLAIM = re.compile(r"[^.\n]*listens on port[^.\n]*\.")
#: What marks the CURRENT note of each fixture pair, for the fake's direction answers.
CURRENT = re.compile(r"[^.\n]*(?:After the move|now listens)[^.\n]*\.")

OLD = "2026-01-01-deploy-port.md"
NEW = "2026-02-01-deploy-port.md"
OLD_B = "2026-03-01-cache-size-limit.md"
NEW_B = "2026-03-05-cache-size-limit.md"

BODIES = {
    OLD: "# Deploy port\n\nThe gateway service listens on port 8080 behind the proxy.\n",
    NEW: "# Deploy port\n\nAfter the move the gateway service listens on port 9090 behind the proxy.\n",
    OLD_B: "# Cache size\n\nThe embedding cache listens on port 7000 and holds 512 MB.\n",
    NEW_B: "# Cache size\n\nThe embedding cache now listens on port 7001 and holds 1024 MB.\n",
    "2026-04-01-unrelated-notes.md": "# Lunch\n\nThe canteen opens at noon on weekdays.\n",
}


class FakeClient:
    """Answers both questions the arbiter asks, from markers in the fixture notes.

    The gate: a pair when both notes state the port claim. The direction check: the note holding
    a `CURRENT` sentence, quoted, unless `direction` says to answer otherwise ("reverse",
    "unclear", "ungrounded"). `fail` fails the first N calls; `fail_direction` the first N
    direction calls.
    """

    def __init__(
        self, probability: int = 95, fail: int = 0, direction: str = "content",
        fail_direction: int = 0,
    ) -> None:
        self.probability = probability
        self.fail = fail
        self.direction = direction
        self.fail_direction = fail_direction
        self.gate_calls = 0
        self.direction_calls = 0

    @property
    def calls(self) -> int:
        return self.gate_calls + self.direction_calls

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        asks_direction = messages[0]["content"] == arb.DIRECTION_PROMPT
        if asks_direction:
            self.direction_calls += 1
        else:
            self.gate_calls += 1
        if self.fail:
            self.fail -= 1
            raise TimeoutError("simulated")
        if asks_direction and self.fail_direction:
            self.fail_direction -= 1
            raise TimeoutError("simulated")
        assert kwargs.get("response_format") == {"type": "json_object"}
        user = messages[1]["content"]
        note_a, note_b = user.removeprefix("Note A:\n").split("\n\n---\n\nNote B:\n")
        if asks_direction:
            return self._direction(note_a, note_b)
        quote_a, quote_b = CLAIM.search(note_a), CLAIM.search(note_b)
        if not (quote_a and quote_b):
            return '{"probability": 5, "subject": "", "quote_a": "", "quote_b": ""}'
        return (
            f'{{"probability": {self.probability}, "subject": "port", '
            f'"quote_a": "{quote_a.group(0).strip()}", "quote_b": "{quote_b.group(0).strip()}"}}'
        )

    def _direction(self, note_a: str, note_b: str) -> str:
        if self.direction == "unclear":
            return '{"current": "unclear", "quote": ""}'
        in_a = CURRENT.search(note_a)
        current, quote = ("A", in_a) if in_a else ("B", CURRENT.search(note_b))
        if self.direction == "reverse":
            current = "B" if current == "A" else "A"
            quote = CLAIM.search(note_b if current == "B" else note_a)
        text = quote.group(0).strip() if quote else ""
        if self.direction == "ungrounded":
            text = "a sentence that appears in neither note at all"
        return f'{{"current": "{current}", "quote": "{text}"}}'


def settings(**overrides: object) -> ArbiterSettings:
    base: dict[str, object] = dict(
        model_id="test/model", revision="r1", base_url="https://openrouter.ai/api/v1",
        threshold=0.90, neighbours=10, max_pairs=200,
    )
    base.update(overrides)
    return ArbiterSettings(**base)  # type: ignore[arg-type]


def arbiter(client: FakeClient, cache: ArbiterCache | None = None, **overrides: object):
    return SupersessionArbiter(client=client, settings=settings(**overrides), cache=cache, workers=2)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("RECALL_TRUTH_EXTRACTION", "RECALL_SUPERSESSION_ARBITER", "RECALL_ARBITER_CACHE"):
        monkeypatch.delenv(name, raising=False)


def test_the_trigger_passes_a_shared_date_or_slug_and_nothing_else() -> None:
    assert triggered("2026-01-01-a.md", "2026-01-01-b.md")
    assert triggered("2026-01-01-deploy-port.md", "2026-02-01-deploy-port.md")
    assert not triggered("2026-01-01-deploy-port.md", "2026-02-01-lunch-menu.md")
    assert not triggered("notes.md", "other.md")


def test_direction_comes_from_metadata_one_signal_at_a_time() -> None:
    stamped = "---\nmodified: 2026-05-02T10:00:00Z\n---\nx\n"
    earlier = "---\nmodified: 2026-05-01\n---\nx\n"
    # `modified` wins over the file-name dates, which point the other way.
    a, b = read_note("2026-09-01-a.md", earlier), read_note("2026-01-01-b.md", stamped)
    older, newer, source = orient(a, b)  # type: ignore[misc]
    assert (older.name, newer.name, source) == ("2026-09-01-a.md", "2026-01-01-b.md", "modified")
    # Only one memo stamped: the dates decide.
    c = read_note("2026-01-01-c.md", "x\n")
    older, newer, source = orient(a, c)  # type: ignore[misc]
    assert (older.name, newer.name, source) == ("2026-01-01-c.md", "2026-09-01-a.md", "file-name date")
    # Same day, no stamps: never a guess.
    assert orient(read_note("2026-01-01-x.md", "x"), read_note("2026-01-01-y.md", "y")) is None
    assert read_note("n.md", earlier).modified == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert read_note("2026-02-30-bad.md", "x").day is None
    assert read_note("2026-02-03-ok.md", "x").day == date(2026, 2, 3)


def test_a_confident_claim_needs_two_grounded_quotes() -> None:
    a, b = "The service listens on port 8080 today.", "The service listens on port 9090 today."
    good = {"probability": 95, "quote_a": "listens on port 8080 today", "quote_b": "listens on port 9090 today"}
    assert score(good, a, b) == 0.95
    assert score({**good, "quote_b": "listens on port 9999 today"}, a, b) == 0.0
    assert score({**good, "quote_a": "port 8080"}, a, b) == 0.0  # under 20 characters
    assert score({"probability": 30}, a, b) == 0.30  # below 0.5 needs no quotes
    assert score({"probability": True}, a, b) == 0.0
    assert score(None, a, b) == 0.0


def _corpus(tmp_path: Path, bodies: dict[str, str] = BODIES) -> dict[str, str]:
    for name, body in bodies.items():
        (tmp_path / name).write_text(body, encoding="utf-8", newline="\n")
    return dict(bodies)


def test_a_run_proposes_the_older_memo_as_superseded() -> None:
    client = FakeClient(probability=90)
    run = arbiter(client).run(BODIES)
    pairs = {(v.older, v.newer) for v in run.verdicts}
    assert pairs == {(OLD, NEW), (OLD_B, NEW_B)}
    assert all(v.probability == 0.90 for v in run.verdicts)  # exactly at the threshold
    below = arbiter(FakeClient(probability=89)).run(BODIES)
    assert below.verdicts == ()


def test_quotes_follow_the_memo_whichever_was_shown_first() -> None:
    # The fixture must exercise both orders, or this test proves nothing.
    assert {blinded_order(OLD, NEW)[0] == OLD, blinded_order(OLD_B, NEW_B)[0] == OLD_B} == {True, False}
    run = arbiter(FakeClient()).run(BODIES)
    by_pair = {(v.older, v.newer): v for v in run.verdicts}
    assert "8080" in by_pair[(OLD, NEW)].quote_older
    assert "9090" in by_pair[(OLD, NEW)].quote_newer
    assert "7000" in by_pair[(OLD_B, NEW_B)].quote_older
    assert "7001" in by_pair[(OLD_B, NEW_B)].quote_newer


def test_a_declared_pair_is_never_sent_to_the_model() -> None:
    declared = {**BODIES, NEW: f"---\nsupersedes: {OLD}\n---\n" + BODIES[NEW]}
    client = FakeClient()
    run = arbiter(client).run(declared)
    assert run.already_declared == 1
    assert {(v.older, v.newer) for v in run.verdicts} == {(OLD_B, NEW_B)}


def test_an_undirected_pair_is_dropped_before_the_call() -> None:
    same_day = {
        "2026-01-01-deploy-port.md": BODIES[OLD],
        "2026-01-01-deploy-port-v2.md": BODIES[NEW],
        # A third memo, or every shared term is in every document and TF-IDF weighs it zero.
        "2026-04-01-unrelated-notes.md": BODIES["2026-04-01-unrelated-notes.md"],
    }
    client = FakeClient()
    run = arbiter(client).run(same_day)
    assert (run.undirected, run.called, client.calls, run.verdicts) == (1, 0, 0, ())


def test_max_pairs_bounds_the_calls_and_counts_the_rest() -> None:
    client = FakeClient()
    run = arbiter(client, max_pairs=1).run(BODIES)
    assert client.gate_calls == 1
    assert run.over_budget == run.candidates - run.already_declared - run.undirected - 1 > 0


def test_a_second_run_calls_nothing_and_a_failure_is_retried(tmp_path: Path) -> None:
    cache = ArbiterCache(tmp_path / "arbiter.sqlite3")
    try:
        first = FakeClient(fail=1)
        run1 = arbiter(first, cache).run(BODIES)
        assert run1.failed == 1
        second = FakeClient()
        arbiter(second, cache).run(BODIES)
        # Only the pair whose gate call failed, and then its direction question.
        assert (second.gate_calls, second.direction_calls) == (1, 1)
        third = FakeClient()
        run3 = arbiter(third, cache).run(BODIES)
        assert third.calls == 0 and run3.cached == 4  # two gate answers, two direction answers
        assert {(v.older, v.newer) for v in run3.verdicts} == {(OLD, NEW), (OLD_B, NEW_B)}
    finally:
        cache.close()


def test_the_arbiter_is_off_by_default_and_refuses_a_malformed_setting(tmp_path: Path) -> None:
    assert arbiter_settings({}) is None
    assert arbiter_settings({"RECALL_SUPERSESSION_ARBITER": "0"}) is None
    with pytest.raises(ValueError, match="RECALL_SUPERSESSION_ARBITER"):
        arbiter_settings({"RECALL_SUPERSESSION_ARBITER": "maybe"})
    with pytest.raises(ValueError, match="RECALL_ARBITER_THRESHOLD"):
        arbiter_settings({"RECALL_SUPERSESSION_ARBITER": "1", "RECALL_ARBITER_THRESHOLD": "0.2"})
    on = arbiter_settings({"RECALL_SUPERSESSION_ARBITER": "1"})
    assert on is not None and on.threshold == 0.90 and on.model_id == arb.DEFAULT_ARBITER_MODEL
    _corpus(tmp_path)
    with pytest.raises(RewriteRefused, match="RECALL_SUPERSESSION_ARBITER=1"):
        corpus_proposals(tmp_path)


def test_rewrite_plan_and_apply_write_supersedes_onto_the_newer_memo(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _corpus(tmp_path)
    cache_path = tmp_path.parent / f"{tmp_path.name}-arbiter.sqlite3"
    client = FakeClient()
    # A fresh cache per resolution, as the real `resolve_arbiter` opens one: `corpus_proposals`
    # closes the arbiter it resolved.
    monkeypatch.setattr(
        arb, "resolve_arbiter", lambda env=None: arbiter(client, ArbiterCache(cache_path))
    )
    main(["rewrite", "plan", str(tmp_path)])
    out, err = capsys.readouterr()
    assert "arbiter:" in err and "2 proposed" in err
    assert f"{OLD} -> {NEW}" in out
    assert 'older: "The gateway service listens on port 8080' in out
    assert 'newer: "After the move the gateway service listens on port 9090' in out
    proposal = next(p for p in corpus_proposals(tmp_path) if p.subject_id == OLD)
    assert (proposal.object_id, proposal.status) == (NEW, "requires_review")
    calls_after_plan = client.calls
    main(["rewrite", "apply", str(tmp_path), "--proposal", proposal.id,
          "--reviewer", "gde", "--note", "ports moved", "--apply"])
    assert client.calls == calls_after_plan  # apply re-derives from the cache, and pays nothing
    assert f"supersedes: {OLD}" in (tmp_path / NEW).read_text(encoding="utf-8")
    assert "supersedes" not in (tmp_path / OLD).read_text(encoding="utf-8")


def test_a_memo_that_supersedes_another_takes_a_second_edge(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The live run's case: the newer memo already declares a DIFFERENT memo.

    `plan` used to call such a proposal DECLARED (it checked only that the key was present),
    then BLOCKED while the key held one value. `supersedes` holds several now, so the proposal
    is ordinary review work, and `apply` adds the second reference beside the first.
    """
    third = "2025-12-01-gateway-draft.md"
    bodies = {**BODIES, third: "# Draft\n\nA first sketch of the gateway, never deployed.\n"}
    bodies[NEW] = f"---\nsupersedes: {third}\n---\n" + BODIES[NEW]
    _corpus(tmp_path, bodies)
    cache_path = tmp_path.parent / f"{tmp_path.name}-arbiter.sqlite3"
    client = FakeClient()
    monkeypatch.setattr(
        arb, "resolve_arbiter", lambda env=None: arbiter(client, ArbiterCache(cache_path))
    )
    main(["rewrite", "plan", str(tmp_path)])
    lines = capsys.readouterr().out.splitlines()
    at = lines.index(f"      {OLD} -> {NEW}")
    mark = next(line for line in reversed(lines[:at]) if not line.startswith("      "))
    assert mark.split()[0] == "review"
    proposal = next(p for p in corpus_proposals(tmp_path) if p.subject_id == OLD)
    main(["rewrite", "apply", str(tmp_path), "--proposal", proposal.id,
          "--reviewer", "gde", "--note", "both replaced", "--apply"])
    meta = parse_frontmatter((tmp_path / NEW).read_text(encoding="utf-8"))[0]
    assert supersedes_targets(meta["supersedes"]) == (third, OLD)


def test_an_empty_supersedes_is_blocked_not_filled_in(tmp_path: Path, monkeypatch, capsys) -> None:
    """`supersedes:` with nothing after it is a human saying "supersedes nothing"."""
    bodies = {**BODIES, NEW: "---\nsupersedes:\n---\n" + BODIES[NEW]}
    _corpus(tmp_path, bodies)
    cache_path = tmp_path.parent / f"{tmp_path.name}-arbiter.sqlite3"
    client = FakeClient()
    monkeypatch.setattr(
        arb, "resolve_arbiter", lambda env=None: arbiter(client, ArbiterCache(cache_path))
    )
    main(["rewrite", "plan", str(tmp_path)])
    lines = capsys.readouterr().out.splitlines()
    at = lines.index(f"      {OLD} -> {NEW}")
    mark = next(line for line in reversed(lines[:at]) if not line.startswith("      "))
    assert "BLOCKED" in mark
    assert "declares the key with no value" in lines[at - 1]


def test_a_disagreement_drops_the_pair_and_is_counted() -> None:
    agreed = arbiter(FakeClient()).run(BODIES)
    assert {(v.older, v.newer) for v in agreed.verdicts} == {(OLD, NEW), (OLD_B, NEW_B)}
    assert all(v.direction_source.endswith("confirmed by the model") for v in agreed.verdicts)
    assert agreed.direction_dropped == 0
    reversed_run = arbiter(FakeClient(direction="reverse")).run(BODIES)
    assert reversed_run.verdicts == () and reversed_run.direction_dropped == 2
    assert "2 dropped by the direction check" in reversed_run.summary()


@pytest.mark.parametrize("answer", ["unclear", "ungrounded"])
def test_an_unclear_or_ungrounded_answer_drops_the_pair(answer: str) -> None:
    run = arbiter(FakeClient(direction=answer)).run(BODIES)
    assert run.verdicts == () and run.direction_dropped == 2


def test_the_check_can_be_switched_off_and_is_on_by_default() -> None:
    client = FakeClient(direction="reverse")
    run = arbiter(client, direction_check=False).run(BODIES)
    assert client.direction_calls == 0
    assert {(v.older, v.newer) for v in run.verdicts} == {(OLD, NEW), (OLD_B, NEW_B)}
    on = arbiter_settings({"RECALL_SUPERSESSION_ARBITER": "1"})
    off = arbiter_settings({"RECALL_SUPERSESSION_ARBITER": "1", "RECALL_ARBITER_DIRECTION_CHECK": "0"})
    assert on is not None and on.direction_check is True
    assert off is not None and off.direction_check is False
    with pytest.raises(ValueError, match="RECALL_ARBITER_DIRECTION_CHECK"):
        arbiter_settings({"RECALL_SUPERSESSION_ARBITER": "1", "RECALL_ARBITER_DIRECTION_CHECK": "maybe"})


def test_the_gate_keys_are_unchanged_and_the_questions_never_share_one() -> None:
    a = arbiter(FakeClient())
    digest = hashlib.sha256()
    for part in (a.provider_id, "test/model", "r1", arb.SYSTEM_PROMPT, "x", "y"):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    # The key every gate answer cached before the check existed: an old cache stays valid.
    assert a.cache_key("x", "y") == digest.hexdigest()
    assert a.cache_key("x", "y", arb.DIRECTION_PROMPT) != a.cache_key("x", "y")


def test_a_failed_direction_call_drops_the_pair_and_is_retried(tmp_path: Path) -> None:
    cache = ArbiterCache(tmp_path / "arbiter.sqlite3")
    try:
        first = arbiter(FakeClient(fail_direction=1), cache).run(BODIES)
        assert len(first.verdicts) == 1 and first.direction_dropped == 1 and first.failed == 1
        retry = FakeClient()
        second = arbiter(retry, cache).run(BODIES)
        assert (retry.gate_calls, retry.direction_calls) == (0, 1)
        assert len(second.verdicts) == 2
    finally:
        cache.close()


def test_a_cache_only_run_calls_nothing_and_counts_what_it_skipped(tmp_path: Path) -> None:
    """15. Cache-only never calls the model: with an empty cache it proposes nothing and says so."""
    cache = ArbiterCache(tmp_path / "arbiter.sqlite3")
    try:
        client = FakeClient()
        run = SupersessionArbiter(
            client=client, settings=settings(), cache=cache, workers=2, cache_only=True
        ).run(BODIES)
        assert client.calls == 0
        assert run.verdicts == () and run.called == 0
        assert run.uncached == 2  # the two gate questions; nothing gated, so no direction question
        assert "2 not in the cache and skipped (cache-only)" in run.summary()
    finally:
        cache.close()


def test_a_cache_only_run_serves_what_a_paid_run_already_cached(tmp_path: Path) -> None:
    """16. After one paid run, a cache-only run proposes the same pairs and spends nothing."""
    cache = ArbiterCache(tmp_path / "arbiter.sqlite3")
    try:
        paid = arbiter(FakeClient(), cache).run(BODIES)
        never = FakeClient()
        free = SupersessionArbiter(
            client=never, settings=settings(), cache=cache, workers=2, cache_only=True
        ).run(BODIES)
        assert never.calls == 0 and free.uncached == 0
        assert {(v.older, v.newer) for v in free.verdicts} == {(v.older, v.newer) for v in paid.verdicts}
    finally:
        cache.close()


def test_a_cache_only_arbiter_needs_no_key_and_cannot_call(tmp_path: Path, monkeypatch) -> None:
    """17. `resolve_arbiter(cache_only=True)` builds without an API key, and its client refuses."""
    monkeypatch.delenv("RECALL_ARBITER_API_KEY", raising=False)
    env = {"RECALL_SUPERSESSION_ARBITER": "1", "RECALL_ARBITER_CACHE": str(tmp_path / "c.sqlite3")}
    resolved = arb.resolve_arbiter(env, cache_only=True)
    assert resolved is not None
    try:
        with pytest.raises(RuntimeError, match="cache-only"):
            resolved._ask("a", "b")
        assert resolved.run(BODIES).called == 0
    finally:
        resolved.close()
