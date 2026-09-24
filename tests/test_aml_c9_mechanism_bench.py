"""Contract proofs for the live C9 mechanism benchmark (``scripts/aml_c9_mechanism_bench.py``).

The benchmark is only worth running if each hard check can go red. Red proof receipts, each a
one-line mutation of the script, applied, the named test run red in its intended assertion, and
reverted (2026-09-24):

* ``test_a_graph_fallback_fails_the_run``: deleting the
  ``x-recall-graph-fallback") == "0"`` clause of the ``_graph_ok`` check leaves
  ``result["passed"]`` True.
* ``test_an_inactive_atomic_rescue_fails_the_run``: deleting the
  ``x-recall-atomic-rescue-active") == "1"`` clause leaves ``result["passed"]`` True.
* ``test_a_cross_user_leak_fails_the_run``: replacing ``_well_formed(leak, {other_session})``
  with ``leak.status == 200`` leaves ``isolation_no_leak`` True.
* ``test_an_accepted_conflicting_replay_fails_the_run``: replacing ``conflict.status == 409``
  with ``conflict.status in (200, 409)`` leaves ``add_conflict_409`` True.
* ``test_a_failed_add_still_deletes_both_users``: moving the delete loop out of ``finally``
  (to after the ``try`` body) leaves ``/v1/delete`` uncalled when an Add raises.
* ``test_every_answer_is_planted_exactly_once``: making ``_assert_answers_are_planted`` return
  immediately lets a filler sentence carrying an answer token through.
"""

from __future__ import annotations

import json
import random
import re
import threading

import pytest

import scripts.aml_c9_mechanism_bench as bench
from recall_aml.specialists import route_query
from scripts.aml_c9_mechanism_bench import NEEDLES, UNANSWERABLE, Call, build_corpus, run


class _FakeC9:
    """A service stub with C9's envelope: receipts by request id, per-user stores, headers."""

    def __init__(self, **faults: object) -> None:
        self.faults = faults
        self.paths: list[str] = []
        self.receipts: dict[tuple[str, str], tuple[str, dict]] = {}
        self.items: dict[str, list[dict]] = {}
        self.lock = threading.Lock()
        self.add_calls: dict[str, int] = {}

    def _headers(self, query: str) -> dict[str, str]:
        return {
            "x-recall-specialist-route": route_query(query),
            "x-recall-graph-attempted": "1",
            "x-recall-graph-fallback": "1" if self.faults.get("graph_fallback") else "0",
            "x-recall-graph-invalid-relations": "0",
            "x-recall-graph-relation-hits": "2",
            "x-recall-atomic-rescue-attempted": "1",
            "x-recall-atomic-rescue-active": "0" if self.faults.get("atomic_inactive") else "1",
            "x-recall-atomic-rescue-fallback": "0",
            "x-recall-atomic-rescue-candidate-available": "1",
        }

    def call(self, path, payload=None, *, timeout=60.0):
        with self.lock:
            return self._call(path, payload)

    def _call(self, path, payload):
        self.paths.append(path)
        if path == "/version":
            return Call(200, {
                "variant": bench.EXPECTED_VARIANT,
                "git_commit": "abc123",
                "graph_sidecar": True,
                "active_components": {"graph_sidecar": True},
                "atomic_rescue": {"enabled": True, "mode": "active", "placement": "fused"},
                "authorized_user_scope": "platform",
            }, {})
        if path == "/v1/add":
            if self.faults.get("add_raises"):
                raise RuntimeError("transport exploded")
            rid = payload["request_id"]
            self.add_calls[rid] = self.add_calls.get(rid, 0) + 1
            if self.faults.get("first_add_503") and self.add_calls[rid] == 1:
                return Call(503, {"error": "service_unavailable"}, {})
            doomed = self.faults.get("add_always_503")
            if doomed and doomed in payload["user_id"]:
                return Call(503, {"error": "service_unavailable"}, {})
            key = (payload["user_id"], payload["request_id"])
            digest = json.dumps(payload, sort_keys=True)
            if key in self.receipts:
                stored, response = self.receipts[key]
                if stored != digest and not self.faults.get("accept_conflict"):
                    return Call(409, {"error": "request_id conflict"}, {})
                return Call(200, response, {})
            store = self.items.setdefault(payload["user_id"], [])
            for turn, message in enumerate(payload["messages"]):
                store.append({
                    "id": f"{payload['request_id']}:{turn}",
                    "content": message["content"],
                    "source": "aml",
                    "session_id": payload["session_id"],
                    "kind": "raw",
                    "score": 0.5,
                })
            response = {
                "success": True, "status": "stored",
                "request_id": payload["request_id"], "user_id": payload["user_id"],
                "session_id": payload["session_id"],
                "raw_count": 17, "compiled_count": 6, "compiler_fallback": False,
            }
            self.receipts[key] = (digest, response)
            return Call(200, response, {})
        if path == "/v1/search":
            user = payload["user_id"]
            pool = list(self.items.get(user, []))
            if (self.faults.get("leak") and user.startswith("c9-mech-bench-other-")) or (
                self.faults.get("leak_concurrent") and user.startswith("c9-mech-conc-")
            ):
                pool += [item for uid, items in self.items.items() if uid != user
                         for item in items]
            words = {w for w in re.findall(r"\w+", payload["query"].lower()) if len(w) > 3}
            scored = sorted(
                pool,
                key=lambda item: -len(words & set(re.findall(r"\w+", item["content"].lower()))),
            )
            if self.faults.get("nondeterministic") and user.startswith("c9-mech-conc-"):
                random.shuffle(scored)
            return Call(200, {"data": scored[: payload["top_k"]]}, self._headers(payload["query"]))
        if path == "/v1/delete":
            removed = len(self.items.pop(payload["user_id"], []))
            return Call(200, {"status": "deleted", "deleted_count": removed}, {})
        raise AssertionError(path)


def _run(**faults: object) -> tuple[dict, _FakeC9]:
    fake = _FakeC9(**faults)
    return run(fake, sleep=lambda _: None), fake


def test_every_answer_is_planted_exactly_once() -> None:
    corpus = build_corpus(20260924)
    for needle in NEEDLES:
        holders = [
            (session["session"], turn)
            for session in corpus
            for turn, message in enumerate(session["messages"])
            if needle.answer in message["content"]
        ]
        assert holders == [(needle.session, needle.turn)], needle.key
    original = bench._FILLER["garden"][0]
    bench._FILLER["garden"][0] = original + " near Lisbon"
    try:
        with pytest.raises(ValueError, match="Lisbon"):
            build_corpus(20260924)
    finally:
        bench._FILLER["garden"][0] = original


def test_each_query_takes_the_route_the_served_router_chooses() -> None:
    """The expected route per query is C9's own router's answer, so a header mismatch is real."""
    for needle in NEEDLES:
        assert route_query(needle.query) == needle.route, needle.key
    assert {needle.route for needle in NEEDLES} == {"code", "context", "multimodal"}
    assert route_query(UNANSWERABLE) in {"code", "context"}


def test_the_corpus_is_large_enough_for_both_mechanisms() -> None:
    """About 80 windows of 160 words: past the atomic prefix of 5 and the graph prefix of 8."""
    words = sum(
        len(message["content"].split())
        for session in build_corpus(20260924)
        for message in session["messages"]
    )
    assert words > 80 * 120


def test_a_healthy_service_passes_and_cleans_up() -> None:
    result, fake = _run()
    assert result["passed"] is True, result["failed_checks"]
    assert result["summary"]["recall_at_100"] == f"{len(NEEDLES)}/{len(NEEDLES)}"
    assert fake.paths.count("/v1/delete") == 2
    assert fake.items == {}


def test_a_graph_fallback_fails_the_run() -> None:
    result, _ = _run(graph_fallback=True)
    assert result["passed"] is False
    assert any(name.endswith("_graph_ok") for name in result["failed_checks"])


def test_an_inactive_atomic_rescue_fails_the_run() -> None:
    result, _ = _run(atomic_inactive=True)
    assert result["passed"] is False
    assert any(name.endswith("_atomic_active") for name in result["failed_checks"])


def test_a_cross_user_leak_fails_the_run() -> None:
    result, _ = _run(leak=True)
    assert result["checks"]["isolation_no_leak"] is False


def test_an_accepted_conflicting_replay_fails_the_run() -> None:
    result, _ = _run(accept_conflict=True)
    assert result["checks"]["add_conflict_409"] is False


def test_a_failed_add_still_deletes_both_users() -> None:
    fake = _FakeC9(add_raises=True)
    with pytest.raises(RuntimeError):
        run(fake, sleep=lambda _: None)
    assert fake.paths.count("/v1/delete") == 2


# Concurrency stages -------------------------------------------------------------------------


def _conc(**faults: object) -> tuple[dict, _FakeC9]:
    fake = _FakeC9(**faults)
    return bench.run_concurrency(fake, sleep=lambda _: None), fake


def test_the_burst_is_exactly_32_adds_and_128_searches_per_stage() -> None:
    result, fake = _conc()
    assert result["passed"] is True, result["failed_checks"]
    stages = result["stages"]
    assert stages["A_adds"]["count"] == 32 and stages["A_adds"]["stored"] == 32
    assert stages["B_searches"]["count"] == 128 and stages["B_searches"]["ok_200"] == 128
    assert stages["C_mixed_adds"]["count"] == 32
    assert stages["C_mixed_searches"]["count"] == 128
    # 8 users x 14 needles, plus 8 repeated needles; the other 8 are the unanswerable question.
    assert stages["B_searches"]["recall_at_100"] == "120/120"
    assert fake.paths.count("/v1/delete") == 16
    assert fake.items == {}


def test_a_retried_add_still_counts_as_stored_and_is_reported() -> None:
    result, _ = _conc(first_add_503=True)
    assert result["passed"] is True, result["failed_checks"]
    assert result["stages"]["A_adds"]["first_attempt_not_200"] == 32
    assert result["stages"]["A_adds"]["retried_statuses"] == [503]


def test_an_add_that_never_lands_fails_the_burst_and_still_cleans_up() -> None:
    result, fake = _conc(add_always_503="-a-3")
    assert result["checks"]["A_adds_all_stored"] is False
    assert fake.paths.count("/v1/delete") == 16


def test_a_cross_tenant_leak_under_load_fails_the_burst() -> None:
    result, _ = _conc(leak_concurrent=True)
    assert result["checks"]["B_searches_isolated"] is False
    assert result["checks"]["C_searches_isolated"] is False


def test_a_ranking_that_changes_under_load_fails_the_burst() -> None:
    result, _ = _conc(nondeterministic=True)
    assert result["checks"]["B_repeats_identical"] is False


def test_the_search_burst_asks_every_needle_of_every_user() -> None:
    tenants = bench._tenants("u", 8, 1)
    jobs = bench._search_jobs(tenants, 128)
    assert len(jobs) == 128
    asked = {(t.user, n.key if n else None) for t, n in jobs}
    assert len(asked) == 8 * (len(NEEDLES) + 1)


def test_a_crash_mid_burst_still_deletes_every_user() -> None:
    fake = _FakeC9(add_raises=True)
    with pytest.raises(RuntimeError):
        bench.run_concurrency(fake, sleep=lambda _: None)
    assert fake.paths.count("/v1/delete") == 16


def test_the_burst_really_runs_every_job_at_once() -> None:
    """32 jobs that each wait for all 32: a burst narrower than its jobs would never finish."""
    barrier = threading.Barrier(32, timeout=10)
    results, _ = bench._burst([barrier.wait for _ in range(32)])
    assert sorted(results) == list(range(32))
