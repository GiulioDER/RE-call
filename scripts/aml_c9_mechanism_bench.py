"""Small pre-run benchmark of a live C9 service: contract, atomizer and grounded graph.

The official AML runs talk to C9 through two calls only, Add and Search. This benchmark uses
exactly those envelopes (``/v1/add`` with request, user and session ids and timestamped messages,
``/v1/search`` with ``top_k: 100``), plus ``/version`` before and ``/v1/delete`` after, so that
everything it proves is proved on the path the platform drives.

Why a new check rather than ``aml_text_route_smoke.py``: that smoke stores three one-line
memories, and the C9 atomic rescue refuses a store with fewer than five dense candidates
(``recall_aml.atomic_views.select_view_rescue``), while the graph can only promote into ranks 9 and
10 (``GRAPH_PROTECTED_PREFIX = 8``). On three memories neither mechanism can do anything, so a
green smoke says nothing about either. Here one throwaway user holds six multi-turn sessions of
roughly 80 raw windows, with short answer-bearing sentences buried in long filler turns, which is
the shape the atomizer exists for, and several answers spread across turns of one session, which
is the shape a grounded ``references`` relation links.

It always deletes both throwaway users at the end, including after a failure. It never touches
another user's data and never calls an AML evaluator.

Hard checks (any failure exits 1): the served variant and components, every Add and Search
status, the Add echo and idempotency contract, read-your-writes after the first Add, item shape
and ``top_k``, user isolation, route headers, graph attempted without fallback and without
invalid relations, atomic rescue attempted and active on the large tenant, and cleanup.
Measured, not gated: answer recall at 10 and 100, graph relation hits, candidates and promotions,
atomic candidate availability and fallback count, compiled record counts and latency. Those are
the numbers a pre-registration predicts; a gate on them would turn a quality question into a
false alarm.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
import random
import statistics
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


EXPECTED_VARIANT = "C9_routed_specialists_grounded_graph_atomic"
#: Statuses AML retries for Add; C9 answers a tenant lock wait with 503.
RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 524})
ADD_ATTEMPTS = 8
SEARCH_ATTEMPTS = 3
ADD_TIMEOUT_S = 1800.0
SEARCH_TIMEOUT_S = 60.0
TOP_K = 100
#: The atomic rescue needs five distinct dense parents; below this many raw windows a fallback is
#: the designed answer rather than a defect, so the large-tenant gate does not apply.
ATOMIC_MIN_WINDOWS = 5


@dataclass(frozen=True)
class Call:
    status: int
    payload: dict[str, Any]
    headers: dict[str, str]
    seconds: float = 0.0
    attempts: int = 1


class Client:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def call(
        self, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 60.0
    ) -> Call:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self._base_url + path,
            data=body,
            method="POST" if body is not None else "GET",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
                return Call(
                    response.status,
                    json.loads(raw) if raw else {},
                    {key.casefold(): value for key, value in response.headers.items()},
                    time.perf_counter() - started,
                )
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"error": raw[:200]}
            return Call(
                exc.code,
                parsed if isinstance(parsed, dict) else {"error": str(parsed)[:200]},
                {key.casefold(): value for key, value in exc.headers.items()},
                time.perf_counter() - started,
            )
        except (URLError, TimeoutError, OSError) as exc:
            # A transport failure is recorded as a status nobody serves, never raised: the
            # benchmark must still reach its cleanup.
            return Call(599, {"error": type(exc).__name__}, {}, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# Corpus. Deterministic for a given seed; answer tokens appear only where they are planted.
# --------------------------------------------------------------------------------------------

_FILLER: dict[str, list[str]] = {
    "garden": [
        "The tomato seedlings on the balcony finally have their second set of leaves",
        "I repotted the basil because the roots were circling the bottom of the pot",
        "Watering in the early morning seems to keep the soil from cracking",
        "The neighbour gave us cuttings from her rosemary hedge last weekend",
        "Slugs got to the lettuce again so I put copper tape around the planter",
        "The compost bin is warm enough now that the peelings break down quickly",
        "I am thinking about a small raised bed for strawberries along the fence",
        "Mulching with straw kept the weeds down far better than I expected",
    ],
    "commute": [
        "The regional train was twenty minutes late because of a signal fault",
        "Cycling along the canal path is faster than the bus when the weather holds",
        "There is roadwork on the ring road so everyone is taking the side streets",
        "I started listening to long history podcasts during the drive",
        "The parking garage near the office raised its monthly rate again",
        "On rainy days the tram is packed and nobody can reach the ticket machine",
        "A colleague suggested leaving fifteen minutes earlier to beat the traffic",
        "The new bike lane on the avenue made the ride feel much safer",
    ],
    "reading": [
        "I am halfway through a long novel about a lighthouse keeper and his family",
        "The library finally got the second volume of that mountaineering memoir",
        "Short essays are easier to read before bed than dense nonfiction",
        "The book club picked a detective story set in a snowbound hotel",
        "I keep a notebook of quotes that I want to look up again later",
        "Audiobooks work well for walking but I lose track of names quickly",
        "The translation of that poetry collection reads very differently from the original",
        "A friend lent me a travel diary about crossing the steppe on horseback",
    ],
    "weather": [
        "It rained all afternoon and the gutters overflowed onto the steps",
        "The forecast promised sun but the fog never lifted over the valley",
        "A strong wind knocked the patio umbrella over twice this week",
        "The first frost came early and caught the late flowers by surprise",
        "Humidity makes the evenings heavy even when the temperature drops",
        "We had a thunderstorm that knocked the power out for an hour",
        "The spring has been mild enough to leave the windows open at night",
        "Snow on the hills made the view from the kitchen window look like a postcard",
    ],
    "cooking": [
        "I tried a slow braised lentil stew with smoked paprika and carrots",
        "The sourdough starter needs feeding twice a day while it is this warm",
        "Roasting the vegetables at a higher heat gave them much better colour",
        "We made fresh pasta on Sunday and the kitchen was covered in flour",
        "A squeeze of lemon at the end lifted the whole pot of soup",
        "The cast iron pan finally has a good seasoning after months of use",
        "I burned the onions while answering the door and had to start again",
        "Homemade stock takes hours but the risotto tastes completely different",
    ],
    "office": [
        "The quarterly all hands ran long because of the question round",
        "Our team moved to the third floor where the light is much better",
        "Someone keeps booking the big meeting room and not showing up",
        "The coffee machine in the kitchen was replaced with a better model",
        "I blocked out mornings for focused work and it has helped a lot",
        "The onboarding checklist for new hires needs a proper cleanup",
        "Hybrid days mean the desks are empty on Mondays and crowded on Wednesdays",
        "We started writing short weekly notes instead of long status meetings",
    ],
}

_CONNECTORS = ["Also,", "Anyway,", "By the way,", "Honestly,", "Meanwhile,", "Oh,", "Still,"]


@dataclass(frozen=True)
class Needle:
    key: str
    session: str
    turn: int
    sentence: str
    answer: str
    query: str
    route: str
    #: What the needle exercises: "atomic" (short fact in a long turn), "graph" (fact linked
    #: across turns of one session), "code", "update", "visual" (PR 719 route guard).
    purpose: str


NEEDLES: tuple[Needle, ...] = (
    Needle("sister_city", "family", 2,
           "Big news, my sister Chiara finally moved to Lisbon and starts at the aquarium in April.",
           "Lisbon", "Which city did my sister Chiara move to?", "code", "atomic"),
    Needle("dog_allergy", "family", 5,
           "The vet said Biscotto, our beagle, reacts badly to chicken, so he only gets salmon kibble now.",
           "salmon kibble", "What food does our beagle eat now because of his allergy?", "code", "atomic"),
    Needle("anniversary", "family", 7,
           "I booked the anniversary dinner at Osteria Ventidue for the evening of the nineteenth.",
           "Ventidue", "Where did I book the anniversary dinner?", "context", "atomic"),
    Needle("tea", "family", 9,
           "For the record, my favourite tea is a smoky lapsang souchong with no sugar at all.",
           "lapsang", "What is my favorite tea?", "context", "atomic"),
    Needle("gym_old", "family", 3,
           "My spin class is on Tuesday evenings at seven, right after work.",
           "Tuesday evenings", "When was my spin class originally scheduled?", "code", "update"),
    Needle("gym_new", "errands", 4,
           "Change of plan, I moved my spin class to Thursday mornings at six from now on.",
           "Thursday mornings", "What day is my spin class now?", "code", "update"),
    Needle("vendor_renewal", "meetings", 2,
           "In the meeting we agreed the freight vendor contract with Halvorsen renews on the first of November.",
           "first of November", "When does the freight vendor contract renew?", "code", "atomic"),
    Needle("budget_owner", "meetings", 6,
           "Priya will own the quarterly budget review and present it to the board herself.",
           "Priya", "Who is presenting the quarterly budget review to the board?", "code", "atomic"),
    Needle("incident_cause", "meetings", 3,
           "The outage on the payments cluster was traced to an expired TLS certificate on node pay-07.",
           "pay-07", "Which node had the expired certificate that caused the payments outage?", "code", "graph"),
    Needle("incident_fix", "meetings", 8,
           "To stop the payments cluster outage happening again, Marek automated certificate renewal with a nightly certbot job.",
           "certbot", "How did we stop the payments outage from happening again?", "code", "graph"),
    Needle("keyerror", "coding", 2,
           "The KeyError came from parse_manifest(path) when a manifest file had no checksum_sha field.",
           "parse_manifest", "Which function raised the KeyError for the missing checksum field?", "code", "code"),
    Needle("pin", "coding", 5,
           "We pinned urllib3 to 1.26.18 because the corporate proxy layer broke on the 2.x series.",
           "1.26.18", "Which urllib3 version did we pin because of the proxy?", "code", "code"),
    Needle("retry_budget", "coding", 7,
           "In worker.yaml the retry budget is max_attempts: 7 with backoff_base: 2.5 seconds.",
           "max_attempts: 7", "What max_attempts value is set in the worker config?", "code", "code"),
    Needle("overlap_fix", "coding", 9,
           "The settings page button overlapped the footer; the fix moved it into the flex container in settings.css.",
           "settings.css", "Is there a screenshot of the settings UI where the button overlapped the footer?",
           "multimodal", "visual"),
)

#: Session ids in Add order; "distractor" sessions carry no needle.
SESSIONS: tuple[tuple[str, int], ...] = (
    ("family", 10),
    ("meetings", 10),
    ("coding", 10),
    ("errands", 6),
    ("distractor-a", 8),
    ("distractor-b", 8),
)
#: A question nothing in the corpus answers; it must still return 200 and a well-formed list.
UNANSWERABLE = "What is the name of my brother's football team?"

_CODE_BLOCK = (
    "Here is the current loader:\n```python\ndef load_all(paths):\n    out = []\n"
    "    for p in paths:\n        out.append(read_one(p))\n    return out\n```\n"
)


def _filler(rng: random.Random, words: int) -> str:
    sentences: list[str] = []
    count = 0
    topics = list(_FILLER)
    while count < words:
        sentence = rng.choice(_FILLER[rng.choice(topics)])
        if rng.random() < 0.3:
            sentence = f"{rng.choice(_CONNECTORS)} {sentence[0].lower()}{sentence[1:]}"
        sentences.append(sentence + ".")
        count += len(sentence.split())
    return " ".join(sentences)


def build_corpus(seed: int, *, start_ms: int = 1_756_000_000_000) -> list[dict[str, Any]]:
    """Sessions as ``{"session": name, "messages": [...]}``, needles buried mid-turn."""
    rng = random.Random(seed)
    placed = {(needle.session, needle.turn): needle for needle in NEEDLES}
    day = 86_400_000
    sessions: list[dict[str, Any]] = []
    for session_index, (name, turns) in enumerate(SESSIONS):
        messages: list[dict[str, Any]] = []
        for turn in range(turns):
            role = "user" if turn % 2 == 0 else "assistant"
            needle = placed.get((name, turn))
            parts = [_filler(rng, 90)]
            if name == "coding" and turn in (4, 6):
                parts.append(_CODE_BLOCK)
            if needle is not None:
                parts.append(needle.sentence)
            parts.append(_filler(rng, 90))
            messages.append(
                {
                    "role": role,
                    "content": " ".join(parts),
                    "timestamp": start_ms + session_index * day + turn * 60_000,
                }
            )
        sessions.append({"session": name, "messages": messages})
    _assert_answers_are_planted(sessions)
    return sessions


def _assert_answers_are_planted(sessions: list[dict[str, Any]]) -> None:
    """Each answer must occur in exactly one message: its needle's. Otherwise a hit proves nothing."""
    for needle in NEEDLES:
        holders = [
            (session["session"], turn)
            for session in sessions
            for turn, message in enumerate(session["messages"])
            if needle.answer in message["content"]
        ]
        if holders != [(needle.session, needle.turn)]:
            raise ValueError(f"answer {needle.answer!r} is held by {holders}, not only its needle")


# --------------------------------------------------------------------------------------------
# Run.
# --------------------------------------------------------------------------------------------


@dataclass
class SearchRecord:
    key: str
    purpose: str
    query: str
    status: int
    attempts: int
    seconds: float
    items: int
    rank: int | None
    route: str | None
    expected_route: str
    headers: dict[str, str] = field(default_factory=dict)


def _retrying(
    client: Client,
    path: str,
    body: dict[str, Any],
    *,
    attempts: int,
    timeout: float,
    sleep: Callable[[float], None],
) -> Call:
    result = Call(599, {}, {})
    started = time.perf_counter()
    for attempt in range(attempts):
        result = client.call(path, body, timeout=timeout)
        if result.status not in RETRYABLE:
            break
        if attempt + 1 < attempts:
            sleep(min(30.0, 2.0 * 2**attempt))
    return Call(
        result.status,
        result.payload,
        result.headers,
        time.perf_counter() - started,
        attempt + 1,
    )


def _rank(call: Call, answer: str) -> int | None:
    for rank, item in enumerate(call.payload.get("data", []), start=1):
        if answer in json.dumps(item.get("content", ""), ensure_ascii=False):
            return rank
    return None


_ITEM_KEYS = {"id", "content", "source", "session_id", "kind", "score"}


def _well_formed(call: Call, sessions: set[str]) -> bool:
    data = call.payload.get("data")
    if not isinstance(data, list) or len(data) > TOP_K:
        return False
    ids = [item.get("id") for item in data if isinstance(item, dict)]
    return (
        len(ids) == len(data)
        and len(set(ids)) == len(ids)
        and all(_ITEM_KEYS <= set(item) for item in data)
        and all(item.get("session_id") in sessions for item in data)
    )


_GRAPH_HEADERS = ("attempted", "fallback", "relation-hits", "candidates", "promoted",
                  "invalid-relations", "top10-order-changed", "top100-membership-changed")
_ATOMIC_HEADERS = ("attempted", "active", "fallback", "candidate-available")


def _diagnostics(call: Call) -> dict[str, str]:
    keep = {f"x-recall-graph-{name}" for name in _GRAPH_HEADERS}
    keep |= {f"x-recall-atomic-rescue-{name}" for name in _ATOMIC_HEADERS}
    keep |= {"x-recall-specialist-route", "x-recall-search-ms", "x-recall-served-commit",
             "x-recall-code-aware-fallback", "x-recall-facet-fallback",
             "x-recall-reranker-fallback"}
    return {key: value for key, value in call.headers.items() if key in keep}


def _header_int(record: SearchRecord, name: str) -> int:
    try:
        return int(record.headers.get(name, "0"))
    except ValueError:
        return 0


def run(
    client: Client,
    *,
    seed: int = 20260924,
    expected_variant: str = EXPECTED_VARIANT,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    nonce = uuid4().hex[:12]
    user = f"c9-mech-bench-{nonce}"
    other = f"c9-mech-bench-other-{nonce}"
    corpus = build_corpus(seed)
    session_id = {s["session"]: f"c9-mech/{nonce}/{s['session']}" for s in corpus}
    other_session = f"c9-mech/{nonce}/other"
    checks: dict[str, bool] = {}
    adds: list[dict[str, Any]] = []
    searches: list[SearchRecord] = []
    extras: dict[str, Any] = {}
    started = clock()

    version = client.call("/version")
    components = version.payload.get("active_components", {}) if version.status == 200 else {}
    atomic_profile = version.payload.get("atomic_rescue") or {}
    checks["version_variant"] = version.payload.get("variant") == expected_variant
    checks["version_graph_sidecar"] = components.get("graph_sidecar") is True and (
        version.payload.get("graph_sidecar") is True
    )
    checks["version_atomic_active"] = isinstance(atomic_profile, dict) and (
        atomic_profile.get("mode") == "active"
    )
    checks["version_platform_scope"] = version.payload.get("authorized_user_scope") == "platform"

    def search(user_id: str, query: str) -> Call:
        return _retrying(
            client, "/v1/search", {"query": query, "user_id": user_id, "top_k": TOP_K},
            attempts=SEARCH_ATTEMPTS, timeout=SEARCH_TIMEOUT_S, sleep=sleep,
        )

    first_body: dict[str, Any] | None = None
    first_response: dict[str, Any] | None = None
    cleanup: dict[str, int] = {}
    try:
        for index, session in enumerate(corpus):
            body = {
                "request_id": f"c9-mech-{nonce}-{index}",
                "user_id": user,
                "session_id": session_id[session["session"]],
                "messages": session["messages"],
            }
            call = _retrying(client, "/v1/add", body, attempts=ADD_ATTEMPTS,
                             timeout=ADD_TIMEOUT_S, sleep=sleep)
            adds.append({
                "session": session["session"],
                "status": call.status,
                "attempts": call.attempts,
                "seconds": round(call.seconds, 3),
                "words": sum(len(m["content"].split()) for m in session["messages"]),
                "raw_count": call.payload.get("raw_count"),
                "compiled_count": call.payload.get("compiled_count"),
                "compiler_fallback": call.payload.get("compiler_fallback"),
                "echo_ok": all(call.payload.get(k) == body[k]
                               for k in ("request_id", "user_id", "session_id")),
            })
            if index == 0:
                first_body, first_response = body, call.payload
                # Read-your-writes: the first Add must be searchable the moment it returns.
                probe = next(n for n in NEEDLES if n.session == session["session"])
                early = search(user, probe.query)
                extras["read_your_writes_rank"] = _rank(early, probe.answer)
                checks["read_your_writes"] = early.status == 200 and (
                    _rank(early, probe.answer) is not None
                )

        # Idempotency: an identical retry answers identically; a changed body under the same
        # request id is a conflict, never a silent overwrite.
        assert first_body is not None
        replay = client.call("/v1/add", first_body, timeout=ADD_TIMEOUT_S)
        checks["add_replay_identical"] = replay.status == 200 and replay.payload == first_response
        altered = dict(first_body, messages=first_body["messages"][:1])
        conflict = client.call("/v1/add", altered, timeout=ADD_TIMEOUT_S)
        extras["conflict_status"] = conflict.status
        checks["add_conflict_409"] = conflict.status == 409

        own_sessions = set(session_id.values())
        for needle in NEEDLES:
            call = search(user, needle.query)
            searches.append(SearchRecord(
                needle.key, needle.purpose, needle.query, call.status, call.attempts,
                round(call.seconds, 3), len(call.payload.get("data", [])),
                _rank(call, needle.answer), call.headers.get("x-recall-specialist-route"),
                needle.route, _diagnostics(call),
            ))
            checks[f"search_{needle.key}_shape"] = call.status == 200 and _well_formed(
                call, own_sessions
            )
        unanswerable = search(user, UNANSWERABLE)
        checks["search_unanswerable_shape"] = unanswerable.status == 200 and _well_formed(
            unanswerable, own_sessions
        )

        # Isolation: a second user with one short session must see only its own session.
        other_add = _retrying(client, "/v1/add", {
            "request_id": f"c9-mech-{nonce}-other",
            "user_id": other,
            "session_id": other_session,
            "messages": [{"role": "user", "content": "My brother supports a small local "
                          "rugby club and never misses a home match.", "timestamp": 1_756_000_000_000}],
        }, attempts=ADD_ATTEMPTS, timeout=ADD_TIMEOUT_S, sleep=sleep)
        leak = search(other, NEEDLES[0].query)
        checks["isolation_other_add"] = other_add.status == 200
        checks["isolation_no_leak"] = leak.status == 200 and _well_formed(leak, {other_session})
    finally:
        for name, uid in (("user", user), ("other", other)):
            deleted = client.call("/v1/delete", {"user_id": uid}, timeout=300.0)
            cleanup[name] = deleted.status
            extras[f"deleted_{name}"] = deleted.payload.get("deleted_count")
        after = search(user, NEEDLES[0].query)
        extras["after_delete_items"] = len(after.payload.get("data", []))
        checks["cleanup_deleted"] = all(status == 200 for status in cleanup.values())
        checks["cleanup_empty_after"] = after.status == 200 and not after.payload.get("data")

    checks["adds_stored"] = len(adds) == len(corpus) and all(a["status"] == 200 for a in adds)
    checks["adds_echo"] = all(a["echo_ok"] for a in adds)
    checks["adds_no_compiler_fallback"] = all(a["compiler_fallback"] is False for a in adds)
    total_windows = sum(a["raw_count"] or 0 for a in adds)
    for record in searches:
        prefix = f"search_{record.key}"
        checks[f"{prefix}_route"] = record.route == record.expected_route
        checks[f"{prefix}_graph_ok"] = (
            record.headers.get("x-recall-graph-attempted") == "1"
            and record.headers.get("x-recall-graph-fallback") == "0"
            and record.headers.get("x-recall-graph-invalid-relations", "0") == "0"
        )
        if total_windows >= ATOMIC_MIN_WINDOWS:
            checks[f"{prefix}_atomic_active"] = (
                record.headers.get("x-recall-atomic-rescue-attempted") == "1"
                and record.headers.get("x-recall-atomic-rescue-active") == "1"
            )

    return {
        "passed": all(checks.values()),
        "failed_checks": sorted(name for name, ok in checks.items() if not ok),
        "checks": checks,
        "served_commit": version.payload.get("git_commit"),
        "variant": version.payload.get("variant"),
        "atomic_profile": atomic_profile,
        "user": user,
        "seed": seed,
        "total_raw_windows": total_windows,
        "adds": adds,
        "searches": [asdict(record) for record in searches],
        "summary": summarize(adds, searches),
        "extras": extras,
        "started_utc": int(started),
        "wall_seconds": round(clock() - started, 3),
    }


def summarize(adds: list[dict[str, Any]], searches: list[SearchRecord]) -> dict[str, Any]:
    ranks = [r.rank for r in searches]

    def within(limit: int, subset: list[SearchRecord]) -> str:
        hits = sum(1 for r in subset if r.rank is not None and r.rank <= limit)
        return f"{hits}/{len(subset)}"

    by_purpose: dict[str, dict[str, str]] = {}
    for purpose in sorted({r.purpose for r in searches}):
        subset = [r for r in searches if r.purpose == purpose]
        by_purpose[purpose] = {"at_10": within(10, subset), "at_100": within(100, subset)}
    search_seconds = [r.seconds for r in searches]
    return {
        "recall_at_1": within(1, searches),
        "recall_at_10": within(10, searches),
        "recall_at_100": within(100, searches),
        "by_purpose": by_purpose,
        "missed": [r.key for r, rank in zip(searches, ranks) if rank is None],
        "graph_relation_hits": sum(_header_int(r, "x-recall-graph-relation-hits") for r in searches),
        "graph_candidates": sum(_header_int(r, "x-recall-graph-candidates") for r in searches),
        "graph_promoted": sum(_header_int(r, "x-recall-graph-promoted") for r in searches),
        "graph_top10_changed": sum(
            _header_int(r, "x-recall-graph-top10-order-changed") for r in searches
        ),
        "searches_with_relation_hits": sum(
            1 for r in searches if _header_int(r, "x-recall-graph-relation-hits") > 0
        ),
        "atomic_candidate_available": sum(
            _header_int(r, "x-recall-atomic-rescue-candidate-available") for r in searches
        ),
        "atomic_fallback": sum(_header_int(r, "x-recall-atomic-rescue-fallback") for r in searches),
        "compiled_records": sum(a["compiled_count"] or 0 for a in adds),
        "add_seconds_max": max((a["seconds"] for a in adds), default=0.0),
        "add_retries": sum(a["attempts"] - 1 for a in adds),
        "search_retries": sum(r.attempts - 1 for r in searches),
        "search_seconds_median": round(statistics.median(search_seconds), 3) if searches else 0.0,
        "search_seconds_max": max(search_seconds, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--expected-variant", default=EXPECTED_VARIANT)
    parser.add_argument("--out", help="also write the JSON report to this path")
    args = parser.parse_args()
    api_key = os.environ.get("RECALL_AML_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RECALL_AML_API_KEY is required")
    report = run(Client(args.base_url, api_key), seed=args.seed,
                 expected_variant=args.expected_variant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
