"""H1 Hindsight linker: candidate interleaving, the online replay, and the mapping the scorer reads.
Each test names its red proof."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aml_h1_hindsight_linker import (  # noqa: E402
    HindsightLinker,
    fact_text,
    interleave,
    keys_from_links,
    load_links,
    parse_decisions,
    reachability,
)


def _vectors(*groups: list[tuple[str, str]]) -> dict[str, Any]:
    """Unit vectors: facts in one group point the same way, groups are orthogonal."""
    out = {}
    for axis, group in enumerate(groups):
        vector = np.zeros(8, dtype=np.float32)
        vector[axis] = 1.0
        for key, value in group:
            out[fact_text(key, value)] = vector
    return out


def test_interleave_alternates_and_keeps_first_occurrence() -> None:
    """Invariant: the dense top candidate is always first and a candidate in both lists appears
    once, at its earliest position (Hindsight's interleave, so the semantic twin is never buried).

    Red proof: dropping the ``not in seen`` check returns ``b`` twice and fails the equality.
    """
    assert interleave(["a", "b", "c"], ["b", "d"]) == ["a", "b", "d", "c"]
    assert interleave([], ["x"]) == ["x"]


def test_a_chosen_candidate_links_the_new_key_and_a_repeat_costs_no_call() -> None:
    """Invariant: when the adjudicator names a candidate, the drifted key joins that cluster, so the
    two statements of one changing fact share a canonical key; a key already seen is resolved
    without asking again.

    Red proof: ignoring the choice (``canonical = key`` in `HindsightLinker.add`) leaves the drifted
    key in its own cluster and fails the ``canonical_of`` equality.
    """
    first, drifted = ("zoom call|date", "April 21"), ("call with the director|scheduled time", "April 22")
    calls: list[list[dict[str, Any]]] = []

    def decide(new_facts, candidates):
        calls.append(new_facts)
        return {1: new_facts[0]["candidates"][0]}

    linker = HindsightLinker(_vectors([first, drifted]), decide)
    linker.add([{"key": first[0], "value": first[1]}])
    record = linker.add([{"key": drifted[0], "value": drifted[1]}])
    assert linker.canonical_of[drifted[0]] == first[0]
    assert record["chosen"] == {drifted[0]: first[0]}
    linker.add([{"key": drifted[0], "value": "April 23"}])
    assert len(calls) == 1


def test_no_candidate_opens_a_cluster_without_a_call_and_null_keeps_facets_apart() -> None:
    """Invariant: a key with no candidate above the floor and no shared word opens its own cluster
    with no model call; a null answer keeps a same-topic, different-entity key apart.

    Red proof: letting an empty candidate list through to ``decide`` (``asked = list(new)``) makes
    the first Add call the model and fails ``calls == 1``.
    """
    alice, bob, other = ("alice|salary", "90k"), ("bob|salary", "80k"), ("garden|colour", "green")
    calls: list[int] = []

    def decide(new_facts, candidates):
        calls.append(len(new_facts))
        return {1: None}

    linker = HindsightLinker(_vectors([alice, bob], [other]), decide)
    linker.add([{"key": alice[0], "value": alice[1]}])
    linker.add([{"key": other[0], "value": other[1]}])
    linker.add([{"key": bob[0], "value": bob[1]}])
    assert calls == [1]
    assert linker.canonical_of == {alice[0]: alice[0], other[0]: other[0], bob[0]: bob[0]}


def test_a_choice_outside_the_facts_own_candidates_is_refused() -> None:
    """Invariant: the model may only pick among the candidates shown for THAT fact; an id shown for
    another fact, or an unknown id, is counted invalid and treated as a new facet.

    Red proof: removing ``target not in shown[key]`` from the validity check accepts the other
    fact's candidate and fails the ``canonical_of`` equality.
    """
    a, b = ("budget|groceries", "300"), ("gift budget|amount", "200")
    a2, b2 = ("grocery budget|monthly", "350"), ("xmas gift|cap", "250")
    vectors = _vectors([a, a2], [b, b2])

    def decide(new_facts, candidates):
        # Fact 1 (a2) is shown only a's cluster; answering with b's cluster id is not allowed.
        other = next(c for c in candidates if c not in new_facts[0]["candidates"])
        return {1: other, 2: "C99"}

    linker = HindsightLinker(vectors, decide, per_leg=1)
    linker.add([{"key": a[0], "value": a[1]}, {"key": b[0], "value": b[1]}])
    record = linker.add([{"key": a2[0], "value": a2[1]}, {"key": b2[0], "value": b2[1]}])
    assert record["invalid"] == 2
    assert linker.canonical_of[a2[0]] == a2[0] and linker.canonical_of[b2[0]] == b2[0]


def test_keys_from_links_counts_clusters_and_rewrites_turn_keys() -> None:
    """Invariant: the scorer sees canonical keys on every turn a raw key cites, and the largest
    cluster counts raw keys per canonical key, as ``resolved_keys`` does.

    Red proof: counting canonical keys instead of raw keys per cluster (``len(clusters)``) reports
    3 and fails the largest-cluster equality. (A first fixture with exactly two clusters, one of
    them of size two, passed under this mutation: 2 clusters and a largest of 2 coincide.)
    """
    records = [
        {"add": "c0:a0", "conversation": 0, "facts": [{"key": "k1", "value": "v", "turns": [1]}]},
        {"add": "c0:a1", "conversation": 0, "facts": [{"key": "k2", "value": "w", "turns": [5]},
                                                       {"key": "k3", "value": "x", "turns": [6]},
                                                       {"key": "k4", "value": "y", "turns": [7]}]},
    ]
    keys, largest = keys_from_links(records, {0: {"k1": "k1", "k2": "k1", "k3": "k3", "k4": "k4"}})
    assert keys[0][5] == {"k1"} and keys[0][6] == {"k3"}
    assert largest[0] == (2, 1)


def test_load_links_ignores_a_failed_attempt_and_keeps_the_rerun() -> None:
    """Invariant: a conversation whose first attempt failed and was rerun is scored from the rerun
    only; the failed attempt's partial decisions do not leak into the diagnostics.

    Red proof: not clearing ``pending`` after an error row carries the failed attempt's decision in
    and fails the decisions-length equality.
    """
    import json
    import tempfile

    rows = [
        {"conversation": 3, "add": "c3:a0", "new": ["a"], "asked": [], "shown": {}, "chosen": {}, "invalid": 0},
        {"conversation": 3, "final": True, "error": "c3:a1: RuntimeError", "mapping": {}},
        {"conversation": 3, "add": "c3:a0", "new": ["a"], "asked": [], "shown": {}, "chosen": {}, "invalid": 0},
        {"conversation": 3, "final": True, "error": None, "mapping": {"a": "a"}},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "links.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        finals, decisions, errors = load_links(path)
    assert finals == {3: {"a": "a"}} and len(decisions) == 1 and len(errors) == 1


def test_reachability_separates_shown_from_chosen() -> None:
    """Invariant: a pair whose later key was shown the earlier cluster but not linked counts as
    shown and not chosen; that is the split that says whether retrieval or adjudication failed.

    Red proof: counting ``chosen`` from ``shown`` (``chosen = shown``) reports 1 chosen and fails.
    """
    records = [
        {"add": "c0:a0", "conversation": 0, "facts": [{"key": "old", "value": "1", "turns": [1]}]},
        {"add": "c0:a1", "conversation": 0, "facts": [{"key": "new", "value": "2", "turns": [9]}]},
    ]
    decisions = [{"conversation": 0, "shown": {"new": ["old"]}, "chosen": {"new": None}}]
    pairs = {0: [{"type": "knowledge_update", "left": [1], "right": [9]}]}
    out = reachability(pairs, records, decisions, {0: {"old": "old", "new": "new"}})
    assert (out["covered"], out["shown"], out["chosen"]) == (1, 1, 0)


def test_parse_decisions_tolerates_junk() -> None:
    """Invariant: a malformed entry is skipped, an empty match is null, never an exception.

    Red proof: removing the ``isinstance(entry, Mapping)`` guard raises on the string entry.
    """
    raw = {"decisions": ["junk", {"fact": "2", "match": " C1 "}, {"fact": 3, "match": ""}, {"fact": None}]}
    assert parse_decisions(raw) == {2: "C1", 3: None}
