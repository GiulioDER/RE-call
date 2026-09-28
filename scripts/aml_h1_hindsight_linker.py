"""H1: link a changing fact the way Hindsight consolidates, retrieve then adjudicate, on W1's facts.

Pre-registration: kept in the maintainer's private research log (H1 Hindsight linker, 2026-09-28).

Prior work: the W1 key-link test (``scripts/aml_w1_link_test.py``) extracted typed facts from BEAM
100K and found that the two statements of one changing fact rarely share a ``subject|attribute``
key (link given coverage 0.192); the in-code resolvers v1 to v3 (``recall_aml.key_resolution``)
merge drifted key STRINGS and are scored by ``scripts/aml_w1_key_resolution_v2.py``'s ``arm``.

vectorize-io/hindsight (MIT, read at 26981c6b) never names a slot. Its consolidation job takes each
new fact, recalls the nearest existing observations (dense and lexical lists interleaved so the
semantic twin is always shown), and asks an LLM, under "match by entity and facet, not topic" and
"prefer update over create", whether the fact updates one of them or creates a new one. This ports
that LINKING decision, not Hindsight's observation rewrite, onto W1's frozen facts, so the result
is scored by exactly the metrics v2 and v3 were:

* a raw key already seen keeps its cluster, with no call (as every resolver does);
* a new raw key gets candidates from the clusters that existed before its Add: the top
  ``PER_LEG`` by bge-small cosine over the cluster's fact texts (Hindsight's default embedder, its
  0.3 semantic floor), interleaved with the top ``PER_LEG`` by key-word Jaccard, at most
  ``MAX_CANDIDATES``; with no candidate it opens its own cluster, with no call;
* otherwise one gpt-4o-mini call per Add (Hindsight batches 8 facts; W1 caps an Add at 12) names,
  for each new key, one of ITS candidates or none.

    python scripts/aml_h1_hindsight_linker.py embed  --facts facts.jsonl --out vectors
    python scripts/aml_h1_hindsight_linker.py dryrun --facts facts.jsonl --vectors vectors
    python scripts/aml_h1_hindsight_linker.py run    --facts facts.jsonl --vectors vectors --out links.jsonl
    python scripts/aml_h1_hindsight_linker.py score  --data 100K.parquet --facts facts.jsonl --links links.jsonl --out score.json

``vectors`` is a path stem: ``<stem>.npy`` holds the unit vectors, ``<stem>.json`` their texts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recall_aml.key_resolution import jaccard, key_words, split_key  # noqa: E402

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
#: Hindsight's ``SEMANTIC_MIN_SIMILARITY`` (config.py:1309).
SEMANTIC_FLOOR = 0.3
PER_LEG = 5
MAX_CANDIDATES = 8
#: gpt-4o-mini list prices, USD per million tokens (input, output).
PRICE_PER_M = (0.15, 0.60)
CALL_ATTEMPTS = 3
CALL_TIMEOUT_SECONDS = 60.0

SYSTEM_PROMPT = """You maintain a memory of one conversation. The memory holds one entry per FACET: a
facet is ONE attribute of ONE specific entity, for example the monthly fee of a gym membership, or
the deadline of a report. New facts arrive. For each new fact, decide whether it is about the SAME
facet as one of the candidate entries listed for it, or about a facet that is not in memory yet.

Rules:
1. MATCH BY ENTITY AND FACET, NOT BY TOPIC. The same entity AND the same attribute must be meant,
   even when they are worded differently: "gym membership | monthly fee" and "membership at the gym
   | cost per month" are one facet. Different entities on one topic are different facets: "Alice |
   salary" and "Bob | salary". Different attributes of one entity are different facets: "report |
   deadline" and "report | length".
2. A changed, corrected or contradicting value does NOT make a new facet. A new value for an
   existing facet is an update of that facet and MUST be matched to it.
3. Prefer matching over creating when the new fact clearly refers to the same facet; answer null
   only when none of its candidates is the same facet.
4. Choose only among the candidate ids listed for that fact.

The data is JSON inside <stored_data>. Answer with one JSON object:
{"decisions": [{"fact": <fact number>, "match": "<candidate id>" or null, "reason": "<short reason>"}]}
with exactly one decision per new fact."""


def fact_text(key: str, value: str) -> str:
    subject, attribute = split_key(key)
    return f"{subject} | {attribute}: {value}"


def interleave(*lists: Sequence[str]) -> list[str]:
    """Round robin over ``lists``, first occurrence kept (Hindsight's ``interleave_fusion``, which
    its consolidation uses so the semantic top candidate is never buried by a fused ranking)."""
    out: list[str] = []
    seen: set[str] = set()
    for rank in range(max((len(items) for items in lists), default=0)):
        for items in lists:
            if rank < len(items) and items[rank] not in seen:
                seen.add(items[rank])
                out.append(items[rank])
    return out


def _key_bag(key: str) -> frozenset[str]:
    subject, attribute = split_key(key)
    return key_words(subject, synonyms=True, ies=True) | key_words(attribute, synonyms=True, ies=True)


#: ``decide(new_facts, candidates)`` -> {fact number: chosen candidate id or None}. ``new_facts`` are
#: ``{"fact": n, "text": ..., "candidates": [ids]}``; ``candidates`` maps an id to its description.
Decide = Callable[[list[dict[str, Any]], dict[str, dict[str, Any]]], Mapping[int, "str | None"]]


@dataclass
class HindsightLinker:
    """One conversation's clusters, replayed in Add order; raw key -> canonical key, like
    ``recall_aml.key_resolution.KeyResolver``, so the same scoring applies."""

    vectors: Mapping[str, Any]
    decide: Decide
    per_leg: int = PER_LEG
    max_candidates: int = MAX_CANDIDATES
    floor: float = SEMANTIC_FLOOR
    aliases: dict[str, list[str]] = field(default_factory=dict)
    canonical_of: dict[str, str] = field(default_factory=dict)
    _texts: dict[str, list[str]] = field(default_factory=dict)
    _latest: dict[str, str] = field(default_factory=dict)

    def _dense(self, text: str) -> list[str]:
        import numpy as np

        query = self.vectors.get(text)
        if query is None:
            return []
        scored = []
        for canonical, texts in self._texts.items():
            members = [self.vectors[t] for t in texts if t in self.vectors]
            if not members:
                continue
            best = float(np.max(np.stack(members) @ query))
            if best >= self.floor:
                scored.append((best, canonical))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [canonical for _, canonical in scored[: self.per_leg]]

    def _lexical(self, key: str) -> list[str]:
        bag = _key_bag(key)
        scored = []
        for canonical, names in self.aliases.items():
            best = max(jaccard(bag, _key_bag(name)) for name in names)
            if best > 0:
                scored.append((best, canonical))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [canonical for _, canonical in scored[: self.per_leg]]

    def candidates(self, key: str, value: str) -> list[str]:
        return interleave(self._dense(fact_text(key, value)), self._lexical(key))[: self.max_candidates]

    def add(self, facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Resolve one Add's facts; returns the decision record (candidates shown, choices)."""
        new: dict[str, str] = {}
        for fact in facts:
            if fact["key"] not in self.canonical_of and fact["key"] not in new:
                new[fact["key"]] = str(fact["value"])
        shown: dict[str, list[str]] = {key: self.candidates(key, value) for key, value in new.items()}
        asked = [key for key in new if shown[key]]
        ids: dict[str, str] = {}
        described: dict[str, dict[str, Any]] = {}
        for key in asked:
            for canonical in shown[key]:
                if canonical not in ids:
                    cid = f"C{len(ids) + 1}"
                    ids[canonical] = cid
                    described[cid] = {"facet": canonical, "also_named": self.aliases[canonical][1:4],
                                      "latest_value": self._latest.get(canonical, "")}
        payload = [{"fact": n + 1, "text": fact_text(key, new[key]), "candidates": [ids[c] for c in shown[key]]}
                   for n, key in enumerate(asked)]
        choices: Mapping[int, str | None] = self.decide(payload, described) if payload else {}
        by_id = {cid: canonical for canonical, cid in ids.items()}
        invalid = 0
        chosen: dict[str, str | None] = {}
        for n, key in enumerate(asked):
            pick = choices.get(n + 1)
            target = by_id.get(pick) if isinstance(pick, str) else None
            if pick is not None and (target is None or target not in shown[key]):
                invalid += 1
                target = None
            chosen[key] = target
        for key in new:
            canonical = chosen.get(key) or key
            self.aliases.setdefault(canonical, []).append(key)
            self.canonical_of[key] = canonical
        for fact in facts:
            canonical = self.canonical_of[fact["key"]]
            text = fact_text(fact["key"], str(fact["value"]))
            if text not in self._texts.setdefault(canonical, []):
                self._texts[canonical].append(text)
            self._latest[canonical] = str(fact["value"])
        return {"new": list(new), "asked": asked, "shown": shown, "chosen": chosen, "invalid": invalid}

    def largest_cluster(self) -> int:
        return max((len(names) for names in self.aliases.values()), default=0)


def _add_index(record: Mapping[str, Any]) -> int:
    return int(str(record["add"]).rsplit(":a", 1)[1])


def by_conversation(records: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        out[int(record["conversation"])].append(record)
    return {c: sorted(adds, key=_add_index) for c, adds in sorted(out.items())}


def keys_from_links(
    records: Iterable[dict[str, Any]], links: Mapping[int, Mapping[str, str]]
) -> tuple[dict[int, dict[int, set[str]]], dict[int, tuple[int, int]]]:
    """``resolved_keys``'s output shape from a stored mapping: canonical keys per conversation per
    turn, and each conversation's (largest cluster, raw keys merged into another)."""
    keys: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    largest: dict[int, tuple[int, int]] = {}
    for conversation, adds in by_conversation(records).items():
        mapping = links.get(conversation, {})
        for record in adds:
            for fact in record["facts"]:
                for turn in fact["turns"]:
                    keys[conversation][int(turn)].add(mapping.get(fact["key"], fact["key"]))
        clusters: dict[str, set[str]] = defaultdict(set)
        for raw, canonical in mapping.items():
            clusters[canonical].add(raw)
        largest[conversation] = (max((len(v) for v in clusters.values()), default=1),
                                 sum(len(v) - 1 for v in clusters.values()))
    return keys, largest


def records_from_links(records: Iterable[dict[str, Any]], links: Mapping[int, Mapping[str, str]]) -> list[dict[str, Any]]:
    """The records with every fact's key replaced by its canonical key (for the W3 check)."""
    out = []
    for conversation, adds in by_conversation(records).items():
        mapping = links.get(conversation, {})
        for record in adds:
            out.append({**record, "facts": [{**f, "key": mapping.get(f["key"], f["key"])} for f in record["facts"]]})
    return out


def parse_decisions(raw: Mapping[str, Any]) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    for entry in raw.get("decisions") or []:
        if not isinstance(entry, Mapping):
            continue
        try:
            number = int(entry.get("fact"))
        except (TypeError, ValueError):
            continue
        match = entry.get("match")
        out[number] = match.strip() if isinstance(match, str) and match.strip() else None
    return out


# ---------------------------------------------------------------------------------------- vectors


def save_vectors(stem: Path, texts: Sequence[str], vectors: Any) -> None:
    import numpy as np

    np.save(stem.with_suffix(".npy"), np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    stem.with_suffix(".json").write_text(json.dumps(list(texts), ensure_ascii=False), encoding="utf-8")


def load_vectors(stem: Path) -> dict[str, Any]:
    import numpy as np

    vectors = np.load(stem.with_suffix(".npy"), allow_pickle=False)
    texts = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return dict(zip(texts, vectors, strict=True))


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed(args: argparse.Namespace) -> None:
    import numpy as np
    from fastembed import TextEmbedding

    records = _read_records(args.facts)
    texts = sorted({fact_text(f["key"], str(f["value"])) for r in records for f in r["facts"]})
    model = TextEmbedding(EMBED_MODEL, threads=args.threads)
    vectors = np.asarray(list(model.embed(texts, batch_size=64)), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    save_vectors(args.out, texts, vectors)
    print(json.dumps({"texts": len(texts), "dim": int(vectors.shape[1]), "model": EMBED_MODEL}))


# ------------------------------------------------------------------------------------------ run


class Usage:
    """Own token accounting: the OpenRouter key is shared, so a balance delta is not this run's."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt = self.completion = self.calls = 0

    @property
    def usd(self) -> float:
        return (self.prompt * PRICE_PER_M[0] + self.completion * PRICE_PER_M[1]) / 1e6

    def wrap(self, client: Any) -> Any:
        create = client.chat.completions.create

        def counted(*a: Any, **kw: Any) -> Any:
            response = create(*a, **kw)
            usage = getattr(response, "usage", None)
            with self.lock:
                self.calls += 1
                self.prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.completion += int(getattr(usage, "completion_tokens", 0) or 0)
            return response

        client.chat.completions.create = counted
        return client


def run(args: argparse.Namespace) -> None:
    from aml_w1_link_test import _balance

    from recall_aml.__main__ import build_openrouter_client
    from recall_aml.compiler import OpenAICompiler

    usage = Usage()
    compiler = OpenAICompiler(usage.wrap(build_openrouter_client(os.environ["OPENROUTER_API_KEY"])))
    vectors = load_vectors(args.vectors)
    conversations = by_conversation(_read_records(args.facts))
    if args.only:
        conversations = {c: v for c, v in conversations.items() if c in set(args.only)}
    done: set[int] = set()
    if args.out.exists():
        done = {row["conversation"] for row in _read_records(args.out) if row.get("final") and not row.get("error")}
    print(json.dumps({"balance_start": round(_balance(), 3), "conversations": len(conversations),
                      "already_done": sorted(done)}), flush=True)
    lock = threading.Lock()
    stop = threading.Event()

    def decide(new_facts: list[dict[str, Any]], candidates: dict[str, dict[str, Any]]) -> Mapping[int, str | None]:
        for attempt in range(4):
            if stop.is_set():
                raise RuntimeError("stopped: budget")
            try:
                raw = compiler.json_object(SYSTEM_PROMPT, {"new_facts": new_facts, "candidates": candidates},
                                           attempts=CALL_ATTEMPTS, timeout_seconds=CALL_TIMEOUT_SECONDS)
                return parse_decisions(raw)
            except Exception as exc:  # BROAD-CATCH: 429 and provider errors back off, then give up
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1) + (20 if "429" in str(exc) else 0))
        return {}

    def conversation(item: tuple[int, list[dict[str, Any]]]) -> None:
        index, adds = item
        if index in done or stop.is_set():
            return
        linker = HindsightLinker(vectors, decide)
        rows = []
        error = None
        for record in adds:
            try:
                decision = linker.add(record["facts"])
            except Exception as exc:  # BROAD-CATCH: a failed conversation is recorded, not linked
                error = f"{record['add']}: {type(exc).__name__}: {exc}"[:300]
                break
            rows.append({"conversation": index, "add": record["add"], **decision})
            if usage.usd > args.max_usd:
                stop.set()
        final = {"conversation": index, "final": True, "error": error,
                 "mapping": dict(linker.canonical_of), "largest_cluster": linker.largest_cluster()}
        with lock:
            with args.out.open("a", encoding="utf-8") as sink:
                for row in rows:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.write(json.dumps(final, ensure_ascii=False) + "\n")
            print(json.dumps({"conversation_done": index, "error": error, "calls": usage.calls,
                              "own_usd": round(usage.usd, 4)}), flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(conversation, conversations.items()))
    print(json.dumps({"calls": usage.calls, "prompt_tokens": usage.prompt, "completion_tokens": usage.completion,
                      "own_usd": round(usage.usd, 4), "stopped": stop.is_set(),
                      "balance_end": round(_balance(), 3)}), flush=True)


# ---------------------------------------------------------------------------------------- score


def load_links(path: Path) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]], list[str]]:
    """Each conversation's final mapping (the last final row wins), its decisions, and errors."""
    finals: dict[int, dict[str, str]] = {}
    decisions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pending: dict[int, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for row in _read_records(path):
        conversation = int(row["conversation"])
        if not row.get("final"):
            pending[conversation].append(row)
            continue
        if row.get("error"):
            errors.append(row["error"])
            pending[conversation] = []
            continue
        finals[conversation] = row["mapping"]
        decisions[conversation] = pending[conversation]
        pending[conversation] = []
    return finals, [d for rows in decisions.values() for d in rows], errors


def reachability(
    pairs_by_conversation: Mapping[int, Sequence[dict[str, Any]]], records: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]], links: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    """Diagnostic, not gated: of the covered knowledge-update pairs not already sharing a raw key,
    how often a later-side key new at its Add was SHOWN a candidate cluster holding an earlier-side
    key, and how often that cluster was CHOSEN. Retrieval bounds linking; this says which failed."""
    turns_of_key: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        for fact in record["facts"]:
            turns_of_key[int(record["conversation"])][fact["key"]].update(int(t) for t in fact["turns"])
    shown_of: dict[int, dict[str, list[str]]] = defaultdict(dict)
    chosen_of: dict[int, dict[str, str | None]] = defaultdict(dict)
    for row in decisions:
        shown_of[int(row["conversation"])].update(row["shown"])
        chosen_of[int(row["conversation"])].update(row["chosen"])
    counts = {"covered": 0, "same_raw_key": 0, "shown": 0, "chosen": 0}
    for index, pairs in pairs_by_conversation.items():
        keys = turns_of_key.get(index, {})
        mapping = links.get(index, {})
        for pair in pairs:
            if pair["type"] != "knowledge_update":
                continue
            left = {k for k, turns in keys.items() if turns & set(pair["left"])}
            right = {k for k, turns in keys.items() if turns & set(pair["right"])}
            if not left or not right:
                continue
            counts["covered"] += 1
            if left & right:
                counts["same_raw_key"] += 1
                continue
            targets = {mapping.get(k, k) for k in left}
            shown = any(set(shown_of[index].get(k, [])) & targets for k in right)
            chosen = any(chosen_of[index].get(k) in targets for k in right)
            counts["shown"] += int(shown)
            counts["chosen"] += int(chosen)
    rest = counts["covered"] - counts["same_raw_key"]
    return {**counts, "shown_rate_of_rest": round(counts["shown"] / rest, 4) if rest else None,
            "chosen_rate_of_shown": round(counts["chosen"] / counts["shown"], 4) if counts["shown"] else None}


def score(args: argparse.Namespace) -> None:
    from aml_w1_key_resolution import evaluate
    from aml_w1_key_resolution_v2 import V1_CHOSEN, arm, current_value_recall
    from aml_w1_key_resolution_v3 import V2_CHOSEN, V3
    from aml_w1_link_test import _rows, pairs_of
    from aml_w3_history_check import check

    rows = _rows(args.data)
    records = _read_records(args.facts)
    links, decisions, errors = load_links(args.links)
    every = list(range(len(rows)))
    missing = [i for i in every if i not in links]

    def h1(event_updates: bool) -> dict[str, Any]:
        keys, largest = keys_from_links(records, links)
        result = evaluate(rows, keys, largest, every)
        history = check(rows, records_from_links(records, links), event_updates=event_updates)
        result["w3"] = history
        result["current_value_recall_given_coverage"] = round(current_value_recall(result, history), 4)
        return result

    out = {
        "conversations_linked": len(links), "conversations_missing": missing, "errors": errors,
        "new_keys": sum(len(d["new"]) for d in decisions), "asked": sum(len(d["asked"]) for d in decisions),
        "merged": sum(1 for d in decisions for v in d["chosen"].values() if v is not None),
        "invalid_choices": sum(d["invalid"] for d in decisions),
        "reachability": reachability({i: pairs_of(rows[i]) for i in every}, records, decisions, links),
        "arms": {
            "h1": h1(event_updates=True),
            "h1_no_event_updates": h1(event_updates=False),
            "exact_keys": arm(rows, records, None, every),
            "v1_chosen": arm(rows, records, V1_CHOSEN, every),
            "v2_chosen": arm(rows, records, V2_CHOSEN, every),
            "v3": arm(rows, records, V3, every, event_updates=True) if V3 else None,
        },
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {name: None if r is None else {
        "link_given_coverage": r["pairs"].get("knowledge_update", {}).get("link_recall_given_coverage"),
        "contradiction_link": r["pairs"].get("contradiction_resolution", {}).get("link_recall"),
        "distractor_rate": r["distractor"]["rate"], "largest_cluster": r["largest_cluster"],
        "current_value_recall_given_coverage": r["current_value_recall_given_coverage"],
    } for name, r in out["arms"].items()}
    print(json.dumps({k: out[k] for k in ("conversations_linked", "conversations_missing", "reachability",
                                          "invalid_choices", "merged", "asked")}
                     | {"errors": len(errors), "arms": summary}, indent=2))


def dryrun(args: argparse.Namespace) -> None:
    """Free: replay with a decide that always answers null, counting calls and prompt sizes."""
    vectors = load_vectors(args.vectors)
    calls = chars = 0

    def decide(new_facts: list[dict[str, Any]], candidates: dict[str, dict[str, Any]]) -> Mapping[int, str | None]:
        nonlocal calls, chars
        calls += 1
        chars += len(SYSTEM_PROMPT) + len(json.dumps({"new_facts": new_facts, "candidates": candidates}))
        return {}

    for adds in by_conversation(_read_records(args.facts)).values():
        linker = HindsightLinker(vectors, decide)
        for record in adds:
            linker.add(record["facts"])
    est_in = chars / 3.6
    print(json.dumps({"calls": calls, "prompt_chars": chars, "est_prompt_tokens": int(est_in),
                      "est_usd_at_150_out_tokens_per_call": round(
                          (est_in * PRICE_PER_M[0] + calls * 150 * PRICE_PER_M[1]) / 1e6, 3)}))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    em = sub.add_parser("embed")
    em.add_argument("--facts", type=Path, required=True)
    em.add_argument("--out", type=Path, required=True)
    em.add_argument("--threads", type=int, default=2)
    for name in ("run", "dryrun"):
        p = sub.add_parser(name)
        p.add_argument("--facts", type=Path, required=True)
        p.add_argument("--vectors", type=Path, required=True)
        if name == "run":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--max-usd", type=float, default=2.0)
            p.add_argument("--only", type=int, nargs="*")
    sc = sub.add_parser("score")
    sc.add_argument("--data", type=Path, required=True)
    sc.add_argument("--facts", type=Path, required=True)
    sc.add_argument("--links", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"embed": embed, "run": run, "score": score, "dryrun": dryrun}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
