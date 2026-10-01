"""W0 session task keys, stage 1: session-level keys as a separate leg over C9's Coding results.

Pre-registration: kept in the maintainer's private research log (W0 session task keys stage 1,
2026-10-01).

Per session of the frozen Coding corpus, two key sources: U, the session's own user messages (no
LLM), and G, short task requests written by gpt-4o-mini from that session's transcript alone. Keys
are embedded with each arm's own model and ranked per task by maximum cosine; that key rank is
fused with the session vote over R1-Coding's recorded top 100, and windows are laid out one per
session first. Nothing is added to the window index.

    python scripts/aml_w0_task_keys.py generate --amb-root AMB --out keys.jsonl --max-usd 1
    python scripts/aml_w0_task_keys.py embed --amb-root AMB --keys keys.jsonl --out key-scores.json
    python scripts/aml_w0_task_keys.py score --a A.json.gz --p P.json.gz --amb-root AMB \
        --key-scores key-scores.json --out task-keys.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RRF_K = 60
BAR = 0.05
MODEL = "openai/gpt-4o-mini"
#: USD per million input and output tokens for gpt-4o-mini (OpenAI list price, as in the AML rules).
PRICE_IN, PRICE_OUT = 0.15, 0.60
KEY_CHARS = 2000
PROFILES = {"A": "voyage-code-4-v1", "P": "qwen3-embedding-8b-mrl1024-openrouter-deepinfra-memory-instruct-v1"}
SOURCES = ("U", "G")
#: (name, base, w); w None is key rank alone; the order breaks leave-one-out ties.
VARIANTS: tuple[tuple[str, str, float | None], ...] = (
    ("served", "served", -1.0),
    ("first", "served", 0.0),
    ("K(1)", "served", 1.0),
    ("K(2)", "served", 2.0),
    ("Konly", "served", None),
    ("L2+first", "L2", 0.0),
    ("L2+K(1)", "L2", 1.0),
    ("L2+K(2)", "L2", 2.0),
)
SYSTEM_PROMPT = (
    "You read one past software engineering session: a transcript between a developer and a coding "
    "agent. Write 3 to 5 short task requests, each one sentence in a developer's voice, that this "
    "session's work fulfils, so that someone facing a similar task later could find this session. "
    "Then write one line naming the repository or project, the files touched, and the outcome. Use "
    "only facts present in the transcript. Reply as JSON: "
    '{"tasks": ["...", "..."], "files_and_outcome": "..."}'
)


# ---------------------------------------------------------------------------------------------
# Keys


def transcript(messages: Sequence[dict[str, Any]]) -> str:
    return "\n\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)


def user_keys(messages: Sequence[dict[str, Any]]) -> list[str]:
    """Key source U: the session's own user messages, each cut to KEY_CHARS."""
    return [str(m["content"])[:KEY_CHARS] for m in messages if m.get("role") == "user" and str(m.get("content", "")).strip()]


def validate_keys(raw: str) -> list[str]:
    """Key source G from the model's JSON: 3 to 5 task strings plus the files line; refuses
    anything else (the session then has no G key)."""
    data = json.loads(raw)
    tasks = data.get("tasks") if isinstance(data, dict) else None
    files = data.get("files_and_outcome") if isinstance(data, dict) else None
    if not isinstance(tasks, list) or not 3 <= len(tasks) <= 5 or not all(isinstance(t, str) and t.strip() for t in tasks):
        raise ValueError("tasks must be 3 to 5 non-empty strings")
    if not isinstance(files, str) or not files.strip():
        raise ValueError("files_and_outcome must be a non-empty string")
    return [t.strip()[:KEY_CHARS] for t in tasks] + [files.strip()[:KEY_CHARS]]


def _sessions(amb_root: Path) -> dict[str, list[dict[str, Any]]]:
    from aml_c7_qualification import load_frozen_corpus
    from aml_c9_coding_window_check import event_messages

    corpus = load_frozen_corpus(amb_root)
    return {relative: event_messages(corpus.corpus_root / relative) for relative in sorted(corpus.sessions)}


def _chat(text: str) -> tuple[str, int, int]:
    body = json.dumps({
        "model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    usage = data.get("usage") or {}
    return data["choices"][0]["message"]["content"], int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def generate(args: argparse.Namespace) -> None:
    done: dict[str, dict[str, Any]] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            previous = json.loads(line)
            done[previous["session_id"]] = previous
    spent = sum(r["tokens_in"] * PRICE_IN + r["tokens_out"] * PRICE_OUT for r in done.values()) / 1e6
    for session_id, messages in _sessions(args.amb_root).items():
        if session_id in done:
            continue
        if spent > args.max_usd:
            print(json.dumps({"stopped": "cap", "usd": round(spent, 4)}), flush=True)
            return
        record: dict[str, Any] = {"session_id": session_id, "user_keys": user_keys(messages)}
        try:
            raw, tokens_in, tokens_out = _chat(transcript(messages))
        except Exception as error:  # noqa: BLE001, a failed call is recorded, never retried silently
            raw, tokens_in, tokens_out = "", 0, 0
            record["error"] = f"call: {type(error).__name__}"
        record.update({"tokens_in": tokens_in, "tokens_out": tokens_out, "raw": raw})
        if raw:
            try:
                record["g_keys"] = validate_keys(raw)
            except (ValueError, json.JSONDecodeError) as error:
                record["error"] = f"invalid: {error}"
        spent += (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
        with args.out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")
    print(json.dumps({"done": True, "usd": round(spent, 4)}), flush=True)


# ---------------------------------------------------------------------------------------------
# Embedding: per arm, source, task and session, the maximum cosine of the task with the keys


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def key_scores(queries: dict[str, list[float]], keys: dict[str, list[list[float]]]) -> dict[str, dict[str, float]]:
    """task -> session -> max cosine over that session's keys (sessions without keys are absent)."""
    return {t: {s: max(_cosine(q, k) for k in vecs) for s, vecs in keys.items() if vecs} for t, q in queries.items()}


def embed(args: argparse.Namespace) -> None:
    from recall.embedding_registry import registered_profile
    from recall.embeddings import embed_passages, embed_query

    from aml_w0_r1_coding import _questions

    questions = _questions(args.amb_root)
    records = [json.loads(line) for line in args.keys.read_text(encoding="utf-8").splitlines()]
    out: dict[str, Any] = {}
    for arm, profile in PROFILES.items():
        embedder = registered_profile(profile).build(env=os.environ)
        queries = {t: list(embed_query(embedder, q.query)) for t, q in questions.items()}
        out[arm] = {}
        for source in SOURCES:
            field = "user_keys" if source == "U" else "g_keys"
            texts = [(r["session_id"], k) for r in records for k in (r.get(field) or [])]
            vectors = embed_passages(embedder, [k for _, k in texts])
            keys: dict[str, list[list[float]]] = {}
            for (session_id, _), vector in zip(texts, vectors, strict=True):
                keys.setdefault(session_id, []).append(list(vector))
            out[arm][source] = key_scores(queries, keys)
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({"arms": list(out), "tasks": len(questions)}))


# ---------------------------------------------------------------------------------------------
# Scoring


def keyed_order(base: Sequence[int], sessions: Sequence[str], scores: dict[str, float], w: float | None) -> list[int]:
    """Window indices: each session's best window, sessions by 1/(k + vote rank) + w/(k + key rank)
    (``w`` None: key rank alone, ties by vote rank), then the remaining windows in base order.
    Sessions without a key score take the last key rank."""
    from aml_w0_r1_offline import session_vote_order

    vote_sessions = []
    for index in session_vote_order(base, sessions, "first"):
        if sessions[index] not in vote_sessions:
            vote_sessions.append(sessions[index])
    vote_rank = {s: r for r, s in enumerate(vote_sessions, start=1)}
    by_key = sorted(vote_sessions, key=lambda s: (-scores.get(s, -math.inf), vote_rank[s]))
    key_rank = {s: r for r, s in enumerate(by_key, start=1)}
    if w is None:
        ordered = sorted(vote_sessions, key=lambda s: (key_rank[s], vote_rank[s]))
    else:
        ordered = sorted(vote_sessions, key=lambda s: (-(1 / (RRF_K + vote_rank[s]) + w / (RRF_K + key_rank[s])), vote_rank[s]))
    best: dict[str, int] = {}
    for index in base:
        best.setdefault(sessions[index], index)
    head = [best[s] for s in ordered]
    chosen = set(head)
    return head + [i for i in base if i not in chosen]


def variant_rr(items: Sequence[dict[str, Any]], query: str, gold: frozenset[str], scores: dict[str, float]) -> dict[str, float]:
    from aml_w0_r1_offline import lexical_order, reciprocal_rank

    sessions = [str(i["session_id"]) for i in items]
    bases = {"served": list(range(len(items))), "L2": lexical_order(query, [str(i["content"]) for i in items], 2.0)}
    out = {}
    for name, base, w in VARIANTS:
        order = bases[base] if name == "served" else keyed_order(bases[base], sessions, scores, w)
        out[name] = reciprocal_rank([sessions[k] for k in order], gold)
    return out


def verdicts(contrasts: dict[str, dict[str, float]]) -> dict[str, bool]:
    """The pre-registered rules on lower bounds: G helps the proxy (vs P served, above 0), G adds to
    the vote (vs L2+first on P, above 0), G closes the gap (vs A served, above -0.05)."""
    return {
        "keys_help_proxy": contrasts["LOO-G(P)-vs-P"]["ci95_low"] > 0,
        "keys_add_to_vote": contrasts["LOO-G(P)-vs-L2first(P)"]["ci95_low"] > 0,
        "keys_close_gap": contrasts["LOO-G(P)-vs-A"]["ci95_low"] > -BAR,
    }


def score(rows: dict[str, dict[str, Any]], questions: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    from aml_w0_embedding_compare import paired_bootstrap
    from aml_w0_r1_offline import loo_select

    def interval(control: list[float], treated: list[float]) -> dict[str, Any]:
        iv = paired_bootstrap(control, treated, scale=1.0)
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in iv.items()}

    tasks = sorted(t for t in rows["A"] if t in rows["P"] and t in questions)
    per: dict[tuple[str, str], dict[str, list[float]]] = {}
    for arm in ("A", "P"):
        for source in SOURCES:
            table: dict[str, list[float]] = {name: [] for name, _, _ in VARIANTS}
            for t in tasks:
                rr = variant_rr(rows[arm][t]["served_facts"]["kept"], questions[t].query, questions[t].gold_sessions, scores[arm][source].get(t, {}))
                for name in table:
                    table[name].append(rr[name])
            per[(arm, source)] = table
    loo = {key: loo_select(table) for key, table in per.items()}
    served = {arm: per[(arm, "G")]["served"] for arm in ("A", "P")}
    contrasts = {
        "LOO-G(P)-vs-P": interval(served["P"], loo[("P", "G")][0]),
        "LOO-G(P)-vs-L2first(P)": interval(per[("P", "G")]["L2+first"], loo[("P", "G")][0]),
        "LOO-G(P)-vs-A": interval(served["A"], loo[("P", "G")][0]),
        "LOO-G(P)-vs-LOO-U(P)": interval(loo[("P", "U")][0], loo[("P", "G")][0]),
        "LOO-G(A)-vs-A": interval(served["A"], loo[("A", "G")][0]),
        "LOO-U(A)-vs-A": interval(served["A"], loo[("A", "U")][0]),
    }
    return {
        "n": len(tasks),
        "variants": {f"{arm}/{source}": {name: round(sum(v) / len(v), 4) for name, v in table.items()} for (arm, source), table in per.items()},
        "loo": {f"{arm}/{source}": {"mrr": round(sum(loo[(arm, source)][0]) / len(tasks), 4), "chosen": dict(Counter(loo[(arm, source)][1]))} for (arm, source) in per},
        "contrasts": contrasts,
        "verdicts": verdicts(contrasts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--amb-root", type=Path, required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--max-usd", type=float, default=1.0)
    emb = sub.add_parser("embed")
    emb.add_argument("--amb-root", type=Path, required=True)
    emb.add_argument("--keys", type=Path, required=True)
    emb.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score")
    for flag in ("--a", "--p", "--amb-root", "--key-scores", "--out"):
        sc.add_argument(flag, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "generate":
        generate(args)
    elif args.mode == "embed":
        embed(args)
    else:
        from aml_w0_r1_coding import _questions, _rows

        result = score({"A": _rows(args.a), "P": _rows(args.p)}, _questions(args.amb_root), json.loads(args.key_scores.read_text(encoding="utf-8")))
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
