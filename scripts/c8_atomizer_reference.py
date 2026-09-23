"""C8 atomic rescue reference on the frozen CAMBench coding corpus.

Two subcommands, both run on VPS3 (VPS2 hosts the official AML Full run):

``probes``
    Sample seeded word spans from the rendered sessions, independently of any atomizer, and ask
    pinned ``openai/gpt-4o-mini-2024-07-18`` for one retrieval question per span. Gold is every
    hosted window that fully contains the span. Output is private JSONL.

``evaluate``
    Replay C8's Code4 retrieval offline: exact dense top 100 over the 1,220 content-only windows,
    the production ``insert_atomic_rescue_dense`` on an artifact written and loaded through the
    production writer and loader, canonical BM25, and unweighted RRF with stable window ties, as
    ``recall_aml.retrieval`` does. Each atomizer arm is compared with an explicit off control.

Probe texts and per-query rows are private and stay outside the repository; only aggregate
metrics are meant for publication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.atomizer import window_bounds, window_views  # noqa: E402
from recall.types import Chunk, ScoredChunk  # noqa: E402
from recall_aml.code4 import rank_bm25_chunks, stable_window_key, word_windows  # noqa: E402
from recall_aml.retrieval import _rrf  # noqa: E402


WINDOW_SIZE = 160
WINDOW_STRIDE = 120
CANDIDATE_K = 100
SEED = 20260922
SPAN_MIN_WORDS = 12
SPAN_MAX_WORDS = 30
SPANS_PER_SESSION = 2
CONTEXT_WORDS = 80
MIN_SPAN_CONTENT_WORDS = 6
QUESTION_MIN_WORDS = 5
QUESTION_MAX_WORDS = 45
MAX_COPIED_RUN = 5
MODEL = "openai/gpt-4o-mini-2024-07-18"
PROVIDER_ROUTING = {"order": ["openai"], "allow_fallbacks": False}
PROBE_BUDGET_USD = 1.00
PRICE_IN, PRICE_OUT = 0.15e-6, 0.60e-6
EMBEDDING_PROFILE = "voyage-code-4-v1"
ARMS: dict[str, dict[str, Any]] = {
    "off": {},
    "sentence": {"strategy": "sentence", "min_words": 6, "max_words": 40},
    "micro": {"strategy": "micro", "micro_size": 24, "micro_stride": 12},
}
CUTOFFS = (1, 5, 6, 8, 10)
_WORD = re.compile(r"[0-9a-z]+")
_ALNUM = re.compile(r"[0-9A-Za-z]")

SYSTEM_PROMPT = (
    "You write retrieval test questions about a software engineering session transcript. "
    "The transcript has been lower-cased and flattened, so punctuation and spacing are unusual. "
    "You receive CONTEXT, with the TARGET passage marked between [[ and ]], and the TARGET alone. "
    "Write exactly one question that a developer who took part in this session might ask weeks "
    "later, whose answer is stated in TARGET. Ask about the specific fact, command, value, file, "
    "error, decision or result that TARGET states. Do not copy more than four consecutive words "
    "from TARGET. Do not mention the target, the passage, the context or the transcript. "
    "If TARGET states no specific checkable fact (for example it is only boilerplate, punctuation, "
    "or a generic remark), set answerable to false and question to an empty string."
)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "retrieval_probe",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answerable", "question"],
            "properties": {
                "answerable": {"type": "boolean"},
                "question": {"type": "string"},
            },
        },
    },
}


# --------------------------------------------------------------------------- corpus and windows


@dataclass(frozen=True)
class Window:
    chunk: Chunk
    session: str
    segment: int
    word_start: int
    word_end: int


def chunk_id(session: str, segment: int) -> str:
    return hashlib.sha256(f"{session}\x00{segment}".encode("utf-8")).hexdigest()[:32]


def build_windows(rendered: Mapping[str, str]) -> list[Window]:
    """Rebuild the hosted content-only windows, with the stable identity C8 stores."""

    windows: list[Window] = []
    for session in sorted(rendered):
        text = rendered[session]
        words = text.split()
        texts = word_windows(text, size=WINDOW_SIZE, stride=WINDOW_STRIDE)
        bounds = window_bounds(len(words), size=WINDOW_SIZE, stride=WINDOW_STRIDE)
        if len(texts) != len(bounds):
            raise RuntimeError("window bounds disagree with hosted word windows")
        for segment, (content, (start, end)) in enumerate(zip(texts, bounds, strict=True)):
            windows.append(
                Window(
                    Chunk(
                        id=chunk_id(session, segment),
                        source=f"aml://session/{session}",
                        text=content,
                        metadata={"source_session_id": session, "segment": segment},
                    ),
                    session,
                    segment,
                    start,
                    end,
                )
            )
    return windows


def gold_segments(word_count: int, start: int, end: int) -> list[int]:
    """Every window that fully contains the word range ``[start, end)``."""

    return [
        segment
        for segment, (window_start, window_end) in enumerate(
            window_bounds(word_count, size=WINDOW_SIZE, stride=WINDOW_STRIDE)
        )
        if window_start <= start and end <= window_end
    ]


def split_for(session: str) -> str:
    return "dev" if hashlib.sha256(session.encode("utf-8")).digest()[0] % 2 == 0 else "confirm"


# --------------------------------------------------------------------------- probe sampling


@dataclass(frozen=True)
class Span:
    session: str
    start: int
    end: int
    text: str
    gold: tuple[int, ...]


def sample_spans(rendered: Mapping[str, str], *, seed: int = SEED) -> list[Span]:
    """Draw seeded spans with no reference to any atomizer's segmentation."""

    rng = random.Random(seed)
    spans: list[Span] = []
    for session in sorted(rendered):
        words = rendered[session].split()
        if len(words) < SPAN_MAX_WORDS * 2:
            continue
        taken: list[tuple[int, int]] = []
        attempts = 0
        while len(taken) < SPANS_PER_SESSION and attempts < 50:
            attempts += 1
            length = rng.randint(SPAN_MIN_WORDS, SPAN_MAX_WORDS)
            start = rng.randrange(0, len(words) - length + 1)
            end = start + length
            if any(start < other_end and other_start < end for other_start, other_end in taken):
                continue
            if sum(1 for word in words[start:end] if _ALNUM.search(word)) < MIN_SPAN_CONTENT_WORDS:
                continue
            taken.append((start, end))
        for start, end in sorted(taken):
            spans.append(
                Span(
                    session,
                    start,
                    end,
                    " ".join(words[start:end]),
                    tuple(gold_segments(len(words), start, end)),
                )
            )
    return spans


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def copied_run(question: str, span: str) -> int:
    """Longest run of consecutive span tokens that the question repeats verbatim."""

    q = _tokens(question)
    s = _tokens(span)
    best = 0
    previous = [0] * (len(s) + 1)
    for q_token in q:
        current = [0] * (len(s) + 1)
        for index, s_token in enumerate(s, start=1):
            if q_token == s_token:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def probe_rejection(answerable: bool, question: str, span: str) -> str | None:
    """Return the frozen reason a generated probe is rejected, or None to keep it."""

    if not answerable:
        return "not_answerable"
    words = question.split()
    if not QUESTION_MIN_WORDS <= len(words) <= QUESTION_MAX_WORDS:
        return "question_length"
    if copied_run(question, span) >= MAX_COPIED_RUN:
        return "copied_span"
    return None


def _context(words: list[str], start: int, end: int) -> str:
    left = words[max(0, start - CONTEXT_WORDS) : start]
    right = words[end : end + CONTEXT_WORDS]
    return " ".join([*left, "[[", *words[start:end], "]]", *right])


def _openrouter(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:  # noqa: S310
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("OpenRouter returned a non-object response")
    return decoded


def generate_probes(
    rendered: Mapping[str, str],
    out: Path,
    *,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    budget_usd: float = PROBE_BUDGET_USD,
) -> dict[str, Any]:
    spans = sample_spans(rendered)
    spent = 0.0
    kept = 0
    reasons: dict[str, int] = {}
    served: set[tuple[str, str]] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index, span in enumerate(spans):
            if spent >= budget_usd:
                raise RuntimeError(f"probe budget exhausted after {index} calls: USD {spent:.4f}")
            words = rendered[span.session].split()
            payload = {
                "model": MODEL,
                "provider": PROVIDER_ROUTING,
                "temperature": 0,
                "max_tokens": 200,
                "response_format": RESPONSE_FORMAT,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "CONTEXT": _context(words, span.start, span.end),
                                "TARGET": span.text,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                "usage": {"include": True},
            }
            response = call(payload)
            usage = response.get("usage") or {}
            cost = usage.get("cost")
            if not isinstance(cost, (int, float)):
                cost = PRICE_IN * usage.get("prompt_tokens", 0) + PRICE_OUT * usage.get(
                    "completion_tokens", 0
                )
            spent += float(cost)
            served.add((str(response.get("model")), str(response.get("provider"))))
            try:
                decoded = json.loads(response["choices"][0]["message"]["content"])
                answerable = bool(decoded["answerable"])
                question = str(decoded["question"]).strip()
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                reasons["malformed"] = reasons.get("malformed", 0) + 1
                continue
            reason = probe_rejection(answerable, question, span.text)
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            kept += 1
            handle.write(
                json.dumps(
                    {
                        "probe_id": hashlib.sha256(
                            f"{span.session}\x00{span.start}\x00{span.end}".encode("utf-8")
                        ).hexdigest()[:16],
                        "session": span.session,
                        "split": split_for(span.session),
                        "span_start": span.start,
                        "span_end": span.end,
                        "span_text": span.text,
                        "question": question,
                        "gold_segments": list(span.gold),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "spans": len(spans),
        "kept": kept,
        "rejections": reasons,
        "usd": round(spent, 6),
        "served": sorted(f"{model}|{provider}" for model, provider in served),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


# --------------------------------------------------------------------------- vectors


class VectorCache:
    """A content addressed vector cache so reruns and extra arms cost no provider call."""

    def __init__(self, path: Path) -> None:
        import numpy as np

        self._np = np
        self.path = path
        self._vectors: dict[str, Any] = {}
        if path.exists():
            loaded = np.load(path)
            for key, vector in zip(loaded["keys"], loaded["vectors"], strict=True):
                self._vectors[str(key)] = vector

    @staticmethod
    def key(kind: str, text: str) -> str:
        return hashlib.sha256(f"{EMBEDDING_PROFILE}\x00{kind}\x00{text}".encode("utf-8")).hexdigest()

    def ensure(
        self, kind: str, texts: Sequence[str], embed: Callable[[list[str]], list[list[float]]]
    ) -> int:
        missing = sorted({text for text in texts if self.key(kind, text) not in self._vectors})
        for offset in range(0, len(missing), 64):
            batch = missing[offset : offset + 64]
            for text, vector in zip(batch, embed(batch), strict=True):
                self._vectors[self.key(kind, text)] = self._np.asarray(vector, dtype=self._np.float32)
        if missing:
            self.save()
        return len(missing)

    def matrix(self, kind: str, texts: Sequence[str]) -> Any:
        np = self._np
        rows = np.stack([self._vectors[self.key(kind, text)] for text in texts]).astype(np.float32)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / norms

    def save(self) -> None:
        np = self._np
        keys = sorted(self._vectors)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, keys=np.asarray(keys), vectors=np.stack([self._vectors[key] for key in keys]))
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- replay


@dataclass(frozen=True)
class Query:
    query_id: str
    kind: str
    text: str
    gold_ids: frozenset[str]
    gold_sessions: frozenset[str]


@dataclass
class ArmOutcome:
    ranked: list[str]
    rescued: str | None
    rescued_prior_rank: int | None
    selector_ms: float
    fallback: bool


def fuse(dense: Sequence[ScoredChunk], lexical: Sequence[ScoredChunk]) -> list[str]:
    """RRF of one dense and one lexical ranking, ordered exactly as the C8 path orders it."""

    by_id = {hit.chunk.id: hit for hit in [*dense, *lexical]}
    fused = _rrf(([hit.chunk.id for hit in dense], [hit.chunk.id for hit in lexical]))
    return sorted(fused, key=lambda item: (-fused[item], stable_window_key(by_id[item].chunk)))


def dense_top(matrix: Any, vector: Any, windows: Sequence[Window]) -> list[ScoredChunk]:
    import numpy as np

    scores = matrix @ vector
    order = np.argsort(-scores, kind="stable")[:CANDIDATE_K]
    return [ScoredChunk(windows[int(index)].chunk, float(scores[int(index)])) for index in order]


def build_artifact(
    arm: str,
    windows: Sequence[Window],
    rendered: Mapping[str, str],
    cache: VectorCache,
    embed: Callable[[list[str]], list[list[float]]],
    directory: Path,
) -> tuple[Any, dict[str, Any]]:
    from recall.atomic_rescue import load_atomic_rescue_artifact, write_atomic_rescue_artifact

    settings = ARMS[arm]
    id_of = {(window.session, window.segment): window.chunk.id for window in windows}
    texts: list[str] = []
    views: list[dict[str, object]] = []
    for session in sorted(rendered):
        for view in window_views(
            rendered[session], window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE, **settings
        ):
            texts.append(view.text)
            views.append(
                {
                    "chunk_id": id_of[(session, view.parent_segment)],
                    "source": session,
                    "parent_ordinal": view.parent_segment,
                    "view_ordinal": view.view_ordinal,
                }
            )
    embedded = cache.ensure("document", texts, embed)
    matrix = cache.matrix("document", texts)
    corpus = hashlib.sha256(
        "\n".join(window.chunk.id for window in windows).encode("utf-8")
    ).hexdigest()
    manifest = write_atomic_rescue_artifact(
        directory / arm,
        matrix=matrix,
        views=views,
        generation_id="cambench-reference",
        calibration_id="uncalibrated-reference",
        pipeline_fingerprint=f"atomizer-{arm}",
        corpus_fingerprint=corpus,
        embedding_profile=EMBEDDING_PROFILE,
        embedding_fingerprint="reference",
        ordinary_chunk_count=len(windows),
        source_commit=os.environ.get("RECALL_SOURCE_COMMIT", "unknown"),
    )
    artifact = load_atomic_rescue_artifact(manifest)
    parents = len({view["chunk_id"] for view in views})
    return artifact, {
        "views": len(views),
        "parents": parents,
        "parent_coverage": round(parents / len(windows), 4),
        "newly_embedded": embedded,
        "load_ms": round(artifact.load_ms, 3),
    }


def replay(
    query: Query,
    vector: Any,
    window_matrix: Any,
    windows: Sequence[Window],
    lexical: Sequence[ScoredChunk],
    artifact: Any | None,
) -> ArmOutcome:
    from recall.atomic_rescue import (
        AtomicRescueArtifactError,
        AtomicRescueSelectionError,
        insert_atomic_rescue_dense,
    )

    dense = dense_top(window_matrix, vector, windows)
    by_id = {window.chunk.id: window.chunk for window in windows}
    rescued: str | None = None
    prior: int | None = None
    fallback = False
    elapsed = 0.0
    if artifact is not None:
        started = time.perf_counter()
        try:
            transformed = insert_atomic_rescue_dense(
                artifact,
                [float(value) for value in vector],
                dense,
                lambda chunk_id, score: ScoredChunk(by_id[chunk_id], score)
                if chunk_id in by_id
                else None,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            rescued = transformed[5].chunk.id
            prior_ids = [hit.chunk.id for hit in dense]
            prior = prior_ids.index(rescued) + 1 if rescued in prior_ids else None
            dense = transformed
        except (AtomicRescueArtifactError, AtomicRescueSelectionError):
            fallback = True
    return ArmOutcome(fuse(dense, lexical), rescued, prior, elapsed, fallback)


def first_hit(ranked: Sequence[str], gold: Iterable[str]) -> int | None:
    wanted = set(gold)
    for rank, item in enumerate(ranked[:CANDIDATE_K], start=1):
        if item in wanted:
            return rank
    return None


def summarise(
    queries: Sequence[Query],
    outcomes: Mapping[str, Sequence[ArmOutcome]],
    session_of: Mapping[str, str],
) -> dict[str, Any]:
    """Arm metrics plus paired gains and losses against the off control."""

    def source_rank(query: Query, outcome: ArmOutcome) -> int | None:
        for rank, item in enumerate(outcome.ranked[:CANDIDATE_K], start=1):
            if session_of[item] in query.gold_sessions:
                return rank
        return None

    report: dict[str, Any] = {}
    control = outcomes["off"]
    base_exact = [first_hit(row.ranked, query.gold_ids) for query, row in zip(queries, control, strict=True)]
    for arm, rows in outcomes.items():
        exact = [first_hit(row.ranked, query.gold_ids) for query, row in zip(queries, rows, strict=True)]
        source = [source_rank(query, row) for query, row in zip(queries, rows, strict=True)]
        entry: dict[str, Any] = {"queries": len(queries)}
        for cutoff in CUTOFFS:
            entry[f"exact@{cutoff}"] = sum(1 for rank in exact if rank is not None and rank <= cutoff)
            entry[f"source@{cutoff}"] = sum(1 for rank in source if rank is not None and rank <= cutoff)
            entry[f"exact_gain@{cutoff}"] = sum(
                1
                for rank, base in zip(exact, base_exact, strict=True)
                if rank is not None and rank <= cutoff and (base is None or base > cutoff)
            )
            entry[f"exact_loss@{cutoff}"] = sum(
                1
                for rank, base in zip(exact, base_exact, strict=True)
                if base is not None and base <= cutoff and (rank is None or rank > cutoff)
            )
        entry["exact_mrr@10"] = round(
            sum(1.0 / rank for rank in exact if rank is not None and rank <= 10) / len(queries), 4
        )
        if arm != "off":
            active = [row for row in rows if not row.fallback]
            entry["attempted"] = len(rows)
            entry["active"] = len(active)
            entry["fallback"] = sum(1 for row in rows if row.fallback)
            entry["candidate_available"] = sum(1 for row in rows if row.rescued is not None)
            entry["rescued_is_exact_gold"] = sum(
                1
                for query, row in zip(queries, rows, strict=True)
                if row.rescued is not None and row.rescued in query.gold_ids
            )
            entry["rescued_from_outside_top100"] = sum(
                1 for row in rows if row.rescued is not None and row.rescued_prior_rank is None
            )
            timings = sorted(row.selector_ms for row in active)
            if timings:
                entry["selector_ms_p50"] = round(statistics.median(timings), 3)
                entry["selector_ms_p95"] = round(timings[max(0, int(0.95 * len(timings)) - 1)], 3)
        report[arm] = entry
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from recall.embeddings import embed_passages, embed_query, resolve_registered_embedder
    from scripts.aml_c7_qualification import load_frozen_corpus

    corpus = load_frozen_corpus(args.amb_root)
    rendered = dict(corpus.rendered)
    windows = build_windows(rendered)
    session_of = {window.chunk.id: window.session for window in windows}
    ids = {(window.session, window.segment): window.chunk.id for window in windows}

    queries: list[Query] = []
    if args.query_set in {"probes", "all"}:
        for line in args.probes.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if args.split != "all" and row["split"] != args.split:
                continue
            queries.append(
                Query(
                    row["probe_id"],
                    "probe",
                    row["question"],
                    frozenset(ids[(row["session"], segment)] for segment in row["gold_segments"]),
                    frozenset({row["session"]}),
                )
            )
    if args.query_set in {"tasks", "all"}:
        for task_id, prompt, gold in corpus.tasks:
            gold_ids = frozenset(window.chunk.id for window in windows if window.session in gold)
            queries.append(Query(task_id, "task", prompt, gold_ids, frozenset(gold)))

    embedder = resolve_registered_embedder(EMBEDDING_PROFILE, dict(os.environ))

    def documents(batch: list[str]) -> list[list[float]]:
        return embed_passages(embedder, batch)

    def questions(batch: list[str]) -> list[list[float]]:
        return [embed_query(embedder, text) for text in batch]

    cache = VectorCache(args.cache)
    spend = {
        "windows": cache.ensure("document", [w.chunk.text for w in windows], documents),
        "queries": cache.ensure(
            "query", [q.text for q in queries], questions),
    }
    window_matrix = cache.matrix("document", [window.chunk.text for window in windows])
    query_matrix = cache.matrix("query", [query.text for query in queries])
    chunks = [window.chunk for window in windows]
    lexical = [
        rank_bm25_chunks(chunks, query.text, k=CANDIDATE_K, stable_ties=True) for query in queries
    ]

    outcomes: dict[str, list[ArmOutcome]] = {}
    artifacts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="c8-atomizer-") as scratch:
        for arm in args.arms:
            artifact = None
            if arm != "off":
                artifact, artifacts[arm] = build_artifact(
                    arm, windows, rendered, cache, documents, Path(scratch)
                )
            outcomes[arm] = [
                replay(query, query_matrix[index], window_matrix, windows, lexical[index], artifact)
                for index, query in enumerate(queries)
            ]
    report: dict[str, Any] = {
        "protocol": "2026-09-22-c8-atomizer-reference",
        "split": args.split,
        "query_set": args.query_set,
        "windows": len(windows),
        "sessions": len(rendered),
        "newly_embedded": spend,
        "artifacts": artifacts,
    }
    for kind in sorted({query.kind for query in queries}):
        selected = [index for index, query in enumerate(queries) if query.kind == kind]
        report[kind] = summarise(
            [queries[index] for index in selected],
            {arm: [rows[index] for index in selected] for arm, rows in outcomes.items()},
            session_of,
        )
    if args.rows is not None:
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        with args.rows.open("w", encoding="utf-8", newline="\n") as handle:
            for index, query in enumerate(queries):
                handle.write(
                    json.dumps(
                        {
                            "query_id": query.query_id,
                            "kind": query.kind,
                            "arms": {
                                arm: {
                                    "exact_rank": first_hit(rows[index].ranked, query.gold_ids),
                                    "rescued": rows[index].rescued,
                                    "rescued_prior_rank": rows[index].rescued_prior_rank,
                                    "rescued_is_gold": rows[index].rescued in query.gold_ids,
                                    "fallback": rows[index].fallback,
                                }
                                for arm, rows in outcomes.items()
                            },
                        }
                    )
                    + "\n"
                )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probes = sub.add_parser("probes")
    probes.add_argument("--amb-root", type=Path, required=True)
    probes.add_argument("--out", type=Path, required=True)
    probes.add_argument("--budget-usd", type=float, default=PROBE_BUDGET_USD)
    run = sub.add_parser("evaluate")
    run.add_argument("--amb-root", type=Path, required=True)
    run.add_argument("--probes", type=Path, required=True)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--split", choices=("dev", "confirm", "all"), required=True)
    run.add_argument("--query-set", choices=("probes", "tasks", "all"), default="all")
    run.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    run.add_argument("--rows", type=Path)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "probes":
        from scripts.aml_c7_qualification import load_frozen_corpus

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required")
        corpus = load_frozen_corpus(args.amb_root)
        summary = generate_probes(
            dict(corpus.rendered),
            args.out,
            call=lambda payload: _openrouter(payload, api_key),
            budget_usd=args.budget_usd,
        )
    else:
        if "off" not in args.arms:
            raise SystemExit("the off control arm is mandatory")
        summary = evaluate(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
