"""LOCOMO — the standard long-term-memory benchmark, run against this library.

Why this file exists
--------------------
Every number this repo publishes was measured on its own corpora. Mem0 and Zep publish LOCOMO
and LongMemEval scores. Until this runner existed, the README's own honesty note applied without
qualification: *"nothing in this README is comparable to a published memory-benchmark result."*

What this measures — and what it deliberately does not
------------------------------------------------------
LOCOMO's headline metric is **LLM-as-a-Judge (J)**: a model reads the retrieved context, writes an
answer, and a judge grades it against the gold answer. Mem0 reports J=66.88, Zep J=65.99
(arXiv:2504.19413, Table 2).

**This runner does not produce a J score, and its numbers must never be placed in a column beside
one.** RE-call has no LLM in its path — it is the retrieval substrate underneath a system like
that, not a QA system. A J score would measure the generator this library does not ship.

What it measures instead is the part RE-call *is* responsible for, and it measures it exactly:

1. **Retrieval (categories 1-4).** LOCOMO annotates every answerable question with the dialog
   turns that contain the answer (``evidence: ["D1:3"]``), and every turn carries that id. So
   "did retrieval surface the right turn" is checkable by string equality — no judge, no
   generator, no scoring model, and therefore no judge variance. `hit@k` here is a *ceiling* on
   whatever J any downstream generator could reach: a turn that was never retrieved cannot be
   answered from.

2. **Abstention (category 5).** This is the reason the file is worth writing.

Category 5 is LOCOMO's adversarial split: 446 questions (22.5% of the dataset) that *look*
answerable and are not — typically an event attributed to the wrong speaker ("What did Caroline
realize after her charity race?" when the race was Melanie's). They carry `adversarial_answer`
instead of `answer`.

An independent audit of the benchmark (github.com/dial481/locomo-audit) reports that **no
published LOCOMO result evaluates them** — the original harness has a broken formatter affecting
444 of the 446, so vendors drop the category. That leaves the one axis this library was built for
unmeasured by the entire field, inside the field's own standard benchmark.

It is also the hardest possible shape for this library, and worth stating plainly before the
numbers arrive: an adversarial question is *lexically almost identical* to an answerable one, so
the retriever will return the wrong-speaker turn at a high cosine. That is the same failure
geometry as the superseded memory that outscores its own successor — the case in the README where
the stale hit wins on similarity (0.806 vs 0.784). Abstention here is not a threshold formality;
it is the thesis under load. A bad score is a real result and gets published as one.

Known defects in the benchmark, inherited by these numbers
----------------------------------------------------------
The same audit finds **99 of 1,540 questions (6.4%) carry incorrect gold answers**, which caps
any J score at ~93.6%. Retrieval scoring here is less exposed — it reads `evidence`, not
`answer` — but the two are annotated by the same pass, so treat the evidence labels as carrying
a comparable error rate. Per-category n ranges from 96 to 841 (8.8x), so the CIs are reported
per category and small-n categories are not separable; this runner never prints a per-category
ranking for that reason.

Usage
-----
::

    curl -sLO https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    python -m recall.eval.locomo --data locomo10.json                 # all 10 conversations
    python -m recall.eval.locomo --data locomo10.json --conversations 2 --k 10

Each of LOCOMO's 10 conversations is an independent world, so each is indexed into **its own
tenant** and searched in isolation. Pooling them would let a question about one conversation
retrieve a turn from another — inflating nothing and corrupting everything.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from recall.embeddings import Embedder
from recall.eval.labelled import _make_embedder
from recall.eval.metrics import wilson_ci
from recall.index import Indexer
from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever
from recall.store import PgVectorStore
from recall.trust import trusted_search

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")

#: LOCOMO's numeric category ids. The audit above notes these do NOT match the order the papers
#: present them in, so they are named here from the data itself (by inspecting members of each),
#: never from a paper's prose.
ANSWERABLE_CATEGORIES = (1, 2, 3, 4)
ADVERSARIAL_CATEGORY = 5

CATEGORY_NAMES = {
    1: "cat1",
    2: "cat2-temporal",
    3: "cat3",
    4: "cat4",
    5: "cat5-adversarial",
}


_SAFE_CORPUS_NAME = re.compile(r"\A[A-Za-z0-9:._-]+\Z")


def _check_corpus_name(value: str, kind: str) -> str:
    """Reject a dataset-derived path component that could escape the corpus directory.

    `dia_id` and `sample_id` come straight from the (third-party, HTTPS-downloaded) LOCOMO JSON,
    and both become filesystem paths — a per-turn filename and the per-conversation corpus dir. A
    `../`, an absolute path, or an embedded separator in a tampered dataset would let the converter
    write outside the workspace. Mirrors `recall.eval.longmemeval._check_session_id`, the guard the
    sibling converter already applies to the same class.
    """
    if not isinstance(value, str) or not _SAFE_CORPUS_NAME.match(value) or value in {".", ".."}:
        raise ValueError(f"unsafe LOCOMO {kind} (path traversal?): {value!r}")
    return value


def _dia_id_to_filename(dia_id: str) -> str:
    """``D1:3`` -> ``D1_3.md``.

    A colon is a legal path character on Linux and a reserved one on Windows (it opens an NTFS
    alternate data stream), and this repo is developed on Windows and run in CI on Linux. Encoding
    it away means the corpus that CI scores is byte-identical to the one a contributor scores.
    """
    _check_corpus_name(dia_id, "dia_id")
    return dia_id.replace(":", "_") + ".md"


def _filename_to_dia_id(filename: str) -> str:
    """Inverse of `_dia_id_to_filename`, for mapping a retrieved file back to ground truth."""
    return Path(filename).stem.replace("_", ":", 1)


def _turn_document(turn: dict[str, Any], session_date: str) -> str:
    """One dialog turn as a standalone markdown document.

    The speaker and the session date are written INTO the body rather than left as metadata,
    because they are frequently the answer: LOCOMO's temporal questions ask *when* something
    happened, and its adversarial questions turn on *who* did it. A retriever that cannot see
    the speaker cannot distinguish the adversarial pair from the real one, and the abstention
    measurement below would be scoring a handicap rather than the library.
    """
    speaker = turn.get("speaker", "unknown")
    text = turn.get("text", "")
    # Some turns carry an image with a generated caption; it is part of what was "said".
    caption = turn.get("blip_caption")
    body = f"{speaker}: {text}"
    if caption:
        body += f"\n\n[shared an image: {caption}]"
    return f"# {speaker} — {session_date}\n\n{body}\n"


def write_conversation_corpus(conversation: dict[str, Any], out_dir: Path) -> int:
    """Materialise one LOCOMO conversation as a directory of per-turn markdown files.

    Files on disk rather than an in-memory shortcut into the store, deliberately: this routes the
    benchmark through `Indexer.index_path` — the same chunking, hashing and frontmatter path that
    real corpora take. A bespoke in-memory loader would be measuring a code path no user runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(
        # Only `session_<int>` keys: this both selects the dialog sessions (excluding
        # `session_N_date_time`) and guards `int(...)` below from a non-numeric `session_*` key,
        # which would otherwise raise ValueError mid-sort instead of being skipped.
        (k for k in conversation if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    written = 0
    for key in sessions:
        turns = conversation[key]
        if not isinstance(turns, list):
            continue
        date = conversation.get(f"{key}_date_time", "unknown date")
        for turn in turns:
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            (out_dir / _dia_id_to_filename(dia_id)).write_text(
                _turn_document(turn, date), encoding="utf-8"
            )
            written += 1
    return written


def _retrieved_dia_ids(hits: list) -> list[str]:
    """Distinct dialog ids behind a result's hits, best rank first."""
    out: list[str] = []
    for h in hits:
        chunk = getattr(h, "chunk", h)
        f = chunk.metadata.get("file")
        if not f:
            continue
        dia = _filename_to_dia_id(f)
        if dia not in out:
            out.append(dia)
    return out


def _hit_by_depth(hits: list, evidence: list[str], depths: Sequence[int]) -> dict[int, bool]:
    """`hit@d` for every depth in `depths`, scored from ONE ranked hit list.

    Truncation is applied to the CHUNK hits and the dialog ids are derived afterwards, which is
    the whole correctness question here. Several chunks can map to the same turn, so the id list
    is shorter than the hit list and slicing *it* to d would reach further down the ranking than
    depth d ever returned — inflating every k below the maximum, and inflating it most where
    chunking is densest. Same trap as `_retrieved_dia_ids` returning distinct ids: the id list is
    a projection of the ranking, not the ranking.
    """
    return {d: any(e in _retrieved_dia_ids(hits[:d]) for e in evidence) for d in depths}


def _rate(flags: list[bool]) -> dict[str, Any]:
    lo, hi = wilson_ci(flags)
    return {
        "n": len(flags),
        "rate": round(sum(flags) / len(flags), 4) if flags else float("nan"),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _depths(ks: Sequence[int] | None, k: int) -> list[int]:
    """Sorted, de-duplicated retrieval depths to score — always including the headline ``k``.

    One source of truth for both ``run`` and ``run_conversation``: ``k`` is folded in
    unconditionally, so every caller — including a direct ``run_conversation`` that passes a ``ks``
    omitting ``k`` — scores the headline depth. Without it, ``hit_by_k[k]`` below could miss.
    """
    return sorted({int(d) for d in ((*ks, k) if ks else (k,))})


def run_conversation(
    conversation: dict[str, Any],
    qa: list[dict[str, Any]],
    *,
    store: PgVectorStore,
    embedder: Embedder,
    k: int,
    corpus_dir: Path,
    ks: Sequence[int] | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> dict[str, Any]:
    """Index one conversation and score every question against it.

    `ks` scores several retrieval depths from ONE retrieval instead of one depth per run. That is
    exact rather than an approximation: `candidate_k` fixes the fused pool independently of `k`,
    so `search(k=n)` returns the first `n` of the same ranking `search(k=N>n)` returns. The k=5
    row of a curve therefore has to reproduce a k=5 run exactly, and `run()` checks that it does.

    Truncation is applied to the CHUNK hits, not to the dialog ids derived from them: several
    chunks can map to one turn, so slicing the id list would score a deeper retrieval than the
    one asked for and inflate every k below the maximum.
    """
    n_turns = write_conversation_corpus(conversation, corpus_dir)
    store.ensure_schema()
    Indexer(store, embedder).index_path(corpus_dir)

    depths = _depths(ks, k)
    max_k = max(depths)
    retriever = HybridRetriever(store, embedder, candidate_k=candidate_k)
    hits_by_cat: dict[int, list[bool]] = {c: [] for c in ANSWERABLE_CATEGORIES}
    abstained: list[bool] = []
    per_question: list[dict[str, Any]] = []

    for q in qa:
        cat = q.get("category")
        question = q.get("question")
        if not question or cat is None:
            continue

        if cat == ADVERSARIAL_CATEGORY:
            # The abstention arm. `trusted_search` is the agent-facing entry point, and its
            # `abstained` flag is the whole answer: the correct behaviour on an unanswerable
            # question is to refuse, regardless of what came back underneath. Pass candidate_k so
            # this arm shares the same fused pool as the depth-curve arm above and the report's
            # top-level `candidate_k` — not silently the retriever's default.
            result = trusted_search(store, embedder, question, k=k, candidate_k=candidate_k)
            abstained.append(result.abstained)
            per_question.append(
                {
                    "question": question,
                    "category": cat,
                    "abstained": result.abstained,
                    "reason": result.reason,
                    "top_cosine": round(result.hits[0].cosine, 4) if result.hits else None,
                }
            )
            continue

        if cat not in hits_by_cat:
            continue
        evidence = [e for e in (q.get("evidence") or []) if isinstance(e, str)]
        if not evidence:
            # No gold evidence -> nothing to score against. Skipped rather than counted as a
            # miss: scoring it either way would be reporting a label gap as a system property.
            continue
        retrieval = retriever.search(question, k=max_k)
        hit_by_k = _hit_by_depth(retrieval.hits, evidence, depths)
        retrieved = _retrieved_dia_ids(retrieval.hits)
        hit = hit_by_k[k]  # k ∈ depths by construction (_depths folds it in)
        hits_by_cat[cat].append(hit)
        per_question.append(
            {
                "question": question,
                "category": cat,
                "evidence": evidence,
                "retrieved": retrieved,
                "hit": hit,
                "hit_by_k": hit_by_k,
            }
        )

    return {
        "sample_id": conversation.get("sample_id"),
        "turns": n_turns,
        "retrieval": {
            CATEGORY_NAMES[c]: _rate(flags) for c, flags in hits_by_cat.items() if flags
        },
        "abstention": _rate(abstained),
        "questions": per_question,
    }


def run(
    data_path: Path,
    *,
    dsn: str,
    embedder_name: str,
    k: int,
    limit: int | None,
    keep_corpus: Path | None,
    table: str,
    ks: Sequence[int] | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> dict[str, Any]:
    conversations = json.loads(data_path.read_text(encoding="utf-8"))
    if limit is not None:
        conversations = conversations[:limit]

    depths = _depths(ks, k)
    embedder = _make_embedder(embedder_name)
    started = time.time()
    per_conversation: list[dict[str, Any]] = []

    workspace = Path(keep_corpus) if keep_corpus else Path(tempfile.mkdtemp(prefix="locomo-"))
    try:
        for i, conv in enumerate(conversations):
            sample_id = conv.get("sample_id") or f"conv{i}"
            _check_corpus_name(str(sample_id), "sample_id")
            # One tenant per conversation. LOCOMO's conversations are unrelated worlds; a shared
            # tenant would let a question retrieve another conversation's turn, and the
            # evidence-id check would not catch it (ids are only unique WITHIN a conversation,
            # so a cross-conversation "D1:3" would score as a hit).
            tenant = f"locomo-{sample_id}"
            corpus_dir = workspace / str(sample_id)
            with PgVectorStore(dsn, dim=embedder.dim, tenant=tenant, table=table) as store:
                res = run_conversation(
                    conv["conversation"],
                    conv.get("qa") or [],
                    store=store,
                    embedder=embedder,
                    k=k,
                    corpus_dir=corpus_dir,
                    ks=depths,
                    candidate_k=candidate_k,
                )
            per_conversation.append(res)
            print(
                f"  [{i + 1}/{len(conversations)}] {sample_id}: "
                f"{res['turns']} turns, "
                f"abstention n={res['abstention']['n']}",
                flush=True,
            )
    finally:
        if keep_corpus is None:
            shutil.rmtree(workspace, ignore_errors=True)

    # Pool across conversations. Per-conversation rates are not averaged: the conversations have
    # different question counts, so a mean of rates would silently weight a 150-question
    # conversation the same as a 250-question one.
    pooled_retrieval: dict[str, list[bool]] = {}
    for res in per_conversation:
        for q in res["questions"]:
            if q["category"] in ANSWERABLE_CATEGORIES and "hit" in q:
                pooled_retrieval.setdefault(CATEGORY_NAMES[q["category"]], []).append(q["hit"])
    pooled_abstain = [
        q["abstained"] for res in per_conversation for q in res["questions"] if "abstained" in q
    ]
    all_hits = [h for flags in pooled_retrieval.values() for h in flags]

    # The depth curve, pooled the same way. Per-category as well as overall: §9a's weakest
    # category (cat3, 0.391 at k=5) is the one a reader will ask about, and "does depth rescue
    # it" is not answerable from an overall rate.
    scored = [
        q
        for res in per_conversation
        for q in res["questions"]
        if q.get("hit_by_k") and q["category"] in ANSWERABLE_CATEGORIES
    ]
    curve: dict[str, Any] = {}
    for d in depths:
        by_cat: dict[str, list[bool]] = {}
        for q in scored:
            by_cat.setdefault(CATEGORY_NAMES[q["category"]], []).append(q["hit_by_k"][d])
        flat = [h for f in by_cat.values() for h in f]
        curve[str(d)] = {
            "overall": _rate(flat),
            "by_category": {c: _rate(f) for c, f in sorted(by_cat.items())},
        }

    # Coherence check: the pooled headline hit@k (from q["hit"] = hit_by_k[k]) must equal the
    # curve's own row at k (from q["hit_by_k"][k]). Both are scored from the SAME single retrieval,
    # so this does NOT independently re-verify that top-k is a prefix of the fused pool — that is a
    # property of the retriever, and one retrieval cannot check it. What it guards is the
    # depth-labeling above: that q["hit"] is recorded at k and not at some other depth. If that
    # drifts the run fails instead of silently reporting a different metric under the same name.
    if str(k) in curve and all_hits:
        primary = curve[str(k)]["overall"]["rate"]
        assert primary == _rate(all_hits)["rate"], (
            f"depth curve at k={k} reads {primary} but the pooled headline hit@k reads "
            f"{_rate(all_hits)['rate']}: q['hit'] is not scored at k"
        )

    return {
        "benchmark": "LOCOMO",
        "metric": f"evidence-turn hit@{k} (retrieval only, no LLM judge)",
        "embedder": embedder_name,
        "k": k,
        "candidate_k": candidate_k,
        "depth_curve": curve,
        "conversations": len(conversations),
        "elapsed_s": round(time.time() - started, 1),
        "retrieval_by_category": {c: _rate(f) for c, f in sorted(pooled_retrieval.items())},
        "retrieval_overall": _rate(all_hits),
        "abstention_adversarial": _rate(pooled_abstain),
        "per_conversation": [
            {kk: vv for kk, vv in res.items() if kk != "questions"} for res in per_conversation
        ],
    }


def _print_report(report: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(f"LOCOMO · {report['conversations']} conversations · embedder={report['embedder']}")
    print(f"metric: {report['metric']}")
    print("=" * 72)
    print()
    print("RETRIEVAL — can the evidence turn be found at all?")
    for cat, r in report["retrieval_by_category"].items():
        lo, hi = r["ci95"]
        print(f"  {cat:<20} hit@{report['k']} {r['rate']:.3f}  [{lo:.2f}, {hi:.2f}]  n={r['n']}")
    o = report["retrieval_overall"]
    print(f"  {'OVERALL':<20} hit@{report['k']} {o['rate']:.3f} "
          f" [{o['ci95'][0]:.2f}, {o['ci95'][1]:.2f}]  n={o['n']}")
    print()
    curve = report.get("depth_curve") or {}
    if len(curve) > 1:
        print("DEPTH — hit@k against retrieval depth, from ONE retrieval per question.")
        print(f"  (candidate pool {report.get('candidate_k')} per leg; the curve cannot exceed it)")
        # Every depth scores the same question set, so all rows share the same category keys.
        cats = list(next(iter(curve.values()))["by_category"].keys())
        print(f"  {'k':<5}{'OVERALL':<12}" + "".join(f"{c:<16}" for c in cats))
        for kk in sorted(curve, key=int):
            row = curve[kk]
            line = f"  {kk:<5}{row['overall']['rate']:<12.3f}"
            line += "".join(f"{row['by_category'][c]['rate']:<16.3f}" for c in cats)
            print(line)
        print()
    print("ABSTENTION — category 5, adversarial. No published LOCOMO result reports this.")
    a = report["abstention_adversarial"]
    if a["n"]:
        print(f"  {'abstain rate':<20} {a['rate']:.3f} "
              f" [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]  n={a['n']}")
        print("  (1.00 = refused every unanswerable question; 0.00 = answered them all)")
    else:
        print("  no adversarial questions in this slice")
    print()
    print("NOT a J score. Not comparable to Mem0's 66.88 or Zep's 65.99 — those measure a")
    print("generator this library does not ship. See the module docstring.")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m recall.eval.locomo",
        description="Run the LOCOMO benchmark against RE-call (retrieval + abstention arms).",
    )
    p.add_argument("--data", required=True, type=Path, help="path to locomo10.json")
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--embedder", default="fastembed", help="fastembed | voyage | hashing | st:PATH")
    p.add_argument("--k", type=int, default=5, help="retrieval depth for hit@k (default 5)")
    p.add_argument(
        "--k-curve",
        default=None,
        help="comma-separated depths to score in ONE pass, e.g. 1,3,5,10,20. Exact, not "
             "approximate: the candidate pool does not depend on k, so top-k is a prefix of "
             "top-max(k). --k stays the headline depth and is always included",
    )
    p.add_argument(
        "--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
        help=f"candidates per retrieval leg before fusion (default {DEFAULT_CANDIDATE_K}). Raise "
             "it to score depths past the default pool — but that changes the fusion, so the "
             "result is a different configuration, not a deeper look at the published one",
    )
    p.add_argument(
        "--conversations", type=int, default=None, help="score only the first N conversations"
    )
    p.add_argument("--out", type=Path, default=None, help="write the full JSON report here")
    p.add_argument(
        "--table",
        default="locomo_chunks",
        # NOT the default `chunks`. A benchmark that shares a table with a dev corpus inherits
        # whatever dimension that corpus was last built at, and the run dies on a dimension
        # mismatch that has nothing to do with the benchmark. Its own table also means a
        # benchmark run can never prune or overwrite real indexed data.
        help="table to index into (default: locomo_chunks, kept apart from any dev corpus)",
    )
    p.add_argument(
        "--keep-corpus",
        type=Path,
        default=None,
        help="write the generated per-turn corpus here and keep it (default: temp dir, deleted)",
    )
    args = p.parse_args(argv)

    if not args.data.exists():
        p.error(
            f"{args.data} not found. Fetch it with:\n"
            "  curl -sLO https://raw.githubusercontent.com/snap-research/locomo/main/data/"
            "locomo10.json"
        )

    ks = None
    if args.k_curve:
        try:
            ks = [int(part) for part in args.k_curve.split(",") if part.strip()]
        except ValueError:
            p.error(f"--k-curve must be comma-separated integers, got {args.k_curve!r}")
        if not ks or any(d < 1 for d in ks):
            p.error("--k-curve depths must all be >= 1")

    if args.k < 1:
        p.error("--k must be >= 1")
    if args.candidate_k < 1:
        p.error("--candidate-k must be >= 1")

    report = run(
        args.data,
        dsn=args.dsn,
        embedder_name=args.embedder,
        k=args.k,
        limit=args.conversations,
        keep_corpus=args.keep_corpus,
        table=args.table,
        ks=ks,
        candidate_k=args.candidate_k,
    )
    _print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"full report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
