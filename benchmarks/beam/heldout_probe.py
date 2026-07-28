"""Does anything separate answerable from unanswerable when the corpus is NOT adversarial?

::

    # $0 of LLM spend. Indexing + retrieval + a local cross-encoder.
    python -m benchmarks.beam.heldout_probe --memory /opt/docs_rag_corpora/memory \
        --embedder router:openai/text-embedding-3-small --out heldout_probe.json

Why this corpus and this construction
-------------------------------------
Five independent signals have now failed to separate BEAM's unanswerable questions from its
answerable ones, and the last one failed in the most informative way: lexical coverage came out
**higher** for unanswerable questions (0.741 vs 0.717), the same inversion cosine shows (0.676 vs
0.641). Two signals sharing no mathematics and no model, inverted identically, is not a property of
embeddings — it is a property of how BEAM's unanswerable questions are BUILT. They ask for details
never stated about topics discussed at length, so the more thoroughly a topic was covered, the more
overlap a question about it finds, and the higher every similarity measure scores exactly where
there is no answer.

That construction is deliberate and legitimate for a hallucination test. It is also not what a
deployment meets. The ordinary unanswerable question is about something the corpus simply does not
contain.

So this builds that case mechanically, with no labelling: hold out N memos, index the rest, and use
each memo's own `description:` line as its query. Descriptions of INDEXED memos are answerable;
descriptions of HELD-OUT memos are unanswerable, because the document they describe is genuinely
absent. No adversarial design, no human judgement.

Predictions, written before running
-----------------------------------
**P1 — cosine WILL separate here, in the right direction**, unlike on BEAM. A held-out memo is
absent, so its best match is some other memo. Predicted gap: unanswerable lower by **0.05-0.15**.

**P2 — but weakly**, because this corpus is topically dense: memo families (`feedback_*`,
`incident_*`, `project_*`) mean a held-out memo about VPS deploys still finds several near
neighbours. A large gap would surprise me.

**P3 — entailment separates better than cosine**, because "does this text answer the query" is a
sharper question than "is this text similar to the query", and it is the only signal that moved the
abstention number on BEAM. If P3 fails — if entailment separates no better than cosine on a corpus
built to be fair to both — then the entailment guard's value is not established anywhere and the
whole abstention lane is empty.

P3 is the one worth being wrong about.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.beam.systems import EMBED_BATCH
from benchmarks.systems import resolve_embedder
from recall.eval.locomo import DEFAULT_DSN
from recall.index import Indexer
from recall.store import PgVectorStore

FRONTMATTER_DESCRIPTION = re.compile(r"^description:\s*(.+)$", re.MULTILINE)
SEED = 1234
TENANT = "heldout"


def _memos(root: Path) -> list[tuple[Path, str]]:
    """(path, description) for every memo that carries a usable description line."""
    out: list[tuple[Path, str]] = []
    for f in sorted(root.glob("*.md")):
        try:
            head = f.read_text(encoding="utf-8", errors="replace")[:2000]
        except OSError:
            continue
        m = FRONTMATTER_DESCRIPTION.search(head)
        if m and len(m.group(1).strip()) > 15:
            out.append((f, m.group(1).strip()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # REQUIRED, not defaulted. The old default pointed at the operator's private memo
    # corpus, so a bare `python -m benchmarks.beam.heldout_probe` indexed private prose —
    # and with the old cloud `--embedder` default, shipped every chunk of it to a
    # third-party embedding API without the caller having asked for the cloud path.
    # SECURITY.md names that combination as in-scope and wanted-reported.
    parser.add_argument("--memory", type=Path, required=True,
                        help="corpus of memos to index (named explicitly: this probe reads "
                             "whatever you point it at, verbatim, into the embedder)")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    # Local by default. Reaching a cloud embedder is an egress boundary and must be an
    # explicit choice by the caller, never the path taken when no choice was made.
    parser.add_argument("--embedder", default="fastembed",
                        help="`fastembed` is local; `router:...`/`voyage:...` send every "
                             "chunk to a third-party API")
    parser.add_argument("--table", default="heldout_chunks")
    parser.add_argument("--holdout", type=int, default=120)
    parser.add_argument("--k", type=int, default=10)
    # BooleanOptionalAction, not `store_true` with `default=True` — the latter could only
    # ever be True, so the cosine-only arm of this probe's own preregistered P3
    # comparison was unrunnable and the `judge is None` branches below were dead code.
    parser.add_argument("--entailment", action=argparse.BooleanOptionalAction, default=True,
                        help="score entailment alongside cosine (--no-entailment to skip "
                             "loading the cross-encoder)")
    parser.add_argument("--out", type=Path, default=Path("heldout_probe.json"))
    args = parser.parse_args()

    memos = _memos(args.memory)
    rng = random.Random(SEED)
    held = set(rng.sample(range(len(memos)), min(args.holdout, len(memos) // 3)))
    indexed = [m for i, m in enumerate(memos) if i not in held]
    holdout = [m for i, m in enumerate(memos) if i in held]
    print(f"memos: {len(memos)} total, {len(indexed)} indexed, {len(holdout)} held out", flush=True)

    embedder = resolve_embedder(args.embedder)

    # Index ONLY the retained memos, into a workspace containing exactly those files — copying is
    # what makes the holdout real. Pointing the indexer at the full directory and filtering later
    # would leave the held-out text in the store, and every "unanswerable" query would find its own
    # document sitting there.
    # A private temp dir per run, cleaned up afterwards — NOT a fixed `/tmp/heldout-corpus`.
    # The old path was predictable and world-writable, and the code unlinked files there BEFORE
    # validating anything, so a pre-created symlink turned this into a delete primitive. It also
    # meant two concurrent arms silently destroyed each other's corpus — the same shape as the
    # incident `run_locomo_arms.sh` added its lock for. `BeamRecallSystem.ingest` already does
    # this correctly; the pattern just was not carried over.
    workspace = Path(tempfile.mkdtemp(prefix="heldout-"))
    try:
        for path, _ in indexed:
            (workspace / path.name).write_text(
                path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
            )

        with PgVectorStore(args.dsn, dim=embedder.dim, tenant=TENANT, table=args.table) as store:
            store.ensure_schema()
            existing = [c.source for c in store.iter_chunks()]
            if existing:
                store.delete_sources(sorted(set(existing)))
            Indexer(store, embedder, batch_chunks=EMBED_BATCH).index_path(workspace)
            n_chunks = store.count()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    print(f"indexed {n_chunks} chunks from {len(indexed)} memos", flush=True)

    judge = None
    if args.entailment:
        from recall.entailment import QnliEntailmentJudge

        judge = QnliEntailmentJudge()

    def _measure(pairs: list[tuple[Path, str]], answerable: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path, description in pairs:
            vector = embedder.embed([description])[0]
            with PgVectorStore(
                args.dsn, dim=embedder.dim, tenant=TENANT, table=args.table
            ) as store:
                hits = store.query_dense(vector, k=args.k)
            top1 = max((h.score for h in hits), default=0.0)
            entailed = None
            if judge is not None and hits:
                # Fraction of the top-k that the judge says answers the query. A COUNT, not a
                # max: §9h showed the shipped any()-style rule is the permissive extreme.
                decisions = judge.judge(description, [h.chunk.text for h in hits])
                entailed = sum(decisions) / len(decisions)
            rows.append(
                {
                    "memo": path.name,
                    "answerable": answerable,
                    "top1": round(top1, 4),
                    "entailed_fraction": round(entailed, 4) if entailed is not None else None,
                }
            )
        return rows

    rows = _measure(indexed, True) + _measure(holdout, False)

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]

    def _summary(sel: list[dict[str, Any]], key: str) -> dict[str, Any]:
        vals = [r[key] for r in sel if r[key] is not None]
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 4) if vals else None,
            "median": round(statistics.median(vals), 4) if vals else None,
        }

    report: dict[str, Any] = {
        "embedder": embedder.name,
        "indexed_memos": len(indexed),
        "held_out_memos": len(holdout),
        "chunks": n_chunks,
        "cosine": {
            "answerable": _summary(ans, "top1"),
            "unanswerable": _summary(una, "top1"),
        },
        "entailment": {
            "answerable": _summary(ans, "entailed_fraction"),
            "unanswerable": _summary(una, "entailed_fraction"),
        },
        "per_question": rows,
    }
    for name in ("cosine", "entailment"):
        a = report[name]["answerable"]["mean"]
        u = report[name]["unanswerable"]["mean"]
        # POSITIVE means answerable scores higher, i.e. the signal points the right way. On BEAM
        # both cosine and lexical coverage came out NEGATIVE.
        report[name]["separation"] = round(a - u, 4) if a is not None and u is not None else None

    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
