"""Is an absolute cosine floor fragile across embedders, or was that an artefact of one sample?

::

    # No LLM spend at all — indexing, retrieval and arithmetic. Hours of CPU, ~$1 of cloud embedding.
    python -m benchmarks.beam.regime_sweep --out regime_sweep.json

The claim under test
--------------------
`DEFAULT_GAP_THRESHOLD = 0.50` is an ABSOLUTE cosine. §9l measured it doing nothing on bge-small
(0 of 54 answerable starved — it sits below that model's observed minimum) and discarding 7 % on
text-embedding-3-small. If that holds, the constant is not a tuned value but an untested one, and
a rate-based floor is the fix.

But §9l is one corpus and 54 questions in one arm, which is the exact profile of two measurements
that reversed at scale earlier in the same session: an n=2 entailment pilot whose policy was net
NEGATIVE at n=30, and a +0.029 k-sweep advantage that became +0.0007 at n=300. Varying the
embedder while holding the corpus fixed also cannot separate an embedder effect from a property of
those three conversations, because the top-1 distribution depends on both.

Design
------
Two corpora that differ in KIND, not just in content, crossed with three embedders:

  - **BEAM** — dense synthetic dialogue, real probing questions as queries.
  - **memory** — 787 curated markdown memos with frontmatter. This is the documental case RE-call
    actually targets, and nothing in this project has ever been measured on it.

The statistic is the VARIANCE of the starve rate across the six conditions: large for the constant
and small for the quantile, or the claim fails.

On the memory corpus the queries are each memo's `description:` line, which is derived from the
memo it retrieves — so those top-1 scores are biased HIGH and the absolute rates there are not
comparable to BEAM's. That is acceptable here and only here: the bias is identical across all three
embedders, and the quantity being compared is the difference BETWEEN embedders on identical
queries, where a shared bias cancels. It would not be acceptable for an accuracy claim.

**This test can fail.** If the three embedders turn out to occupy similar cosine regimes, embedder
independence is not a real problem, 0.50 is fine, and the whole line closes as unnecessary.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import iter_conversations
from benchmarks.systems import resolve_embedder
from recall.eval.locomo import DEFAULT_DSN
from recall.index import Indexer
from collections.abc import Sequence
from recall.store import DEFAULT_TENANT, PgVectorStore

#: (label, embedder spec). bge-small is the shipped default; bge-large is a second LOCAL model, so
#: a difference between them cannot be blamed on "local vs cloud"; the third is the cloud model
#: whose regime started this.
EMBEDDERS = (
    ("bge-small", "fastembed"),
    ("bge-large", "fastembed:BAAI/bge-large-en-v1.5"),
    ("text-embedding-3-small", "router:openai/text-embedding-3-small"),
)

ABSOLUTE_FLOORS = (0.50, 0.40)
QUANTILES = (0.05, 0.10)

FRONTMATTER_DESCRIPTION = re.compile(r"^description:\s*(.+)$", re.MULTILINE)

#: Single tenant for the memory corpus — it is one flat document set, not per-conversation.
MEMORY_TENANT = "regime-mem"


def _memory_queries(root: Path, limit: int | None = None) -> list[str]:
    """One query per memo, taken from its `description:` frontmatter line.

    Files without a description are skipped rather than substituted with their filename: a
    filename is not a query, and padding the set with unrealistic ones would move the very
    distribution being measured.
    """
    out: list[str] = []
    for f in sorted(root.glob("*.md")):
        try:
            head = f.read_text(encoding="utf-8", errors="replace")[:2000]
        except OSError:
            continue
        m = FRONTMATTER_DESCRIPTION.search(head)
        if m and len(m.group(1).strip()) > 15:
            out.append(m.group(1).strip())
        if limit and len(out) >= limit:
            break
    return out


def _beam_queries(shards: Path, conversations: int) -> list[tuple[str, str]]:
    """(tenant, question) for the first `conversations` BEAM conversations."""
    out: list[tuple[str, str]] = []
    for conv in iter_conversations(shards, "1M", list(range(conversations))):
        tenant = f"beam-1m-{conv.index}"
        out.extend((tenant, q.question) for q in conv.questions)
    return out


def _top1_scores(dsn: str, table: str, embedder: Any,
                 queries: Sequence[tuple[str | None, str]],
                 k: int) -> list[float]:
    """Top-1 score per query. A query returning nothing contributes 0.0, which is the honest
    encoding: no candidate at all is the extreme of a bad score, not a missing observation."""
    scores: list[float] = []
    for tenant, text in queries:
        vector = embedder.embed([text])[0]
        with PgVectorStore(
            dsn, dim=embedder.dim, tenant=tenant or DEFAULT_TENANT, table=table
        ) as store:
            hits = store.query_dense(vector, k=k)
        scores.append(max((h.score for h in hits), default=0.0))
    return scores


def _rates(scores: list[float]) -> dict[str, Any]:
    s = sorted(scores)
    n = len(s)
    q = {str(x): round(s[min(int(x * n), n - 1)], 4) for x in QUANTILES}
    return {
        "n": n,
        "min": round(s[0], 4),
        "median": round(statistics.median(s), 4),
        "max": round(s[-1], 4),
        "quantile_floors": q,
        "starve_rate_absolute": {
            str(f): round(sum(1 for v in s if v < f) / n, 4) for f in ABSOLUTE_FLOORS
        },
        "starve_rate_quantile": {
            str(x): round(sum(1 for v in s if v < float(q[str(x)])) / n, 4) for x in QUANTILES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--shards", type=Path, default=Path("/opt/recall-beam-data/shards_1M"))
    parser.add_argument("--memory", type=Path, default=Path("/opt/docs_rag_corpora/memory"))
    parser.add_argument("--beam-conversations", type=int, default=5)
    parser.add_argument("--k", type=int, default=45)
    parser.add_argument("--out", type=Path, default=Path("regime_sweep.json"))
    args = parser.parse_args()

    beam_q = _beam_queries(args.shards, args.beam_conversations)
    mem_q = [(MEMORY_TENANT, q) for q in _memory_queries(args.memory)]
    print(f"queries: BEAM {len(beam_q)}, memory {len(mem_q)}", flush=True)

    report: dict[str, Any] = {"conditions": {}, "beam_queries": len(beam_q),
                              "memory_queries": len(mem_q)}

    for label, spec in EMBEDDERS:
        embedder = resolve_embedder(spec)
        print(f"\n=== {label} (dim {embedder.dim}) ===", flush=True)

        # --- memory corpus: one tenant, index once ---
        table = f"regime_mem_{label.replace('-', '_').replace('.', '')}"[:60]
        with PgVectorStore(
            args.dsn, dim=embedder.dim, tenant=MEMORY_TENANT, table=table
        ) as store:
            store.ensure_schema()
            existing = sum(1 for _ in store.iter_chunks())
            if not existing:
                Indexer(store, embedder, batch_chunks=32).index_path(args.memory)
                print(f"  memory indexed: {sum(1 for _ in store.iter_chunks())} chunks", flush=True)
        scores = _top1_scores(args.dsn, table, embedder, mem_q, args.k)
        report["conditions"][f"memory/{label}"] = _rates(scores)
        print(f"  memory done: {report['conditions'][f'memory/{label}']}", flush=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

        # --- BEAM: one tenant per conversation ---
        btable = f"regime_beam_{label.replace('-', '_').replace('.', '')}"[:60]
        for conv in iter_conversations(args.shards, "1M", list(range(args.beam_conversations))):
            tenant = f"beam-1m-{conv.index}"
            with PgVectorStore(args.dsn, dim=embedder.dim, tenant=tenant, table=btable) as store:
                store.ensure_schema()
                if sum(1 for _ in store.iter_chunks()):
                    continue
                workspace = Path(f"/tmp/regime-{label}-{conv.index}")
                workspace.mkdir(parents=True, exist_ok=True)
                for i, turn in enumerate(conv.turns):
                    (workspace / f"turn_{i:06d}.md").write_text(
                        f"# {turn['role']}\n\n{turn['content']}\n", encoding="utf-8"
                    )
                Indexer(store, embedder, batch_chunks=32).index_path(workspace)
            print(f"  beam conv {conv.index} indexed", flush=True)
        scores = _top1_scores(args.dsn, btable, embedder, beam_q, args.k)
        report["conditions"][f"beam/{label}"] = _rates(scores)
        print(f"  beam done: {report['conditions'][f'beam/{label}']}", flush=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # The headline: how much does each rule's behaviour move across conditions?
    for family, key in (("absolute", "starve_rate_absolute"), ("quantile", "starve_rate_quantile")):
        params = ABSOLUTE_FLOORS if family == "absolute" else QUANTILES
        report[f"spread_{family}"] = {
            str(p): {
                "rates": {c: v[key][str(p)] for c, v in report["conditions"].items()},
                "spread": round(
                    max(v[key][str(p)] for v in report["conditions"].values())
                    - min(v[key][str(p)] for v in report["conditions"].values()), 4
                ),
            }
            for p in params
        }
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\n=== SPREAD (max-min del tasso di affamamento fra le condizioni) ===")
    print(json.dumps({k: v for k, v in report.items() if k.startswith("spread_")}, indent=1))


if __name__ == "__main__":
    main()
