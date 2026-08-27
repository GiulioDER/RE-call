"""Screen the three surviving retrieval directions on the 14 recorded miss sessions.

    python -u scripts/agent_ab_screen_directions.py \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control \
        --archive ~/.claude/archive/agent-ab-skill-001

Preregistered in `docs/preregistrations/2026-08-27-three-direction-screen.md`. Read-only: it
builds nothing, generates nothing, writes only its own artifact.

It drives the SHIPPED `HybridRetriever` rather than reimplementing retrieval, in three leg
configurations, because the point is what this system can reach and not what a hand-rolled query
can:

- `dense`   — `use_sparse=False`, cosine only
- `lexical` — `use_dense=False`, `ts_rank` only
- `fused`   — both, unweighted RRF: exactly what production serves today

against two query sources:

- the sessions' recorded GOAL queries (screens A and B)
- each session's OWN recorded `Write`/`Edit` payload or `Bash` command, one query per payload
  (screen C, the "search with the operation you are about to perform" hypothesis)

Endpoint everywhere: the governing memo in TOP-5, the depth a session receives. Coverage is also
reported at k in {5, 20, 40, 100, 200}; 40 is called out because it is production's real fused
pool (each leg takes candidate_k=20, so dense ∪ lexical is at most 40).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TOP_K = 5
K_GRID = (5, 20, 40, 100, 200)
CANDIDATE_K = 200
MISSED_FAMILIES = ("ts-lf-rewrite", "ts-worktree-import", "ts-sample-covers-tail")
#: fastembed's bge-small; the corpus was built with it.
EMBEDDER_ID = "fastembed"
DIM = 384


def sessions(archive: Path) -> list[dict]:
    """On-arm `memory_only` sessions of the three missed families, with goal queries and drafts."""

    rows: list[dict] = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = str(record.get("task_id", ""))
        base = task_id.split("#")[0]
        metadata = record.get("metadata") or {}
        if base not in MISSED_FAMILIES or record.get("variant") != "recall_on":
            continue
        if metadata.get("locus") != "memory_only":
            continue
        memo = str(metadata.get("governing_memo") or "")
        queries, drafts = [], []
        for call in record.get("tool_calls") or []:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            if "recall_search" in name and args.get("query"):
                queries.append(str(args["query"]))
            elif name in ("Write", "Edit", "NotebookEdit"):
                payload = str(args.get("content") or args.get("new_string") or "")
                if payload.strip():
                    drafts.append(payload)
            elif name == "Bash" and args.get("command"):
                drafts.append(str(args["command"]))
        retrieved = " ".join(str(c) for c in (record.get("retrieved_contexts") or []))
        rows.append(
            {
                "task_id": task_id,
                "base": base,
                "memo": memo,
                "queries": queries,
                "drafts": drafts,
                "hit_in_run": bool(memo and memo in retrieved),
            }
        )
    return rows


def rank_of(retriever, query: str, memo: str) -> int | None:
    """1-based rank of the memo's best chunk in this retriever's result, or None if absent.

    ⛔ **A retrieval error is NOT a miss and must never be caught here.** The first version of this
    function swallowed exceptions and returned None, and every one of the 84 retrievals failed with
    `UndefinedColumn` while the screen printed a flawless 0/14 across all six columns — a perfect,
    entirely fabricated null, in a probe written to decide which direction to pursue. It is the
    exact failure `[[a-null-is-the-cheapest-result-to-fabricate]]` was written about, one hour
    after writing it. Let the exception kill the run.
    """

    wanted = f"{memo}.md"
    result = retriever.search(query, k=CANDIDATE_K)
    for index, hit in enumerate(result.hits, start=1):
        # `Chunk.source`, not `.source_uri`. Getting this wrong was the THIRD defect in this one
        # script that produced a flawless 0/14: a `getattr(..., "source_uri", "")` default meant
        # every comparison was against an empty string and nothing ever matched. Read the field,
        # do not default it.
        if Path(str(hit.chunk.source)).name == wanted:
            return index
    return None


def best(ranks: list[int | None]) -> int | None:
    found = [r for r in ranks if r is not None]
    return min(found) if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--generation", default=None, help="active generation id; asked if absent")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import psycopg

    # `resolve_embedder`, not `resolve_registered_embedder`: the corpus was built by the CLI with
    # RECALL_EMBEDDER=fastembed, and that is the resolver the CLI uses. Asking the registry for a
    # profile id would silently pick a DIFFERENT model, and every tenant here is 384/1024-wide, so
    # a wrong choice returns a confidently ranked list rather than an error.
    from recall.embeddings import resolve_embedder
    from recall.generation_store import GenerationStore
    from recall.retriever import HybridRetriever

    generation = args.generation
    if not generation:
        with psycopg.connect(args.dsn, connect_timeout=20) as conn:
            row = conn.execute(
                "SELECT generation_id FROM recall_generations WHERE state = 'active' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            raise SystemExit("no active generation in that database")
        generation = row[0]
    print(f"corpus: {args.dsn.rsplit('/', 1)[-1]}  generation: {generation}")

    rows = sessions(Path(args.archive).expanduser())
    missed = [r for r in rows if not r["hit_in_run"]]
    print(f"{len(rows)} sessions in the missed families; {len(missed)} scored as misses")
    for base in MISSED_FAMILIES:
        print(f"  {base:<26} {sum(1 for r in missed if r['base'] == base)}")
    if len(missed) != 14:
        raise SystemExit(
            f"expected the registered 14 miss sessions, recovered {len(missed)}; the screen's "
            "population is not the one the record fixes, so it is not run"
        )

    embedder = resolve_embedder(EMBEDDER_ID)
    # `GenerationStore`, constructed exactly as `recall_mcp/stores.py` constructs it for a served
    # tenant. A plain `PgVectorStore` pointed at `recall_chunks_v1` reads the LEGACY column set
    # (`id`, `source`, ...) and every query raises `UndefinedColumn`, which the first version of
    # this screen then reported as a clean 0/14.
    store = GenerationStore(args.dsn, dim=DIM, tenant="default")
    store.check_schema()
    legs = {
        "dense": HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_sparse=False),
        "lexical": HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_dense=False),
        "fused": HybridRetriever(store, embedder, candidate_k=CANDIDATE_K),
    }

    # ⛔ POSITIVE CONTROL, and the reason it exists: the first run of this screen reported a
    # flawless 0/14 in all six columns while every single retrieval was failing with
    # `UndefinedColumn`. An all-zero screen and a broken screen are the same output. So: query each
    # leg with a memo's own distinctive text and require it back at rank 1. If a retriever cannot
    # find a document by quoting it, no number below means anything.
    control_memo = "python-write-text-crlf-churn"
    control_query = "Path.write_text on Windows injects CRLF against a tree configured eol=lf"
    for leg_name, retriever in legs.items():
        rank = rank_of(retriever, control_query, control_memo)
        print(f"  positive control [{leg_name}]: {control_memo} at rank {rank}")
        if rank is None or rank > TOP_K:
            raise SystemExit(
                f"POSITIVE CONTROL FAILED on the {leg_name} leg: quoting a memo's own content did "
                f"not return it in the top {TOP_K} (rank {rank}). The instrument is broken, and "
                "every screen below it would read as a clean null. Not run."
            )

    results: list[dict] = []
    for row in missed:
        entry = {k: row[k] for k in ("task_id", "base", "memo")}
        entry["n_queries"] = len(row["queries"])
        entry["n_drafts"] = len(row["drafts"])
        for leg_name, retriever in legs.items():
            entry[f"goal_{leg_name}"] = best(
                [rank_of(retriever, q, row["memo"]) for q in row["queries"]]
            )
            entry[f"draft_{leg_name}"] = best(
                [rank_of(retriever, d, row["memo"]) for d in row["drafts"]]
            )
        results.append(entry)
        print(
            f"  {row['task_id']:<26} "
            + "  ".join(
                f"{src[0]}{leg[0]}={entry[f'{src}_{leg}']}"
                for src in ("goal", "draft")
                for leg in ("dense", "lexical", "fused")
            )
        )

    columns = [
        f"{source}_{leg}"
        for source in ("goal", "draft")
        for leg in ("dense", "lexical", "fused")
    ]
    summary = {}
    print(f"\n{'screen':<16} " + "  ".join(f"k<={k:<5}" for k in K_GRID))
    for column in columns:
        ranks = [r[column] for r in results]
        cov = {k: sum(1 for x in ranks if x is not None and x <= k) for k in K_GRID}
        summary[column] = {"coverage": cov, "of": len(results)}
        print(f"{column:<16} " + "  ".join(f"{cov[k]:>2}/{len(results):<4}" for k in K_GRID))

    per_family: dict[str, dict] = defaultdict(dict)
    for column in columns:
        for base in MISSED_FAMILIES:
            subset = [r[column] for r in results if r["base"] == base]
            per_family[base][column] = {
                "top5": sum(1 for x in subset if x is not None and x <= TOP_K),
                "n": len(subset),
            }
    print(f"\nper family, at TOP-{TOP_K}:")
    for base, block in per_family.items():
        print(f"  {base}")
        for column, cell in block.items():
            print(f"    {column:<16} {cell['top5']}/{cell['n']}")

    payload = {
        "database": args.dsn.rsplit("/", 1)[-1],
        "generation": generation,
        "population": len(results),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "k_grid": list(K_GRID),
        "embedder": EMBEDDER_ID,
        "summary": summary,
        "per_family": per_family,
        "sessions": results,
    }
    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "direction-screen.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
