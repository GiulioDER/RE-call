"""K-2 v2 on LoCoMo: embed the stored top-30 items and report the mechanism share. Voyage only.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-same-subject-adjacency-v2.md, step 3 and
apparatus checks 2 and 3.

The stored retrieval (``collected-S.json.gz``) keeps each item's window text but no vector, so each
distinct window text in the top 30 of the committed 720 questions is embedded once with the
registered profile C9's primary index uses (``voyage-code-4-v1``, passage mode). Where the collect
left the same text in its embedding cache, the cached vector (what the index holds) is compared with
the recomputed one; they must agree to cosine at least 0.99 (check 3). The vectors are written keyed
by item id for ``scripts/aml_t1k2_locomo.py``'s K2v2 arm, and the share of questions whose top 30
v2 reorders is reported against the record's gates (check 2 is the render-only property).

    python scripts/aml_k2v2_vectors.py --collected C.json.gz --draw draw.json \\
        --cache embeddings-S.sqlite --vectors vectors.json.gz --out mechanism.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aml_t1k2_locomo import _items, load_collected  # noqa: E402

from recall.cache import EmbeddingCache, cache_key  # noqa: E402
from recall.embeddings import embed_passages, embedding_profile, resolve_registered_embedder  # noqa: E402
from recall_aml.conflict_order import TAU_V2, WINDOW, same_subject_adjacent  # noqa: E402
from recall_aml.window_format import dated_items  # noqa: E402

AGREEMENT_FLOOR = 0.99


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--draw", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    collected = load_collected(args.collected)
    chosen = set(json.loads(args.draw.read_text(encoding="utf-8"))["ids"])
    rows = [row for row in collected["rows"] if row["id"] in chosen]
    texts: dict[str, str] = {}
    for row in rows:
        for item in row["items"][:WINDOW]:
            texts.setdefault(str(item["id"]), str(item.get("content") or ""))
    ids = sorted(texts)

    embedder = resolve_registered_embedder("voyage-code-4-v1", {"VOYAGE_API_KEY": os.environ["VOYAGE_API_KEY"]})
    fresh: list[list[float]] = []
    for start in range(0, len(ids), 64):
        fresh.extend(embed_passages(embedder, [texts[i] for i in ids[start : start + 64]]))
    vectors = dict(zip(ids, fresh, strict=True))

    # Check 3, read-only: work on a copy so nothing is ever written into the collect's cache.
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / "cache.sqlite"
        shutil.copyfile(args.cache, copy)
        profile = embedding_profile(embedder)
        keys = {i: cache_key(profile, embedder.dim, texts[i], "passage") for i in ids}
        with EmbeddingCache(copy) as cache:
            cached = cache.get_many(list(keys.values()))
    agreement = [cosine(vectors[i], cached[keys[i]]) for i in ids if keys[i] in cached]

    moved = 0
    render_only = True
    for row in rows:
        base = dated_items(_items(row["items"]))
        v2 = same_subject_adjacent(base, vectors=[vectors.get(item.id) for item in base[:WINDOW]])
        moved += [item.id for item in v2[:WINDOW]] != [item.id for item in base[:WINDOW]]
        render_only &= sorted(item.model_dump_json() for item in v2) == sorted(item.model_dump_json() for item in base)
        render_only &= [item.id for item in v2[WINDOW:]] == [item.id for item in base[WINDOW:]]
    share = moved / len(rows)
    with gzip.open(args.vectors, "wt", encoding="utf-8") as sink:
        json.dump(vectors, sink)
    result = {
        "questions": len(rows),
        "distinct_items_embedded": len(ids),
        "tau": TAU_V2,
        "k2v2_share_top30_order_changed": share,
        "gate_inert_below_0_05": share < 0.05,
        "gate_links_everything_above_0_90": share > 0.90,
        "check_2_render_only": render_only,
        "check_3_cache_hits": len(agreement),
        "check_3_min_cosine_vs_index": min(agreement) if agreement else None,
        "check_3_passed": (min(agreement) >= AGREEMENT_FLOOR) if agreement else None,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
