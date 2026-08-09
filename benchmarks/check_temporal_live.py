"""End-to-end: after the ingest change, do REAL indexed chunks carry a validity window?

Prior work: same search as `benchmarks/check_temporal_inert.py`; see that file's docstring for the
two memos that shaped this (supersession and validity time are already shipped; what was missing
was that benchmark data never populated a window).

This is the empirical half. `check_temporal_inert.py` and `tests/test_locomo_turn_validity.py`
both work on the DOCUMENT, so they prove the text changed. Neither proves the window survives the
Indexer's chunking and reaches the store, which is where it has to arrive for `research_search` to
act on it. A document-level pass with a store-level failure would look exactly like success.

PRE-REGISTERED:

  P1  Every chunk from a real LOCOMO conversation now has a non-null `valid_from`.
      (Before the change this was 0 of N; that is the measured baseline in check_temporal_inert.)
  P2  Asking as-of a date BEFORE a session makes that session's turns `not_yet_valid`, so the
      count of usable hits at an early reference time is strictly lower than at wall clock.
  P3  CONTROL: at wall clock, nothing is `not_yet_valid`, because every session predates today.
      If P3 fails, the reference time is not reaching the trust layer and P2 proves nothing.

Run:  RECALL_DSN=postgresql://recall:recall@localhost:5432/recall \
      python benchmarks/check_temporal_live.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone

from recall.embeddings import FastEmbedEmbedder
from recall.eval.locomo import index_conversation
from recall.frontmatter import validity_bounds
from recall.store import PgVectorStore
from recall.types import TrustedResult

from benchmarks._trust import bench_search

DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")
TENANT = "temporal-live-check"
TABLE = "chunks_temporal_check"


def main() -> int:
    conv = json.load(open("locomo10.json", encoding="utf-8"))[0]
    inner = conv["conversation"]
    dates = sorted(v for k, v in inner.items() if k.endswith("_date_time"))
    print(f"conversation sessions: {len(dates)}  first={dates[0]!r}")

    emb = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    with PgVectorStore(DSN, dim=emb.dim, tenant=TENANT, table=TABLE) as store:
        store.ensure_schema()
        store.drop_table()
        store.ensure_schema()
        n = index_conversation(store, emb, inner, allow_existing=True)
        print(f"indexed turns: {n}")

        metas = [c.metadata for c in store.iter_chunks()]

        total = len(metas)
        with_window = sum(1 for m in metas if validity_bounds(m)[0] is not None)
        print(f"\nP1  chunks with a non-null valid_from: {with_window}/{total}")
        p1 = total > 0 and with_window == total

        # A reference time before the whole conversation, and wall clock.
        early = datetime(2000, 1, 1, tzinfo=timezone.utc)
        wall = datetime.now(timezone.utc)
        q = "What did they say about their plans?"

        # `bench_search`, not a bare `research_search`: P2 and P3 below are BOTH read off
        # `hit.verdict`, and development mode without an explicit calibration rewrites every
        # verdict to `unverified`. That makes P2 (`not_yet_valid > 0`) structurally unreachable
        # and P3 (`not_yet_valid == 0` at wall clock) vacuously true — so this pre-registered
        # check would print "NOT live" and exit 1 whatever the validity layer did, and its own
        # control would be the thing certifying that nothing was wrong. Exactly the
        # "document-level pass with a store-level failure would look exactly like success"
        # this file was written to rule out, one layer further down.
        res_early = bench_search(store, emb, q, k=10, now=early)
        res_wall = bench_search(store, emb, q, k=10, now=wall)

        def verdicts(res: TrustedResult) -> Counter:
            return Counter(h.verdict for h in res.hits)

        v_early, v_wall = verdicts(res_early), verdicts(res_wall)
        print(f"P2  verdicts asked as-of 2000-01-01 : {dict(v_early)}")
        print(f"P3  verdicts asked at wall clock     : {dict(v_wall)}")

        p2 = v_early.get("not_yet_valid", 0) > 0
        p3 = v_wall.get("not_yet_valid", 0) == 0

        print()
        for name, ok in (("P1 every chunk carries a window", p1),
                         ("P2 early as-of yields not_yet_valid", p2),
                         ("P3 CONTROL wall clock yields none", p3)):
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")

        print()
        if not p3:
            print("VERDICT: CONTROL FAILED -- reference time is not reaching the trust layer.")
            return 1
        print("VERDICT:", "temporal layer is LIVE end to end" if (p1 and p2) else "NOT live")
        store.drop_table()
        return 0 if (p1 and p2 and p3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
