"""Does a DOUBLED corpus explain §9a's published pool-100 number?

The question
-----------
`results/locomo/postfix_pool100.json` publishes hit@5 **0.5957** at `candidate_k=100`. Re-run today
on a clean corpus at the same stated configuration, the same arm reads **0.6615** — higher at every
depth (+0.004 / +0.066 / +0.063 / +0.039 at k=1/5/10/20).

It is not code drift: the only commit touching the retrieval path between the published artifact
(`3ee36ed`, 2026-07-26) and this branch is `9eb3bc1`, whose sole change to `recall/store.py` is a
`tenant` property accessor, and which does not touch `recall/retriever.py` at all.

It is not the `ef_search` truncation that voided the *earlier* pool-100 control: `query_dense(k=100)`
returns a full 100 rows today, measured directly.

The hypothesis
--------------
`9eb3bc1`'s own commit message documents a contamination incident: two run scripts wrote into the
same tables, every tenant held its corpus **twice** (11,764 rows against a correct 5,882), nothing
errored, and *"every depth of the LOCOMO curve came in about 0.05 low"* — the mechanism being that a
fixed-size candidate pool holds roughly half as many DISTINCT documents when every document appears
twice.

Today's clean arm C sits ~0.05-0.066 ABOVE the published pool-100 figure. That is the same signature,
inverted. So: **was `postfix_pool100.json` measured on a doubled corpus?**

The timing permits it. The guard that now refuses to index over an existing corpus landed in
`9eb3bc1` on 2026-07-28 — *after* both published artifacts were produced on 07-26. A doubled run was
possible then and is impossible now.

Preregistered predictions (written before running)
--------------------------------------------------
- **C2 (doubled, pool 100) lands within ±0.01 of 0.5957.** If it does, the published pool-100 number
  is contaminated and §9a's "a 5× deeper pool dilutes" rests on it.
- **A2 (doubled, pool 20) falls well below 0.6706.** This is the control that makes the result
  interpretable rather than a coincidence: the contamination note says doubling costs ~0.05 at every
  depth, so it must hurt pool 20 too. If A2 is unaffected, the mechanism is depth-specific and the
  C2 match would be luck.
- **Consequence if both hold:** the two published artifacts came from *different corpus states* —
  pool-20 clean (it reproduces to 0.0000 today) and pool-100 doubled. Odd but entirely possible
  before the guard existed.

Falsification: C2 landing near 0.66, or anywhere that is not ~0.596, refutes the hypothesis and the
discrepancy needs another explanation. That outcome ships too.

Usage
-----
::

    python -m scripts.probe_doubled_corpus --data locomo10.json --dsn "$RECALL_DSN" \
        --candidate-k 100 --out results/wrrf/doubled_pool100.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recall.eval import locomo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure a LOCOMO arm on a deliberately doubled corpus")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--dsn", required=True)
    p.add_argument("--embedder", default="fastembed")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument("--table", required=True, help="dedicated table; it WILL be indexed into twice")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)

    common = dict(
        dsn=a.dsn,
        embedder_name=a.embedder,
        k=a.k,
        limit=None,
        keep_corpus=None,
        table=a.table,
        ks=[1, 5, 10, 20],
        candidate_k=a.candidate_k,
    )

    # Pass 1 — a normal, clean index. Its report is discarded; only the side effect matters.
    print("pass 1/2: indexing a clean corpus", flush=True)
    locomo.run(a.data, **common)

    # Pass 2 — index the SAME tenants again. `allow_existing` is the deliberate override for the
    # guard that now makes this impossible by accident; here it is the entire point of the probe.
    print("pass 2/2: indexing the SAME tenants again -> doubled corpus", flush=True)
    report = locomo.run(a.data, allow_existing=True, **common)

    report["probe"] = "doubled_corpus"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    o = report["retrieval_overall"]
    print(f"\nDOUBLED corpus, pool {a.candidate_k}: hit@5 {o['rate']:.4f}  n={o['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
