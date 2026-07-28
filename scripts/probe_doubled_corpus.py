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
        --candidate-k 100 --table dbl_c2 --out /tmp/doubled_pool100.json

`--table` is REQUIRED (the table is indexed into twice, so it must be a dedicated one), and
`--out` deliberately points at a scratch path: `results/wrrf/doubled_pool100.json` is a RETAINED
artifact backing the §9a retraction, and `results/ARTIFACTS.md` forbids aiming `--out` at one.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from recall.eval import locomo

#: A bare SQL identifier. The table name reaches a `count(*)` that psycopg cannot parameterise
#: (identifiers are not values), so it is validated rather than trusted.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def corpus_rows(dsn: str, table: str) -> int:
    """Total chunk rows in `table`, across every tenant.

    Table-wide rather than per-tenant: LOCOMO indexes one tenant per conversation, and the
    doubling this probe induces is a property of the whole table.
    """
    import psycopg

    if not _IDENTIFIER.match(table):
        raise SystemExit(f"refusing to interpolate {table!r} into SQL: not a bare identifier")
    with psycopg.connect(dsn) as conn:
        row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608 - validated
    return int(row[0]) if row else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a LOCOMO arm on a deliberately doubled corpus"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--table", required=True, help="dedicated table; it WILL be indexed into twice"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="permit writing over an existing --out (retained artifacts are cited; see ARTIFACTS.md)",
    )
    args = parser.parse_args(argv)

    if args.out.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.out} exists. Retained artifacts back published claims and a re-run would "
            f"replace one with numbers from a different config. Choose a scratch path, or pass "
            f"--overwrite if you really mean to replace it."
        )

    common = dict(
        dsn=args.dsn,
        embedder_name=args.embedder,
        k=args.k,
        limit=None,
        keep_corpus=None,
        table=args.table,
        ks=[1, 5, 10, 20],
        candidate_k=args.candidate_k,
    )

    # Pass 1 — a normal, clean index. Its report is discarded; only the side effect matters.
    print("pass 1/2: indexing a clean corpus", flush=True)
    locomo.run(args.data, **common)
    rows_pass1 = corpus_rows(args.dsn, args.table)

    # Pass 2 — index the SAME tenants again. `allow_existing` is the deliberate override for the
    # guard that now makes this impossible by accident; here it is the entire point of the probe.
    print("pass 2/2: indexing the SAME tenants again -> doubled corpus", flush=True)
    report = locomo.run(args.data, allow_existing=True, **common)
    rows_pass2 = corpus_rows(args.dsn, args.table)

    # The probe's whole premise is "the corpus is now doubled", and until this check it was
    # asserted rather than measured: nothing counted a row, and `locomo.run`'s report carries no
    # countable field. So if pass 2 had failed to double — a guard change, a wrong --table, a
    # `delete_sources` in between — the probe would have emitted a clean-looking number under the
    # label DOUBLED and REFUTED the contamination hypothesis on an apparatus that never ran the
    # treatment. That is precisely the failure this probe exists to document, reproduced by the
    # probe itself. RESEARCH_PROTOCOL.md names this invariant, and records that only a row count
    # ever caught it.
    if rows_pass1 == 0:
        # Checked separately, because `0 != 2 * 0` is False: an empty table would otherwise
        # SATISFY the doubling invariant and ship an artifact stamped `doubling_verified: true`
        # over a corpus that was never indexed at all. A guard whose degenerate case passes is
        # the failure mode this probe exists to document.
        raise SystemExit(
            f"APPARATUS FAILURE: pass 1 left ZERO rows in {args.table!r}. Nothing was indexed, so "
            f"there is no corpus to double and nothing here measures the doubled condition."
        )
    if rows_pass2 != 2 * rows_pass1:
        raise SystemExit(
            f"APPARATUS FAILURE: pass 1 left {rows_pass1} rows in {args.table!r}, pass 2 left "
            f"{rows_pass2} — expected exactly {2 * rows_pass1}. The corpus was NOT doubled, so "
            f"whatever this run measured is not the doubled condition. Refusing to write an "
            f"artifact labelled 'doubled_corpus'."
        )

    report["probe"] = "doubled_corpus"
    report["corpus"] = {
        "table": args.table,
        "rows_after_pass1": rows_pass1,
        "rows_after_pass2": rows_pass2,
        "doubling_verified": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    overall = report["retrieval_overall"]
    print(f"\nrows: {rows_pass1} -> {rows_pass2} (doubling verified)")
    print(f"DOUBLED corpus, pool {args.candidate_k}: "
          f"hit@5 {overall['rate']:.4f}  n={overall['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
