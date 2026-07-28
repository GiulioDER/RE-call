"""Refuse to score a tenant a LOCOMO reader is about to read that holds zero rows.

Companion to the two builder-side guards in `recall.eval.locomo.index_conversation`:
the PRE-condition (its message: "tenant ... already holds N chunk(s)") catches a sequential
re-index over an already-populated tenant, and the POST-condition (its message: "... table
CONCURRENTLY") catches a second writer landing mid-index. Both catch a corpus that came out too
BIG — doubled, by a re-run or a race.

Neither catches the opposite failure: a corpus that came out too SMALL. On 2026-07-27 a build job
(`recall.eval.locomo`) died silently after indexing 1 of 10 conversations, leaving `locomo_chunks`
holding a single populated tenant (419 rows) against a correct 5,882. Nothing raised —
`index_conversation`'s guards only cover the tenant IT is writing; the nine tenants it never got
to are simply absent, which looks identical to "not built yet" from outside. The two readers that
later score against that table (`recall.eval.locomo_abstention`, `recall.eval.locomo_entailment_
sweep`) do not build anything — each opens a store per tenant, calls `store.count()`, and goes
straight to scoring. Had one run against the partial corpus, it would have returned a perfectly
plausible-looking separation number computed over 10% of the intended data — indistinguishable,
from the output alone, from a real result.

`check_tenants_populated` is the fix, factored out so both readers share ONE check instead of each
growing (or each forgetting to grow) its own copy. It takes the tenant -> row-count mapping the
caller already has — every LOCOMO reader calls `store.count()` per tenant for its own provenance
block anyway — and refuses if any of THOSE tenants are empty. It never invents a roster of its
own: a hardcoded "10" would fire (or fail to fire) on the wrong thing the moment someone runs
`--conversations 3`.

Deliberately DB-free: this module imports nothing beyond the standard library. The caller reads
Postgres (via `PgVectorStore.count()`) and hands this function the plain result; this function
never opens a connection itself. That keeps the policy ("is this run's corpus complete enough to
score") unit-testable with a plain dict — no live database, no psycopg, no pgvector — and keeps a
reader that imports this module from pulling in a transitive DB dependency it did not ask for just
to get the check.
"""
from __future__ import annotations

from collections.abc import Mapping


def check_tenants_populated(counts: Mapping[str, int], *, table: str) -> None:
    """Raise if any tenant this run is about to score against holds zero rows.

    `counts` must be scoped by the CALLER to exactly the tenants this run iterates — built from
    the same (already `--conversations`/`limit`-sliced) conversation list the caller is about to
    score, one `store.count()` per tenant. This function does not know, and does not need to
    know, how many tenants exist in the full dataset; it only ever looks at what it was handed.
    A deliberately limited run (`--conversations 3`) therefore checks exactly 3 tenants and can
    never fail because some OTHER tenant, outside this run, happens to be empty — and it can
    never fail to catch a partial run just because the full dataset happens to have 10.

    Every empty tenant is collected and named in ONE error, not just the first: "1 of 10 present"
    is the fact an operator needs in order to judge how much rebuilding is required, and stopping
    at the first empty tenant would hide that eight more are also missing.
    """
    empty = sorted(t for t, c in counts.items() if c == 0)
    if not empty:
        return
    total = len(counts)
    present = total - len(empty)
    raise RuntimeError(
        f"UNBUILT-TENANT: only {present}/{total} tenant(s) in table {table!r} are populated — "
        f"{len(empty)} hold ZERO rows and were never indexed: {empty}. A reader scores against a "
        f"corpus recall.eval.locomo already built; an empty tenant here means that build died "
        f"partway through (or never ran for it), and scoring it anyway would silently report a "
        f"plausible-looking number computed over missing data. Rebuild the corpus first: "
        f"`python -m recall.eval.locomo --data locomo10.json --table {table}` — then re-run this "
        f"reader."
    )
