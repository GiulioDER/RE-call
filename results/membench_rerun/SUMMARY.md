# mem-bench re-run, RE-call 0.7.0 — 2026-08-01

Prior work searched 2026-08-01. `docs_search(source_type="memory")` is UNAVAILABLE (the corpus is
served from VPS2, which is down; the VPS3 mirror has no `docs_chunks` relation), so this fell back
to `MEMORY.md` + grep over `~/.claude/.../memory` and the two repos. Load-bearing, and this run
exists because of them: MEMORY.md records that the eight board artifacts "restano `(declared)`
finché non li ri-esegui (serve Postgres+pgvector)" and "⛔ NON editarli", and that the board carries
RE-call **at the defaults**, not a tuned config. `benchmarks/membench/README.md` is the other
load-bearing source; it is what predicted the flat temporal result below.

Both artifacts **VERIFY OK** against mem-bench's own verifier, which recomputes every figure from
the raw rows and ignores the submitted summary.

Config is **byte-identical** to the board's existing rows (`{"dim": 384, "embedder":
"BAAI/bge-small-en-v1.5", "reranker": null}`), so this is like-for-like. The only difference is
`system_version`: the board declares **0.6.0**, this run reports **0.7.0** read live from
`recall.__version__`.

## Isolation (axis 2)

| metric | board (0.6.0, declared) | this run (0.7.0, verified) |
|---|---|---|
| `leak_rate` | 0.0000 | **0.0000** |
| `indeterminate_leak_rate` | 0.0000 | **0.0000** |
| `own_tenant_recall` | 0.8417 | **0.9917** |
| `own_recall_T1-disjoint` | 0.8250 | 1.0000 |
| `own_recall_T2-near-duplicate` | 0.9000 | 0.9750 |
| `own_recall_T3-targeted` | 0.8000 | 1.0000 |

Zero leakage held, and own-tenant recall rose by 0.15. ⚠️ **This is NOT attributable to the
2026-08-01 session.** It spans 0.6.0 → 0.7.0, which covers far more than one day's work. Anyone
citing it needs to bisect the versions first.

## Temporal (axis 3)

| metric | board | this run | delta |
|---|---|---|---|
| `covering_selection_rate` | 0.2417 | 0.2500 | +1 instance |
| `contradiction_rate` | 0.7500 | 0.7417 | −1 instance |
| `covering_T1-stable` | 0.9667 | 0.9667 | 0 |
| `covering_T2-revised-ref-after` | 0.0000 | 0.0000 | 0 |
| `covering_T3-revised-ref-before` | 0.0000 | 0.0000 | 0 |
| `covering_T4-role-confusion` | 0.0000 | 0.0333 | +1 of 30 |
| `recency_trap_rate` | 0.0000 | 0.0000 | 0 |
| `answered_rate` | 1.0000 | 1.0000 | 0 |

**Flat.** 120 instances means one instance is 0.0083, so every non-zero delta above is a single
instance. There is no signal here.

🔑 **And that is expected, not disappointing.** `benchmarks/membench/README.md` says it outright:
`known_as_of` is **not wired into the temporal adapter**, and axis 3 scores **valid** time while
the 2026-08-01 work (edge dating, fan-in by assertion time, `first_indexed_at`) is **transaction**
time. The session's work is invisible to this axis by construction. Reading "we did temporal work
and the temporal axis did not move" as a negative result would be reading the wrong instrument.

T2 and T3 sitting at exactly 0.0000 on both runs is the real finding, and it predates today: the
system does not select the covering instance for a revised reference in either direction.

## What this run actually buys

The board's eight artifacts render `(declared)` because they assert 0.6.0 with nothing confirming
it. These two are **verified** at 0.7.0, recomputed from raw rows. That is the value here, not the
numbers moving.

## Not done

- Only 2 of the 8 artifacts (RE-call isolation + temporal). The abstention axis was not re-run.
- **Nothing was submitted.** These live in `results/membench_rerun/`, not `submissions/`; the
  verifier noted "not under submissions/, convention not applicable". Publishing is a separate
  decision, and mem-bench is still private pending the flip.
- A calibration warning fires on both runs: no calibration exists for `BAAI/bge-small-en-v1.5`, so
  abstention used the untuned 0.50 cosine floor. The board's run has the same gap, so it does not
  break the comparison, but neither run is measuring a calibrated system.

---

# Follow-up: larger manifests, and the bisect

## 1. Larger existing manifests (no new versions minted)

| axis / manifest | instances | result |
|---|---|---|
| isolation **v3** | 120 | `leak_rate` 0.0000 · **`own_tenant_recall` 1.0000** |
| temporal **v4** | 180 | `covering_selection_rate` 0.1889 · `contradiction_rate` 0.7944 · `recency_trap_rate` 0.0167 |

Both `VERIFY OK`. These are **not** comparable to the board, which publishes v1 on both axes.

Temporal v4 per tier, and the last row is the one to read:

| tier | covering | trap |
|---|---|---|
| T1-stable | 1.0000 | 0.0000 |
| T2-revised-ref-after | 0.0000 | 0.0000 |
| T3-revised-ref-before | 0.1000 | 0.0333 |
| T4-role-confusion | 0.0000 | 0.0333 |
| T5-boundary | 0.0333 | 0.0333 |
| **T6-write-time** | **0.0000** | 0.0000 |

🔑 **v4 introduced a `T6-write-time` tier, which is transaction time — precisely what the
2026-08-01 work implements — and RE-call scores 0.0000 covering on it.** That is not a measurement
of the new code: `benchmarks/membench/README.md` states `known_as_of` is not wired into the
temporal adapter. T6 is the tier that would exercise it, and the adapter cannot answer it. Wiring
it is a concrete, bounded piece of work with a metric already waiting for it.

## 2. The isolation "gain" is NOT RE-call. Do not claim it.

The bisect was run with the **adapter held fixed at master** and only `recall/` varying:

| commit | `recall.__version__` | `own_tenant_recall` |
|---|---|---|
| `ff7bb5d` (earliest the current adapter can run) | 0.6.0 | **0.9917** |
| `70996ea` (master) | 0.7.0 | **0.9917** |
| board artifact, 2026-07-31 | 0.6.0 (declared) | **0.8417** |

Everything else is identical and was checked, not assumed:

- **manifest**: my v1 run records digest `b36155831d4aae20…`, byte-identical to the board's.
- **config**: `{"dim": 384, "embedder": "BAAI/bge-small-en-v1.5", "reranker": null}` on both.
- **`candidate_k`**: the adapter passes 20, which IS `DEFAULT_CANDIDATE_K`, so it is a no-op.
- **stability**: 0.9917 reproduced on three independent runs.

So the library scores the same at 0.6.0 and at 0.7.0, on the same manifest, with the same config.
**The delta lives in the adapter, not in RE-call.** The board's row was produced on 2026-07-31 by
the adapter copy that lived on VPS2, which was lost when that volume failed; the adapter in git is
a reconstruction. The current one cannot even RUN against 0.6.0 (`trusted_search()` gained
`candidate_k` at `ff7bb5d`, after the 0.6.0 release), which is direct evidence the two differ.

🔑 **That is worse than `(declared)`.** `(declared)` means the version is unconfirmed. This figure
cannot be regenerated at all, because the code that produced it no longer exists. Any writeup
claiming "own-tenant recall improved from 0.84 to 0.99" would be attributing an adapter difference
to the library.

**What would settle it:** nothing available. The only honest move is to replace the board row with
a reproducible one and say the earlier figure is superseded rather than beaten.

---

# 3. Option A tested: the temporal axis was measuring nothing

**The library did not change. The adapter was starving it.**

`membench/axes/temporal/run.py` offers an optional `load_metadata` hook and says why in its own
comment: *"the intervals as structured data, for systems that accept them ... this axis measures
SELECTION, not extraction from prose (spec section 8)."* mem-bench ships a reference
implementation that uses it. RE-call's adapter never implemented it, so **0 of 420 stored chunks
carried `valid_from`, `valid_until`, or any clock at all**. `now=reference_time` was threaded into
`trusted_search` with nothing to compare against, so `expired` and `not_yet_valid` could not fire.

Every temporal figure this project has published was scoring plain semantic retrieval.

## Result, manifest v4 (180 instances), library untouched

| tier | before (inert) | + valid time | + interval translation |
|---|---|---|---|
| T1-stable | 1.0000 | 1.0000 | 1.0000 |
| T2-revised-ref-after | 0.0000 | 1.0000 | 1.0000 |
| T3-revised-ref-before | 0.1000 | 0.9667 | 0.9667 |
| T4-role-confusion | 0.0000 | 0.9667 | 0.9667 |
| T5-boundary | 0.0333 | 0.0000 | 0.9667 |
| T6-write-time | 0.0000 | 1.0000 | 1.0000 |
| **`covering_selection_rate`** | **0.1889** | 0.8222 | **0.9833** |
| **`contradiction_rate`** | **0.7944** | 0.1611 | **0.0000** |
| `recency_trap_rate` | 0.0167 | 0.0000 | 0.0000 |

All `VERIFY OK`.

## Two corrections to my own scoping

🔑 **I said T6-write-time could not be answered from valid time alone. Wrong, and the run said so.**
T6 puts the reference time between `asserted_at` and `effective_from`, i.e. the fact is announced
but not yet in force. A valid-time window handles that on its own: the announced assertion is
`not_yet_valid` and is excluded, so the still-effective predecessor is served. T6 went to 1.0000
with **no** transaction time wired. The case for wiring `known_as_of` here is now weaker, not
stronger, and it should not be built on this evidence.

⚠️ **The first attempt double-indexed the corpus.** Re-indexing under a fresh `mkdtemp` meant
`replace_sources` keyed on a new absolute path and INSERTED: 840 rows for 420 documents, every fact
present twice, half of them with no validity metadata. Caught by counting rows before reading any
score. One work directory per adapter instance fixes it.

## The interval translation, and why it is not fitting

mem-bench's covering predicate is **half-open** — `reference_time < effective_to`
(`axes/temporal/corpus.py:63`). RE-call reads `valid_until` as **inclusive to end-of-day**. Passing
`effective_to` through unchanged kept every assertion alive one day too long, which is the entire
`T5-boundary = 0.0000`. Converting the exclusive end to the previous day is a translation between
two interval conventions, which is what an adapter is for. Reporting the zero without it would have
published an adapter artefact as a RE-call weakness.

## Is this leakage? No, and it was checked

- `METADATA_FIELDS` is `(entity, field, kind, effective_from, effective_to, asserted_at)` —
  **`covering_doc_ids` is not in it.** The adapter never sees which document is the answer.
- The answer is *which* assertion covers the reference time. RE-call still derives that itself,
  through `_verdict`, from the window and the question's `now`.
- `asserted_at` is available and deliberately **not** used, so this is a one-variable change.

## What this does and does not license

- ✅ The temporal axis can now measure RE-call's validity layer. Before, it could not.
- ⛔ **Not a RE-call improvement.** The library is byte-identical. Any writeup saying "covering
  went from 0.19 to 0.98" without saying "because the harness was not passing the intervals" would
  be claiming a capability gain for a measurement fix.
- ⛔ The board's published temporal rows were produced by the inert adapter. They are superseded,
  not beaten.
