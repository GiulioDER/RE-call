# §9a's published pool-100 number was measured on a doubled corpus

**Date:** 2026-07-28 · **Verdict: `results/locomo/postfix_pool100.json` is contaminated. The
"5× deeper pool dilutes" finding built on it is an artifact.**
**Preregistration:** `scripts/probe_doubled_corpus.py` module docstring, committed `d5e1baa`
**before** the probe ran. **Both predictions hit.**

## The claim under test

`results/FINDINGS.md` §9a publishes:

> **A deeper pool measurably *dilutes* a fused prefix** — −0.075 at k=5, −0.073 at k=20 — which is
> §7's mechanism showing up on cue: RRF gives `dense[r]` and `sparse[r]` identical scores, so a
> five-fold deeper pool interleaves five times as many low-rank candidates into every prefix.

That rests on pool-20 **0.671** against pool-100 **0.5957**. A clean re-run today reproduces the
pool-20 figure exactly and the pool-100 figure not at all.

## The measurement

Four configurations, one session, same harness, same embedder, n=1,536 throughout.

| k | published §9a pool-100 | **doubled corpus, pool 100** | clean corpus, pool 100 |
|---|---|---|---|
| 1 | 0.3900 | **0.3913** | 0.3939 |
| 5 | **0.5957** | **0.5944** | **0.6615** |
| 10 | 0.6901 | **0.6914** | 0.7533 |
| 20 | 0.7819 | **0.7819** | 0.8210 |

**A deliberately doubled corpus reproduces the published pool-100 curve to within ±0.0013 at every
depth, and exactly at k=20.** The clean corpus does not, missing by up to +0.066.

### The control that makes it interpretable

| pool 20 | hit@5 |
|---|---|
| published §9a | 0.6706 |
| clean re-run | **0.6706** (Δ 0.0000) |
| doubled corpus | **0.6081** (Δ −0.0625) |

Doubling costs **−0.0625** at pool 20, so the contamination is *not* depth-specific — it would have
been plainly visible in the pool-20 artifact had that one been affected. It was not: pool-20
reproduces to **0.0000**.

So the two published artifacts came from **different corpus states** — pool-20 clean, pool-100
doubled. That is exactly the combination the preregistration predicted, and it was possible because
the guard that now refuses to index over an existing corpus landed in `9eb3bc1` on **2026-07-28**,
*after* both artifacts were produced on **07-26**.

## What this corrects

**The dilution is real but roughly one eighth of the published size.**

| | k=5 |
|---|---|
| published: pool 20 → pool 100 | 0.6706 → 0.5957 = **−0.0749** |
| measured clean today | 0.6706 → 0.6615 = **−0.0091** |

The *direction* survives — a deeper pool still costs a little — but "measurably dilutes" at −0.075
is an artifact of a corpus in which every document appeared twice, halving the DISTINCT documents a
fixed-size candidate pool could hold. The mechanism §9a attributes it to (RRF scoring `dense[r]` and
`sparse[r]` identically) may well be real; this measurement simply does not evidence it at anything
like that magnitude.

**This is the second correction to §9a's pool control.** The first withdrew its k=50 row, when the
pool-100 arm turned out to have run against an `hnsw.ef_search`-capped dense leg. The k=50 row was
withdrawn then; the k=5/10/20 rows were re-run and published — and those re-runs are what this
finding now implicates.

## Three explanations eliminated before the probe

Each by measurement, not argument:

- **Code drift.** The only commit touching the retrieval path between the published artifact
  (`3ee36ed`) and this branch is `9eb3bc1`, whose sole change to `recall/store.py` is a `tenant`
  property accessor; it does not touch `recall/retriever.py`.
- **`ef_search` truncation** — the cause of the *first* retraction. `query_dense(k=100)` returns a
  full 100 rows today, measured directly against the arm's own table.
- **Nondeterminism.** Pool-20 reproduces to 0.0000 at four depths in the same session.

## How it was found

Not by looking. Phase 1 preregistered an apparatus check — *"C ≠ 0.596 ± 0.01 → fail the run"* —
purely so that a weighted-fusion result could not be read against a baseline that had drifted. That
gate fired, and the investigation followed from it.

The irony is worth recording: the contamination this identifies is the same failure mode that
`9eb3bc1` was itself written to prevent, and its own commit message describes the signature —
*"every depth of the LOCOMO curve came in about 0.05 low"* — which is what made the hypothesis
formulable at all.

## Consequences

1. ~~**`results/FINDINGS.md` §9a's pool-100 row and its "measurably dilutes" paragraph need
   correcting.** Not done here — this branch is a research branch and the correction belongs to
   whoever owns that section.~~ **Done**, in the change that carried this file onto `master`: §9a
   and `RESULTS.md` now show the retraction notice and the corrected figures.
2. **Phase 1's motivating premise was largely false.** Weighted fusion was built to counter a −0.075
   dilution that measures −0.009. It also lost on its own terms (−0.0261 at pool 20), so the
   conclusion does not change — but the *motive* was much weaker than the spec claimed.
3. ~~**`results/locomo/postfix_pool100.json` should be withdrawn or re-run**, not annotated.~~
   **Deleted** in the same change, on the precedent of `9eb3bc1`'s own handling of contaminated
   artifacts: *"an annotated wrong number in `results/` is still a number someone can read off a
   table."* `arm_C_rrf_pool100.json` is the clean re-run that replaces it, and everything that
   pointed at the deleted file now points there.

## Reproduce

```bash
python -m scripts.probe_doubled_corpus --data locomo10.json --dsn "$RECALL_DSN" \
    --candidate-k 100 --table dbl_c2 --out results/wrrf/doubled_pool100.json
python -m scripts.probe_doubled_corpus --data locomo10.json --dsn "$RECALL_DSN" \
    --candidate-k 20  --table dbl_a2 --out results/wrrf/doubled_pool20.json
```

Artifacts: `doubled_pool100.json`, `doubled_pool20.json`, `arm_A_rrf_pool20.json`,
`arm_C_rrf_pool100.json`, all in this directory.
