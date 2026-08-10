# The Answerability Ladder v3 — embedder generality

**Date:** 2026-07-29 · **System:** RE-call at shipped defaults, embedder `thenlper/gte-base`
**Archived manifest:** `results/ladder/manifest_v2.jsonl` — **the same frozen manifest as v2**, digest
`5534c61356acaa7b62ac5a79dbec7383674fc052984d10c1d0cc89e26a532bd5`, verified at run start via
`--expected-digest`
**Archived responses:** `results/ladder/responses_v3_gte.jsonl`, 1 200 scored, none missing
**Pre-registration:** [`benchmarks/PREREGISTRATION-ladder-v3.md`](../../benchmarks/PREREGISTRATION-ladder-v3.md)
**Analysis:** `benchmarks/ladder/analyze_v3.py`, **committed before this arm finished** and
validated by reproducing every published v2 figure

**Prior work searched** before the design was fixed — `docs_search(source_type="memory")` on
"voyage calibrated abstention threshold arm ladder rerank best configuration". The two load-bearing
hits are cited in place below and both changed this arm rather than decorating it:
[[project-recall-threshold-embedder-fragile-2026-07-28]] (0.50 is inert on `bge-small` *and*
`bge-large`, which is where P4 came from) and
[[project-recall-beam-bestconfig-blocked-2026-07-28]] (on `voyage-4-large` the same constant
starves 23.3 % — the opposite failure, and the reason this arm's result must not be generalised to
API embedders).

## Verdict: all four pre-registered predictions PASS

```
                          v2 (bge-small, 384d)   v3 (gte-base, 768d)
AUC answerable vs r=0.00         0.567                  0.570
AUC answerable vs r=0.25         0.784                  0.794
AUC answerable vs r=0.50         0.841                  0.841
AUC answerable vs r=0.75         0.921                  0.925
AUC answerable vs r=1.00         0.968                  0.976

monotone (non-incr.) per Q     173/200 = 0.865       173/200 = 0.865
responses below 0.50 floor       1 / 1200               0 / 1200
```

**The discrimination curve reproduces across embedder families — within 0.010 at every rung, and
identical at `r=0.50`.** Different family (GTE vs BGE), different dimensionality (768 vs 384), and
entirely non-overlapping cosine scales (0.7620–0.9332 against 0.4945–0.8238). The *shape* survives
all of it.

**The absolute magnitudes do not transfer, and that was the point of testing sign rather than
size.** The within-unanswerable paired deltas are ~2.1× smaller here:

| rung | v2 delta | v3 delta |
|---|---|---|
| `r=0.25` | −0.0397 | −0.0176 [−0.0201, −0.0152] |
| `r=0.50` | −0.0539 | −0.0234 [−0.0259, −0.0208] |
| `r=0.75` | −0.0837 | −0.0390 [−0.0422, −0.0358] |
| `r=1.00` | −0.1100 | −0.0531 [−0.0565, −0.0497] |

Every CI still excludes zero, monotone throughout. Had this file pre-registered "reproduces v2's
−0.1100" it would now read as a failure of a real effect — which is exactly why P1–P3 were fixed on
sign, monotonicity and ordering.

**`173/200` appearing in both arms is a coincidence, and I checked rather than reported it.** The
two monotone sets are *not* the same questions — 156 of 173 overlap. The per-step triples also
differ (155/32/187 vs 161/30/191). So the identical count is chance; what is substantive is that
**90 % of the questions monotone under `bge-small` are also monotone under `gte-base`** — agreement
per question, not merely in aggregate.

## The near-rung limit reproduces — and this is the load-bearing result

`AUC 0.570` at `r=0.00`, against v2's `0.567`. **Barely above chance, on a second independent
embedder family.** With the whole topic present and only the supporting turn removed, top-1 cosine
cannot tell an answerable question from an unanswerable one. That was the limit the v2 addendum
added; it is now shown not to be a `bge-small` artefact.

Discrimination crosses AUC 0.90 between `r=0.50` and `r=0.75` in **both** arms — i.e. only once
half to three-quarters of the topic is gone.

## P4: the shipped floor is inert on a second family, decisively

**0 of 1200** responses fall below 0.50, over an observed range of [0.7620, 0.9332] — the entire
distribution sits above 0.76. v2 had 1 of 1200. Two independent embedder families on which the
shipped constant cannot express a signal the system demonstrably has. Predicted in advance from
[[project-recall-threshold-embedder-fragile-2026-07-28]], not discovered here.

| | prediction | measured | |
|---|---|---|---|
| **P1** (kill condition) | gradient reproduces, CI excludes 0 | −0.0531 [−0.0565, −0.0497] | ✅ |
| **P2** | rung means monotone | monotone | ✅ |
| **P3** | per-question ≥ 0.70 | 0.865 | ✅ |
| **P4** | floor inert (≤ 0.02 every rung) | 0.000 | ✅ |

## The deflating reading, which §5 pre-registered and which now looks correct

The pre-registration said: *"If P1 holds, the most likely reading is mundane — any dense retriever's
top-1 similarity falls as you delete the relevant documents. That would make the axis real but
unsurprising."*

**That is the honest reading of this arm.** The reproduction is so clean across two unrelated
families that the mechanism is almost certainly generic to dense retrieval, not a property of
RE-call. The benchmark's interesting content therefore is *not* "the gradient exists" — it is where
v2 put it: **a graded signal sitting underneath a binary decision that cannot express it**, plus the
sharp limit that at the boundary itself the signal is worth almost nothing.

## What this does NOT establish

- **Not a quality comparison.** Nothing here says `gte-base` is better or worse than `bge-small`.
  Retrieval quality is unmeasured; only the shape of the distance response.
- **Two families is not "embedder-independent."** It raises confidence and says nothing about API
  embedders — and note that on `voyage-4-large` the 0.50 floor *starves 23.3 %*
  ([[project-recall-beam-bestconfig-blocked-2026-07-28]]), the opposite failure from inertness. A
  third arm there would likely look different from both of these.
- **Not a single-variable change.** Family, size and dimensionality all moved together; the arm
  cannot attribute the agreement to any one of them.
- **Same corpus, same distractors, no judge.** A shared-corpus artefact would reproduce here rather
  than be caught, and answer *correctness* remains unmeasured.
