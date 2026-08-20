# Pre-registration: does an offline-generated query set certify `re-call-docs` under voyage-4?

**Date:** 2026-08-20   **Status:** measured 2026-08-20, both arms; every prediction below is unedited

Tenant `re-call-docs` on VPS2 (`recall_repos`, port 55432), embedder `voyage:voyage-4`, corpus
`~/recall-repos/src/RE-call/**/*.md` (155 files, 1,624 KB).

## The question

Building a generation for `re-call-docs` with `voyage:voyage-4` and calibrating it against a query
set from `recall.wizard.queryset.generate_offline`, does the artifact **certify**, i.e. is the
separability 95% lower bound at or above `MIN_SEPARABILITY = 0.90`?

## Why it is not already answered

`docs/preregistrations/2026-08-16-generated-calibration-query-sets.md` measured exactly this
question on the same corpus family (this repository's `docs/`) and answered yes: the offline arm
certified at AUC 0.9806, lower bound 0.9496, threshold 0.7050. **That run used `fastembed` bge-small
at 384 dimensions.** This one uses voyage-4 at 1024, and a threshold does not transfer across
models, so neither the certification nor the threshold carries over.

The specific reason to doubt it: measured 2026-08-17 and recorded in
`calibrated-thresholds-and-the-overlap`, **voyage-4 returned cosines of only 0.269 to 0.413 on
`re-call-docs`**, where voyage-code-3 on the sibling code corpus returned 0.480 to 0.834. That is a
compressed regime, and compression leaves less room between the classes.

## What I predict

**P1. It certifies.** Separability 95% lower bound **at or above 0.90**, point AUC **0.95 to 1.00**.
Reasoning: the offline generator's failure mode measured on 2026-08-16 was a thinner, noisier set
rather than an unseparable one, and cosine compression shifts both classes together rather than
overlapping them. Separability is rank-based and therefore scale-free, so compression alone should
not hurt it.

**P2. The threshold lands far below the bge-small precedent.** Between **0.25 and 0.45**, against
the 0.7050 that bge-small produced on this corpus. Directly extrapolated from the 0.269 to 0.413
band measured on this exact corpus and model on 2026-08-17.

**P3. Certification and usable error will disagree, and this is the interesting one.** Even if the
AUC certifies, the per-class error at the fitted threshold will be **worse** than the memory
tenant's, whose false abstain is 9.09% and false confirm 3.57%. Predict false confirm **above 10%**
on at least one of the two classes. Compression squeezes the classes toward each other in absolute
terms even when their ORDER is preserved, and it is the fixed cut that pays for that, not the AUC.

## What would falsify this

- **P1 falsified** if the lower bound is below 0.90, or the point AUC below 0.95. Then an offline
  set cannot calibrate this tenant under voyage-4, and the honest outcome is that `re-call-docs`
  stays in development mode until either a labelled set or an LLM key exists.
- **P2 falsified** by a threshold outside 0.25 to 0.45.
- **P3 falsified** if both per-class error rates come in at or below 10%, which would mean
  compression costs nothing at the operating point and my reason for worrying about it is wrong.

## How it will be measured

1. Build a manifest over the 155 markdown files, then a generation with `voyage:voyage-4` and
   `chunk_text`, matching how the legacy `re-call-docs` tenant was chunked.
2. **Verify the apparatus before reading any number**: the generation must report
   `embedder_model = voyage:voyage-4`, and a query whose answer is in the corpus must retrieve the
   file that contains it. A calibration measured against a mis-built generation is meaningless.
3. Generate the query set with `generate_offline`, using **the same chunker the generation was
   built with**. `chunks_from_directory`'s docstring is explicit that a mismatch here breaks the
   invariant invisibly: measured on this repository's own `pipeline.py`, `chunk_text` and
   `chunk_code` produce 20 chunks against 8 with no exact string in common.
4. `recall calibration calibrate`, then read, in this order: certified yes/no, separability and its
   interval, the fitted threshold, and the per-class error at that threshold. Metrics named by
   their denominators: false abstain over the answerable count, false confirm over the unanswerable
   count.
5. Publish and promote ONLY if it certifies.

## What I already know

- `2026-08-16-generated-calibration-query-sets.md`: offline certifies on this corpus family at
  bge-small, 0.9806 / 0.9496 / 0.7050. LLM arm was better (AUC 1.0000) and is preferred wherever a
  key exists. **No LLM key is present on VPS2** (checked: no `OPENAI_API_KEY`,
  `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` in its `.env`), so the offline arm is not a choice
  here, it is the only arm.
- `calibrated-thresholds-and-the-overlap`: in 4 of 4 corpora the answerable and unanswerable
  distributions OVERLAP, and separability of 0.97 to 0.99 hid it completely. Read both numbers.
- `2026-08-20-calibration-carry-forward.md`: the memory tenant's operating point, for comparison.

## Confounds I can name now

1. **The offline generator's "unanswerable" queries come from a fixed subject list filtered against
   the corpus.** On a corpus that is itself about software, retrieval systems and calibration, that
   list may be less disjoint than it was for a generic corpus, which would depress AUC for a reason
   that has nothing to do with voyage-4.
2. **Labels are machine-generated and unreviewed**, the same unretired confound as every other
   calibration in this project.
3. **The legacy `re-call-docs` tenant chunked with `chunk_text`; I am asserting rather than
   verifying that.** If it in fact used a different chunker, the new generation is a different
   corpus from the one the client has been querying, and comparisons to its behaviour are void.
4. **One tenant, one embedder, one generator arm.** Nothing here licenses a claim about offline
   generation in general under compressed-cosine models.

## Result, voyage-4 arm (2026-08-20)

**Status:** measured. The predictions above are unedited.

Generation `gen_2d25ad7a26284894a259365e7c5de355`, 155 objects, 2,703 chunks, built in 4m13s.
Query set from `generate_offline(per_class=40, seed=0)` over 2,734 chunks read with `chunk_text`,
the chunker the generation was built with.

**Apparatus check passed before any calibration number was read.** The binding reported
`embedder_model = voyage:voyage-4` at dimension 1024, and "what are the generation states and how
does promotion work" retrieved `GENERATIONS.md`.

| | predicted | measured | |
|---|---|---|---|
| certified | **yes** | **NO** | falsified |
| separability | 0.95 to 1.00 | **0.8419**, CI [0.7540, 0.9297] | falsified |
| threshold | 0.25 to 0.45 | **0.382** | confirmed |
| per-class error above 10% | yes, at least one class | **both**: false abstain 25.00%, false confirm 27.50% | confirmed |

```
answerable min/max : 0.2895 / 0.7167
gap        min/max : 0.2660 / 0.4836
overlap min(a)-max(g): -0.1941
```

### P1 is falsified and the reason is the opposite of the one I argued

I predicted certification on the grounds that **separability is rank-based and therefore
scale-free**, so compressing the cosines could not hurt it. That reasoning is correct about
separability and wrong about this corpus: the classes do not merely sit closer together, they
**interleave**. The worst answerable query scores 0.2895 while the best gap query scores 0.4836, an
overlap of −0.1941 — an order of magnitude worse than the −0.048 measured on the memory corpus.
Compression was never the mechanism. voyage-4 simply does not order this corpus's answerable
queries above its off-topic ones.

P3 was right for a reason that also turns out to be wrong: I expected the ordering to hold and the
fixed cut to pay for compression. The cut is paying for a genuine ordering failure instead.

### What it does not show

It does not show that the offline generator failed. Confound 1 named exactly this risk — that a
subject list filtered against a corpus about software and retrieval would be less disjoint than for
a generic corpus. With a gap ceiling of 0.4836 against an answerable floor of 0.2895, generator and
embedder are not separable from each other by this run. The bge arm below is what tells them apart:
**same corpus, same query set, same chunker, one variable changed.**

## Second prediction: the bge-large arm (2026-08-20, before measuring)

Operator direction after seeing the above: use bge for docs too. Registered before running it.

**P4. bge-large certifies on this corpus.** Separability 95% lower bound **at or above 0.90**,
point AUC **0.95 to 1.00**. Grounded in two prior measurements rather than hope: the 2026-08-16
study measured the offline arm on this same corpus family at bge-**small** and got AUC 0.9806 with
a lower bound of 0.9496, and the memory tenant under bge-**large** measured 0.9870 four days later.

**P5. The threshold lands near the bge-small precedent, not near voyage-4's.** Between **0.62 and
0.78**, against 0.7050 for bge-small on this corpus and 0.712 for bge-large on memory.

**P6. The overlap shrinks by at least an order of magnitude**, from −0.1941 to no worse than
−0.05, i.e. into the band the memory corpus showed (−0.048).

Falsified by: a lower bound below 0.90, a threshold outside 0.62 to 0.78, or an overlap worse than
−0.05. If P4 fails as well, the offline generator is the common factor across two embedders and the
honest conclusion is that this corpus needs labelled queries, not a different model.

## Result, bge-large arm (2026-08-20)

**Status:** measured. P4, P5 and P6 above are unedited.

Generation `gen_362409d421804b7ab7a50865f4e2f362`, 155 objects, 2,703 chunks, built in **4h17m**
under a 600% cgroup cap. **Same corpus, same `chunk_text`, same query set** (digest
`0a7a27cc445e0a7b…`, byte-identical to the voyage-4 arm's). One variable changed.

| | predicted | measured | |
|---|---|---|---|
| certified | **yes** | **YES** | confirmed |
| separability | 0.95 to 1.00 | **0.9756**, CI [0.9408, 1.0000] | confirmed |
| threshold | 0.62 to 0.78 | **0.637** | confirmed |
| overlap no worse than −0.05 | | **−0.0802** | **falsified** |

```
answerable min/max : 0.6047 / 0.8429
gap        min/max : 0.5074 / 0.6849
false abstain 10.00% of 40    false confirm 10.00% of 40
```

### The A/B, which is the point of holding everything else fixed

| | voyage-4 | bge-large |
|---|---|---|
| certified | **no** | **yes** |
| separability | 0.8419 [0.7540, 0.9297] | **0.9756 [0.9408, 1.0000]** |
| threshold | 0.382 | 0.637 |
| false abstain | 25.00% | **10.00%** |
| false confirm | 27.50% | **10.00%** |
| overlap | −0.1941 | **−0.0802** |

**The offline generator is exonerated.** Confound 1 raised the possibility that a subject list
filtered against a corpus about software and retrieval would be insufficiently disjoint, depressing
AUC for reasons unrelated to the embedder. The same generated set, unchanged, moves from 0.8419 to
0.9756 when only the embedder changes. The generator was not the cause; voyage-4 was.

### P6 is falsified, and by a factor that matters

I predicted the overlap would shrink into the memory corpus's band, no worse than −0.05. It came in
at **−0.0802**: 2.4x better than voyage-4's −0.1941, and still 1.7x worse than the memory corpus's
−0.048. So bge-large orders this corpus well enough to certify, but the classes still interleave
more than they do on memos. Both per-class errors landing on exactly 10.00% (4 of 40 each) is the
visible consequence: the fitted cut has nowhere clean to sit.

This is the fifth corpus in a row where the answerable and unanswerable distributions overlap,
which is the standing observation from `calibrated-thresholds-and-the-overlap` — 4 of 4 at the
time, now 5 of 5 — and the reason to read the overlap next to the AUC rather than instead of it.

### Apparatus, stated honestly

The binding reported `embedder_model = BAAI/bge-large-en-v1.5` at 1024, and the generation
validated at 155 sources and 2,703 chunks, matching the voyage-4 arm exactly.

**The registered control query did NOT return its expected top-1.** "what are the generation states
and how does promotion work" returned `ARTIFACTS.md` at 0.7012, with `GENERATIONS.md` at ranks 2, 4
and 5 (0.6408, 0.6373, 0.6348). Under voyage-4 the same query did return `GENERATIONS.md` first.
Followed up before treating the numbers as final: "how does calibration bind to a generation"
returns `CALIBRATION.md` at 0.7908 top-1, a clean hit. The corpus is loaded and retrieving the
right documents; the control was a weak discriminator, not a failed ingest. Recorded because a
control that misses is worth stating even when the follow-up clears it.

### Operational consequence

`re-call-docs` can now leave development mode: certified calibration at threshold 0.637, bound to
`gen_362409d421804b7ab7a50865f4e2f362`, under the SAME embedder as `memory`. All three tenants the
live client uses are now generation-backed and certified, two of them on bge-large and
`re-call-code-gen` on voyage-code-3.

⚠️ Both error rates sit at exactly the 10% bound. That is a working threshold, not a comfortable
one, and the next corpus delta on this tenant should be watched rather than assumed.
