# Pre-registration: does an offline-generated query set certify `re-call-docs` under voyage-4?

**Date:** 2026-08-20   **Status:** predicted, not yet measured

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
