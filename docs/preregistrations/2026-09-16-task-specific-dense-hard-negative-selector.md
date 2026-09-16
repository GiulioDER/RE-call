# Task specific selector with real dense hard negatives

Status: predicted, not yet measured.

Registered 2026-09-16 before collecting candidate pools, constructing training pairs, training a
model, or scoring any validation or internal test row.

## Question

Can a small cross encoder trained on RE-call's automatic extractive gold choose a better first
evidence chunk than Context 4 dense rank one when its negatives come from the same real production
dense top 20 distribution?

An earlier MiniLM fine tune on LOCOMO did not improve retrieval. Its recorded reentry condition was
training with negatives mined from real dense or fused retrieval pools rather than BM25. This
experiment satisfies that condition and changes no other model family.

This is a consumed development experiment. Even a pass cannot authorize serving. It can authorize
one fresh, naturally worded, source disjoint validation.

## Frozen population and split

Use only answerable rows from
`docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json`, SHA256
`66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`.

Use the split protocol and seed from
`docs/preregistrations/2026-09-16-extractive-gold-shape-source-split-audit.md`. The audit result
artifact SHA256 is `842e9bcbd279449222f773c20d27d6556cb3331ea388b6b51d3ca3788c055f18`.

Before collection, exclude every answerable row whose `answer_span_sha256` appears more than once
anywhere in the 250 answerable rows. This exclusion is global and uses no retrieval result. The
expected retained population is 186 train, 33 validation, and 29 internal test rows. Refuse if the
counts differ.

The matched absent identifier controls are excluded completely. This experiment does not train,
tune, or evaluate a null class and makes no answerability claim.

## Frozen serving lineage and candidates

Collect the original dense top 20 through a pinned production benchmark MCP process with graph
expansion off, reranking off, and source conditioning off. The required lineage is:

1. generation `gen_e3702629f8844cf5bd9293da36a7c7ab`
2. calibration `cal_308ad1eeecb74075a7668f878f781b51`
3. pipeline fingerprint
   `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`
4. corpus fingerprint
   `8307f711b21fff830d2ce2c222011a91dfdf711597561f8bb021ab93841babae`
5. embedding profile `voyage-context-4-v1`
6. retrieval profile `fast`

Require exactly 20 unique dense candidates per row in ascending dense rank. Save query text,
candidate chunk ID, source, chunk text, dense rank, and dense cosine only in a private artifact.
Join gold source and exact span labels locally after retrieval. Record hashes of every private
input and output in the public aggregate.

Reconstruct the exact bearing positive chunk from the frozen local source and recorded ordinal.
Require the complete frozen span in that chunk. A positive need not have been retrieved in the top
20. For each train query, use that exact bearing chunk once as label 1 and the first four dense
candidates that do not contain the complete answer span as label 0. A same source but nonbearing
chunk is a valid negative because the target is exact evidence selection. Refuse if any query has
fewer than four negatives or if a positive text is also labelled negative for that query.

## Frozen model and training

Use `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`, the same pinned base used by RE-call's local
reranker. Train `BinaryCrossEntropyLoss` with these fixed settings:

1. seed `20260916`
2. two epochs
3. learning rate `1e-5`
4. batch size 16
5. warmup ratio 0.1
6. maximum sequence length 512
7. AdamW defaults supplied by Sentence Transformers
8. no null examples, score mixing, threshold tuning, class weighting, negative resampling, or
   alternate checkpoint

Training and all model scoring run on VPS2 under the shared model lock, with an 8 GB memory cap,
zero swap, a 250 percent CPU quota, four Torch threads, and lowered priority. Record exact library
versions, elapsed time, model weight drift, and a SHA256 digest over the saved model files. The
model and row level artifacts remain private.

## Frozen scoring

Score the base model and the trained model on every query and its unchanged dense top 20. Rank by
descending cross encoder score, breaking an exact score tie by original dense rank. Do not mix the
cross encoder score with dense cosine.

For dense order, base model order, and trained model order, report gold source and exact span reach
at ranks 1, 3, 5, 10, and 20. Report rank one gains, losses, and changed selection precision for
gold source and exact span relative to dense rank one. Report membership preservation and the
number of pools reordered by training relative to the base model.

The base model is diagnostic. The decision comparison is trained model order versus original dense
order.

## Validation gate and sealed internal test

Score validation first. Do not score or summarize internal test unless every validation condition
passes:

1. all 33 memberships are preserved
2. trained exact rank one exceeds dense exact rank one by at least two rows
3. trained gold rank one exceeds dense gold rank one by at least two rows
4. exact rank one losses are at most one row
5. gold rank one losses are at most one row
6. changed rank one precision is at least 0.50 for both exact span and gold source
7. trained weights differ from the pinned base and at least one validation pool order differs from
   base model order

If the validation gate fails, return `STOP_TASK_SPECIFIC_SELECTOR_VALIDATION` and leave internal
test unscored.

If validation passes, score internal test once. Return
`PROCEED_FRESH_NATURAL_SELECTOR_VALIDATION` only if every condition below passes:

1. all 29 memberships are preserved
2. trained exact rank one exceeds dense exact rank one by at least two rows
3. trained gold rank one exceeds dense gold rank one by at least two rows
4. exact rank one losses are at most one row
5. gold rank one losses are at most one row
6. changed rank one precision is at least 0.50 for both exact span and gold source

Otherwise return `STOP_TASK_SPECIFIC_SELECTOR_INTERNAL_TEST`. Do not tune any frozen choice on the
observed rows.

## Prediction

I predict validation will pass with at least two net exact rank one gains and two net gold rank one
gains. The real uncertainty is transfer to internal test. I predict internal test will produce at
least two exact and two gold gains with at most one loss on each measure, because the supervision
matches the exact evidence target and the negatives now come from the deployment candidate
distribution. The prediction is intentionally stronger than a generic reranking improvement: the
experiment is useful only if it improves the first result that the user sees.

## Pre measurement execution amendment

Recorded before any model load, training step, or validation score. VPS2 has the pinned Torch,
Transformers, and Sentence Transformers stack but does not have the optional `datasets` package.
To avoid modifying the production environment, training will use the installed
`CrossEncoder.old_fit` data loader adapter. Its default one label loss is
`torch.nn.BCEWithLogitsLoss`, which is the binary cross entropy with logits objective frozen above,
and its optimizer is AdamW. The seed, epochs, learning rate, batch size, warmup ratio, maximum
length, examples, and all gates remain unchanged.
