# AML multimodal MemEye Brand admission pilot

Date: 2026-09-20

## Question

Does preserving original image evidence improve final answer quality over supplied-text retrieval,
and does a dedicated multimodal embedding leg improve retrieval and answer quality beyond image
preservation alone?

This is a local directional experiment over public MemEye data. It is not an AML hosted evaluation,
does not use AML private data, and cannot be reported as an AML leaderboard score.

## Frozen sources

The dataset is `MemEyeBench/MemEye` at Hugging Face revision
`f139f89d08bf0f66986153b5ff010cac24ba383d`. The measured file is
`data/dialog/Brand_Memory_Test.json`, SHA256
`4aed4963277f742fe06f68b0910d7c1cfec5d0ed240904d291e02887712fbf78`.
It contains 42 sessions, 72 dialogue rounds, 30 distinct referenced images, and 29 MCQ questions.

The reference evaluation behavior is pinned to `MinghoKwok/MemEye` commit
`0358e70d714980980dfd8c87903384db474f0b16`. Its four option rotations and mean exact-match
aggregation are part of this protocol. The exact pinned MCQ system prompt will be used without
editing.

The public source images are loaded only from the pinned dataset revision. Their paths must resolve
under `data/image/Brand_Memory_Test/`. The runner must reject traversal, missing images, unsupported
formats, a decoded image larger than 10 MiB, or a dataset JSON hash mismatch.

## Frozen arms

1. `MM0_caption` is the supplied-text control. It stores user and assistant text, does not retain
   image bytes, and returns text evidence.
2. `MM1_preserve` uses the same text retrieval path, retains each original image, and returns exact
   ordered text and image evidence.
3. `MM2_dual` uses the same preserved evidence and adds `voyage-multimodal-3.5` document and query
   embeddings fused with the text ranking through the already frozen reciprocal-rank fusion.

Despite its historical name, `MM0_caption` receives no generated caption in this experiment. The
dataset `image_caption` field is not sent to any arm because it is an annotation and would weaken
the intended visual-necessity test. No arm receives question labels, answers, clue round IDs, axis
labels, or option correctness during Add or Search.

The Voyage based arms are an Industry configuration under the AML rules visible on 2026-09-20.
They are not represented as Academic compliant. The answer model is downstream evaluation only and
is not part of Add or Search.

## Frozen ingestion

Each dataset dialogue round is one Add request. Its dataset round ID is used as `session_id`, so a
returned `session_id` can be compared mechanically with annotated clue rounds. The request ID is a
deterministic digest of dataset revision, arm, user ID, and round ID.

The user message contains its text followed by every `input_image` in listed order. The assistant
message follows as text. A round without an image remains text only. The session date is preserved
as a UTC timestamp. All 72 rounds are added in source order under one fresh user ID per arm. Every
arm uses a fresh isolated table and tenant namespace.

Add must return HTTP 200 and the exact request, user, and session identifiers. The runner repeats
one deterministic Add request per arm and requires an identical idempotent response before Search.

## Frozen Search and Answer protocol

For each of the 29 questions, the runner executes all four upstream option rotations. Search gets
the original question as `query`, the current rotation's four option strings as `options`, and
`top_k=100`. The response must remain a ranked prefix, contain at most 100 items, and remain within
the AML 30 MiB decoded-media response limit.

The fixed local Answer proxy is `openai/gpt-4o-mini` through OpenRouter, temperature zero, with the
exact pinned MemEye MCQ system prompt. Each answer request contains the current rotated question and
options plus the Search evidence in returned order. Original images are passed as images, not as
Base64 text. The request uses the longest ranked prefix that fits the AML 117,760-token Answer input
budget. It may not skip an earlier item to admit a later one. Any budget estimate, truncation point,
provider usage, retry, and failure is recorded.

Answer output is reduced with the pinned MemEye choice extractor. A rotation scores one only when
the extracted letter equals that rotation's answer key. Per-question debiased exact match is the
mean over four rotations. The primary answer metric is mean debiased exact match over 29 questions.
The runner also reports strict question accuracy, where all four rotations must be correct, and
choice-position frequencies.

Provider retries are limited to three attempts for 408, 409, 425, 429, 500, 502, 503, 504, and 524.
Every retry must reuse the identical payload. A nonretryable response or an exhausted retry records
the arm as incomplete. No failed arm is silently resumed or scored.

## Frozen retrieval measurements

For each Search result, returned `session_id` is compared with the question's annotated clue round
IDs. Metrics are computed per rotation and then averaged per question:

1. Any-clue Recall at 5, 10, and 100.
2. Complete-clue Recall at 5, 10, and 100.
3. Mean reciprocal rank of the first clue.
4. Mean fraction of clue rounds retrieved at 5, 10, and 100.
5. The same metrics by MemEye X visual-granularity label and Y reasoning-depth label.

Because some questions annotate more clues than fit at a small cutoff, complete-clue Recall is
reported but never interpreted without the mean clue fraction and clue count distribution.

## Frozen operational measurements

The run records Add and Search latency, Search response bytes, decoded media bytes, returned item
count, answer prompt tokens, answer completion tokens, provider usage, persisted media bytes,
contract failures, retries, fallbacks, and residual rows after cleanup. Latency is descriptive and
is not an admission gate on this small run.

Artifacts include the resolved configuration, pinned identities, dataset and prompt hashes,
per-request privacy-safe trace, per-question scores, aggregate metrics, selector verdict,
independent mechanical audit, and `SHA256SUMS`. Questions, answers, image bytes, Base64 values,
credentials, and complete model prompts must not be copied into repository artifacts.

## Frozen predictions

1. `MM1_preserve` will exceed `MM0_caption` mean debiased exact match by at least 0.05 because the
   Brand questions are validated for visual necessity and MM0 cannot return the source images.
2. `MM2_dual` will exceed `MM1_preserve` any-clue Recall at 10 by at least 0.05 because visual
   document embeddings should bridge questions about visible traits that are absent from dialogue
   text.
3. `MM2_dual` will exceed `MM1_preserve` mean debiased exact match by at least 0.02. This is the
   hardest prediction because better clue ranking may not change the fixed Answer proxy.
4. Every arm will preserve Recall at 100 relative to its own corpus size, but MM1 and MM2 may return
   fewer than 72 image-bearing items when the decoded-media response cap binds.
5. MM2 will have higher Add and Search latency than MM1.

## Frozen admission selector

The selector emits one of `ADMIT_FULL_MEMEYE`, `RETRIEVAL_ONLY`, `PRESERVATION_ONLY`, `NO_GAIN`, or
`INVALID`.

`INVALID` applies if an identity or hash differs, an arm is incomplete, isolation or cleanup fails,
the Answer configuration differs across arms, gold data reaches Add or Search, response ordering is
altered after Search, or a required artifact is missing.

`ADMIT_FULL_MEMEYE` requires all of the following:

1. MM1 mean debiased exact match exceeds MM0 by at least 0.05.
2. MM2 any-clue Recall at 10 exceeds MM1 by at least 0.05.
3. MM2 mean debiased exact match exceeds MM1 by at least 0.02.
4. MM2 any-clue Recall at 100 is not lower than MM1.
5. MM2 causes no contract, isolation, response-budget, or cleanup regression.

`RETRIEVAL_ONLY` applies when conditions 2, 4, and 5 pass but condition 3 fails.
`PRESERVATION_ONLY` applies when conditions 1 and 5 pass but condition 2 fails.
`NO_GAIN` applies when the run is valid and none of the preceding verdicts applies.

Only `ADMIT_FULL_MEMEYE` authorizes a new preregistration for all eight public MemEye MCQ scenarios.
No verdict authorizes an AML hosted Full run, an Open-answer run, or a product claim.

## Frozen safety and stopping rules

The experiment has a USD 10 provider cost ceiling and a four-hour wall-clock ceiling. It stops
before the next provider call when either ceiling is reached. It also stops on a new OOM event,
available VPS2 memory below 8 GB, load above 20 on two consecutive observations, identity drift, a
real concurrent embedding or indexing worker, cross-tenant evidence, or residual data after an arm's
cleanup.

Every arm uses fresh immutable result paths. Partial and failed artifacts are preserved. Apparatus
defects may be repaired only with a recorded red then green proof, an appended amendment below the
marker, new frozen commits, and a new result path. Scientific gates, predictions, thresholds,
dataset membership, prompts, and scoring rules may not be changed after measurement begins.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had been run when this record was committed.
