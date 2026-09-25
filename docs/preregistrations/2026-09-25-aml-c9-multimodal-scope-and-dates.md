# Pre-registration: C9 multimodal scope (MM-1) and dated image items (MM-3)

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Baseline is the served C9 (`C9_routed_specialists_grounded_graph_atomic`)
at `3eb447c4`, the day-zero baseline of the round-two plan. This is a local directional
experiment over public data. It is not an AML hosted evaluation and not a leaderboard score.

## Why

The official AML Multimodal Full scored 31.64 on C8, with visual_reasoning 19.35,
information_extraction 22.13 and direct_recall 23.60. Reading the served code on `origin/master`
on 2026-09-25 found two behaviours no record had named as a cause:

1. **Images reach the reader only on the multimodal route.** In `SearchService.search`
   (`recall_aml/service.py`), the Voyage multimodal leg and `render_preserved` run only when
   `not context_specialist or specialist_route == "multimodal"`. C8 and C9 set
   `context_specialist=True`, and `route_query` (`recall_aml/specialists.py`) returns
   `multimodal` only for a query that carries an image part or matches `_VISUAL_SIGNAL`, ten
   English words (image, photo, picture, screenshot, diagram, chart, figure, visual, screen, ui).
   Any other question is answered from text windows alone: the supplied text of an image message
   (`content_text`, `recall_aml/multimodal.py`), or the placeholder `[image evidence]` when the
   message had no text. The image itself is never returned.
2. **Image items carry no date in their content.** `dated_items` (`recall_aml/window_format.py`)
   returns any item whose content is not a plain string unchanged, so on the multimodal route every
   image-bearing item reaches the reader with its date only in the optional `created_at` field,
   which AML may not render. The same header on text items was worth +53.75 on LoCoMo temporal
   (`docs/preregistrations/2026-09-25-aml-c9-window-format.md`).

🔁 A correction to the round-two plan page written earlier the same day: it said image memories
come back "as the placeholder". That holds only for image messages with no text. A message with
text (AML sends captions with images) is retrievable by that text; what is missing is the image.

## The questions

- **MM-1:** On MemEye questions that C9 routes away from the multimodal route, does returning the
  images of multimodal memories raise MCQ debiased exact match against served C9?
- **MM-3:** Does prefixing each image-bearing item's content with its own date raise debiased exact
  match on MemEye evolutionary-synthesis questions (Y3), without lowering it on all questions?

## Arms

All arms Search one ingested C9 tenant per scenario. Only Search behaviour differs, so ingest,
compile and Add-time embeddings are shared and cannot confound the comparison.

| Arm | Non-multimodal route (the questions MM-1 is about) | Multimodal route |
|---|---|---|
| **B** | served C9: text windows only | served C9: visual leg, fusion, preserved images |
| **B′** | B again, Searched separately (noise floor; Voyage query embeddings are not deterministic) | as B |
| **P** (MM-1 preserve) | same text ranking; image-bearing hits rendered with their original images via `render_preserved`; no visual leg | as B |
| **D** (MM-1 dual) | visual leg run and fused as on the multimodal route, then preserved rendering | as B |
| **P+t, D+t** (MM-3) | P or D with the date part added to every image-bearing item | as B, plus the date part |

Implementation constraints, fixed now:

- MM-1 is a default-off variant setting. When the tenant's visual store returns no hit, P and D
  must be byte-identical to B, so a text-only tenant (every Textual and Coding user) is unchanged.
- MM-3 is a pure render transform: it adds exactly one leading text part `[YYYY-MM-DD HH:MM UTC] `
  to a list-content item that has `created_at`, and changes nothing else. P+t and D+t are computed
  by applying that same production function to the stored P and D Search responses, so the dated
  and undated arms are paired on identical retrieval.

## What I predict

Written before any of the numbers below exist. My past effect predictions ran two to four times
too high (memory `i-over-predict-effect-magnitudes`), so the bands are deliberately low.

**Stage 0, route census (no model spend).** Share of questions that served C9 routes to
`multimodal`, over the question text only:

| Dataset | n | Predicted share |
|---|---:|---|
| MemEye MCQ, all 8 public scenarios | 371 | 0.20 to 0.50 |
| MemLens (question set shared by all four lengths) | 789 | 0.10 to 0.40 |
| MobileMem-Omni, English questions | as released | 0.15 to 0.45 |
| MobileMem-Omni, Chinese questions, if released separately | as released | 0.00 to 0.05 |

**Stage 1, retrieval and delivery on MemEye, non-multimodal stratum** (questions B routes away
from `multimodal`):

| Metric | B | P | D |
|---|---|---|---|
| Clue image delivered inside the answer-packing prefix | **exactly 0.000** | 0.75 to 0.95 | 0.75 to 0.95 |
| Any-clue Recall@10 by session, minus B | 0 | within B′ noise | −0.05 to +0.03 |
| Median items admitted to the answer prefix | 100 | 60 to 100 | 60 to 100 |

**Stage 2, answers on MemEye** (debiased exact match, gpt-4o-mini, frozen MemEye MCQ prompt, four
option rotations per question):

| Contrast | Stratum | Predicted |
|---|---|---|
| D − B | non-multimodal | **+0.03**, band +0.01 to +0.06 |
| P − B | non-multimodal | +0.01, band −0.03 to +0.04 |
| (D − B on X3∪X4) − (D − B on X1∪X2) | non-multimodal | positive: pixel and instance questions gain more |
| B′ − B | non-multimodal | within ±0.03 |
| D+t − D | Y3, all routes | **+0.03**, band 0.00 to +0.08 |
| D+t − D | all questions | −0.01 to +0.02 |

## What would falsify this

- **MM-1:** D − B below +0.01 on the non-multimodal stratum, or its paired 95% bootstrap CI lower
  bound below −0.03. If P ≥ D, the visual leg adds nothing beyond attaching images to what text
  retrieval already found.
- **Mechanism:** the X3∪X4 subgroup gaining no more than X1∪X2 falsifies "native images help
  because captions lose instance and pixel detail", even if the total is positive.
- **MM-3:** D+t − D on Y3 at or below 0.00, or below −0.01 on all questions.
- **Stage 0:** a MemEye share above 0.80 would make MM-1 nearly moot on MemEye: stop after Stage 1
  and move the test to MemLens or MobileMem.

## Decision rule

- Recommend **MM-1** to the user if D − B ≥ +0.03 on the non-multimodal stratum, its CI lower bound
  is above −0.02, |B′ − B| is smaller than D − B, and every apparatus check passes. Prefer P over D
  when P − D ≥ −0.01, since P adds no Search-time embedding.
- Recommend **MM-3** if D+t − D ≥ +0.02 on Y3 and ≥ −0.01 on all questions. It is a render-only
  change of the kind already adopted for text (H), so the bar is non-inferiority plus a positive
  point estimate where it should act.
- Either change moves the C9 baseline, so it needs an explicit user decision before it serves
  (memory `official-textual-coding-config-graph-on`). Neither may serve during an AML job.

## How it will be measured

**Stage 0.** A census script over the pinned public question files, applying the served
`route_query` to each question string. MemEye: `MemEyeBench/MemEye` on Hugging Face, the eight
`data/dialog/<scenario>.json` MCQ files, revision pinned by SHA at download. MemLens:
`xiyuRenBill/MEMLENS`, `dataset_32k.json`, questions only. MobileMem: `zjunlp/MobileMem`, the Omni
QA files. A dataset that cannot be downloaded is reported as unavailable, never imputed.

**Stage 1.** VPS3 testbench, never the official C9 on VPS2. One C9 service built from this branch,
against a fresh table. Per scenario:

1. Ingest every round as the frozen MemEye harness does (`build_add_requests` in
   `scripts/aml_multimodal_memeye.py`: one Add per round, user text plus images, session date at
   12:00 UTC).
2. Search each question under B, B′, P and D (four option rotations each, `top_k` 100, the
   question as the query and the rotation's options as `options`), store every response.
3. Delete the tenant and verify an empty Search.

Metrics: `retrieval_metrics` from the frozen harness; "clue image delivered" means at least one
returned item whose `session_id` is a clue round carries an `image_url` part among the items that
`pack_answer_content` admits under the 117,760-token budget.

**Stage 2.** The frozen MemEye MCQ answer path (`pack_answer_content`, `answer_rotation`,
`extract_choice`): `openai/gpt-4o-mini`, temperature 0, 16 output tokens, image detail `low`, the
pinned `sys_prompt_mcq.txt`. Answered only where arms can differ:

- B, B′, P and D on the non-multimodal stratum.
- D and D+t on every question (for MM-3).

Paired bootstrap over questions, 10,000 resamples, seed 20260925.

**Spend.** The OpenRouter key must not be the one the official C9 uses, or the balance must be
checked first (memory `2026-09-25-openrouter-credit-exhausted-c9-shares-key`). Cap: USD 60 on the
harness's conservative ledger (`max(reported, prompt × $1/M + completion × $4/M)`), about USD 10 at
gpt-4o-mini's actual rates. Voyage embedding spend is logged but not capped. Stage 2 starts only
after Stage 1 passes its apparatus checks and the user approves the spend.

## Apparatus checks, fixed now

1. **Text-only tenant unchanged.** A unit test shows P, D and MM-3 return byte-identical Search
   output to B for a tenant with no image rows. It must first be seen red against a deliberate
   mutation that removes the empty-visual-store guard.
2. **B delivers no image on the non-multimodal stratum.** B's clue-image-delivered rate there is
   exactly 0.000 and above 0 on the multimodal stratum. Anything else means the harness is not
   measuring the route gate.
3. **The visual leg ran where it should.** From the response's `specialist_route` and a logged
   visual-leg flag: in B the leg runs on exactly the multimodal-routed questions; in D it runs on
   every question.
4. **MM-3 is render-only.** On every stored P and D response, the dated copy has the same item ids,
   order, scores and parts, plus exactly one leading date part on each list-content item with
   `created_at`.
5. **Noise is measured, not assumed.** B′ against B: share of identical ranked session lists and
   the answer-score difference.
6. **Answers are real.** A valid option letter on at least 98% of rotations in every arm; empty or
   invalid answers are counted, not dropped.
7. **Cleanup.** Every tenant deleted, with an empty Search after deletion.

## What I already know

- MemEye Brand v2 (29 questions, one scenario, 2026-09-20): caption 0.4655, preserve 0.4310,
  dual 0.4741 debiased exact match; any-clue Recall@10 0.9655 in all three. Verdict `NO_GAIN`
  (`docs/preregistrations/2026-09-20-aml-multimodal-memeye-brand-v2.md`). Those arms had no router,
  so they measured preservation and dual retrieval in general, not the route gate this record
  targets. Preserving images LOWERED the score against captions by 0.034 there; that is the
  strongest reason the P and D bands above are low.
- MemEye has 371 MCQ questions across 8 scenarios (counted 2026-09-25 from the files, nothing
  else computed): X1 48, X2 48, X3 142, X4 133; Y1 119, Y2 204, Y3 48. The MemEye authors report
  that captions lose instance and pixel detail and that retrieval can rank stale visual evidence
  above updates.
- The official Multimodal Full had a median 33% of each user's rows hidden on the multimodal route
  (defect fixed by #719) and no timestamps on preserved items (fixed by #725). Neither fix touches
  the route gate or dated content for images.
- The date header on text items: LoCoMo temporal +53.75, Coding MRR unchanged.

## Confounds I can name now

- **MemEye is one of AML's own Multimodal sources** (the frontend lists MemEye Open and MemEye
  MCQ). The public set may overlap the evaluation set, so a gain here is not held out from AML.
  Neither change is tuned on MemEye: both are generic mechanisms whose settings are fixed above.
- **Reader mismatch.** AML's multimodal answer model and image detail are unknown. Stage 2 uses
  gpt-4o-mini at `low` detail (512 px), which can erase pixel-level (X4) evidence and would bias
  MM-1 toward null.
- **Token budget.** Images displace text in the answer prefix (1,000 tokens each by the harness
  estimate). A null or loss can come from displaced text rather than from useless images;
  Stage 1's admitted-items metric exists to tell those apart.
- **Voyage query nondeterminism.** 31% of ranked lists differ between identical runs (memory
  `voyage-query-embeddings-are-not-deterministic`); B′ is the control.
- **Query and option wording.** Stage 0 measures route share on public wording; AML's own
  questions may carry visual words more or less often.
- **MM-3 dates on MemEye are per session at 12:00 UTC**, so the header can order sessions but
  never times within a day.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

### Amendment 1, 2026-09-25, before any measurement

Written after the record was committed (`e785b010`) and before any stage ran. The predictions,
falsifiers and decision rule above are unchanged.

1. **Host.** Everything runs on VPS3. The official AML Textual Full is running on VPS2, and the
   user's instruction is that the VPS2 API is never touched or disconnected. No stage queries,
   loads or restarts anything on VPS2.
2. **Model.** No gpt-4o-mini call anywhere in this experiment (user instruction 2026-09-25:
   gpt-4o-mini only after the numbers confirm an improvement and all experiments are finished).
   - Stage 2 reader: `deepseek/deepseek-v4.1-flash` on OpenRouter, which lists `text` and
     `image` input (checked 2026-09-25 on `/api/v1/models`; `deepseek/deepseek-v4-flash` is
     text-only and could not measure MM-1). Temperature 0, the frozen MemEye MCQ prompt, image
     parts as the harness sends them. Output capped at 16 tokens with reasoning disabled; a
     runaway generation is an apparatus failure, not an answer.
   - Stage 1 ingest: C9's Add-time compiler is pointed at the same DeepSeek model through a
     default-off environment override on VPS3 only. The ingest is shared by every arm, so this
     cannot favour one arm, but it means Stage 1 does not reproduce served C9's compiled records.
   - The predictions were written for a gpt-4o-mini reader. They stand as written; a reader
     mismatch is a named confound, not a reason to re-predict.
3. **Credit floor.** VPS3 carries the same OpenRouter key as the official C9 (key label
   `sk-or-v1-90a...7b8`, no limit; balance USD 52.47 at 14:15 UTC). Every paid stage reads the
   account balance first and refuses to start, and stops mid-run, if the balance is below
   USD 40, so this experiment can never be what drains the official run. Paid stages also need an
   explicit user go.
4. **MobileMem file.** The census uses `omni/filtered_questions.jsonl` at revision
   `14c086312c61b0e13cf588afd2b67a5b1664b33a` (2,308 questions: 8 English and 8 Chinese users)
   as the primary MobileMem set, and reports the unfiltered `omni/questions.jsonl` (9,308) beside
   it. The Chinese row of the prediction table is the `language == "Chinese"` users. MemLens is
   `dataset_32k.parquet` at `afa101a1907cc37db40b50d649547964387b96b7`, question column only
   (789). MemEye is the eight MCQ files at `main` as downloaded 2026-09-25; the census records
   each file's SHA-256.

## Result, Stage 0 route census (2026-09-25)

**Status:** Stage 0 measured; Stages 1 and 2 not yet run.

Command, on VPS3 at `c871f594` (the served `route_query` is unchanged from `3eb447c4`):
`python scripts/aml_mm_route_census.py --data-dir data --output census-stage0.json`.
Output: `results/aml-mm-scope-dates/census-stage0.json`, with each source file's SHA-256.

| Dataset | n | Routed to `multimodal` | Predicted | Gap |
|---|---:|---:|---|---|
| MemEye MCQ, 8 scenarios | 371 | **0.216** (80) | 0.20 to 0.50 | inside, at the floor |
| MemLens 32K | 789 | **0.048** (38) | 0.10 to 0.40 | **below the band** |
| MobileMem-Omni filtered, English | 1,171 | **0.059** (69) | 0.15 to 0.45 | **below the band** |
| MobileMem-Omni filtered, Chinese | 1,137 | **0.000** (0) | 0.00 to 0.05 | inside |
| MobileMem-Omni unfiltered, all | 9,308 | 0.019 (175) | reported only | |

Over the three primary sets, 187 of 4,268 questions (4.4%) reach the route on which C9 returns
images. The other 95.6% are answered from text alone, whatever the memory holds.

Breakdowns, same file:
- MemEye by visual granularity: X1 0.104, X2 0.083, X3 0.225, X4 0.293. By scenario, from 0.000
  (Card Playlog, Outdoor Navigation) to 0.765 (Personal Health Dashboard). Brand, the only scenario
  the 2026-09-20 pilot ran, is 0.103.
- MobileMem English by type: visual_reasoning 0.245 (249), temporal_reasoning 0.021,
  knowledge_update 0.000, single_hop 0.004. Even the questions labelled visual reasoning mostly
  miss the image route.
- MemLens by type: information_extraction 0.081, temporal_reasoning 0.072, multi_session 0.028,
  knowledge_update 0.000, answer_refusal 0.000.

**What the gap means.** I over-predicted the share on two of four sets by at least half again.
My error was assuming a question about an image would name the medium ("photo", "screenshot").
Mostly it names the content instead ("what was on the receipt", "which route did I take"). That
is the direction that makes MM-1 matter more, not less. The Stage 0 stop condition (MemEye share
above 0.80) is not met, so Stage 1 proceeds, on MemEye's 291 non-multimodal questions.

This is a census of public wording, not of AML's own questions, and it says nothing yet about
whether returning the images helps the reader. Stages 1 and 2 measure that.

### Amendment 2, 2026-09-25, during Stage 1 and before any Stage 2 answer is scored

Written while Stage 1 was running (2 of 8 scenarios complete) and before Stage 2. Predictions,
falsifiers and the decision rule are unchanged.

1. **The Stage 2 reader is pinned to one provider**, `DeepInfra`, with fallbacks off
   (`READER_PROVIDER` in `scripts/aml_mm_scope_stage2.py`, `9bb0e276`). Two unpinned probe calls
   were served by DeepInfra and by Sail Research; a provider mix has emptied answers in this
   repository before (memory `openrouter-qwen3-provider-mix-empties-answers`).
2. **The reader sees images (apparatus check, measured).** One MemEye Brand image with the question
   "Name the brand or logo shown in this image": with the image, "Burger King, with main colours
   blue, red, yellow, and white", 978 prompt tokens; without it, "Coca-Cola, red and white", 24.
3. **B′ cannot measure retrieval noise.** The three arm processes share one on-disk embedding cache
   (`RECALL_AML_EMBED_CACHE_PATH`) and `CachedEmbedder.embed_query` caches query vectors, so every
   arm, B′ included, searches with the same vector for the same question. On the first 77 questions
   B and B′ returned identical ranked lists on every rotation (1.0). This removes Voyage query
   nondeterminism as a confound between arms; B′ now measures only reader noise in Stage 2.
4. **DeepSeek compiles fall back more often than gpt-4o-mini's.** Brand: 26 of 42 text-only Adds
   fell back to raw windows (journal: 33 `ValueError`, 9 `ValidationError`, 5 `JSONDecodeError`
   over both completed scenarios). The ingest is shared by every arm, so this cannot favour one,
   but compiled records and graph promotions are thinner than served C9's.
5. **Measured Stage 2 cost.** Six probe answers: the first rotation of a question cost USD 0.0058
   (about 19k prompt tokens with its images), later rotations USD 0.0001 to 0.0002 through prompt
   caching. Estimate for the full plan (about 1,600 question-arm pairs): USD 6 to 11. At a balance
   of USD 47.56 and the USD 40 floor, the floor can stop Stage 2 before it finishes; the run then
   stops and resumes after a top-up, it does not continue below the floor.

## Result, Stage 1 retrieval and delivery (2026-09-25)

**Status:** Stage 1 measured; Stage 2 (answers) not yet run.

Run `s1-20260925T144704Z` on VPS3 at `7442c02b` (arms B, B′, P, D over one ingest; resumed once with
four scenarios in parallel at `2de68028`, which only reordered work: Adds replay from fixed request
ids). 5,936 Searches (371 questions × 4 rotations × 4 arms), every scenario deleted and verified
empty. Report: `results/aml-mm-scope-dates/stage1-report.json`
(`scripts/aml_mm_scope_report.py`).

**Apparatus checks:** all pass. B delivers no image off the multimodal route (0.000) and does on it
(1.000); the visual leg ran on exactly B's multimodal-routed questions and on every D question;
MM-3's dated copies change nothing but the one added part, on every P and D row; B and B′ returned
identical ranked lists on every rotation (1.0, the shared query cache of amendment 2); no question
missing.

**Non-multimodal stratum, 291 questions** (mean over rotations, then over questions):

| Metric | B | B′ | P | D | Predicted |
|---|---:|---:|---:|---:|---|
| Clue image delivered in the answer prefix | **0.000** | 0.000 | 0.918 | 0.993 | B exactly 0; P, D 0.75 to 0.95 |
| Any-clue Recall@10 by session | 0.495 | 0.495 | 0.495 | **0.818** | D − B −0.05 to +0.03 |
| Any-clue Recall@100 by session | 0.955 | 0.955 | 0.955 | 1.000 | |
| Median items admitted to the prefix | 100 | 100 | 100 | 100 | 60 to 100 |
| Image items admitted, mean | 0.0 | 0.0 | 30.1 | 37.1 | |

**D − B on Recall@10: +0.323**, against a predicted −0.05 to +0.03. D is better on 97 questions and
worse on 3. On the 80 multimodal-routed questions all four arms are identical by construction
(Recall@10 0.950 each).

By scenario (non-multimodal stratum, Recall@10, B → D): Outdoor Navigation 0.000 → 0.964 (n 28),
Social Chat 0.061 → 0.455 (33), Cartoon 0.378 → 0.892 (74), Personal Health 0.583 → 0.917 (12),
Multi-Scene 0.700 → 0.867 (30), Brand 0.846 → 0.962 (26), Home Renovation 0.775 → 0.850 (40),
Card Playlog 0.688 → 0.708 (48). By visual granularity: X1 +0.465 (43), X2 +0.614 (44), X3 +0.236
(110), X4 +0.223 (94).

**What the gap means.** I predicted the visual leg would add almost nothing to retrieval, because
MemEye Brand had shown any-clue Recall@10 of 0.9655 for every arm. Brand is the scenario where text
retrieval is already strongest (0.846 here); in scenarios whose image rounds carry little text
(route screenshots, chat screenshots, cartoon frames) the served text leg rarely ranks the clue round
at all, and the image leg finds it. The whole prediction rested on the one scenario that least
needed the leg. The granularity split runs opposite to the answer-level mechanism prediction
(X3∪X4 gaining more): for RETRIEVAL the gain is largest on scene- and region-level questions, which
are the ones a whole-image embedding matches best. Whether the answers follow, and on which axis, is
Stage 2's question.

**Stage 2 is blocked by the credit floor, not by a gate.** Stage 1 passes every condition for
Stage 2. Both paid runs queued behind it (this Stage 2 and the T-1 LoCoMo answers) refused to start
at an OpenRouter balance of USD 34.88, then 33.58, below the USD 40 floor of amendment 1, because
the official Textual Full is drawing on the same key. Nothing was spent. Stage 2 starts when the
balance is back above the floor.
