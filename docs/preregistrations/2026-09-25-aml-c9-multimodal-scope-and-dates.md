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
