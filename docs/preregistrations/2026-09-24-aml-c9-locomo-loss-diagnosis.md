# Pre-registration: where does C9 lose LoCoMo answers when the evidence is already returned?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

With C9 serving Search at top_k 100 and AML's own LoCoMo-Refined Answer and accuracy prompts
applied to what it returns, what share of the WRONG answers falls in each loss bucket below, and
which bucket is largest among those where the evidence was present?

This is a diagnosis, not a feature test. Its only purpose is to choose which Add-time LLM
candidate, if any, is worth building for the second AML Textual Full run:
1. grounded supersession links (currentness),
2. standing rule and preference records,
3. temporal anchoring of relative dates.

Candidate 2 cannot be decided on LoCoMo, which holds few standing rules.

## Why it is being asked

On 2026-09-23, C9's turn hit@100 on these 1,535 questions was 99.87%, so the Answer model almost
always has the evidence. Points are lost in reading it, and nothing in this repository measures
how. The ATM precedent (`atm-over-abstention-is-the-largest-mechanism`) is that the largest
recoverable loss was one nobody predicted: over-abstention with the evidence in hand, +5.81 QS.

## What I predict

Accuracy is the AML accuracy prompt's CORRECT rate over all 1,535 questions. Shares are of the
WRONG answers.

| quantity | prediction |
| --- | --- |
| accuracy, all questions | 68% to 80% |
| category with the lowest accuracy | 3 |
| RETRIEVAL_MISS share of wrong | 0% to 3% |
| ABSTAINED share of wrong | 5% to 20% |
| TEMPORAL share of wrong | 20% to 40%, the largest single bucket |
| STALE share of wrong | 1% to 8% |
| LIST share of wrong | 8% to 20% |
| DISTRACTOR share of wrong | 8% to 20% |
| INFERENCE share of wrong | 8% to 25% |
| JUDGE share of wrong | 3% to 12% |
| GOLD_ISSUE share of wrong | 5% to 15% |
| accuracy, first gold item at rank 1 to 10, minus rank 11 to 100 | +5 to +20 points |

Reasoning. With 100 windows of 160 words, the Answer model reads most of each conversation, so
this sits close to a full-context reader. Published full-context gpt-4o-mini results on LoCoMo are
in the low 70s under a similar binary judge. LoCoMo's temporal category is 320 of 1,535 questions,
and the AML judge is strict on time granularity and forbids converting relative to absolute
dates, while the Answer prompt asks for exactly that conversion. So I expect temporal errors to
dominate. LoCoMo holds few changing facts, so STALE should be small. Applying my own correction
([[i-over-predict-effect-magnitudes]]) to the rank effect gives the wide low band.

## Decision rule, fixed now

Applied in order, and the first that fires decides:

1. If JUDGE plus GOLD_ISSUE is at least 30% of wrong answers, LoCoMo is too noisy to steer by.
   Build nothing yet, and repeat the diagnosis on a LongMemEval sample.
2. If ABSTAINED is at least 15% of wrong answers, the next work is on abstention with evidence
   present, which is an ordering and evidence problem, not an Add-time LLM one.
3. If STALE is at least 10% of wrong answers, build candidate 1.
4. If TEMPORAL is the largest bucket with evidence present, audit its members first. Split them
   into date-arithmetic errors, which anchoring can fix, and granularity or relative-form
   mismatches, which anchoring could make worse. Build candidate 3 only if arithmetic errors are
   the larger part.
5. Otherwise, no Add-time candidate is supported by LoCoMo.

Nothing here changes the first official Full run, which stays on C9 as decided.

## What would falsify this

- Accuracy outside 68% to 80%.
- TEMPORAL not the largest bucket, or outside 20% to 40%.
- Any other bucket share outside its band.
- The rank effect outside +5 to +20 points.

## How it will be measured

- **Script:** `scripts/aml_locomo_loss_diagnosis.py`, committed with this record. It imports the
  corpus builder and the evidence matcher unchanged from `scripts/aml_locomo_route_compare.py`,
  the 2026-09-23 harness.
- **Collect, on VPS3:**
  - The service is built in process at `337f2537`, the C9 commit served officially. Variant
    `C9_routed_specialists_grounded_graph_atomic`, graph on, atomic stage active.
  - A fresh database of its own, and a copy of the 2026-09-23 embedding cache.
  - One Add per LoCoMo session, as on 2026-09-23. Then one Search per question with the served
    router, top_k 100, and the returned items are kept. Then the users are deleted.
- **Dataset:** `locomo10.json`, SHA256 `79fa87e9…`. The questions are the 1,535 in categories
  1 to 4 whose evidence resolves, exactly as on 2026-09-23.
- **Answer, locally:**
  - The official `render_answer_prompt` from `data/locomo-refined/pipeline.py`, imported at runtime
    from `github.com/AML-memory/agent-memory-leaderboard` at `1b8142bf`.
  - Items render as AML's own timestamped block (`- [created_at] content`), in rank order.
  - All items go under `speaker_1_memories`, named after both speakers, because one user holds the
    whole conversation here.
  - `openai/gpt-4o-mini` through OpenRouter, temperature 0, as the official `complete()` sends it.
- **Judge:** the official `render_accuracy_prompt` and `parse_judge_label`, with the same model
  and settings. An unparsed label counts as WRONG and is reported.
- **Classify each WRONG answer:**
  - `RETRIEVAL_MISS` when turn hit@100 is 0.
  - Else `ABSTAINED` when the answer matches the fixed `ABSTENTION` pattern in the script.
  - Else one label from the fixed `CLASSIFIER_PROMPT`, by `openai/gpt-4.1` at temperature 0. It
    sees the question, gold, answer and the gold evidence turns with session dates.
- **First gold rank:** the rank of the first returned item containing a gold turn, by the same
  matcher that scored turn hit on 2026-09-23.
- **Cost cap:** 15 USD across answer, judge, self-check and classify, enforced in the script from
  OpenRouter's reported cost. My estimate is about 7 USD.

## Apparatus checks

1. **Canary:** the verbatim text of one ingested turn, used as the query, must score turn hit@10 = 1.
2. **Retrieval parity with 2026-09-23:** turn hit@10 within 1.0 point of 93.09 and turn hit@100 at
   least 99.5%. The corpus is re-added, and the Add-time compiler is not deterministic, so exact
   equality is not expected.
3. **Judge known answer:** every 15th question judged with its own gold answer as the generated
   answer must come back CORRECT at least 95% of the time. Otherwise the judge stand-in is not
   fit, and the bucket shares are not reported.
4. **Budget:** the longest Answer prompt must fit AML's 117,760-token input window, checked by
   character count. Otherwise the run is not a faithful copy of what AML sends.
5. **Classifier audit:** I read 30 classified answers, drawn by seed 0, and record my agreement.
   Below 70% agreement, the bucket shares are reported as unreliable and the decision rule does
   not fire.
6. The rule buckets are unit tested, and each test is proved red by mutation:
   `tests/test_aml_locomo_loss_diagnosis.py`.

## What I already know

- `c9-context4-route-worse-on-locomo` (2026-09-23): router turn hit@10 93.09, hit@100 99.87.
- `atm-over-abstention-is-the-largest-mechanism` (2026-08-22): on ATM, 139 wrong refusals were
  13.72 QS lost, the biggest recoverable mechanism.
- `aml-cycle1-open-method-lessons`: temporal updates and stale evidence are ranking or state
  problems. One entry saw no gain from a second Search pass when all top-k evidence is returned.
- `aml-textual-track-evidence`: the AML Answer prompt asks for absolute dates, while the judge
  demands the gold's granularity and forbids converting relative forms.

## Confounds I can name now

- **The Answer and judge models are stand-ins.** AML's public pipeline takes both from environment
  variables, and the live models are not disclosed here. gpt-4o-mini matches the Add-time model
  rule, not necessarily the Answer model. A stronger reader would lose fewer points, and in
  different places.
- **The speaker split is not AML's.** The official template has two speaker blocks, filled by how
  AML ingests a conversation. That is unknown here, so everything goes in one block.
- **LoCoMo is not LoCoMo-Refined.** The refined gold may fix some of what GOLD_ISSUE will catch.
- **The classifier is an LLM.** Its labels are the weakest link, hence apparatus check 5. Two
  buckets overlap: a temporal list error could be TEMPORAL or LIST.
- **Hit@100 uses turn text matching.** Compiled records that paraphrase a turn do not count, so
  RETRIEVAL_MISS can be slightly overstated.

## Result (2026-09-24)

**Status:** measured

**Run facts.**
- Collect `full1` on VPS3: C9 at `337f2537`, 272 Adds, 1,535 Searches, 0 failures, 0 retries,
  3,090 s in total.
- Collect artifact kept on VPS3 at `/home/sentiment/loss-diag-20260924/full1.json.gz`: SHA256
  `c677e89e…`, decompressed `84bb28b5…`. It is not committed, since it holds 40 MB of LoCoMo text.
- Answers, judge labels, classifier labels and the report are in
  `docs/results/2026-09-24-aml-c9-locomo-loss-diagnosis/`.
- OpenRouter spend: 4.42 USD answer, 0.16 judge, 0.48 classify, plus 0.02 for the judge self-check.

**Apparatus checks.**

| check | required | measured | pass |
| --- | --- | --- | --- |
| 1. canary | turn hit@10 = 1 | 1 | yes |
| 2. retrieval parity | turn hit@10 within 1.0 of 93.09, hit@100 at least 99.5 | 93.36, 99.87 | yes |
| 3. judge known answer | at least 95% CORRECT | 103 of 103 | yes |
| 4. budget | longest prompt inside 117,760 tokens | 91,441 characters | yes |
| 5. classifier audit | at least 70% agreement | 27 of 30 (90%) | yes |
| 6. rule-bucket tests | red by mutation | 3 of 3 | yes |

In the audit I disagreed on #17, #21 and #26:
- **#17:** the gold comes from an image caption, and the classifier's evidence block shows turn text
  only. That is a harness limitation.
- **#21:** missing specificity, not a list error.
- **#26:** the gold evidence turn belongs to a different speaker than the question names. That is a
  gold issue, so GOLD_ISSUE is probably undercounted.

**Predictions against measurements.** Shares are of the 379 WRONG answers.

| quantity | predicted | measured | in band |
| --- | --- | --- | --- |
| accuracy, all 1,535 | 68% to 80% | **75.3%** (1,156) | yes |
| lowest-accuracy category | 3 | 3, at 54.3% (1: 57.4%, 2: 59.1%, 4: 89.8%) | yes |
| RETRIEVAL_MISS | 0% to 3% | 0.3% (1) | yes |
| ABSTAINED | 5% to 20% | **1.8%** (7) | no, below |
| TEMPORAL | 20% to 40%, largest bucket | 28.8% (109), **second largest** | share yes, rank no |
| STALE | 1% to 8% | 2.1% (8) | yes |
| LIST | 8% to 20% | **31.7% (120), the largest** | no, above |
| DISTRACTOR | 8% to 20% | 19.5% (74) | yes |
| INFERENCE | 8% to 25% | 12.7% (48) | yes |
| JUDGE | 3% to 12% | 2.9% (11) | no, just below |
| GOLD_ISSUE | 5% to 15% | **0.3%** (1) | no, below |
| rank 1 to 10 minus rank 11 to 100 | +5 to +20 points | +16.4 (76.4% of 1,433 against 60.0% of 100) | yes |

**Gap.**
- **LIST errors are the largest loss, not temporal ones.** 89 of the 120 are category 1 multi-hop
  questions ("what games does Nate play", "what tricks do James's pets know"). The items are spread
  across sessions, every one of them was returned, and the reader listed only some. I predicted
  8% to 20% and it is 31.7%, which is the error worth learning from: I modelled the reader's failure
  as reading dates, when its biggest failure is collecting.
- **Abstention is almost absent**, at 7 answers. The ATM precedent did not transfer, because the
  AML Answer prompt tells the reader not to refuse.
- **GOLD_ISSUE is near zero** as labelled, against a 5% to 15% prediction. The audit suggests the
  classifier undercounts it, because its evidence block omits image captions.
- The rank effect is in band, but it is confounded by question difficulty: questions whose gold
  sits deep may simply be harder.

**Decision, by the rule fixed above.**
1. JUDGE plus GOLD_ISSUE is 3.2%, not at least 30%. Does not fire.
2. ABSTAINED is 1.8%, not at least 15%. Does not fire.
3. STALE is 2.1%, not at least 10%. Does not fire.
4. TEMPORAL is not the largest bucket; LIST is. Does not fire.
5. **Fires: no Add-time candidate among the three is supported by LoCoMo.** Currentness (1) is ruled
   out by STALE at 2.1%. Temporal anchoring (3) is not licensed as the next build, though temporal
   errors are 28.8%. Rules and preferences (2) cannot be decided on LoCoMo.

**Exploratory, not pre-registered.**
- **What anchoring could fix.** Of the 12 TEMPORAL answers in the audit sample, about 6 are
  anchoring errors that dating relative expressions at Add could fix: the message date given
  instead of the event date, or "last Friday" left unconverted. About 4 are granularity or
  relative-form mismatches that anchoring could make worse.
- **Accuracy by route.** The route was recomputed with the served `route_query`; the harness had
  read it from the response body, where it is never present, and the report's by-route table is
  therefore empty.

  | route, category | n | accuracy |
  | --- | ---: | ---: |
  | context, 2 temporal | 214 | 57.5% |
  | code, 2 temporal | 104 | 61.5% |
  | context, 4 single hop | 184 | 88.6% |
  | code, 4 single hop | 632 | 91.1% |
  | multimodal, 4 single hop | 25 | 64.0% |

  The route is chosen by the question text, so these differences mix question type with route.
  The compiled-records counterfactual (`2026-09-24-aml-c9-compiled-records-counterfactual.md`) is
  the controlled test for the context route.
- **A new candidate, not on the pre-registered list:** an Add-time aggregation view. For each
  person, it would hold one record listing the items of one kind (games, pets, trips) with links
  to their source turns. It targets the LIST bucket directly and would need its own
  pre-registration, on questions this run has not already shown me.

## Note (2026-09-24, after the result)

At the user's request, the VPS3 run directory `/home/sentiment/loss-diag-20260924` and database
`lossdiag_20260924` were deleted. With them went the 40 MB collect artifact `full1.json.gz` (SHA256
`c677e89e…`) that the result above names. What remains: its hashes, the committed answers, judge
labels, classifier labels and report, and a regenerable path through `scripts/aml_locomo_loss_diagnosis.py
collect` at `337f2537`. The Add-time compiler is not deterministic, so a regenerated collect will not
reproduce the hash.
