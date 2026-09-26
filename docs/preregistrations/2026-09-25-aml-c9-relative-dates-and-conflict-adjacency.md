# Pre-registration: C9 resolved relative dates (T-1) and same-subject adjacency (K-2)

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Baseline is the served C9 at `3eb447c4` (the round-two day-zero baseline):
content-only windows, each returned text item prefixed with its own date (`dated_search_content`,
arm H of `docs/preregistrations/2026-09-25-aml-c9-window-format.md`). Local directional
experiment over public data; not an AML hosted evaluation.

Both changes are Search-time render transforms, default off, that touch neither what is stored,
embedded or ranked by retrieval. So both are measured by applying the production function to
already-stored retrieval, and every arm is paired on identical evidence.

## Why

- **T-1.** C1 is the most mechanism-shaped gap on the AML board: Cycle 1 Textual median 16.4,
  leaders 58 to 65, and the leaders that disclose a method resolve relative dates against the
  message timestamp (MemOS's extraction prompt says so). On C9, BEAM temporal_reasoning is 0.290
  (measured before dates were served) and LoCoMo category 2 is about 60. The LoCoMo loss audit
  (`docs/preregistrations/2026-09-24-aml-c9-locomo-loss-diagnosis.md`) found about 6 of 12
  sampled temporal errors were anchoring ("last Friday" left unconverted, or the message date given
  instead of the event date), and about 4 were granularity or relative-form mismatches that
  anchoring could make worse. H gives the reader each item's date; the reader still does the
  arithmetic.
- **K-2.** BEAM contradiction_resolution is 0.063, the worst type measured; Cycle 1 D2 median is
  19.6, with one entry at 48.3. The compiler's `supersedes` field was used 0 times in 263,662
  compiled rows, so nothing in C9 ever puts an old and a new statement about the same thing side
  by side. Items arrive in relevance order, where the two versions can sit far apart.

## The questions

- **T-1:** Does annotating each relative time expression in a returned item with its resolved
  absolute date raise LoCoMo category 2 accuracy and BEAM temporal_reasoning against H, without
  lowering overall accuracy?
- **K-2:** Does placing returned items that share a subject but carry different dates next to
  each other, newest first, raise BEAM contradiction_resolution plus knowledge_update against H,
  without lowering overall accuracy or Coding MRR?

## The two transforms, fixed now

**T-1, `resolve_relative_times`** (new module `recall_aml/temporal_render.py`; variant flag
`resolved_relative_times`, default off; `RECALL_AML_RESOLVE_RELATIVE_TIMES` for an experiment).
For each text item with `created_at`, every match of the fixed pattern list below gets a bracket
inserted right after it. The anchor is the item's own `created_at` date in UTC. The original words
stay, so a relative-form gold answer is still reachable. Nothing else in the item changes.

| Expression (case-insensitive) | Inserted |
|---|---|
| today, tonight, this morning, this afternoon, this evening | `[= YYYY-MM-DD]` (anchor day) |
| yesterday, last night | `[= YYYY-MM-DD]` (anchor minus 1 day) |
| the day before yesterday | `[= YYYY-MM-DD]` (minus 2) |
| tomorrow | `[= YYYY-MM-DD]` (plus 1) |
| N days / weeks / months / years ago (N as digits, a, one to twelve, a couple of = 2, a few = 3) | days and weeks: `[≈ YYYY-MM-DD]`; months: `[≈ YYYY-MM]`; years: `[≈ YYYY]` |
| last / this / next week | `[week of YYYY-MM-DD]` (the Monday) |
| last / this / next weekend | `[weekend of YYYY-MM-DD]` (the Saturday) |
| last / this / next month | `[= YYYY-MM]` |
| last / this / next year | `[= YYYY]` |
| last / this / next Monday … Sunday | `[= YYYY-MM-DD]`: last = the most recent such day strictly before the anchor; next = the first strictly after; this = the one in the anchor's Monday-based week |

**K-2, `same_subject_adjacent`** (new module `recall_aml/conflict_order.py`; variant flag
`same_subject_order`, default off; `RECALL_AML_SAME_SUBJECT_ORDER`). Over the top 30 returned
items only:

1. An item's subject tokens are its lowercased alphabetic words of at least 4 letters, minus a
   fixed English stopword list, minus any word present in at least half of those 30 items (which
   removes speaker names and the conversation's constant vocabulary).
2. Two items are linked when their subject-token Jaccard is at least 0.35 and their `created_at`
   fall on different UTC days.
3. Groups are connected components of the links, capped at 4 members (kept greedily by highest
   link weight to the group's best-ranked member).
4. Each group of 2 or more is moved to the rank of its best-ranked member, its members newest
   first; every other item keeps its relative order. Items ranked 31 to 100 are untouched.

No parameter above is tuned on any measured set; each is fixed here.

## Arms

All read by `deepseek/deepseek-v4.1-flash` (user instruction 2026-09-25: no gpt-4o-mini until
experiments finish), temperature 0, reasoning off, one pinned OpenRouter provider, judged by the
same model with AML's own judge prompts through the existing harnesses.

| Arm | Content the reader sees |
|---|---|
| **H** | served baseline: stored items, `dated_items` applied |
| **H′** | H answered and judged a second time, concurrently (reader and judge noise floor) |
| **T1** | H, then `resolve_relative_times` |
| **K2** | H, then `same_subject_adjacent` |

## Data

- **LoCoMo**: the stored C9 retrieval `collected-S.json.gz` on VPS3
  (`/home/sentiment/reader-dates`, 1,535 questions, 100 items each, written 2026-09-24 by the
  reader-dates collect, content-only windows). Answered through `scripts/aml_locomo_loss_diagnosis.py`
  (`answer`, `judge`) with AML's LoCoMo prompts. To bound cost: all 320 category 2 questions, plus
  400 of the other 1,215 drawn once with `random.Random(20260925)`, stratified by category. The
  draw is written to a file and committed before any answer is generated.
- **BEAM 100K**: the stored C9 probe retrieval of 2026-09-24 (400 questions, 20 conversations,
  `385c6074`), with AML's BEAM prompts through `benchmarks/beam/aml_c9_probe.py`. That file is on
  VPS2, which is hands-off while the official Textual Full runs. BEAM is measured only after the
  user lifts that rule; if it is not lifted by the time LoCoMo is done, the BEAM half is re-collected
  on VPS3 from the same public data, and this record says which.
- **Coding guard for K-2**: the stored Coding retrieval `coding-K3.json.gz` on VPS3 (34 prompts,
  served C9 with dated content). MRR is recomputed after K-2's reordering; no model call.

## What I predict

Written before either transform exists in code. My effect predictions have run two to four times
too high (memory `i-over-predict-effect-magnitudes`), so the bands sit low.

**Mechanism (no model call):**

| Metric | Predicted |
|---|---|
| LoCoMo questions whose top 20 items contain at least one resolved expression | 0.60 to 0.90 |
| Resolved expressions per LoCoMo question, top 20 | median 2 to 8 |
| BEAM questions with at least one K-2 group in the top 30 | 0.30 to 0.70 |
| LoCoMo questions with at least one K-2 group in the top 30 | 0.20 to 0.60 |
| Coding prompts whose top 10 changes under K-2 | 0.05 to 0.40 |

**Answers:**

| Contrast | Set | Predicted |
|---|---|---|
| T1 − H | LoCoMo category 2 (320) | **+2 points**, band −2 to +6 |
| T1 − H | LoCoMo, the 720 answered | +0.5, band −1.0 to +2.0 |
| T1 − H | BEAM temporal_reasoning (40) | +0.03, band −0.04 to +0.10 |
| K2 − H | BEAM contradiction_resolution + knowledge_update (80) | **+0.02**, band −0.02 to +0.06 |
| K2 − H | BEAM contradiction_resolution alone (40) | +0.01, band −0.02 to +0.05 |
| K2 − H | LoCoMo, the 720 answered | −0.3, band −1.5 to +0.5 |
| K2 − H | BEAM 10-type mean | −0.005, band −0.02 to +0.01 |
| K2 − K0 | Coding K3 MRR | −0.01, band −0.04 to 0.00 |
| H′ − H | LoCoMo 720 | within ±1.5 points |

The T-1 band reaches below zero on purpose: the audit says some of its targets could get worse.
The K-2 contradiction prediction is near zero because 34 of 40 BEAM contradiction answers are a
bare "Yes." or "No." under AML's "be direct and concise" answer prompt, which no context change
removes.

## What would falsify this

- **T-1:** T1 − H on LoCoMo category 2 at or below 0, or its paired 95% CI entirely below +0.5;
  or the mechanism rate below 0.30 (the transform barely fires, so any result is noise).
- **K-2:** K2 − H on BEAM contradiction plus knowledge update at or below 0; or groups forming on
  more than 90% of questions (the similarity rule is linking everything, not subjects).
- **Apparatus:** |H′ − H| on LoCoMo above the T-1 effect, in which case neither is claimed.

## Decision rule

- **Recommend T-1** if T1 − H on LoCoMo category 2 is at least +2.0 with a CI lower bound above 0,
  T1 − H on the 720 is at least −0.5, BEAM temporal is at least −0.02 (when measured), and
  |H′ − H| is smaller than the category 2 effect.
- **Recommend K-2** if K2 − H on BEAM contradiction plus knowledge update is at least +0.03,
  LoCoMo 720 at least −0.5, BEAM 10-type mean at least −0.01, and Coding MRR at least −0.01.
- Both change the C9 baseline, so both need an explicit user decision, and neither may be deployed
  while an AML job is running.

## How it will be measured

1. Build both transforms with default-off flags and unit tests, each seen red against a deliberate
   mutation of its production line before green (apparatus check 1).
2. Commit the LoCoMo draw file.
3. Offline on VPS3: materialise H, T1 and K2 views of `collected-S.json.gz` by applying the
   production functions; compute the mechanism metrics; recompute Coding MRR under K-2.
4. Answer and judge H, H′, T1 and K2 on the 720 LoCoMo questions concurrently.
5. BEAM, when available (see Data), the same four arms on 400 questions.
6. Paired bootstrap over questions, 10,000 resamples, seed 20260925.

**Spend.** DeepSeek V4.1 Flash only. Estimated USD 12 to 18 for LoCoMo and USD 4 to 7 for BEAM.
Cap USD 25 across both. Every paid step reads the OpenRouter balance first and stops below USD 40,
because the key is shared with the official C9.

## Apparatus checks, fixed now

1. Unit tests with red proof: T-1 only inserts brackets after matched phrases (removing every
   inserted bracket gives back the original text byte for byte); ids, order and scores unchanged;
   one test per row of the T-1 table on a fixed anchor date. K-2 returns the same multiset of items
   with identical contents; ranks 31 to 100 unchanged; an item with no link keeps its relative order.
2. The render-only property above re-checked on every stored LoCoMo and BEAM row.
3. Mechanism metrics reported before any answer is scored.
4. H′ answered concurrently with the other arms, not afterwards.
5. Valid-answer rate per arm; empty answers counted as wrong, never dropped.
6. The LoCoMo draw is identical across arms (same question ids, verified by hash).

## What I already know

- H on LoCoMo with a DeepSeek Flash reader: 70.49 overall, category 2 61.56
  (`docs/preregistrations/2026-09-25-aml-c9-window-format.md`). That reader was
  `deepseek-v4-flash-0731`; this record uses `deepseek-v4.1-flash`, so H is re-answered here and not
  taken from that run.
- Chronological order for every question: event ordering +0.10, temporal −0.13, information
  extraction −0.17 on BEAM (`docs/preregistrations/2026-09-24-c9-beam-ability-probe.md`). K-2
  reorders only within same-subject groups inside the top 30, but that result is the main reason
  its LoCoMo band reaches below zero.
- The aggregation view (new synthesised per-person records) cost multi-hop −3.19. K-2 adds no
  record and no text.
- Temporal anchoring at Add was named by the loss diagnosis and never licensed; this is its first
  test, moved to render time so storage, embeddings and BM25 are untouched.

## Confounds I can name now

- **Anchor error.** A window's `created_at` is its Add's latest message time. For LoCoMo one Add is
  one session, so one date; for multi-day Adds the anchor can be late, which would resolve
  "yesterday" to the wrong day.
- **Judge strictness.** AML's LoCoMo judge demands the gold's granularity and relative form, so an
  answer made more precise by T-1 can be marked wrong. The original phrase is kept for this reason.
- **Reader.** DeepSeek V4.1 Flash is not AML's reader. Both effects depend on how a reader uses
  bracketed dates and item order.
- **Subset.** LoCoMo is answered on 720 of 1,535 questions; the overall figure is a stratified
  estimate, not the full set.
- **Public data.** LoCoMo and BEAM are among AML's Textual sources; neither transform is tuned on
  them, but a gain is not held out from AML.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

### Amendment 1, 2026-09-25, before any measurement

Written after both transforms were built (`58cc540b`) and before any mechanism metric or answer.
Predictions, falsifiers and decision rule are unchanged.

1. **The LoCoMo draw is committed**: `results/aml-t1k2/locomo-draw.json`, 720 ids (category 2:
   320; category 1: 93; category 3: 30; category 4: 277), seed 20260925, drawn from
   `collected-S.json.gz` with SHA-256 `243f48eb…9f74`; ids SHA-256 `6c4f0d01…9571`.
2. **Schedule, at the user's instruction to keep OpenRouter use low**: LoCoMo starts only after
   the MM-1/MM-3 Stage 2 run has finished, never alongside it. Two workers. For each question the
   four arms are answered back to back in an order that rotates with the question, which is how
   "concurrently" is met here: reader drift over the run falls on every arm alike.
3. **Reader and judge**: `deepseek/deepseek-v4.1-flash` for both, pinned to `DeepInfra` with
   fallbacks off, temperature 0, reasoning off, output capped at 300 tokens for an answer and 400
   for a verdict (runaway generations near 131k tokens cost money here on 2026-09-25 when
   uncapped). Driver: `scripts/aml_t1k2_locomo.py`, prompts from the pinned AML checkout via
   `scripts/aml_locomo_loss_diagnosis.py`.
4. **UNPARSED verdicts count as not correct** in the primary figures, in every arm alike. The
   older DeepSeek judge left 114 of 1,535 (7.4%) unparsed on the window-format run. A sensitivity
   figure with any pair containing an UNPARSED verdict dropped is reported beside the primary one.
5. **The Coding guard for K-2 cannot use the stored retrieval**: `coding-K3.json.gz` keeps ids,
   sessions and kinds per item, but no content and no dates, and K-2 needs both. The guard needs a
   Coding re-collect on VPS3 that keeps item content; it runs after LoCoMo, and K-2 is not
   recommended until it has.
6. **Spend cap for the LoCoMo half: USD 18**, inside the record's USD 25 for both halves.

## Result, mechanism metrics on LoCoMo (2026-09-25)

**Status:** mechanism measured on the 720 drawn questions; no answer generated yet.
`python scripts/aml_t1k2_locomo.py mechanism` at `90c7188e` on VPS3, no model call. Output:
`results/aml-t1k2/locomo-mechanism.json`.

| Metric | Measured | Predicted | Gap |
|---|---|---|---|
| Questions with at least one T-1 resolution in the top 20 | **0.999** (719 of 720) | 0.60 to 0.90 | above the band |
| T-1 resolutions per question, top 20, median | **9** | 2 to 8 | above the band |
| Questions whose top 30 K-2 reorders | **0.000** (0 of 720) | 0.20 to 0.60 | **below the band: K-2 never fires** |
| Render-only property on every row (both transforms) | holds | required | |

**Why K-2 never fires (diagnosis, no parameter changed).** Dates are present (median 15 distinct
days among the top 30), and each item keeps a median of 42.5 subject words after filtering. But
LoCoMo's returned items are 160-word windows of chat, and the best cross-day subject Jaccard per
question has a median of 0.156, a 90th percentile of 0.190 and a maximum of 0.250; 5.9% of
questions reach 0.20 and 0.1% reach 0.25. The pre-registered threshold, 0.35, is unreachable.
I fixed it thinking of short single statements, not of windows, without looking at a single
window. The falsifier I wrote guarded only the opposite failure (linking everything).

So K-2 as registered is inert on LoCoMo: its arm would be byte-identical to H on all 720
questions, and K2 − H there is exactly 0 by construction. That is recorded as the LoCoMo result
for K-2. Whether it fires on BEAM is still open. The threshold is NOT retuned on this data; a
K-2 with a different similarity signal needs its own pre-registration.

### Amendment 2, 2026-09-25, before any answer

The K2 arm is dropped from the LoCoMo answer run: it would repeat H exactly, at a quarter of the
LoCoMo spend, against the user's instruction to keep OpenRouter use low. LoCoMo answers H, H2 and
T1 (`--arms H,H2,T1`). Every T-1 prediction, falsifier and decision rule is unchanged.

### Amendment 3, 2026-09-25, before any answer

The LoCoMo half's credit floor is lowered from USD 40 to **USD 5**, on the user's instruction the
same day (DeepSeek costs are small; the user accepts the risk to the official Textual Full sharing
the key). It runs after the MM-1/MM-3 Stage 2 answers, never alongside them. Everything else in
amendments 1 and 2 is unchanged, including the USD 18 cap and arms H, H2 and T1.

### T-1 LoCoMo result, 2026-09-26

Arms H, H′ and T1 on the committed 720-question draw (all 320 category 2), reader and judge
`deepseek/deepseek-v4.1-flash` pinned to one provider, AML's LoCoMo answer and accuracy prompts,
arms interleaved per question. `scripts/aml_t1k2_locomo.py` at `4861b083` for the first 1,761
answers; the run then died on an unretried OpenRouter 520 and was resumed from its own output at
`9646fca2`, whose only change to anything the run imports is adding 520, 522 and 524 to the retry
set (verified by diff). The resume refused once below the USD 5 floor and ran after the user's
top-up. 2,160 answers, 0 duplicate pairs, 0 unparsed verdicts, USD 3.96. Output:
`results/aml-t1k2/locomo-score.json`.

| Contrast | Set | Predicted | Measured [95% CI] | wins / losses |
|---|---|---|---|---|
| T1 − H | LoCoMo category 2 (320) | +2 points, band −2 to +6 | **+14.69** [+9.06, +20.31] | 70 / 23 |
| T1 − H | LoCoMo, the 720 | +0.5, band −1.0 to +2.0 | **+6.67** [+3.89, +9.44] | 79 / 31 |
| H′ − H | category 2 / all 720 | (noise) | −1.56 [−4.38, +1.25] / −0.42 [−1.94, +1.11] | |

Accuracy: H 56.67 (category 2: 32.19), H′ 56.25 (30.63), T1 63.33 (46.88).

**Against the decision rule:** T1 − H on category 2 is +14.69 ≥ +2.0 with a CI lower bound of
+9.06; on the 720 it is +6.67 ≥ −0.5; |H′ − H| (1.56) is far below the effect. **The LoCoMo half
recommends T-1.** The rule's BEAM condition (temporal at least −0.02) is not measured: the BEAM
half still waits for VPS2 or a VPS3 re-collect, so the recommendation is not complete.

**Both predictions were far too low, seven and thirteen times the point estimates**, the same
direction as MM-1 Stage 2 the same day. The mechanism result (a resolution on 719 of 720 questions,
median 9 per question) was already far above its band, and I did not revisit the answer prediction.

**Why the effect is this large here, which qualifies it:**

1. **This reader does not resolve relative dates.** H scores 32.19 on category 2, against 61.56 for
   the same view with `deepseek-v4-flash-0731` (`docs/preregistrations/2026-09-25-aml-c9-window-format.md`).
   `deepseek-v4.1-flash` copies the phrase ("Last year", "Next month", "yesterday") instead of
   computing the date. T-1 writes the date into the text, and the reader copies that instead
   ("Last year (2022)", "2023-05-07"). A reader that does the arithmetic itself would gain less.
2. **AML's LoCoMo judge rule punishes relative answers, and this judge applies it unevenly.** The
   accuracy prompt forbids converting a relative expression, so "Last year" is wrong against gold
   "2022". The same judge marked "Yesterday (2023-05-07)" wrong and "Last year (2022)" right. T-1's
   gain is partly a gain in answer FORM that AML's own rule rewards; that part should transfer to
   any judge that applies the rule, but its size depends on how often the official reader emits a
   relative phrase, which is not known.
3. **One run per arm**, as registered; the H′ replicate bounds the reader and judge noise.

The planned confirmation with `gpt-4o-mini` (the user's instruction: only after the DeepSeek numbers
and all experiments are in) is where point 1 is tested: if gpt-4o-mini's H already resolves dates,
the gain will shrink toward what that reader leaves unresolved.

### Amendment 4, 2026-09-26, before any BEAM answer: the BEAM half as it will run

The user lifted the VPS2 hands-off rule on 2026-09-26, so the BEAM half uses the stored retrieval
this record named: `/root/c9-beam-probe-out-20260924.tgz` from VPS2 (official C9 `385c6074`,
2026-09-24, 400 questions, 20 conversations, 100 items each, nonce `67c487cdf4`), copied to VPS3,
with the BEAM data file `beam100k.jsonl` (sha256 prefix `62a950489d597b68`, identical on both hosts).
Nothing is re-collected and no Voyage call is made.

1. **Arms: H, H′ and T1 on all 400 questions.** K-2 is closed (three versions, this record's
   successors), so its arm is dropped, as it was for LoCoMo in amendment 3.
2. **H is built offline.** The stored retrieval predates the date header (#761), so its items carry
   `created_at` but no header in content. H applies the production `dated_items` to the stored
   items; T1 applies `resolve_relative_times(dated_items(...))`, exactly as on LoCoMo. H′ is H
   answered separately.
3. **Answer and judge** are `benchmarks/beam/aml_c9_probe.py`'s: AML's BEAM answer prompt, its batch
   rubric judge on the 0 / 0.5 / 1 scale, and the event-ordering alignment score, from the pinned
   AML checkout `1b8142b`. Only the model call is replaced: `deepseek/deepseek-v4.1-flash`, one
   pinned provider, reasoning off, temperature 0, for both answer and judge (the probe used
   Qwen3-14B). So these BEAM numbers are not comparable with the 2026-09-24 probe's levels, only
   with each other.
4. **Spend cap USD 7** (the record's estimate was 4 to 7), balance floor USD 5, never alongside
   another OpenRouter job.

Predictions, falsifiers and the decision rule are unchanged: T1 − H on BEAM temporal_reasoning
(40) is predicted +0.03, band −0.04 to +0.10, and the rule needs it at least −0.02.
