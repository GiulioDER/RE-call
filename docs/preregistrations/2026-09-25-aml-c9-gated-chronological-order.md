# Pre-registration: E-1, chronological order only for questions that ask about order

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Baseline: served C9 at `3eb447c4`. Local directional experiment over public
data; not an AML hosted evaluation.

## Why

On BEAM 100K through C9, answering from the same returned items put in chronological order changed
each question type as follows (Qwen3-14B reader, `docs/preregistrations/2026-09-24-c9-beam-ability-probe.md`):
event_ordering **+0.101**, temporal_reasoning −0.132, information_extraction −0.167,
multi_session_reasoning −0.070, summarization −0.027 (`docs/results/2026-09-25-c9-beam-diagnosis.md`).
Event ordering is the one type it helps, and it is C9's second-lowest BEAM type (0.224, above only
contradiction resolution's 0.062). So the order helps where the question is about order and hurts
elsewhere. E-1 applies it only where the question asks.

Two lessons from 2026-09-25 shape this record. Question-keyed mechanisms reach few questions
(4.4% of multimodal questions name the medium, 2.5% to 17% name a time; memory
`questions-name-their-content-not-their-time-or-form`), so the gate's reach is counted first, for
free. And 40 BEAM questions of one type carry a reader noise of about 0.08 between identical runs
(`docs/preregistrations/2026-09-25-f1-storyline-replay.md`), so each gated question is answered
several times per arm.

## The mechanism, fixed now

New module `recall_aml/order_gate.py`, variant flag `ordering_gate`, default off, experiment override
`RECALL_AML_ORDERING_GATE`:

1. **Gate.** `asks_for_order(question)` is true when the question matches, case-insensitive, any of:
   `in what order`, `in which order`, `what order`, `chronological`, `chronologically`, `sequence of`,
   `the order (in which|that|of)`, `order (did|do|were|was)`, `timeline`, `which came first`,
   `which happened first`, `what happened (first|next|last|before|after)`, `list .* in (the )?order`,
   `rank .* by (date|time)`, `earliest to latest`, `oldest to newest`, `first to last`. Nothing else.
2. **Action.** When the gate is true, the returned items are re-sorted by `created_at`, earliest
   first; items without one keep their relative order after the dated ones. The same items, the same
   text; nothing added or dropped. When false, Search output is byte-identical to today.

The pattern list is written here before any census and is not edited afterwards; a gate that misses
or over-fires is a result, not something to tune.

## Stage 0, census (free, no model)

| Set | n | Predicted gate firing |
|---|---:|---|
| BEAM 100K, event_ordering | 40 | at least 0.70 (the gate's recall on its target) |
| BEAM 100K, the other 360 | 360 | at most 0.05 (false fires) |
| ScriptMem, ordering questions (as ScriptMem labels them) | from X-1's draw | 0.50 to 0.90 |
| ScriptMem, other questions | from X-1's draw | at most 0.05 |
| LoCoMo, all | 1,986 | 0.00 to 0.03 |
| LongMemEval-S, all | 500 | 0.01 to 0.06 |

**Gate:** Stage 1 runs only if recall on BEAM event_ordering is at least 0.70 and the false-fire rate
on BEAM's other types is at most 0.05. ScriptMem is counted when X-1's draw exists and does not gate.

## Stage 1, answers (only if Stage 0 passes)

- **Retrieval.** BEAM 100K's stored C9 retrieval of 2026-09-24 once the VPS2 hands-off rule is lifted,
  or a VPS3 re-collect of BEAM 100K otherwise (the record says which), with `dated_items` applied as
  the served baseline. ScriptMem's ordering questions from X-1's Stage B collect, when it exists.
- **Arms**, built offline from the same stored retrieval with the production functions: **H** (served),
  **H′** (H again), **E1** (H with the gate's reordering where it fires).
- **Only gated questions are answered**: on every other question E1 equals H by construction.
- **Three answers per question per arm**, interleaved, averaged per question, to cut reader noise.
- Reader and judge: `deepseek/deepseek-v4.1-flash`, one pinned provider, reasoning off, bounded
  output; AML's BEAM and ScriptMem prompts and scoring from the pinned checkout.

| Contrast | Set | Predicted |
|---|---|---|
| E1 − H | BEAM event_ordering, gated questions | **+0.05**, band 0.00 to +0.10 |
| E1 − H | ScriptMem ordering, gated questions | +0.02, band −0.05 to +0.08 |
| H′ − H | BEAM gated questions | within ±0.04 |
| E1 − H, whole BEAM set, counting ungated questions as unchanged | all 400 | +0.004, band −0.002 to +0.010 |

The BEAM prediction is half the ungated +0.101: a different reader (DeepSeek, not Qwen3-14B), and my
effect predictions have run two to four times too high (memory `i-over-predict-effect-magnitudes`).

## What would falsify this

- Stage 0: recall below 0.70 on BEAM event_ordering, or false fires above 0.05 on its other types.
- Stage 1: E1 − H at or below 0 on BEAM's gated event_ordering questions, or |H′ − H| as large as
  E1 − H.

## Decision rule

Recommend E-1 if Stage 0 passes, E1 − H on BEAM's gated event_ordering questions is at least +0.04
with |H′ − H| smaller, and ScriptMem's ordering questions are not worse by more than the noise
floor. Then the X-1 held-out rule applies (non-inferior on an X-1 source not used here) before any
serving decision, which is the user's. Never deployed during an AML job.

## Apparatus checks, fixed now

1. `asks_for_order` unit tests, one per pattern and several near-misses ("order a pizza", "in order
   to", "what did I order"), each seen red against a deliberate mutation before green.
2. With the gate false, Search output byte-identical to the flag off (red-proved by forcing the gate).
3. E1 on a gated question returns the same multiset of items as H, sorted by `created_at`.
4. Valid-answer rate at least 98% per arm.

## Spend

Stage 0: none. Stage 1: about 40 BEAM questions × 3 arms × 3 answers, plus ScriptMem's gated
questions, with judges: about USD 1 to 3 at DeepSeek V4.1 Flash rates; cap USD 5, the USD 5 balance
floor the user set on 2026-09-25, never alongside another OpenRouter job.

## What I already know

- The ungated BEAM result above, and that time order and session summaries were the only BEAM levers
  that helped event ordering (both +0.10), both hurting elsewhere
  (`docs/results/2026-09-25-c9-beam-diagnosis.md`).
- BEAM scores event ordering as evidence F1 times a normalised Kendall tau (memory
  `2026-09-18-aml-textual-track-evidence`), so order matters directly to the score.
- ScriptMem scores ordering by exact match: an approximate order scores zero.

## Confounds I can name now

- **The gate may fit BEAM's wording.** BEAM's event-ordering questions may share phrasing that the
  pattern list happens to match; ScriptMem's ordering questions and the false-fire counts on
  LoCoMo and LongMemEval are the checks.
- **`created_at` is when a window was stored,** so chronological order is conversation order, not
  event order; a memory told late about an early event sorts late. That is what the ungated test
  measured too.
- **Reader mismatch.** DeepSeek here, Qwen3-14B in the ungated measurement and in AML's BEAM reader.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

### Stage 0 result, 2026-09-25

`scripts/aml_e1_census.py` at `75d36563` on VPS3, no model and no service, in a throwaway uv
environment (the arms' venv has no pyarrow and was not touched). Inputs by sha256 prefix: BEAM
`100K.parquet` `c0519be25907005b`, `locomo10.json` `79fa87e90f040813`,
`longmemeval_s_cleaned.json` `d6f21ea9d60a0d56`; ScriptMem from `memorax-ai/ScriptMem` at
`22ac7e7e`, all four scripts. Output: `results/aml-e1/census.json`.

| Set | n | Predicted | Measured | |
|---|---:|---|---:|---|
| BEAM 100K, event_ordering | 40 | at least 0.70 | **0.975** (39) | held |
| BEAM 100K, the other 360 | 360 | at most 0.05 | **0.014** (5) | held |
| ScriptMem, ordering | 36 | 0.50 to 0.90 | **0.250** (9) | **falsified, low** |
| ScriptMem, other | 421 | at most 0.05 | **0.024** (10) | held |
| LoCoMo, all | 1,986 | 0.00 to 0.03 | **0.000** (0) | held |
| LongMemEval-S, all | 500 | 0.01 to 0.06 | **0.016** (8) | held |

**The gate passes, so Stage 1 is authorised.** It has not started: its BEAM half needs stored
retrieval, which is on VPS2 (hands-off while the official Textual run lasts) or a VPS3 re-collect,
and it must not share OpenRouter with the MM-1 Stage 2 and T-1 LoCoMo runs now in progress.

What the counts say beyond the gate:

1. **The BEAM hit is one phrase.** 39 of BEAM's 40 event-ordering questions match
   `the order (in which|that|of)`; `list .* in (the )?order` adds nothing that phrase does not
   already catch. The one miss is "Can you list in order how...", which the list pattern does not
   match because it needs a word between "list" and "in". So BEAM's recall measures BEAM's template,
   which is the overfitting risk the record named, now observed.
2. **ScriptMem shows the risk is real.** Its ordering questions say "what is the correct order"
   and "from nearest to farthest", neither of which is on the list, so the gate reaches 9 of 36.
   Several of them ask for a backward causal chain (nearest to farthest from an endpoint), which
   earliest-first presentation does not match in direction. Nine gated questions cannot resolve an
   effect, so ScriptMem in Stage 1 is a direction check at most.
3. **The five BEAM false fires are all `timeline`** used as a noun ("editing timelines", "the
   development timeline"), in contradiction resolution, information extraction, multi-session
   reasoning and two summarization questions. Reordering those is the ungated loss on a small scale.
4. **ScriptMem's "false" fires are mostly not false.** Eight of its ten are single-choice questions
   asking for a "sequence of events" or a "chronological progression"; they ask about order, and
   ScriptMem labels them by answer format rather than by what is asked. Two fire on option text
   (`timeline` inside a lettered option), because ScriptMem's question field carries its options.
5. **LongMemEval-S's eight are all true ordering questions**, all in temporal-reasoning ("What is
   the order of the six museums I visited from earliest to latest?"). No other type fires.
   LongMemEval-S therefore has more gated questions that genuinely ask for order than ScriptMem
   does, and is the better second dataset for Stage 1 if X-1's Stage B collects it.

The pattern list is unchanged, as fixed. A revised list (adding "correct order", "list in order",
"nearest to farthest", dropping bare `timeline`) would be a new pre-registration, and it would have
been written after seeing these questions, so it could only be tested on data not used here.

🔁 **Correction to point 4 above, 2026-09-25, same hour; the point is left as written.** It says
"Eight of its ten are single-choice questions asking for a 'sequence of events' or a 'chronological
progression'". Recounted from `results/aml-e1/census.json`: eight of the ten are single-choice, but
**six** of the ten ask about order (four "sequence of events/actions/interactions", two
"chronological progression"). The other four are not about order: "Hovstad's sequence of remarks"
(multi-select), "a timeline difference" (single-choice), and two that fire only on option text
(`sequence of` in a multi-select option, `timeline` in a single-choice option). So ScriptMem's
false fires by what is asked are 4 of 421 (0.010), not 2.

### Note, 2026-09-25, after Stage 0 and before Stage 1: ScriptMem cannot enter Stage 1

X-1's Stage A found that ScriptMem's public release omits the script conversations
(`docs/preregistrations/2026-09-25-aml-c9-source-coverage-baseline.md`, Stage A result). Its
questions can be counted, which Stage 0 did, but there is no memory to Add and so no retrieval to
reorder. The ScriptMem rows of Stage 1 and of the decision rule cannot be measured and will be
reported as unavailable. LongMemEval-S, whose eight gated questions are all genuine ordering
questions (Stage 0, point 5), is the only second dataset left; with eight questions it is a
direction check, not a measurement.

### Amendment 1, 2026-09-26, before any Stage 1 answer: Stage 1 as it will run

The user lifted the VPS2 hands-off rule on 2026-09-26, so Stage 1 uses the stored retrieval this
record named (the 2026-09-24 probe of official C9 `385c6074`, the same file T-1's amendment 4
uses), with no re-collect.

1. **Questions:** the union of the questions E-1's and E-2's gates fire on in BEAM 100K: 45 (40
   event_ordering, 2 summarization, 1 each of contradiction_resolution, information_extraction and
   multi_session_reasoning). E-1 fires on 44, E-2 on 40.
2. **Arms, three answers each** (arm names `X@1..3`): **H** (the production `dated_items` on the
   stored items, which predate the date header), **H′** (H again), **E1** (H re-sorted by
   `created_at`, earliest first, where E-1's gate fires, else H) and **E2** (the same under E-2's
   gate; E-2's record says it joins this run as one arm). A question's arm score is the mean of its
   three answers.
3. **Reader and judge:** `benchmarks/beam/aml_c9_probe.py`'s AML BEAM answer prompt, batch rubric
   judge and event-ordering alignment score, with the model call replaced by
   `deepseek/deepseek-v4.1-flash`, one pinned provider, reasoning off (`scripts/aml_beam_offline_arms.py`).
   The registered event-ordering metric is the alignment score (F1 times normalised Kendall tau);
   the rubric mean is reported beside it.
4. **Order of runs:** after T-1's BEAM half finishes, never alongside it. Cap USD 5, floor USD 5.
5. **LongMemEval-S** (the only second dataset left, 8 gated questions) runs when X-1 Stage B has
   stored its retrieval, with its own AML prompt and judge; it is a direction check, as recorded.

Predictions and the decision rule are unchanged.

### Amendment 2, 2026-09-26, during Stage 1: paused, then resumed with the same cap

Stage 1 started automatically when T-1's BEAM run ended at its cap, and I paused it at 289 of 540
answers because the cost per BEAM answer (about USD 0.0073, 21,000 prompt tokens) made my estimate
to the user wrong. The user decided on 2026-09-26 to finish it after T-1's BEAM half. It resumes from
its own output (written answers kept); the total cap stays USD 5, so the resumed process may spend
at most USD 3.00 more.

### Stage 1 result, 2026-09-26

Amendments 1 and 2: 45 gated BEAM 100K questions, stored 2026-09-24 retrieval, arms H, H′, E1, E2 ×
3 answers each (540 answers, all judged; 10 judge failures excluded and counted), DeepSeek V4.1
Flash answering and judging with AML's BEAM prompts. A question's arm score is the mean of its three
answers. Spend: USD 2.46 in the resumed process plus about USD 2.1 before the pause (289 answers at
about USD 0.0073 each; that process was stopped before it logged its spend), about USD 4.6 against
the USD 5 cap. Output: `results/aml-e2/stage1-beam-score.json`.

| Contrast | Set | Predicted | Alignment (registered) | Rubric mean |
|---|---|---|---|---|
| E1 − H | event_ordering, E-1 gated (39) | +0.05, band 0.00 to +0.10 | **−0.006** [−0.021, +0.009], 7 / 9 | +0.036 [−0.030, +0.103], 14 / 14 |
| H′ − H | event_ordering (40) | within ±0.04 | −0.004 [−0.012, +0.004] | +0.020 [−0.011, +0.053] |
| E1 − H | all 45 | not predicted | +0.003 [−0.015, +0.020] | +0.038 [−0.021, +0.099] |
| E1 − H | the 4 non-ordering questions E-1 fires on | not predicted | +0.084 [−0.004, +0.167] | same |

**Falsified: E1 − H on BEAM's gated event_ordering questions is at or below 0 on the registered
metric.** The decision rule (at least +0.04, noise smaller) is not met on either metric: the rubric
mean's +0.036 is below the bar and under twice the replicate difference. **E-1 is not recommended.**

**A limit that qualifies the null.** The alignment score is near its floor for every arm (0.019 to
0.029): with this reader's concise answers, the alignment of the listed events to the reference
barely registers any order. The +0.101 that motivated E-1 was measured with Qwen3-14B on the probe's
own score. So this run says gated time order does not help a DeepSeek reader on this metric; it does
not measure what it would do for the reader AML uses. ScriptMem stays unavailable, and the
LongMemEval-S direction check waits for X-1 Stage B.

### LongMemEval-S direction check, 2026-09-26: unavailable on X-1's draw

X-1 Stage B finished all 120 LongMemEval-S questions it drew (stratified, 20 per question type), so
the direction check this record names could run on their stored retrieval. **E-1's gate fires on 0
of those 120.** The 8 gated questions Stage 0 found are in the full 500, and none was drawn. There
is nothing to reorder, so the check is reported as unavailable, at no cost; drawing the 8 on
purpose would need a new collect and would still be a direction check on 8 questions. E-1 stays
not recommended on its BEAM result.
