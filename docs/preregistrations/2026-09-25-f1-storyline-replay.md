# Pre-registration: does an Add-time gpt-4o-mini storyline lift BEAM summarization (AML F1)?

**Date:** 2026-09-25   **Status:** predicted, not yet measured

## The question

AML's Textual smoke on C9 scored 0 on F1 (summarization and long-history synthesis), on one to
three questions. The public BEAM 100K probe (`2026-09-24-c9-beam-ability-probe.md`, branch
`claude/c9-beam-probe`) found the cause is the reader, not retrieval: summarization scored
**0.304** while the rubric evidence was in the returned 100 items for **0.950** of the points, and
time order did not help (-0.027). AML's reader is told to be "direct and concise" in 512 tokens and
leans on the top of the context. BEAM's summarization rubrics have 5 to 8 points, one per phase of
a long history, each with named specifics.

AML's live rules say "Search must not generate final answers or disguise answers as memory
records" (API guide, read 2026-09-25), so a summary written at Search for the question is out.
The rules expect gpt-4o-mini at Add. This test asks whether a synthesis built **at Add, before any
question exists**, and placed on top of the SAME stored retrieval **only for questions that ask for
a summary**, raises the summarization rubric score. It targets the SECOND Full run (user decision
2026-09-25); C9 is not changed.

## Design

Harness: `benchmarks/beam/f1_storyline_replay.py`, tests `tests/test_beam_f1_storyline_replay.py`
(8 tests, 6 mutations shown red, recorded in the test docstring). Answering and judging call the
probe's own functions (`benchmarks/beam/aml_c9_probe.py`, copied unchanged from
`claude/c9-beam-probe` at `74bfc681`): AML's BEAM answer prompt, batch rubric judge, Qwen3-14B with
thinking off for both, temperature 0.

Inputs (sha256 prefixes):

| file | sha256 | origin |
|---|---|---|
| `beam100k.jsonl` | `62a950489d597b68` | the probe's input, 20 conversations, 400 questions |
| `retrieval.jsonl` | `9b3336f52965d123` | the probe's stored C9 Search results, 400 x 100 items, C9 `385c6074` |
| `locomo10.json` | `79fa87e90f040813` | 1,986 LoCoMo questions, only to count gate false fires |

**Simulated Add stream.** Each BEAM session is cut the way AML's Textual adapter cuts it: a chunk
closes at 20 messages or 2,000 words, never crossing a session. Conversations run in parallel;
chunks within a user run in order, as C9's per-user lock serialises Adds. Each chunk is ONE
`openai/gpt-4o-mini` call (temperature 0, JSON mode) that receives the current storyline and the
chunk, and returns a digest of the chunk (80 words at most) and the updated storyline (600 words at
most, dated phases, oldest first, compress oldest first). The builder prompt is `BUILDER_SYSTEM`,
sha256 prefix `44d57274f2c68723`. A failed call keeps the previous storyline (fail closed).

**Gate.** `GATE`, a frozen regex for summary intent in English plus five Chinese words, sha256 of
the pattern prefix `db8d3caec553aa39`. No LLM gate in this stage: on BEAM a regex will match
"Can you provide a ... summary" trivially, so any LLM gate result here would say nothing about
AML's hidden phrasings.

**Arms**, all capped at 100 items (the platform's `top_k`):

| arm | gated questions | other questions |
|---|---|---|
| `r0` | stored items unchanged (concurrent replicate of the probe's `returned`) | same |
| `story` | final storyline at rank 1, then the stored items | r0's answer reused (byte-identical context) |
| `digests` | the user's digests, chronological, BM25 top 24 if more, then stored items | r0's answer reused |
| `story_digests` | storyline, then digests, then stored items | r0's answer reused |
| `story_all` | storyline at rank 1 on EVERY question, ungated (exploratory) | same |

Every BEAM question is asked after the whole conversation, so the storyline used is the one after
the user's last Add. `storycover` asks the probe's coverage judge whether the storyline ALONE holds
each summarization rubric point (the mechanism metric). Paired bootstrap, 10,000 resamples, seed
20260925. Model spend cap: USD 8 in total.

**Contamination.** While designing I read four BEAM **1M** summarization questions with their
rubrics (to learn the rubric shape). I have read no BEAM 100K summarization question, rubric or
answer. The prompt and the gate are frozen by this commit.

## Predictions

`[[i-over-predict-effect-magnitudes]]`: eleven of twelve past predictions were too high by 2 to 4
times, so the effect bands below are deliberately modest.

| quantity | band | point |
|---|---|---|
| `r0` summarization (replicate of 0.304) | 0.25 to 0.36 | 0.30 |
| `story` minus `r0`, summarization | +0.03 to +0.15 | +0.07 |
| `digests` minus `r0`, summarization | -0.02 to +0.10 | +0.03 |
| `story_digests` minus `r0`, summarization | +0.02 to +0.15 | +0.07 |
| storyline-alone coverage of summarization rubric points | 0.35 to 0.70 | 0.50 |
| `story_all` minus `r0`, mean over the 9 other types | -0.05 to +0.01 | -0.02 |
| `story_all` minus `r0`, multi_session_reasoning | -0.05 to +0.08 | +0.01 |
| `story_all` minus `r0`, event_ordering (rubric) | -0.03 to +0.10 | +0.03 |
| gate fires, BEAM summarization | 40 of 40 | |
| gate fires, the other 360 BEAM questions | 0 to 3 | 1 |
| gate fires, 1,986 LoCoMo questions | 0 to 10 | 2 |
| builder latency per Add, p50 | 5 to 15 s | 8 s |
| builder completion tokens per Add, p50 | 600 to 1,100 | 850 |
| builder spend, whole split | under USD 1.50 | USD 0.80 |

## Apparatus checks (a failed check voids the reading, not the run)

1. Every chunk built; at most 2% failed calls.
2. `r0` summarization lands within 0.07 of the probe's 0.304. Outside that, reader drift is too
   large for the deltas to be read.
3. In every gated arm, each ungated question's answer is `r0`'s, byte for byte.
4. No arm answers from more than 100 items.
5. Judge left at most 2 summarization questions unscored in any arm.

## Decision rule, fixed now

Stage 1 (building it into a C9 successor, tested end to end on VPS3) is licensed only if BOTH hold:

- `story` or `story_digests` raises summarization by **+0.05 or more, with the 95% CI lower bound
  above 0**;
- the gate fires on at most 1% of non-summary questions, on BEAM and on LoCoMo separately.

If both arms pass, Stage 1 builds the cheaper one unless the other leads by 0.03 or more. If
`story_all` is non-inferior on the 9 other types (CI lower bound above -0.02), that is recorded as
exploratory evidence for widening the gate, NOT as licence to drop it. Anything else closes this
design; the result goes to the user either way.

## What this cannot show

- The platform's real reader, judge and context format are not public; this uses the probe's
  assumptions (content only, returned order, Qwen3-14B).
- The AML chunking rule is taken from the API guide, not observed on AML Textual data.
- The storyline is not grounded to anchor ids here; Stage 1 must add grounding, and a
  hallucinated specific would be judged as a wrong answer, not flagged.
- The regex gate matching BEAM's phrasing says nothing about AML's hidden F1 phrasings.
- One public split, 40 summarization questions: the CI will be wide.
- Real Add latency inside C9 (the extra call runs beside the 17.9 s p50 compile) is estimated from
  the builder's own latency, not measured in the service.

## Result

(appended below after the run; nothing above is edited)

### Amendment 1, 2026-09-25, before any outcome was read

Nothing in the sections above is edited. Two apparatus failures stopped the first build; no arm
had been answered with a storyline and no judgement had been run when this was written.

1. **The OpenRouter account ran out of credit** (855 total, 853.22 used). 402 responses were
   recorded as fail-closed Adds, which misrepresents an apparatus failure as service behaviour.
   The build now stops on a 402 (`46f41eb7`), and the first 59 `r0` answers were discarded because
   the provider shrinks `max_tokens` when credit runs low.
2. **gpt-4o-mini ignores the 600-word storyline limit in the prompt.** Storylines reached 1,068
   words (p90 933), and past the 1,400-token output cap the JSON was cut off (`finish_reason:
   length`, reproduced on conversation 19 chunk 15); at temperature 0 the six retries fail
   identically, so 48 Adds were fail-closed. That is a defect in the design, not noise: a storyline
   that outgrows its cap stops updating.

**Change (the only one):** the builder's output cap is 2,600 tokens, and whenever the returned
storyline exceeds 700 words a second gpt-4o-mini call (`COMPRESS_SYSTEM`) rewrites it to at most
450 words, keeping every dated phase. This is the "compression" half of F1 enforced by a mechanism
rather than by a prompt instruction. Two tests cover it, both shown red by mutation.

**Consequences for reading the result.** The storyline is rebuilt from scratch for all 20
conversations with the amended builder; the first build is kept aside as
`memory.v1-abandoned.jsonl` and is not used by any arm. The builder latency, token and spend
predictions are read against the main call; compression calls are reported separately, and their
cost is added to the spend. The `r0` answers are independent of the builder and are kept. The
answers for `r0` start roughly 30 minutes before the storyline arms; that gap is recorded as a
drift risk (`[[llm-reader-runs-drift-between-sessions]]`).

### Amendment 2, 2026-09-25, before any storyline arm was answered

The exploratory `story_all` arm (storyline on every question, ungated) is **not run**. Two
reasons, both known before any storyline arm existed:

- Another session is testing the ungated placement directly on the same stored retrieval:
  `docs/preregistrations/2026-09-25-c9-beam-session-summaries.md` on `claude/c9-beam-probe`
  (per-session gpt-4o-mini summaries, undated, placed before the items on all 400 questions).
  Running a second ungated arm would duplicate it; its result is cited instead.
- The OpenRouter account shared with the official C9 has little credit left, and `story_all` is
  the largest remaining spend (400 long answers) with no decision riding on it.

The three `story_all` predictions in the table above are therefore left unscored, and the
widening-the-gate clause of the decision rule cannot fire. Nothing else changes.

### Amendment 3, 2026-09-25, before any storyline arm was answered

The amended build (Amendment 1) failed 31 of its first 833 Adds (3.7%), past apparatus check 1.
All 31 were the COMPRESSION call, not the builder: gpt-4o-mini mostly does not compress when asked
(253 of 338 successful compressions still returned more than 700 words), so a long reply passed
the 1,400-token compression cap and truncated its JSON. Failing closed then kept the previous
storyline, the next Add grew it again, and the next compression failed the same way, so affected
storylines stopped taking in new content.

**Change:** the compression cap is 2,600 tokens, and a failed compression keeps the NEW
uncompressed storyline (fail forward) instead of the previous one. A 402 still stops the build.
One new test, red by mutation. Rows up to each conversation's first failure are kept unchanged
(the change only alters calls that failed); each conversation resumes from its first failure.

**Finding recorded now, independent of the outcome:** prompt-level and call-level compression are
both unreliable with gpt-4o-mini. Its storylines level off near 1,000 to 1,200 words on their own
(final median 981, max 1,167 in the Amendment 1 build), so the practical bound is that plateau,
not the 600 or 450 words asked for. A C9 successor must bound length in code (for example by
dropping the oldest phases past a hard word count), not by instruction.

### Amendment 4, 2026-09-25, after `r0` was judged and before any storyline arm was answered

**Apparatus check 2 FAILED.** `r0` summarization is **0.3752** against the probe's **0.3038**, a
gap of **+0.0715** against the pre-registered limit of 0.07. It is narrow, and it is not the only
drift: on byte-identical contexts, 21 of 40 summarization scores changed from the probe's run,
information_extraction moved **-0.098** and temporal_reasoning **-0.111**. So the Qwen3-14B reader
and judge through OpenRouter vary far more between runs than I assumed.

By the rule written above, this **voids the confirmatory reading**: whatever the storyline arms
show, this run cannot license Stage 1 on its own. The prediction for `r0` (0.25 to 0.36, point
0.30) is also falsified high.

**Change:** one arm is added, `r0b`, which re-answers the 40 gated questions from `r0`'s exact
items, concurrently with `story`, `digests` and `story_digests`, and is judged alongside them. It
measures the noise floor any storyline effect must clear. The arms are then read as exploratory:
each storyline arm is reported against both `r0` and `r0b`, and an effect is called only if it
exceeds the `r0b` minus `r0` gap by a clear margin. A confirmatory test, if warranted, needs its
own pre-registration on data this run has not touched (a BEAM 500K or 1M subset).

### Result, 2026-09-25

Readout: `docs/results/2026-09-25-f1-storyline-replay-readout.txt`, produced by
`benchmarks/beam/f1_storyline_readout.py` over the run directory (not committed: 60 MB of model
output, kept in the session scratchpad). All arms answered and judged; `story`, `digests`,
`story_digests` and `r0b` concurrently, `r0` about 45 minutes earlier.

**Apparatus checks.** 1 passed (final build: 9 of 1,070 Adds failed, 0.8%; 551 compressions, 0
failed compressions). **2 FAILED** (Amendment 4: `r0` 0.3752 against 0.3038). 3 passed (every
reused answer byte-identical to `r0`; every non-summary judgement identical to `r0`). 4 passed (at
most 100 items). 5 passed (0 unscored summarization questions in every arm).

**Summarization, 40 questions:**

| arm | mean | minus `r0` [95% CI] | minus `r0b` [95% CI] | minus pooled baseline [95% CI] |
|---|---|---|---|---|
| `r0` | 0.375 | | | |
| `r0b` (identical items) | 0.298 | **-0.077** [-0.178, +0.016] | | |
| `story` | 0.320 | -0.055 [-0.164, +0.046] | +0.022 [-0.067, +0.111] | -0.017 [-0.103, +0.067] |
| `digests` | 0.275 | -0.100 [-0.214, +0.010] | -0.024 [-0.131, +0.081] | -0.062 [-0.163, +0.034] |
| `story_digests` | 0.263 | -0.112 [-0.228, +0.001] | -0.035 [-0.135, +0.067] | -0.074 [-0.169, +0.022] |

The pooled baseline is the per-question mean of `r0` and `r0b`.

**Reading.** No storyline arm is distinguishable from the baseline. The noise floor is the finding
that dominates: re-answering byte-identical contexts moved summarization by -0.077, as large as
any arm's effect. The best arm, `story`, sits at -0.017 against the pooled baseline, with a CI
that rules out the +0.07 point I predicted at its upper end only barely (+0.067). Adding digests
lowered the point estimates in both arms that carried them, as the aggregation view did on
LoCoMo (`[[c9-aggregation-view-hurt-list-answers]]`), but not beyond the noise.

**Mechanism (exploratory).** The evidence is not the constraint. The storyline ALONE holds 0.706
of the rubric points (coverage judge, 40 questions), on top of the 0.95 already in the 100 items.
The constraint is the reader's answer: AML's BEAM prompt asks it to "Be direct and concise", and it
writes a median of **68 words** (`r0`) or **74.5** (`story`) for a question whose rubric has a
median of **5** points (range 3 to 8). It is credited with a median of 2 points in `r0` and 1 in
`story`. Putting the synthesis on top did not make the answer longer or more complete; the reader
still compresses to a few sentences and picks which points to keep. A memory system cannot change
the reader's prompt, so on this reader, content placed at Search cannot buy F1 points.

**Predictions against outcomes.**

| prediction | band | measured | verdict |
|---|---|---|---|
| `r0` summarization | 0.25 to 0.36 | 0.375 | falsified high |
| `story` minus `r0` | +0.03 to +0.15 | -0.055 | falsified low |
| `digests` minus `r0` | -0.02 to +0.10 | -0.100 | falsified low |
| `story_digests` minus `r0` | +0.02 to +0.15 | -0.112 | falsified low |
| storyline-alone coverage | 0.35 to 0.70 | 0.706 | falsified high, narrowly |
| `story_all` rows (3) | | not run | unscored (Amendment 2) |
| gate, BEAM summarization | 40 of 40 | 40 of 40 | held |
| gate, other 360 BEAM | 0 to 3 | 0 | held |
| gate, 1,986 LoCoMo | 0 to 10 | 0 | held |
| builder latency p50 | 5 to 15 s | 11.5 s (p90 20.5 s) | held |
| builder completion tokens p50 | 600 to 1,100 | 1,047 | held |
| builder spend, whole split | under USD 1.50 | USD 1.84 (final build) | falsified high |

`[[i-over-predict-effect-magnitudes]]` again: every effect prediction was too high, and this time
the direction was wrong too.

**Decision, by the rule above:** Stage 1 is NOT licensed. The confirmatory reading is void (check
2), and even read exploratorily no arm reaches +0.05, let alone with a CI above 0. This design
(Add-time storyline plus a summary-intent gate, on top of the stored items) is closed for the AML
BEAM reader.

**What survives, for the next idea.**

1. The gate works and is free: 40 of 40, 0 of 2,346 false fires. Any future F1 mechanism can use it.
2. gpt-4o-mini does not honour length limits, by instruction or by a compression call: 450 of 551
   compressions still returned more than 700 words, and final storylines reached 2,130 words. A
   C9 successor must bound length in code.
3. Run-to-run noise on this reader and judge is about 0.08 on 40 summarization questions. Any F1
   test on this apparatus needs a concurrent replicate, and 40 questions cannot resolve an effect
   under about 0.10.
4. The measured lever is the reader's brevity, which the participant does not control. The only
   levers left to a memory system are the ORDER and FORM of what the reader sees: for example one
   storyline item written as a short itemised list of phases that a 70-word answer can copy
   whole. That is a new hypothesis, not tested here.

Model spend for this run: builder USD 1.84 (final build) plus about USD 0.6 in abandoned builds;
Qwen3-14B answers and judgements about USD 2.5. The shared OpenRouter balance read USD 8.41 at the end.
